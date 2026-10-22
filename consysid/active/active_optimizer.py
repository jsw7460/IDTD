"""CMA-ES optimizer for active exploration command sequence optimization.

Maximizes tr(FIM) by optimizing Bezier curve control points that
parameterize the command trajectory fed to a pretrained policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import cma

from consysid.active.bezier_commands import BezierCommandGenerator
from consysid.active.console import (
    print_active_header,
    print_active_iteration,
    print_active_rollout_step,
    print_active_rollout_done,
    print_active_convergence,
)
from consysid.active.env_partitioner import EnvPartitioner
from consysid.active.fim_evaluator import FIMEvaluator
from consysid.sysid.parameters import SysIDParameters


# Gait mode order mirrors ``bezier_commands.GAIT_MODES`` keys (0..3).
# Used for both console formatting and wandb key naming so the two
# stay in lockstep — change in one place if a new mode is ever added.
GAIT_NAMES = ("pronk", "trot", "pace", "bound")


@dataclass
class ActiveCMAConfig:
    """CMA-ES optimizer config for active exploration."""

    sigma0: float = 0.3
    max_iterations: int = 100
    population_size: int = 16
    seed: int = 42
    tol_fun: float = 1e-8

    save_interval: int = 50
    """Every this many iterations, dump a numbered checkpoint
    alongside the always-overwritten ``active_commands_latest.npz``.
    Mirrors ``CMAOptimizerConfig.save_interval`` from identification so both
    pipelines expose the same "periodic + latest" crash-safe pattern.
    Set to ``0`` to disable numbered snapshots (``_latest`` still
    overwritten every iteration)."""

    parallel_candidates: bool = True
    """When ``True`` (default), all ``population_size`` CMA-ES candidates
    are evaluated in a single physics rollout — env count scales to
    ``population_size × num_groups × (param_dim + 1)`` and every
    candidate's command sequence is injected simultaneously to its own
    env block. Matches how identification parallelises candidates
    (``evaluate_parallel`` in ``sysid/open_loop_evaluator.py``).

    Set to ``False`` to fall back to the legacy per-candidate rollout
    loop (one rollout per candidate, env count = ``num_groups × (N+1)``)
    — useful as a correctness oracle when diagnosing the parallel path.
    """


class ActiveExplorationOptimizer:
    """CMA-ES optimizer for active exploration (active exploration).

    Finds command sequences that maximize Fisher Information Matrix (FIM)
    trace, producing trajectories most informative for system identification.

    The optimizer searches over Bezier curve control points that
    parameterize the 14D command trajectory fed to a pretrained
    gait-conditioned policy. Each candidate command sequence is
    evaluated by rolling out the policy in (1 main + N aux) envs
    and computing finite-difference FIM.

    Each CMA-ES iteration:
        1. Sample ``population_size`` candidate Bezier control point vectors
        2. For each candidate, decode control points into a per-timestep
           14D command trajectory via Bernstein basis evaluation
        3. Roll out the trajectory: inject commands into the policy,
           step the simulator, sync aux states to main periodically
        4. Compute tr(FIM) from state differences between main and aux envs
        5. Report ``-tr(FIM)`` to CMA-ES (minimize negative = maximize)
    """

    def __init__(
        self,
        bezier_gen: BezierCommandGenerator,
        partitioner: EnvPartitioner,
        fim_evaluator: FIMEvaluator,
        params: SysIDParameters,
        config: ActiveCMAConfig,
        delta_gen: Any | None = None,
        barrier_coef: float = 0.0,
        relative_threshold: float = 0.8,
        r_baseline: float = 0.0,
    ):
        """
        Args:
            delta_gen: Optional ``BezierActionDeltaGenerator``. When
                set, the CMA candidate vector is augmented with the
                delta's parameters (``[x_command, x_delta]``) and the
                rollout step adds the decoded ``Δa_t`` to the policy's
                action before stepping the env.
            barrier_coef: Coefficient on the soft locomotion-stability
                barrier ``barrier_coef * max(0, α·R_baseline − R)`` added
                to the FIM cost. ``0`` (default) disables the barrier
                even when ``delta_gen`` is set.
            relative_threshold: ``α``. Fraction of ``r_baseline`` below
                which the barrier kicks in. ``0.8`` allows up to 20 %
                drop in mean per-env episode return.
            r_baseline: Reference mean per-env episode return measured
                in a one-shot delta=0 / Bezier-init rollout. Used as
                the threshold scale; consumed only when
                ``barrier_coef > 0``.
        """
        self.bezier_gen = bezier_gen
        self.partitioner = partitioner
        self.fim_evaluator = fim_evaluator
        self.params = params
        self.config = config
        self.delta_gen = delta_gen
        self.barrier_coef = float(barrier_coef)
        self.relative_threshold = float(relative_threshold)
        self.r_baseline = float(r_baseline)
        # Cache of per-parameter labels for the per-parameter coverage
        # diagnostics logged inside :meth:`optimize`.
        self._param_labels_cache: list[str] | None = None
        # Latest-iter snapshot — refreshed every CMA-ES iteration with
        # *that* iter's best candidate (not the global best). Lets a
        # downstream consumer pick the most recent command instead of
        # the global-best one when CMA-ES landed on the global best
        # early and later iterations discovered a (slightly) worse cost.
        self._latest_x: np.ndarray | None = None
        self._latest_cost: float | None = None

    def _param_index_labels(self) -> list[str]:
        """Map each FIM param index → ``'term[sub]'`` (e.g. ``'kp_all[2]'``).

        Cached. Index layout matches ``theta_0`` / ``_term_slices``.
        """
        if self._param_labels_cache is None:
            labels = [f"idx{i}" for i in range(self.params.dim)]
            for name, (s, e) in self.params._term_slices.items():
                for i in range(s, e):
                    labels[i] = f"{name}[{i - s}]"
            self._param_labels_cache = labels
        return self._param_labels_cache

    def optimize(
        self,
        env: Any,
        policy: Any,
        theta_surrogate: np.ndarray,
        wandb_run: Any | None = None,
        sim_type: str = "",
        sysid_config_name: str = "",
        log_dir: str = "",
        checkpoint_fn: Callable[[int, np.ndarray, float, list], None] | None = None,
    ) -> dict:
        """Run CMA-ES optimization to find the best command sequence.

        Args:
            env: JaxRLWorld environment with ``num_groups * (param_dim + 1)``
                parallel envs for finite-difference FIM computation.
            policy: Pretrained gait-conditioned policy wrapper. Must have
                ``get_action(obs, robot_states)`` method.
            theta_surrogate: (dim,) surrogate physical parameters from
                identification (e.g. mass, CoM). Used to set the operating
                point around which FIM is computed.
            wandb_run: Optional wandb run for logging FIM metrics.
            checkpoint_fn: Optional per-iteration save callback. Called
                with ``(iteration, global_best_x, global_best_cost,
                history)`` at the end of every CMA-ES iteration so the
                runner can emit ``active_commands_latest.npz`` +
                periodic numbered snapshots. Mirrors
                ``CMAOptimizer.optimize(checkpoint_fn=...)`` in identification.

        Returns:
            Dict containing:
                - ``best_x``: Best Bezier control point vector found
                - ``best_fim``: Corresponding tr(FIM) value
                - ``velocity_cmds``: (total_steps, 3) decoded velocity commands
                - ``gait_cmds``: (total_steps, 11) decoded gait commands
                - ``history``: Per-iteration optimization history
        """
        cfg = self.config
        gen = self.bezier_gen
        cmd_lo, cmd_hi = gen.get_bounds()

        # Optional action-delta augmentation: stack the delta's CMA
        # parameters after the command's so a single CMA-ES treats
        # ``[x_command, x_delta]`` as one candidate vector. With
        # ``delta_gen=None`` the layout collapses back to the legacy
        # command-only one and every code path below behaves identically.
        if self.delta_gen is not None:
            d_lo, d_hi = self.delta_gen.get_bounds()
            lower = np.concatenate([cmd_lo, d_lo])
            upper = np.concatenate([cmd_hi, d_hi])
            x0 = np.concatenate([gen.get_initial_x(), self.delta_gen.get_initial_x()])
            self._cmd_dim = gen.param_dim
        else:
            lower, upper = cmd_lo, cmd_hi
            x0 = gen.get_initial_x()
            self._cmd_dim = gen.param_dim

        opts = {
            "seed": cfg.seed,
            "maxiter": cfg.max_iterations,
            "popsize": cfg.population_size,
            "tolfun": cfg.tol_fun,
            "bounds": [lower.tolist(), upper.tolist()],
        }

        es = cma.CMAEvolutionStrategy(
            x0.tolist(),
            cfg.sigma0,
            opts,
        )

        # Apply surrogate theta (+ perturbations) to all envs once.
        # ``theta_default`` is the term-wise URDF nominal (params.theta_0
        # at this point holds the fresh term defaults — this code path
        # never invokes ``params.theta_0 = theta_init`` since explore
        # uses ``theta_surrogate`` directly as the env configuration);
        # the partitioner only consumes it for
        # ``theta_prior_mode == 'default_to_surrogate'``. ``term_slices``
        # is similarly only consumed when ``theta_prior_gaussian_std``
        # is a per-term dict.
        theta_batch = self.partitioner.build_theta_batch(
            theta_surrogate,
            theta_default=self.params.theta_0,
            term_slices=self.params._term_slices,
        )
        self.params.apply_to_env(theta_batch)

        history = []
        # CMA-ES minimizes ``cost`` (unified objective). We track the
        # running best as the smallest mean-cost seen so far; for
        # display and backward-compat logging we also expose
        # ``-best_cost`` under the legacy ``fim`` keys (identical to
        # the old ``tr(F)`` signal in SPI-Active mode).
        best_cost = float("inf")
        best_x = None

        print_active_header(
            sim_type=sim_type,
            sysid_config_name=sysid_config_name,
            bezier_dim=gen.param_dim,
            popsize=cfg.population_size,
            num_groups=self.partitioner.cfg.num_groups,
            param_dim=self.partitioner.param_dim,
            total_envs=self.partitioner.total_envs,
            num_segments=gen.cfg.num_segments,
            segment_duration=gen.cfg.segment_duration,
            bezier_degree=gen.cfg.bezier_degree,
            optimized_commands=gen.cfg.optimized_commands,
            delta_param=self.partitioner.cfg.delta_param,
            ksync_steps=self.partitioner.cfg.ksync_steps,
            sigma0=cfg.sigma0,
            theta_surrogate=theta_surrogate,
            wandb_url=wandb_run.url if wandb_run is not None else None,
            log_dir=log_dir,
        )

        # One-time: publish the FIM param index → name map so the
        # per-parameter coverage panels are human-readable on the run's
        # Overview / Config page.
        if wandb_run is not None:
            labels = self._param_index_labels()
            wandb_run.config.update(
                {"param_index_map": {str(i): lab for i, lab in enumerate(labels)}},
                allow_val_change=True,
            )

        iteration = 0
        while not es.stop():
            candidates = es.ask()
            costs = []
            iter_num_terminated = 0       # worst candidate (max over population)
            iter_mean_terminated = 0.0    # population mean
            iter_best_terminated = 0      # this iter's best candidate (argmin cost)

            import time as _time
            t_iter_start = _time.time()

            # Per-iteration telemetry accumulators for wandb/console.
            # Per-parameter decoupling coverage ``c_i`` averaged over the
            # group axis, one row per candidate → (num_cand, N).
            per_param_c_per_cand: list[np.ndarray] = []

            if cfg.parallel_candidates:
                # Single rollout covers every candidate. ``fim_result`` has
                # per-(P, G) tensors — reduce the group axis here so the
                # downstream wandb / CMA-ES plumbing sees one scalar per
                # candidate, matching the serial path semantics.
                candidates_np = np.stack(
                    [np.asarray(c, dtype=np.float32) for c in candidates], axis=0,
                )
                fim_result = self._evaluate_candidates_parallel(
                    candidates_np, env, policy, theta_surrogate,
                )
                cost_pg = fim_result["cost"]                              # (P, G)
                # Reduce the group axis according to FIMConfig.cost_aggregation.
                # ``"mean"`` (default) → Bayesian-average cost (current behaviour).
                # ``"worst_case"``     → max cost across groups (robust min-max);
                #                       CMA-ES then minimises the *worst* group
                #                       cost so commands are informative under
                #                       every θ_g in the prior. Cost itself is
                #                       in minimise-form (negative info), hence
                #                       ``.max(dim=-1)`` for worst-case info.
                _agg = self.fim_evaluator.cfg.cost_aggregation
                if _agg == "mean":
                    cost_per_cand = cost_pg.mean(dim=-1)
                elif _agg == "worst_case":
                    cost_per_cand = cost_pg.max(dim=-1).values
                else:
                    raise ValueError(
                        f"FIMConfig.cost_aggregation must be 'mean' or "
                        f"'worst_case', got {_agg!r}"
                    )
                # Optional locomotion-stability barrier on the action-
                # delta ablation. ``barrier_coef * max(0, α·R_b − R)``
                # is added per-candidate so destabilising deltas (mean
                # per-env return below ``α·R_baseline``) get a strictly
                # positive penalty proportional to how far below they
                # fell. ``barrier_coef == 0`` (default) leaves the cost
                # untouched, so the legacy command-only path is byte-
                # identical to its pre-ablation behaviour.
                return_per_cand = fim_result.get("return_per_cand", None)
                if self.barrier_coef > 0.0 and return_per_cand is not None:
                    threshold = self.relative_threshold * self.r_baseline
                    barrier = self.barrier_coef * torch.clamp(
                        threshold - return_per_cand, min=0.0,
                    )
                    cost_per_cand = cost_per_cand + barrier
                    iter_barrier_mean = float(barrier.mean().item())
                    iter_return_mean = float(return_per_cand.mean().item())
                else:
                    iter_barrier_mean = 0.0
                    iter_return_mean = (
                        float(return_per_cand.mean().item())
                        if return_per_cand is not None
                        else 0.0
                    )
                costs = cost_per_cand.cpu().numpy().tolist()              # (P,)
                # Per-parameter coverage ``c_i``: (P, G, N) → average
                # over the group axis → (P, N), one row per candidate.
                per_param_c_pg = fim_result["decoupling_per_param_c"]
                per_param_c_per_cand = list(
                    per_param_c_pg.mean(dim=1).cpu().numpy(),
                )

                # Per-candidate terminated-group counts. ``max`` is the
                # worst candidate (a CMA-ES tail sample that fell — always
                # near num_groups, so misleading on its own); ``mean`` is
                # the population health; ``best`` is the terminated count
                # of the lowest-cost candidate (the solution CMA is
                # actually converging to — this is the one to watch).
                group_term = self.fim_evaluator.group_terminated          # (P, G)
                term_per_cand = group_term.sum(dim=-1)                    # (P,)
                iter_num_terminated = int(term_per_cand.max().item())
                iter_mean_terminated = float(term_per_cand.float().mean().item())
                iter_best_terminated = int(term_per_cand[cost_per_cand.argmin()].item())
            else:
                for ci, candidate in enumerate(candidates):
                    print(
                        f"\r  evaluating candidate {ci + 1}/{len(candidates)}",
                        end="", flush=True,
                    )

                    x = np.array(candidate, dtype=np.float32)
                    fim_result = self._evaluate_candidate(
                        x, env, policy, theta_surrogate,
                    )

                    # ``cost`` is already in "minimize" form. Tensor
                    # shape is ``(1, G)`` (P=1 in serial path). Reduce
                    # the group axis with the same ``cost_aggregation``
                    # rule as the parallel path so the two paths are
                    # behaviourally identical for any given config.
                    cost_tensor = fim_result["cost"]      # (1, G)
                    _agg = self.fim_evaluator.cfg.cost_aggregation
                    if _agg == "mean":
                        cost = float(cost_tensor.mean())
                    elif _agg == "worst_case":
                        cost = float(cost_tensor.max())
                    else:
                        raise ValueError(
                            f"FIMConfig.cost_aggregation must be 'mean' "
                            f"or 'worst_case', got {_agg!r}"
                        )
                    costs.append(cost)

                    # Per-parameter coverage ``c_i``: (1, G, N) → average
                    # over the group axis → (N,).
                    per_param_c_per_cand.append(
                        fim_result["decoupling_per_param_c"].mean(dim=1)[0]
                        .cpu().numpy(),
                    )

                    # Termination count from the authoritative source.
                    cand_terminated = int(self.fim_evaluator.group_terminated.sum().item())
                    iter_num_terminated = max(iter_num_terminated, cand_terminated)

            print("\r" + " " * 80 + "\r", end="", flush=True)

            es.tell(candidates, costs)
            iteration += 1
            rollout_time = _time.time() - t_iter_start

            costs_arr = np.array(costs)
            best_idx = int(np.argmin(costs_arr))
            best_iter_cost = float(costs_arr[best_idx])

            # Always-update latest snapshot: this iter's best candidate
            # (x, cost). Independent of the global-best gate below so a
            # script that prefers "last iter's commands" can read it
            # straight out of the npz.
            self._latest_x = np.array(candidates[best_idx], dtype=np.float32)
            self._latest_cost = float(best_iter_cost)

            if best_iter_cost < best_cost:
                best_cost = best_iter_cost
                best_x = np.array(candidates[best_idx], dtype=np.float32)

            # ── Gait composition of the current global best ───────────────
            # The Bezier vector embeds 4 logits per segment that softmax-
            # blend pronk / trot / pace / bound (see
            # ``bezier_commands.decode``). Surface that distribution so
            # we can see *which gait* CMA-ES is converging toward, not
            # just the FIM scalar. Averaged over segments → 4 scalars per
            # iteration (one per mode), whose sum is 1. Per-segment
            # composition is captured in the wandb summary at the end of
            # ``optimize`` for finer-grained inspection.
            # Velocity-only configs have no gait logits — skip the gait
            # composition diagnostics entirely.
            if gen.cfg.use_gait:
                gait_probs = gen.decode_gait_probs(best_x)         # (num_seg, 4)
                gait_mean_probs = gait_probs.mean(axis=0)          # (4,)
                gait_dom_idx = int(np.argmax(gait_mean_probs))
                gait_dom_prob = float(gait_mean_probs[gait_dom_idx])
            else:
                gait_mean_probs = None
                gait_dom_idx = None
                gait_dom_prob = None

            # The console display frames the score as "higher is
            # better" by mirroring the minimisation cost with a sign
            # flip.
            print_active_iteration(
                iteration=iteration,
                max_iterations=cfg.max_iterations,
                fim_best=-best_iter_cost,
                fim_mean=float(-costs_arr.mean()),
                fim_global_best=-best_cost,
                sigma=float(es.sigma),
                num_terminated=iter_num_terminated,
                mean_terminated=iter_mean_terminated,
                best_terminated=iter_best_terminated,
                num_groups=self.partitioner.cfg.num_groups,
                rollout_time=rollout_time,
                wandb_url=wandb_run.url if wandb_run is not None else None,
                log_dir=log_dir,
            )
            # Compact gait summary line — pairs with the per-iter FIM
            # display and answers "what is best_x walking like?". Format
            # mirrors the rest of the iteration block (2-space indent,
            # ``key=value`` pairs).
            if gait_mean_probs is not None:
                print(
                    f"  gait[best]: pronk={gait_mean_probs[0]:.2f} "
                    f"trot={gait_mean_probs[1]:.2f} "
                    f"pace={gait_mean_probs[2]:.2f} "
                    f"bound={gait_mean_probs[3]:.2f}  "
                    f"(dominant: {GAIT_NAMES[gait_dom_idx]} {gait_dom_prob:.2f})",
                    flush=True,
                )
            # Compact θ-surrogate summary — same key=value style as
            # ``gait[best]``. Per-term scalars are printed in full;
            # multi-dim terms (e.g. base_inertial 10D log-Cholesky φ)
            # are abbreviated to ``[v0, v1, ...]<N>D`` so the line stays
            # one-row at typical terminal widths. The values shown are
            # the *anchor* θ_surrogate that the run was launched with;
            # under ``theta_prior_mode != "fixed"`` each group's main env
            # uses a *different* sampled θ_g around this anchor (per-
            # group spread is logged once at startup by EnvPartitioner
            # and stays constant across iters since the prior seed is
            # fixed — re-printing per iter would just repeat).
            term_strs: list[str] = []
            for name in self.params.terms:
                start, end = self.params._term_slices[name]
                vals = theta_surrogate[start:end]
                if len(vals) == 1:
                    term_strs.append(f"{name}={float(vals[0]):.3f}")
                else:
                    head = ", ".join(f"{float(v):+.2f}" for v in vals[:2])
                    term_strs.append(f"{name}=[{head},...]{len(vals)}D")
            print(
                "  θ_surrogate: " + "  ".join(term_strs),
                flush=True,
            )

            # Wandb logging — the minimization-form ``cost/*`` (what
            # CMA-ES actually sees), a "higher is better" ``fim/*``
            # mirror (= -cost), convergence/rollout diagnostics, and the
            # per-parameter decoupling coverage ``c_i`` so the run can be
            # inspected axis by axis.
            if wandb_run is not None:
                # (num_cand, N) → mean over the population → (N,).
                per_param_c_mean = (
                    np.mean(np.stack(per_param_c_per_cand, axis=0), axis=0)
                    if per_param_c_per_cand else np.zeros(0)
                )
                wandb_run.log({
                    "iteration": iteration,
                    # Unified cost (CMA-ES objective).
                    "cost/best": best_iter_cost,
                    "cost/mean": float(costs_arr.mean()),
                    "cost/std": float(costs_arr.std()),
                    "cost/global_best": best_cost,
                    # Coverage summary: the population-mean coverage over
                    # all axes, and the weakest axis (the bottleneck the
                    # ``log`` aggregation is pushing on).
                    "coverage/c_mean": (
                        float(per_param_c_mean.mean())
                        if per_param_c_mean.size else 0.0
                    ),
                    "coverage/c_min": (
                        float(per_param_c_mean.min())
                        if per_param_c_mean.size else 0.0
                    ),
                    # "Higher is better" mirror (= -cost).
                    "fim/best": float(-best_iter_cost),
                    "fim/mean": float(-costs_arr.mean()),
                    "fim/global_best": float(-best_cost),
                    "fim/std": float(costs_arr.std()),
                    # Convergence diagnostics.
                    "cma/sigma": float(es.sigma),
                    "rollout/terminated_groups": iter_num_terminated,
                    "rollout/terminated_groups_mean": iter_mean_terminated,
                    "rollout/terminated_groups_best": iter_best_terminated,
                    "rollout/time_s": rollout_time,
                    # Action-delta diagnostics. Always logged (zero when
                    # ``delta_gen`` is unset / ``barrier_coef == 0``) so
                    # the same wandb panel works for command-only and
                    # action-delta runs.
                    "action_delta/return_mean": iter_return_mean,
                    "action_delta/return_baseline": float(self.r_baseline),
                    "action_delta/return_threshold": float(
                        self.relative_threshold * self.r_baseline
                    ),
                    "action_delta/barrier_mean": iter_barrier_mean,
                    "action_delta/delta_norm_best": (
                        float(np.linalg.norm(best_x[self._cmd_dim:]))
                        if (best_x is not None and self.delta_gen is not None) else 0.0
                    ),
                    # Gait composition of the current global best —
                    # softmax probabilities over (pronk, trot, pace,
                    # bound) averaged across all Bezier segments. Sums to
                    # 1.0 each step.
                    **(
                        {
                            **{
                                f"gait/prob_{name}": float(p)
                                for name, p in zip(GAIT_NAMES, gait_mean_probs)
                            },
                            "gait/dominant_idx": gait_dom_idx,
                            "gait/dominant_prob": gait_dom_prob,
                        }
                        if gait_mean_probs is not None else {}
                    ),
                }, step=iteration)

                # Per-parameter coverage, keyed by the physical param
                # label so a wandb panel regex ``coverage/c/.*`` reads as
                # "how decoupled-identifiable each parameter became".
                if per_param_c_mean.size:
                    labels = self._param_index_labels()
                    cov_log = {
                        f"coverage/c/{i:02d}_{labels[i]}": float(
                            per_param_c_mean[i],
                        )
                        for i in range(per_param_c_mean.shape[0])
                    }
                    wandb_run.log(cov_log, step=iteration)
            history.append({
                "iteration": iteration,
                "best_cost": best_iter_cost,
                "mean_cost": float(costs_arr.mean()),
                "global_best_cost": best_cost,
            })

            # Checkpoint: always refresh ``active_commands_latest.npz``;
            # additionally emit a numbered snapshot every
            # ``save_interval`` iterations. Identical format to the
            # final save so any snapshot can be loaded by data collection /
            # viz_bezier / downstream consumers via the same path.
            if checkpoint_fn is not None:
                checkpoint_fn(
                    iteration, best_x, best_cost, history,
                    getattr(self, "_latest_x", None),
                    getattr(self, "_latest_cost", None),
                )

        # Decode best command sequence
        velocity_cmds, gait_cmds = self.bezier_gen.decode(best_x)

        print_active_convergence(
            iteration=iteration,
            best_fim=-best_cost,
            stop_reason=es.stop(),
        )

        # Final gait composition of the converged best_x — both the
        # segment-averaged distribution (interpretation: "what gait did
        # the optimizer pick?") and the per-segment breakdown
        # (interpretation: "did the optimizer prefer a single gait
        # throughout, or switch mid-trajectory?").
        if self.bezier_gen.cfg.use_gait:
            final_gait_probs = self.bezier_gen.decode_gait_probs(best_x)  # (num_seg, 4)
            final_gait_mean = final_gait_probs.mean(axis=0)               # (4,)
        else:
            final_gait_probs = None
            final_gait_mean = None

        if wandb_run is not None:
            summary = {
                "final/best_cost": float(best_cost),
                "final/best_fim": float(-best_cost),  # legacy alias
                "final/iterations": iteration,
            }
            if final_gait_mean is not None:
                summary["final/gait_dominant"] = GAIT_NAMES[int(np.argmax(final_gait_mean))]
                for name, p in zip(GAIT_NAMES, final_gait_mean):
                    summary[f"final/gait_prob_{name}"] = float(p)
                # Per-segment finer-grained breakdown — useful when the
                # mean-over-segments hides a "trot first half, bound second
                # half" pattern. Segments are indexed from 0.
                for seg_idx in range(final_gait_probs.shape[0]):
                    for name, p in zip(GAIT_NAMES, final_gait_probs[seg_idx]):
                        summary[f"final/gait_seg{seg_idx}_prob_{name}"] = float(p)
            wandb_run.summary.update(summary)

        # ``best_fim`` is exposed as ``-cost`` so downstream consumers
        # (eval, notebook plotting) can read a "higher is better" score.
        return {
            "best_x": best_x,
            "best_cost": float(best_cost),
            "best_fim": float(-best_cost),
            "velocity_cmds": velocity_cmds,
            "gait_cmds": gait_cmds,
            "history": history,
            # Latest-iter snapshot (final iteration's best candidate).
            # See :class:`ActiveExplorationOptimizer.__init__` docstring
            # on ``self._latest_x`` for when this differs from the
            # global best and how to choose between them downstream.
            "latest_x": getattr(self, "_latest_x", None),
            "latest_cost": getattr(self, "_latest_cost", None),
        }

    @torch.no_grad()
    def _evaluate_candidate(
        self,
        x: np.ndarray,
        env: Any,
        policy: Any,
        theta_surrogate: np.ndarray,
    ) -> dict:
        """Evaluate a single Bezier control point vector.

        Decodes ``x`` into a command trajectory, rolls out the policy
        in (1 main + N aux) envs, and returns the decoupling-coverage
        cost plus its per-parameter coverage diagnostics.

        Args:
            x: Flat Bezier control point vector from CMA-ES. Length is
                ``num_segments * (num_control_points * continuous_dim + 4)``.
                Contains per-segment Bezier control points for each optimized
                command dimension, plus 4 gait-mode logits per segment.
            env: JaxRLWorld environment (already configured with correct
                number of envs for partitioning).
            policy: Pretrained policy with ``get_action(obs, robot_states)``.
            theta_surrogate: (dim,) surrogate parameters. Applied to main
                envs as-is and to aux envs with +delta perturbation on
                each parameter dimension.

        Returns:
            Dict of per-``(1, G)`` tensors:
              - ``cost``: decoupling-coverage objective (minimize)
              - ``decoupling_per_param_c``: ``(1, G, N)`` per-parameter
                coverage ``c_i``
        """
        gen = self.bezier_gen
        part = self.partitioner
        fim = self.fim_evaluator
        cfg_bezier = gen.cfg

        # 1. Decode Bezier → command trajectory
        velocity_cmds_np, gait_cmds_np = gen.decode(x)
        velocity_cmds = torch.tensor(velocity_cmds_np, device=part.device, dtype=torch.float32)
        gait_cmds = (
            torch.tensor(gait_cmds_np, device=part.device, dtype=torch.float32)
            if gait_cmds_np is not None else None
        )

        # 2. Re-apply surrogate theta (needed because env.reset may alter params)
        theta_batch = part.build_theta_batch(
            theta_surrogate,
            theta_default=self.params.theta_0,
            term_slices=self.params._term_slices,
        )
        self.params.apply_to_env(theta_batch)

        # 3. Reset all envs
        env.reset()

        # 4. Reset FIM accumulator
        fim.reset()

        # 5. Inject initial commands. Serial path always uses
        # ``num_candidates=1`` — ``broadcast_commands`` now expects a
        # ``(P, ...)``-leading tensor so we keep a size-1 leading dim.
        G = part.cfg.num_groups
        vel_t = velocity_cmds[0].unsqueeze(0)  # (1, 3)
        gait_t = gait_cmds[0].unsqueeze(0) if gait_cmds is not None else None
        part.broadcast_commands(env, vel_t, gait_t)

        total_steps = cfg_bezier.total_steps(gen.dt)
        ksync = part.cfg.ksync_steps

        # 6. Rollout loop
        num_terminated_groups = 0
        for t in range(total_steps):
            # Inject commands for this timestep
            vel_t = velocity_cmds[t].unsqueeze(0)
            gait_t = gait_cmds[t].unsqueeze(0) if gait_cmds is not None else None
            part.broadcast_commands(env, vel_t, gait_t)

            # Get observation and action
            obs = env.obs_manager.get_observation()
            robot_states = env.get_robot_state()
            action = policy.get_action(obs, robot_states)

            # Step physics
            obs, rewards, terminated, truncated, infos = env.step(action)

            # State sync: copy main → aux every ksync_steps
            if ksync > 0 and (t + 1) % ksync == 0:
                self._sync_states(env, part)

            # Read state and accumulate FIM
            state_all = fim.read_state_vector(env)
            fim.accumulate_step(state_all)

            # Handle grouped resets. Return mask now has shape ``(P, G)``
            # but serial ``P=1``, so ``.sum()`` and ``.any()`` collapse
            # it to the same scalar semantics as before.
            group_reset, reset_ids = part.handle_grouped_resets(terminated, truncated)
            if group_reset.any():
                num_terminated_groups += int(group_reset.sum().item())
                fim.mark_terminated(group_reset)

                # Re-inject next step's command to all envs (including
                # the ones that just reset, since env.step cleared their
                # externally-controlled flag).
                if len(reset_ids) > 0 and t + 1 < total_steps:
                    nxt = min(t + 1, total_steps - 1)
                    vel_next = velocity_cmds[nxt].unsqueeze(0)   # (1, 3)
                    gait_next = gait_cmds[nxt].unsqueeze(0) if gait_cmds is not None else None
                    part.broadcast_commands(env, vel_next, gait_next)

            # Progress indicator (every 10% of rollout)
            if (t + 1) % max(1, total_steps // 10) == 0:
                print_active_rollout_step(t, total_steps, num_terminated_groups, G)

        print_active_rollout_done()
        return fim.compute_fim_objective()

    @torch.no_grad()
    def _evaluate_candidates_parallel(
        self,
        candidates_np: np.ndarray,
        env: Any,
        policy: Any,
        theta_surrogate: np.ndarray,
    ) -> dict:
        """Evaluate every CMA-ES candidate in a single physics rollout.

        Mirrors the identification ``evaluate_parallel`` pattern but with
        per-candidate **command trajectories** (identification only needed
        per-candidate θ; here θ is shared and commands differ). Policy
        inference, physics step, and FIM accumulation all run at
        ``total_envs = P × G × (N+1)`` in one pass.

        Args:
            candidates_np: ``(P, bezier_param_dim)`` CMA-ES candidate
                vectors stacked along the leading axis.
            env: JaxRLWorld env with ``total_envs`` configured to match
                the partitioner's ``num_candidates = P``.
            policy: Pretrained policy — handles the full batch.
            theta_surrogate: ``(param_dim,)`` surrogate θ from identification.
                Shared across candidates.

        Returns:
            ``fim.compute_fim_objective()`` dict with ``(P, G)`` tensors.
        """
        gen = self.bezier_gen
        part = self.partitioner
        fim = self.fim_evaluator
        cfg_bezier = gen.cfg
        P = candidates_np.shape[0]

        if P != part.num_candidates:
            raise ValueError(
                f"candidates leading dim {P} != partitioner.num_candidates "
                f"{part.num_candidates}. The partitioner must be built with "
                f"num_candidates=population_size when "
                f"ActiveCMAConfig.parallel_candidates=True."
            )

        # 1. Decode every candidate's Bezier curves into full trajectories.
        #    Stack to ``(P, total_steps, 3)`` and ``(P, total_steps, 11)``.
        #    With ``delta_gen`` set, the candidate vector is
        #    ``[x_command, x_delta]``; we slice off the command head and
        #    decode the delta tail into ``(P, total_steps, num_actuated)``.
        cmd_dim = self._cmd_dim
        vel_list, gait_list = [], []
        K_list: list[np.ndarray | None] = []
        delta_list: list[np.ndarray] = []
        for p in range(P):
            x_cand = candidates_np[p]
            vel_np, gait_np = gen.decode(x_cand[:cmd_dim])
            vel_list.append(vel_np)
            gait_list.append(gait_np)
            K_list.append(gen.decode_K(x_cand[:cmd_dim]))
            if self.delta_gen is not None:
                delta_list.append(self.delta_gen.decode(x_cand[cmd_dim:]))

        # Stack per-candidate K into ``(P, cmd_dim_total, n_feat)`` when
        # state feedback is active (every candidate gets the same K
        # shape -- the Bezier config's feature tuple is shared). When
        # the feature tuple is empty, ``K_list`` is ``[None] * P`` and
        # we leave ``K_all`` as ``None`` so the rollout loop branches
        # to the legacy zero-feedback path.
        fb_features = gen.cfg.state_feedback_features
        if fb_features and K_list[0] is not None:
            K_all = torch.tensor(
                np.stack(K_list, axis=0),
                device=part.device,
                dtype=torch.float32,
            )
        else:
            K_all = None
        vel_all = torch.tensor(
            np.stack(vel_list, axis=0), device=part.device, dtype=torch.float32,
        )
        # gait_list is [None]*P for velocity-only configs (use_gait=False).
        gait_all = (
            torch.tensor(
                np.stack(gait_list, axis=0), device=part.device, dtype=torch.float32,
            )
            if gen.cfg.use_gait else None
        )
        delta_all = (
            torch.tensor(np.stack(delta_list, axis=0), device=env.device, dtype=torch.float32)
            if self.delta_gen is not None
            else None
        )
        total_steps = cfg_bezier.total_steps(gen.dt)
        ksync = part.cfg.ksync_steps

        # 2. Apply surrogate θ — same (main + aux perturbations) pattern
        #    repeats across every candidate block.
        theta_batch = part.build_theta_batch(
            theta_surrogate,
            theta_default=self.params.theta_0,
            term_slices=self.params._term_slices,
        )
        self.params.apply_to_env(theta_batch)

        # 3. Reset all envs once. Unlike the serial path we do this a
        #    single time — θ stays fixed for the whole iteration so
        #    there is nothing to re-apply between candidates.
        env.reset()

        # 4. Reset FIM accumulator (allocates ``(P, G, N, N)`` tensors).
        fim.reset()

        # 5. Inject t=0 commands. ``vel_all[:, 0, :]`` is ``(P, 3)`` —
        #    each candidate's first-step cmd, broadcast internally to
        #    that candidate's G*(N+1) envs.
        part.broadcast_commands(
            env, vel_all[:, 0, :],
            gait_all[:, 0, :] if gait_all is not None else None,
        )

        G = part.cfg.num_groups
        total_groups = P * G  # used only for progress display denom
        num_terminated_groups = 0
        envs_per_cand = G * (part.param_dim + 1)

        # Per-env return accumulator — only used when the optional
        # action-delta barrier is active. Storing it always (even when
        # ``barrier_coef == 0``) is cheap (one ``num_envs``-long fp32)
        # and lets the caller surface mean return for diagnostics.
        return_acc = torch.zeros(env.num_envs, device=env.device)

        # 6. Rollout loop
        for t in range(total_steps):
            # Per-candidate command at this timestep. When the Bezier
            # config carries a state-feedback gain ``K``, add
            # ``K @ mean_features`` per candidate before broadcasting
            # so the command becomes a (partial) closed-loop function
            # of robot state. The mean is over the candidate's own
            # G*(N+1) envs so all envs in a candidate still receive an
            # identical command (FIM finite-diff requires that;
            # per-env command would break ``(∂s/∂θ)`` at fixed cmd).
            if K_all is not None:
                from consysid.active.state_feedback import extract_features
                features = extract_features(env, fb_features)        # (num_envs, n_feat)
                features_pc = features.reshape(
                    P, envs_per_cand, -1,
                ).mean(dim=1)                                          # (P, n_feat)
                correction = torch.bmm(
                    K_all, features_pc.unsqueeze(-1),
                ).squeeze(-1)                                          # (P, cmd_dim_total)
                vel_t = vel_all[:, t, :] + correction[:, :3]
                if gait_all is not None:
                    gait_t = gait_all[:, t, :] + correction[:, 3:]
                    part.broadcast_commands(env, vel_t, gait_t)
                else:
                    part.broadcast_commands(env, vel_t, None)
            else:
                part.broadcast_commands(
                    env, vel_all[:, t, :],
                    gait_all[:, t, :] if gait_all is not None else None,
                )

            obs = env.obs_manager.get_observation()
            robot_states = env.get_robot_state()
            action = policy.get_action(obs, robot_states)

            # Optional per-step Δa: each candidate's row is replicated
            # across its own ``envs_per_cand`` block so the partition
            # layout stays consistent with the command broadcast.
            if delta_all is not None:
                delta_pt = delta_all[:, t, :]                    # (P, na)
                delta_full = (
                    delta_pt.unsqueeze(1)
                    .expand(P, envs_per_cand, delta_pt.shape[-1])
                    .reshape(P * envs_per_cand, -1)
                )
                # ``action`` may be wider than ``num_actuated`` if the
                # action manager exposes non-actuated channels; only
                # add to the leading actuated DoFs.
                na = delta_full.shape[-1]
                action = action.clone()
                action[:, :na] = action[:, :na] + delta_full[: action.shape[0], :na]

            obs, rewards, terminated, truncated, infos = env.step(action)
            return_acc = return_acc + rewards

            if ksync > 0 and (t + 1) % ksync == 0:
                self._sync_states(env, part)

            state_all = fim.read_state_vector(env)
            fim.accumulate_step(state_all)

            group_reset, reset_ids = part.handle_grouped_resets(
                terminated, truncated,
            )
            if group_reset.any():
                num_terminated_groups += int(group_reset.sum().item())
                fim.mark_terminated(group_reset)

                if len(reset_ids) > 0 and t + 1 < total_steps:
                    nxt = min(t + 1, total_steps - 1)
                    part.broadcast_commands(
                        env, vel_all[:, nxt, :],
                        gait_all[:, nxt, :] if gait_all is not None else None,
                    )

            if (t + 1) % max(1, total_steps // 10) == 0:
                print_active_rollout_step(
                    t, total_steps, num_terminated_groups, total_groups,
                )

        print_active_rollout_done()
        result = fim.compute_fim_objective()
        # Per-candidate mean episode return (mean over each candidate's
        # ``envs_per_cand`` block). Always populated; consumed by the
        # outer optimizer only when ``barrier_coef > 0``.
        result["return_per_cand"] = return_acc.view(P, envs_per_cand).mean(dim=-1)
        return result

    @staticmethod
    def _sync_states(env: Any, part: EnvPartitioner) -> None:
        """Copy main env states to all auxiliary envs within each
        ``(candidate, group)`` pair.

        Called every ``ksync_steps`` to prevent auxiliary envs from
        diverging too far from the main env due to different physical
        parameters. Without periodic sync, the finite-difference
        gradient estimate becomes meaningless.

        Uses JaxRLWorld's ``RobotStateWriter`` API which properly writes
        through to the simulator's internal state (warp arrays + eval_fk
        for Newton, direct API for Genesis).

        Sync is strictly within a ``(p, g)`` pair — different candidates
        see different trajectories, so we must never copy across the
        candidate axis.
        """
        robot_data = env.get_robot_data()
        writer = env.get_robot_state_writer()
        # ``main_indices`` is ``(P, G)``; flatten to ``(P*G,)`` for the
        # writer API (it expects a 1-D ``env_ids`` tensor) and index the
        # robot-data arrays accordingly.
        main_idx_flat = part.main_indices.reshape(-1)                # (P*G,)

        main_pos = robot_data.root_link_pos_w[main_idx_flat]         # (P*G, 3)
        main_quat = robot_data.root_link_quat_w[main_idx_flat]       # (P*G, 4)
        main_lin_vel = robot_data.root_link_lin_vel_w[main_idx_flat] # (P*G, 3)
        main_ang_vel = robot_data.root_link_ang_vel_w[main_idx_flat] # (P*G, 3)
        main_jpos = robot_data.joint_pos[main_idx_flat]              # (P*G, nj)
        main_jvel = robot_data.joint_vel[main_idx_flat]              # (P*G, nj)

        # Thrust-motor robots: the propeller RPM is dynamic state on par
        # with joint velocities and must resync main → aux with the rest
        # of the state. Left unsynced, an aux env's stale RPM difference
        # keeps feeding kinematic divergence that is NOT caused by its
        # θ_i perturbation, biasing the finite-difference Jacobian.
        propeller_term = env.act_manager._terms.get("propeller_thrust")
        if propeller_term is not None:
            rpm = propeller_term._actuator.current_rpm
            main_rpm = rpm[main_idx_flat]                            # (P*G, motors)

        # For each param dim i, copy the main state of every (p, g)
        # pair into the matching aux_i env of that same pair.
        # ``aux_indices(i)`` is ``(P, G)``; flatten matches the order of
        # ``main_idx_flat`` so the row-wise correspondence holds.
        for i in range(part.param_dim):
            aux_idx = part.aux_indices(i).reshape(-1)                # (P*G,)
            writer.set_root_pose(main_pos, main_quat, env_ids=aux_idx)
            writer.set_root_velocity(main_lin_vel, main_ang_vel, env_ids=aux_idx)
            writer.set_dof_positions(main_jpos, env_ids=aux_idx)
            writer.set_dof_velocities(main_jvel, env_ids=aux_idx)
            if propeller_term is not None:
                rpm[aux_idx] = main_rpm
