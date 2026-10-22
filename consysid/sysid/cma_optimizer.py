"""CMA-ES based system identification optimizer.

Wraps cma (pycma) to optimize physical parameters by minimizing
open-loop prediction error in Genesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import cma

from consysid.sysid.clip_dataset import ClipDataset
from consysid.sysid.console import print_header, print_iteration, print_convergence
from consysid.sysid.parameters import SysIDParameters
from consysid.sysid.open_loop_evaluator import OpenLoopEvaluator


@dataclass
class CMAOptimizerConfig:
    """CMA-ES optimizer configuration."""
    sigma0: float = 0.5
    tol_stagnation: int | None = None
    """Override pycma's ``tolstagnation`` (iterations of cost stagnation
    before stopping). Default ``None`` keeps pycma's auto value. Set large
    (e.g. = max_iterations) when using ``sigma_cap``: a capped σ flattens
    cost so the default stagnation check stops early even though gt_error
    is still improving (cost-best ≠ gt_error-best)."""
    sigma_cap: float | None = None
    """Optional hard upper bound on the CMA step size σ, re-applied after
    every ``es.tell``. CMA-ES inflates σ when costs stall; on a biped this
    pushes candidates into violent PD / inertial combos that fall (joint_vel
    blows up → nan → 1e6 penalty → costs stall harder → σ inflates more —
    a spiral). Capping σ keeps candidates near the current mean so rollouts
    stay physical. ``None`` (default) leaves σ unconstrained (Go2 etc.)."""
    max_iterations: int = 300
    population_size: int = 64
    seed: int = 42
    tol_fun: float = 1e-6
    save_interval: int = 50
    """Every this many iterations, dump a numbered checkpoint alongside the
    always-overwritten ``sysid_result_latest.npz``. Matches JaxRLWorld's
    ``runner_cfg.save_interval`` pattern (periodic + latest)."""


class CMAOptimizer:
    """CMA-ES optimization loop for system identification.

    Each iteration:
      1. Sample B candidates from N(μ, σ²C)
      2. For each candidate:
         a. Apply θ to ALL parallel envs (same params everywhere)
         b. Evaluate over clips (distributed across envs)
      3. Update CMA-ES distribution
    """

    def __init__(
        self,
        params: SysIDParameters,
        evaluator: OpenLoopEvaluator,
        clip_dataset: ClipDataset,
        config: CMAOptimizerConfig,
    ):
        self.params = params
        self.evaluator = evaluator
        self.clip_dataset = clip_dataset
        self.config = config

    def optimize(
        self,
        wandb_run: Any | None = None,
        trajectory_name: str = "",
        sim_type: str = "",
        sysid_config_name: str = "",
        horizon_range: tuple[int, int] = (0, 0),
        log_dir: str = "",
        checkpoint_fn: Callable[[int, np.ndarray, float, list], None] | None = None,
    ) -> dict:
        """Run CMA-ES optimization.

        Args:
            wandb_run: Active wandb run for logging. None to skip.
            trajectory_name: Dataset filename for display.
            sim_type: Simulator name for display.
            sysid_config_name: Config class name for display.
            horizon_range: Clip horizon range for display.

        Returns:
            Dict with keys: 'theta_best', 'cost_best', 'history'.
        """
        cfg = self.config
        lower, upper = self.params.get_bounds()
        term_names = list(self.params.terms.keys())

        opts = {
            "seed": cfg.seed,
            "maxiter": cfg.max_iterations,
            "popsize": cfg.population_size,
            "tolfun": cfg.tol_fun,
            "bounds": [lower.tolist(), upper.tolist()],
            "CMA_diagonal": False,
        }
        # When σ is capped (sigma_cap), CMA can't widen the search to escape
        # a flat-cost region, so the default ``tolstagnation`` fires early and
        # stops while gt_error is still improving (cost-best ≠ gt_error-best).
        # ``tol_stagnation`` (when set) overrides it so the run continues to
        # ``maxiter`` instead of giving up on a cost plateau.
        if cfg.tol_stagnation is not None:
            opts["tolstagnation"] = cfg.tol_stagnation

        es = cma.CMAEvolutionStrategy(
            self.params.theta_0.tolist(),
            cfg.sigma0,
            opts,
        )

        history = []
        global_best_cost = float("inf")
        global_best_theta = self.params.theta_0.copy()

        print_header(
            sim_type=sim_type,
            sysid_config_name=sysid_config_name,
            trajectory=trajectory_name,
            dim=self.params.dim,
            popsize=cfg.population_size,
            num_clips=self.clip_dataset.num_clips,
            num_envs=self.evaluator.num_envs,
            horizon_range=horizon_range,
            sigma0=cfg.sigma0,
            term_names=term_names,
            term_slices=self.params._term_slices,
            theta_0=self.params.theta_0,
            lower=lower,
            upper=upper,
            wandb_url=wandb_run.url if wandb_run is not None else None,
            log_dir=log_dir,
        )

        iteration = 0
        while not es.stop():
            # Re-sample clips each iteration (like mini-batch in SGD)
            clips = self.clip_dataset.sample_clips()

            candidates = es.ask()
            n_candidates = len(candidates)
            candidate_thetas = np.array(candidates, dtype=np.float32)

            # Evaluate all candidates in parallel (single physics pass).
            pred_costs_arr = self.evaluator.evaluate_parallel(
                clips, self.params, candidate_thetas,
            )

            # Regularization for all candidates at once
            reg_costs_arr = self.params.regularization(candidate_thetas)
            costs_arr = pred_costs_arr + reg_costs_arr

            # A diverging candidate (e.g. a PD / inertial combo that makes a
            # biped fall → self-collision contacts overflow the solver →
            # nan state → nan cost) must not poison es.tell, which would
            # collapse the whole CMA run from that iteration on. Map any
            # non-finite cost to a large finite penalty so CMA-ES treats it
            # as a very bad point and steers away. Robust robots (Go2
            # quadruped) never fall under any candidate, so their costs stay
            # finite and this is a no-op; bipeds (G1) hit it routinely.
            costs_arr = np.where(np.isfinite(costs_arr), costs_arr, 1e6)

            es.tell(candidates, costs_arr.tolist())

            # Hard-cap σ to break the biped inflation spiral (see sigma_cap):
            # without this CMA keeps widening the search into ever-more
            # violent PD candidates that just fall.
            if cfg.sigma_cap is not None and es.sigma > cfg.sigma_cap:
                es.sigma = cfg.sigma_cap

            iteration += 1
            best_idx = np.argmin(costs_arr)
            best_theta = candidate_thetas[best_idx]

            # Track global best
            if costs_arr[best_idx] < global_best_cost:
                global_best_cost = float(costs_arr[best_idx])
                global_best_theta = best_theta.copy()

            print_iteration(
                iteration=iteration,
                costs_arr=costs_arr,
                pred_costs_arr=pred_costs_arr,
                reg_costs_arr=reg_costs_arr,
                best_idx=best_idx,
                best_theta=best_theta,
                global_best_cost=global_best_cost,
                sigma=float(es.sigma),
                term_names=term_names,
                term_slices=self.params._term_slices,
                theta_0=self.params.theta_0,
                trajectory_name=trajectory_name,
                wandb_url=wandb_run.url if wandb_run is not None else None,
                term_costs=getattr(self.evaluator, '_last_term_costs', None),
                max_iterations=cfg.max_iterations,
                theta_gt=self.params.theta_gt,
                num_episodes=self.clip_dataset.num_episodes,
                term_decoders={
                    n: t.decoder for n, t in self.params.terms.items()
                },
                log_dir=log_dir,
            )

            # wandb logging
            if wandb_run is not None:
                log = {
                    "iteration": iteration,
                    # Population cost stats
                    "cost/best": float(costs_arr[best_idx]),
                    "cost/mean": float(costs_arr.mean()),
                    "cost/std": float(costs_arr.std()),
                    "cost/median": float(np.median(costs_arr)),
                    "cost/worst": float(costs_arr.max()),
                    # Cost breakdown for best candidate
                    "cost/best_pred": float(pred_costs_arr[best_idx]),
                    "cost/best_reg": float(reg_costs_arr[best_idx]),
                    # Population prediction cost stats
                    "cost/pred_mean": float(pred_costs_arr.mean()),
                    "cost/pred_std": float(pred_costs_arr.std()),
                    "cost/reg_mean": float(reg_costs_arr.mean()),
                    # Global best (across all iterations)
                    "cost/global_best": global_best_cost,
                    # CMA-ES internal state
                    "cma/sigma": float(es.sigma),
                    "cma/condition_number": float(es.D.max() / es.D.min()) if es.D.min() > 0 else float("inf"),
                    "cma/axis_ratio": float(es.D.max() / es.D.min()) if es.D.min() > 0 else float("inf"),
                }

                # Per-term cost breakdown for best candidate
                last_term_costs = getattr(self.evaluator, '_last_term_costs', {})
                for name, per_cand in last_term_costs.items():
                    log[f"loss/{name}_best"] = float(per_cand[best_idx])
                    log[f"loss/{name}_mean"] = float(per_cand.mean())

                # Per-term best θ values (named by term)
                theta_gt = self.params.theta_gt
                for name in term_names:
                    start, end = self.params._term_slices[name]
                    for d in range(end - start):
                        log[f"theta_best/{name}_{d}"] = float(best_theta[start + d])
                        log[f"theta_default/{name}_{d}"] = float(self.params.theta_0[start + d])
                    # Distance from default for this term
                    dev = best_theta[start:end] - self.params.theta_0[start:end]
                    log[f"theta_dev/{name}_l2"] = float(np.linalg.norm(dev))
                    # GT error (if available)
                    if theta_gt is not None:
                        gt_vals = theta_gt[start:end]
                        if not np.any(np.isnan(gt_vals)):
                            gt_err = float(np.linalg.norm(best_theta[start:end] - gt_vals))
                            log[f"gt_error/{name}_l2"] = gt_err
                            for d in range(end - start):
                                log[f"gt_error/{name}_{d}"] = float(abs(best_theta[start + d] - gt_vals[d]))

                    # Physical-space metrics for reparameterized terms.
                    term = self.params.terms[name]
                    if term.decoder is not None:
                        dec = term.decoder(best_theta[start:end])
                        dec_def = term.decoder(self.params.theta_0[start:end])
                        if "mass" in dec:
                            log[f"physical/{name}/mass"] = float(dec["mass"])
                            log[f"physical_default/{name}/mass"] = float(dec_def["mass"])
                        if "com" in dec:
                            com_arr = np.asarray(dec["com"]).flatten()
                            com_def = np.asarray(dec_def["com"]).flatten()
                            for d, v in enumerate(com_arr):
                                log[f"physical/{name}/com_{d}"] = float(v)
                                log[f"physical_default/{name}/com_{d}"] = float(com_def[d])
                        if "inertia" in dec:
                            I = np.asarray(dec["inertia"])
                            I_def = np.asarray(dec_def["inertia"])
                            for d in range(3):
                                log[f"physical/{name}/I_diag_{d}"] = float(I[d, d])
                                log[f"physical_default/{name}/I_diag_{d}"] = float(I_def[d, d])
                            log[f"physical/{name}/I_fro"] = float(np.linalg.norm(I))
                            log[f"physical_default/{name}/I_fro"] = float(np.linalg.norm(I_def))
                        if theta_gt is not None and not np.any(np.isnan(theta_gt[start:end])):
                            gt_dec = term.decoder(theta_gt[start:end])
                            if "mass" in dec and "mass" in gt_dec:
                                log[f"physical_err/{name}/mass"] = float(
                                    abs(float(dec["mass"]) - float(gt_dec["mass"]))
                                )
                            if "com" in dec and "com" in gt_dec:
                                log[f"physical_err/{name}/com_l2"] = float(
                                    np.linalg.norm(
                                        np.asarray(dec["com"]) - np.asarray(gt_dec["com"])
                                    )
                                )
                            if "inertia" in dec and "inertia" in gt_dec:
                                log[f"physical_err/{name}/inertia_fro"] = float(
                                    np.linalg.norm(
                                        np.asarray(dec["inertia"]) - np.asarray(gt_dec["inertia"])
                                    )
                                )

                # CMA-ES mean (distribution center)
                cma_mean = np.array(es.mean, dtype=np.float32)
                for name in term_names:
                    start, end = self.params._term_slices[name]
                    for d in range(end - start):
                        log[f"cma_mean/{name}_{d}"] = float(cma_mean[start + d])

                # Per-dimension std (sqrt of diagonal of covariance)
                cma_std = es.sigma * es.D
                for name in term_names:
                    start, end = self.params._term_slices[name]
                    for d in range(end - start):
                        log[f"cma_std/{name}_{d}"] = float(cma_std[start + d])

                wandb_run.log(log, step=iteration)

            history.append({
                "iteration": iteration,
                "best_cost": float(costs_arr[best_idx]),
                "mean_cost": float(costs_arr.mean()),
                "best_theta": best_theta,
                "best_pred_cost": float(pred_costs_arr[best_idx]),
                "best_reg_cost": float(reg_costs_arr[best_idx]),
            })

            # Checkpoint: always refresh latest; additionally emit a numbered
            # snapshot every ``save_interval`` iterations. Identical format to
            # the final save so downstream consumers (active exploration)
            # can load any snapshot via the same path.
            if checkpoint_fn is not None:
                checkpoint_fn(iteration, global_best_theta, global_best_cost, history)

        result = es.result
        theta_best = np.array(result.xbest, dtype=np.float32)

        print_convergence(
            iteration=iteration,
            theta_best=theta_best,
            cost_best=float(result.fbest),
            term_names=term_names,
            term_slices=self.params._term_slices,
            theta_0=self.params.theta_0,
            stop_reason=es.stop(),
        )

        # Log final summary to wandb
        if wandb_run is not None:
            summary = {
                "final/cost_best": float(result.fbest),
                "final/iterations": iteration,
                "final/stop_reason": str(es.stop()),
            }
            for name in term_names:
                start, end = self.params._term_slices[name]
                for d in range(end - start):
                    summary[f"final/theta/{name}_{d}"] = float(theta_best[start + d])
                    summary[f"final/theta_default/{name}_{d}"] = float(self.params.theta_0[start + d])
                    summary[f"final/theta_delta/{name}_{d}"] = float(
                        theta_best[start + d] - self.params.theta_0[start + d]
                    )
            wandb_run.summary.update(summary)

        return {
            "theta_best": theta_best,
            "cost_best": float(result.fbest),
            "history": history,
        }