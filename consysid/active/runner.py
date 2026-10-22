"""Active exploration runner — orchestrates the active-exploration stage.

Loads a pretrained policy, creates the (1 main + N aux) env structure,
and runs CMA-ES to find command sequences that maximize Fisher Information.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from consysid.active.active_optimizer import ActiveCMAConfig, ActiveExplorationOptimizer
from consysid.active.bezier_action_delta import (
    BezierActionDeltaConfig,
    BezierActionDeltaGenerator,
)
from consysid.active.bezier_commands import BezierCommandConfig, BezierCommandGenerator
from consysid.active.env_partitioner import EnvPartitionConfig, EnvPartitioner
from consysid.active.fim_evaluator import FIMConfig, FIMEvaluator
from consysid.configs.sysid_term_config import SysIDConfig
from consysid.sysid.parameters import SysIDParameters
from consysid.sysid.result_io import load_theta_best


@dataclass
class ActiveExplorationConfig:
    """All settings for an active exploration run."""

    # Policy checkpoint
    checkpoint_path: str = ""
    wandb_run_path: str | None = None
    wandb_checkpoint_iter: int | None = None

    # Surrogate theta from identification — file path / wandb ref / explicit override.
    sysid_result_path: str = ""
    """Path or wandb artifact reference to a identification ``sysid_result.npz``
    whose ``theta_best`` is used as the FIM linearisation point.
    Accepts a server-local path or a wandb ref like
    ``"<entity>/SysID/sysid-result:vN"``; resolution goes through
    :func:`consysid.io.resolve_artifact_path`. Ignored if
    :attr:`theta_surrogate` is set."""

    theta_surrogate: Any = None
    """Optional explicit ``theta_surrogate`` override — bypass the
    file-loading path and inject the linearisation θ directly from
    the script. Takes precedence over :attr:`sysid_result_path`.
    Accepts:

    *  ``np.ndarray`` of shape ``(params.dim,)`` — flat vector.
    *  ``dict[str, float | sequence]`` — partial term-name dict;
       missing terms inherit ``term.default`` (URDF nominal).

    Useful for ablation runs that linearise FIM at hand-tuned points
    (e.g. ``θ_default``, ``θ_GT`` if known) instead of a identification
    estimate, or for chained explore→refine→explore′ loops where
    the refinement output is loaded from wandb but plumbed in directly
    without going through a file. Same dispatch as
    :attr:`SysIDRunConfig.theta_init` —
    :meth:`SysIDParameters.build_theta`."""

    uniform_delta: bool = False
    """Ablation: replace the per-parameter FD perturbation vector
    (:meth:`SysIDParameters.build_delta_vec`, bound-width scaled) with a
    uniform ``partition.delta_param`` on every axis. At
    ``decoupling_power=1`` the objective is scale-invariant in the
    per-axis δ, so any performance change isolates the FINITE-DIFFERENCE
    linearization quality — not the objective itself."""

    # SysID parameter config (defines which params to explore)
    sysid_config: SysIDConfig | None = None
    sim_type: str = "newton"
    env_cfgs: Any = None

    # Active exploration
    bezier: BezierCommandConfig = field(default_factory=BezierCommandConfig)
    fim: FIMConfig = field(default_factory=FIMConfig)
    partition: EnvPartitionConfig = field(default_factory=EnvPartitionConfig)
    cma: ActiveCMAConfig = field(default_factory=ActiveCMAConfig)

    # ── Optional action-delta ablation ──────────────────────────────────────
    # When ``action_delta`` is set, CMA-ES jointly optimises the Bezier
    # command AND a per-DoF Bezier action delta added on top of the
    # policy's output (a_t^exec = π(s_t, c_t) + Δa_t). The augmented
    # CMA candidate vector is ``[x_command, x_delta]``; everything else
    # in the explore pipeline (env partition, FIM, multi-Schur cost,
    # wandb logging, checkpointing) runs unchanged. ``barrier_coef=0``
    # (default) leaves the cost untouched even when ``action_delta`` is
    # set — useful for an "uncostrained Δa" ablation. ``barrier_coef>0``
    # adds ``barrier_coef * max(0, α·R_baseline − R_actual)`` to the
    # cost, where ``R_baseline`` is the mean per-env episode return of
    # a one-shot delta=0 / Bezier-init rollout measured before CMA
    # starts. This keeps locomotion stable while Δa explores low-torque
    # regimes that the pretrained policy alone cannot reach.
    action_delta: BezierActionDeltaConfig | None = None
    barrier_coef: float = 0.0
    relative_threshold: float = 0.8

    # Output. Leave as None to auto-create
    # ``outputs/sysid/explorations/YYYY-MM-DD/HH-MM-SS/active_commands.npz``,
    # mirroring the identification ``_setup_log_dir`` convention. Pass an explicit
    # path to override.
    output: str | None = None
    use_wandb: bool = True
    wandb_project: str = "SysID-Active"

    upload_wandb_artifact: bool = False
    """When ``True`` and ``use_wandb`` is ``True``, the final
    ``active_commands.npz`` is logged to the wandb run as an artifact
    (type ``"active-commands"``). Makes the optimised commands
    portable across servers — data collection can refer to it via
    ``TrajectoryCollectorConfig.command_trajectory_path =
    "<entity>/<project>/active-commands:vN"`` instead of needing the
    server-local file path. Default ``False`` keeps the legacy
    local-only behaviour."""
    wandb_artifact_name: str = "active-commands"
    wandb_artifact_aliases: list[str] | None = None

    seed: int = 42


def _setup_log_dir() -> str:
    """Create timestamped output directory: outputs/sysid/explorations/YYYY-MM-DD/HH-MM-SS."""
    from datetime import datetime

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    log_dir = Path("outputs") / "sysid" / "explorations" / date_str / time_str
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def _build_active_result_dict(
    best_x: np.ndarray,
    best_cost: float,
    history: list[dict],
    config: "ActiveExplorationConfig",
    theta_surrogate: np.ndarray,
    delta_vec: np.ndarray,
    latest_x: np.ndarray | None = None,
    latest_cost: float | None = None,
) -> dict:
    """Assemble the exact payload written to ``active_commands.npz``.

    Shared by the periodic checkpoint callback and the final save so
    that every snapshot has a bit-identical schema — the data-collection
    stage and the visualiser can load any of them via the same path.
    """
    opt_names = list(config.bezier.optimized_commands.keys())
    opt_bounds = np.array(
        [list(config.bezier.optimized_commands[n]) for n in opt_names],
        dtype=np.float32,
    )
    default_gait_keys = list(config.bezier.default_gait.keys())
    default_gait_vals = np.array(
        [config.bezier.default_gait[k] for k in default_gait_keys],
        dtype=np.float32,
    )

    if history:
        history_costs = np.array([h["best_cost"] for h in history])
    else:
        history_costs = np.empty((0,), dtype=np.float32)

    # Action-delta ablation metadata — saved only when ``action_delta``
    # is set. data collection (data_collection._load_command_trajectory)
    # detects ``delta_param_dim > 0`` and rebuilds the
    # ``BezierActionDeltaGenerator`` from these fields to decode the
    # ``best_x`` tail. ``delta_param_dim = 0`` (or key absent on legacy
    # npz) signals command-only — collect skips the delta hook entirely
    # and stays bit-identical to the pre-ablation behaviour.
    delta_meta: dict[str, Any] = {"delta_param_dim": 0}
    if config.action_delta is not None:
        ad = config.action_delta
        ncp_d = ad.num_control_points
        nj = ad.num_actuated
        delta_param_dim = ad.num_segments * ncp_d * nj
        delta_meta = {
            "delta_param_dim": int(delta_param_dim),
            "action_delta_segment_duration": float(ad.segment_duration),
            "action_delta_num_segments": int(ad.num_segments),
            "action_delta_bezier_degree": int(ad.bezier_degree),
            "action_delta_num_actuated": int(ad.num_actuated),
            "action_delta_bound": float(ad.bound),
        }

    # Latest-iter snapshot — exposed alongside the global-best fields
    # so a collect script can opt into "use the most recent command"
    # instead of the global-best one via the ``use_latest_command`` flag
    # on TrajectoryCollectorConfig. Empty array when no latest snapshot
    # is available.
    latest_x_arr = (
        np.asarray(latest_x, dtype=np.float32)
        if latest_x is not None
        else np.empty((0,), dtype=np.float32)
    )
    latest_cost_val = float(latest_cost) if latest_cost is not None else float("nan")

    return dict(
        best_x=best_x,
        best_cost=float(best_cost),
        best_fim=float(-best_cost),  # "higher is better" alias = -best_cost
        theta_surrogate=theta_surrogate,
        delta_vec=delta_vec,
        segment_duration=config.bezier.segment_duration,
        num_segments=config.bezier.num_segments,
        bezier_degree=config.bezier.bezier_degree,
        # Velocity-only explore (e.g. G1) sets use_gait=False → best_x has
        # no gait channels. The collect-side decoder MUST rebuild the same
        # layout, so persist the flag (older npz lacking it fall back to
        # the collector's ``command_use_gait`` override / use_gait=True).
        use_gait=bool(config.bezier.use_gait),
        # Primary-channel routing metadata: which env command term the
        # decoded 3-vector drives and the default value of non-optimized
        # dims. Older npz files lack both keys and the collect-side
        # loader falls back to "velocity" / (0, 0, 0) — the locomotion
        # behaviour, bit-identical to before these fields existed.
        command_term=str(config.bezier.command_term),
        primary_defaults=np.asarray(config.bezier.primary_defaults, dtype=np.float32),
        gait_logit_bound=config.bezier.gait_logit_bound,
        optimized_command_names=np.array(opt_names),
        optimized_command_bounds=opt_bounds,
        default_gait_keys=np.array(default_gait_keys),
        default_gait_values=default_gait_vals,
        # State-feedback feature tuple -- empty (length-0) when the
        # explore run used the open-loop Bezier path. Collect-side
        # ``_load_command_trajectory`` keys on ``state_feedback_features
        # in data.files`` so older npz files without this key fall
        # straight through to the legacy no-feedback decoder.
        state_feedback_features=np.array(
            list(config.bezier.state_feedback_features),
        ),
        # Latest-iter snapshot (populated whenever CMA-ES ran).
        latest_x=latest_x_arr,
        latest_cost=latest_cost_val,
        fim_objective=config.fim.objective,
        # Propagate the *original* sysid_result_path spec verbatim so
        # the auto-discovery chain stays portable: when the upstream
        # identification was uploaded to wandb the spec is a wandb ref, when
        # it lived only locally the spec is a path. Either form is
        # resolvable by ``resolve_artifact_path`` at the next stage's
        # load time. Resolving here would bake in a server-local cache
        # path that's invalid on every other server.
        theta_init_path=str(config.sysid_result_path),
        history_costs=history_costs,
        **delta_meta,
    )


def _make_active_checkpoint_fn(
    log_dir: str,
    save_interval: int,
    config: "ActiveExplorationConfig",
    theta_surrogate: np.ndarray,
    delta_vec: np.ndarray,
    wandb_run: Any | None = None,
):
    """Build the per-iteration checkpoint callback passed to the active
    optimizer.

    Exposes a "periodic + latest" save pattern:
        - ``active_commands_latest.npz`` is rewritten every iteration
        - ``active_commands_iter{it:06d}.npz`` is emitted every
          ``save_interval`` iterations
    Both files use the identical schema as the final
    ``active_commands.npz`` so any snapshot plugs directly into the
    data-collection stage / ``viz_bezier`` / etc.

    When ``wandb_run`` is provided AND
    ``config.upload_wandb_artifact`` is set, every numbered local
    snapshot is *also* uploaded to wandb as a fresh artifact version.
    Explore runs are long (often several hours), so a crash or
    preemption mid-training otherwise loses the partial result —
    periodic upload lets downstream stages resume from the latest
    successfully-uploaded version (``entity/project/active-commands:
    latest``) on a different server. Upload failures are caught and
    logged so a transient network hiccup doesn't tear down the explore
    run; the local snapshot is always written first.
    """
    from consysid.sysid.result_io import atomic_savez

    log_path = Path(log_dir)
    latest_path = log_path / "active_commands_latest.npz"

    upload_enabled = (
        config.upload_wandb_artifact and wandb_run is not None
    )

    def checkpoint_fn(
        iteration: int,
        best_x: np.ndarray,
        best_cost: float,
        history: list[dict],
        latest_x: np.ndarray | None = None,
        latest_cost: float | None = None,
    ) -> None:
        data = _build_active_result_dict(
            best_x=np.asarray(best_x, dtype=np.float32),
            best_cost=float(best_cost),
            history=history,
            config=config,
            theta_surrogate=theta_surrogate,
            delta_vec=delta_vec,
            latest_x=latest_x,
            latest_cost=latest_cost,
        )
        atomic_savez(latest_path, data)
        if save_interval > 0 and iteration % save_interval == 0:
            numbered = log_path / f"active_commands_iter{iteration:06d}.npz"
            atomic_savez(numbered, data)

            if upload_enabled:
                # Each numbered snapshot becomes a new wandb artifact
                # version. Aliases combine the user-supplied list with
                # an iteration-specific tag so users can pin a specific
                # iteration via either ``:vN`` (auto) or
                # ``:iter000300`` (semantic). The auto-managed
                # ``latest`` alias always tracks the newest version,
                # which is the recommended ref for downstream data collection
                # collect resuming a crashed explore run.
                from consysid.io import upload_artifact

                aliases = list(config.wandb_artifact_aliases or [])
                aliases.append(f"iter{iteration:06d}")
                try:
                    upload_artifact(
                        local_path=str(numbered),
                        artifact_name=config.wandb_artifact_name,
                        artifact_type="active-commands",
                        wandb_run=wandb_run,
                        aliases=aliases,
                        metadata={
                            "iteration": iteration,
                            "best_cost": float(best_cost),
                            "sim_type": config.sim_type,
                            "sysid_config": (
                                type(config.sysid_config).__name__
                                if config.sysid_config is not None
                                else "<unknown>"
                            ),
                            "sysid_result_path": str(config.sysid_result_path),
                            "fim_objective": config.fim.objective,
                            "is_intermediate": True,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    # Network / wandb-side glitches mustn't tear down
                    # a long explore run. Surface the error inline so
                    # the user notices but continue with local-only
                    # snapshots.
                    print(
                        f"[active checkpoint] wandb upload at iter "
                        f"{iteration} failed: {exc}\n"
                        f"  local snapshot still saved at {numbered}"
                    )

    return checkpoint_fn


@torch.no_grad()
def _measure_baseline_return(
    env: Any,
    policy: Any,
    bezier_gen: BezierCommandGenerator,
    partitioner: EnvPartitioner,
    total_steps: int,
) -> float:
    """One Δa=0 / Bezier-init rollout → mean per-env episode return.

    Used as the locomotion-stability reference (``R_baseline``) for the
    action-delta ablation's relative-threshold barrier. Sized by the
    same env partition that CMA-ES will use, so the broadcast / reset
    layout matches the optimisation rollout exactly. Called only when
    ``ActiveExplorationConfig.barrier_coef > 0``.
    """
    x_init = bezier_gen.get_initial_x()
    vel_np, gait_np = bezier_gen.decode(x_init)
    vel_t = torch.tensor(vel_np, device=partitioner.device, dtype=torch.float32)
    gait_t = (
        torch.tensor(gait_np, device=partitioner.device, dtype=torch.float32)
        if gait_np is not None else None
    )

    P = partitioner.num_candidates
    env.reset()
    # Single command broadcast across every candidate slot — baseline
    # is a property of the policy + bezier-init, not of any specific
    # CMA candidate.
    partitioner.broadcast_commands(
        env, vel_t[0:1].expand(P, -1),
        gait_t[0:1].expand(P, -1) if gait_t is not None else None,
    )

    return_acc = torch.zeros(env.num_envs, device=env.device)
    for t in range(total_steps):
        partitioner.broadcast_commands(
            env, vel_t[t : t + 1].expand(P, -1),
            gait_t[t : t + 1].expand(P, -1) if gait_t is not None else None,
        )
        obs = env.obs_manager.get_observation()
        robot_states = env.get_robot_state()
        action = policy.get_action(obs, robot_states)
        _, rewards, _, _, _ = env.step(action)
        return_acc = return_acc + rewards
    return float(return_acc.mean().item())


def _create_env(sim_type: str, env_cfgs: Any):
    """Create a headless env for active exploration."""
    import importlib
    from jaxrlworld.rl.evals.sim_initializers import get_initializer

    cfgs = env_cfgs
    sim_name = "Genesis" if sim_type == "genesis" else "Newton"
    init = get_initializer(sim_name)

    cfgs.visualization.show_viewer = False
    cfgs.visualization.record_video = False
    # ``enable_noise`` was removed from ``ObservationConfig`` — noise is
    # now gated per-group via ``ObservationGroupConfig.enable_corruption``.
    # ``disable_corruption`` flips every group to deterministic, so the
    # FIM finite-difference signal is not contaminated by obs noise.
    from jaxrlworld.rl.configs.common_config_classes import disable_corruption
    disable_corruption(cfgs.observation)

    # Strip the policy-training preset's noise events. Explore *keeps*
    # ``reset_*`` so the robot starts standing every episode -- without
    # them the trunk drops from the mjcf default 1.5 m on Mujoco /
    # similarly misplaced on Newton, the policy sees an out-of-
    # distribution initial pose, and the FIM rollout becomes a fall
    # rather than an informative excitation.
    if hasattr(cfgs, "event"):
        from consysid.configs.event_filter import disable_dr_events
        disable_dr_events(cfgs.event, keep_reset=True)

    init.init_device()
    env = init.init_environment(cfgs)
    env.reset()
    return env


def _load_policy(env, config: ActiveExplorationConfig):
    """Load pretrained policy from checkpoint."""
    from jaxrlworld.rl.evals.policy_wrappers import PolicyWrapper
    from jaxrlworld.rl.utils.checkpoint import load_checkpoint_metadata, load_runner

    # Resolve checkpoint path
    checkpoint_path = config.checkpoint_path
    if config.wandb_run_path is not None:
        from jaxrlworld.rl.utils.wandb_checkpoint import get_wandb_checkpoint
        checkpoint_path, was_cached = get_wandb_checkpoint(
            wandb_run_path=config.wandb_run_path,
            iteration=config.wandb_checkpoint_iter,
        )
        status = "cached" if was_cached else "downloaded"
        print(f"Using {status} wandb checkpoint: {checkpoint_path}")

    metadata = load_checkpoint_metadata(checkpoint_path)

    # Build eval configs matching checkpoint. ``resolve_cross_sim_config``
    # imports the preset_module recorded in the checkpoint metadata; when
    # that module has been renamed or removed by a later framework
    # refactor (e.g. ``go2_flat`` → ``go2``), the import fails. In that
    # case fall back to ``config.env_cfgs`` -- the explore script builds
    # it from the *current* preset which is what the policy's obs /
    # action layout has to match anyway.
    from jaxrlworld.rl.evals.sim_initializers import resolve_cross_sim_config
    try:
        eval_cfgs = resolve_cross_sim_config(metadata, config.sim_type)
    except (ValueError, ModuleNotFoundError, AttributeError) as exc:
        if config.env_cfgs is None:
            raise
        print(
            f"[_load_policy] cross-sim resolve failed ({exc}); "
            f"falling back to config.env_cfgs."
        )
        from copy import deepcopy
        eval_cfgs = deepcopy(config.env_cfgs)

    # Copy nn/algo from checkpoint
    from copy import deepcopy
    cfgs = deepcopy(eval_cfgs)
    train_config = metadata.get("config", {})
    train_algo = train_config.get("algorithm", {})
    if train_algo:
        cfgs.algorithm = type(cfgs.algorithm).from_dict(train_algo)
    train_nn = train_config.get("nn", {})
    if train_nn:
        cfgs.nn = type(cfgs.nn).from_dict(train_nn)

    runner = load_runner(
        env=env,
        checkpoint_path=checkpoint_path,
        cfgs=cfgs,
        use_wandb=False,
    )
    runner.set_eval_mode()

    policy = PolicyWrapper.from_runner(runner, env.device)
    return policy


def run_active_exploration(config: ActiveExplorationConfig):
    """Run active exploration (active exploration).

    Steps:
        1. Compute param_dim and create env with num_groups * (param_dim + 1) envs
        2. Build SysIDParameters (defines term layout)
        3. Load surrogate theta from an identification result, remapped to the current term layout
        4. Load pretrained policy
        5. Build components (Bezier, Partitioner, FIM, Optimizer)
        6. Run CMA-ES optimization
        7. Save best command sequence
    """
    # 0. Resolve output path — auto-create a timestamped directory if the
    # caller didn't specify one, so explore runs land next to identification
    # runs under outputs/sysid/{explorations,identifications}/<date>/<time>/.
    # Resolve to absolute so every console/header line shows an unambiguous
    # path regardless of the CWD the script was launched from (identification
    # console pattern — see sysid.console.print_iteration's "Log Dir" row).
    if config.output is None:
        log_dir = _setup_log_dir()
        output_path = Path(log_dir) / "active_commands.npz"
    else:
        output_path = Path(config.output)
        log_dir = str(output_path.parent)
    output_path = output_path.resolve()
    log_dir = str(Path(log_dir).resolve())
    print(f"Active exploration log directory: {log_dir}")

    # The Bezier decoder and the partitioner's env injection must target
    # the same command term — a mismatch would silently optimize commands
    # that never reach the env (the injected term stays at its default).
    if config.partition.command_term != config.bezier.command_term:
        raise ValueError(
            f"partition.command_term={config.partition.command_term!r} != "
            f"bezier.command_term={config.bezier.command_term!r}; both must "
            f"name the same env command term."
        )

    # 1. Determine param_dim — must match SysIDParameters.dim, so use the
    # same ``iter_terms`` traversal (walks the full MRO). The previous
    # ``__class__.__dict__`` shortcut only saw fields declared on the leaf
    # config class, silently dropping any term inherited from a parent
    # (e.g. Go2NewtonInertialAllFrictionConfig inherits base_inertial and
    # foot_friction from its base classes). That made total_envs too
    # small for inherited configs and the partitioner later indexed past
    # the actual env tensor — surfacing as a CUDA assert.
    from jaxrlworld.rl.configs.base_config import iter_terms
    from consysid.configs.sysid_term_config import SysIDTermConfig
    sysid_terms = iter_terms(config.sysid_config, SysIDTermConfig)
    sysid_params_dim = sum(t.dim for t in sysid_terms.values())
    print(f"SysID param dim: {sysid_params_dim}")

    # Compute num_envs from partition + parallel-candidate config.
    # ``parallel_candidates=True`` evaluates every CMA-ES candidate in a
    # single rollout, so the env tensor grows by the population size.
    # With ``False`` we fall back to the legacy serial loop (P=1) — the
    # partitioner layout stays the same, just one candidate at a time.
    param_dim = sysid_params_dim
    num_candidates_total = (
        config.cma.population_size if config.cma.parallel_candidates else 1
    )
    total_envs = (
        num_candidates_total * config.partition.num_groups * (param_dim + 1)
    )
    config.env_cfgs.env.num_envs = total_envs

    # Set episode length to cover the full active exploration trajectory
    # so the env doesn't truncate mid-rollout
    total_trajectory_s = config.bezier.num_segments * config.bezier.segment_duration
    config.env_cfgs.env.episode_length_s = total_trajectory_s + 10.0  # margin
    mode_str = "parallel" if config.cma.parallel_candidates else "serial"
    print(
        f"Creating {total_envs} envs "
        f"({num_candidates_total} candidate(s) × {config.partition.num_groups} groups × "
        f"({param_dim}+1) envs/group, {mode_str} mode)"
    )
    print(f"Episode length set to {config.env_cfgs.env.episode_length_s:.0f}s "
          f"(trajectory {total_trajectory_s:.0f}s + 10s margin)")

    # 2. Create env
    env = _create_env(config.sim_type, config.env_cfgs)

    # 3. Build SysIDParameters (needed before loading θ_surrogate so the
    # loader can remap term-by-term into this config's slot layout —
    # different configs that share the same terms may lay them out in
    # different orders, see result_io.load_theta_best).
    params = SysIDParameters(env, config.sysid_config)

    # 4. Resolve the FIM-linearisation θ. Priority:
    #     1. ``config.theta_surrogate`` — explicit dict / np.ndarray
    #     2. ``config.sysid_result_path`` — file or wandb ref
    #
    # Both forms dispatch through ``params.build_theta(spec)`` so the
    # same vector layout lands in env regardless of source. A wandb
    # ref in ``sysid_result_path`` is auto-downloaded; a dict in
    # ``theta_surrogate`` lets the user run an explore ablation at a
    # hand-tuned θ (e.g. URDF nominal, or θ_GT for synthetic checks)
    # without going through a file at all.
    if config.theta_surrogate is not None:
        seed_spec: Any = config.theta_surrogate
        seed_label = "config.theta_surrogate (explicit override)"
        if config.sysid_result_path:
            print(
                "[run_active_exploration] config.theta_surrogate is set; "
                "config.sysid_result_path is ignored."
            )
    elif config.sysid_result_path:
        seed_spec = config.sysid_result_path
        seed_label = (
            f"config.sysid_result_path={config.sysid_result_path!r}"
        )
    else:
        raise ValueError(
            "Active explore needs a θ_surrogate — set "
            "``config.theta_surrogate`` (dict / np.ndarray) or "
            "``config.sysid_result_path`` (str path or wandb ref)."
        )

    theta_surrogate = params.build_theta(seed_spec)
    print(
        f"Loaded surrogate θ from {seed_label}\n"
        f"  values: {theta_surrogate}"
    )

    # 5. Load policy
    policy = _load_policy(env, config)

    # 6. Build components
    # Derive control dt from env config (physics_dt * decimation)
    env_dt = config.env_cfgs.scene.dt * config.env_cfgs.env.decimation
    print(f"Control dt: {env_dt:.4f}s (physics_dt={config.env_cfgs.scene.dt}, "
          f"decimation={config.env_cfgs.env.decimation})")
    bezier_gen = BezierCommandGenerator(config.bezier, dt=env_dt)

    # Per-dim FD step, derived from the ``SysIDParameters`` slicing so
    # the partitioner, apply loop and FIM evaluator agree on which dim
    # maps to which term.
    delta_vec = params.build_delta_vec(fallback=config.partition.delta_param)
    if config.uniform_delta:
        delta_vec = np.full(
            params.dim, float(config.partition.delta_param), dtype=np.float64,
        )
        print(
            f"[uniform_delta] every axis perturbed by "
            f"{config.partition.delta_param} (per-parameter delta_vec bypassed)"
        )
    print(
        f"Param dim: {params.dim}; "
        f"delta range [{float(delta_vec.min()):.4f}, "
        f"{float(delta_vec.max()):.4f}]"
    )
    print(
        f"FIM objective: decoupling_coverage — "
        f"cost = −Σ_i log(c_i + eps), "
        f"c_i = max_s Schur(F_s, {{i}}) / F_ii(s)^α, "
        f"eps={config.fim.schur_eps}"
    )

    partitioner = EnvPartitioner(
        config.partition,
        params.dim,
        env.device,
        delta_vec=delta_vec,
        num_candidates=num_candidates_total,
    )
    fim_evaluator = FIMEvaluator(
        partitioner,
        config.fim,
        env.device,
    )

    # 7. Wandb init
    wandb_run = None
    if config.use_wandb:
        import wandb
        # ``WANDB_NAME`` (e.g. a batch job's RUN_NAME) takes precedence so
        # batch sweeps land under a meaningful per-job run name.
        run_name = os.environ.get("WANDB_NAME") or (
            f"active_{config.sim_type}_{type(config.sysid_config).__name__}"
        )
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                "sim_type": config.sim_type,
                "sysid_config": type(config.sysid_config).__name__,
                "sysid_result_path": config.sysid_result_path,
                "theta_surrogate": theta_surrogate.tolist(),
                "param_dim": params.dim,
                "bezier_dim": bezier_gen.param_dim,
                "num_groups": config.partition.num_groups,
                "num_segments": config.bezier.num_segments,
                "segment_duration": config.bezier.segment_duration,
                "delta_param": config.partition.delta_param,
                "delta_vec": delta_vec.tolist(),
                "ksync_steps": config.partition.ksync_steps,
                "population_size": config.cma.population_size,
                "sigma0": config.cma.sigma0,
                "fim/objective": config.fim.objective,
                "fim/schur_eps": config.fim.schur_eps,
                "fim/coverage_eps": config.fim.coverage_eps,
                "fim/decoupling_power": (
                    config.fim.decoupling_power
                    if np.isscalar(config.fim.decoupling_power)
                    else list(config.fim.decoupling_power)
                ),
                "fim/termination_discount": config.fim.termination_discount,
                # Action-delta knobs. ``action_delta`` is None when it is
                # off; the locomotion barrier is governed by
                # ``barrier_coef`` / ``relative_threshold``.
                "action_delta/enabled": config.action_delta is not None,
                "action_delta/bound": (
                    config.action_delta.bound
                    if config.action_delta is not None else None
                ),
                "action_delta/bezier_degree": (
                    config.action_delta.bezier_degree
                    if config.action_delta is not None else None
                ),
                "action_delta/num_actuated": (
                    config.action_delta.num_actuated
                    if config.action_delta is not None else None
                ),
                "barrier_coef": config.barrier_coef,
                "relative_threshold": config.relative_threshold,
            },
        )

    # 8. Run optimization. ``checkpoint_fn`` emits a mid-run
    # ``active_commands_latest.npz`` every iteration plus numbered
    # snapshots every ``save_interval`` iters — same schema as the
    # final save below so any snapshot is drop-in loadable by data collection
    # / ``viz_bezier`` etc.
    # Optional action-delta ablation: build the delta generator and
    # measure the locomotion-stability baseline (mean per-env episode
    # return with Δa=0 and the Bezier command at its CMA init). The
    # baseline is captured *before* CMA-ES starts so it stays a fixed
    # reference across iterations — relative-α (e.g. 0.8) then keeps
    # the cost barrier scale-free across policies.
    delta_gen = None
    r_baseline = 0.0
    if config.action_delta is not None:
        if (
            config.action_delta.segment_duration != config.bezier.segment_duration
            or config.action_delta.num_segments != config.bezier.num_segments
        ):
            raise ValueError(
                "action_delta and bezier must share segment_duration and "
                "num_segments (lock-step time grid)."
            )
        delta_gen = BezierActionDeltaGenerator(config.action_delta, env_dt)
        if config.barrier_coef > 0.0:
            r_baseline = _measure_baseline_return(
                env, policy, bezier_gen, partitioner,
                bezier_gen.cfg.total_steps(env_dt),
            )
            print(
                f"[active] action-delta ablation enabled — "
                f"R_baseline = {r_baseline:.4f}, threshold = "
                f"{config.relative_threshold * r_baseline:.4f} "
                f"(barrier_coef = {config.barrier_coef})"
            )
        else:
            print(
                "[active] action-delta enabled (barrier_coef = 0 → "
                "unconstrained Δa, no baseline measurement)"
            )

    optimizer = ActiveExplorationOptimizer(
        bezier_gen, partitioner, fim_evaluator, params, config.cma,
        delta_gen=delta_gen,
        barrier_coef=config.barrier_coef,
        relative_threshold=config.relative_threshold,
        r_baseline=r_baseline,
    )
    checkpoint_fn = _make_active_checkpoint_fn(
        log_dir=log_dir,
        save_interval=config.cma.save_interval,
        config=config,
        theta_surrogate=theta_surrogate,
        delta_vec=delta_vec,
        wandb_run=wandb_run,
    )
    result = optimizer.optimize(
        env, policy, theta_surrogate, wandb_run,
        sim_type=config.sim_type,
        sysid_config_name=type(config.sysid_config).__name__,
        log_dir=log_dir,
        checkpoint_fn=checkpoint_fn,
    )

    # 9. Final save — same helper as the mid-run checkpoint so the
    # canonical ``active_commands.npz`` is byte-compatible with every
    # ``active_commands_latest.npz`` written during the loop.
    from consysid.sysid.result_io import atomic_savez
    final_data = _build_active_result_dict(
        best_x=result["best_x"],
        best_cost=result["best_cost"],
        history=result["history"],
        config=config,
        theta_surrogate=theta_surrogate,
        delta_vec=delta_vec,
        latest_x=result.get("latest_x"),
        latest_cost=result.get("latest_cost"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(output_path, final_data)
    print(f"\nSaved active exploration result to {output_path}")
    print(f"  best_x dim: {result['best_x'].shape}")
    print(f"  best_cost: {result['best_cost']:.6f}")
    print(f"  best_fim (higher is better, = -best_cost): {result['best_fim']:.6f}")
    print(f"  To decode: BezierCommandGenerator(config, dt=your_env_dt).decode(best_x)")

    # Optional: upload as wandb artifact for cross-server portability.
    # The returned canonical ref can be plugged directly into
    # ``TrajectoryCollectorConfig.command_trajectory_path`` of the
    # downstream data collection script — same field accepts either
    # local path or wandb ref via ``resolve_artifact_path``.
    if (
        config.upload_wandb_artifact
        and wandb_run is not None
    ):
        from consysid.io import upload_artifact

        artifact_ref = upload_artifact(
            local_path=str(output_path),
            artifact_name=config.wandb_artifact_name,
            artifact_type="active-commands",
            wandb_run=wandb_run,
            aliases=config.wandb_artifact_aliases,
            metadata={
                "sim_type": config.sim_type,
                "sysid_config": type(config.sysid_config).__name__,
                "sysid_result_path": str(config.sysid_result_path),
                "best_cost": float(result["best_cost"]),
                "fim_objective": config.fim.objective,
                "num_segments": config.bezier.num_segments,
                "segment_duration": config.bezier.segment_duration,
            },
        )
        wandb_run.summary["artifact_ref"] = artifact_ref

    if wandb_run is not None:
        wandb_run.finish()

    return result
