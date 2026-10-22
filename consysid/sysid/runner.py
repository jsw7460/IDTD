"""Reusable SysID runner — extracted from scripts/run_sysid.py.

Call ``run_sysid()`` from a preset script with all configuration
baked in, eliminating the need for argparse.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from consysid.configs.sysid_term_config import SysIDConfig
from consysid.sysid.clip_dataset import ClipDataset
from consysid.sysid.cma_optimizer import CMAOptimizer, CMAOptimizerConfig
from consysid.sysid.open_loop_evaluator import EvaluatorConfig
from consysid.sysid.parameters import SysIDParameters
from consysid.sysid.result_io import atomic_savez, load_theta_best


@dataclass
class SysIDRunConfig:
    """All settings needed for a single SysID run."""

    trajectory: str | list[str]
    """Trajectory dataset(s) to identify against. Accepts:

    *  ``str`` — single npz path or wandb artifact reference (legacy
       behaviour, bit-identical to before this field was widened).
    *  ``list[str]`` — multiple datasets pooled for joint MLE
       refinement (passive collect + active collect).
       Use case: refinement that wants to leverage *both*
       the large passive dataset (low info per sample but high sample
       count) and the small active dataset (high info per sample, low
       count). See ``dataset_sampling`` for how the per-iter clip
       budget is split.
    """

    sysid_config: SysIDConfig
    sim_type: str
    env_cfgs: Any

    num_clips: int = 4096
    horizon_range: tuple[int, int] = (3, 10)
    seed: int = 42
    output: str = "sysid_result.npz"
    use_wandb: bool = True
    wandb_project: str = "SysID"

    upload_wandb_artifact: bool = False
    """When ``True`` and ``use_wandb`` is ``True``, the final
    ``sysid_result.npz`` is logged to the wandb run as an artifact
    (type ``"sysid-result"``). This makes the result portable across
    servers — downstream stages can refer to it by its returned
    ``entity/project/sysid-result:vN`` reference instead of needing
    a server-local file path. Default ``False`` keeps the legacy
    local-only behaviour for in-place development."""

    wandb_artifact_name: str = "sysid-result"
    """Artifact name in the wandb registry. Default ``"sysid-result"``
    matches the ``artifact_type`` so the wandb UI groups every identification
    output under one node. Override per-run if you want to
    distinguish separate identification runs' artifacts in the registry."""

    wandb_artifact_aliases: list[str] | None = None
    """Optional human-readable aliases attached to the uploaded
    artifact, in addition to the auto-assigned ``vN`` version. Useful
    for tagging a result with a descriptive name like
    ``["pd-gains-friction"]`` so downstream configs can reference it
    stably without needing to look up the version number."""

    dataset_sampling: str = "pooled"
    """How to split ``num_clips`` across multiple ``trajectory`` datasets.
    Ignored when ``trajectory`` is a single string.

    *  ``"pooled"``     — clip budget proportional to each dataset's
                          sample volume (``T_i * N_i``). Each (env, t)
                          cell across the union has equal probability;
                          larger datasets dominate per-iter signal
                          (which equals their MLE weight in expectation).
    *  ``"stratified"`` — equal share per dataset. Same asymptotic
                          estimator as ``pooled`` once iter-count is
                          large, but small datasets get guaranteed
                          per-iter representation so CMA-ES sees their
                          gradient signal early. Recommended when one
                          dataset is much smaller than the others
                          (e.g. active collect ≪ passive collect by ~100x)."""

    theta_init_path: str | None = None
    """Optional path/wandb-ref to a prior ``sysid_result.npz`` whose
    ``theta_best`` is used as the CMA-ES starting point **and** the
    regularization center for this run: a refinement run should start
    from and pull toward the θ obtained in a previous identification
    cycle, not the nominal URDF defaults. Term names and dims must
    match the current ``sysid_config`` (checked on load).

    Accepts a server-local path or a wandb artifact reference like
    ``"<entity>/SysID/sysid-result:vN"``. Resolution goes through
    :func:`consysid.io.resolve_artifact_path`."""

    theta_init: Any = None
    """Optional explicit ``theta_init`` override — bypass the
    file-loading path and inject a value directly from the script.
    Takes precedence over both ``theta_init_path`` and the
    auto-discovery via trajectory metadata. Accepts:

    *  ``np.ndarray`` of shape ``(params.dim,)`` — flat vector.
    *  ``dict[str, float | sequence]`` — partial term-name dict;
       missing terms inherit ``term.default``. Useful when the user
       wants to set just a few values from an earlier identification result without
       touching the rest::

           theta_init={
               "kp_hip": 15.0,
               "kd_hip": 0.8,
               "foot_friction": 0.3,
               # base_inertial / joint_friction / kp_knee / kd_knee
               # all keep term.default
           }

    Both forms are dispatched through
    :meth:`SysIDParameters.build_theta`, so the same machinery serves
    identification init, active-exploration linearisation, and any future
    callers."""

    cma: CMAOptimizerConfig = field(default_factory=CMAOptimizerConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)



def create_env(sim_type: str, env_cfgs: Any):
    """Create a headless env for the specified simulator."""
    from jaxrlworld.rl.evals.sim_initializers import get_initializer

    cfgs = env_cfgs

    sim_name = "Genesis" if sim_type == "genesis" else "Newton"
    init = get_initializer(sim_name)

    cfgs.visualization.show_viewer = False
    cfgs.visualization.record_video = False
    # ``enable_noise`` was removed from ``ObservationConfig`` — noise is
    # now gated per-group via ``ObservationGroupConfig.enable_corruption``.
    # ``disable_corruption`` flips every group to deterministic.
    from jaxrlworld.rl.configs.common_config_classes import disable_corruption
    disable_corruption(cfgs.observation)

    # Strip the policy-training preset's noise events. Identification
    # path *also* drops ``reset_*`` because the open-loop evaluator
    # overwrites t0 state from the dataset per-clip, and residual
    # velocity-field perturbations from ``reset_root`` were observed to
    # inject a ~1.65 % bias into PD identification.
    if hasattr(cfgs, "event"):
        from consysid.configs.event_filter import disable_dr_events
        disable_dr_events(cfgs.event, keep_reset=False)

    init.init_device()
    env = init.init_environment(cfgs)
    env.reset()
    return env


def build_evaluator(sim_type: str, env, eval_config: EvaluatorConfig):
    """Build simulator-specific evaluator."""
    if sim_type == "genesis":
        from consysid.sysid.open_loop_evaluator import GenesisOpenLoopEvaluator
        return GenesisOpenLoopEvaluator(env, eval_config)
    elif sim_type == "newton":
        from consysid.sysid.open_loop_evaluator import NewtonOpenLoopEvaluator
        return NewtonOpenLoopEvaluator(env, eval_config)
    else:
        raise ValueError(f"Unsupported sim: {sim_type}")


def _build_wandb_config(config: SysIDRunConfig, params: SysIDParameters) -> dict:
    """Flatten all config into a single dict for wandb.config."""
    from dataclasses import asdict

    sysid_cfg = config.sysid_config
    term_details = {}
    for name, term in params.terms.items():
        start, end = params._term_slices[name]
        # ``term.bounds`` may be ``(scalar, scalar)`` (broadcast across all
        # dims) or ``(list, list)`` (per-dim). Normalise to per-dim lists
        # for wandb so the panel always sees a flat list of length ``dim``.
        lo_raw, hi_raw = term.bounds
        if isinstance(lo_raw, (int, float)):
            lo_list = [float(lo_raw)] * term.dim
        else:
            lo_list = [float(x) for x in lo_raw]
        if isinstance(hi_raw, (int, float)):
            hi_list = [float(hi_raw)] * term.dim
        else:
            hi_list = [float(x) for x in hi_raw]
        term_details[name] = {
            "dim": term.dim,
            "default": params.theta_0[start:end].tolist(),
            "bounds": [lo_list, hi_list],
            "reg_weight": term.reg_weight,
            "params": {k: str(v) for k, v in term.params.items()},
        }

    lower, upper = params.get_bounds()

    # ``trajectory`` may be ``str`` or ``list[str]``. wandb accepts
    # both; normalising to a list here keeps the panel column types
    # consistent across single-dataset and multi-dataset runs.
    if isinstance(config.trajectory, (str, bytes)):
        traj_for_log: list[str] = [str(config.trajectory)]
    else:
        traj_for_log = [str(t) for t in config.trajectory]

    return {
        "sim_type": config.sim_type,
        "env_cfgs_class": type(config.env_cfgs).__qualname__,
        "trajectory": traj_for_log,
        "sysid_config_class": type(sysid_cfg).__qualname__,
        "num_envs": config.env_cfgs.env.num_envs,
        "num_clips": config.num_clips,
        "horizon_range": list(config.horizon_range),
        "seed": config.seed,
        "dataset_sampling": config.dataset_sampling,
        "total_dim": params.dim,
        "theta_0": params.theta_0.tolist(),
        "bounds_lower": lower.tolist(),
        "bounds_upper": upper.tolist(),
        "term_names": list(params.terms.keys()),
        "terms": term_details,
        "cma/sigma0": config.cma.sigma0,
        "cma/max_iterations": config.cma.max_iterations,
        "cma/population_size": config.cma.population_size,
        "cma/seed": config.cma.seed,
        "cma/tol_fun": config.cma.tol_fun,
        "evaluator/weights": config.evaluator.weights,
    }


def resolve_trajectory(trajectory: str, cache_dir: str = "wandb_datasets") -> str:
    """Backward-compatible alias for :func:`consysid.io.resolve_artifact_path`.

    Older callsites used this function for
    trajectory-dataset resolution specifically. The implementation is
    now generic — any artifact spec (dataset / sysid_result / active
    commands) flows through the same code path. This wrapper keeps the
    legacy import surface stable while routing all calls through the
    shared utility.
    """
    from consysid.io import resolve_artifact_path

    return resolve_artifact_path(trajectory, cache_dir=cache_dir)


def _build_result_dict(
    theta_best: np.ndarray,
    cost_best: float,
    history: list[dict],
    params: SysIDParameters,
    sim_type: str,
    sysid_config_name: str,
) -> dict:
    """Assemble the exact payload written to ``sysid_result.npz``.

    Shared by the periodic checkpoint callback and the final save so that
    every snapshot has a bit-identical schema — active exploration
    can load any of them through the same ``_load_theta_surrogate`` path.
    """
    decoded_fields: dict[str, np.ndarray] = {}
    for name, term in params.terms.items():
        if term.decoder is None:
            continue
        start, end = params._term_slices[name]
        physical = term.decoder(theta_best[start:end])
        for key, val in physical.items():
            decoded_fields[f"{name}__{key}"] = np.asarray(val)

    if history:
        history_costs = np.array([h["best_cost"] for h in history])
        history_thetas = np.array([h["best_theta"] for h in history])
    else:
        history_costs = np.empty((0,), dtype=np.float32)
        history_thetas = np.empty((0, params.dim), dtype=np.float32)

    return dict(
        theta_best=theta_best,
        cost_best=cost_best,
        theta_0=params.theta_0,
        dim=params.dim,
        sysid_config=sysid_config_name,
        sim_type=sim_type,
        # term_names + term_dims together define the saved vector's layout
        # so loaders (load_theta_best in result_io.py) can remap term-by-term
        # even if a different config class places terms in a different order.
        term_names=list(params.terms.keys()),
        term_dims=[t.dim for t in params.terms.values()],
        history_costs=history_costs,
        history_thetas=history_thetas,
        **decoded_fields,
    )


def _make_checkpoint_fn(
    log_dir: str,
    params: SysIDParameters,
    sim_type: str,
    sysid_config_name: str,
    save_interval: int,
) -> Any:
    """Build the per-iteration checkpoint callback passed to the optimizer.

    Mirrors JaxRLWorld's ``base_runner.post_iteration`` split:
        - ``sysid_result_latest.npz`` is rewritten every iteration
        - ``sysid_result_iter{it:06d}.npz`` is emitted every ``save_interval``
    """
    log_path = Path(log_dir)
    latest_path = log_path / "sysid_result_latest.npz"

    def checkpoint_fn(
        iteration: int,
        global_best_theta: np.ndarray,
        global_best_cost: float,
        history: list[dict],
    ) -> None:
        data = _build_result_dict(
            theta_best=np.asarray(global_best_theta, dtype=np.float32),
            cost_best=float(global_best_cost),
            history=history,
            params=params,
            sim_type=sim_type,
            sysid_config_name=sysid_config_name,
        )
        atomic_savez(latest_path, data)
        if save_interval > 0 and iteration % save_interval == 0:
            numbered = log_path / f"sysid_result_iter{iteration:06d}.npz"
            atomic_savez(numbered, data)

    return checkpoint_fn


def _setup_log_dir() -> str:
    """Create timestamped output directory: outputs/sysid/identifications/YYYY-MM-DD/HH-MM-SS."""
    from datetime import datetime

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    log_dir = Path("outputs") / "sysid" / "identifications" / date_str / time_str
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def _discover_theta_init_path(trajectory_path: str) -> str | None:
    """Look inside a resolved trajectory npz for a propagated ``theta_init_path``.

    Populated by the explore → collect pipeline: the active-exploration
    stage's ``active_commands.npz`` records the absolute path of the
    identification result that seeded the exploration, and the
    collector copies that value into the collected trajectory npz. A
    later refinement run can then auto-pick it up without the user
    having to repeat the path in every stage's config.

    Returns ``None`` for trajectories that do not carry this field.
    """
    try:
        data = np.load(trajectory_path, allow_pickle=True)
    except (OSError, ValueError):
        # Artifact or unreadable — leave auto-discovery inert.
        return None
    if "theta_init_path" not in data.files:
        return None
    val = data["theta_init_path"]
    # ``np.savez`` may serialize a single string as a 0-d array; extract it.
    if isinstance(val, np.ndarray) and val.shape == ():
        val = val.item()
    return str(val) if val else None


def run_sysid(config: SysIDRunConfig):
    """Run CMA-ES system identification.

    This is the core logic previously in ``scripts/run_sysid.py:main()``.
    No internal algorithm changes — only the entry point interface changed.
    """
    # Setup output directory
    log_dir = _setup_log_dir()
    print(f"Log directory: {log_dir}")

    # Normalize ``trajectory`` to a list so the multi-dataset code
    # path is the single source of truth. Single-string callers stay
    # bit-identical to the legacy single-npz behaviour (one-element
    # list collapses to the same per-iter sampling).
    if isinstance(config.trajectory, (str, bytes)):
        trajectory_specs = [str(config.trajectory)]
    else:
        trajectory_specs = [str(t) for t in config.trajectory]

    # Resolve every spec (local file or wandb artifact reference).
    trajectory_paths = [resolve_trajectory(t) for t in trajectory_specs]
    # ``trajectory_path`` (singular) is kept as an alias for the first
    # resolved path — used by the theta_init_path auto-discovery below
    # which only inspects a single npz at a time.
    trajectory_path = trajectory_paths[0]

    # Load clips. ``ClipDataset`` accepts a list and handles per-
    # dataset budget allocation according to ``dataset_sampling``.
    print("Loading trajectory clips...")
    clip_dataset = ClipDataset(
        npz_paths=trajectory_paths,
        horizon_range=config.horizon_range,
        num_clips=config.num_clips,
        seed=config.seed,
        sampling=config.dataset_sampling,
    )
    print(f"  Clips per iteration: {len(clip_dataset)} (re-sampled each iteration)")

    # Create env
    print(f"Creating {config.sim_type} env with {config.env_cfgs.env.num_envs} parallel envs...")
    env = create_env(config.sim_type, config.env_cfgs)

    # Build parameters and evaluator
    params = SysIDParameters(env, config.sysid_config)
    evaluator = build_evaluator(config.sim_type, env, config.evaluator)

    # Optional: seed CMA-ES and regularization from a previous run's
    # θ_best (warm-started refinement). ``load_theta_best`` validates term
    # set / per-term dim and transparently reorders across configs that
    # share the same term names but in different ``iter_terms`` order.
    #
    # Resolution precedence:
    #   1. config.theta_init_path  — explicit user override (highest)
    #   2. trajectory npz's embedded ``theta_init_path`` — auto-discovered
    #      from the explore → collect propagation chain
    #   3. None — fall back to nominal URDF defaults
    #
    # When both 1 and 2 are present and differ we honour the explicit
    # override but print a loud warning so the user notices the pipeline
    # metadata is being bypassed.
    # Scan every dataset for an embedded ``theta_init_path`` — passive
    # passive collects often have none (collected before any SysID
    # result existed), while active collects always carry one
    # propagated through the explore → collect chain. Use the first
    # dataset that has it; warn if multiple disagree (common pipeline
    # setups should agree, but we don't silently drop the conflict).
    auto_theta_init_path: str | None = None
    discovered_paths: list[tuple[str, str]] = []
    for tp in trajectory_paths:
        candidate = _discover_theta_init_path(tp)
        if candidate is not None:
            discovered_paths.append((tp, candidate))
            if auto_theta_init_path is None:
                auto_theta_init_path = candidate
    if len(discovered_paths) > 1:
        unique_targets = {p for _, p in discovered_paths}
        if len(unique_targets) > 1:
            print(
                "[run_sysid] WARNING: multiple datasets carry conflicting "
                "theta_init_path metadata:\n"
                + "\n".join(
                    f"    {tp}  →  {p}" for tp, p in discovered_paths
                )
                + f"\n  Using the first one ({auto_theta_init_path}). "
                "Override with config.theta_init_path to silence."
            )
    effective_theta_init_path: str | None
    if config.theta_init_path is not None:
        effective_theta_init_path = config.theta_init_path
        if (
            auto_theta_init_path is not None
            and str(Path(auto_theta_init_path).resolve())
                != str(Path(config.theta_init_path).resolve())
        ):
            print(
                f"[run_sysid] trajectory metadata suggests "
                f"theta_init_path={auto_theta_init_path!r} but "
                f"config.theta_init_path={config.theta_init_path!r} takes "
                f"precedence — pipeline auto-wiring bypassed."
            )
    elif auto_theta_init_path is not None:
        effective_theta_init_path = auto_theta_init_path
        print(
            f"[run_sysid] auto-discovered theta_init_path from trajectory "
            f"metadata: {auto_theta_init_path}"
        )
    else:
        effective_theta_init_path = None

    # Resolution priority for the seeding θ:
    #   1. ``config.theta_init`` (explicit dict / np.ndarray override)  ← highest
    #   2. ``config.theta_init_path`` (str path or wandb ref)
    #   3. ``auto_theta_init_path`` (auto-discovered from trajectory metadata)
    #   4. fall-through: keep ``params.theta_0`` (term defaults)
    #
    # Every form is dispatched through ``params.build_theta(spec)`` so
    # the same ``(dim,)`` vector lands in ``params.theta_0`` regardless
    # of source — every downstream consumer (CMA-ES init, regularisation
    # centre, wandb log) sees identical layout.
    seed_spec: Any = None
    seed_source_label: str = ""
    if config.theta_init is not None:
        seed_spec = config.theta_init
        seed_source_label = "config.theta_init (explicit override)"
        if config.theta_init_path is not None:
            print(
                "[run_sysid] config.theta_init is set; "
                "config.theta_init_path is ignored."
            )
    elif config.theta_init_path is not None:
        seed_spec = config.theta_init_path
        seed_source_label = f"config.theta_init_path={config.theta_init_path!r}"
        if (
            auto_theta_init_path is not None
            and auto_theta_init_path != config.theta_init_path
        ):
            print(
                f"[run_sysid] trajectory metadata suggests "
                f"theta_init_path={auto_theta_init_path!r} but "
                f"config.theta_init_path={config.theta_init_path!r} "
                f"takes precedence — pipeline auto-wiring bypassed."
            )
    elif auto_theta_init_path is not None:
        seed_spec = auto_theta_init_path
        seed_source_label = (
            f"auto-discovered from trajectory metadata "
            f"({auto_theta_init_path!r})"
        )

    if seed_spec is not None:
        try:
            theta_init = params.build_theta(seed_spec)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"theta_init seed could not be resolved:\n"
                f"  source: {seed_source_label}\n"
                f"  spec:   {seed_spec!r}\n"
                f"The path/wandb-ref may be stale or unreachable. Set "
                f"``config.theta_init_path`` / ``config.theta_init`` to "
                f"a valid value, or both to ``None`` to fall back to "
                f"nominal defaults."
            ) from exc
        print(
            f"Seeding CMA-ES + regularization center from "
            f"{seed_source_label}\n"
            f"  prior θ_best: {theta_init}"
        )
        params.theta_0 = theta_init

    # Init wandb
    wandb_run = None
    if config.use_wandb:
        import wandb

        # ``WANDB_NAME`` (e.g. the batch job's RUN_NAME) is used as the wandb
        # GROUP so a seed sweep — same RUN_NAME, different ``--seed`` — lands
        # under one group, with each run named ``<RUN_NAME>_<seed>``. So a
        # txt line ``IDF_SPI_Extended ... --seed 47`` becomes group
        # ``IDF_SPI_Extended`` / run ``IDF_SPI_Extended_47``. Without
        # WANDB_NAME (local runs) fall back to a descriptive single name and
        # no group.
        wandb_name_env = os.environ.get("WANDB_NAME")
        if wandb_name_env:
            run_group = wandb_name_env
            run_name = f"{wandb_name_env}_{config.seed}"
        else:
            run_group = None
            run_name = f"{config.sim_type}_{type(config.sysid_config).__name__}"
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=run_name,
            group=run_group,
            dir=log_dir,
            config=_build_wandb_config(config, params),
        )

    # CMA-ES — build the checkpoint callback so mid-run snapshots share the
    # exact schema of the final save (any snapshot is drop-in loadable).
    sysid_config_qualname = type(config.sysid_config).__qualname__
    checkpoint_fn = _make_checkpoint_fn(
        log_dir=log_dir,
        params=params,
        sim_type=config.sim_type,
        sysid_config_name=sysid_config_qualname,
        save_interval=config.cma.save_interval,
    )

    # ``trajectory_name`` is a display string (used by the console
    # header's ``Path(...).name`` call). Multi-dataset runs collapse
    # the list to "<first basename> (+N more)" so the header stays
    # single-line; the full list lives in ``_build_wandb_config`` /
    # the resolved-paths log emitted by ``ClipDataset``.
    if isinstance(config.trajectory, (str, bytes)):
        trajectory_display = str(config.trajectory)
    else:
        traj_list = list(config.trajectory)
        if len(traj_list) == 1:
            trajectory_display = str(traj_list[0])
        else:
            trajectory_display = (
                f"{traj_list[0]} (+{len(traj_list) - 1} more)"
            )

    # CMA-ES identification run.
    optimizer = CMAOptimizer(
        params, evaluator, clip_dataset, config.cma,
    )
    result = optimizer.optimize(
        wandb_run=wandb_run,
        trajectory_name=trajectory_display,
        sim_type=config.sim_type,
        sysid_config_name=type(config.sysid_config).__name__,
        horizon_range=config.horizon_range,
        log_dir=log_dir,
        checkpoint_fn=checkpoint_fn,
    )


    # Final save — same helper as mid-run checkpoints, so the canonical
    # ``sysid_result.npz`` is byte-compatible with every ``sysid_result_latest``
    # written during the loop.
    final_data = _build_result_dict(
        theta_best=result["theta_best"],
        cost_best=result["cost_best"],
        history=result["history"],
        params=params,
        sim_type=config.sim_type,
        sysid_config_name=sysid_config_qualname,
    )
    output_path = Path(log_dir) / "sysid_result.npz"
    atomic_savez(output_path, final_data)
    print(f"\nResult saved to {output_path}")

    # Optional: upload the final result as a wandb artifact so cross-
    # server pipelines can refer to it by ``entity/project/sysid-result:vN``
    # instead of needing the local file path. Returns the canonical
    # reference string; callers can spot-check it in stdout, copy into
    # downstream config fields (``ActiveExplorationConfig.sysid_result_path``,
    # ``SysIDRunConfig.theta_init_path``, etc.), or harvest it from the
    # wandb run summary which we update below.
    if (
        config.upload_wandb_artifact
        and wandb_run is not None
    ):
        from consysid.io import upload_artifact

        artifact_ref = upload_artifact(
            local_path=str(output_path),
            artifact_name=config.wandb_artifact_name,
            artifact_type="sysid-result",
            wandb_run=wandb_run,
            aliases=config.wandb_artifact_aliases,
            metadata={
                "sim_type": config.sim_type,
                "sysid_config": sysid_config_qualname,
                "trajectory": trajectory_specs,
                "dataset_sampling": config.dataset_sampling,
                "best_cost": float(result["cost_best"]),
                "term_names": list(params.terms.keys()),
            },
        )
        # Surface the canonical ref in the wandb run summary so the
        # next stage's user can copy-paste it without scrolling logs.
        wandb_run.summary["artifact_ref"] = artifact_ref

    if wandb_run is not None:
        wandb_run.finish()

    return result