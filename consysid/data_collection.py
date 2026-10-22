"""Trajectory data collection from MuJoCo (real-world proxy).

Loads a checkpoint (any simulator), deploys the policy in MuJoCo,
and records raw physical state trajectories for downstream SysID.

Robot type is auto-detected from the checkpoint metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from consysid.active.bezier_commands import (
    BezierCommandConfig,
    BezierCommandGenerator,
)
from consysid.active.state_feedback import extract_features


def _detect_robot_key_local(metadata: dict) -> str:
    """Robot label for trajectory metadata, extended with robots that
    exist only in this workspace's private overlay (JaxRLWorld's public
    ``_detect_robot_key`` must not reference them).

    jaxrlworld import deferred like every other jaxrlworld import in this
    module (they load the sim stacks and are kept out of import time).
    """
    from jaxrlworld.rl.evals.sim_initializers import _detect_robot_key

    task_name = metadata.get("config", {}).get("env", {}).get("task_name", "").lower()
    if "crazyflie" in task_name:
        return "crazyflie"
    return _detect_robot_key(metadata)


@dataclass
class TrajectoryCollectorConfig:
    """Configuration for trajectory collection."""
    checkpoint_path: str = ""
    wandb_run_path: str | None = None
    wandb_checkpoint_iter: int | None = None
    num_envs: int = 64
    num_episodes: int = 100
    max_steps: int | None = None
    """Optional hard cap on the rollout length (control steps). The
    collect loop stops at ``min(num_episodes reached, max_steps)``.
    ``None`` (default) keeps the episode-count-only stop. Set this to
    run a FIXED-duration rollout (e.g. to capture a rare long-surviving
    episode) instead of stopping as soon as ``num_episodes`` short
    episodes have completed."""
    output_path: str | None = None
    """Where to save the collected trajectory npz. ``None`` (default) +
    ``command_trajectory_path`` set → the trajectory lands next to the
    command file at ``<command_trajectory_dir>/collected_trajectory.npz``,
    mirroring the identification / exploration auto-dated convention so a collect
    run inherits the explore run's timestamp directory. If
    ``command_trajectory_path`` is also ``None``, a fresh timestamped
    directory under ``outputs/sysid/collections/<date>/<time>/`` is
    created instead."""
    seed: int = 42
    deterministic: bool = True
    show_viewer: bool = False
    viewer_type: str = "viser"
    viser_port: int = 8080
    xml_path: str | None = None
    command_trajectory_path: str | None = None
    """Path to active_commands.npz from the active-exploration stage. When set, the collector
    injects the decoded Bezier command trajectory at each step instead of
    using the env's automatic command resampling."""
    command_use_gait: bool | None = None
    """Override for ``BezierCommandConfig.use_gait`` when decoding a legacy
    active_commands.npz that predates the ``use_gait`` metadata key. The
    flag governs the best_x layout (gait channels present or not), so a
    velocity-only run (e.g. G1, ``use_gait=False``) whose npz lacks the
    key MUST set this to ``False`` or the decoder slices the wrong stride.
    ``None`` → trust the npz's ``use_gait`` if present, else default True
    (the historical Go2 gait-conditioned behaviour)."""
    use_latest_command: bool = True
    """When ``True``, decode the latest-iter snapshot (final iteration's
    best candidate) saved as ``latest_x`` in active_commands.npz instead
    of the global-best snapshot (``best_x``). Useful when CMA-ES landed
    on a tight global best early and later iterations found a slightly
    worse but preferable command. Default ``True``; no-op on npz files
    that do not carry the latest-snapshot field."""
    upload_wandb: bool = False
    """Upload collected dataset to wandb Artifacts after saving."""
    wandb_artifact_alias: str = ""
    """Alias for the uploaded artifact (e.g. 'newton-multibehavior-default-50t').
    Used to identify and download specific datasets. Required when upload_wandb=True."""
    preset: Any | None = None
    """Optional preset instance (e.g. ``Go2FlatSysIDCollectConfig``) that
    bakes SysID ground-truth values into the env config. When set, takes
    precedence over the metadata-resolved cross-sim config, so ``customize_env``
    is no longer needed for kp/kd/joint_friction/foot_friction. The preset
    must expose ``build()`` (returning a ConfigsForRun) and may optionally
    expose ``runtime_setup(env)`` for overrides that can't be expressed at
    config level (e.g. mjwarp-tensor patches like foot mu)."""


class TrajectoryCollector:
    """Collects physical state trajectories from MuJoCo using a trained policy.

    Uses the evaluator's cross-sim infrastructure to load a checkpoint
    and deploy it in MuJoCo. Robot type is auto-detected from the checkpoint.

    Args:
        config: Collector settings (checkpoint path, num_envs, etc.).
        customize_env: Optional callback ``(env) -> None`` called after
            env creation. Use this for runtime modifications to the
            simulation model (friction, mass, etc.) that must be applied
            directly to the batched model, not the spec.
    """

    def __init__(
        self,
        config: TrajectoryCollectorConfig,
        customize_env: Any | None = None,
    ):
        self.config = config

        # Import JaxRLWorld components
        from jaxrlworld.rl.evals.sim_initializers import (
            detect_sim_type,
            get_initializer,
            resolve_cross_sim_config,
        )
        from jaxrlworld.rl.evals.policy_wrappers import PolicyWrapper
        from jaxrlworld.rl.utils.checkpoint import load_checkpoint_metadata, load_runner

        # Resolve checkpoint_path from wandb if needed
        if config.wandb_run_path is not None:
            from jaxrlworld.rl.utils.wandb_checkpoint import get_wandb_checkpoint
            checkpoint_path, was_cached = get_wandb_checkpoint(
                wandb_run_path=config.wandb_run_path,
                iteration=config.wandb_checkpoint_iter,
            )
            status = "cached" if was_cached else "downloaded"
            print(f"Using {status} wandb checkpoint: {checkpoint_path}")
            config.checkpoint_path = checkpoint_path

        if not config.checkpoint_path:
            raise ValueError("Either checkpoint_path or wandb_run_path must be provided.")

        # Load checkpoint metadata
        self.metadata = load_checkpoint_metadata(config.checkpoint_path)
        self._robot_key = _detect_robot_key_local(self.metadata)
        train_sim_type = detect_sim_type(self.metadata)
        print(f"Checkpoint trained on: {train_sim_type} (robot: {self._robot_key})")

        # Resolve MuJoCo config: explicit preset takes precedence over the
        # metadata-resolved cross-sim config so SysID-specific overrides
        # (PD gains, joint friction, etc.) can be baked in without
        # runtime ``customize_env`` patching.
        if config.preset is not None:
            eval_cfgs = config.preset.build()
            print(
                f"Using explicit preset: {type(config.preset).__name__} "
                f"(metadata-based cross-sim resolve skipped)"
            )
        else:
            eval_cfgs = resolve_cross_sim_config(self.metadata, "mujoco")

        # Prepare cross-sim configs (copy obs/nn/algo from checkpoint).
        # We always do this so the policy sees the obs structure it was
        # trained against, even when an explicit preset is provided.
        eval_cfgs = self._prepare_cross_sim_configs(eval_cfgs, self.metadata)

        # Override entity spec_fn if custom XML is provided
        if config.xml_path is not None:
            self._override_entity_with_xml(eval_cfgs, config.xml_path)

        # Override num_envs and viewer
        eval_cfgs.env.num_envs = config.num_envs
        eval_cfgs.scene.num_envs = config.num_envs
        if getattr(eval_cfgs.scene, "mjlab_scene_cfg", None) is not None:
            eval_cfgs.scene.mjlab_scene_cfg.num_envs = config.num_envs
        eval_cfgs.visualization.show_viewer = config.show_viewer
        if config.show_viewer:
            eval_cfgs.visualization.viewer_type = config.viewer_type
            eval_cfgs.visualization.viser_port = config.viser_port
        eval_cfgs.visualization.record_video = False

        # Disable obs noise for data collection
        from jaxrlworld.rl.configs.common_config_classes import disable_corruption
        disable_corruption(eval_cfgs.observation)

        # Strip the policy-training preset's noise events. Collect
        # *keeps* ``reset_*`` so the robot starts standing every
        # episode -- without them the Mujoco trunk drops from the mjcf
        # default 1.5 m, the policy sees an out-of-distribution initial
        # pose, and the collected trajectory is garbage from step 0.
        if hasattr(eval_cfgs, 'event'):
            from consysid.configs.event_filter import disable_dr_events
            disable_dr_events(eval_cfgs.event, keep_reset=True)

        # Initialize MuJoCo env
        self._init = get_initializer("MujocoEnv")
        self.device = self._init.init_device()
        self.env = self._init.init_environment(
            eval_cfgs,
            seed=config.seed,
        )
        self.env.reset()

        # Apply preset-level runtime overrides (e.g. foot mu patch +
        # ground max() workaround). Only invoked if the preset class
        # exposes ``runtime_setup``; ``Go2FlatSysIDCollectConfig`` does.
        if config.preset is not None and hasattr(config.preset, "runtime_setup"):
            config.preset.runtime_setup(self.env)

        # Apply user-supplied runtime env customizations (friction, mass,
        # etc.). Kept for backward compatibility with the legacy
        # ``customize_env`` collect scripts; new SysID scripts should use
        # ``preset`` instead.
        if customize_env is not None:
            customize_env(self.env)

        # Load runner and build policy wrapper
        self.runner = load_runner(
            env=self.env,
            checkpoint_path=config.checkpoint_path,
            cfgs=eval_cfgs,
            use_wandb=False,
        )
        self.runner.set_eval_mode()

        # Build joint permutation for cross-sim eval
        joint_perm = self._build_joint_permutation()
        self.policy = PolicyWrapper.from_runner(
            self.runner, self.device, joint_perm=joint_perm
        )

        self.eval_cfgs = eval_cfgs

        # Load command trajectory from active exploration
        self._cmd_velocity = None
        self._cmd_gait = None
        # Env command term the primary channel is injected into. Overridden
        # by the ``command_term`` metadata of newer active-commands npz
        # files; the default keeps legacy locomotion artifacts working.
        self._cmd_term_name = "velocity"
        # Optional per-step action delta (only populated when the explore
        # explore run had the action-delta ablation enabled). Tensor
        # shape ``(total_steps, num_actuated)``; added on top of the
        # policy's action at every collect step. ``None`` falls back to
        # the legacy command-only collect path.
        self._cmd_delta = None
        # Optional state-feedback gain matrix ``K`` (only populated when
        # the explore run had a non-empty
        # ``state_feedback_features``). When set, the per-step command
        # becomes ``cmd_bezier(t) + K @ features_t`` instead of the
        # open-loop Bezier baseline.
        self._cmd_K: torch.Tensor | None = None
        self._cmd_K_feature_names: tuple[str, ...] = ()
        # Forward the identification θ_init path the explore run was seeded from,
        # so a later refinement run on the dataset we're about to collect can
        # auto-discover the correct prior θ. Populated in
        # ``_load_command_trajectory`` when the active_commands.npz
        # contains the ``theta_init_path`` key (added in the explore save).
        self._theta_init_path: str | None = None
        if config.command_trajectory_path is not None:
            self._load_command_trajectory(config.command_trajectory_path)

    @staticmethod
    def _override_entity_with_xml(cfgs, xml_path: str) -> None:
        """Replace robot entity's spec_fn to load from a custom XML file.

        The rest of the entity config (init_state, articulation, collisions)
        is preserved from the auto-resolved preset.  If the custom XML does
        not embed mesh assets, they are borrowed from the original spec_fn.
        """
        import mujoco

        xml_path = Path(xml_path).resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"XML not found: {xml_path}")

        entity_name = cfgs.scene.robot_entity_name
        entity = cfgs.scene.entities[entity_name]
        original_spec_fn = entity.spec_fn

        def custom_spec_fn() -> mujoco.MjSpec:
            spec = mujoco.MjSpec.from_file(str(xml_path))
            if original_spec_fn is not None and callable(original_spec_fn):
                orig_spec = original_spec_fn()
                if orig_spec.assets and not spec.assets:
                    spec.assets = orig_spec.assets
            return spec

        entity.spec_fn = custom_spec_fn

    def _load_command_trajectory(self, path: str) -> None:
        """Load and decode command trajectory from active exploration result.

        The npz file carries the full ``BezierCommandConfig`` payload
        (channel names, bounds, gait defaults, ``gait_logit_bound``)
        alongside ``best_x``. All of those must round-trip into the
        decoder — without them ``BezierCommandConfig`` falls back to
        its 4-channel default ``{vx, vy, wz, body_pitch}`` and
        silently mis-slices ``best_x`` whenever the explore run
        optimised more channels (e.g. the 9-channel scalar ablation
        run produces a 206-float vector that the 4-channel decoder
        chops into a misaligned 96-float window).

        Older ``active_commands.npz`` files predate the channel-
        metadata save and use the 4-channel default layout, so the
        fallback path is the right decoder for them — the warning
        below makes the regime explicit so it is never silent.

        ``path`` accepts either a server-local file path or a wandb
        artifact reference (``entity/project/active-commands:vN``);
        ``resolve_artifact_path`` auto-downloads + caches the wandb
        form so cross-server collect runs work without manual rsync.
        """
        from consysid.io import resolve_artifact_path

        local_path = resolve_artifact_path(path)
        if local_path != path:
            print(
                f"[TrajectoryCollector] resolved command_trajectory_path "
                f"{path} → {local_path}"
            )
        data = np.load(local_path, allow_pickle=True)
        # Choose between the global-best snapshot (``best_x``) and the
        # latest-iter snapshot (``latest_x``). When ``use_latest_command``
        # is set but the npz predates the latest snapshot or the field
        # is empty, fall through to the legacy path so old explore
        # runs stay loadable.
        use_latest = bool(self.config.use_latest_command)
        if (
            use_latest
            and "latest_x" in data.files
            and np.asarray(data["latest_x"]).size > 0
        ):
            best_x = data["latest_x"]
            latest_cost = (
                float(data["latest_cost"])
                if "latest_cost" in data.files else float("nan")
            )
            print(
                f"[TrajectoryCollector] use_latest_command=True → "
                f"using latest-iter snapshot (cost={latest_cost:.6f})"
            )
        else:
            best_x = data["best_x"]
            if use_latest:
                print(
                    "[TrajectoryCollector] use_latest_command=True but "
                    "active_commands.npz has no usable latest snapshot — "
                    "falling back to global-best ``best_x``."
                )

        # Forward the identification seed path through the pipeline (see
        # __init__). When the npz does not carry this field the collected
        # trajectory simply won't either, and a later refinement run
        # falls back to explicit / nominal θ as usual.
        if "theta_init_path" in data.files:
            self._theta_init_path = str(data["theta_init_path"])

        cfg_kwargs: dict[str, Any] = dict(
            segment_duration=float(data["segment_duration"]),
            num_segments=int(data["num_segments"]),
            bezier_degree=int(data["bezier_degree"]),
        )

        if "optimized_command_names" in data.files:
            opt_names = [str(n) for n in data["optimized_command_names"]]
            opt_bounds = np.asarray(data["optimized_command_bounds"], dtype=np.float32)
            cfg_kwargs["optimized_commands"] = {
                name: (float(opt_bounds[i, 0]), float(opt_bounds[i, 1]))
                for i, name in enumerate(opt_names)
            }
        else:
            print(
                "WARNING: active_commands.npz lacks 'optimized_command_names' "
                "— falling back to BezierCommandConfig 4-channel default "
                "(vx, vy, wz, body_pitch). This is the correct decoder ONLY "
                "for pre-channel-metadata explore runs; any newer multi-"
                "channel run must regenerate its npz so the layout matches."
            )

        if "default_gait_keys" in data.files:
            gait_keys = [str(k) for k in data["default_gait_keys"]]
            gait_vals = np.asarray(data["default_gait_values"], dtype=np.float32)
            cfg_kwargs["default_gait"] = {
                k: float(v) for k, v in zip(gait_keys, gait_vals)
            }

        if "gait_logit_bound" in data.files:
            cfg_kwargs["gait_logit_bound"] = float(data["gait_logit_bound"])

        # Primary-channel routing (added for the Crazyflie target-position
        # explore runs). Older npz files lack both keys -> the locomotion
        # defaults ("velocity" term, zero defaults) apply unchanged.
        if "command_term" in data.files:
            self._cmd_term_name = str(data["command_term"])
        if "primary_defaults" in data.files:
            cfg_kwargs["primary_defaults"] = tuple(
                float(v) for v in np.asarray(data["primary_defaults"])
            )

        # Optional state-feedback feature tuple (added in 2026-06; legacy
        # active_commands.npz files without it default to the no-feedback
        # path, preserving bit-identical behaviour).
        if "state_feedback_features" in data.files:
            cfg_kwargs["state_feedback_features"] = tuple(
                str(n) for n in data["state_feedback_features"]
            )

        # use_gait governs the best_x stride (gait channels present or
        # not). Trust the npz when it carries the flag; otherwise fall back
        # to the collector's explicit ``command_use_gait`` override, else
        # the historical gait-conditioned default (True). A velocity-only
        # npz that lacks the key with NO override would slice the wrong
        # stride and crash in BezierCommandGenerator.decode.
        if "use_gait" in data.files:
            cfg_kwargs["use_gait"] = bool(data["use_gait"])
        elif self.config.command_use_gait is not None:
            cfg_kwargs["use_gait"] = bool(self.config.command_use_gait)

        config = BezierCommandConfig(**cfg_kwargs)

        env_dt = self._control_step_dt()
        gen = BezierCommandGenerator(config, dt=env_dt)
        vel_np, gait_np = gen.decode(best_x)

        self._cmd_velocity = torch.tensor(vel_np, device=self.device, dtype=torch.float32)
        # Velocity-only decoders (use_gait=False, e.g. G1) return gait_np=None;
        # keep _cmd_gait None so the collect loop's gait-optional path skips it.
        self._cmd_gait = (
            torch.tensor(gait_np, device=self.device, dtype=torch.float32)
            if gait_np is not None
            else None
        )

        # State-feedback gain ``K``: sliced from ``best_x`` tail by the
        # Bezier generator when ``state_feedback_features`` is non-empty.
        # ``decode_K`` returns ``None`` on the no-feedback path so the
        # collect step loop branches around the matmul.
        K_np = gen.decode_K(best_x)
        if K_np is not None:
            self._cmd_K = torch.tensor(
                K_np, device=self.device, dtype=torch.float32,
            )
            self._cmd_K_feature_names = tuple(config.state_feedback_features)
            print(
                f"Loaded state-feedback K: shape={K_np.shape}, "
                f"features={self._cmd_K_feature_names}"
            )
        opt_channel_names = list(config.optimized_commands.keys())
        print(
            f"Loaded active command trajectory: {vel_np.shape[0]} steps "
            f"({config.num_segments} segments x {config.segment_duration}s, "
            f"dt={env_dt}, {len(opt_channel_names)} optimised channels: "
            f"{opt_channel_names})"
        )

        # Optional action-delta: present iff the explore run had the
        # command + action-delta ablation enabled. The npz then carries
        # the delta's Bezier meta and ``delta_param_dim > 0``; we slice
        # the matching tail of ``best_x`` and decode it into a per-step
        # (T, num_actuated) tensor stored on ``self._cmd_delta``. Older
        # explore runs (command-only) have ``delta_param_dim`` absent
        # or zero, so the legacy collect path is unchanged.
        delta_param_dim = int(data["delta_param_dim"]) if "delta_param_dim" in data.files else 0
        if delta_param_dim > 0:
            from consysid.active.bezier_action_delta import (
                BezierActionDeltaConfig,
                BezierActionDeltaGenerator,
            )

            ad_cfg = BezierActionDeltaConfig(
                segment_duration=float(data["action_delta_segment_duration"]),
                num_segments=int(data["action_delta_num_segments"]),
                bezier_degree=int(data["action_delta_bezier_degree"]),
                num_actuated=int(data["action_delta_num_actuated"]),
                bound=float(data["action_delta_bound"]),
            )
            delta_gen = BezierActionDeltaGenerator(ad_cfg, dt=env_dt)
            cmd_param_dim = gen.param_dim
            expected_total = cmd_param_dim + delta_gen.param_dim
            if best_x.shape[0] < expected_total:
                raise ValueError(
                    f"best_x length {best_x.shape[0]} < expected "
                    f"command({cmd_param_dim}) + delta({delta_gen.param_dim}) "
                    f"= {expected_total}. Trajectory metadata and best_x "
                    f"are inconsistent."
                )
            delta_x = best_x[cmd_param_dim : cmd_param_dim + delta_gen.param_dim]
            delta_np = delta_gen.decode(delta_x)
            self._cmd_delta = torch.tensor(delta_np, device=self.device, dtype=torch.float32)
            print(
                f"Loaded action-delta trajectory: shape={delta_np.shape}, "
                f"bound=±{ad_cfg.bound} rad ({ad_cfg.num_segments} "
                f"segments × degree {ad_cfg.bezier_degree})"
            )

    def _prepare_cross_sim_configs(self, eval_cfgs, metadata: dict):
        """Copy obs/nn/algo from checkpoint for cross-sim evaluation."""
        from copy import deepcopy

        cfgs = deepcopy(eval_cfgs)
        train_config = metadata.get("config", {})

        # Copy algorithm config
        train_algo = train_config.get("algorithm", {})
        if train_algo:
            cfgs.algorithm = type(cfgs.algorithm).from_dict(train_algo)

        # Copy nn config
        train_nn = train_config.get("nn", {})
        if train_nn:
            cfgs.nn = type(cfgs.nn).from_dict(train_nn)

        # Copy observation config
        train_obs = train_config.get("observation", {})
        if train_obs:
            train_obs_group = train_obs.get("obs_group")
            if train_obs_group is not None:
                cfgs.observation.obs_group = train_obs_group

        return cfgs

    def _build_joint_permutation(self):
        """Build joint permutation for cross-sim eval if needed."""
        from jaxrlworld.rl.envs.multi_sim_world import (
            _JointPermutation, MultiSimWorld,
        )

        eval_names = list(self.env.act_manager.actuated_joint_names)
        canonical_names = self.metadata.get("canonical_joint_names")
        if canonical_names is None:
            print("WARNING: No canonical_joint_names in metadata, skipping permutation.")
            return None

        def _bare(name: str) -> str:
            return name.rsplit("/", 1)[-1]

        canonical_bare = [_bare(n) for n in canonical_names]
        eval_bare = [_bare(n) for n in eval_names]

        if canonical_bare == eval_bare:
            print("Joint ordering matches canonical — no permutation needed.")
            return None

        joint_slices = MultiSimWorld._find_joint_obs_slices(
            self.env, self.env.num_actions,
        )
        obs_dims = self.env.obs_manager.calculate_obs_dim()

        perm = _JointPermutation(
            canonical_names=canonical_names,
            sim_names=eval_names,
            obs_group_joint_slices=joint_slices,
            obs_group_dims=obs_dims,
            device=self.device,
        )
        print(f"Joint permutation built: {len(canonical_names)} joints")
        return perm

    def _control_step_dt(self) -> float:
        """Control-step ``dt`` in seconds.

        Mujoco path -- ``physics_dt`` is the substep size and
        ``decimation`` is the number of substeps per control step, so
        the control step is the product. The Newton subclass overrides
        this because ``NewtonSceneConfig`` stores the control step
        directly under ``scene.dt`` (substeps live inside the solver
        and are not multiplied here).
        """
        return (
            self.eval_cfgs.scene.physics_dt
            * self.eval_cfgs.env.decimation
        )

    def _read_applied_torques(self) -> torch.Tensor:
        """Read per-joint applied torque after ``env.step``.

        Mujoco path -- ``env.robot_data.applied_torque`` already wraps
        ``qfrc_actuator`` re-indexed into act-manager joint order. The
        Newton subclass overrides this to sum each explicit actuator's
        ``applied_effort`` instead, matching the sim2sim collect's
        torque source to what ``NewtonOpenLoopEvaluator._read_applied
        _torques`` reads during the identification forward sim.
        """
        return self.env.robot_data.applied_torque

    def _extract_pd_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract PD gains (Kp, Kd) in actuated-joint order.

        Reads from JaxRLWorld's explicit actuator instances (e.g.
        DelayedPDActuator), which hold the actual stiffness/damping
        used for PD torque computation. Falls back to MuJoCo model
        actuator parameters if no explicit actuators are present.
        """
        act_mgr = self.env.act_manager
        num_joints = act_mgr.total_action_dim

        kp = np.zeros(num_joints, dtype=np.float32)
        kd = np.zeros(num_joints, dtype=np.float32)

        if act_mgr._has_explicit_actuators:
            for actuator, joint_idx in act_mgr._actuators:
                # actuator.stiffness/damping: (num_envs, num_group_joints)
                # Take from env 0, they're the same across envs
                kp_vals = actuator.stiffness[0].cpu().numpy()
                kd_vals = actuator.damping[0].cpu().numpy()
                for i, j in enumerate(joint_idx):
                    kp[j] = kp_vals[i]
                    kd[j] = kd_vals[i]
        else:
            # Fallback: read from MuJoCo model (implicit actuators)
            mj_model = self.env.scene_manager.mj_model
            joint_names = list(act_mgr.actuated_joint_names)

            def _bare(name: str) -> str:
                return name.rsplit("/", 1)[-1]

            mj_jnt_id_to_act_idx = {}
            for j, name in enumerate(joint_names):
                for mj_jid in range(mj_model.njnt):
                    if _bare(mj_model.joint(mj_jid).name) == _bare(name):
                        mj_jnt_id_to_act_idx[mj_jid] = j
                        break

            for act_idx in range(mj_model.nu):
                jnt_id = mj_model.actuator_trnid[act_idx, 0]
                if jnt_id in mj_jnt_id_to_act_idx:
                    j = mj_jnt_id_to_act_idx[jnt_id]
                    kp[j] = mj_model.actuator_gainprm[act_idx, 0]
                    kd[j] = -mj_model.actuator_biasprm[act_idx, 2]

        return kp, kd

    def _read_motor_rpm(self) -> torch.Tensor | None:
        """Current per-motor RPM state, ``(num_envs, num_motors)``.

        ``None`` (default) means the robot has no thrust-motor actuator
        and the ``motor_rpm`` channel is not recorded — the npz layout
        for joint-actuated robots is unchanged. Sim-specific collectors
        override this for propeller robots.
        """
        return None

    @torch.no_grad()
    def collect(self) -> dict[str, Any]:
        """Run rollouts and collect trajectory data.

        Returns:
            Dictionary containing trajectory arrays and metadata, ready for np.savez.
        """
        config = self.config
        num_envs = config.num_envs
        target_episodes = config.num_episodes

        # Buffers: list of per-step arrays
        all_root_pos = []       # [T, N, 3]
        all_root_quat = []      # [T, N, 4]
        all_root_lin_vel = []   # [T, N, 3]
        all_root_ang_vel = []   # [T, N, 3]
        all_joint_pos = []      # [T, N, num_joints]
        all_joint_vel = []      # [T, N, num_joints]
        all_joint_torques = []  # [T, N, num_joints]
        all_actions = []        # [T, N, num_actions]
        all_processed_actions = []  # [T, N, num_actions]
        all_policy_obs = []     # [T, N, obs_dim]
        all_dones = []          # [T, N]  — True at steps where episode ended
        all_motor_rpm = []      # [T, N, num_motors] — thrust robots only

        # Episode tracking
        completed_episodes = 0
        ep_reward_buf = torch.zeros(num_envs, device=self.device)
        ep_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        ep_returns = []
        ep_lengths = []

        # Get initial obs
        obs = self.env.obs_manager.get_observation()
        robot_states = self.env.get_robot_state()

        # Command injection setup
        has_cmd_traj = self._cmd_velocity is not None
        if has_cmd_traj:
            cmd_len = self._cmd_velocity.shape[0]
            all_env_ids = torch.arange(num_envs, device=self.device)
            print(f"Collecting with active commands ({cmd_len} steps, looping)")
        else:
            cmd_len = 0

        print(f"Collecting {target_episodes} episodes across {num_envs} envs...")

        step = 0
        max_steps = config.max_steps
        while completed_episodes < target_episodes and (
            max_steps is None or step < max_steps
        ):
            # Inject command from active exploration trajectory. When
            # the explore run used the state-feedback variant,
            # add ``K @ features_t`` per-env on top of the Bezier
            # baseline -- ``features_t`` is computed afresh every step
            # so each env's command tracks its own state in closed loop.
            if has_cmd_traj:
                t = step % cmd_len
                vel_t = self._cmd_velocity[t].unsqueeze(0).expand(num_envs, -1)
                # Robots without a gait command term (e.g. G1 velocity-only)
                # leave ``_cmd_gait`` None — inject velocity alone.
                has_gait = self._cmd_gait is not None
                if has_gait:
                    gait_t = self._cmd_gait[t].unsqueeze(0).expand(num_envs, -1)
                if self._cmd_K is not None:
                    features = extract_features(
                        self.env, self._cmd_K_feature_names,
                    )                                                    # (num_envs, n_feat)
                    # K shape (14, n_feat) for gait robots, (3, n_feat) for
                    # velocity-only. features @ K^T -> (num_envs, cmd_dim)
                    correction = features @ self._cmd_K.T
                    vel_t = vel_t + correction[:, :3]
                    if has_gait:
                        gait_t = gait_t + correction[:, 3:]
                cmd = {self._cmd_term_name: vel_t}
                if has_gait:
                    cmd["gait"] = gait_t
                self.env.command_manager.set_commands(
                    env_ids=all_env_ids, **cmd,
                )

            # Get action from policy
            obs = self.env.obs_manager.get_observation()
            robot_states = self.env.get_robot_state()
            action = self.policy.get_action(
                obs, robot_states, deterministic=config.deterministic
            )


            # Record state BEFORE stepping (state at which action was taken).
            # Kinematic state + action are correctly aligned at this point —
            # ``state(t)`` is the input that ``action(t)`` will be applied
            # to, so they belong in the same row of the trajectory tensor.
            #
            # Use ``env.robot_data`` (the cached ``MujocoRobotData``
            # wrapper) instead of ``scene_manager.robot.data`` (raw
            # mjlab ``EntityData``). The wrapper reindexes every joint-
            # axis field by ``act_manager.joint_ids`` so columns line up
            # with the simulation-side
            # ``open_loop_evaluator._read_applied_torques`` output, which
            # is in act-manager order. Without the wrapper the columns
            # would follow URDF ``<joint>`` declaration order — a Go2-
            # specific coincidence that the rest of the SysID pipeline
            # silently relied on; switching to the wrapper makes the
            # alignment explicit so a different robot / URDF order
            # cannot silently swap columns.
            robot_data = self.env.robot_data
            all_root_pos.append(robot_data.root_link_pos_w.cpu().numpy().copy())
            all_root_quat.append(robot_data.root_link_quat_w.cpu().numpy().copy())
            all_root_lin_vel.append(robot_data.root_link_lin_vel_w.cpu().numpy().copy())
            all_root_ang_vel.append(robot_data.root_link_ang_vel_w.cpu().numpy().copy())
            all_joint_pos.append(robot_data.joint_pos.cpu().numpy().copy())
            all_joint_vel.append(robot_data.joint_vel.cpu().numpy().copy())
            all_actions.append(action.cpu().numpy().copy())
            actor_obs = obs["actor"] if isinstance(obs, dict) else obs
            all_policy_obs.append(actor_obs.cpu().numpy().copy())
            # Thrust-motor robots carry hidden actuator state (motor RPM).
            # Record the PRE-step RPM alongside the kinematic snapshot so
            # open-loop replay can restore the exact actuator state a
            # clip's first action was computed against — with a motor
            # time constant of ~5 control steps, clip horizons of 1-5
            # steps live entirely inside the RPM transient and skipping
            # the restore would bias every motor-parameter estimate.
            rpm = self._read_motor_rpm()
            if rpm is not None:
                all_motor_rpm.append(rpm.cpu().numpy().copy())
            # NOTE: torque is intentionally NOT appended here — before
            # the step it still holds the *previous* action's torque (or
            # zero on the very first iteration after reset). Recording
            # it post-step (below) gives ``torque(t) = compute(action(t),
            # state(t))`` which matches what ``open_loop_evaluator``
            # reads on the simulation side via ``_read_applied_torques``
            # after ``_apply_actions(action_t)``. Without this fix the
            # torque column lags the state/action columns by one step,
            # injecting a silent bias into any identification cost with
            # non-zero ``joint_torques`` weight.

            # Optional per-step action delta from the explore run.
            # ``self._cmd_delta`` is ``None`` for command-only explore
            # runs (legacy + most production), so this branch is a no-op
            # there and the collect path stays bit-identical to before
            # the ablation hook landed. When set, ``Δa(t)`` is added on
            # top of the policy's PD target so the collected trajectory
            # carries the same low-torque excitation that the explore
            # objective discovered.
            if self._cmd_delta is not None:
                t_delta = step % self._cmd_delta.shape[0]
                delta_t = self._cmd_delta[t_delta]  # (num_actuated,)
                na = delta_t.shape[0]
                action = action.clone()
                action[:, :na] = action[:, :na] + delta_t.unsqueeze(0).expand(num_envs, -1)

            # Step environment — this triggers actuator.compute(action(t),
            # state(t)) and writes the resulting torque into qfrc_actuator,
            # which we read immediately below to fill in the ``torque(t)``
            # slot of the trajectory tensor.
            obs, rewards, terminated, truncated, infos = self.env.step(action)
            # Reindexed to act-manager joint order (see ``robot_data``
            # comment above) — ``applied_torque`` ≡ ``qfrc_actuator[:,
            # joint_ids]`` on ``MujocoRobotData``.
            all_joint_torques.append(
                self._read_applied_torques().cpu().numpy().copy()
            )
            all_processed_actions.append(
                self.env.act_manager.processed_actions.cpu().numpy().copy()
            )
            robot_states = self.env.get_robot_state()
            step += 1

            # Accumulate per-env episode stats
            ep_reward_buf += rewards
            ep_length_buf += 1

            # Track episode completions
            reset_mask = terminated | truncated
            all_dones.append(reset_mask.cpu().numpy().copy())
            if reset_mask.any():
                done_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
                for idx in done_ids:
                    ep_returns.append(ep_reward_buf[idx].item())
                    ep_lengths.append(ep_length_buf[idx].item())
                ep_reward_buf[done_ids] = 0.0
                ep_length_buf[done_ids] = 0
                completed_episodes += len(done_ids)
                self.policy.notify_reset(reset_mask.cpu().numpy())

                # Re-inject commands after reset (reset clears _externally_controlled)
                if has_cmd_traj:
                    t_next = step % cmd_len
                    vel_t = self._cmd_velocity[t_next].unsqueeze(0).expand(num_envs, -1)
                    cmd = {self._cmd_term_name: vel_t[done_ids]}
                    if self._cmd_gait is not None:
                        gait_t = self._cmd_gait[t_next].unsqueeze(0).expand(num_envs, -1)
                        cmd["gait"] = gait_t[done_ids]
                    self.env.command_manager.set_commands(
                        env_ids=done_ids, **cmd,
                    )

            if step % 100 == 0:
                print(f"  Step {step}: {completed_episodes}/{target_episodes} episodes")

        mean_return = np.mean(ep_returns) if ep_returns else 0.0
        mean_length = np.mean(ep_lengths) if ep_lengths else 0.0
        print(f"Collection done: {completed_episodes} episodes, {step} total steps")
        print(f"  Mean return: {mean_return:.2f}, Mean episode length: {mean_length:.1f}")

        # Stack arrays: [T, N, ...]
        trajectory = {
            "root_pos": np.stack(all_root_pos, axis=0),
            "root_quat": np.stack(all_root_quat, axis=0),
            "root_lin_vel": np.stack(all_root_lin_vel, axis=0),
            "root_ang_vel": np.stack(all_root_ang_vel, axis=0),
            "joint_pos": np.stack(all_joint_pos, axis=0),
            "joint_vel": np.stack(all_joint_vel, axis=0),
            "joint_torques": np.stack(all_joint_torques, axis=0),
            "actions": np.stack(all_actions, axis=0),
            "processed_actions": np.stack(all_processed_actions, axis=0),
            "policy_observations": np.stack(all_policy_obs, axis=0),
            "dones": np.stack(all_dones, axis=0),
        }
        if all_motor_rpm:
            trajectory["motor_rpm"] = np.stack(all_motor_rpm, axis=0)

        # Add metadata
        joint_names = list(self.env.act_manager.actuated_joint_names)
        canonical_names = self.metadata.get("canonical_joint_names", joint_names)
        dt = self._control_step_dt()

        trajectory["timestamps"] = np.arange(step, dtype=np.float64) * dt
        trajectory["joint_names"] = np.array(joint_names, dtype=object)
        trajectory["canonical_joint_names"] = np.array(canonical_names, dtype=object)
        trajectory["dt"] = np.array(dt)
        trajectory["sim_duration"] = np.array(step * dt)
        trajectory["data_frequency"] = np.array(int(round(1.0 / dt)))
        trajectory["num_envs"] = np.array(num_envs)
        trajectory["num_steps"] = np.array(step)
        trajectory["num_episodes"] = np.array(completed_episodes)
        trajectory["robot_type"] = self._robot_key

        # PD gains from actuators
        kp, kd = self._extract_pd_gains()
        trajectory["pd_gain_kp"] = kp
        trajectory["pd_gain_kd"] = kd

        # Forward the identification seed path (picked up from the active
        # commands file in ``_load_command_trajectory``) so a later
        # refinement run can auto-discover it from the trajectory alone.
        if self._theta_init_path is not None:
            trajectory["theta_init_path"] = np.array(self._theta_init_path)

        return trajectory

    @torch.no_grad()
    def visualize(self) -> None:
        """Play the decoded active-command trajectory in the viewer.

        Uses :class:`jaxrlworld.rl.vis.viser.play_viewer.ViserPlayViewer`
        — same viewer that ``PolicyEvaluator.play()`` builds — so the
        full GUI (Speed Slower / 1x / Faster, Pause, Reset, FPS / SPS
        readouts, term overlays) is available exactly like in the
        evaluator scripts under ``jaxrlworld/scripts/evaluation/``.

        Time-indexed Bezier commands are injected via a thin
        ``ViserPlayViewer`` subclass that overrides ``_execute_step``
        to set ``env.command_manager`` commands before the viewer's
        own policy / step path runs.

        Requires:
          * ``config.command_trajectory_path`` set — otherwise there
            are no commands to inject.
          * ``config.show_viewer=True`` — otherwise the underlying env
            is headless and the viewer has nothing to render against.
        """
        if self._cmd_velocity is None:
            raise RuntimeError(
                "visualize() requires command_trajectory_path to be set "
                "so the Bezier curve can be decoded."
            )
        if not self.config.show_viewer:
            raise RuntimeError(
                "visualize() expects show_viewer=True — otherwise there "
                "is no viewer to render into."
            )

        from jaxrlworld.rl.vis.viser.play_viewer import ViserPlayViewer

        env = self.env
        cmd_velocity = self._cmd_velocity
        cmd_gait = self._cmd_gait
        cmd_term = self._cmd_term_name
        # Optional action-delta companion (None on command-only runs).
        cmd_delta = self._cmd_delta
        cmd_delta_len = cmd_delta.shape[0] if cmd_delta is not None else 0
        # Optional state-feedback gain (None unless the explore run set
        # state_feedback_features). Same closed-loop K @ features_t as
        # collect() so the preview matches the collected behaviour.
        cmd_K = self._cmd_K
        cmd_K_feats = self._cmd_K_feature_names
        device = self.device
        num_envs = self.config.num_envs
        cmd_len = cmd_velocity.shape[0]
        all_env_ids = torch.arange(num_envs, device=device)

        # Subclass ``ViserPlayViewer`` to weave Bezier-command injection
        # into the viewer's main step. Pre-step (before policy.get_action)
        # we set the time-indexed velocity / gait command on every env;
        # post-step we re-inject onto reset envs (env.step clears the
        # externally-controlled command flag on reset, mirroring the
        # legacy headless visualize() loop). Looping is handled by
        # ``cmd_step % cmd_len`` so the curve wraps forever; Speed
        # Slower / 1x / Faster from the parent class still scale the
        # physics-time budget, so the same number of injections happens
        # at the rate the user picks.
        deterministic = self.config.deterministic

        class _BezierCommandViewer(ViserPlayViewer):
            """Same as ``ViserPlayViewer`` but injects the time-indexed
            Bezier command before each policy step and re-injects the
            next-step command on envs that just reset (``env.step``
            clears the externally-controlled command flag on reset).
            We re-implement the obs → policy → step body locally rather
            than calling ``super()._execute_step`` because that base
            method does not expose the per-step ``terminated``/
            ``truncated`` masks the re-injection needs.
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._cmd_step = 0

            def _execute_step(viewer_self) -> bool:  # noqa: N805
                import traceback as _tb

                # Pre-step: set t-th Bezier command on every env. Add the
                # closed-loop state-feedback correction K @ features_t
                # (computed afresh from each env's current state) when the
                # explore run used state feedback, mirroring collect().
                t = viewer_self._cmd_step % cmd_len
                vel_t = cmd_velocity[t].unsqueeze(0).expand(num_envs, -1)
                has_gait = cmd_gait is not None
                if has_gait:
                    gait_t = cmd_gait[t].unsqueeze(0).expand(num_envs, -1)
                if cmd_K is not None:
                    features = extract_features(env, cmd_K_feats)
                    correction = features @ cmd_K.T          # (num_envs, cmd_dim)
                    vel_t = vel_t + correction[:, :3]
                    if has_gait:
                        gait_t = gait_t + correction[:, 3:]
                cmd = {cmd_term: vel_t}
                if has_gait:
                    cmd["gait"] = gait_t
                env.command_manager.set_commands(
                    env_ids=all_env_ids, **cmd,
                )

                try:
                    with torch.no_grad():
                        obs = env.obs_manager.get_observation()
                        robot_states = env.get_robot_state()
                        action = viewer_self.policy.get_action(
                            obs, robot_states, deterministic=deterministic,
                        )
                        # Optional per-step action delta (None on legacy
                        # command-only runs). Same overlay as the
                        # ``collect()`` path so the viewer previews
                        # exactly what would land in the dataset.
                        if cmd_delta is not None:
                            t_d = viewer_self._cmd_step % cmd_delta_len
                            delta_t = cmd_delta[t_d]
                            na = delta_t.shape[0]
                            action = action.clone()
                            action[:, :na] = action[:, :na] + delta_t.unsqueeze(0).expand(
                                num_envs, -1,
                            )
                        _, _, terminated, truncated, _ = env.step(action)
                        viewer_self._step_count += 1
                        viewer_self._stats_steps += 1

                        reset_mask = terminated | truncated
                        if reset_mask.any():
                            viewer_self.policy.notify_reset(
                                reset_mask.cpu().numpy(),
                            )
                            # Re-inject next-step command on freshly
                            # reset envs so the Bezier curve resumes
                            # immediately.
                            t_next = (viewer_self._cmd_step + 1) % cmd_len
                            vel_next = cmd_velocity[t_next].unsqueeze(0).expand(
                                num_envs, -1,
                            )
                            done_ids = reset_mask.nonzero(
                                as_tuple=False,
                            ).squeeze(-1)
                            cmd = {cmd_term: vel_next[done_ids]}
                            if cmd_gait is not None:
                                gait_next = cmd_gait[t_next].unsqueeze(0).expand(
                                    num_envs, -1,
                                )
                                cmd["gait"] = gait_next[done_ids]
                            env.command_manager.set_commands(
                                env_ids=done_ids, **cmd,
                            )
                        viewer_self._cmd_step += 1
                        return True
                except Exception:
                    viewer_self._last_error = _tb.format_exc()
                    print(
                        f"[PlayViewer] Exception during step:\n"
                        f"{viewer_self._last_error}",
                    )
                    viewer_self.pause()
                    return False

        play_scene = self._init.create_play_scene(env)
        viewer = _BezierCommandViewer(
            env=env,
            play_scene=play_scene,
            policy=self.policy,
            port=self.config.viser_port,
        )
        print(
            f"Launching ViserPlayViewer on port {self.config.viser_port}: "
            f"{cmd_len} Bezier-command steps will loop forever. "
            f"Use the GUI's Speed buttons (Slower / 1x / Faster) to "
            f"adjust playback. Ctrl+C in the terminal exits."
        )
        try:
            viewer.run()
        except KeyboardInterrupt:
            print("\nViewer interrupted.")

    def _resolve_output_path(self) -> Path:
        """Decide where the collected npz lands.

        Priority:
          1. Explicit ``output_path`` — respected verbatim.
          2. Falls alongside ``command_trajectory_path`` in that file's
             parent directory (so the explore run and its data
             collection share one timestamp folder).
          3. Fresh ``outputs/sysid/collections/<date>/<time>/`` if neither
             is set — mirrors the identification ``_setup_log_dir`` convention.
        """
        if self.config.output_path is not None:
            return Path(self.config.output_path)
        if self.config.command_trajectory_path is not None:
            cmd_path = Path(self.config.command_trajectory_path)
            return cmd_path.parent / "collected_trajectory.npz"
        from datetime import datetime
        now = datetime.now()
        log_dir = (
            Path("outputs") / "sysid" / "collections"
            / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
        )
        return log_dir / "collected_trajectory.npz"

    def collect_and_save(self) -> Path:
        """Collect trajectories and save to npz file.

        Returns:
            Path to saved file.
        """
        trajectory = self.collect()

        output_path = self._resolve_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(output_path), **trajectory)

        print(f"Saved trajectory to {output_path}")
        print(f"  Shapes:")
        for key, val in trajectory.items():
            if isinstance(val, np.ndarray) and val.ndim > 0:
                print(f"    {key}: {val.shape}")

        if self.config.upload_wandb:
            self._upload_to_wandb(output_path, trajectory)

        return output_path

    def _upload_to_wandb(self, file_path: Path, trajectory: dict | None = None) -> None:
        """Upload dataset to wandb Artifacts with metadata."""
        import wandb

        # Build metadata from trajectory data or config
        metadata = {
            "robot_type": self._robot_key,
            "num_envs": self.config.num_envs,
            "num_episodes": self.config.num_episodes,
            "seed": self.config.seed,
            "deterministic": self.config.deterministic,
            "output_path": str(file_path),
        }
        if trajectory is not None:
            metadata.update({
                "num_steps": int(trajectory.get("num_steps", 0)),
                "dt": float(trajectory.get("dt", 0)),
                "sim_duration": float(trajectory.get("sim_duration", 0)),
                "data_frequency": int(trajectory.get("data_frequency", 0)),
            })
        if self.config.xml_path:
            metadata["xml_path"] = self.config.xml_path
        if self.config.checkpoint_path:
            metadata["checkpoint_path"] = self.config.checkpoint_path
        if self.config.wandb_run_path:
            metadata["policy_wandb_run"] = self.config.wandb_run_path
        if self.config.command_trajectory_path:
            metadata["command_trajectory"] = self.config.command_trajectory_path

        alias = self.config.wandb_artifact_alias
        if not alias:
            raise ValueError("wandb_artifact_alias is required when upload_wandb=True")

        run = wandb.init(
            project="SysID",
            job_type="data-collection",
        )
        artifact = wandb.Artifact(
            name="SysID-Dataset",
            type="dataset",
            metadata=metadata,
        )
        artifact.add_file(str(file_path), name=file_path.name)
        run.log_artifact(artifact, aliases=["latest", alias])
        run.finish()
        print(f"Uploaded to wandb: SysID-Dataset:{alias} ({file_path.name})")
        print(f"  Download with: trajectory='<entity>/SysID/SysID-Dataset:{alias}'")
