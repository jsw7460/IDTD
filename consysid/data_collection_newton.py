"""Newton trajectory collection.

Sister of :mod:`consysid.data_collection` -- the Mujoco collect uses a
different simulator from the one the policy trained in (identified
parameters absorb any cross-sim mismatch). This module instead collects
in **Newton**, the same simulator the policy was trained in and the
same simulator the identification runs its forward sims in, so the
identification target is the pure parameter shift (no hidden simulator
bias).

The class inherits :class:`consysid.data_collection.TrajectoryCollector`
and overrides only what is sim-specific:

* :meth:`__init__` builds the env via the Newton initializer and
  consumes a *pre-built* ``env_cfgs`` (callers apply the GT shift on
  the source ``Go2FlatConfig`` via
  :func:`apply_kp_kdknee_footfric_to_robot_overrides` and pass the
  built ``env_cfgs`` in -- mirroring how the active-exploration stage
  wires Newton).
* :meth:`_read_applied_torques` sums the explicit actuators'
  ``applied_effort`` instead of reading the Mujoco-only
  ``robot_data.applied_torque`` wrapper.
* :meth:`_extract_pd_gains` skips the implicit-actuator MuJoCo
  fallback (Newton always exposes explicit actuators).

Every other method (``_load_command_trajectory``, the rollout loop in
:meth:`collect`, save / resolve / wandb upload paths) is sim-agnostic
and reused verbatim through inheritance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from consysid.data_collection import (
    TrajectoryCollector,
    TrajectoryCollectorConfig,
    _detect_robot_key_local,
)


@dataclass
class NewtonTrajectoryCollectorConfig(TrajectoryCollectorConfig):
    """Newton sim2sim collect config.

    Adds one Newton-specific field on top of the Mujoco config:

    * ``env_cfgs`` -- a pre-built ``ConfigsForRun`` returned by
      ``preset.build()`` where the source ``Go2FlatConfig.robot.*_override``
      slots have already been populated (typically via
      :func:`apply_kp_kdknee_footfric_to_robot_overrides`). The
      collector then skips the Mujoco-side preset-resolve path and
      uses this ``env_cfgs`` directly. Setting ``preset`` or
      ``xml_path`` is invalid here -- they are Mujoco-only.
    """

    env_cfgs: Any = None


class NewtonTrajectoryCollector(TrajectoryCollector):
    """Collect physical state trajectories in Newton.

    Used in the single-simulator experimental setup: policy was trained
    on Newton, the collected trajectories come from Newton (with
    ``CURRENT_GT`` applied via ``*_override`` slots), and identification
    runs in Newton. No cross-sim gap → the identified ``theta_best`` is
    the pure GT shift.
    """

    def __init__(
        self,
        config: NewtonTrajectoryCollectorConfig,
        customize_env: Any | None = None,
    ):
        # Deliberately do NOT call ``super().__init__`` -- the Mujoco
        # init is structured around ``resolve_cross_sim_config`` + a
        # mujoco-only preset's ``runtime_setup`` hook, neither of which
        # applies here. We mirror its post-conditions instead so every
        # inherited method below (``collect`` / ``visualize`` / save /
        # wandb upload) can use the same attribute surface.
        self.config = config

        if config.env_cfgs is None:
            raise ValueError(
                "NewtonTrajectoryCollectorConfig.env_cfgs is required. "
                "Build it from the Newton-side ``Go2FlatConfig`` after "
                "applying ``apply_*_to_robot_overrides`` on its robot "
                "slot and call ``preset.build()`` before constructing "
                "this collector."
            )
        if config.preset is not None or config.xml_path is not None:
            raise ValueError(
                "NewtonTrajectoryCollectorConfig does not accept "
                "``preset`` or ``xml_path`` -- those are Mujoco-only. "
                "Apply your GT shift to ``Go2FlatConfig.robot.*_override`` "
                "slots and pass the resulting built ``env_cfgs`` instead."
            )

        from jaxrlworld.rl.evals.sim_initializers import (
            detect_sim_type,
            get_initializer,
        )
        from jaxrlworld.rl.evals.policy_wrappers import PolicyWrapper
        from jaxrlworld.rl.utils.checkpoint import (
            load_checkpoint_metadata,
            load_runner,
        )

        # Resolve checkpoint_path from wandb if needed -- shared with the
        # Mujoco init, but reproduced here because we are not calling
        # ``super().__init__``.
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
            raise ValueError(
                "Either checkpoint_path or wandb_run_path must be provided."
            )

        # Load checkpoint metadata for downstream consumers (joint perm,
        # robot_key for trajectory metadata).
        self.metadata = load_checkpoint_metadata(config.checkpoint_path)
        self._robot_key = _detect_robot_key_local(self.metadata)
        train_sim_type = detect_sim_type(self.metadata)
        print(
            f"Checkpoint trained on: {train_sim_type} "
            f"(robot: {self._robot_key})"
        )

        eval_cfgs = config.env_cfgs

        # Copy obs/nn/algo from checkpoint so the policy sees the exact
        # input shape it was trained against -- mirrors the Mujoco init's
        # ``_prepare_cross_sim_configs`` step but in-place here.
        train_config = self.metadata.get("config", {})
        train_algo = train_config.get("algorithm", {})
        if train_algo:
            eval_cfgs.algorithm = type(eval_cfgs.algorithm).from_dict(
                train_algo,
            )
        train_nn = train_config.get("nn", {})
        if train_nn:
            eval_cfgs.nn = type(eval_cfgs.nn).from_dict(train_nn)
        train_obs = train_config.get("observation", {})
        if train_obs:
            train_obs_group = train_obs.get("obs_group")
            if train_obs_group is not None:
                eval_cfgs.observation.obs_group = train_obs_group

        # Override num_envs / viewer (same fields the Mujoco init writes).
        eval_cfgs.env.num_envs = config.num_envs
        eval_cfgs.scene.num_envs = config.num_envs
        eval_cfgs.visualization.show_viewer = config.show_viewer
        if config.show_viewer:
            eval_cfgs.visualization.viewer_type = config.viewer_type
            eval_cfgs.visualization.viser_port = config.viser_port
        eval_cfgs.visualization.record_video = False

        # Disable obs corruption -- noise has no role in a sim2sim
        # identification experiment.
        from jaxrlworld.rl.configs.common_config_classes import disable_corruption

        disable_corruption(eval_cfgs.observation)

        # Strip noise events but keep ``reset_*`` so the robot starts
        # standing every episode (same rationale as the Mujoco collect).
        if hasattr(eval_cfgs, "event"):
            from consysid.configs.event_filter import disable_dr_events

            disable_dr_events(eval_cfgs.event, keep_reset=True)

        # Init Newton env -- same path the active-exploration stage and identification
        # use, so the robot config (PD / friction overrides already
        # written into ``eval_cfgs.robot`` by the caller) takes effect
        # exactly the same way.
        self._init = get_initializer("Newton")
        self.device = self._init.init_device()
        self.env = self._init.init_environment(eval_cfgs, seed=config.seed)
        self.env.reset()

        # Optional user runtime hook (e.g. mass-tensor patch) -- same
        # contract as the Mujoco path so existing customize_env helpers
        # can be reused if they touch sim-agnostic state.
        if customize_env is not None:
            customize_env(self.env)

        self.runner = load_runner(
            env=self.env,
            checkpoint_path=config.checkpoint_path,
            cfgs=eval_cfgs,
            use_wandb=False,
        )
        self.runner.set_eval_mode()

        joint_perm = self._build_joint_permutation()
        self.policy = PolicyWrapper.from_runner(
            self.runner, self.device, joint_perm=joint_perm,
        )

        self.eval_cfgs = eval_cfgs

        # Mirror the Mujoco init's command-trajectory state so the
        # inherited ``collect`` loop's ``has_cmd_traj`` branch works as-is.
        self._cmd_velocity = None
        self._cmd_gait = None
        self._cmd_delta = None
        self._cmd_K = None
        self._cmd_K_feature_names = ()
        self._segment_param_indices = None
        self._segment_size_steps = 0
        self._param_names = None
        self._theta_init_path = None
        if config.command_trajectory_path is not None:
            self._load_command_trajectory(config.command_trajectory_path)

    # ── Sim-specific overrides ───────────────────────────────────────────

    def _control_step_dt(self) -> float:
        """Newton stores the control step directly in ``scene.dt``."""
        return float(self.eval_cfgs.scene.dt)

    def _read_applied_torques(self) -> torch.Tensor:
        """Newton: sum explicit actuators' ``applied_effort``.

        Matches the read path
        :meth:`NewtonOpenLoopEvaluator._read_applied_torques` so the
        collect-side torque column lines up byte-for-byte with what
        the identification forward sim records on the same trajectory.
        """
        act_mgr = self.env.act_manager
        actuators = act_mgr._actuators
        nj = act_mgr.total_action_dim
        torques = torch.zeros(self.env.num_envs, nj, device=self.device)
        for act, idx in actuators:
            torques[:, idx] = act.applied_effort
        return torques

    def _read_motor_rpm(self) -> torch.Tensor | None:
        """Pre-step motor RPM of the propeller actuator, or ``None``.

        Keys on the ``propeller_thrust`` action term (the thrust-robot
        marker); joint-actuated robots return ``None`` so the base
        collect loop skips the ``motor_rpm`` channel entirely.
        """
        term = self.env.act_manager._terms.get("propeller_thrust")
        if term is None:
            return None
        return term._actuator.current_rpm

    def _extract_pd_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Newton always uses explicit actuators -- no MuJoCo fallback."""
        act_mgr = self.env.act_manager
        num_joints = act_mgr.total_action_dim

        kp = np.zeros(num_joints, dtype=np.float32)
        kd = np.zeros(num_joints, dtype=np.float32)

        if not act_mgr._has_explicit_actuators:
            if "propeller_thrust" in act_mgr._terms:
                # Thrust-motor robot: there is no PD torque path, so the
                # kp/kd metadata columns are structurally meaningless —
                # store zeros of action width. The motor parameters
                # (kf/km/tau) live on the actuator buffers, not here.
                return kp, kd
            raise RuntimeError(
                "NewtonTrajectoryCollector expects explicit actuators "
                "(``act_manager._has_explicit_actuators == True``); the "
                "MJ-model fallback path is Mujoco-specific and does not "
                "apply here."
            )

        for actuator, joint_idx in act_mgr._actuators:
            kp_vals = actuator.stiffness[0].cpu().numpy()
            kd_vals = actuator.damping[0].cpu().numpy()
            for i, j in enumerate(joint_idx):
                kp[j] = kp_vals[i]
                kd[j] = kd_vals[i]

        return kp, kd
