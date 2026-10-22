"""Open-loop cost evaluator for SysID.

Sets initial state from trajectory clips, replays recorded actions,
and computes weighted prediction error against ground truth.
Genesis and Newton subclasses handle simulator-specific state access.

Cost terms follow SPI-Active (Table 2):
  Position, Velocity, Quaternion, Angular velocity,
  Joint position, Joint velocity, Joint torque.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from consysid.sysid.clip_dataset import Clip


DEFAULT_WEIGHTS = {
    "root_pos": 4.0,
    "root_vel": 2.0,
    "root_quat": 2.0,
    "root_ang_vel": 0.5,
    "joint_pos": 3.0,
    "joint_vel": 0.1,
    "joint_torques": 0.01,
}


@dataclass
class EvaluatorConfig:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


@dataclass
class SimState:
    """Full simulator state returned by _read_state."""
    root_pos: torch.Tensor       # [B, 3]
    root_quat: torch.Tensor      # [B, 4]
    root_lin_vel: torch.Tensor   # [B, 3]
    root_ang_vel: torch.Tensor   # [B, 3]
    joint_pos: torch.Tensor      # [B, nj]
    joint_vel: torch.Tensor      # [B, nj]
    joint_torques: torch.Tensor  # [B, nj]


class OpenLoopEvaluator(ABC):
    """Base evaluator: clips distributed across parallel envs."""

    def __init__(self, env, config: EvaluatorConfig | None = None):
        self.env = env
        self.config = config or EvaluatorConfig()
        self.device = env.device
        self.num_envs = env.num_envs

    @torch.no_grad()
    def evaluate(self, clips: list[Clip]) -> float:
        """Evaluate current env parameters over all clips. Returns scalar cost."""
        horizon_groups: dict[int, list[Clip]] = defaultdict(list)
        for clip in clips:
            horizon_groups[clip.horizon].append(clip)

        total_cost = 0.0
        num_evaluated = 0

        for h, group_clips in horizon_groups.items():
            for batch_start in range(0, len(group_clips), self.num_envs):
                batch_clips = group_clips[batch_start:batch_start + self.num_envs]
                batch_cost = self._evaluate_batch(batch_clips)
                total_cost += batch_cost.sum()
                num_evaluated += len(batch_clips)

        return total_cost / num_evaluated

    @torch.no_grad()
    def evaluate_parallel(
        self,
        clips: list[Clip],
        params: "SysIDParameters",
        candidate_thetas: np.ndarray,
    ) -> np.ndarray:
        """Evaluate all candidates in parallel using env partitioning.

        Each candidate gets ``clips_per_candidate`` envs. The same clips
        are replicated across all candidate groups, but each group has
        different physical parameters (θ).

        Args:
            clips: Sampled clips (must all have the same horizon).
            params: SysIDParameters for applying θ to env.
            candidate_thetas: (n_candidates, dim) array.

        Returns:
            (n_candidates,) array of mean prediction costs.
        """
        n_cand = candidate_thetas.shape[0]
        clips_per_cand = self.num_envs // n_cand
        if clips_per_cand == 0:
            raise ValueError(
                f"Not enough envs ({self.num_envs}) for {n_cand} candidates. "
                f"Need at least {n_cand} envs."
            )

        # Use only as many clips as fit
        clips = clips[:clips_per_cand]
        B_per = len(clips)
        B_total = n_cand * B_per  # total envs used

        # Max horizon across all clips; shorter clips will be masked
        H_max = max(c.horizon for c in clips)

        # 1. Apply different θ per candidate group (same clips, different
        # physical parameters across groups).
        theta_batch = np.zeros((self.num_envs, params.dim), dtype=np.float32)
        for ci in range(n_cand):
            start_env = ci * B_per
            end_env = start_env + B_per
            theta_batch[start_env:end_env] = candidate_thetas[ci]
        params.apply_to_env(theta_batch)

        # 2. Set initial state: replicate clips for each candidate group
        expanded_clips = clips * n_cand  # [clip0..clipN, clip0..clipN, ...]
        self._set_batch_initial_state(expanded_clips)
        self._reset_actuators()

        # 3. Pre-build GT tensors, padded to H_max
        na = clips[0].actions.shape[1]
        nj = clips[0].joint_pos.shape[1]

        # Pad clips to H_max+1 states and H_max actions
        def _pad_state(arr, target_len):
            """Pad [H+1, D] to [target_len, D] by repeating last row."""
            if arr.shape[0] >= target_len:
                return arr[:target_len]
            pad_rows = target_len - arr.shape[0]
            return np.concatenate([arr, np.tile(arr[-1:], (pad_rows, 1))], axis=0)

        def _pad_action(arr, target_len):
            """Pad [H, D] to [target_len, D] with zeros."""
            if arr.shape[0] >= target_len:
                return arr[:target_len]
            pad_rows = target_len - arr.shape[0]
            return np.concatenate([arr, np.zeros((pad_rows, arr.shape[1]), dtype=arr.dtype)], axis=0)

        all_root_pos = np.stack([_pad_state(c.root_pos, H_max + 1) for c in clips])
        all_root_quat = np.stack([_pad_state(c.root_quat, H_max + 1) for c in clips])
        all_root_lin_vel = np.stack([_pad_state(c.root_lin_vel, H_max + 1) for c in clips])
        all_root_ang_vel = np.stack([_pad_state(c.root_ang_vel, H_max + 1) for c in clips])
        all_joint_pos = np.stack([_pad_state(c.joint_pos, H_max + 1) for c in clips])
        all_joint_vel = np.stack([_pad_state(c.joint_vel, H_max + 1) for c in clips])
        all_joint_torques = np.stack([_pad_state(c.joint_torques, H_max + 1) for c in clips])
        all_actions = np.stack([_pad_action(c.actions, H_max) for c in clips])

        # Per-clip valid mask: [B_per, H_max] — True if step t is valid for this clip
        valid_mask_np = np.zeros((B_per, H_max), dtype=np.float32)
        for i, clip in enumerate(clips):
            valid_mask_np[i, :clip.horizon] = 1.0

        # Tile for all candidates
        gt_root_pos = torch.tensor(np.tile(all_root_pos, (n_cand, 1, 1)), device=self.device, dtype=torch.float32)
        gt_root_quat = torch.tensor(np.tile(all_root_quat, (n_cand, 1, 1)), device=self.device, dtype=torch.float32)
        gt_root_lin_vel = torch.tensor(np.tile(all_root_lin_vel, (n_cand, 1, 1)), device=self.device, dtype=torch.float32)
        gt_root_ang_vel = torch.tensor(np.tile(all_root_ang_vel, (n_cand, 1, 1)), device=self.device, dtype=torch.float32)
        gt_joint_pos = torch.tensor(np.tile(all_joint_pos, (n_cand, 1, 1)), device=self.device, dtype=torch.float32)
        gt_joint_vel = torch.tensor(np.tile(all_joint_vel, (n_cand, 1, 1)), device=self.device, dtype=torch.float32)
        gt_joint_torques = torch.tensor(np.tile(all_joint_torques, (n_cand, 1, 1)), device=self.device, dtype=torch.float32)
        gt_actions = torch.tensor(np.tile(all_actions, (n_cand, 1, 1)), device=self.device, dtype=torch.float32)
        valid_mask = torch.tensor(np.tile(valid_mask_np, (n_cand, 1)), device=self.device, dtype=torch.float32)

        # Initial sim position
        init_state = self._read_state(B_total)
        sim_pos_init = init_state.root_pos.clone()
        gt_pos_init = gt_root_pos[:, 0, :]

        w = self.config.weights
        term_names = list(w.keys())
        clip_costs = torch.zeros(B_total, device=self.device)
        # Per-term cost accumulators
        term_costs = {name: torch.zeros(B_total, device=self.device) for name in term_names}
        valid_steps = torch.zeros(B_total, device=self.device)

        for t in range(H_max):
            # Actions: fill used envs from pre-built tensor
            actions_t = torch.zeros(self.num_envs, na, device=self.device)
            actions_t[:B_total] = gt_actions[:, t, :]

            self._apply_actions(actions_t)
            self._step_physics()
            # Read applied_effort AFTER the decimation loop so the
            # measurement reflects the last substep's compute — matching
            # data_collection, which records ``robot_data.applied_torque``
            # (= MuJoCo's ``qfrc_actuator``) after ``env.step`` returns.
            # Reading before _step_physics would give the substep-0
            # pre-compute, when joint_pos is still farthest from target.
            applied_torques = self._read_applied_torques(B_total)

            sim = self._read_state(B_total)
            sim.joint_torques = applied_torques

            gt_idx = t + 1
            gt_pos = gt_root_pos[:, gt_idx, :]
            gt_quat = gt_root_quat[:, gt_idx, :]
            gt_lin_vel = gt_root_lin_vel[:, gt_idx, :]
            gt_ang_vel = gt_root_ang_vel[:, gt_idx, :]
            gt_jpos = gt_joint_pos[:, gt_idx, :]
            gt_jvel = gt_joint_vel[:, gt_idx, :]
            gt_jtorques = gt_joint_torques[:, t, :]

            sim_pos_delta = sim.root_pos - sim_pos_init
            gt_pos_delta = gt_pos - gt_pos_init

            dot = (sim.root_quat * gt_quat).sum(dim=1)
            quat_err = 1.0 - dot ** 2

            # Compute per-term errors
            errs = {
                "root_pos": ((sim_pos_delta - gt_pos_delta) ** 2).sum(dim=1),
                "root_vel": ((sim.root_lin_vel - gt_lin_vel) ** 2).sum(dim=1),
                "root_quat": quat_err,
                "root_ang_vel": ((sim.root_ang_vel - gt_ang_vel) ** 2).sum(dim=1),
                "joint_pos": ((sim.joint_pos - gt_jpos) ** 2).sum(dim=1),
                "joint_vel": ((sim.joint_vel - gt_jvel) ** 2).sum(dim=1),
                "joint_torques": ((sim.joint_torques - gt_jtorques) ** 2).sum(dim=1),
            }

            err = torch.zeros(B_total, device=self.device)
            mask_t = valid_mask[:, t]
            for name in term_names:
                weighted = w[name] * errs[name]
                err += weighted
                term_costs[name] += weighted * mask_t

            clip_costs += err * mask_t
            valid_steps += mask_t

        # Aggregate per candidate: sum over steps already accumulated in clip_costs,
        # just average over clips (matching sequential: sum-over-steps / num_clips).
        per_env_costs = clip_costs[:B_total].cpu().numpy()
        per_cand_costs = per_env_costs.reshape(n_cand, B_per).mean(axis=1)

        # Per-term breakdown
        self._last_term_costs = {}
        for name in term_names:
            per_env = term_costs[name][:B_total].cpu().numpy()
            per_cand = per_env.reshape(n_cand, B_per).mean(axis=1)
            self._last_term_costs[name] = per_cand

        return per_cand_costs

    def _evaluate_batch(self, clips: list[Clip]) -> np.ndarray:
        """Evaluate a batch of clips (same horizon, one clip per env)."""
        B = len(clips)
        H = clips[0].horizon
        w = self.config.weights
        nj = clips[0].joint_pos.shape[1]

        self._set_batch_initial_state(clips)
        self._reset_actuators()

        # Pre-build all GT tensors at once (numpy stack -> single GPU transfer)
        na = clips[0].actions.shape[1]
        all_root_pos = np.stack([c.root_pos for c in clips], axis=0)
        all_root_quat = np.stack([c.root_quat for c in clips], axis=0)
        all_root_lin_vel = np.stack([c.root_lin_vel for c in clips], axis=0)
        all_root_ang_vel = np.stack([c.root_ang_vel for c in clips], axis=0)
        all_joint_pos = np.stack([c.joint_pos for c in clips], axis=0)
        all_joint_vel = np.stack([c.joint_vel for c in clips], axis=0)
        all_joint_torques = np.stack([c.joint_torques for c in clips], axis=0)
        all_actions = np.stack([c.actions for c in clips], axis=0)

        gt_root_pos = torch.tensor(all_root_pos, device=self.device, dtype=torch.float32)
        gt_root_quat = torch.tensor(all_root_quat, device=self.device, dtype=torch.float32)
        gt_root_lin_vel = torch.tensor(all_root_lin_vel, device=self.device, dtype=torch.float32)
        gt_root_ang_vel = torch.tensor(all_root_ang_vel, device=self.device, dtype=torch.float32)
        gt_joint_pos = torch.tensor(all_joint_pos, device=self.device, dtype=torch.float32)
        gt_joint_vel = torch.tensor(all_joint_vel, device=self.device, dtype=torch.float32)
        gt_joint_torques = torch.tensor(all_joint_torques, device=self.device, dtype=torch.float32)
        gt_actions = torch.tensor(all_actions, device=self.device, dtype=torch.float32)

        init_state = self._read_state(B)
        sim_pos_init = init_state.root_pos.clone()
        gt_pos_init = gt_root_pos[:, 0, :]

        clip_costs = torch.zeros(B, device=self.device)

        for t in range(H):
            actions_t = torch.zeros(self.num_envs, na, device=self.device)
            actions_t[:B] = gt_actions[:, t, :]

            self._apply_actions(actions_t)
            self._step_physics()
            applied_torques = self._read_applied_torques(B)

            sim = self._read_state(B)
            sim.joint_torques = applied_torques

            gt_idx = t + 1
            sim_pos_delta = sim.root_pos - sim_pos_init
            gt_pos_delta = gt_root_pos[:, gt_idx, :] - gt_pos_init

            dot = (sim.root_quat * gt_root_quat[:, gt_idx, :]).sum(dim=1)
            quat_err = 1.0 - dot ** 2

            err = torch.zeros(B, device=self.device)
            err += w["root_pos"] * ((sim_pos_delta - gt_pos_delta) ** 2).sum(dim=1)
            err += w["root_vel"] * ((sim.root_lin_vel - gt_root_lin_vel[:, gt_idx, :]) ** 2).sum(dim=1)
            err += w["root_quat"] * quat_err
            err += w["root_ang_vel"] * ((sim.root_ang_vel - gt_root_ang_vel[:, gt_idx, :]) ** 2).sum(dim=1)
            err += w["joint_pos"] * ((sim.joint_pos - gt_joint_pos[:, gt_idx, :]) ** 2).sum(dim=1)
            err += w["joint_vel"] * ((sim.joint_vel - gt_joint_vel[:, gt_idx, :]) ** 2).sum(dim=1)
            err += w["joint_torques"] * ((sim.joint_torques - gt_joint_torques[:, t, :]) ** 2).sum(dim=1)
            clip_costs += err

        return clip_costs.cpu().numpy()

    @abstractmethod
    def _set_batch_initial_state(self, clips: list[Clip]) -> None: ...

    @abstractmethod
    def _apply_actions(self, actions: torch.Tensor) -> None: ...

    @abstractmethod
    def _step_physics(self) -> None: ...

    @abstractmethod
    def _read_state(self, B: int) -> SimState:
        """Returns full simulator state sliced to [:B]."""
        ...

    @abstractmethod
    def _read_applied_torques(self, B: int) -> torch.Tensor:
        """Read applied torques AFTER _step_physics' decimation loop.

        Newton/Genesis ``_step_physics`` re-invokes
        ``act_manager.apply_actions`` every substep, so ``act.applied_effort``
        is overwritten each iteration. Reading after the decimation loop
        captures the last substep's compute — the same time point at which
        ``data_collection`` records ``robot_data.applied_torque`` (post
        ``env.step``). Reading before the loop would yield the substep-0
        pre-compute (joint_pos farthest from target, torque inflated).
        """
        ...

    def _reset_actuators(self) -> None:
        """Reset actuator buffers and disable delay for open-loop replay.

        Delayed PD actuators have a ring buffer of past targets. Without
        resetting, stale targets from previous clips leak into the current
        evaluation. Delay is set to 0 because clip actions are already
        the processed (post-delay) targets from the original collection.
        """
        act_mgr = self.env.act_manager
        if not act_mgr._has_explicit_actuators:
            return
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        for actuator, _ in act_mgr._actuators:
            actuator.reset(all_env_ids)
            # Disable delay: clip actions are already post-delay targets
            if hasattr(actuator, '_delay'):
                actuator._delay[:] = 0


class GenesisOpenLoopEvaluator(OpenLoopEvaluator):

    def __init__(self, env, config: EvaluatorConfig | None = None):
        super().__init__(env, config)
        self.robot = env.scene_manager.robot
        self.dof_ids = env.act_manager.actuated_dof_ids

    def _set_batch_initial_state(self, clips: list[Clip]) -> None:
        B = len(clips)
        N = self.num_envs
        nj = clips[0].joint_pos.shape[1]

        root_pos = torch.zeros(N, 3, device=self.device)
        root_quat = torch.zeros(N, 4, device=self.device)
        jpos = torch.zeros(N, nj, device=self.device)
        all_vel = torch.zeros(N, 6 + nj, device=self.device)

        for i, clip in enumerate(clips):
            root_pos[i] = torch.tensor(clip.root_pos[0], device=self.device)
            root_quat[i] = torch.tensor(clip.root_quat[0], device=self.device)
            jpos[i] = torch.tensor(clip.joint_pos[0], device=self.device)
            all_vel[i, 0:3] = torch.tensor(clip.root_lin_vel[0], device=self.device)
            all_vel[i, 3:6] = torch.tensor(clip.root_ang_vel[0], device=self.device)
            all_vel[i, 6:] = torch.tensor(clip.joint_vel[0], device=self.device)

        self.robot.set_pos(root_pos)
        self.robot.set_quat(root_quat)
        self.robot.set_dofs_position(jpos, dofs_idx_local=self.dof_ids, zero_velocity=False)
        self.robot.set_dofs_velocity(all_vel)

    def _apply_actions(self, actions: torch.Tensor) -> None:
        # ``clip.actions`` is the raw policy output. Run
        # ``process_actions`` to get ``scale·raw + offset`` into
        # ``_processed_action_history[0]`` (matching the data-side
        # ``World.step`` pipeline) and push that processed target to
        # Genesis' built-in PD.
        self.env.act_manager.process_actions(actions)
        target = self.env.act_manager.processed_actions
        self.robot.control_dofs_position(target, self.dof_ids)

    def _step_physics(self) -> None:
        self.env._step_physics()

    def _read_applied_torques(self, B: int) -> torch.Tensor:
        # Genesis: explicit PD computes torques and stores via apply_actions.
        # Read the actuator's computed effort directly.
        actuators = self.env.act_manager._actuators
        if actuators:
            # Concatenate all actuator efforts in canonical joint order
            nj = self.env.act_manager.total_action_dim
            torques = torch.zeros(self.num_envs, nj, device=self.device)
            for act, idx in actuators:
                torques[:, idx] = act.applied_effort
            return torques[:B]
        return torch.zeros(B, self.env.act_manager.total_action_dim, device=self.device)

    def _read_state(self, B: int) -> SimState:
        return SimState(
            root_pos=self.robot.get_pos()[:B],
            root_quat=self.robot.get_quat()[:B],
            root_lin_vel=self.robot.get_vel()[:B],
            root_ang_vel=self.robot.get_ang()[:B],
            joint_pos=self.robot.get_dofs_position(dofs_idx_local=self.dof_ids)[:B],
            joint_vel=self.robot.get_dofs_velocity(dofs_idx_local=self.dof_ids)[:B],
            joint_torques=torch.zeros(B, len(self.dof_ids), device=self.device),  # filled by _read_applied_torques
        )


class NewtonOpenLoopEvaluator(OpenLoopEvaluator):

    def __init__(self, env, config: EvaluatorConfig | None = None):
        super().__init__(env, config)
        import warp as wp

        self._scene_mgr = env.scene_manager
        self._act_mgr = env.act_manager

        model = self._scene_mgr.model
        self._num_worlds = model.world_count
        self._q_per_world = model.joint_coord_count // self._num_worlds
        self._qd_per_world = model.joint_dof_count // self._num_worlds
        self._bodies_per_world = model.body_count // self._num_worlds

        # Canonical (act-manager) → native joint_q / joint_qd slot indices.
        # Clips store joint_pos / joint_vel / actions in canonical order
        # (data_collection writes via MujocoRobotData reindex), so any
        # write into / read from native arrays must go through these maps.
        # Hard-coded "free joint = first 7/6 slots, actuated = next nj
        # contiguous slots" is wrong: URDF parse order is arbitrary and
        # may interleave passive joints.
        indexing = self._act_mgr._indexing
        self._q_idx = indexing.newton_q_indices
        self._qd_idx = indexing.newton_qd_indices
        nj = self._act_mgr.total_action_dim
        self._nj = nj

        # Thrust-motor robots (Crazyflie): the propeller actuator's RPM is
        # hidden dynamic state that must be restored per clip — see
        # ``Clip.motor_rpm``. ``None`` for joint-actuated robots.
        self._propeller_term = self._act_mgr._terms.get("propeller_thrust")

    def _set_batch_initial_state(self, clips: list[Clip]) -> None:
        import warp as wp
        import newton

        B = len(clips)
        N = self._num_worlds
        qpw = self._q_per_world
        qdpw = self._qd_per_world

        # Start from current state (preserves default values for non-actuated joints)
        joint_q_data = wp.to_torch(self._scene_mgr.state_0.joint_q).reshape(N, qpw).clone()
        joint_qd_data = wp.to_torch(self._scene_mgr.state_0.joint_qd).reshape(N, qdpw).clone()

        for i, clip in enumerate(clips):
            # Root position: use clip position directly (no env spacing)
            joint_q_data[i, 0:3] = torch.tensor(clip.root_pos[0], device=self.device)

            # Root pose: MuJoCo wxyz → Newton xyzw
            root_quat_wxyz = clip.root_quat[0]
            joint_q_data[i, 3:7] = torch.tensor(
                [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]],
                device=self.device,
            )
            joint_q_data[i, self._q_idx] = torch.tensor(
                clip.joint_pos[0], device=self.device
            )

            joint_qd_data[i, 0:3] = torch.tensor(clip.root_lin_vel[0], device=self.device)
            joint_qd_data[i, 3:6] = torch.tensor(clip.root_ang_vel[0], device=self.device)
            joint_qd_data[i, self._qd_idx] = torch.tensor(
                clip.joint_vel[0], device=self.device
            )

        if self._propeller_term is not None:
            # Restore the hidden motor-RPM state each clip's first action
            # was computed against. Datasets collected before the
            # ``motor_rpm`` channel existed cannot be evaluated on a
            # thrust robot — the 1-5 step horizons sit entirely inside
            # the RPM transient, so a silent zero-RPM start would bias
            # every motor-parameter estimate. Crash instead.
            actuator = self._propeller_term._actuator
            rpm0 = torch.zeros(
                B, actuator.current_rpm.shape[1], device=self.device,
            )
            for i, clip in enumerate(clips):
                if clip.motor_rpm is None:
                    raise ValueError(
                        "Thrust-motor robot but the clip carries no "
                        "motor_rpm channel — re-collect the dataset with "
                        "the RPM-recording collector."
                    )
                rpm0[i] = torch.tensor(clip.motor_rpm[0], device=self.device)
            actuator.current_rpm[:B] = rpm0

        flat_q = wp.from_torch(joint_q_data.flatten(), dtype=wp.float32)
        flat_qd = wp.from_torch(joint_qd_data.flatten(), dtype=wp.float32)

        # Write to BOTH states to handle double-buffering swap
        for state in (self._scene_mgr.state_0, self._scene_mgr.state_1):
            wp.copy(state.joint_q, flat_q)
            wp.copy(state.joint_qd, flat_qd)
            newton.eval_fk(
                self._scene_mgr.model, state.joint_q, state.joint_qd, state
            )

    def _apply_actions(self, actions: torch.Tensor) -> None:
        # ``clip.actions`` is the raw policy output (ClipDataset reads
        # ``npz["actions"]`` into the field). Run ``process_actions``
        # so the same ``raw → clip → scale·raw + offset`` transform
        # (and any settle-step / per-term logic) that ``World.step``
        # applied during data collection runs again here. The
        # decimation loop in ``_step_physics`` reads
        # ``_processed_action_history[0]`` each substep, which
        # ``process_actions`` has just populated.
        #
        # This mirrors deployment: on hardware, firmware turns the
        # policy's raw output into a PD target via the same scale/offset.
        # Identifying physics from raw also exposes any cross-env
        # ``_scale``/``_offset`` divergence as a non-zero cost rather
        # than a silent bias.
        self._act_mgr.process_actions(actions)

    def _step_physics(self) -> None:
        self.env._step_physics()

    def _read_applied_torques(self, B: int) -> torch.Tensor:
        # Newton: explicit PD writes ``applied_effort`` per substep via
        # ``act_manager.apply_actions`` inside ``_step_physics``'s
        # decimation loop. Reading after the loop yields the last
        # substep's compute, matching data-collection's post-step
        # ``qfrc_actuator`` snapshot.
        actuators = self._act_mgr._actuators
        if actuators:
            nj = self._act_mgr.total_action_dim
            torques = torch.zeros(self.env.num_envs, nj, device=self.device)
            for act, idx in actuators:
                torques[:, idx] = act.applied_effort
            return torques[:B]
        return torch.zeros(B, self._nj, device=self.device)

    def _read_state(self, B: int) -> SimState:
        import warp as wp

        state = self._scene_mgr.state_0
        N = self._num_worlds
        qpw = self._q_per_world
        qdpw = self._qd_per_world

        joint_q = wp.to_torch(state.joint_q).reshape(N, qpw)
        joint_qd = wp.to_torch(state.joint_qd).reshape(N, qdpw)

        root_pos = joint_q[:B, 0:3]

        # Newton xyzw → MuJoCo wxyz for comparison
        root_quat_xyzw = joint_q[:B, 3:7]
        root_quat = torch.stack([
            root_quat_xyzw[:, 3],  # w
            root_quat_xyzw[:, 0],  # x
            root_quat_xyzw[:, 1],  # y
            root_quat_xyzw[:, 2],  # z
        ], dim=1)

        root_lin_vel = joint_qd[:B, 0:3]
        root_ang_vel = joint_qd[:B, 3:6]
        joint_pos = joint_q[:B][:, self._q_idx]
        joint_vel = joint_qd[:B][:, self._qd_idx]

        return SimState(
            root_pos=root_pos,
            root_quat=root_quat,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_torques=torch.zeros(B, self._nj, device=self.device),  # filled by _read_applied_torques
        )
