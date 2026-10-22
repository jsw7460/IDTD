"""Genesis simulator apply functions for SysID parameters.

All functions follow the signature:
    func(env, values: np.ndarray, env_ids: torch.Tensor, **params) -> None

where ``values`` has shape ``(B, dim)`` and ``env_ids`` shape ``(B,)``.
"""
from __future__ import annotations

import numpy as np
import torch


def apply_mass(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    link_name: str = "base",
) -> None:
    """Set absolute mass on a link. values: (B, 1) in kg."""
    robot = env.scene_manager.robot
    link = robot.get_link(link_name)
    default_mass = float(link.get_mass())

    mass_shifts = values[:, 0] - default_mass
    mass_shifts_t = torch.tensor(
        mass_shifts, dtype=torch.float32, device=env.device,
    ).unsqueeze(1)  # (B, 1)

    robot.set_mass_shift(
        mass_shift=mass_shifts_t,
        links_idx_local=[link.idx_local],
        envs_idx=env_ids,
    )


def apply_com(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    link_name: str = "base",
) -> None:
    """Set CoM offset on a link. values: (B, 3) in metres."""
    robot = env.scene_manager.robot
    link = robot.get_link(link_name)

    com_shifts_t = torch.tensor(
        values, dtype=torch.float32, device=env.device,
    ).unsqueeze(1)  # (B, 1, 3)

    robot.set_COM_shift(
        com_shift=com_shifts_t,
        links_idx_local=[link.idx_local],
        envs_idx=env_ids,
    )


def apply_kp(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    dof_pattern: str | list[str] = ".*",
) -> None:
    """Set PD proportional gains. values: (B, N_matched_dofs)."""
    from jaxrlworld.rl.utils.entity_utils import find_dofs

    robot = env.scene_manager.robot
    if isinstance(dof_pattern, str):
        dof_pattern = [dof_pattern]
    dof_ids, _ = find_dofs(robot, dof_pattern)

    kp_t = torch.tensor(values, dtype=torch.float32, device=env.device)
    robot.set_dofs_kp(kp=kp_t, dofs_idx_local=dof_ids, envs_idx=env_ids)


def apply_kd(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    dof_pattern: str | list[str] = ".*",
) -> None:
    """Set PD derivative gains. values: (B, N_matched_dofs)."""
    from jaxrlworld.rl.utils.entity_utils import find_dofs

    robot = env.scene_manager.robot
    if isinstance(dof_pattern, str):
        dof_pattern = [dof_pattern]
    dof_ids, _ = find_dofs(robot, dof_pattern)

    kd_t = torch.tensor(values, dtype=torch.float32, device=env.device)
    robot.set_dofs_kv(kv=kd_t, dofs_idx_local=dof_ids, envs_idx=env_ids)


def apply_contact_friction(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    link_names: list[str] | None = None,
) -> None:
    """Set contact friction ratio on links. values: (B, 1)."""
    robot = env.scene_manager.robot

    if link_names is None:
        links_idx = list(range(robot.n_links))
    else:
        links_idx = [robot.get_link(n).idx_local for n in link_names]

    n_links = len(links_idx)
    # Broadcast scalar friction to all selected links
    friction_t = torch.tensor(
        values[:, 0:1], dtype=torch.float32, device=env.device,
    ).expand(-1, n_links)  # (B, n_links)

    robot.set_friction_ratio(
        friction_ratio=friction_t,
        links_idx_local=links_idx,
        envs_idx=env_ids,
    )


def apply_armature(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    dof_pattern: str | list[str] = ".*",
) -> None:
    """Set joint armature. values: (B, N_matched_dofs)."""
    from jaxrlworld.rl.utils.entity_utils import find_dofs

    robot = env.scene_manager.robot
    if isinstance(dof_pattern, str):
        dof_pattern = [dof_pattern]
    dof_ids, _ = find_dofs(robot, dof_pattern)

    arm_t = torch.tensor(values, dtype=torch.float32, device=env.device)
    robot.set_dofs_armature(armature=arm_t, dofs_idx_local=dof_ids, envs_idx=env_ids)
