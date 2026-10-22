"""Robot-state summary feature extractor for the Bezier command's
state-feedback term.

The Bezier command generator produces a time-indexed open-loop command
trajectory; the feedback term turns it into a state-aware closed-loop
controller by adding ``K @ features_t`` to the Bezier baseline. ``K``
is jointly optimised with the Bezier control points by CMA-ES. This
module is the single source of truth for which ``features_t`` the
caller can request via :attr:`BezierCommandConfig.state_feedback_features`.
"""

from __future__ import annotations

from typing import Any

import torch


# Canonical names accepted by :func:`extract_features`. Add a new entry
# here and a matching ``elif`` branch in ``extract_features`` to expose
# a new summary feature to the Bezier feedback path.
FEATURE_NAMES: tuple[str, ...] = (
    "base_height",
    "base_lin_vel_x",
    "base_lin_vel_y",
    "base_lin_vel_z",
    "base_ang_vel_x",
    "base_ang_vel_y",
    "base_ang_vel_z",
    "joint_vel_abs_mean",
    "joint_pos_diff_mean",
)


def extract_features(env: Any, names: tuple[str, ...]) -> torch.Tensor:
    """Extract the named summary features from ``env.get_robot_data()``.

    Args:
        env: Any env exposing ``get_robot_data()`` with the standard
            ``root_link_pos_w`` / ``root_link_lin_vel_w`` /
            ``root_link_ang_vel_w`` / ``joint_pos`` / ``joint_vel`` /
            ``default_joint_pos`` attributes.
        names: Tuple of feature names to extract, drawn from
            :data:`FEATURE_NAMES`.

    Returns:
        ``(num_envs, len(names))`` tensor on the env's device.
        Returns ``(num_envs, 0)`` when ``names`` is empty so a caller
        can blindly ``features @ K.T`` even on the no-feedback path
        (matrix multiply against an ``(N, 0)`` matrix gives ``(N, 0)``
        which the broadcast layer ignores).
    """
    robot_data = env.get_robot_data()
    if not names:
        # Empty placeholder with the right leading dim so callers do
        # not need a separate ``is None`` branch.
        n_envs = int(robot_data.root_link_pos_w.shape[0])
        return torch.zeros(
            (n_envs, 0),
            dtype=torch.float32,
            device=robot_data.root_link_pos_w.device,
        )

    # Per-feature scale divisor so all features land roughly in the
    # ``[-1, 1]`` range. Without this CMA-ES has to learn a per-axis
    # ``K`` magnitude that varies by orders of magnitude (joint_vel
    # spans ~10 rad/s while base_height stays near 0.34 m); the
    # learned correction would then be wild on the high-magnitude
    # axes during the first iterations and blow the policy out of its
    # distribution.
    cols: list[torch.Tensor] = []
    for name in names:
        if name == "base_height":
            cols.append(robot_data.root_link_pos_w[:, 2])
        elif name == "base_lin_vel_x":
            cols.append(robot_data.root_link_lin_vel_w[:, 0] / 2.0)
        elif name == "base_lin_vel_y":
            cols.append(robot_data.root_link_lin_vel_w[:, 1] / 2.0)
        elif name == "base_lin_vel_z":
            cols.append(robot_data.root_link_lin_vel_w[:, 2] / 2.0)
        elif name == "base_ang_vel_x":
            cols.append(robot_data.root_link_ang_vel_w[:, 0] / 3.0)
        elif name == "base_ang_vel_y":
            cols.append(robot_data.root_link_ang_vel_w[:, 1] / 3.0)
        elif name == "base_ang_vel_z":
            cols.append(robot_data.root_link_ang_vel_w[:, 2] / 3.0)
        elif name == "joint_vel_abs_mean":
            cols.append(robot_data.joint_vel.abs().mean(dim=-1) / 10.0)
        elif name == "joint_pos_diff_mean":
            default = robot_data.default_joint_pos
            cols.append((robot_data.joint_pos - default).abs().mean(dim=-1))
        else:
            raise ValueError(
                f"Unknown feedback feature {name!r}. Available: "
                f"{FEATURE_NAMES}"
            )

    return torch.stack(cols, dim=-1).to(dtype=torch.float32)
