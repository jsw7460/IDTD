"""Sysid env event-term filter shared by identification / explore / collect.

All three SysID env builders strip the policy-training preset's noise-
inducing event terms (DR, push, random reset) before handing the env
to CMA-ES / open-loop replay / trajectory collection. This module is
the single source of truth for *which* event terms survive that strip,
keyed by a per-caller ``keep_reset`` flag.

Two prefixes always survive:

* ``set_*`` -- the SSOT-installed GT-pin maintainers
  (``set_foot_friction`` / ``set_joint_friction``) that re-write the
  configured ``CURRENT_GT`` value into mjwarp's model arrays at every
  reset. Without them the sim friction drifts away from ``CURRENT_GT``
  and the SysID candidate's view of "the real world" becomes
  inconsistent with the data.

One prefix is path-dependent:

* ``reset_*`` (``reset_root`` / ``reset_dof_pos``) -- standing-pose
  initialisers. *Collect* and *explore* paths need them so the robot
  spawns standing; *identification* path overwrites t0 state per-clip
  from the dataset, so the reset event is redundant *and* its
  residual velocity-field perturbations have been observed to inject
  a ~1.65 % bias into PD identification.
"""

from __future__ import annotations

from jaxrlworld.rl.configs.base_config import iter_terms
from jaxrlworld.rl.configs.events.event_term_config import EventTermConfig


def disable_dr_events(event_cfg, *, keep_reset: bool) -> None:
    """Strip every event term except SSOT GT-pin maintainers (``set_*``),
    optionally keeping reset-time pose initialisers (``reset_*``).

    Args:
        event_cfg: ``EventConfig`` dataclass instance whose event-term
            attributes are mutated in place. Attributes whose name does
            not pass the allow-list become ``None``.
        keep_reset: ``True`` for *collect* / *explore* paths. ``reset_*``
            terms survive so the robot starts standing every episode
            (Newton: ``reset_root_state_uniform`` places the trunk at
            ~0.34 m; without it MuJoCo's mjcf default trunk pos = 1.5 m
            drops the robot from the sky and corrupts every collected
            trajectory). ``False`` for *identification* -- the open-loop
            evaluator overwrites t0 state per-clip from the dataset so
            the reset pose is irrelevant, and small reset-time
            velocity-field residuals were observed to inject a ~1.65 %
            bias into PD identification (see
            ``scripts/sanity/single_term_only/kp_knee.py``).
    """
    for name, term in iter_terms(event_cfg, EventTermConfig).items():
        if name.startswith("set_"):
            continue
        if keep_reset and name.startswith("reset_"):
            continue
        setattr(event_cfg, name, None)
