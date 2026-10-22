"""G1 (29-DOF) Newton SysID preset configurations.

Sibling of ``go2_newton.py`` for the G1 humanoid. Identifies the base
link's log-Cholesky pseudo-inertia (10D) plus per-joint PD gains (kp 29 +
kd 29) — friction is intentionally NOT identified for G1. The apply
functions (``apply_log_cholesky_inertial``, ``apply_kp``, ``apply_kd``)
are robot-agnostic; only the body/joint patterns and per-joint defaults
differ from Go2.

The kp/kd defaults are the nominal motor-class gains from
``rl/configs/robots/g1_29dof.py``, laid out in the g1.xml actuated-joint
order (matches ``act_manager.actuated_joint_names``; the collect script
prints that order so the layout can be verified on the first run).
"""
from dataclasses import dataclass

import numpy as np

from consysid.configs.sysid_term_config import SysIDConfig, SysIDTermConfig
from consysid.sysid.log_cholesky import (
    decode_phi_to_physical_dict,
    encode_physical_to_phi,
    mujoco_inertial_to_body_inertia,
)
from consysid.sysid.param_terms import newton as nf
from jaxrlworld.rl.configs.robots.g1_29dof import (
    DAMPING_4010,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    DAMPING_ANKLE,
    DAMPING_WAIST,
    STIFFNESS_4010,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
    STIFFNESS_ANKLE,
    STIFFNESS_WAIST,
)


# G1 base link ("pelvis") inertial from g1.xml (<body name="pelvis">'s
# <inertial>). MuJoCo stores inertia in principal-axis frame; rebuild the
# body-frame 3x3 tensor so it matches Newton's body_inertia convention.
_G1_BASE_MASS = 3.813
_G1_BASE_COM = np.array([0.0, 0.0, -0.07605])
_G1_BASE_DIAGINERTIA = np.array([0.010549, 0.0093089, 0.0079184])
_G1_BASE_IQUAT_WXYZ = np.array([1.0, 0.0, -0.000399148, 0.0])
_G1_BASE_INERTIA = mujoco_inertial_to_body_inertia(
    _G1_BASE_DIAGINERTIA, _G1_BASE_IQUAT_WXYZ,
)
_G1_BASE_PHI = encode_physical_to_phi(
    _G1_BASE_MASS, _G1_BASE_COM, _G1_BASE_INERTIA,
).tolist()


def _g1_inertial_gt_encoder(gt: dict) -> np.ndarray:
    """Convert user-supplied physical GT dict to log-Cholesky phi."""
    return encode_physical_to_phi(
        mass=gt["mass"],
        com=np.asarray(gt["com"]),
        inertia=np.asarray(gt["inertia"]),
    )


# Per-dim bounds around the nominal phi. A scalar (-5, 5) lets CMA wander
# into non-physical bodies (m ≈ 0.25 kg, CoM ≈ 0.9 m) that blow up the
# Newton sim and produce nan losses. Tight per-dim margins keep mass / CoM
# physical while still covering the mass-shifted GT (pelvis 3.813 → 5.813,
# CoM / inertia unchanged). The mass channel (alpha = phi[0]) and the CoM
# offsets (t1,t2,t3 = phi[7:10]) get the tightest margins since those are
# what diverge; the shape channels (d, s) stay looser.
# GT is a mass-only shift (pelvis 3.813→5.813; CoM / inertia stay nominal),
# so the GT phi differs from the nominal phi essentially only in the mass
# channel (alpha) and slightly in the CoM-coupled terms — the shape
# channels (d diagonal, s off-diagonal) stay ~nominal (s≈0). The earlier
# wide margins (d/s = ±1.5) let CMA push s to ±1.4, which blows the inertia
# tensor up to ~6 (600× nominal) → non-physical body → Newton nan. Tighten
# the shape channels hard (they shouldn't move) and the mass/CoM channels
# enough to cover the mass-shifted GT but not a 2× mass.
#                alpha  d1    d2    d3    s12   s13   s23   t1    t2    t3
_G1_BI_MARGIN = [0.25, 0.50, 0.50, 0.50, 0.30, 0.30, 0.30, 0.35, 0.35, 1.0]
_G1_BI_LO = [p - m for p, m in zip(_G1_BASE_PHI, _G1_BI_MARGIN)]
_G1_BI_HI = [p + m for p, m in zip(_G1_BASE_PHI, _G1_BI_MARGIN)]


# Actuated-joint selection: the same three patterns the G1 robot config
# uses for ``actuated_dof_patterns`` (29 DOFs; excludes the floating base,
# the legacy hand_palm joints, and the welded foot-frame dummies).
_G1_ACTUATED_PATTERNS = (
    r"left_(?!hand_palm_joint|foot_frame_joint).*",
    r"right_(?!hand_palm_joint|foot_frame_joint).*",
    r"waist_(?!support_joint).*",
)

# Actuated joints in g1.xml definition order (= act_manager.actuated_
# joint_names ordering, the column order kp_all / kd_all use). Exported so
# physics.py can map a per-DOF GT list onto the robot override dict and the
# SysID ground_truth array consistently.
_G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Nominal per-joint PD in g1.xml actuated-joint order:
#   L leg : hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
#   R leg : (same six)
#   waist : yaw, roll, pitch
#   L arm : shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
#           wrist_roll, wrist_pitch, wrist_yaw
#   R arm : (same seven)
# Each entry is the motor-class gain that joint's name resolves to in the
# robot config's p_gains / d_gains regex map.
_G1_KP_NOMINAL = [
    STIFFNESS_7520_14, STIFFNESS_7520_22, STIFFNESS_7520_14,
    STIFFNESS_7520_22, STIFFNESS_ANKLE, STIFFNESS_ANKLE,        # L leg
    STIFFNESS_7520_14, STIFFNESS_7520_22, STIFFNESS_7520_14,
    STIFFNESS_7520_22, STIFFNESS_ANKLE, STIFFNESS_ANKLE,        # R leg
    STIFFNESS_7520_14, STIFFNESS_WAIST, STIFFNESS_WAIST,        # waist
    STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020,
    STIFFNESS_5020, STIFFNESS_4010, STIFFNESS_4010,             # L arm
    STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020,
    STIFFNESS_5020, STIFFNESS_4010, STIFFNESS_4010,             # R arm
]
_G1_KD_NOMINAL = [
    DAMPING_7520_14, DAMPING_7520_22, DAMPING_7520_14,
    DAMPING_7520_22, DAMPING_ANKLE, DAMPING_ANKLE,              # L leg
    DAMPING_7520_14, DAMPING_7520_22, DAMPING_7520_14,
    DAMPING_7520_22, DAMPING_ANKLE, DAMPING_ANKLE,              # R leg
    DAMPING_7520_14, DAMPING_WAIST, DAMPING_WAIST,              # waist
    DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020,
    DAMPING_5020, DAMPING_4010, DAMPING_4010,                   # L arm
    DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020,
    DAMPING_5020, DAMPING_4010, DAMPING_4010,                   # R arm
]


@dataclass
class G1NewtonInertialPDConfig(SysIDConfig):
    """G1 Newton: log-Cholesky base inertial (10) + per-joint kp/kd
    (29 + 29) = 68D. No friction.

      * ``base_inertial`` (10D): pelvis mass / CoM / full inertia as a
        single unconstrained log-Cholesky phi (every step is informative).
      * ``kp_all`` / ``kd_all`` (29D each): per-joint stiffness / damping,
        matched by ``_G1_ACTUATED_PATTERNS`` in actuated-joint order.
    """

    base_inertial = SysIDTermConfig(
        func=nf.apply_log_cholesky_inertial,
        dim=10,
        default=_G1_BASE_PHI,
        bounds=(_G1_BI_LO, _G1_BI_HI),
        reg_weight=0.0,
        params={"body_pattern": r".*pelvis$"},
        gt_encoder=_g1_inertial_gt_encoder,
        decoder=decode_phi_to_physical_dict,
    )
    # Bounds kept just wide enough to cover both the nominal gains
    # (kp 14..99, kd 0.9..6.3) and the heterogeneous GT (kp 10.7..84.2,
    # kd 0.68..8.5). The earlier (5,150)/(0.1,15) range let CMA reach
    # kp≈150 / kp≈5 extremes that destabilise the biped (it falls →
    # self-collision contacts overflow nefc → nan); tightening to the
    # physical range removes those divergent candidates.
    kp_all = SysIDTermConfig(
        func=nf.apply_kp,
        dim=29,
        default=_G1_KP_NOMINAL,
        bounds=(8.0, 110.0),
        reg_weight=0.0,
        params={"joint_patterns": _G1_ACTUATED_PATTERNS},
        delta_param=2.0,
    )
    kd_all = SysIDTermConfig(
        func=nf.apply_kd,
        dim=29,
        default=_G1_KD_NOMINAL,
        bounds=(0.3, 10.0),
        reg_weight=0.0,
        params={"joint_patterns": _G1_ACTUATED_PATTERNS},
        delta_param=0.1,
    )


@dataclass
class G1NewtonPDOnlyConfig(G1NewtonInertialPDConfig):
    """Diagnostic/ablation: identify per-joint kp/kd only (58D).

    Disables ``base_inertial`` (kept at nominal) so the base-inertial
    log-Cholesky search can't drive non-physical bodies that destabilise
    the sim. Use this to isolate whether base_inertial divergence — not
    PD or solver buffers — is the source of nan losses.
    """

    base_inertial = None
