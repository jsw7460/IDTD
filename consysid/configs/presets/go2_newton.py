"""Go2 Newton SysID preset configurations."""
from dataclasses import dataclass

import numpy as np

from consysid.configs.sysid_term_config import SysIDConfig, SysIDTermConfig
from consysid.sysid.log_cholesky import (
    decode_phi_to_physical_dict,
    encode_physical_to_phi,
    mujoco_inertial_to_body_inertia,
)
from consysid.sysid.param_terms import newton as nf


# Go2 base link ("trunk") URDF defaults from
# Mjlab/src/mjlab/asset_zoo/robots/unitree_go2/xmls/go2.xml.
# MuJoCo stores inertia in principal-axis frame; we rebuild the body-frame
# 3x3 tensor here so it matches Newton's body_inertia convention.
_GO2_BASE_MASS = 6.921
_GO2_BASE_COM = np.array([0.021112, 0.0, -0.005366])
_GO2_BASE_DIAGINERTIA = np.array([0.107027, 0.0980771, 0.0244531])
_GO2_BASE_IQUAT_WXYZ = np.array(
    [-0.000543471, 0.713435, -0.00173769, 0.700719]
)
_GO2_BASE_INERTIA = mujoco_inertial_to_body_inertia(
    _GO2_BASE_DIAGINERTIA, _GO2_BASE_IQUAT_WXYZ,
)
_GO2_BASE_PHI = encode_physical_to_phi(
    _GO2_BASE_MASS, _GO2_BASE_COM, _GO2_BASE_INERTIA,
).tolist()


def _go2_inertial_gt_encoder(gt: dict) -> np.ndarray:
    """Convert user-supplied physical GT dict to log-Cholesky phi."""
    return encode_physical_to_phi(
        mass=gt["mass"],
        com=np.asarray(gt["com"]),
        inertia=np.asarray(gt["inertia"]),
    )


@dataclass
class Go2NewtonMassComConfig(SysIDConfig):
    """Go2 Newton: mass + CoM (4D)."""
    base_mass = SysIDTermConfig(
        func=nf.apply_mass,
        dim=1,
        default=6.921,
        bounds=(3.0, 20.0),
        reg_weight=0.005,
        params={"body_pattern": ".*trunk$"},
    )
    base_com = SysIDTermConfig(
        func=nf.apply_com,
        dim=3,
        default=[0.0, 0.0, 0.0],
        bounds=(-0.5, 0.5),
        reg_weight=1.0,
        params={"body_pattern": ".*trunk$"},
    )


@dataclass
class Go2NewtonMassComFrictionConfig(Go2NewtonMassComConfig):
    """Go2 Newton: mass + CoM + foot friction (5D)."""
    base_mass = SysIDTermConfig(
        func=nf.apply_mass,
        dim=1,
        default=6.921,
        bounds=(3.0, 20.0),
        reg_weight=0.001,
        params={"body_pattern": ".*trunk$"},
    )
    foot_friction = SysIDTermConfig(
        func=nf.apply_contact_friction,
        dim=1,
        default=1.0,
        bounds=(0.1, 2.0),
        reg_weight=0.05,
        params={},
    )


@dataclass
class Go2NewtonInertialConfig(SysIDConfig):
    """Go2 Newton: log-Cholesky pseudo-inertia (10D).

    Subsumes mass + CoM + full 3x3 inertia tensor of the base link into
    a single 10D unconstrained parameter phi (Sobanbabu et al. 2025,
    Eq. 2). Every phi decodes to a physically valid (m, r, I) triple,
    so CMA-ES can explore R^10 without box constraints on individual
    physical quantities.

    Do not combine this with ``Go2NewtonMassComConfig``-style terms —
    both would write ``body_mass``/``body_com`` and collide.
    """
    base_inertial = SysIDTermConfig(
        func=nf.apply_log_cholesky_inertial,
        dim=10,
        default=_GO2_BASE_PHI,
        bounds=(-5.0, 5.0),
        reg_weight=0.0,
        params={"body_pattern": ".*trunk$"},
        gt_encoder=_go2_inertial_gt_encoder,
        decoder=decode_phi_to_physical_dict,
    )


@dataclass
class Go2NewtonInertialFrictionConfig(Go2NewtonInertialConfig):
    """10D log-Cholesky inertial + 1D foot friction = 11D."""
    foot_friction = SysIDTermConfig(
        func=nf.apply_contact_friction,
        dim=1,
        default=1.0,
        bounds=(0.1, 2.0),
        reg_weight=0.0,
        params={},
    )


@dataclass
class Go2NewtonJointFrictionConfig(SysIDConfig):
    """Go2 Newton: joint friction loss only (12D, one per leg DOF).

    Useful as a minimal standalone test of the joint-friction apply path.
    Default from go2.xml ``<default class="go2">``: frictionloss=0.2 N·m.
    """
    joint_friction = SysIDTermConfig(
        func=nf.apply_joint_friction,
        dim=12,
        default=[0.0] * 12,
        bounds=(0.0, 2.0),
        reg_weight=0.01,
        params={"joint_patterns": (r".*_joint$",)},
    )


@dataclass
class Go2NewtonInertialAllFrictionConfig(Go2NewtonInertialFrictionConfig):
    """10D log-Cholesky inertial + 1D foot contact friction + 12D joint
    friction = 23D.

    Simultaneously identifies the full base-link inertial parameters,
    foot-ground contact friction, and per-leg joint Coulomb friction.
    """
    joint_friction = SysIDTermConfig(
        func=nf.apply_joint_friction,
        dim=12,
        default=[0.0] * 12,
        bounds=(0.0, 2.0),
        reg_weight=0.01,
        params={"joint_patterns": (r".*_joint$",)},
    )


@dataclass
class Go2NewtonDecoupledInertialAllFrictionConfig(Go2NewtonInertialAllFrictionConfig):
    """Same 23D parameter set as ``Go2NewtonInertialAllFrictionConfig``
    but with θ/φ category tags set for the decoupled active-exploration
    objective ``λ_off · ‖I(θ,φ)‖_F − λ_θ · tr(I(θ,θ)) − λ_φ · tr(I(θ,θ))``:

        θ (target):   base_inertial (log-Cholesky, 10D) — mass, CoM, inertia
        φ (nuisance): foot_friction (1D), joint_friction (12D)

    Use alongside ``FIMConfig.decoupled(...)`` during active exploration.
    The SPI-Active baseline parent class is unchanged — running it
    with ``FIMConfig()`` defaults still yields the original
    ``-tr(F)`` objective.

    Per-term ``delta_param`` is also specified here: foot friction is
    bounded (0.1, 2.0) so δ=0.05 is a ~2.5% relative step, and joint
    friction has a natural scale near 0.2 N·m so δ=0.02 is ~10%.
    log-Cholesky phi is unconstrained O(1) so the default δ=0.1
    inherited from the partition config remains appropriate.
    """
    base_inertial = SysIDTermConfig(
        func=nf.apply_log_cholesky_inertial,
        dim=10,
        default=_GO2_BASE_PHI,
        bounds=(-5.0, 5.0),
        reg_weight=0.01,
        params={"body_pattern": ".*trunk$"},
        gt_encoder=_go2_inertial_gt_encoder,
        decoder=decode_phi_to_physical_dict,
        category="theta",
    )
    foot_friction = SysIDTermConfig(
        func=nf.apply_contact_friction,
        dim=1,
        default=1.0,
        bounds=(0.1, 2.0),
        reg_weight=0.05,
        params={},
        category="phi",
        delta_param=0.05,
    )
    joint_friction = SysIDTermConfig(
        func=nf.apply_joint_friction,
        dim=12,
        default=[0.0] * 12,
        bounds=(0.0, 2.0),
        reg_weight=0.01,
        params={"joint_patterns": (r".*_joint$",)},
        category="phi",
        delta_param=0.02,
    )


# ---------------------------------------------------------------------------
# Full 36D: log-Cholesky base_inertial (10) + per-joint PD (12 + 12) +
# scalar joint_friction (1) + foot_friction (1).
#
# Collect side bakes hip/knee-group scalar PD GT (homogeneous across
# the four legs) via ``apply_to_robot_overrides``, so a correct SysID
# fit ends up with the four hip joints at the same kp/kd value (and
# likewise for thighs and calves). The per-joint search space is
# intentionally larger than the collect-side parameterisation so the
# 1D-per-joint aliasing structure is exposed -- different joints
# become identifiable in different gait phases (swing-heavy → calf
# joints, stance-heavy → foot_friction + hip joints, etc.), the regime
# where decoupling-coverage exploration separates axes that a uniform
# ``-tr(F)`` objective treats identically.
# ---------------------------------------------------------------------------


@dataclass
class Go2NewtonFullMixedConfig(SysIDConfig):
    """36D full mixed: base_inertial (10) + per-joint kp/kd (24) +
    joint_friction scalar (1) + foot_friction (1).

    A harder target than the 16D setup. Each axis has a distinct phase
    signature:

      * ``base_inertial``: every dynamic step (gravity / acceleration)
      * ``kp_all`` / ``kd_all`` per joint: that joint's swing or stance
        phase (swing knee for kd_calf, stance hip for kp_hip, etc.)
      * ``joint_friction``: high-velocity motion (lift-off / impact)
      * ``foot_friction``: stance only (slip threshold)
    """

    base_inertial = SysIDTermConfig(
        func=nf.apply_log_cholesky_inertial,
        dim=10,
        default=_GO2_BASE_PHI,
        bounds=(
            [0.55, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -2.0, -0.3, -0.3],
            [1.50,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  2.0,  0.3,  0.3],
        ),
        reg_weight=0.0,
        params={"body_pattern": ".*trunk$"},
        gt_encoder=_go2_inertial_gt_encoder,
        decoder=decode_phi_to_physical_dict,
    )

    # Per-joint PD (24D). Defaults match JaxRLWorld's
    # ``STIFFNESS_HIP`` / ``STIFFNESS_KNEE`` / ``DAMPING_HIP`` /
    # ``DAMPING_KNEE`` constants in robots/go2.py; the [hip, thigh,
    # calf] pattern repeats for each of the four legs in the joint
    # ordering ``act_manager.actuated_joint_names`` exposes
    # (FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_..., RR_...).
    kp_all = SysIDTermConfig(
        func=nf.apply_kp,
        dim=12,
        default=[17.6918, 17.6918, 65.2191] * 4,
        bounds=(5.0, 100.0),
        reg_weight=0.0,
        params={"joint_patterns": (r".*_joint$",)},
        delta_param=2.0,
    )
    kd_all = SysIDTermConfig(
        func=nf.apply_kd,
        dim=12,
        default=[1.1263, 1.1263, 4.1520] * 4,
        bounds=(0.1, 10.0),
        reg_weight=0.0,
        params={"joint_patterns": (r".*_joint$",)},
        delta_param=0.1,
    )

    # Scalar joint_friction (1D, broadcast across all 12 leg joints)
    # -- matches the ``joint_frictionloss`` field on the collect preset
    # which is also scalar. ``apply_joint_friction_scalar`` broadcasts
    # the 1-element value to every matched joint.
    joint_friction = SysIDTermConfig(
        func=nf.apply_joint_friction_scalar,
        dim=1,
        default=0.0,
        bounds=(0.0, 5.0),
        reg_weight=0.0,
        params={"joint_patterns": (r".*_joint$",)},
        delta_param=0.05,
    )

    foot_friction = SysIDTermConfig(
        func=nf.apply_contact_friction,
        dim=1,
        default=1.0,
        bounds=(0.1, 2.0),
        reg_weight=0.0,
        params={},
        delta_param=0.05,
    )


class Go2NewtonFullPerFrictionConfig(SysIDConfig):
    """50D full mixed with HETEROGENEOUS friction:
    base_inertial (10) + per-joint kp/kd (24) + per-DOF joint_friction (12)
    + per-foot foot_friction (4).

    Same as ``Go2NewtonFullMixedConfig`` but the two scalar friction terms
    become per-dimension, paired with ``GT_V3``'s heterogeneous friction.
    All 36D terms are re-declared (not inherited) so ``iter_terms``'s MRO
    walk keeps the term order base_inertial → kp_all → kd_all →
    joint_friction → foot_friction (an override-only subclass would reorder
    the overridden terms to the front and shift the theta layout).
    """

    base_inertial = SysIDTermConfig(
        func=nf.apply_log_cholesky_inertial,
        dim=10,
        default=_GO2_BASE_PHI,
        bounds=(
            [0.55, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -2.0, -0.3, -0.3],
            [1.50,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  2.0,  0.3,  0.3],
        ),
        reg_weight=0.0,
        params={"body_pattern": ".*trunk$"},
        gt_encoder=_go2_inertial_gt_encoder,
        decoder=decode_phi_to_physical_dict,
    )
    kp_all = SysIDTermConfig(
        func=nf.apply_kp,
        dim=12,
        default=[17.6918, 17.6918, 65.2191] * 4,
        bounds=(5.0, 100.0),
        reg_weight=0.0,
        params={"joint_patterns": (r".*_joint$",)},
        delta_param=2.0,
    )
    kd_all = SysIDTermConfig(
        func=nf.apply_kd,
        dim=12,
        default=[1.1263, 1.1263, 4.1520] * 4,
        bounds=(0.1, 10.0),
        reg_weight=0.0,
        params={"joint_patterns": (r".*_joint$",)},
        delta_param=0.1,
    )
    # Per-DOF joint friction (12D). ``apply_joint_friction`` writes one
    # value per matched DOF (FL_hip, FL_thigh, FL_calf, FR_..., order of
    # ``act_manager.actuated_joint_names``).
    joint_friction = SysIDTermConfig(
        func=nf.apply_joint_friction,
        dim=12,
        default=[0.0] * 12,
        bounds=(0.0, 5.0),
        reg_weight=0.0,
        params={"joint_patterns": (r".*_joint$",)},
        delta_param=0.05,
    )
    # Per-foot foot friction (4D, FL/FR/RL/RR in foot-collision-shape
    # order). ``apply_contact_friction`` accepts (B, 4).
    foot_friction = SysIDTermConfig(
        func=nf.apply_contact_friction,
        dim=4,
        default=[1.0] * 4,
        bounds=(0.1, 2.0),
        reg_weight=0.0,
        params={},
        delta_param=0.05,
    )


@dataclass
class Go2NewtonMassComGainsConfig(Go2NewtonMassComConfig):
    """Go2 Newton: mass + CoM + PD gains (28D).

    Newton uses implicit PD via ``joint_target_ke``/``joint_target_kd``.
    Joint selection is name-based via ``robot_view``: the default
    ``joint_patterns=(r".*_joint$",)`` picks up Go2's 12 actuated leg
    joints while skipping the ``floating_base`` free joint.
    """
    kp_all = SysIDTermConfig(
        func=nf.apply_kp,
        dim=12,
        default=[20.0] * 12,
        bounds=(10.0, 40.0),
        reg_weight=0.001,
        params={"joint_patterns": (r".*_joint$",)},
    )
    kd_all = SysIDTermConfig(
        func=nf.apply_kd,
        dim=12,
        default=[0.5] * 12,
        bounds=(0.1, 2.0),
        reg_weight=0.01,
        params={"joint_patterns": (r".*_joint$",)},
    )


# ---------------------------------------------------------------------------
# Scalar-joint-friction ablation configs (12D total instead of 23D).
#
# ``Go2NewtonInertialAllFrictionConfig`` identifies per-joint Coulomb friction
# (12D) which is the physically-accurate setting but also the dimension that
# most dilutes objective-specific effects in the ablation: per-joint friction
# is weakly identifiable from walking data, so finite-sample CMA-ES noise on
# those 12 dims swamps any Schur / bi-Schur improvement.
#
# These presets collapse joint friction to a SINGLE scalar shared across all
# 12 actuated joints (``apply_joint_friction_scalar`` broadcasts at apply
# time), leaving the base-inertial structure (10D log-Cholesky) intact. Net
# param_dim = 10 + 1 (foot) + 1 (joint scalar) = 12. CMA-ES converges many
# times faster and objective comparisons become much cleaner — use this for
# baseline / Schur / bi-Schur ablation studies on the objective itself.
# ---------------------------------------------------------------------------


@dataclass
class Go2NewtonInertialFrictionScalarConfig(Go2NewtonInertialFrictionConfig):
    """10D log-Cholesky inertial + 1D foot contact friction + 1D **scalar**
    joint friction = 12D.

    Baseline-compatible variant: every term keeps the default
    ``category="theta"`` so ``FIMConfig`` with default λ's reduces to
    pure SPI-Active ``cost = -tr(F)`` over the full 12D block.
    """
    joint_friction = SysIDTermConfig(
        func=nf.apply_joint_friction_scalar,
        dim=1,
        default=0.2,
        bounds=(0.0, 2.0),
        reg_weight=0.0,
        params={"joint_patterns": (r".*_joint$",)},
    )


@dataclass
class Go2NewtonDecoupledInertialFrictionScalarConfig(
    Go2NewtonInertialFrictionScalarConfig,
):
    """Scalar-friction decoupled variant (12D total):

        θ (target, 10D)   : base_inertial (log-Cholesky)
        φ (nuisance, 2D)  : foot_friction (1D) + joint_friction scalar (1D)

    Minimal φ block — the Schur / bi-Schur inversions are over at most a
    2×2, so Tikhonov regularisation is effectively free. Designed as the
    low-dim ablation companion to
    ``Go2NewtonDecoupledInertialAllFrictionConfig``: same structural
    split, dimension collapsed so objective differences aren't drowned
    by finite-sample noise across 23 jointly-optimised parameters.

    Per-term ``delta_param`` mirrors the 23D preset's rationale
    (friction δ smaller than inertial-φ's natural O(1) scale).
    """
    base_inertial = SysIDTermConfig(
        func=nf.apply_log_cholesky_inertial,
        dim=10,
        default=_GO2_BASE_PHI,
        bounds=(-5.0, 5.0),
        reg_weight=0.01,
        params={"body_pattern": ".*trunk$"},
        gt_encoder=_go2_inertial_gt_encoder,
        decoder=decode_phi_to_physical_dict,
        category="theta",
    )
    foot_friction = SysIDTermConfig(
        func=nf.apply_contact_friction,
        dim=1,
        default=1.0,
        bounds=(0.1, 2.0),
        reg_weight=0.05,
        params={},
        category="phi",
        delta_param=0.05,
    )
    joint_friction = SysIDTermConfig(
        func=nf.apply_joint_friction_scalar,
        dim=1,
        default=0.2,
        bounds=(0.0, 2.0),
        reg_weight=0.01,
        params={"joint_patterns": (r".*_joint$",)},
        category="phi",
        delta_param=0.02,
    )


# ---------------------------------------------------------------------------
# Scalar PD-gain extension (16D = 10 inertial + 1 foot fric + 1 joint fric
# scalar + 2 kp scalars (hip/thigh vs calf groups) + 2 kd scalars).
#
# Matches the joint grouping used in collect-time presets
# (``Go2GaitConditionedSysIDCollectConfig``): the GT generates one kp/kd
# pair per actuator group (.*_hip_joint + .*_thigh_joint) and another
# for (.*_calf_joint), so identifying 4 scalars instead of per-joint
# 12+12 = 24 keeps SysID dim low and converges faster.
# ---------------------------------------------------------------------------


@dataclass
class Go2NewtonInertialFrictionScalarPDConfig(Go2NewtonInertialFrictionScalarConfig):
    """16D: 10 log-Cholesky inertial + 1 foot friction + 1 joint friction
    scalar + 4 PD scalar gains (kp_hip, kd_hip, kp_knee, kd_knee).

    Pair with the matching collect preset that overrides the same four
    PD scalars. Hip/thigh share one kp/kd group (motors are spec-matched
    in Unitree's setup), calf gets its own (higher-stiffness, larger
    armature).

    Defaults track ``jaxrlworld/rl/configs/robots/go2.py`` constants:
        STIFFNESS_HIP  ≈ 17.69   DAMPING_HIP  ≈ 1.13
        STIFFNESS_KNEE ≈ 65.22   DAMPING_KNEE ≈ 4.15
    Bounds give SysID room on either side of these defaults so the
    "real-like" values (~0.7x kp, ~0.7x kd) sit comfortably inside.
    """

    # Tighten ``base_inertial`` per-dim bounds. The base class's uniform
    # (-5, 5) bound on the log-Cholesky phi vector lets the decoded mass
    # range from ~0.045 g to ~22 tons because mass = e^(2·phi[0]) ·
    # (1 + ‖phi[7:10]‖²) — phi[0] = 5 alone gives mass ≈ 22000 kg.
    # When PD gains are co-identified the optimizer has more freedom to
    # absorb residuals into mass, and we observed runaway estimates of
    # 80–150 kg in practice. Constrain phi component-wise so decoded
    # mass stays in a plausible Go2 range:
    #     phi[0] (≈ ½·log m)         ∈ [0.55, 1.50]
    #     phi[1:4] (d_i, log scale)  ∈ [-2.0,  2.0]
    #     phi[4:7] (s_ij coupling)   ∈ [-1.5,  1.5]
    #     phi[7:10] (CoM-related t)  ∈ [-0.3,  0.3]
    base_inertial = SysIDTermConfig(
        func=nf.apply_log_cholesky_inertial,
        dim=10,
        default=_GO2_BASE_PHI,
        bounds=(
            #  α    d1    d2    d3    s12   s13   s23   t1    t2    t3
            [0.55, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -2.0, -0.3, -0.3],
            [1.50,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  2.0,  0.3,  0.3],
        ),
        reg_weight=0.0,
        params={"body_pattern": ".*trunk$"},
        gt_encoder=_go2_inertial_gt_encoder,
        decoder=decode_phi_to_physical_dict,
    )

    # Per-term ``delta_param`` overrides the EnvPartitionConfig global
    # (typically 0.1) for active-exploration FIM finite-differences. The
    # natural scale of PD gains spans two orders of magnitude (kp_knee ≈
    # 65 vs kd_hip ≈ 1.1), so a uniform 0.1 would be either trivially
    # small (~0.15% on kp_knee → buried in numerical noise) or
    # overzealous (~9% on kd_hip → outside the locally-linear regime).
    # Values below target ~5% relative perturbation around defaults.
    kp_hip = SysIDTermConfig(
        func=nf.apply_kp_scalar,
        dim=1,
        default=17.6918,
        bounds=(10.0, 20.0),
        reg_weight=0.0,
        params={"joint_patterns": (r".*_hip_joint", r".*_thigh_joint")},
        delta_param=2.0,
    )
    kp_knee = SysIDTermConfig(
        func=nf.apply_kp_scalar,
        dim=1,
        default=65.2191,
        # default=60.0,
        bounds=(50.0, 70.0),
        reg_weight=0.0,
        params={"joint_patterns": (r".*_calf_joint",)},
        delta_param=3.0,
    )
    kd_hip = SysIDTermConfig(
        func=nf.apply_kd_scalar,
        dim=1,
        default=1.1263,
        bounds=(0.5, 1.8),
        reg_weight=0.0,
        params={"joint_patterns": (r".*_hip_joint", r".*_thigh_joint")},
        delta_param=0.05,
    )
    kd_knee = SysIDTermConfig(
        func=nf.apply_kd_scalar,
        dim=1,
        default=4.1520,
        bounds=(2.0, 5.0),
        reg_weight=0.0,
        params={"joint_patterns": (r".*_calf_joint",)},
        delta_param=0.5,
    )


# ---------------------------------------------------------------------------
# 4D demo: identify kp_hip / kp_knee / kd_knee / foot_friction. Each
# axis carries a distinct phase signature (stance posture / push-off /
# swing retract / contact slip) so the identifiable subset varies along
# the trajectory's swing/stance pattern -- a minimal axis set where
# decoupling-coverage exploration separates axes that a uniform
# ``-tr(F)`` objective aggregates, on a per-axis ``|theta - theta*|``
# chart. base_inertial / joint_friction / kd_hip are pinned at the
# nominal MJCF.
# ---------------------------------------------------------------------------


@dataclass
class Go2NewtonKpKdKneeFootFricConfig(Go2NewtonInertialFrictionScalarPDConfig):
    """4D ablation: kp_hip / kp_knee / kd_knee / foot_friction.

    Inherits term definitions verbatim from
    ``Go2NewtonInertialFrictionScalarPDConfig`` (PD bounds / delta_param
    / regularisation already tuned) and zeros out every other field via
    the SysIDConfig ``None`` disable pattern -- the remaining axes are
    skipped by ``iter_terms`` so the matching apply functions are never
    called and the simulator value (the nominal MJCF default) survives
    the run. Pairs with
    ``scripts/go2/collect/kp_kdknee_footfric_overrides.py`` and the
    4D identification / refinement / explore scripts.
    """

    base_inertial = None
    joint_friction = None
    kd_hip = None
    # kp_hip / kp_knee / kd_knee / foot_friction inherited unchanged.


# ---------------------------------------------------------------------------
# Multi-group split for the ``multi_schur`` FIM objective (16D, three
# blocks). Pure parameter relabelling on top of
# ``Go2NewtonInertialFrictionScalarPDConfig`` — same dim, same defaults,
# same bounds; only ``category`` changes so the FIM evaluator can carve
# the matrix into three blocks instead of one or two.
#
#   inertial (10D) ← base_inertial
#   pd       ( 4D) ← kp_hip, kd_hip, kp_knee, kd_knee
#   friction ( 2D) ← foot_friction, joint_friction
#
# Multi-schur per-group conditional Fisher: for each block i,
#     S_i = F_ii − F_{i,~i} · F_{~i,~i}^{-1} · F_{~i,i}
# rewards commands that produce motion identifying block i even when
# the other two blocks are co-estimated. Cleaner than bi_schur in
# this case because inertial / PD / friction have distinct physical
# roles and each contributes to a different regime of joint torques
# (gravity & accel terms vs. position/velocity tracking vs. contact &
# Coulomb dissipation).
# ---------------------------------------------------------------------------


@dataclass
class Go2NewtonMultiGroupInertialFrictionPDConfig(
    Go2NewtonInertialFrictionScalarPDConfig,
):
    """3-group split of the 16D PD+friction+inertial config for
    ``FIMConfig.objective="multi_schur"``.

    Inherits every term from ``Go2NewtonInertialFrictionScalarPDConfig``
    and only re-tags the ``category`` field so the FIM evaluator
    discovers three groups (``"inertial"``, ``"pd"``, ``"friction"``)
    instead of the default single ``"theta"`` block. Defaults / bounds
    / delta_param / reg_weight all carried over unchanged — set those
    on the parent class if you want to retune.
    """

    # ── inertial group (10D) ───────────────────────────────────────────
    base_inertial = SysIDTermConfig(
        func=nf.apply_log_cholesky_inertial,
        dim=10,
        default=_GO2_BASE_PHI,
        bounds=(
            #  α    d1    d2    d3    s12   s13   s23   t1    t2    t3
            [0.55, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -0.3, -0.3, -0.3],
            [1.70,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  0.3,  0.3,  0.3],
        ),
        reg_weight=0.01,
        params={"body_pattern": ".*trunk$"},
        gt_encoder=_go2_inertial_gt_encoder,
        decoder=decode_phi_to_physical_dict,
        category="inertial",
    )

    # ── friction group (2D) ────────────────────────────────────────────
    foot_friction = SysIDTermConfig(
        func=nf.apply_contact_friction,
        dim=1,
        default=1.0,
        bounds=(0.1, 2.0),
        reg_weight=0.05,
        params={},
        category="friction",
    )
    joint_friction = SysIDTermConfig(
        func=nf.apply_joint_friction_scalar,
        dim=1,
        default=0.2,
        bounds=(0.0, 2.0),
        reg_weight=0.01,
        params={"joint_patterns": (r".*_joint$",)},
        category="friction",
    )

    # ── pd group (4D) ──────────────────────────────────────────────────
    kp_hip = SysIDTermConfig(
        func=nf.apply_kp_scalar,
        dim=1,
        default=17.6918,
        bounds=(5.0, 30.0),
        reg_weight=0.001,
        params={"joint_patterns": (r".*_hip_joint", r".*_thigh_joint")},
        delta_param=1.0,
        category="pd",
    )
    kp_knee = SysIDTermConfig(
        func=nf.apply_kp_scalar,
        dim=1,
        default=65.2191,
        bounds=(20.0, 100.0),
        reg_weight=0.001,
        params={"joint_patterns": (r".*_calf_joint",)},
        delta_param=3.0,
        category="pd",
    )
    kd_hip = SysIDTermConfig(
        func=nf.apply_kd_scalar,
        dim=1,
        default=1.1263,
        bounds=(0.1, 3.0),
        reg_weight=0.01,
        params={"joint_patterns": (r".*_hip_joint", r".*_thigh_joint")},
        delta_param=0.05,
        category="pd",
    )
    kd_knee = SysIDTermConfig(
        func=nf.apply_kd_scalar,
        dim=1,
        default=4.1520,
        bounds=(0.5, 6.0),
        reg_weight=0.01,
        params={"joint_patterns": (r".*_calf_joint",)},
        delta_param=0.2,
        category="pd",
    )


# ---------------------------------------------------------------------------
# Physics-aligned 3-group split. Same 16D parameter set as
# ``Go2NewtonMultiGroupInertialFrictionPDConfig`` but ``joint_friction``
# moves out of the friction block (which bundled it with foot_friction
# purely on a name match) and into the PD block.
#
# Rationale: in the joint-torque equation
#     τ_joint = kp·(q_des − q) + kd·(−q̇) − f_j·sign(q̇)
# joint_friction shares the same channel as PD gains and is irreducibly
# entangled with them in steady-state walking — multi_schur's between-
# block Schur correction can never decouple variables in the *same*
# block, so putting them together lets the optimizer exploit Schur
# where it actually helps (between actuator / inertial / contact)
# instead of fighting an unwinnable within-block fight.
#
# foot_friction lives in the contact constraint
#     |F_tangent| ≤ μ · F_normal
# which is independent of the joint-torque channel, so it stays on its
# own as the ``contact`` block.
# ---------------------------------------------------------------------------


@dataclass
class Go2NewtonMultiGroupInertialActuatorContactConfig(
    Go2NewtonInertialFrictionScalarPDConfig,
):
    """Physics-aligned 3-group split of the 16D PD+friction+inertial
    config for ``FIMConfig.objective="multi_schur"``.

    Differs from ``Go2NewtonMultiGroupInertialFrictionPDConfig`` only
    in *where* ``joint_friction`` lives:

    +-------------------+--------------+----------------+
    | term              | old grouping | new grouping   |
    +===================+==============+================+
    | base_inertial     | inertial     | inertial       |
    | foot_friction     | friction     | contact        |
    | joint_friction    | friction     | **actuator**   |
    | kp/kd hip+knee    | pd           | actuator       |
    +-------------------+--------------+----------------+

    Resulting block sizes: inertial=10D, actuator=5D, contact=1D.
    Inherits every function / dim / default / bounds / delta_param
    from the parent class — only ``category`` labels change.
    """

    # ── inertial group (10D) ───────────────────────────────────────────
    base_inertial = SysIDTermConfig(
        func=nf.apply_log_cholesky_inertial,
        dim=10,
        default=_GO2_BASE_PHI,
        bounds=(
            #  α    d1    d2    d3    s12   s13   s23   t1    t2    t3
            [0.55, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -2.0, -0.3, -0.3],
            [1.70,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  2.0,  0.3,  0.3],
        ),
        reg_weight=0.01,
        params={"body_pattern": ".*trunk$"},
        gt_encoder=_go2_inertial_gt_encoder,
        decoder=decode_phi_to_physical_dict,
        category="inertial",
    )

    # ── contact group (1D) — foot-ground friction only ─────────────────
    foot_friction = SysIDTermConfig(
        func=nf.apply_contact_friction,
        dim=1,
        default=1.0,
        bounds=(0.1, 2.0),
        reg_weight=0.05,
        params={},
        category="contact",
    )

    # ── actuator group (5D) — PD gains + joint Coulomb friction ────────
    # All five enter the joint torque equation directly and share the
    # actuator-dynamics channel; bundling them lets multi_schur stop
    # trying to decouple the irreducibly-entangled PD↔jf pair.
    joint_friction = SysIDTermConfig(
        func=nf.apply_joint_friction_scalar,
        dim=1,
        default=0.2,
        bounds=(0.0, 2.0),
        reg_weight=0.01,
        params={"joint_patterns": (r".*_joint$",)},
        category="actuator",
    )
    kp_hip = SysIDTermConfig(
        func=nf.apply_kp_scalar,
        dim=1,
        default=17.6918,
        bounds=(5.0, 30.0),
        reg_weight=0.001,
        params={"joint_patterns": (r".*_hip_joint", r".*_thigh_joint")},
        delta_param=1.0,
        category="actuator",
    )
    kp_knee = SysIDTermConfig(
        func=nf.apply_kp_scalar,
        dim=1,
        default=65.2191,
        bounds=(20.0, 100.0),
        reg_weight=0.001,
        params={"joint_patterns": (r".*_calf_joint",)},
        delta_param=3.0,
        category="actuator",
    )
    kd_hip = SysIDTermConfig(
        func=nf.apply_kd_scalar,
        dim=1,
        default=1.1263,
        bounds=(0.1, 3.0),
        reg_weight=0.01,
        params={"joint_patterns": (r".*_hip_joint", r".*_thigh_joint")},
        delta_param=0.05,
        category="actuator",
    )
    kd_knee = SysIDTermConfig(
        func=nf.apply_kd_scalar,
        dim=1,
        default=4.1520,
        bounds=(0.5, 6.0),
        reg_weight=0.01,
        params={"joint_patterns": (r".*_calf_joint",)},
        delta_param=0.2,
        category="actuator",
    )


@dataclass
class Go2NewtonMultiGroupInertialPDFrictionContactConfig(
    Go2NewtonMultiGroupInertialActuatorContactConfig,
):
    """4-group split — splits ``joint_friction`` out of the actuator
    block into its own group, so ``multi_schur`` runs Schur-style
    marginalisation on the PD↔joint_friction pair as well.

    The parent ``...InertialActuatorContactConfig`` bundles PD and
    joint_friction together under ``category="actuator"`` on the
    rationale that they enter the joint-torque equation jointly. In
    practice the resulting same-group confound shows up as a
    systematic +12 % bias on both axes (PD ↑ ↔ joint_friction ↑
    cancel in trajectory space). Promoting ``joint_friction`` to its
    own category gives ``multi_schur`` a separable boundary to
    decouple it from PD.

    Block sizes change (10, 5, 1) → (10, 4, 1, 1)::

        inertial        10D — base_inertial
        pd               4D — kp/kd hip+knee
        joint_friction   1D — scalar joint Coulomb friction
        contact          1D — foot_friction

    Pair with ``FIMConfig.schur_group_weights`` reflecting the new
    split, e.g.::

        schur_group_weights={
            "inertial":       1.0,  # per-dim 0.1
            "pd":             2.0,  # per-dim 0.5
            "joint_friction": 2.0,  # per-dim 2.0
            "contact":       20.0,  # per-dim 20.0
        }
    """

    joint_friction = SysIDTermConfig(
        func=nf.apply_joint_friction_scalar,
        dim=1,
        default=0.2,
        bounds=(0.0, 2.0),
        reg_weight=0.01,
        params={"joint_patterns": (r".*_joint$",)},
        category="joint_friction",
    )
    kp_hip = SysIDTermConfig(
        func=nf.apply_kp_scalar,
        dim=1,
        default=17.6918,
        bounds=(5.0, 30.0),
        reg_weight=0.001,
        params={"joint_patterns": (r".*_hip_joint", r".*_thigh_joint")},
        delta_param=1.0,
        category="pd",
    )
    kp_knee = SysIDTermConfig(
        func=nf.apply_kp_scalar,
        dim=1,
        default=65.2191,
        bounds=(20.0, 100.0),
        reg_weight=0.001,
        params={"joint_patterns": (r".*_calf_joint",)},
        delta_param=3.0,
        category="pd",
    )
    kd_hip = SysIDTermConfig(
        func=nf.apply_kd_scalar,
        dim=1,
        default=1.1263,
        bounds=(0.1, 3.0),
        reg_weight=0.01,
        params={"joint_patterns": (r".*_hip_joint", r".*_thigh_joint")},
        delta_param=0.05,
        category="pd",
    )
    kd_knee = SysIDTermConfig(
        func=nf.apply_kd_scalar,
        dim=1,
        default=4.1520,
        bounds=(0.5, 6.0),
        reg_weight=0.01,
        params={"joint_patterns": (r".*_calf_joint",)},
        delta_param=0.2,
        category="pd",
    )
