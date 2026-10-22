"""Single source of truth for SysID physics parameter sets.

Centralizes the ground-truth, prior, and fitted physics values that
used to live as hardcoded constants in eight different scripts
(identification, refinement, data collection, Newton / MuJoCo
training, evaluation, contact-sanity diagnostics). Each named
:class:`PhysicsSet` groups the six identifiable PD / friction scalars
with the inertial-XML path so all sites import the same instance rather
than reproducing the values inline.

A run-wide experiment is selected by reassigning :data:`CURRENT_GT`
to a named set:

:data:`CURRENT_GT` is the "real world" the SysID pipeline tries to
identify -- the value baked into every collect environment, used as
the ``ground_truth`` label on every identification term, pinned by
the mujoco-ceiling training run, and pinned again by the mujoco-side
``eval_with_sysid_gt.py``. identification data is collected under it and
the identified theta_best should recover it (up to noise / bias).

To run a new shift, declare ``GT_V2`` below and switch the
``CURRENT_GT`` assignment -- every consumer picks up the change
without further edits.

Three adapter helpers translate :class:`PhysicsSet` into the format
each consumer expects:

* :func:`apply_to_robot_overrides` writes the six fields onto the
  ``Go2Config.*_override`` slots consumed by JaxRLWorld's Newton /
  MuJoCo Go2 builders.
* :func:`apply_to_collect_preset` writes the same six fields onto
  :class:`Go2GaitConditionedSysIDCollectConfig` (used by
  :class:`TrajectoryCollector` for collection and
  :class:`PolicyEvaluator` for cross-sim evaluation).
* :func:`apply_gt_to_sysid_config` writes the six fields into the
  matching ``SysIDTermConfig.ground_truth`` slots so the identification
  loss panel reports ``|theta - theta*|`` accurately.

:func:`load_fitted_from_npz` rebuilds a :class:`PhysicsSet` from a
``sysid_result.npz`` (identification output) for the post-identification
training / eval arm.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from consysid.configs.presets.g1_newton import (
    _G1_JOINT_NAMES,
    _G1_KD_NOMINAL,
    _G1_KP_NOMINAL,
)
from consysid.sysid.log_cholesky import decode_phi_to_physical_dict

if TYPE_CHECKING:
    from consysid.configs.collect_presets.go2_gait_conditioned import (
        Go2GaitConditionedSysIDCollectConfig,
    )
    from consysid.configs.sysid_term_config import SysIDConfig


@dataclass(frozen=True)
class PhysicsSet:
    """Named bundle of the six scalar SysID parameters + inertial XML.

    Frozen so accidentally mutating a shared instance can't drift one
    site away from the rest of the run. Use :func:`dataclasses.replace`
    to derive variants without touching the original.

    Attributes:
        kp_hip: Stiffness shared by hip + thigh joints.
        kd_hip: Damping shared by hip + thigh joints.
        kp_knee: Stiffness for calf joints.
        kd_knee: Damping for calf joints.
        joint_frictionloss: Joint Coulomb friction shared across the
            12 leg DOFs (named ``joint_frictionloss`` here to match
            MJCF / actuator-config nomenclature; the SysID term is
            ``joint_friction``).
        foot_friction: Foot-ground slide friction (``mu``).
        mjcf_path: Optional repo-relative path to an MJCF whose
            ``<inertial>`` tags carry this set's mass / CoM / inertia.
            ``None`` means "consumer keeps its default trunk MJCF".
    """

    kp_hip: float
    kd_hip: float
    kp_knee: float
    kd_knee: float
    joint_frictionloss: float
    foot_friction: float
    mjcf_path: str | None = None

    # ── Per-leg PD (Newton sim2sim only) ──
    # When set, ``apply_perleg_pd_to_robot_overrides`` writes these
    # ``{"FL": {"hip": ..., "knee": ...}, ...}`` dicts onto the
    # ``Go2Config.kp_per_leg_override`` / ``kd_per_leg_override`` slots
    # so the four legs ship distinct PD gains. The scalar ``kp_hip`` /
    # ``kp_knee`` etc. above stay populated with the mean (or any
    # representative value) so consumers that ignore ``kp_per_leg``
    # still see a sane scalar fallback.
    kp_per_leg: dict[str, dict[str, float]] | None = None
    kd_per_leg: dict[str, dict[str, float]] | None = None

    # ── Per-DOF / per-foot friction (Newton sim2sim only) ──
    # When set, identification recovers heterogeneous friction instead of
    # the two scalars above. ``joint_friction_per_joint`` is a 12-vector in
    # ``act_manager.actuated_joint_names`` order (FL_hip, FL_thigh, FL_calf,
    # FR_hip, ...); ``foot_friction_per_foot`` is a 4-vector (FL, FR, RL, RR
    # foot). The scalars ``joint_frictionloss`` / ``foot_friction`` stay
    # populated with the mean so scalar-friction consumers (Mujoco/Genesis
    # branches, 36D configs) keep a sane fallback. ``None`` → scalar path.
    joint_friction_per_joint: list[float] | None = None
    foot_friction_per_foot: list[float] | None = None


# ===========================================================================
# Named sets - declare a new GT_V2 per experiment.
# ===========================================================================

# Inertial XML for v1 (mass / CoM / inertia edits encoded in the
# file; PD / friction values come from the PhysicsSet).
_MJCF_V1: str = "./SysID/assets/robots/unitree_go2/xmls/go2_mass_modified.xml"

# Experiment v1 -- the "real world" SysID is trying to identify, baked
# into every collect environment, used as the ground_truth label for
# every identification term, and pinned by the mujoco-ceiling training
# / mujoco-side evaluation runs.

GT_V1: PhysicsSet = PhysicsSet(
    kp_hip=10.0,
    kd_hip=0.6,
    kp_knee=50.0,
    kd_knee=2.0,
    joint_frictionloss=1.5,
    foot_friction=0.3,
    mjcf_path=_MJCF_V1,
)


# Experiment v2 -- per-leg heterogeneous PD on top of v1's friction /
# inertial shift. The four legs ship distinct stiffness/damping so a
# correct SysID fit recovers four pairs of (kp_hip, kp_knee, kd_hip,
# kd_knee) rather than two scalars. Each leg's PD becomes informative
# only during its own stance phase, so the identifiable parameter
# subset is phase-dependent — the regime where decoupling-coverage
# exploration separates the axes that a uniform trace objective would
# aggregate together.
#
# The scalar ``kp_hip`` / ``kp_knee`` / ``kd_hip`` / ``kd_knee`` here
# carry the four-leg mean so consumers that ignore ``kp_per_leg`` (the
# Mujoco / Genesis branches, sanity scripts that haven't migrated) still
# see a representative scalar GT instead of garbage.
_KP_PER_LEG_V2: dict[str, dict[str, float]] = {
    "FL": {"hip": 12.0, "knee": 45.0},
    "FR": {"hip": 15.0, "knee": 55.0},
    "RL": {"hip": 18.0, "knee": 60.0},
    "RR": {"hip": 22.0, "knee": 70.0},
}
_KD_PER_LEG_V2: dict[str, dict[str, float]] = {
    "FL": {"hip": 0.7, "knee": 2.5},
    "FR": {"hip": 0.5, "knee": 2.0},
    "RL": {"hip": 0.8, "knee": 3.0},
    "RR": {"hip": 0.4, "knee": 3.5},
}

_KP_HIP_MEAN_V2 = sum(v["hip"] for v in _KP_PER_LEG_V2.values()) / 4
_KP_KNEE_MEAN_V2 = sum(v["knee"] for v in _KP_PER_LEG_V2.values()) / 4
_KD_HIP_MEAN_V2 = sum(v["hip"] for v in _KD_PER_LEG_V2.values()) / 4
_KD_KNEE_MEAN_V2 = sum(v["knee"] for v in _KD_PER_LEG_V2.values()) / 4

GT_V2: PhysicsSet = PhysicsSet(
    kp_hip=_KP_HIP_MEAN_V2,
    kd_hip=_KD_HIP_MEAN_V2,
    kp_knee=_KP_KNEE_MEAN_V2,
    kd_knee=_KD_KNEE_MEAN_V2,
    joint_frictionloss=1.5,
    foot_friction=0.2,
    mjcf_path=_MJCF_V1,
    kp_per_leg=_KP_PER_LEG_V2,
    kd_per_leg=_KD_PER_LEG_V2,
)


# Experiment v3 -- v2's per-leg PD + heterogeneous per-DOF joint friction
# and per-foot foot friction on top. Raises the identified space from 36D
# to 50D (base 10 + kp 12 + kd 12 + joint_friction 12 + foot_friction 4)
# and adds strongly coupled, harder-to-identify friction axes -- the
# regime where decoupling-coverage should separate from SPI-Active's
# uniform -tr(F).
_JOINT_FRICTION_PER_JOINT_V3: list[float] = [
    1.0, 1.3, 1.6,   # FL hip, thigh, calf
    1.2, 1.5, 1.8,   # FR
    0.8, 1.1, 1.4,   # RL
    1.4, 1.7, 2.0,   # RR
]
_FOOT_FRICTION_PER_FOOT_V3: list[float] = [0.15, 0.25, 0.35, 0.20]  # FL FR RL RR

GT_V3: PhysicsSet = PhysicsSet(
    kp_hip=_KP_HIP_MEAN_V2,
    kd_hip=_KD_HIP_MEAN_V2,
    kp_knee=_KP_KNEE_MEAN_V2,
    kd_knee=_KD_KNEE_MEAN_V2,
    joint_frictionloss=sum(_JOINT_FRICTION_PER_JOINT_V3) / 12,   # mean fallback
    foot_friction=sum(_FOOT_FRICTION_PER_FOOT_V3) / 4,          # mean fallback
    mjcf_path=_MJCF_V1,
    kp_per_leg=_KP_PER_LEG_V2,
    kd_per_leg=_KD_PER_LEG_V2,
    joint_friction_per_joint=_JOINT_FRICTION_PER_JOINT_V3,
    foot_friction_per_foot=_FOOT_FRICTION_PER_FOOT_V3,
)


# ===========================================================================
# Active selection - change THIS line to switch experiment.
# ===========================================================================

CURRENT_GT: PhysicsSet = GT_V3


# ===========================================================================
# Adapter helpers.
# ===========================================================================


def apply_to_robot_overrides(robot_cfg, p: PhysicsSet) -> None:
    """Write ``p`` into the ``Go2Config.*_override`` slots in place.

    Consumed by JaxRLWorld's Newton / MuJoCo Go2 builders at
    scene-build time (PD) and at every reset (frictions). Setting
    ``robot_cfg.mjcf_path`` forces the MJCF branch of the builder
    (the URDF branch is currently commented out, so ``urdf_path`` is
    blanked to avoid an ambiguous double-source).
    """
    robot_cfg.kp_hip_override = p.kp_hip
    robot_cfg.kd_hip_override = p.kd_hip
    robot_cfg.kp_knee_override = p.kp_knee
    robot_cfg.kd_knee_override = p.kd_knee
    robot_cfg.joint_frictionloss_override = p.joint_frictionloss
    robot_cfg.foot_friction_override = p.foot_friction
    if p.mjcf_path is not None:
        robot_cfg.urdf_path = ""
        robot_cfg.mjcf_path = p.mjcf_path


def apply_perleg_pd_to_robot_overrides(robot_cfg, p: PhysicsSet) -> None:
    """Newton-only variant: write per-leg PD dicts to ``Go2Config``.

    Routes ``p.kp_per_leg`` / ``p.kd_per_leg`` into the per-leg slots
    consumed by ``_newton_builders``, which build a joint-pattern →
    value dict and feed it to ``IdealPDActuatorCfg.stiffness`` /
    ``damping``. Friction + MJCF still use the same scalar slots as
    ``apply_to_robot_overrides`` since those have no per-leg variant.

    Raises:
        ValueError: ``p.kp_per_leg`` or ``p.kd_per_leg`` is None
            (would silently fall through to the scalar PD path).
    """
    if p.kp_per_leg is None or p.kd_per_leg is None:
        raise ValueError(
            "apply_perleg_pd_to_robot_overrides requires both "
            "p.kp_per_leg and p.kd_per_leg to be set; use "
            "apply_to_robot_overrides for the scalar PD case.",
        )
    robot_cfg.kp_per_leg_override = p.kp_per_leg
    robot_cfg.kd_per_leg_override = p.kd_per_leg
    robot_cfg.joint_frictionloss_override = p.joint_frictionloss
    robot_cfg.foot_friction_override = p.foot_friction
    if p.mjcf_path is not None:
        robot_cfg.urdf_path = ""
        robot_cfg.mjcf_path = p.mjcf_path


def apply_to_collect_preset(
    preset: "Go2GaitConditionedSysIDCollectConfig",
    p: PhysicsSet,
) -> None:
    """Write ``p`` into the collect-side preset fields in place.

    Mirrors :func:`apply_to_robot_overrides` for
    :class:`Go2GaitConditionedSysIDCollectConfig`, the preset shared
    by :class:`TrajectoryCollector` (collection) and
    :class:`PolicyEvaluator`-based eval scripts.
    """
    preset.kp_hip = p.kp_hip
    preset.kd_hip = p.kd_hip
    preset.kp_knee = p.kp_knee
    preset.kd_knee = p.kd_knee
    preset.joint_frictionloss = p.joint_frictionloss
    preset.foot_friction = p.foot_friction
    if p.mjcf_path is not None:
        preset.xml_path = p.mjcf_path


def apply_kp_kdknee_footfric_to_robot_overrides(
    robot_cfg, p: PhysicsSet,
) -> None:
    """Write ``kp_hip`` / ``kp_knee`` / ``kd_knee`` / ``foot_friction``
    from ``p`` onto ``Go2Config.*_override`` slots; leave every other
    override slot at ``None`` (= MJCF nominal default).

    Newton-side companion of
    :func:`apply_kp_kdknee_footfric_to_collect_preset` -- the Mujoco
    collect preset mutates actuator stiffness/damping at MJCF spec-
    build time, whereas Newton consumes the same four scalars from
    the ``Go2Config`` override slots and threads them into its own
    actuator config + foot geom mu at ``preset.build()`` time.

    Disabled override slots stay ``None`` so the Newton sim falls
    back to the MJCF default for those axes -- this is what keeps
    the the identification sim env's non-target axes in lockstep with the
    collect env's non-target axes (no cross-axis bias).
    """
    robot_cfg.kp_hip_override = p.kp_hip
    robot_cfg.kp_knee_override = p.kp_knee
    robot_cfg.kd_knee_override = p.kd_knee
    robot_cfg.foot_friction_override = p.foot_friction


def apply_kp_kdknee_footfric_to_collect_preset(
    preset: "Go2GaitConditionedSysIDCollectConfig",
    p: PhysicsSet,
) -> None:
    """Write ``kp_hip`` / ``kp_knee`` / ``kd_knee`` / ``foot_friction``
    from ``p`` onto ``preset``; leave everything else at the preset's
    nominal default.

    Targeted 4D set: each retained axis has a phase-dependent dynamics
    signature so the identifiable subset varies across the trajectory --
        * ``kp_hip``        -- stance posture / body support
        * ``kp_knee``       -- swing retract + stance push-off
        * ``kd_knee``       -- swing high-velocity retract (dominant in
                               flight phase)
        * ``foot_friction`` -- stance only (contact slip)

    ``kd_hip`` / ``joint_frictionloss`` / ``xml_path`` are left at
    nominal so the collect env's non-target axes match what the
    identification sim will start from, isolating the four-axis shift.
    """
    preset.kp_hip = p.kp_hip
    preset.kp_knee = p.kp_knee
    preset.kd_knee = p.kd_knee
    preset.foot_friction = p.foot_friction


def apply_gt_to_sysid_config(sysid_cfg: "SysIDConfig", gt: PhysicsSet) -> None:
    """Pin identification ``SysIDTermConfig.ground_truth`` slots from ``gt``.

    Only the six PD + friction terms are populated; ``base_inertial``
    carries its ground truth through the encoded log-Cholesky phi
    vector, which each preset script handles separately via its own
    ``PHI_GT`` constant.

    The SysID term name for joint Coulomb friction is
    ``joint_friction`` while the PhysicsSet field uses MJCF-style
    ``joint_frictionloss``; this helper bridges the two.

    Terms set to ``None`` on ``sysid_cfg`` (the SysIDConfig "disable"
    pattern) are skipped -- callers can run single-term ablations
    (e.g. "identify ``kp_knee`` only, fix the rest at GT") with the
    same helper as the full identification runs.
    """
    pairs = (
        ("kp_hip", gt.kp_hip),
        ("kd_hip", gt.kd_hip),
        ("kp_knee", gt.kp_knee),
        ("kd_knee", gt.kd_knee),
    )
    # ── Per-joint kp_all / kd_all → 12D ground_truth array ──
    # The 36D ``Go2NewtonFullMixedConfig`` (and any other per-joint PD
    # config) doesn't expose the scalar ``kp_hip`` / ``kd_hip`` names
    # in the loop below; instead we map ``gt.kp_per_leg`` (when set) or
    # the four scalars (fallback) onto the 12D array slot. The joint
    # order matches ``act_manager.actuated_joint_names``:
    #   FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
    #   RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
    def _expand_to_12(per_leg: dict | None, scalar_hip: float, scalar_knee: float):
        if per_leg is not None:
            return np.array([
                v for leg in ("FL", "FR", "RL", "RR") for v in (
                    per_leg[leg]["hip"],
                    per_leg[leg]["hip"],
                    per_leg[leg]["knee"],
                )
            ], dtype=np.float32)
        return np.array([scalar_hip, scalar_hip, scalar_knee] * 4, dtype=np.float32)

    for arr_name, per_leg, scalar_hip, scalar_knee in (
        ("kp_all", gt.kp_per_leg, gt.kp_hip, gt.kp_knee),
        ("kd_all", gt.kd_per_leg, gt.kd_hip, gt.kd_knee),
    ):
        if not hasattr(sysid_cfg, arr_name):
            continue
        term = getattr(sysid_cfg, arr_name)
        if term is None:
            continue
        term.ground_truth = _expand_to_12(per_leg, scalar_hip, scalar_knee)

    # ── joint_friction / foot_friction: scalar OR per-DOF / per-foot ──
    # The term's own ``dim`` decides: dim==1 keeps the scalar GT (36D
    # config / Mujoco / Genesis branches), dim>1 expands to the per-list
    # vector (preferred) or broadcasts the scalar mean (fallback).
    def _friction_gt(term, per_list, scalar):
        if int(term.dim) <= 1:
            return scalar
        if per_list is not None:
            arr = np.asarray(per_list, dtype=np.float32)
            if arr.shape != (int(term.dim),):
                raise ValueError(
                    f"friction per-list len {arr.shape} != term.dim {term.dim}",
                )
            return arr
        return np.full(int(term.dim), float(scalar), dtype=np.float32)

    for fname, per_list, scalar in (
        ("joint_friction", gt.joint_friction_per_joint, gt.joint_frictionloss),
        ("foot_friction", gt.foot_friction_per_foot, gt.foot_friction),
    ):
        if not hasattr(sysid_cfg, fname):
            continue
        term = getattr(sysid_cfg, fname)
        if term is None:
            continue
        term.ground_truth = _friction_gt(term, per_list, scalar)

    for name, val in pairs:
        # ``hasattr`` first so per-joint configs (e.g.
        # ``Go2NewtonFullMixedConfig`` which exposes ``kp_all``/``kd_all``
        # instead of the scalar ``kp_hip``/``kd_hip``/``kp_knee``/
        # ``kd_knee``) don't trip an ``AttributeError`` here; their
        # ground_truth labelling has to be set explicitly per-array by
        # the caller, not via this scalar bridge.
        if not hasattr(sysid_cfg, name):
            continue
        term = getattr(sysid_cfg, name)
        if term is not None:
            term.ground_truth = val


def load_fitted_from_npz(path: str) -> PhysicsSet:
    """Rebuild a :class:`PhysicsSet` from a identification
    ``sysid_result.npz``.

    Pulls each scalar term from the saved ``theta_best`` vector via
    the accompanying ``term_names`` / ``term_dims`` layout.
    ``base_inertial`` is intentionally skipped -- its 10D log-Cholesky
    phi has to be decoded and substituted into an MJCF by hand, so
    the returned set has ``mjcf_path=None`` and the caller is expected
    to set it explicitly (typically to the human-edited XML produced
    from the same npz).

    Raises:
        ValueError: ``term_names`` / ``term_dims`` metadata is absent
            (legacy npz) or any required PD / friction term is not
            1-D in the saved layout.
        KeyError: a required PD / friction term is missing from the
            saved ``term_names``.
    """
    data = np.load(path, allow_pickle=True)
    theta = np.asarray(data["theta_best"]).astype(np.float64)

    if "term_names" not in data.files or "term_dims" not in data.files:
        raise ValueError(
            f"sysid_result at {path} is missing term_names / term_dims "
            f"layout metadata - cannot map theta_best to named scalars."
        )
    names = [str(n) for n in data["term_names"]]
    dims = [int(d) for d in data["term_dims"]]

    slices: dict[str, tuple[int, int]] = {}
    off = 0
    for n, d in zip(names, dims):
        slices[n] = (off, off + d)
        off += d

    def scalar(name: str) -> float:
        if name not in slices:
            raise KeyError(
                f"sysid_result at {path} has no term named {name!r}; "
                f"available: {list(slices)}"
            )
        lo, hi = slices[name]
        if hi - lo != 1:
            raise ValueError(
                f"term {name!r} in {path} is not scalar (dim={hi - lo}); "
                f"cannot reduce to a PhysicsSet field."
            )
        return float(theta[lo])

    return PhysicsSet(
        kp_hip=scalar("kp_hip"),
        kd_hip=scalar("kd_hip"),
        kp_knee=scalar("kp_knee"),
        kd_knee=scalar("kd_knee"),
        joint_frictionloss=scalar("joint_friction"),
        foot_friction=scalar("foot_friction"),
        mjcf_path=None,
    )


def load_fitted_perleg_from_npz(path: str) -> PhysicsSet:
    """Rebuild a per-leg :class:`PhysicsSet` from a per-joint
    ``sysid_result.npz`` (the 36D ``Go2NewtonFullMixedConfig`` output).

    The per-joint PD lives in the 12D ``kp_all`` / ``kd_all`` terms in
    ``act_manager.actuated_joint_names`` order::

        [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
         RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]

    Each leg's ``hip`` value is taken from its hip slot (index ``leg*3``)
    and ``knee`` from its calf slot (index ``leg*3 + 2``). ``thigh`` (the
    middle slot) shares the hip group in the actuator wiring; this loader
    keeps the hip value as the group representative — matching how
    :func:`apply_perleg_pd_to_robot_overrides` re-expands ``hip`` over
    ``(hip|thigh)``. ``joint_friction`` / ``foot_friction`` are scalar,
    same as :func:`load_fitted_from_npz`. ``base_inertial`` (log-Cholesky
    phi) is skipped — the caller sets ``mjcf_path`` to the human-edited
    XML produced from the same npz.

    The four-scalar ``kp_hip`` / ``kp_knee`` / ``kd_hip`` / ``kd_knee``
    fields are filled with the four-leg mean so consumers that ignore
    ``kp_per_leg`` still see a sane scalar fallback.

    Raises:
        ValueError: ``term_names`` / ``term_dims`` metadata is absent, or
            ``kp_all`` / ``kd_all`` are not 12-D in the saved layout.
        KeyError: ``kp_all`` / ``kd_all`` / ``joint_friction`` /
            ``foot_friction`` missing from the saved ``term_names``.
    """
    data = np.load(path, allow_pickle=True)
    theta = np.asarray(data["theta_best"]).astype(np.float64)

    if "term_names" not in data.files or "term_dims" not in data.files:
        raise ValueError(
            f"sysid_result at {path} is missing term_names / term_dims "
            f"layout metadata - cannot map theta_best to named terms."
        )
    names = [str(n) for n in data["term_names"]]
    dims = [int(d) for d in data["term_dims"]]

    slices: dict[str, tuple[int, int]] = {}
    off = 0
    for n, d in zip(names, dims):
        slices[n] = (off, off + d)
        off += d

    def _term(name: str, expected_dim: int) -> np.ndarray:
        if name not in slices:
            raise KeyError(
                f"sysid_result at {path} has no term named {name!r}; "
                f"available: {list(slices)}"
            )
        lo, hi = slices[name]
        if hi - lo != expected_dim:
            raise ValueError(
                f"term {name!r} in {path} has dim={hi - lo}, expected "
                f"{expected_dim}."
            )
        return theta[lo:hi]

    kp_all = _term("kp_all", 12)
    kd_all = _term("kd_all", 12)

    legs = ("FL", "FR", "RL", "RR")

    def _per_leg(arr: np.ndarray) -> dict[str, dict[str, float]]:
        return {
            leg: {"hip": float(arr[i * 3 + 0]), "knee": float(arr[i * 3 + 2])}
            for i, leg in enumerate(legs)
        }

    kp_per_leg = _per_leg(kp_all)
    kd_per_leg = _per_leg(kd_all)

    def _mean(per_leg: dict, key: str) -> float:
        return sum(v[key] for v in per_leg.values()) / len(per_leg)

    return PhysicsSet(
        kp_hip=_mean(kp_per_leg, "hip"),
        kd_hip=_mean(kd_per_leg, "hip"),
        kp_knee=_mean(kp_per_leg, "knee"),
        kd_knee=_mean(kd_per_leg, "knee"),
        joint_frictionloss=float(_term("joint_friction", 1)[0]),
        foot_friction=float(_term("foot_friction", 1)[0]),
        mjcf_path=None,
        kp_per_leg=kp_per_leg,
        kd_per_leg=kd_per_leg,
    )


def load_fitted_perfriction_from_npz(
    path: str,
    mjcf_path: str | None = None,
) -> PhysicsSet:
    """Rebuild a heterogeneous-friction :class:`PhysicsSet` from a 50D
    ``sysid_result.npz`` (the ``Go2NewtonFullPerFrictionConfig`` output).

    Layout: ``base_inertial`` 10 + ``kp_all`` 12 + ``kd_all`` 12 +
    ``joint_friction`` 12 + ``foot_friction`` 4. The per-joint PD maps to
    per-leg slots with the same convention as
    :func:`load_fitted_perleg_from_npz` (hip slot represents the
    ``(hip|thigh)`` actuator group, calf slot → ``knee``); the frictions
    land in the per-dim fields (``joint_friction_per_joint`` /
    ``foot_friction_per_foot``) consumed by
    :func:`apply_perdim_friction_to_robot_overrides`, with their means as
    the scalar fallbacks.

    ``base_inertial`` (log-Cholesky phi) is NOT handled here — pass the
    MJCF produced by :func:`write_fitted_base_xml` (from the same npz)
    via ``mjcf_path`` so the identified trunk inertial travels the XML
    route like every other inertial override.

    Raises:
        ValueError / KeyError: layout metadata absent or a term missing /
            mis-sized (e.g. a 36D scalar-friction result was passed —
            use :func:`load_fitted_perleg_from_npz` for those).
    """
    data = np.load(path, allow_pickle=True)
    theta = np.asarray(data["theta_best"]).astype(np.float64)

    if "term_names" not in data.files or "term_dims" not in data.files:
        raise ValueError(
            f"sysid_result at {path} is missing term_names / term_dims "
            f"layout metadata - cannot map theta_best to named terms."
        )
    names = [str(n) for n in data["term_names"]]
    dims = [int(d) for d in data["term_dims"]]

    slices: dict[str, tuple[int, int]] = {}
    off = 0
    for n, d in zip(names, dims):
        slices[n] = (off, off + d)
        off += d

    def _term(name: str, expected_dim: int) -> np.ndarray:
        if name not in slices:
            raise KeyError(
                f"sysid_result at {path} has no term named {name!r}; "
                f"available: {list(slices)}"
            )
        lo, hi = slices[name]
        if hi - lo != expected_dim:
            raise ValueError(
                f"term {name!r} in {path} has dim={hi - lo}, expected "
                f"{expected_dim}."
            )
        return theta[lo:hi]

    kp_all = _term("kp_all", 12)
    kd_all = _term("kd_all", 12)
    jf = _term("joint_friction", 12)
    ff = _term("foot_friction", 4)

    legs = ("FL", "FR", "RL", "RR")

    def _per_leg(arr: np.ndarray) -> dict[str, dict[str, float]]:
        return {
            leg: {"hip": float(arr[i * 3 + 0]), "knee": float(arr[i * 3 + 2])}
            for i, leg in enumerate(legs)
        }

    kp_per_leg = _per_leg(kp_all)
    kd_per_leg = _per_leg(kd_all)

    def _mean(per_leg: dict, key: str) -> float:
        return sum(v[key] for v in per_leg.values()) / len(per_leg)

    return PhysicsSet(
        kp_hip=_mean(kp_per_leg, "hip"),
        kd_hip=_mean(kd_per_leg, "hip"),
        kp_knee=_mean(kp_per_leg, "knee"),
        kd_knee=_mean(kd_per_leg, "knee"),
        joint_frictionloss=float(jf.mean()),
        foot_friction=float(ff.mean()),
        mjcf_path=mjcf_path,
        kp_per_leg=kp_per_leg,
        kd_per_leg=kd_per_leg,
        joint_friction_per_joint=[float(v) for v in jf],
        foot_friction_per_foot=[float(v) for v in ff],
    )


def write_fitted_base_xml(
    sysid_npz_path: str,
    out_path: str,
    template_xml: str = "./SysID/assets/robots/unitree_go2/xmls/go2.xml",
    body_name: str = "trunk",
) -> str:
    """Bake an identified ``base_inertial`` into an MJCF's base body.

    Decodes the log-Cholesky ``base_inertial`` phi (10D) from a
    ``sysid_result.npz`` to physical ``(mass, com, inertia-about-com)``
    and substitutes ``body_name``'s ``<inertial>`` in ``template_xml``
    (the NOMINAL robot xml — Go2 ``trunk``/go2.xml by default, G1
    callers pass ``body_name="pelvis"`` + g1.xml): ``pos`` ← com,
    ``mass`` ← mass, and the tensor as ``fullinertia`` (MuJoCo: about
    the com, in the body frame — the same convention
    ``decode_phi_batched`` returns). Any ``diaginertia`` / ``quat`` /
    ``iquat`` on the tag is removed so ``fullinertia`` is authoritative.

    Returns ``out_path`` (parent dirs created), ready to hand to
    :func:`load_fitted_perfriction_from_npz` as ``mjcf_path``.

    Raises:
        KeyError / ValueError: ``base_inertial`` missing or mis-sized.
        ValueError: trunk ``<inertial>`` not found in the template.
    """
    data = np.load(sysid_npz_path, allow_pickle=True)
    theta = np.asarray(data["theta_best"]).astype(np.float64)
    names = [str(n) for n in data["term_names"]]
    dims = [int(d) for d in data["term_dims"]]
    off = 0
    phi = None
    for n, d in zip(names, dims):
        if n == "base_inertial":
            if d != 10:
                raise ValueError(
                    f"base_inertial in {sysid_npz_path} has dim={d}, expected 10."
                )
            phi = theta[off:off + d]
            break
        off += d
    if phi is None:
        raise KeyError(
            f"sysid_result at {sysid_npz_path} has no base_inertial term; "
            f"available: {names}"
        )

    phys = decode_phi_to_physical_dict(phi)
    mass = float(phys["mass"])
    com = np.asarray(phys["com"], dtype=np.float64)
    inertia = np.asarray(phys["inertia"], dtype=np.float64)

    tree = ET.parse(template_xml)
    root = tree.getroot()

    # The template references meshes/textures RELATIVE to its own
    # directory (e.g. meshdir="assets"). The fitted XML lands elsewhere
    # (outputs/...), so absolutize the asset dirs against the template's
    # location — otherwise the loader resolves them against out_path and
    # crashes on a missing .obj.
    template_dir = Path(template_xml).resolve().parent
    compiler_el = root.find("compiler")
    if compiler_el is None:
        compiler_el = ET.SubElement(root, "compiler")
    for attr, default in (("meshdir", "assets"), ("texturedir", None)):
        cur = compiler_el.get(attr, default)
        if cur is not None and not Path(cur).is_absolute():
            compiler_el.set(attr, str(template_dir / cur))

    inertial_el = None
    for body in root.iter("body"):
        if body.get("name") == body_name:
            inertial_el = body.find("inertial")
            break
    if inertial_el is None:
        raise ValueError(f"{body_name} <inertial> not found in {template_xml}")

    for attr in ("diaginertia", "quat", "iquat", "fullinertia"):
        if attr in inertial_el.attrib:
            del inertial_el.attrib[attr]
    inertial_el.set("pos", " ".join(f"{v:.8g}" for v in com))
    inertial_el.set("mass", f"{mass:.8g}")
    # MuJoCo fullinertia order: (Ixx, Iyy, Izz, Ixy, Ixz, Iyz).
    full = (
        inertia[0, 0], inertia[1, 1], inertia[2, 2],
        inertia[0, 1], inertia[0, 2], inertia[1, 2],
    )
    inertial_el.set("fullinertia", " ".join(f"{v:.10g}" for v in full))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="unicode", xml_declaration=False)
    print(
        f"[write_fitted_base_xml] {body_name}: {sysid_npz_path} → {out}\n"
        f"  mass={mass:.4f}  com={com.round(5).tolist()}"
    )
    return str(out)


def apply_perdim_friction_to_robot_overrides(robot_cfg, p: PhysicsSet) -> None:
    """Newton-only: write heterogeneous per-DOF / per-foot friction.

    Routes p.joint_friction_per_joint (12) and p.foot_friction_per_foot
    (4) into the per-dim override slots consumed by _newton_builders,
    which install set_joint_friction_per_dim / set_foot_friction_per_foot
    reset_dr events so the heterogeneous ground truth is re-applied every
    reset. Clears the scalar friction overrides. Call AFTER
    apply_perleg_pd_to_robot_overrides.
    """
    if p.joint_friction_per_joint is None or p.foot_friction_per_foot is None:
        raise ValueError(
            "apply_perdim_friction_to_robot_overrides requires both "
            "p.joint_friction_per_joint and p.foot_friction_per_foot; use "
            "apply_perleg_pd_to_robot_overrides for the scalar friction "
            "case.",
        )
    robot_cfg.joint_friction_per_joint_override = list(p.joint_friction_per_joint)
    robot_cfg.foot_friction_per_foot_override = list(p.foot_friction_per_foot)
    robot_cfg.joint_frictionloss_override = None
    robot_cfg.foot_friction_override = None


# ===========================================================================
# G1 (29-DOF) — per-joint PD + base inertial only (NO friction).
# ===========================================================================
# G1 is a separate robot with its own parameter bundle: kp/kd are 29-vectors
# in g1.xml actuated-joint order (``_G1_JOINT_NAMES``), and the base inertial
# GT travels through a mass-modified pelvis MJCF (created separately). Go2's
# PhysicsSet (hip/knee scalars, 4-leg/4-foot structure) does not apply here,
# so G1 gets its own frozen dataclass + adapters.


@dataclass(frozen=True)
class G1PhysicsSet:
    """G1 SysID parameter bundle: per-DOF PD (29 + 29) + inertial MJCF.

    Attributes:
        kp_per_dof: 29 per-joint stiffness values in ``_G1_JOINT_NAMES``
            order.
        kd_per_dof: 29 per-joint damping values, same order.
        mjcf_path: Repo-relative path to the mass-modified pelvis MJCF
            whose ``<inertial>`` carries the base-inertial ground truth.
            ``None`` keeps the nominal g1.xml.
    """

    kp_per_dof: list[float]
    kd_per_dof: list[float]
    mjcf_path: str | None = None


# Mass-modified inertial XML for G1 (created separately; the PD GT lives in
# the vectors below).
_G1_MJCF: str = "./SysID/assets/robots/g1/xmls/g1_mass_modified.xml"

# Heterogeneous PD ground truth: the nominal motor-class gains scaled
# per joint so the PD axes have a non-trivial identification target — the
# role GT_V3's per-joint friction plays for Go2. Shifts kept modest
# (kp ~±12%, kd ~±20%) so the GT robot stays close to the trained
# dynamics; a larger kp shift (the earlier ±35%) moved the biped too far.
# All scaled values stay inside the kp_all/kd_all config bounds. The
# inertial shift comes from the mass-modified MJCF. Adjust to taste.
_G1_KP_GT_SCALE = [
    1.12, 0.90, 1.08, 0.88, 1.10, 0.93,        # L leg
    1.05, 0.91, 1.10, 0.89, 1.07, 1.11,        # R leg
    0.90, 1.10, 0.88,                          # waist
    1.12, 0.88, 1.09, 0.94, 1.10, 0.90, 1.06,  # L arm
    0.91, 1.08, 0.89, 1.11, 0.88, 1.07, 1.00,  # R arm
]
_G1_KD_GT_SCALE = [
    0.85, 1.18, 0.90, 1.20, 0.82, 1.12,        # L leg
    1.15, 0.88, 1.10, 0.83, 1.18, 0.85,        # R leg
    1.18, 0.84, 1.12,                          # waist
    0.80, 1.20, 0.88, 1.15, 0.83, 1.18, 0.90,  # L arm
    1.12, 0.85, 1.20, 0.82, 1.15, 0.88, 1.00,  # R arm
]

GT_G1: G1PhysicsSet = G1PhysicsSet(
    kp_per_dof=[n * s for n, s in zip(_G1_KP_NOMINAL, _G1_KP_GT_SCALE)],
    kd_per_dof=[n * s for n, s in zip(_G1_KD_NOMINAL, _G1_KD_GT_SCALE)],
    mjcf_path=_G1_MJCF,
)


def apply_to_g1_robot_overrides(robot_cfg, p: G1PhysicsSet) -> None:
    """Write G1 per-DOF PD + inertial MJCF onto the G1 robot config.

    Maps the 29-vectors onto ``kp_per_dof_override`` / ``kd_per_dof_
    override`` (a ``{joint_name: value}`` map the actuator consumes
    natively) and swaps the MJCF to the mass-modified inertial XML.
    Callers also set ``use_ideal_pd_actuator=True`` on the preset so the
    explicit-PD path is used. Friction is left untouched (not identified
    for G1).
    """
    n = len(_G1_JOINT_NAMES)
    if len(p.kp_per_dof) != n or len(p.kd_per_dof) != n:
        raise ValueError(
            f"G1 PD vectors must be length {n}; got kp={len(p.kp_per_dof)}, "
            f"kd={len(p.kd_per_dof)}.",
        )
    robot_cfg.kp_per_dof_override = {
        name: float(v) for name, v in zip(_G1_JOINT_NAMES, p.kp_per_dof)
    }
    robot_cfg.kd_per_dof_override = {
        name: float(v) for name, v in zip(_G1_JOINT_NAMES, p.kd_per_dof)
    }
    if p.mjcf_path is not None:
        robot_cfg.urdf_path = ""
        robot_cfg.mjcf_path = p.mjcf_path


def apply_gt_to_g1_sysid_config(sysid_cfg: "SysIDConfig", p: G1PhysicsSet) -> None:
    """Pin G1 identification ``kp_all`` / ``kd_all`` ground_truth from ``p``.

    ``base_inertial`` carries its GT through the encoded log-Cholesky phi
    (resolved from the mass-modified MJCF by the term's ``gt_encoder``),
    same as the Go2 path, so only the PD arrays are set here. Mirrors
    :func:`apply_gt_to_sysid_config`'s kp_all/kd_all branch.
    """
    for arr_name, vec in (("kp_all", p.kp_per_dof), ("kd_all", p.kd_per_dof)):
        if not hasattr(sysid_cfg, arr_name):
            continue
        term = getattr(sysid_cfg, arr_name)
        if term is None:
            continue
        term.ground_truth = np.asarray(vec, dtype=np.float32)


def load_fitted_g1_from_npz(
    path: str,
    mjcf_path: str | None = None,
) -> G1PhysicsSet:
    """Rebuild a :class:`G1PhysicsSet` from a 68D G1 ``sysid_result.npz``
    (the ``G1NewtonInertialPDConfig`` output: base_inertial 10 + kp_all 29
    + kd_all 29).

    The 29-vectors land in ``kp_per_dof`` / ``kd_per_dof`` in
    ``_G1_JOINT_NAMES`` order — the layout
    :func:`apply_to_g1_robot_overrides` maps onto the per-joint override
    dicts. ``base_inertial`` (log-Cholesky phi) is NOT handled here —
    pass the MJCF produced by :func:`write_fitted_base_xml` (same npz,
    ``body_name="pelvis"``, g1.xml template) via ``mjcf_path``.

    Raises:
        ValueError / KeyError: layout metadata absent, or ``kp_all`` /
            ``kd_all`` missing / not 29-D.
    """
    data = np.load(path, allow_pickle=True)
    theta = np.asarray(data["theta_best"]).astype(np.float64)

    if "term_names" not in data.files or "term_dims" not in data.files:
        raise ValueError(
            f"sysid_result at {path} is missing term_names / term_dims "
            f"layout metadata - cannot map theta_best to named terms."
        )
    names = [str(n) for n in data["term_names"]]
    dims = [int(d) for d in data["term_dims"]]

    slices: dict[str, tuple[int, int]] = {}
    off = 0
    for n, d in zip(names, dims):
        slices[n] = (off, off + d)
        off += d

    def _term(name: str, expected_dim: int) -> np.ndarray:
        if name not in slices:
            raise KeyError(
                f"sysid_result at {path} has no term named {name!r}; "
                f"available: {list(slices)}"
            )
        lo, hi = slices[name]
        if hi - lo != expected_dim:
            raise ValueError(
                f"term {name!r} in {path} has dim={hi - lo}, expected "
                f"{expected_dim}."
            )
        return theta[lo:hi]

    n = len(_G1_JOINT_NAMES)
    kp = _term("kp_all", n)
    kd = _term("kd_all", n)

    return G1PhysicsSet(
        kp_per_dof=[float(v) for v in kp],
        kd_per_dof=[float(v) for v in kd],
        mjcf_path=mjcf_path,
    )
