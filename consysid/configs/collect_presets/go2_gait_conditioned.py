"""Go2 gait-conditioned SysID-collect preset.

Subclass of JaxRLWorld's :class:`Go2GaitConditionedMujocoConfig`
(the mujoco-side training preset for the gait-conditioned policy)
that bakes ground-truth SysID parameter values directly into the env
config. Inheriting from the **same** preset the policy was trained
against keeps the obs/command interface identical — the policy sees
the gait commands and dof_pos_nominal_difference / clock_inputs /
all_commands observation terms it expects, just with different
underlying physics (modified PD gains, joint friction, foot mu).

All overrides live as fields on this class so the collect script
declares them once and the preset handles propagation. No runtime
``customize_env`` patching required:

    * PD gains (kp/kd) and ``joint_frictionloss`` → mutated on the
      ``IdealPDActuatorCfg`` instances inside the built scene's
      ``ArticulationCfg.actuators``. The base preset's
      ``DelayedPDActuatorCfg`` is swapped to ``IdealPDActuatorCfg``
      via ``use_ideal_pd_actuator=True`` (honoured by all three sim
      builders), so SysID doesn't have to identify command delay as
      a nuisance parameter. mjlab's BuiltinMotorActuator forwards
      stiffness / damping / frictionloss to MuJoCo at spec-build
      time.
    * ``foot_friction`` → applied via :meth:`runtime_setup` after
      env.reset(). Lives outside the actuator config so it cannot
      ride the spec-build path; the runtime patch also installs the
      ground-geom max() workaround.

If you train a different variant (e.g. ``Go2FlatConfig`` directly,
or a future ``Go2ScaffoldedTDMPC2MujocoConfig``), create a parallel
collect preset that inherits from THAT variant — keep the inheritance
chain matched to the trained policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from jaxrlworld.rl.actuators import IdealPDActuatorCfg
from jaxrlworld.rl.configs.presets.go2.mujoco.gait_conditioned import (
    Go2GaitConditionedMujocoConfig,
)


@dataclass
class Go2GaitConditionedSysIDCollectConfig(Go2GaitConditionedMujocoConfig):
    """Go2 gait-conditioned mujoco preset with SysID parameter overrides.

    All override fields default to ``None``, meaning "preserve base
    preset value". Set the ones you want to deviate.
    """

    # ── Actuator class override ────────────────────────────────────────
    # Strip the trained policy's command-delay actuator at collect time
    # so SysID doesn't have to identify it as a nuisance parameter. The
    # base ``Go2FlatConfig`` honours this flag in all three sim builders
    # (mujoco / newton / genesis) — no further plumbing required.
    use_ideal_pd_actuator: bool = True

    # ── PD gain overrides (None → preserve STIFFNESS_HIP/KNEE etc.) ────
    # Defaults from jaxrlworld/rl/configs/robots/go2.py:
    #   STIFFNESS_HIP  ≈ 17.6918   DAMPING_HIP  ≈ 1.1263
    #   STIFFNESS_KNEE ≈ 65.2191   DAMPING_KNEE ≈ 4.1520
    kp_hip: float | None = None
    kd_hip: float | None = None
    kp_knee: float | None = None
    kd_knee: float | None = None

    # ── Joint Coulomb friction (N·m) ───────────────────────────────────
    # None → preserve ActuatorBaseCfg default (0.0). Applied to every
    # actuator group covering leg joints, so a single scalar suffices.
    joint_frictionloss: float | None = None

    # ── Foot-ground slide friction (mu) ────────────────────────────────
    # None → preserve XML default (= 0.4 from go2.xml's foot class).
    # Applied at runtime to mjwarp ``geom_friction[..., 0]`` for both
    # foot and ground geoms; see :meth:`runtime_setup`.
    foot_friction: float | None = None

    # ── Inertial XML override (mass / CoM / inertia) ───────────────────
    # None → use the base preset's auto-resolved Go2 XML. Set to a path
    # of an XML file with edited ``<inertial>`` tags to inject SysID-
    # identified mass / CoM / inertia values without touching any of
    # the other physics knobs. The override replaces the robot entity's
    # ``spec_fn`` at build() time so the rest of the entity config
    # (init_state, articulation, collisions) and any mesh assets that
    # the custom XML doesn't embed are preserved from the base preset.
    #
    # Usage symmetry: the collect script ``sysid_preset_overrides.py``
    # currently passes ``xml_path`` through ``TrajectoryCollectorConfig``
    # which applies the same override post-hoc; setting it here gives
    # the *eval* side the same capability (PolicyEvaluator has no such
    # config field) and is the recommended path for new scripts.
    xml_path: str | None = None

    # ── Build hook ─────────────────────────────────────────────────────

    def build(self):
        cfgs = super().build()
        self._apply_actuator_overrides(cfgs)
        if self.xml_path is not None:
            self._apply_xml_override(cfgs, self.xml_path)
        return cfgs

    @staticmethod
    def _apply_xml_override(cfgs, xml_path: str) -> None:
        """Replace robot entity's spec_fn to load from a custom XML file.

        Mirrors :meth:`consysid.data_collection.TrajectoryCollector
        ._override_entity_with_xml` line by line so collect-time and
        eval-time XML injection take identical effect on the resolved
        scene config. Mesh assets embedded only in the original XML
        (not the override) are borrowed from the original spec_fn so
        the substitute XML can be a thin file with just inertial tag
        edits.
        """
        from pathlib import Path

        import mujoco

        path = Path(xml_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"XML not found: {path}")

        entity_name = cfgs.scene.robot_entity_name
        entity = cfgs.scene.entities[entity_name]
        original_spec_fn = entity.spec_fn

        def custom_spec_fn() -> "mujoco.MjSpec":
            spec = mujoco.MjSpec.from_file(str(path))
            if original_spec_fn is not None and callable(original_spec_fn):
                orig_spec = original_spec_fn()
                if orig_spec.assets and not spec.assets:
                    spec.assets = orig_spec.assets
            return spec

        entity.spec_fn = custom_spec_fn

    def _apply_actuator_overrides(self, cfgs) -> None:
        """Mutate actuator configs in place to install SysID GT values.

        Walks ``cfgs.scene.entities["robot"].articulation.actuators`` and
        overwrites ``stiffness`` / ``damping`` / ``frictionloss`` based
        on the joint-name patterns each actuator group covers. Group
        membership is detected by checking for ``"calf"`` substring in
        the actuator's ``target_names_expr`` — Go2's preset splits into
        ``(.*_hip_joint, .*_thigh_joint)`` and ``(.*_calf_joint,)``.

        Mutations in place: ``actuators`` is a tuple of dataclass
        instances, so individual field writes propagate even though
        the tuple itself is immutable.
        """
        actuators = cfgs.scene.entities["robot"].articulation.actuators
        for act in actuators:
            if not isinstance(act, IdealPDActuatorCfg):
                # Only PD-style actuator configs carry stiffness/damping;
                # ImplicitActuator / ActuatorNet variants would need a
                # different override path. Skip silently for non-Go2
                # presets that may share this class but use other
                # actuator types.
                continue

            patterns = " ".join(act.target_names_expr)
            is_calf = "calf" in patterns

            kp_override = self.kp_knee if is_calf else self.kp_hip
            kd_override = self.kd_knee if is_calf else self.kd_hip

            if kp_override is not None:
                act.stiffness = kp_override
            if kd_override is not None:
                act.damping = kd_override
            if self.joint_frictionloss is not None:
                act.frictionloss = self.joint_frictionloss

    # ── Runtime hook ───────────────────────────────────────────────────

    def runtime_setup(self, env) -> None:
        """Apply runtime patches that can't be expressed at preset level.

        Currently only ``foot_friction`` — sets the foot collision
        geoms' slide-friction component to ``self.foot_friction`` and
        mirrors the value onto every ground geom so MuJoCo's max()
        contact-friction rule yields the foot's value as the effective
        pair friction (the ground's higher default would otherwise win).

        Called by :class:`TrajectoryCollector` after ``env.reset()``.
        No-op when ``foot_friction is None``.
        """
        if self.foot_friction is None:
            return

        robot = env.scene_manager.scene["robot"]

        # 1. Foot collision geoms.
        foot_geom_names = (
            "FR_foot_collision", "FL_foot_collision",
            "RR_foot_collision", "RL_foot_collision",
        )
        local_geom_ids, _ = robot.find_geoms(foot_geom_names)
        foot_geom_ids = robot.indexing.geom_ids[local_geom_ids]
        env.scene_manager.model.geom_friction[
            :, foot_geom_ids, 0,
        ] = self.foot_friction

        # 2. Ground geoms (max() rule workaround). Detect by name.
        mj_model = env.scene_manager.mj_model
        ground_indices = [
            i for i in range(mj_model.ngeom)
            if any(
                tok in mj_model.geom(i).name.lower()
                for tok in ("terrain", "ground", "plane")
            )
        ]
        for gi in ground_indices:
            env.scene_manager.model.geom_friction[
                :, gi, 0,
            ] = self.foot_friction

        # 3. Verification print (env 0).
        applied = env.scene_manager.model.geom_friction[
            0, foot_geom_ids, 0,
        ].cpu().numpy()
        print(
            f"[Go2GaitConditionedSysIDCollectConfig.runtime_setup] "
            f"foot mu={self.foot_friction} applied to {len(foot_geom_ids)} "
            f"foot geoms (env 0): {applied}; mirrored onto "
            f"{len(ground_indices)} ground geoms"
        )
