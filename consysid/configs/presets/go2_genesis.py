"""Go2 Genesis SysID preset configurations."""
from dataclasses import dataclass

from consysid.configs.sysid_term_config import SysIDConfig, SysIDTermConfig
from consysid.sysid.param_terms import genesis as gf


@dataclass
class Go2GenesisMassComConfig(SysIDConfig):
    """Go2 Genesis: mass + CoM (4D)."""
    base_mass = SysIDTermConfig(
        func=gf.apply_mass,
        dim=1,
        default=6.921,
        bounds=(3.0, 20.0),
        reg_weight=0.0005,
        params={"link_name": "base"},
    )
    base_com = SysIDTermConfig(
        func=gf.apply_com,
        dim=3,
        default=[0.0, 0.0, 0.0],
        bounds=(-0.5, 0.5),
        reg_weight=1.0,
        params={"link_name": "base"},
    )


@dataclass
class Go2GenesisMassComGainsConfig(Go2GenesisMassComConfig):
    """Go2 Genesis: mass + CoM + PD gains (8D).

    Go2 has 12 DOFs in 2 groups: hip (4 joints) and knee (8 joints).
    Gains are per-group scalars broadcast to all joints in the group.
    """
    kp_hip = SysIDTermConfig(
        func=gf.apply_kp,
        dim=4,
        default=[20.0] * 4,
        bounds=(10.0, 40.0),
        reg_weight=0.001,
        params={"dof_pattern": [".*_hip_joint"]},
    )
    kd_hip = SysIDTermConfig(
        func=gf.apply_kd,
        dim=4,
        default=[0.5] * 4,
        bounds=(0.1, 2.0),
        reg_weight=0.01,
        params={"dof_pattern": [".*_hip_joint"]},
    )
    kp_leg = SysIDTermConfig(
        func=gf.apply_kp,
        dim=8,
        default=[20.0] * 8,
        bounds=(10.0, 40.0),
        reg_weight=0.001,
        params={"dof_pattern": [".*_thigh_joint", ".*_calf_joint"]},
    )
    kd_leg = SysIDTermConfig(
        func=gf.apply_kd,
        dim=8,
        default=[0.5] * 8,
        bounds=(0.1, 2.0),
        reg_weight=0.01,
        params={"dof_pattern": [".*_thigh_joint", ".*_calf_joint"]},
    )
