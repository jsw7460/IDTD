"""Go2 Newton active exploration config presets."""

from consysid.active.active_optimizer import ActiveCMAConfig
from consysid.active.bezier_commands import BezierCommandConfig
from consysid.active.env_partitioner import EnvPartitionConfig
from consysid.active.fim_evaluator import FIMConfig
from consysid.active.runner import ActiveExplorationConfig
from consysid.configs.presets.go2_newton import (
    Go2NewtonMassComConfig,
    Go2NewtonMassComFrictionConfig,
)


def get_mass_com_config(**overrides) -> ActiveExplorationConfig:
    """Active exploration for mass + CoM (4D) identification."""
    cfg = ActiveExplorationConfig(
        sysid_config=Go2NewtonMassComConfig(),
        sim_type="newton",
        bezier=BezierCommandConfig(
            segment_duration=4.0,
            num_segments=10,
            bezier_degree=10,
        ),
        fim=FIMConfig(termination_discount=0.5),
        partition=EnvPartitionConfig(
            num_groups=4,
            delta_param=0.1,
            ksync_steps=5,
        ),
        cma=ActiveCMAConfig(
            sigma0=0.3,
            max_iterations=100,
            population_size=16,
        ),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def get_mass_com_friction_config(**overrides) -> ActiveExplorationConfig:
    """Active exploration for mass + CoM + friction (5D) identification."""
    cfg = ActiveExplorationConfig(
        sysid_config=Go2NewtonMassComFrictionConfig(),
        sim_type="newton",
        bezier=BezierCommandConfig(
            segment_duration=4.0,
            num_segments=10,
            bezier_degree=10,
        ),
        fim=FIMConfig(termination_discount=0.5),
        partition=EnvPartitionConfig(
            num_groups=4,
            delta_param=0.1,
            ksync_steps=5,
        ),
        cma=ActiveCMAConfig(
            sigma0=0.3,
            max_iterations=100,
            population_size=16,
        ),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
