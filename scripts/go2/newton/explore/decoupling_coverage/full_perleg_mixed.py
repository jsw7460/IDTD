"""Decoupling-coverage explore -- 36D full mixed, per-leg PD GT.

Command-only Bezier exploration on the 36D ``Go2NewtonFullMixedConfig``
(base_inertial 10 + per-joint kp/kd 24 + joint_friction 1 +
foot_friction 1) with the ``decoupling_coverage`` objective.

An absolute-information trace always prefers the high-information
ENTANGLED motion (walking) over the DECOUPLED one (scrape) that
actually identifies kp better. The per-parameter decoupling ratio
``Schur(F_s,{i}) / F_ii(s)`` — best segment per parameter, summed in log
space — instead rewards drawing every parameter out cleanly in at least
one segment. Segments specialise (one exposes kp, another the base
inertial, ...) and their union covers the full set, scored by
decoupling quality rather than raw magnitude.

The ratio is whitening-invariant and all 36 dims are scored (no subset
selection), so only ``schur_eps`` / ``coverage_eps`` /
``decoupling_power`` govern the objective.
"""

import argparse
import os
import sys
from pathlib import Path

_release_root = next(
    p for p in Path(__file__).resolve().parents if (p / "consysid").is_dir()
)
sys.path.insert(0, str(_release_root))
sys.path.insert(
    0, os.environ.get("JAXRLWORLD_PATH", str(_release_root / "JaxRLWorld")),
)

from jaxrlworld.rl.configs.presets.go2.newton.gait_conditioned import get_config

from consysid.active.active_optimizer import ActiveCMAConfig
from consysid.active.bezier_commands import BezierCommandConfig
from consysid.active.env_partitioner import EnvPartitionConfig
from consysid.active.fim_evaluator import FIMConfig
from consysid.active.runner import ActiveExplorationConfig, run_active_exploration
from consysid.configs.presets.go2_newton import Go2NewtonFullMixedConfig


def _parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population_size", type=int, default=25)
    parser.add_argument("--sigma0", type=float, default=0.25)
    parser.add_argument(
        "--policy", type=str, required=True,
        help="wandb run path of the trained control policy to drive "
             "during exploration (entity/project/run_id).",
    )
    parser.add_argument(
        "--theta", type=str, required=True,
        help="Stage-1 identification result (wandb artifact ref or local "
             "npz path) used as the FIM linearization point.",
    )
    # num_envs is derived by the active runner from population_size ×
    # num_groups × (param_dim + 1); there is no --num_clips for explore.
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    args = _parse_cli()
    env_cfgs = get_config()

    BEZIER_NUM_SEGMENTS = 5
    BEZIER_SEGMENT_DURATION = 2.0
    CONTROL_STEP_S = 0.02
    EXPECTED_HORIZON = int(
        BEZIER_NUM_SEGMENTS * BEZIER_SEGMENT_DURATION / CONTROL_STEP_S,
    )

    config = ActiveExplorationConfig(
        wandb_run_path=args.policy,
        sysid_config=Go2NewtonFullMixedConfig(),
        sysid_result_path=args.theta,
        upload_wandb_artifact=True,
        wandb_artifact_aliases=["explore-decoupling-coverage-full-perleg-mixed"],
        sim_type="newton",
        bezier=BezierCommandConfig(
            segment_duration=BEZIER_SEGMENT_DURATION,
            num_segments=BEZIER_NUM_SEGMENTS,
            bezier_degree=5,
            optimized_commands={
                "vx": (-0.8, 0.8),
                "vy": (-0.6, 0.6),
                "wz": (-0.8, 0.8),
                "body_pitch": (-0.2, 0.2),
                "body_height": (-0.10, 0.05),
                "footswing_height": (0.03, 0.15),
                "stance_width": (0.22, 0.32),
                "stance_length": (0.35, 0.45),
                "gait_freq": (2.5, 3.5),
            },
        ),
        fim=FIMConfig(
            objective="decoupling_coverage",
            schur_eps=1e-6,          # inv regularisation eps
            coverage_eps=1e-6,       # log-floor eps
            termination_discount=1.0,
            cost_aggregation="mean",
            num_segments=BEZIER_NUM_SEGMENTS,
            expected_horizon=EXPECTED_HORIZON,
            decoupling_power=1.0,
        ),
        partition=EnvPartitionConfig(
            num_groups=8,
            delta_param=0.1,
            ksync_steps=5,
            theta_prior_mode="fixed",
        ),
        cma=ActiveCMAConfig(
            sigma0=args.sigma0,
            max_iterations=3000,
            population_size=args.population_size,
            parallel_candidates=True,
            save_interval=50,
            tol_fun=1e-3,
        ),
        env_cfgs=env_cfgs,
    )

    run_active_exploration(config)
