"""Decoupling-coverage explore -- G1 velocity, command-only (no gait).

Our method (``objective="decoupling_coverage"``) on the 68D
``G1NewtonInertialPDConfig`` (base_inertial 10 + kp 29 + kd 29). G1 is
velocity-conditioned only, so the Bezier optimizes vx/vy/wz with
``use_gait=False``. Command-only: no per-step action-delta and no
locomotion-stability barrier — that pairing lives in
``g1_velocity_action_delta.py``. Same FIM objective as that run, so the
only difference is the absence of the joint-level perturbation.
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

from jaxrlworld.rl.configs.presets.g1_29dof.mlp import get_config

from consysid.active.active_optimizer import ActiveCMAConfig
from consysid.active.bezier_commands import BezierCommandConfig
from consysid.active.env_partitioner import EnvPartitionConfig
from consysid.active.fim_evaluator import FIMConfig
from consysid.active.runner import ActiveExplorationConfig, run_active_exploration
from consysid.configs.presets.g1_newton import G1NewtonInertialPDConfig


def _parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population_size", type=int, default=20)
    parser.add_argument("--sigma0", type=float, default=0.25)
    parser.add_argument("--decoupling_power", type=float, default=1.0)
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
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    args = _parse_cli()
    env_cfgs = get_config(sim="newton", use_ideal_pd_actuator=True)
    env_cfgs.scene.solver_cfg.njmax = 3000
    env_cfgs.scene.solver_cfg.nconmax = 200

    BEZIER_NUM_SEGMENTS = 5
    BEZIER_SEGMENT_DURATION = 2.0
    CONTROL_STEP_S = 0.02
    EXPECTED_HORIZON = int(
        BEZIER_NUM_SEGMENTS * BEZIER_SEGMENT_DURATION / CONTROL_STEP_S,
    )

    config = ActiveExplorationConfig(
        wandb_run_path=args.policy,
        sysid_config=G1NewtonInertialPDConfig(),
        # G1 identification prior (68D) — the FIM linearization point.
        sysid_result_path=args.theta,
        upload_wandb_artifact=True,
        wandb_artifact_aliases=["decoupling-g1-velocity"],
        sim_type="newton",
        bezier=BezierCommandConfig(
            segment_duration=BEZIER_SEGMENT_DURATION,
            num_segments=BEZIER_NUM_SEGMENTS,
            bezier_degree=5,
            use_gait=False,
            optimized_commands={
                "vx": (-2.0, 2.0),
                "vy": (-1.0, 1.0),
                "wz": (-0.75, 0.75),
            },
        ),
        fim=FIMConfig(
            objective="decoupling_coverage",
            schur_eps=1e-6,
            coverage_eps=1e-6,
            decoupling_power=args.decoupling_power,
            termination_discount=1.0,
            cost_aggregation="mean",
            num_segments=BEZIER_NUM_SEGMENTS,
            expected_horizon=EXPECTED_HORIZON,
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
