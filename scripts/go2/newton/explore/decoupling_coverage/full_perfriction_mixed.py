"""Decoupling-coverage explore + action-delta -- 50D heterogeneous friction.

Our method (``objective="decoupling_coverage"``) on the 50D
``Go2NewtonFullPerFrictionConfig`` (base 10 + kp 12 + kd 12 +
joint_friction 12 + foot_friction 4) with ``GT_V3``. Same action-delta +
barrier setup as the SPI-Active run so the only difference is the FIM
objective.

Per-parameter decoupling coverage:
    c_i  = max_s [ Schur(F_s,{i}) / F_ii(s)^alpha ]
    cost = -sum_i log(c_i + eps)
rewards a trajectory where EVERY parameter (incl. the 12 per-DOF joint
frictions and 4 per-foot frictions) is drawn out cleanly in at least one
segment. ``decoupling_power=1.0`` is the pure scale-free ratio; lower it
toward 0.5-0.9 to fold in an information gate against trivial (near-zero
info) solutions.
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
from consysid.active.bezier_action_delta import BezierActionDeltaConfig
from consysid.active.bezier_commands import BezierCommandConfig
from consysid.active.env_partitioner import EnvPartitionConfig
from consysid.active.fim_evaluator import FIMConfig
from consysid.active.runner import ActiveExplorationConfig, run_active_exploration
from consysid.configs.presets.go2_newton import Go2NewtonFullPerFrictionConfig


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
    env_cfgs = get_config()

    BEZIER_NUM_SEGMENTS = 5
    BEZIER_SEGMENT_DURATION = 2.0
    CONTROL_STEP_S = 0.02
    EXPECTED_HORIZON = int(
        BEZIER_NUM_SEGMENTS * BEZIER_SEGMENT_DURATION / CONTROL_STEP_S,
    )

    config = ActiveExplorationConfig(
        wandb_run_path=args.policy,
        sysid_config=Go2NewtonFullPerFrictionConfig(),
        sysid_result_path=args.theta,
        upload_wandb_artifact=True,
        wandb_artifact_aliases=["explore-decoupling-full-perfriction-action-delta"],
        sim_type="newton",
        bezier=BezierCommandConfig(
            segment_duration=BEZIER_SEGMENT_DURATION,
            num_segments=BEZIER_NUM_SEGMENTS,
            bezier_degree=5,
            optimized_commands={
                "vx": (-1.5, 1.5),
                "vy": (-0.5, 0.5),
                "wz": (-1.0, 1.0),
                "body_pitch": (-0.3, 0.3),
                "body_height": (-0.25, 0.15),
                "footswing_height": (0.03, 0.35),
                "stance_width": (0.2, 0.35),
                "stance_length": (0.25, 0.5),
                "gait_freq": (2, 4),
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
