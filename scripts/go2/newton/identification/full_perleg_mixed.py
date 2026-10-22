"""Go2 Newton SysID -- 36D full mixed, per-leg PD GT.

Same sysid_config as ``full_mixed.py`` (the 36D space already has
per-joint kp_all / kd_all so it can recover the four-leg distinct GT
without any layout change), but consumes the per-leg passive dataset
produced by ``collect/full_perleg_newton_overrides.py``. The CMA
search target is now per-joint heterogeneous: the four hip joints
should land at four distinct kp values and likewise for calves.
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

from consysid.sysid.open_loop_evaluator import EvaluatorConfig

from consysid.configs.physics import CURRENT_GT, apply_gt_to_sysid_config
from consysid.configs.presets.go2_newton import Go2NewtonFullMixedConfig
from consysid.sysid.cma_optimizer import CMAOptimizerConfig
from consysid.sysid.runner import SysIDRunConfig, run_sysid
from jaxrlworld.rl.configs.presets.go2.mlp import get_config


def _parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trajectory", nargs="+", required=True,
        help="One or more wandb artifact refs or local npz paths.",
    )
    p.add_argument("--population_size", type=int, default=32)
    p.add_argument("--num_clips", type=int, default=256)
    p.add_argument("--sigma0", type=float, default=1.0)
    p.add_argument("--max_iterations", type=int, default=5000)
    p.add_argument(
        "--seed", type=int, default=42,
        help="CMA-ES + env seed. Vary across runs for independent "
             "identifications (e.g. CI bars in visualize_sysid_result).",
    )
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    args = _parse_cli()

    env_cfgs = get_config(sim="newton").with_cli_overrides()
    env_cfgs.env.num_envs = args.num_clips * args.population_size
    env_cfgs.scene.solver_cfg.ccd_iterations = 10

    config = SysIDRunConfig(
        trajectory=args.trajectory,
        upload_wandb_artifact=True,
        wandb_artifact_aliases=["full-perleg-mixed"],
        sysid_config=Go2NewtonFullMixedConfig(),
        sim_type="newton",
        env_cfgs=env_cfgs,
        horizon_range=(1, 5),
        use_wandb=True,
        num_clips=args.num_clips,
        seed=args.seed,
        cma=CMAOptimizerConfig(
            population_size=args.population_size,
            max_iterations=args.max_iterations,
            sigma0=args.sigma0,
            seed=args.seed,
        ),
        evaluator=EvaluatorConfig(weights={
            "root_pos": 4.0,
            "root_vel": 2.0,
            "root_quat": 2.0,
            "root_ang_vel": 0.5,
            "joint_pos": 3.0,
            "joint_vel": 0.1,
            "joint_torques": 0.0,
        }),
    )

    apply_gt_to_sysid_config(config.sysid_config, CURRENT_GT)

    run_sysid(config)
