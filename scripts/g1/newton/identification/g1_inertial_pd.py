"""G1 Newton SysID -- 68D base inertial + per-joint PD (no friction).

Identifies ``G1NewtonInertialPDConfig`` (base_inertial 10 + kp 29 + kd 29)
against ``GT_G1``. Uses the explicit-PD actuator (``use_ideal_pd_actuator``)
so kp/kd map onto a clean torque path. Consumes a dataset collected with
``collect/g1_velocity_newton.py``.
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

from consysid.configs.physics import GT_G1, apply_gt_to_g1_sysid_config
from consysid.configs.presets.g1_newton import (
    G1NewtonInertialPDConfig,
    G1NewtonPDOnlyConfig,
)
from consysid.sysid.cma_optimizer import CMAOptimizerConfig
from consysid.sysid.runner import SysIDRunConfig, run_sysid
from jaxrlworld.rl.configs.presets.g1_29dof.mlp import get_config


def _parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trajectory", nargs="+", required=True,
        help="One or more wandb artifact refs or local npz paths.",
    )
    p.add_argument(
        "--pd_only", action="store_true",
        help="Diagnostic: identify PD only (fix base_inertial at nominal) "
             "to isolate whether base_inertial divergence causes nan.",
    )
    p.add_argument("--population_size", type=int, default=32)
    p.add_argument("--num_clips", type=int, default=128)
    p.add_argument("--sigma0", type=float, default=1.0)
    p.add_argument("--max_iterations", type=int, default=5000)
    p.add_argument(
        "--seed", type=int, default=42,
        help="CMA-ES + env seed. Vary across runs for independent "
             "identifications (e.g. CI bars in visualize).",
    )
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    args = _parse_cli()

    env_cfgs = get_config(
        sim="newton", use_ideal_pd_actuator=True,
    ).with_cli_overrides()
    env_cfgs.env.num_envs = args.num_clips * args.population_size
    env_cfgs.scene.solver_cfg.ccd_iterations = 10
    # G1 (29 DOF + self-collision) generates many more contacts than the
    # flat default solver buffers (njmax=1500) can hold — replayed SysID
    # candidates overflow nefc (~1663 needed) and the solver returns nan.
    env_cfgs.scene.solver_cfg.njmax = 3000
    env_cfgs.scene.solver_cfg.nconmax = 200

    config = SysIDRunConfig(
        trajectory=args.trajectory,
        upload_wandb_artifact=True,
        wandb_artifact_aliases=["g1-inertial-pd"],
        sysid_config=(
            G1NewtonPDOnlyConfig() if args.pd_only
            else G1NewtonInertialPDConfig()
        ),
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
            # Cap σ so CMA can't inflate into violent PD candidates that
            # make the biped fall (joint_vel blowup / nan). G1-specific.
            sigma_cap=0.3,
            # With σ capped, cost flattens and the default stagnation check
            # stops early while gt_error is still improving — disable it so
            # the run goes to max_iterations.
            tol_stagnation=args.max_iterations,
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

    apply_gt_to_g1_sysid_config(config.sysid_config, GT_G1)

    run_sysid(config)
