# IDTD — Informationally Decoupled Trajectory Design for Sim-to-Real System Identification

Paper: https://arxiv.org/abs/2610.10905

Active exploration for system identification. Given a trained control
policy and a parametric physics model, IDTD searches for a command
trajectory that makes **every** physical parameter identifiable, then
identifies those parameters from the resulting rollout.

The objective is a per-parameter *decoupling coverage*: for each
parameter `i` and trajectory segment `s`, the single-axis conditional
Fisher ratio

```
c_i = max_s  Schur(F_s, {i}) / F_ii(s)^α
cost = - Σ_i log(c_i + ε)
```

rewards a trajectory on which every parameter is drawn out cleanly
(decoupled from the rest) in at least one segment, rather than one that
simply maximises aggregate information and leaves weakly-excited axes
entangled. The exponent `α` (`decoupling_power`) is an information gate:
`α = 1` is the pure scale-free ratio, and `α < 1` folds in an `F_ii^{1-α}`
factor that pushes the motion to excite otherwise weakly-observed axes.
`α` may be a scalar or a per-parameter vector (see the explore scripts).

## Layout

```
consysid/            the package (physics configs, parameters, FIM
                     evaluator, CMA optimiser, open-loop evaluator,
                     active-exploration runner, data collection)
scripts/
  examples/          standalone toy examples (pure NumPy + CMA)
  g1/  go2/          per-robot identification + exploration scripts
```

## Dependencies

Install the Python dependencies:

```
pip install -r requirements.txt
```

This code additionally depends on **JaxRLWorld**, which is not on PyPI.
Obtain a JaxRLWorld checkout separately and make it importable in one of
two ways:

* place it at `./JaxRLWorld` (the default the scripts look for), or
* point `JAXRLWORLD_PATH` at your checkout:

  ```
  export JAXRLWORLD_PATH=/path/to/JaxRLWorld
  ```

Each script resolves the package root automatically (it looks for the
directory that contains `consysid/`), so scripts can be run from any
working directory once `consysid` and JaxRLWorld are importable.

## Pipeline

Two steps, each a standalone script:

1. **Identify** parameters from a trajectory dataset (open-loop CMA-ES
   fit against recorded rollouts).
2. **Explore**: given the identified parameters as the linearisation
   point and a trained policy, optimise a command (Bezier) trajectory
   under the decoupling-coverage objective. The resulting command
   trajectory is exported for the next data-collection round.

### Identification

```
python scripts/g1/newton/identification/g1_inertial_pd.py \
    --trajectory ENTITY/PROJECT/DATASET:vN

python scripts/go2/newton/identification/full_perfriction_mixed.py \
    --trajectory ENTITY/PROJECT/DATASET:vN
```

`--trajectory` accepts one or more local `.npz` paths or wandb artifact
references.

### Active exploration

```
python scripts/g1/newton/explore/decoupling_coverage/g1_velocity.py \
    --policy ENTITY/PROJECT/POLICY_RUN \
    --theta  ENTITY/PROJECT/sysid-result:vN

python scripts/go2/newton/explore/decoupling_coverage/full_perfriction_mixed_action_delta.py \
    --policy ENTITY/PROJECT/POLICY_RUN \
    --theta  ENTITY/PROJECT/sysid-result:vN
```

* `--policy` is the wandb run of the trained control policy to drive
  during exploration.
* `--theta` is an identification result (the FIM linearisation point).
* `--decoupling_power` overrides `α` uniformly for every parameter;
  omit it to use each script's per-term `ALPHA_PER_TERM` table.

The `*_action_delta` variants additionally optimise a per-step joint
action offset under a locomotion-stability barrier.

## Toy example

```
python scripts/examples/linear_dynamics.py
```

A self-contained linear-dynamics demonstration of the decoupling-coverage
objective (NumPy + CMA only, no simulator or policy required).

## Citation

If you use this code, please cite:

> *Informationally Decoupled Trajectory Design for Sim-to-Real System
> Identification.* https://arxiv.org/abs/2610.10905
