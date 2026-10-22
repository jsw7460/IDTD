"""Bezier-spline action delta for active exploration.

Lives next to ``BezierCommandConfig`` / ``BezierCommandGenerator`` and
shares the same time grid (``segment_duration`` × ``num_segments``)
so command and delta advance in lock-step. The delta is added on top
of the policy's output at every physics step::

    a_t^exec = π(s_t, c_t) + Δa_t

The (typically small, bounded) Δa lets active exploration push the
policy slightly off-distribution to excite low-torque regimes that
the pretrained policy alone cannot reach (e.g. the Coulomb-friction
threshold), without breaking locomotion when ``bound`` is kept tight.

When ``ActiveExplorationConfig.action_delta`` is left as ``None``
(default), nothing here is built and the standard command-only
explore path is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np


@dataclass
class BezierActionDeltaConfig:
    """Per-DoF Bezier-spline action delta config.

    The time grid (``segment_duration`` × ``num_segments``) MUST match
    the paired ``BezierCommandConfig`` so command and delta advance
    in lock-step; the active runner asserts this at startup.
    """

    segment_duration: float = 4.0
    """Must match ``BezierCommandConfig.segment_duration``."""
    num_segments: int = 10
    """Must match ``BezierCommandConfig.num_segments``."""
    bezier_degree: int = 3
    """Lower than the command Bezier (default 10) on purpose: a smooth,
    low-frequency delta is enough to push the policy slightly off its
    training distribution without injecting high-frequency noise that
    a small ``bound`` couldn't tame anyway."""
    num_actuated: int = 12
    """Number of actuated DoFs that the delta covers. Default matches
    Go2's 12 leg joints."""
    bound: float = 0.05
    """Per-control-point bound in radians (PD-target offset). 0.05 rad
    ≈ 3°. Larger destabilises the policy; smaller starves friction-
    threshold excitation."""

    @property
    def num_control_points(self) -> int:
        return self.bezier_degree + 1

    def steps_per_segment(self, dt: float) -> int:
        return int(self.segment_duration / dt)

    def total_steps(self, dt: float) -> int:
        return self.num_segments * self.steps_per_segment(dt)


class BezierActionDeltaGenerator:
    """CMA parameter vector → per-step ``(T, num_actuated)`` Δa trajectory.

    Layout per segment::

        [cp_0_dof_0, ..., cp_0_dof_{nj-1},
         cp_1_dof_0, ..., cp_{deg}_dof_{nj-1}]

    All control points for one segment are contiguous, mirroring
    :class:`BezierCommandGenerator`'s layout.
    """

    def __init__(self, cfg: BezierActionDeltaConfig, dt: float):
        self.cfg = cfg
        self.dt = dt
        self._params_per_segment = cfg.num_control_points * cfg.num_actuated

        # Bernstein basis on uniform t∈[0,1] within a segment.
        sps = cfg.steps_per_segment(dt)
        t = np.linspace(0.0, 1.0, sps, dtype=np.float64)
        n = cfg.bezier_degree
        basis = np.zeros((len(t), n + 1), dtype=np.float64)
        for i in range(n + 1):
            basis[:, i] = comb(n, i) * (t**i) * ((1.0 - t) ** (n - i))
        self._basis = basis  # (sps, n+1)

    @property
    def param_dim(self) -> int:
        return self.cfg.num_segments * self._params_per_segment

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.full(self.param_dim, -self.cfg.bound, dtype=np.float32)
        hi = np.full(self.param_dim, self.cfg.bound, dtype=np.float32)
        return lo, hi

    def get_initial_x(self) -> np.ndarray:
        # Δa = 0 at init — rely on the command Bezier to steer the
        # policy and let CMA-ES discover non-zero deltas only if they
        # help FIM.
        return np.zeros(self.param_dim, dtype=np.float32)

    def decode(self, x: np.ndarray) -> np.ndarray:
        """``x`` of length :attr:`param_dim` → ``(total_steps, num_actuated)`` Δa."""
        cfg = self.cfg
        ncp = cfg.num_control_points
        nj = cfg.num_actuated
        sps = cfg.steps_per_segment(self.dt)
        out = np.zeros((cfg.num_segments * sps, nj), dtype=np.float32)
        for seg in range(cfg.num_segments):
            base = seg * self._params_per_segment
            cps = np.asarray(x[base : base + ncp * nj], dtype=np.float64)
            cps = cps.reshape(ncp, nj)  # (ncp, nj)
            samples = self._basis @ cps  # (sps, nj)
            out[seg * sps : (seg + 1) * sps, :] = samples.astype(np.float32)
        return out
