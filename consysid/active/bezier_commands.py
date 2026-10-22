"""Bezier curve command trajectory generator for active exploration.

Parameterizes command sequences as 10-degree Bezier curves per segment,
following SPI-Active Appendix A.3. CMA-ES optimizes the control points
and discrete gait modes to maximize Fisher Information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np


# Gait mode definitions: (phase, offset, bound)
# Matches JaxRLWorld GaitCommandTerm gait categories
GAIT_MODES = {
    0: (0.0, 0.0, 0.0),    # pronk
    1: (0.5, 0.0, 0.0),    # trot
    2: (0.0, 0.5, 0.0),    # pace
    3: (0.0, 0.0, 0.5),    # bound
}

# Stacked (4, 3) array of [phase, offset, bound] per mode — used for
# vectorised softmax blending (decode() takes ``probs @ GAIT_MODE_VECTORS``).
GAIT_MODE_VECTORS = np.array(
    [GAIT_MODES[i] for i in range(4)], dtype=np.float32,
)

# Default gait command values for non-optimized dimensions
DEFAULT_GAIT = {
    "gait_freq": 3.0,
    "gait_duration": 0.5,
    "footswing_height": 0.08,
    "body_height": 0.0,
    "body_pitch": 0.0,
    "body_roll": 0.0,
    "stance_width": 0.25,
    "stance_length": 0.40,
}

# Maps command name → (target_array, column_index)
# "velocity" is the internal tag for the PRIMARY 3D command channel —
# the array injected into the env command term named by
# ``BezierCommandConfig.command_term``. For locomotion robots that term
# is the velocity command [vx(0), vy(1), wz(2)]; for the quadrotor it
# is the 3D target position [x(0), y(1), z(2)] (``target_x/y/z``
# aliases below).
# "gait" array: [freq(0), phase(1), offset(2), bound(3), duration(4),
#                footswing(5), body_height(6), body_pitch(7), body_roll(8),
#                stance_width(9), stance_length(10)]
COMMAND_MAPPING = {
    "vx":              ("velocity", 0),
    "vy":              ("velocity", 1),
    "wz":              ("velocity", 2),
    "target_x":        ("velocity", 0),
    "target_y":        ("velocity", 1),
    "target_z":        ("velocity", 2),
    "gait_freq":       ("gait", 0),
    "footswing_height":("gait", 5),
    "body_height":     ("gait", 6),
    "body_pitch":      ("gait", 7),
    "body_roll":       ("gait", 8),
    "stance_width":    ("gait", 9),
    "stance_length":   ("gait", 10),
}


@dataclass
class BezierCommandConfig:
    """Bezier curve command parameterization config."""

    segment_duration: float = 4.0
    """Duration of each command segment in seconds."""
    num_segments: int = 10
    """Number of segments (~40s total trajectory)."""
    bezier_degree: int = 10
    """Degree of Bezier curve (11 control points)."""

    optimized_commands: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "vx": (-1.0, 1.0),
        "vy": (-0.6, 0.6),
        "wz": (-1.0, 1.0),
        "body_pitch": (-0.4, 0.4),
    })
    """Command names to optimize via Bezier, with (lower, upper) bounds.
    Available names: vx, vy, wz, gait_freq, footswing_height,
    body_height, body_pitch, body_roll, stance_width, stance_length."""

    default_gait: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_GAIT))
    """Fixed default values for non-optimized gait dims."""

    gait_logit_bound: float = 5.0
    """CMA-ES box bounds for the 4 gait-mode logits (one per
    segment). Larger values give the optimizer more room to sharpen
    the softmax distribution toward a single gait — at ``±5`` softmax
    sees up to ~99 % mass on one mode, at ``±3`` only ~87 %. Must be
    strictly greater than ``|get_initial_x()`` 's logit bias (currently
    set to put strong trot mass at init); CMA-ES's boundary-handling
    inverse transform diverges at the exact boundary, so any init
    value equal to ``gait_logit_bound`` will crash with
    ``ValueError: argument of inverse must be within the given
    bounds``."""

    use_gait: bool = True
    """When False, the generator emits velocity-only command trajectories
    — no gait array, no gait-mode logits — for robots without a gait
    command term (e.g. G1 velocity-tracking). ``default_gait`` /
    ``gait_logit_bound`` are then unused, ``param_dim`` drops the 4
    logits/segment, and ``decode`` returns ``(velocity, None)``."""

    command_term: str = "velocity"
    """Name of the env command-manager term the primary 3D channel
    drives. ``"velocity"`` for locomotion robots (vx/vy/wz);
    ``"target_position"`` for the Crazyflie waypoint policy
    (target_x/y/z). Persisted in the active-commands npz so the
    collect-side injection targets the same term."""

    primary_defaults: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Value of each primary-channel dim when it is NOT in
    ``optimized_commands``. Locomotion keeps the zero default; the
    Crazyflie sets ``(0, 0, 1.0)`` because the hover policy was trained
    with target z fixed at 1.0 m (only x/y are optimized)."""

    state_feedback_features: tuple[str, ...] = ()
    """Names of robot-state summary features used as inputs to the
    linear feedback term added on top of the Bezier-decoded command::

        cmd_t = cmd_bezier(t) + K @ features_t

    ``K`` is a ``(cmd_dim_total=14, len(features))`` matrix whose
    flattened entries are appended to the CMA-ES parameter vector --
    CMA jointly optimises the Bezier control points *and* the
    state-feedback gains. Empty tuple (default) disables the feedback
    so the Bezier-only behaviour is preserved bit-identically.

    Available feature names are resolved by
    :func:`consysid.active.state_feedback.extract_features`."""

    state_feedback_gain_bound: float = 0.1
    """Per-element CMA box bound on ``K`` entries. With features
    normalised to ``~[-1, 1]`` by
    :func:`consysid.active.state_feedback.extract_features`, ``K`` of
    magnitude ``0.1`` produces a feedback correction of magnitude
    ``~0.1`` -- about 10-15 % of the typical command box (e.g.
    velocity bound ``±0.8``), enough to be meaningful but not enough
    to wash out the Bezier baseline or push the policy out of its
    training distribution during the first CMA-ES iterations."""

    @property
    def continuous_dim(self) -> int:
        return len(self.optimized_commands)

    @property
    def num_control_points(self) -> int:
        return self.bezier_degree + 1

    def steps_per_segment(self, dt: float) -> int:
        """Number of timesteps per segment for a given control dt."""
        return int(self.segment_duration / dt)

    def total_steps(self, dt: float) -> int:
        """Total timesteps for a given control dt."""
        return self.num_segments * self.steps_per_segment(dt)


class BezierCommandGenerator:
    """Generates 14D command trajectories from CMA-ES parameter vectors.

    CMA-ES parameter layout per segment::

        [cp_0_dim0, ..., cp_0_dimC, ..., cp_N_dimC, logit_0, ..., logit_3]

    Total params per segment = num_control_points * continuous_dim + 4
    Total params = num_segments * params_per_segment
    """

    def __init__(self, config: BezierCommandConfig, dt: float):
        """
        Args:
            config: Bezier parameterization config.
            dt: Environment control timestep. Determines how many
                discrete points to sample from each Bezier curve.
                Obtained from ``env_cfg.scene.physics_dt * env_cfg.env.decimation``.
        """
        self.cfg = config
        self.dt = dt

        # Ordered list of (name, lo, hi, target_array, col_index)
        self._opt_dims: list[tuple[str, float, float, str, int]] = []
        for name, (lo, hi) in config.optimized_commands.items():
            if name not in COMMAND_MAPPING:
                raise ValueError(
                    f"Unknown command '{name}'. "
                    f"Available: {list(COMMAND_MAPPING.keys())}"
                )
            target, col = COMMAND_MAPPING[name]
            self._opt_dims.append((name, lo, hi, target, col))

        # Velocity-only configs may not optimize gait dims (there is no
        # gait array to write them into).
        if not config.use_gait:
            for name, _lo, _hi, target, _col in self._opt_dims:
                if target == "gait":
                    raise ValueError(
                        f"use_gait=False but optimized_commands has gait dim "
                        f"'{name}'; velocity-only configs may only optimize "
                        f"vx / vy / wz."
                    )

        cdim = config.continuous_dim
        # The 4 gait-mode logits are appended per segment only when gait is
        # active; velocity-only robots (use_gait=False) drop them.
        n_gait_logits = 4 if config.use_gait else 0
        self._params_per_segment = config.num_control_points * cdim + n_gait_logits

        # Precompute Bernstein basis for uniform timesteps within a segment
        sps = config.steps_per_segment(dt)
        t = np.linspace(0, 1, sps, dtype=np.float64)
        self._basis = self._bernstein_basis(config.bezier_degree, t)

        # State-feedback ``K`` (cmd_dim_total × n_features). ``cmd_dim_total``
        # is the decoded command width — 14 (velocity 3 + gait 11) with gait,
        # 3 (velocity only) without. ``n_features == 0`` is the legacy
        # Bezier-only path and produces a zero-size suffix on the CMA vector.
        self._cmd_dim_total = 14 if config.use_gait else 3
        self._fb_n_feat = len(config.state_feedback_features)
        self._K_dim = self._cmd_dim_total * self._fb_n_feat

    @property
    def param_dim(self) -> int:
        """Total CMA-ES parameter dimension (Bezier control points +
        optional state-feedback gain matrix)."""
        return self.cfg.num_segments * self._params_per_segment + self._K_dim

    @staticmethod
    def _bernstein_basis(n: int, t: np.ndarray) -> np.ndarray:
        """Evaluate all Bernstein basis polynomials of degree n.

        B_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i)

        Returns:
            (T, n+1) array of basis values.
        """
        basis = np.zeros((len(t), n + 1), dtype=np.float64)
        for i in range(n + 1):
            basis[:, i] = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        return basis

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) bounds for CMA-ES parameter vector."""
        lower_seg = []
        upper_seg = []

        # Control point bounds: repeat per control point
        for _ in range(self.cfg.num_control_points):
            for _, lo, hi, _, _ in self._opt_dims:
                lower_seg.append(lo)
                upper_seg.append(hi)

        # Gait logit bounds (only when gait is active)
        if self.cfg.use_gait:
            for _ in range(4):
                lower_seg.append(-self.cfg.gait_logit_bound)
                upper_seg.append(self.cfg.gait_logit_bound)

        # Tile across segments
        lower = np.tile(lower_seg, self.cfg.num_segments).astype(np.float32)
        upper = np.tile(upper_seg, self.cfg.num_segments).astype(np.float32)

        # Append per-element bounds for the state-feedback K matrix.
        # No-op when ``state_feedback_features`` is empty.
        if self._K_dim > 0:
            k_lo = np.full(
                self._K_dim, -self.cfg.state_feedback_gain_bound, dtype=np.float32,
            )
            k_hi = np.full(
                self._K_dim, self.cfg.state_feedback_gain_bound, dtype=np.float32,
            )
            lower = np.concatenate([lower, k_lo])
            upper = np.concatenate([upper, k_hi])
        return lower, upper

    def get_initial_x(self) -> np.ndarray:
        """Return a reasonable initial parameter vector (mid-range commands)."""
        x = np.zeros(self.param_dim, dtype=np.float32)
        cdim = self.cfg.continuous_dim
        ncp = self.cfg.num_control_points

        for seg in range(self.cfg.num_segments):
            offset = seg * self._params_per_segment
            for cp in range(ncp):
                for d, (_, lo, hi, _, _) in enumerate(self._opt_dims):
                    x[offset + cp * cdim + d] = (lo + hi) / 2.0
            # Strong trot bias under softmax decoding. Two earlier
            # variants both crashed every candidate in iter 1:
            #   * ``logit[trot]=0`` (uniform softmax): decoded gait
            #     (phase, offset, bound) = (0.125, 0.125, 0.125) — no
            #     canonical gait pattern, policy could not stabilise.
            #   * ``logit[trot]=1`` (~47 % trot, others ~17 %): even
            #     CMA-ES candidates leaning into trot only reached
            #     ~55 % trot mass, the remaining ~45 % "leaked" into
            #     pronk/pace/bound and produced an unstable gait soup.
            # ``logit[trot]=2`` puts ~71 % mass on trot at init with
            # ~10 % per other mode — comfortably inside the policy's
            # stable-walking regime while still leaving room for CMA-
            # ES to discover that another gait scores better on the
            # FIM objective (mode switching only requires a logit
            # delta of ~+2 on a competitor, well within ``sigma0=0.25``
            # × ``max_iterations=300`` reachable distance). If even
            # this still crashes every candidate, push to 3.0 (max
            # ``gait_logit_bound``) to recover near-argmax behaviour.
            if self.cfg.use_gait:
                x[offset + ncp * cdim + 1] = 1.0

        return x

    def decode_gait_probs(self, x: np.ndarray) -> np.ndarray:
        """Return per-segment softmax probabilities over the 4 gait modes.

        Companion to :meth:`decode` — exposes the softmax distribution
        that the decode-time blend uses, so callers (typically the
        active-exploration optimizer) can log which gait CMA-ES is
        gravitating toward without having to recompute the blend.

        Args:
            x: Flat parameter vector of length ``param_dim``.

        Returns:
            ``(num_segments, 4)`` array. Row ``s`` is
            ``softmax(logits_s)`` over modes ``[pronk, trot, pace, bound]``
            and sums to 1.
        """
        cfg = self.cfg
        cdim = cfg.continuous_dim
        ncp = cfg.num_control_points
        probs = np.zeros((cfg.num_segments, 4), dtype=np.float32)
        for seg in range(cfg.num_segments):
            offset = seg * self._params_per_segment
            logits = x[offset + ncp * cdim: offset + ncp * cdim + 4]
            shifted = logits - np.max(logits)
            p = np.exp(shifted)
            probs[seg] = p / p.sum()
        return probs

    def decode_K(self, x: np.ndarray) -> np.ndarray | None:
        """Slice the state-feedback gain matrix ``K`` from ``x``.

        Returns:
            ``(cmd_dim_total=14, len(state_feedback_features))`` array
            -- ``None`` when ``state_feedback_features`` is empty so
            the caller can branch on a single ``is None`` check.
        """
        if self._fb_n_feat == 0:
            return None
        base_dim = self.cfg.num_segments * self._params_per_segment
        K_flat = x[base_dim: base_dim + self._K_dim]
        return K_flat.reshape(self._cmd_dim_total, self._fb_n_feat)

    def decode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode CMA-ES vector to velocity and gait command trajectories.

        Args:
            x: Flat parameter vector of length ``param_dim``.

        Returns:
            velocity_cmds: (total_steps, 3) — [vx, vy, wz]
            gait_cmds: (total_steps, 11) — full gait command vector
        """
        cfg = self.cfg
        cdim = cfg.continuous_dim
        ncp = cfg.num_control_points
        sps = cfg.steps_per_segment(self.dt)
        total_steps = cfg.total_steps(self.dt)

        velocity_cmds = np.zeros((total_steps, 3), dtype=np.float32)
        # Non-optimized primary dims sit at their configured default
        # (all-zero for locomotion; e.g. z=1.0 for the drone target).
        velocity_cmds[:] = np.asarray(cfg.primary_defaults, dtype=np.float32)
        # Velocity-only configs (use_gait=False) emit no gait trajectory.
        if not cfg.use_gait:
            gait_cmds = None
        else:
            gait_cmds = np.zeros((total_steps, 11), dtype=np.float32)
            # Fill gait defaults for all timesteps
            gait_cmds[:, 0] = cfg.default_gait["gait_freq"]
            gait_cmds[:, 4] = cfg.default_gait["gait_duration"]
            gait_cmds[:, 5] = cfg.default_gait["footswing_height"]
            gait_cmds[:, 6] = cfg.default_gait["body_height"]
            gait_cmds[:, 7] = cfg.default_gait["body_pitch"]
            gait_cmds[:, 8] = cfg.default_gait["body_roll"]
            gait_cmds[:, 9] = cfg.default_gait["stance_width"]
            gait_cmds[:, 10] = cfg.default_gait["stance_length"]

        for seg in range(cfg.num_segments):
            offset = seg * self._params_per_segment
            t_start = seg * sps
            t_end = t_start + sps

            # Extract control points: (ncp, cdim)
            cp_flat = x[offset: offset + ncp * cdim]
            control_points = cp_flat.reshape(ncp, cdim)

            # Clip control points to bounds
            for d, (_, lo, hi, _, _) in enumerate(self._opt_dims):
                control_points[:, d] = np.clip(control_points[:, d], lo, hi)

            # Evaluate Bezier curve: (sps, cdim)
            curve = self._basis @ control_points

            # Clip curve values and write to output arrays
            for d, (_, lo, hi, target, col) in enumerate(self._opt_dims):
                values = np.clip(curve[:, d], lo, hi)
                if target == "velocity":
                    velocity_cmds[t_start:t_end, col] = values
                else:
                    gait_cmds[t_start:t_end, col] = values

            # Softmax blend over the 4 gait modes — replaces the prior
            # argmax-based discrete selection. Argmax is piecewise-
            # constant in logit space, so small CMA-ES perturbations
            # never flipped the mode (and the population stayed glued
            # to whichever gait the init bias preferred). Softmax makes
            # the gait choice fully smooth in logits, restoring CMA-ES's
            # ability to traverse mode boundaries gradient-free.
            #
            # The blended (phase, offset, bound) triple is also strictly
            # within the policy's training distribution: GaitCommandTerm
            # samples each of phase / offset / bound continuously from
            # [0, 1] during pretraining, while the 4 corners
            # {pronk, trot, pace, bound} live at component values in
            # {0, 0.5}. Any softmax convex combination therefore stays
            # inside the policy's seen domain (in fact a *strict subset*
            # of it).
            if cfg.use_gait:
                logits = x[offset + ncp * cdim: offset + ncp * cdim + 4]
                shifted = logits - np.max(logits)         # numerical-safe softmax
                probs = np.exp(shifted)
                probs = probs / probs.sum()               # (4,)
                blended = probs @ GAIT_MODE_VECTORS       # (3,) [phase, offset, bound]
                gait_cmds[t_start:t_end, 1] = blended[0]
                gait_cmds[t_start:t_end, 2] = blended[1]
                gait_cmds[t_start:t_end, 3] = blended[2]

        return velocity_cmds, gait_cmds
