"""Fisher Information Matrix evaluator via finite differences.

Computes FIM from state differences between main and auxiliary
environments, following SPI-Active Eq. 12:

    F(theta, pi) = sigma^{-2} * sum_t (df/dtheta)^T (df/dtheta)

where the gradient is approximated by finite differences:

    df/dtheta_i ≈ (state_main - state_aux_i) / delta

All accumulators are ``(P, G, …)`` shaped — ``P=num_candidates`` lets
a single rollout cover every CMA-ES candidate simultaneously. Serial
callers construct the partitioner with ``num_candidates=1``; the math
collapses to the legacy ``(G, …)`` shapes via a size-1 leading axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch

from consysid.active.env_partitioner import EnvPartitioner


FIMObjective = Literal["decoupling_coverage"]
"""Accepted value for ``FIMConfig.objective``. Exposed as a module-level
alias so downstream code can annotate variables with the same type."""


@dataclass
class FIMConfig:
    """Fisher Information Matrix computation config.

    CMA-ES minimizes ``cost`` for the ``"decoupling_coverage"``
    objective: the trajectory is split into ``num_segments`` time
    slices, a per-segment FIM is accumulated, and each parameter is
    scored by its best segment,

        ``c_i = max_s [ Schur(F_s, {i}) / F_ii(s)^α ]``
        ``cost = −Σ_i log(c_i + ε)``

    where ``Schur(F_s, {i}) / F_ii(s)`` is the single-axis decoupling
    ratio in ``[0, 1]`` (how cleanly parameter ``i`` is identifiable in
    segment ``s`` after marginalising every other parameter) and ``α``
    (:attr:`decoupling_power`) sets the information gate (see there).
    The ``−Σ log`` makes the weakest axis dominate, so EVERY parameter
    must be drawn out in at least one segment.
    """

    state_channels: tuple[str, ...] | None = None
    """Which state channels enter the finite-difference Jacobian.
    ``None`` (default) keeps the full vector — root_pos, root_quat,
    root_lin_vel, root_ang_vel, joint_pos, joint_vel — bit-identical to
    the full-vector behavior. Hardware-faithful exploration restricts
    this to the channels the physical robot can actually measure (e.g.
    drop root_pos / root_lin_vel when there is no mocap), so the
    optimizer cannot pump information into unobservable directions."""

    objective: FIMObjective = "decoupling_coverage"
    """Which FIM cost to feed CMA-ES. Only ``"decoupling_coverage"`` is
    supported (see the class docstring)."""

    schur_eps: float = 1e-6
    """Tikhonov regularisation added to the Fisher diagonal before each
    single-axis inversion. Large enough to keep the inverse stable when
    a parameter is poorly identified, small enough to be negligible when
    it carries O(100-1000) information."""

    termination_discount: float = 1.0
    """Multiplicative discount applied to a terminated group's cost
    (1.0 = no penalty, 0.0 = discard it entirely), so a candidate whose
    rollout fell over is penalised rather than rewarded."""

    cost_aggregation: str = "mean"
    """How the per-(candidate, group) cost tensor is reduced along the
    group axis to produce the scalar CMA-ES objective per candidate.

    Only meaningful when groups carry *different* θ_g (i.e.
    ``EnvPartitionConfig.theta_prior_mode != 'fixed'``); under
    ``"fixed"`` mode every group sees the same θ and mean / max are
    interchangeable up to MC noise.

    Modes:
      * ``"mean"``        — Bayesian average of FIM across the group
        prior. CMA-ES maximises the *expected* Fisher information
        under ``p(θ)``. Default; matches the legacy single-θ path
        when groups share θ.
      * ``"worst_case"``  — robust min-max. CMA-ES maximises the
        *minimum* per-group Fisher information across the prior — the
        commands are guaranteed to be informative under the *least
        favourable* θ_g. More conservative; useful when the surrogate
        is suspected to be biased and the prior covers the GT
        somewhere in its support.

    Cost is in minimisation form (negative info), so ``"worst_case"``
    is implemented as ``cost.max(dim=-1)`` (the worst cost = least
    info) — symmetric to the standard min-max E-optimal design."""

    # ── Trajectory-segmentation parameters ──────────────────────────────
    # The trajectory is split into ``num_segments`` equal time slices; a
    # per-segment FIM is accumulated, and each parameter is scored by its
    # best segment (coverage), so different segments are free to
    # specialise on different parameters.
    num_segments: int = 4
    """Number of equal-length trajectory segments. Each parameter is
    scored by its best segment, so a natural choice is
    ``num_segments == (number of parameter groups)`` to give the
    trajectory room to specialise on each group."""

    expected_horizon: int = 200
    """Nominal trajectory length used to compute
    ``segment_size = expected_horizon // num_segments``.
    ``accumulate_step``'s internal counter divides each step into a
    segment via ``min(step // segment_size, num_segments - 1)`` so a
    longer-than-expected horizon still lands in a valid segment. Set to
    the explore-time rollout length (``Bezier``
    ``num_segments * segment_duration / dt``)."""

    coverage_eps: float = 1e-6
    """Tikhonov ε for the single-axis conditional-Fisher inverse and the
    ``log`` floor in the ``−Σ_i log(c_i + ε)`` cost."""

    decoupling_power: float | Sequence[float] = 1.0
    """(``"decoupling_coverage"`` only) Exponent ``α`` on the denominator:
    per-param score ``= Schur(F_s,{i}) / F_ii(s)^α = F_ii^{1-α} · (decoupling
    ratio)``. ``α=1`` (default) → pure scale-free decoupling ratio (no info
    weighting). ``0<α<1`` folds in an
    information gate ``F_ii^{1-α}`` so a parameter must actually be excited
    (not just trivially "decoupled" at ~0 info) to score — and a motion
    that excites it MORE while staying decoupled scores higher. ``α=0`` →
    absolute single-axis Schur (info only, drifts back toward the
    high-information entangled motion). Sweet spot typically ``0.5–0.9``.

    A scalar applies one ``α`` to every parameter. A length-``param_dim``
    sequence sets ``α`` per parameter (axis order matching ``theta``):
    give weakly-excited axes a lower ``α`` so the info gate forces the
    motion to excite them, and well-excited axes ``α=1`` for pure
    decoupling. Build it from a per-term dict with
    :func:`consysid.sysid.parameters.build_per_param_vector`."""

    coverage_seg_agg: str = "max"
    """(``"decoupling_coverage"`` only) How the per-param score aggregates
    over segments. ``"max"`` (default) = COVERAGE — each parameter is
    scored by its single best segment, so different segments are free to
    specialise on different axes. ``"mean"`` averages over segments,
    removing the coverage structure (every segment must serve every
    axis)."""

    coverage_param_agg: str = "log"
    """(``"decoupling_coverage"`` only) How per-param scores combine into
    the scalar cost. ``"log"`` (default) ``-Σ_i log(c_i + ε)`` — the
    weakest axis dominates, so EVERY parameter must be drawn out.
    ``"linear"`` ``-Σ_i c_i`` is a plain sum a few strong axes can
    carry."""


class FIMEvaluator:
    """Accumulates per-step finite-difference gradients into FIM.

    State vector per env: root_pos(3) + root_quat(4) + root_lin_vel(3)
    + root_ang_vel(3) + joint_pos(nj) + joint_vel(nj).

    For Go2 with 12 joints: 3+4+3+3+12+12 = 37D.

    Accumulator shape: ``(P, G, N, N)`` — one (N, N) FIM matrix per
    ``(candidate, group)`` pair. Per-group averaging over ``G`` and
    per-candidate reduction happen in ``compute_fim_objective``'s
    consumer (typically ``.mean(dim=-1)`` to collapse groups).
    """

    def __init__(
        self,
        partitioner: EnvPartitioner,
        config: FIMConfig,
        device: torch.device,
    ):
        self.partitioner = partitioner
        self.cfg = config
        self.device = device

        # Per-dim delta for the FD Jacobian. Pulled from the partitioner
        # so the same vector is used on both the perturbation side
        # (build_theta_batch) and the gradient side (accumulate_step) —
        # any drift between the two would bias the Jacobian.
        self._delta_vec = torch.from_numpy(
            partitioner.delta_vec,
        ).to(device=device, dtype=torch.float32)

        self._group_terminated: torch.Tensor | None = None

        # Per-segment FIM accumulator. Allocated lazily in :meth:`reset`.
        self._XtX_seg: torch.Tensor | None = None
        self._step_per_seg: torch.Tensor | None = None
        self._global_step: int = 0

    def reset(self) -> None:
        """Reset accumulated FIM for a new evaluation."""
        P = self.partitioner.num_candidates
        G = self.partitioner.cfg.num_groups
        N = self.partitioner.param_dim
        self._group_terminated = torch.zeros(P, G, dtype=torch.bool, device=self.device)

        # Per-segment FIM buffers (``(num_seg, N, N)`` per group).
        num_seg = max(1, int(self.cfg.num_segments))
        self._XtX_seg = torch.zeros(P, G, num_seg, N, N, device=self.device)
        self._step_per_seg = torch.zeros(P, G, num_seg, device=self.device)
        self._global_step = 0

    def read_state_vector(self, env: Any) -> torch.Tensor:
        """Read full state from all envs as a flat tensor.

        Returns:
            (total_envs, state_dim) tensor.
        """
        robot_data = env.get_robot_data()

        parts = {
            "root_pos": robot_data.root_link_pos_w,          # (E, 3)
            "root_quat": robot_data.root_link_quat_w,        # (E, 4)
            "root_lin_vel": robot_data.root_link_lin_vel_w,  # (E, 3)
            "root_ang_vel": robot_data.root_link_ang_vel_w,  # (E, 3)
            "joint_pos": robot_data.joint_pos,               # (E, nj)
            "joint_vel": robot_data.joint_vel,               # (E, nj)
        }
        channels = self.cfg.state_channels or tuple(parts)
        unknown = [c for c in channels if c not in parts]
        if unknown:
            raise ValueError(
                f"FIMConfig.state_channels contains unknown channel(s) "
                f"{unknown}; valid: {tuple(parts)}"
            )
        return torch.cat([parts[c] for c in channels], dim=1)

    def accumulate_step(self, state_all: torch.Tensor) -> None:
        """Accumulate one timestep's gradient contribution to FIM.

        For each non-terminated ``(candidate, group)`` pair and each
        param dim ``i``:
            grad_i = (state_main - state_aux_i) / delta_i
            XtX   += grad @ grad^T   # (N, N)

        Per-dim ``delta`` matters once different terms carry different
        natural scales (e.g. ``0.1`` kg vs ``0.1`` unitless friction);
        applying the same scalar δ everywhere would systematically
        distort the cross-block Jacobian.

        Args:
            state_all: ``(total_envs, state_dim)`` from read_state_vector.
        """
        part = self.partitioner
        P = part.num_candidates
        G = part.cfg.num_groups
        N = part.param_dim
        gs = part._group_size

        # Reshape to (P, G, group_size, state_dim)
        state_grouped = state_all.reshape(P, G, gs, -1)

        # Main state: (P, G, 1, state_dim)
        main_state = state_grouped[:, :, 0:1, :]

        # Aux states: (P, G, N, state_dim)
        aux_states = state_grouped[:, :, 1:, :]

        # Finite-difference gradient: (P, G, N, state_dim).
        # ``_delta_vec`` is (N,); broadcast by reshaping to (1, 1, N, 1).
        delta_broadcast = self._delta_vec.view(1, 1, N, 1)
        gradient = (main_state - aux_states) / delta_broadcast

        # XtX: (P, G, N, N) = gradient @ gradient^T per (candidate, group).
        # ``matmul`` broadcasts the leading dims so we get a per-(P,G)
        # outer product in one kernel launch.
        XtX = torch.matmul(gradient, gradient.transpose(-1, -2))

        # Mask out terminated (candidate, group) pairs — keeps FIM frozen
        # at the last alive step so downstream cost is invariant to
        # trailing dead time. ``(P, G)`` broadcast up to ``(P, G, N, N)``
        # via ``view(P, G, 1, 1)``.
        alive_mask = ~self._group_terminated  # (P, G)
        XtX_masked = XtX * alive_mask.view(P, G, 1, 1).to(XtX.dtype)

        # Per-segment accumulation.
        # ``_global_step`` advances once per ``accumulate_step`` call
        # regardless of per-group termination — segment indexing is a
        # function of trajectory time, not per-group alive count. The
        # alive_mask still gates the XtX add per (P, G) so a terminated
        # group contributes zeros to its current segment.
        self._global_step += 1
        num_seg = self._XtX_seg.shape[2]
        segment_size = max(
            1, int(self.cfg.expected_horizon) // num_seg,
        )
        seg_idx = min((self._global_step - 1) // segment_size, num_seg - 1)
        self._XtX_seg[:, :, seg_idx] += XtX_masked
        self._step_per_seg[:, :, seg_idx] += alive_mask.to(
            self._step_per_seg.dtype,
        )

    def mark_terminated(self, group_mask: torch.Tensor) -> None:
        """Mark ``(candidate, group)`` pairs as terminated.

        Args:
            group_mask: ``(P, G)`` bool tensor.
        """
        self._group_terminated |= group_mask

    @property
    def group_terminated(self) -> torch.Tensor:
        """Read-only view of which ``(P, G)`` pairs terminated. Callers
        use this for diagnostics so they don't re-derive termination
        from the scalar cost — which is ambiguous once the cost mixes
        reward and penalty channels."""
        return self._group_terminated

    def compute_fim_objective(self) -> dict:
        """Compute the decoupling-coverage cost plus per-parameter
        diagnostics. Returns a dict with ``(P, G)``-shaped tensors:

            cost                   = decoupling-coverage cost (lower is
                                     better)
            decoupling_per_param_c = (P, G, N) per-parameter coverage
                                     score ``c_i`` (best segment, after
                                     the chosen aggregations)

        Callers typically reduce over the group axis (``.mean(dim=-1)``)
        to obtain a per-candidate ``(P,)`` score before feeding CMA-ES.
        Serial evaluation (``P=1``) gets back the ``(G,)`` semantics
        after squeezing the leading dim.

        The per-parameter decoupling coverage credits each parameter its
        best segment::

            c_i  = max_s [ Schur(F_s, {i}) / F_ii(s)^α ]
            cost = - Σ_i log(c_i + eps_log)

        so a trajectory is rewarded when every parameter is drawn out
        cleanly (decoupled from the rest) in at least one segment.
        """
        cost, diagnostics = self._compute_decoupling_coverage_cost(
            eps=self.cfg.schur_eps,
        )
        result = {"cost": cost}
        result.update(diagnostics)
        return result

    # ── Decoupling-coverage implementation ───────────────────────────────
    def _compute_decoupling_coverage_cost(
        self, eps: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Per-(candidate, group) per-parameter decoupling coverage cost.

        For each parameter ``i`` the single-axis DECOUPLING ratio in a
        segment is::

            d_i(s) = Schur(F_s, {i}) / F_ii(s)
                   = 1 / ( (F_s + eps I)^{-1}_ii  ·  F_ii(s) )      in [0, 1]

        (~1 → ``i`` is drawn out cleanly given the rest; ~0 → entangled;
        the ratio is scale-free in each axis.) Each parameter is
        credited its BEST segment, then aggregated::

            c_i  = max_s d_i(s)
            cost = - Σ_i log(c_i + eps_log)

        Minimising this rewards a trajectory whose every parameter is
        decoupled-identifiable in at least one segment — segments
        specialise (one exposes kp, another the base inertial, ...) and
        their union covers the full set, scored by decoupling quality
        rather than absolute information.
        """
        assert self._XtX_seg is not None and self._step_per_seg is not None
        P = self.partitioner.num_candidates
        G = self.partitioner.cfg.num_groups
        N = self.partitioner.param_dim
        num_seg = self._XtX_seg.shape[2]
        device = self.device
        eps_log = float(self.cfg.coverage_eps)

        valid = self._step_per_seg > 0                       # (P, G, num_seg)
        denom = self._step_per_seg.clamp(min=1).view(P, G, num_seg, 1, 1)
        M_seg = torch.where(
            valid.view(P, G, num_seg, 1, 1).expand_as(self._XtX_seg),
            self._XtX_seg / denom,
            torch.zeros_like(self._XtX_seg),
        )
        M_seg_np = M_seg.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()

        eye_N = np.eye(N)
        # alpha: a scalar broadcast to every axis, or a per-axis (N,)
        # vector (axis order matches theta). ``m_diag ** alpha`` below
        # broadcasts element-wise either way.
        dp = self.cfg.decoupling_power
        if np.isscalar(dp):
            alpha: float | np.ndarray = float(dp)
        else:
            alpha = np.asarray(dp, dtype=np.float64)
            if alpha.shape != (N,):
                raise ValueError(
                    f"decoupling_power vector must have shape ({N},) "
                    f"(param_dim), got {alpha.shape}"
                )
        seg_agg = str(self.cfg.coverage_seg_agg)
        param_agg = str(self.cfg.coverage_param_agg)
        if seg_agg not in ("max", "mean"):
            raise ValueError(
                f"coverage_seg_agg must be 'max' or 'mean', got {seg_agg!r}"
            )
        if param_agg not in ("log", "linear"):
            raise ValueError(
                f"coverage_param_agg must be 'log' or 'linear', got {param_agg!r}"
            )
        cost_np = np.zeros((P, G), dtype=np.float32)
        per_param_c_np = np.zeros((P, G, N), dtype=np.float32)
        for p in range(P):
            for g in range(G):
                c_acc = np.zeros(N, dtype=np.float64)
                n_valid = 0
                for s in range(num_seg):
                    if not bool(valid_np[p, g, s]):
                        continue
                    F_seg = M_seg_np[p, g, s]
                    try:
                        inv_diag = np.diag(np.linalg.inv(F_seg + eps * eye_N))
                    except np.linalg.LinAlgError:
                        continue
                    m_diag = np.diag(F_seg)
                    # Schur(F,{i}) / F_ii^α  =  F_ii^{1-α} · (decoupling ratio).
                    # α=1 → pure ratio; α<1 folds in an info gate F_ii^{1-α}.
                    schur_i = 1.0 / np.maximum(inv_diag, 1e-12)
                    val = schur_i / np.maximum(m_diag, eps) ** alpha
                    if seg_agg == "max":
                        np.maximum(c_acc, val, out=c_acc)
                    else:                      # "mean" — average over segments
                        c_acc += val
                        n_valid += 1
                if seg_agg == "mean" and n_valid > 0:
                    c_acc /= n_valid
                per_param_c_np[p, g] = c_acc
                if param_agg == "log":
                    cost_np[p, g] = -float(np.sum(np.log(c_acc + eps_log)))
                else:                          # "linear" — plain sum
                    cost_np[p, g] = -float(np.sum(c_acc))

        cost = torch.from_numpy(cost_np).to(device=device, dtype=torch.float32)

        # Termination discount (same convention as the other objectives:
        # multiply a terminated group's cost toward zero = worse).
        if self.cfg.termination_discount != 1.0 and self._group_terminated.any():
            d = self.cfg.termination_discount
            discount = torch.where(
                self._group_terminated,
                torch.full_like(cost, d),
                torch.ones_like(cost),
            )
            cost = cost * discount

        diagnostics: dict[str, Any] = {
            "decoupling_per_param_c": torch.from_numpy(per_param_c_np).to(
                device=device,
            ),
        }
        return cost, diagnostics
