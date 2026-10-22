"""Environment partitioner for finite-difference FIM computation.

Manages the (candidate × group × (1 main + N aux)) env hierarchy used
by active exploration. ``num_candidates=1`` reproduces the
legacy single-candidate serial layout exactly — callers only opt into
candidate-parallel evaluation by constructing with ``num_candidates>1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class EnvPartitionConfig:
    """Configuration for main/aux environment partitioning."""

    num_groups: int = 1
    """Number of independent trajectory groups (parallel FIM evaluations
    **within** a single candidate). When ``theta_prior_mode != 'fixed'``
    the groups also serve as Bayesian / robust BOED θ samples — each
    group's main env then sits at a *different* θ_g drawn from the
    chosen prior, and the FIM is aggregated across groups according to
    ``FIMConfig.cost_aggregation``."""
    delta_param: float = 0.1
    """Finite-difference perturbation magnitude."""
    ksync_steps: int = 5
    """Sync aux states to main every this many physics steps."""

    command_term: str = "velocity"
    """Env command-manager term the primary 3D command channel is
    injected into by :meth:`EnvPartitioner.broadcast_commands`. Must
    match ``BezierCommandConfig.command_term`` (``"velocity"`` for
    locomotion, ``"target_position"`` for the Crazyflie)."""

    # ── BOED: θ prior over the ``num_groups`` main envs ───────────────
    theta_prior_mode: str = "fixed"
    """How each group's main-env θ is chosen for FIM linearization. The
    active exploration objective is sensitive to this choice — when
    ``θ_surrogate`` is far from ``θ_GT`` the FIM gradient at that single
    point can mislead the optimizer (active exploration produces commands that
    look informative under the wrong linearization). Sampling θ across
    a *plausible region* and aggregating FIM is robust optimal
    experimental design (BOED).

    Modes:
      * ``"fixed"``                — every group's main env at
        ``θ_surrogate``. Equivalent to the legacy single-linearization
        SPI-Active behaviour; group axis is pure MC noise reduction.
      * ``"default_to_surrogate"`` — group g's main at
        ``α_g · θ_surrogate + (1−α_g) · θ_default`` with ``α_g ~ U[0,1]``.
        Cheap heuristic — anchors one end at the URDF-nominal default
        (which is at least physically valid) and the other at the Stage
        1 estimate. Useful when identification may have over-/under-shot.
      * ``"gaussian"``             — group g's main at
        ``θ_surrogate + N(0, Σ)`` with diagonal ``Σ`` from
        ``theta_prior_gaussian_std`` (scalar broadcast, or per-term
        dict). Standard Bayesian BOED with hand-tuned uncertainty.
      * ``"explicit"``             — group g's main at
        ``theta_prior_explicit[g]``. ``num_groups``-long list of θ
        vectors for ablation / debugging."""
    theta_prior_seed: int = 42
    """RNG seed for prior sampling. Fixed → deterministic θ_g
    assignment across iterations (the same K θ samples linearise every
    candidate evaluation in a given run). Different seed across runs
    gives different θ realisations."""
    theta_prior_gaussian_std: Any = None
    """``"gaussian"`` mode standard deviation. Either:
      * ``float``       — same σ for every dim,
      * ``dict[str, float]`` — per-term σ (key = term name, value =
        σ broadcast over that term's dims).
    Required when ``theta_prior_mode == "gaussian"``; ignored
    otherwise."""
    theta_prior_explicit: Any = None
    """``"explicit"`` mode θ list. Length must equal ``num_groups``;
    each entry is a ``(param_dim,)`` array (or list) used as the
    main-env θ for that group. Required when
    ``theta_prior_mode == "explicit"``."""


class EnvPartitioner:
    """Manages ``(P candidates × G groups × (1 main + N aux))`` env layout.

    Layout: ``total_envs = num_candidates * num_groups * (param_dim + 1)``

    Candidate ``p``, Group ``g``:
        - main env:  ``p*(G*(N+1)) + g*(N+1)``
        - aux env i: ``p*(G*(N+1)) + g*(N+1) + 1 + i``   for i = 0..N-1

    All envs within the same candidate receive the same per-step command
    (different candidates have different command sequences). Aux env
    ``i`` holds parameter ``i`` perturbed by ``+δ_i`` from the surrogate
    θ — used for finite-difference Jacobian estimation.
    """

    def __init__(
        self,
        config: EnvPartitionConfig,
        param_dim: int,
        device: torch.device,
        delta_vec: np.ndarray | None = None,
        num_candidates: int = 1,
    ):
        """
        Args:
            delta_vec: Optional per-dim finite-difference step, shape
                ``(param_dim,)``. ``None`` → broadcast the scalar
                ``config.delta_param`` across all dims (legacy uniform
                behavior — matches SPI-Active baseline). Provide a
                per-dim vector when terms have different natural scales
                (built via ``SysIDParameters.build_delta_vec``).
            num_candidates: How many CMA-ES candidates to evaluate in
                parallel within a single rollout. ``1`` = legacy serial
                path (one candidate per rollout call). ``>1`` expands
                ``total_envs`` by that factor so a single
                physics/policy pass covers all candidates.
        """
        self.cfg = config
        self.param_dim = param_dim
        self.device = device

        if num_candidates < 1:
            raise ValueError(f"num_candidates must be ≥1, got {num_candidates}")
        self.num_candidates = num_candidates

        if delta_vec is None:
            delta_vec = np.full(param_dim, config.delta_param, dtype=np.float32)
        delta_vec = np.asarray(delta_vec, dtype=np.float32)
        if delta_vec.shape != (param_dim,):
            raise ValueError(
                f"delta_vec shape {delta_vec.shape} != (param_dim={param_dim},)"
            )
        if np.any(delta_vec <= 0.0):
            raise ValueError(
                f"delta_vec must be strictly positive, got min={delta_vec.min()}"
            )
        self.delta_vec = delta_vec

        P = num_candidates
        G = config.num_groups
        N = param_dim
        self._group_size = N + 1  # (1 main + N aux)
        self._candidate_size = G * self._group_size
        total = P * self._candidate_size

        # Flat index tensor, reshaped to the 3-level hierarchy. Indices
        # run contiguous through the env tensor the simulator exposes.
        all_flat = torch.arange(total, device=device)
        self._all_indices = all_flat
        self._cand_group_env_indices = all_flat.reshape(P, G, self._group_size)

        # Main env index per (candidate, group): (P, G)
        self._main_indices = self._cand_group_env_indices[:, :, 0].contiguous()

        # Aux env indices: (P, G, N) — ``_aux_indices[:, :, i]`` gives
        # the env indices of the aux env for parameter ``i`` across all
        # (candidate, group) pairs.
        self._aux_indices_tensor = self._cand_group_env_indices[:, :, 1:].contiguous()

        # Group membership: (P, G, group_size) — full (candidate, group) env list.
        self._group_envs = self._cand_group_env_indices

    @property
    def total_envs(self) -> int:
        return self.num_candidates * self._candidate_size

    @property
    def main_indices(self) -> torch.Tensor:
        """``(P, G)`` main-env indices. Use ``.reshape(-1)`` for a flat
        ``(P*G,)`` view when the downstream op needs a single dim."""
        return self._main_indices

    def aux_indices(self, param_idx: int) -> torch.Tensor:
        """Env indices of aux envs for ``param_idx`` across all
        (candidate, group) pairs. Shape ``(P, G)``."""
        return self._aux_indices_tensor[:, :, param_idx]

    @property
    def group_envs(self) -> torch.Tensor:
        """``(P, G, group_size)`` tensor of env indices."""
        return self._group_envs

    def build_theta_batch(
        self,
        theta_surrogate: np.ndarray,
        theta_default: np.ndarray | None = None,
        term_slices: dict[str, tuple[int, int]] | None = None,
    ) -> np.ndarray:
        """Build per-env parameter array with finite-difference perturbations.

        For every ``(candidate, group)`` cell a main env and ``param_dim``
        aux envs are assigned:
          * main env θ = ``θ_g``,
          * aux env i  = ``θ_g + δ_i · e_i``.

        How ``θ_g`` is chosen across the ``num_groups`` groups is set by
        ``EnvPartitionConfig.theta_prior_mode``:

          * ``"fixed"``                — ``θ_g = θ_surrogate`` for every g
            (legacy SPI-Active behaviour).
          * ``"default_to_surrogate"`` — ``θ_g = α_g θ_surrogate + (1−α_g)
            θ_default`` with ``α_g ~ U[0,1]`` (requires ``theta_default``).
          * ``"gaussian"``             — ``θ_g = θ_surrogate + N(0, Σ)``
            with ``Σ`` from ``theta_prior_gaussian_std`` (scalar or
            per-term dict; per-term dict requires ``term_slices``).
          * ``"explicit"``             — ``θ_g = theta_prior_explicit[g]``
            from the config.

        θ does not vary across candidates — candidates differ in their
        **command sequence** (see ``broadcast_commands``), not in θ.

        Args:
            theta_surrogate: ``(dim,)`` surrogate θ from identification.
            theta_default: ``(dim,)`` default-anchor θ (typically the
                term-wise URDF nominal). Required for
                ``"default_to_surrogate"`` mode; ignored otherwise.
            term_slices: ``{name: (start, end)}`` per-term slice into
                the flat θ vector. Required for ``"gaussian"`` mode when
                ``theta_prior_gaussian_std`` is a dict; ignored
                otherwise. Typically obtained as
                ``SysIDParameters._term_slices``.

        Returns:
            ``(total_envs, dim)`` array.
        """
        cfg = self.cfg
        mode = cfg.theta_prior_mode
        G = cfg.num_groups
        dim = len(theta_surrogate)

        # ── Sample one θ per group ──────────────────────────────────────
        # Shape ``(G, dim)``. Same θ_g is used for every candidate's group g
        # (BOED prior is a property of the run, not of individual CMA-ES
        # candidates).
        rng = np.random.default_rng(cfg.theta_prior_seed)

        theta_per_group: np.ndarray
        if mode == "fixed":
            theta_per_group = np.tile(theta_surrogate, (G, 1)).astype(np.float32)
        elif mode == "default_to_surrogate":
            if theta_default is None:
                raise ValueError(
                    "theta_prior_mode='default_to_surrogate' requires "
                    "theta_default to be passed; got None."
                )
            theta_default = np.asarray(theta_default, dtype=np.float32)
            if theta_default.shape != (dim,):
                raise ValueError(
                    f"theta_default shape {theta_default.shape} != "
                    f"theta_surrogate shape ({dim},)"
                )
            alphas = rng.uniform(0.0, 1.0, size=G).astype(np.float32)
            theta_per_group = (
                alphas[:, None] * theta_surrogate[None, :]
                + (1.0 - alphas)[:, None] * theta_default[None, :]
            ).astype(np.float32)
        elif mode == "gaussian":
            std_spec = cfg.theta_prior_gaussian_std
            if std_spec is None:
                raise ValueError(
                    "theta_prior_mode='gaussian' requires "
                    "theta_prior_gaussian_std to be set."
                )
            sigma = self._resolve_gaussian_std(std_spec, dim, term_slices)
            noise = rng.standard_normal(size=(G, dim)).astype(np.float32)
            theta_per_group = (
                theta_surrogate[None, :] + sigma[None, :] * noise
            ).astype(np.float32)
        elif mode == "explicit":
            ex = cfg.theta_prior_explicit
            if ex is None:
                raise ValueError(
                    "theta_prior_mode='explicit' requires "
                    "theta_prior_explicit to be a list of θ vectors."
                )
            theta_per_group = np.asarray(ex, dtype=np.float32)
            if theta_per_group.shape != (G, dim):
                raise ValueError(
                    f"theta_prior_explicit must have shape "
                    f"(num_groups={G}, dim={dim}), got "
                    f"{theta_per_group.shape}"
                )
        else:
            raise ValueError(
                f"theta_prior_mode must be 'fixed', 'default_to_surrogate', "
                f"'gaussian', or 'explicit'; got {mode!r}"
            )

        # ── Tile (G, dim) → (P, G, group_size, dim) and apply FD perturbation ──
        # Layout: total = P × G × (1 + N). For each (p, g, slot) the row's
        # θ is the group-g vector; slot 0 = main, slot 1..N = aux i with
        # +δ_i on dim i.
        total = self.total_envs
        P = self.num_candidates
        theta_per_group_t = np.broadcast_to(
            theta_per_group[None, :, None, :],
            (P, G, self._group_size, dim),
        ).astype(np.float32, copy=True)

        # Apply FD perturbation on aux slots: aux env i (slot 1+i) gets +δ_i on dim i.
        for i in range(self.param_dim):
            theta_per_group_t[:, :, 1 + i, i] += float(self.delta_vec[i])

        theta_batch = theta_per_group_t.reshape(total, dim)
        if mode != "fixed":
            self._log_prior_summary(theta_per_group, theta_surrogate)
        return theta_batch

    @staticmethod
    def _resolve_gaussian_std(
        spec: Any,
        dim: int,
        term_slices: dict[str, tuple[int, int]] | None,
    ) -> np.ndarray:
        """Resolve ``theta_prior_gaussian_std`` into a ``(dim,)`` σ vector.

        Accepts either a scalar (broadcast across all dims) or a dict
        keyed by term name (requires ``term_slices`` to map names to
        per-dim slices). Missing keys default to 0 (no perturbation on
        those dims) — explicit per-term opt-in is safer than implicit
        broadcast for high-curvature terms like log-Cholesky φ.
        """
        if isinstance(spec, (int, float)):
            return np.full(dim, float(spec), dtype=np.float32)
        if isinstance(spec, dict):
            if term_slices is None:
                raise ValueError(
                    "theta_prior_gaussian_std as dict requires term_slices "
                    "(passed via build_theta_batch); got None."
                )
            sigma = np.zeros(dim, dtype=np.float32)
            for name, val in spec.items():
                if name not in term_slices:
                    raise ValueError(
                        f"theta_prior_gaussian_std key {name!r} not in "
                        f"term_slices {list(term_slices)}"
                    )
                start, end = term_slices[name]
                sigma[start:end] = float(val)
            return sigma
        raise TypeError(
            f"theta_prior_gaussian_std must be float or dict, got "
            f"{type(spec).__name__}"
        )

    def _log_prior_summary(
        self,
        theta_per_group: np.ndarray,
        theta_surrogate: np.ndarray,
    ) -> None:
        """One-shot console print summarising the per-group θ spread.

        Helps verify that the prior is doing what the user expects:
        ``fixed`` mode never prints; non-fixed modes print min/max/range
        per dim across the K group samples so an outlier σ or wrong
        anchor stands out immediately.
        """
        # Avoid spamming if called every iter — gate on attribute that we
        # set on first call.
        if getattr(self, "_prior_logged", False):
            return
        self._prior_logged = True

        lo = theta_per_group.min(axis=0)
        hi = theta_per_group.max(axis=0)
        spread = hi - lo
        print(
            f"\n[EnvPartitioner] theta_prior_mode="
            f"{self.cfg.theta_prior_mode!r} — sampled θ across "
            f"{self.cfg.num_groups} groups\n"
            f"  surrogate min: {theta_surrogate.min():.4f} "
            f"max: {theta_surrogate.max():.4f}\n"
            f"  group θ_g min:  {lo.min():.4f}   "
            f"group θ_g max:  {hi.max():.4f}\n"
            f"  per-dim spread (max - min) summary: "
            f"min={spread.min():.4f}  median={np.median(spread):.4f}  "
            f"max={spread.max():.4f}\n"
        )

    def broadcast_commands(
        self,
        env: Any,
        velocity_cmds: torch.Tensor,
        gait_cmds: torch.Tensor | None,
    ) -> None:
        """Inject per-candidate commands to every env of that candidate.

        Within a single candidate all ``G*(N+1)`` envs share the same
        command — FIM cares about ``(∂s/∂θ)`` **at fixed command**, so
        main and aux must be driven identically. Across candidates the
        commands differ because that is what CMA-ES is searching over.

        Args:
            env: JaxRLWorld environment.
            velocity_cmds: ``(num_candidates, 3)`` tensor.
            gait_cmds: ``(num_candidates, 11)`` tensor, or ``None`` for
                velocity-only robots (no gait command term).
        """
        P = self.num_candidates
        if velocity_cmds.shape[0] != P:
            raise ValueError(
                f"broadcast_commands expected leading dim = num_candidates={P}, "
                f"got velocity {tuple(velocity_cmds.shape)}"
            )
        if gait_cmds is not None and gait_cmds.shape[0] != P:
            raise ValueError(
                f"broadcast_commands expected gait leading dim = {P}, "
                f"got {tuple(gait_cmds.shape)}"
            )

        # Each candidate's row is replicated across its G*(N+1) envs.
        cmd = {
            self.cfg.command_term: velocity_cmds.repeat_interleave(
                self._candidate_size, dim=0,
            ),
        }
        if gait_cmds is not None:
            cmd["gait"] = gait_cmds.repeat_interleave(self._candidate_size, dim=0)

        env.command_manager.set_commands(
            env_ids=self._all_indices, **cmd,
        )

    def handle_grouped_resets(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine which ``(candidate, group)`` pairs need full reset.

        If any env within a ``(p, g)`` group terminates/truncates, that
        entire group must be reset to maintain finite-difference
        validity. Termination propagates only within a ``(p, g)`` pair
        — different candidates are always independent.

        Args:
            terminated: (total_envs,) bool tensor.
            truncated: (total_envs,) bool tensor.

        Returns:
            group_reset_mask: ``(P, G)`` bool tensor — which
                ``(candidate, group)`` pairs need reset.
            expanded_reset_ids: ``(K,)`` int env indices to reset.
        """
        reset = terminated | truncated  # (total_envs,)
        P = self.num_candidates
        G = self.cfg.num_groups
        gs = self._group_size

        reset_per_group = reset.reshape(P, G, gs).any(dim=-1)  # (P, G)

        # Expand back to all envs in reset groups: (P, G, gs) → (total_envs,)
        expanded = reset_per_group.unsqueeze(-1).expand(-1, -1, gs).reshape(-1)
        expanded_reset_ids = expanded.nonzero(as_tuple=False).flatten()

        return reset_per_group, expanded_reset_ids
