"""SysID term configuration — analogous to JaxRLWorld's RewardTermConfig.

Each SysIDTermConfig defines one identifiable physical parameter (or group),
including how to apply it to the simulator. Compose them in a SysIDConfig
dataclass and use ``iter_terms()`` to discover them automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SysIDTermConfig:
    """Configuration for a single system identification parameter term.

    Attributes:
        func: Apply function with signature
              ``(env, values: np.ndarray, env_ids: Tensor, **params) -> None``.
              ``values`` has shape ``(B, dim)``.
        dim: Number of scalar parameters in this term.
        default: Default parameter value(s). Scalar for dim=1, list for dim>1.
        bounds: ``(lower, upper)`` applied element-wise. Scalar bounds are
                broadcast across all dims; list bounds specify per-dim.
        reg_weight: L2 regularization weight for deviation from default.
        params: Extra keyword arguments forwarded to ``func``.
    """
    func: Callable
    dim: int
    default: float | list[float] = 0.0
    bounds: tuple[Any, Any] = (-1.0, 1.0)
    reg_weight: float = 0.01
    params: dict[str, Any] = field(default_factory=dict)
    ground_truth: float | list[float] | None = None
    """Known ground-truth value(s) for this parameter. When set,
    the optimizer logs ``|theta - gt|`` each iteration for tracking
    estimation accuracy. ``None`` means unknown."""

    ground_truth_physical: dict[str, Any] | None = None
    """Optional ground truth expressed in physical quantities (rather
    than in the term's internal optimization coordinates). Used when
    ``func`` applies a reparameterization such as log-Cholesky — the
    user supplies ``{"mass": ..., "com": ..., "inertia": ...}`` and
    ``gt_encoder`` converts it to the length-``dim`` internal vector."""

    gt_encoder: Callable[[dict], Any] | None = None
    """Converts ``ground_truth_physical`` to a length-``dim`` vector in
    the same coordinate space as ``func`` expects. Only consulted when
    ``ground_truth_physical`` is set."""

    decoder: Callable[[Any], dict] | None = None
    """Inverse of ``gt_encoder``: maps an internal vector (e.g. the
    optimizer's best ``theta``) back to a physical-quantity dict for
    logging and downstream consumption. Saved alongside the raw
    ``theta_best`` in ``sysid_result.npz``."""

    category: str = "theta"
    """Free-form group label used by Schur-style FIM objectives to
    partition the Fisher Information Matrix into per-group blocks.

    Binary use (``"theta"``/``"phi"``) drives the legacy ``schur`` /
    ``bi_schur`` objectives. Arbitrary labels (e.g. ``"inertial"``,
    ``"pd"``, ``"friction"``) drive the generalised ``multi_schur``
    objective which computes one conditional Fisher info per group:

        S_i = F_ii − F_{i,~i} · F_{~i,~i}^{-1} · F_{~i,i}

    where ``~i`` is the union of all *other* groups' dims. Default
    ``"theta"`` keeps every term in a single group — combined with
    ``FIMConfig.lambda_off=0`` and ``lambda_phi=0`` that reproduces
    the SPI-Active baseline exactly."""

    delta_param: float | None = None
    """Optional per-term finite-difference step for FIM Jacobian
    estimation. ``None`` (default) → fall back to the global
    ``EnvPartitionConfig.delta_param`` (uniform across all dims, the
    SPI-Active behavior). Set this when the term's natural scale
    differs substantially from the global default — e.g. a 10% relative
    perturbation on a ~0.1 N·m joint friction and on a ~7 kg base mass
    are physically very different events, and using the same absolute
    δ biases the cross-term Jacobian estimate."""


@dataclass
class SysIDConfig:
    """Base class for SysID configurations.

    Define identifiable parameter terms as class attributes of type
    ``SysIDTermConfig``. Use ``iter_terms(config, SysIDTermConfig)``
    to discover them. Set a term to ``None`` to disable it.

    Example::

        @dataclass
        class MyConfig(SysIDConfig):
            base_mass = SysIDTermConfig(func=apply_mass, dim=1, ...)
            base_com  = SysIDTermConfig(func=apply_com,  dim=3, ...)

        # Extend and disable a term:
        @dataclass
        class MyConfig2(MyConfig):
            base_com = None  # disabled
            kp = SysIDTermConfig(func=apply_kp, dim=4, ...)
    """
    pass
