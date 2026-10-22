"""Log-Cholesky parameterization of the pseudo-inertia matrix.

Reference: Sobanbabu et al. 2025 (SPI-Active), Eq. 2.

The physical inertial parameters of a rigid body — mass ``m``, CoM ``r``,
and symmetric inertia tensor ``I`` (CoM-centered) — are packed into the
4x4 pseudo-inertia matrix

    J = [[ Σ,  h ],
         [ h^T, m ]],

where ``h = m·r`` and ``Σ = ½·tr(I)·I₃ − I``. Physical validity requires
``J ≻ 0`` (symmetric positive definite). To turn the constrained
estimation problem into an unconstrained one, we factor ``J = L·Lᵀ``
(lower-Cholesky) and optimize over the Cholesky factors in log-space.

Parameter vector ``φ ∈ ℝ¹⁰`` (used throughout the SysID pipeline):

    φ = [α, d1, d2, d3, s12, s13, s23, t1, t2, t3]

with the convention

    L[0,0] = exp(α + d1)       L[1,0] = exp(α) · s12
    L[1,1] = exp(α + d2)       L[2,0] = exp(α) · s13
    L[2,2] = exp(α + d3)       L[2,1] = exp(α) · s23
    L[3,3] = exp(α)            L[3,0] = exp(α) · t1
                               L[3,1] = exp(α) · t2
                               L[3,2] = exp(α) · t3

(All other entries are 0.) Under this convention every ``φ ∈ ℝ¹⁰``
decodes to a physically valid ``(m, r, I)`` triple, and the map is a
bijection onto the interior of the SPD cone of pseudo-inertia matrices.

Note: the SPI-Active paper writes the factorization as ``J = U·Uᵀ`` with
upper-triangular U. Lower- and upper-Cholesky are mathematically
equivalent parameterizations; we use the lower convention because it
matches ``numpy.linalg.cholesky`` / ``torch.linalg.cholesky`` natively.
Component semantics (α = log scale, d_i = diagonal shape, s_ij = off-
diagonals, t_i = CoM moment) are preserved.
"""
from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Batched decode: CMA-ES sample vectors → physical (m, r, I) for all envs.
# ---------------------------------------------------------------------------


def decode_phi_batched(
    phi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode ``(B, 10)`` log-Cholesky vectors to physical inertial params.

    Args:
        phi: ``(B, 10)`` tensor of log-Cholesky coordinates.

    Returns:
        ``(mass, com, inertia)`` with shapes ``(B,)``, ``(B, 3)``,
        ``(B, 3, 3)`` on the same device/dtype as ``phi``. The returned
        inertia tensors are symmetric positive definite by construction.
    """
    if phi.ndim != 2 or phi.shape[1] != 10:
        raise ValueError(f"phi must have shape (B, 10), got {tuple(phi.shape)}")

    B = phi.shape[0]
    device = phi.device
    dtype = phi.dtype

    alpha = phi[:, 0]
    d1, d2, d3 = phi[:, 1], phi[:, 2], phi[:, 3]
    s12, s13, s23 = phi[:, 4], phi[:, 5], phi[:, 6]
    t1, t2, t3 = phi[:, 7], phi[:, 8], phi[:, 9]

    e_alpha = torch.exp(alpha)

    L = torch.zeros(B, 4, 4, dtype=dtype, device=device)
    L[:, 0, 0] = torch.exp(alpha + d1)
    L[:, 1, 1] = torch.exp(alpha + d2)
    L[:, 2, 2] = torch.exp(alpha + d3)
    L[:, 3, 3] = e_alpha
    L[:, 1, 0] = e_alpha * s12
    L[:, 2, 0] = e_alpha * s13
    L[:, 2, 1] = e_alpha * s23
    L[:, 3, 0] = e_alpha * t1
    L[:, 3, 1] = e_alpha * t2
    L[:, 3, 2] = e_alpha * t3

    # J = L Lᵀ is symmetric positive definite by construction.
    J = L @ L.transpose(-1, -2)

    mass = J[:, 3, 3]                                  # (B,)
    h = J[:, :3, 3]                                    # (B, 3)
    com = h / mass.unsqueeze(-1)                       # (B, 3)

    Sigma = J[:, :3, :3]                               # (B, 3, 3) — origin frame
    # J packs the ORIGIN-frame second moment Σ_origin = Σ_com + m·r·rᵀ
    # (see encode_physical_to_phi). Shift back to the CoM frame first, then
    # invert Σ_com = ½·tr(I)·I₃ − I via tr(I) = 2·tr(Σ_com) ⇒
    # I = tr(Σ_com)·I₃ − Σ_com.
    eye3 = torch.eye(3, dtype=dtype, device=device).expand(B, 3, 3)
    outer_com = com.unsqueeze(-1) * com.unsqueeze(-2)        # (B, 3, 3) r·rᵀ
    Sigma_com = Sigma - mass.view(B, 1, 1) * outer_com
    trace_Sigma = Sigma_com.diagonal(dim1=-2, dim2=-1).sum(-1)   # (B,)
    inertia = trace_Sigma.view(B, 1, 1) * eye3 - Sigma_com

    # Symmetrize to kill any tiny floating-point asymmetry before handing
    # off to the solver (eig3 expects exactly symmetric input).
    inertia = 0.5 * (inertia + inertia.transpose(-1, -2))

    return mass, com, inertia


# ---------------------------------------------------------------------------
# Single-vector encode/decode (numpy). Used for defaults, ground truth,
# and saving results to disk.
# ---------------------------------------------------------------------------


def encode_physical_to_phi(
    mass: float,
    com: np.ndarray,
    inertia: np.ndarray,
) -> np.ndarray:
    """Encode physical ``(m, r, I)`` to the 10-D log-Cholesky vector ``φ``.

    Args:
        mass: Scalar mass [kg], must be positive.
        com: ``(3,)`` CoM position [m] relative to the body frame origin.
        inertia: ``(3, 3)`` symmetric positive-definite inertia tensor
            [kg·m²], expressed at the CoM in the body frame.

    Returns:
        ``(10,)`` numpy array ``[α, d1, d2, d3, s12, s13, s23, t1, t2, t3]``
        such that :func:`decode_phi_batched` round-trips to the same
        physical quantities (up to floating point).
    """
    mass_f = float(mass)
    if mass_f <= 0.0:
        raise ValueError(f"mass must be positive, got {mass_f}")

    com_arr = np.asarray(com, dtype=np.float64).reshape(3)
    I = np.asarray(inertia, dtype=np.float64).reshape(3, 3)
    I = 0.5 * (I + I.T)  # symmetrize defensively

    h = mass_f * com_arr
    # Origin-frame second moment Σ_origin = Σ_com + m·r·rᵀ. The CoM-frame
    # Σ = ½·tr(I)·I₃ − I alone mixes frames with the origin-frame h and
    # makes the 4x4 pseudo-inertia NOT SPD once the CoM sits far from the
    # body origin (Go2 trunk CoM ≈ 0.005 m slips through; the G1 pelvis at
    # z = −0.076 m fails). Adding the parallel-axis term restores genuine
    # SPD-ness for any valid body; decode subtracts it back out.
    Sigma = 0.5 * np.trace(I) * np.eye(3) - I + mass_f * np.outer(com_arr, com_arr)

    J = np.zeros((4, 4), dtype=np.float64)
    J[:3, :3] = Sigma
    J[:3, 3] = h
    J[3, :3] = h
    J[3, 3] = mass_f

    # Check positive definiteness before factoring for a clearer error.
    eigvals = np.linalg.eigvalsh(J)
    if np.min(eigvals) <= 0.0:
        raise ValueError(
            f"pseudo-inertia matrix is not positive definite "
            f"(min eigenvalue={np.min(eigvals):.3e}); check (m, r, I) inputs"
        )

    L = np.linalg.cholesky(J)  # lower, J = L Lᵀ

    alpha = float(np.log(L[3, 3]))
    e_alpha = float(L[3, 3])

    d1 = float(np.log(L[0, 0])) - alpha
    d2 = float(np.log(L[1, 1])) - alpha
    d3 = float(np.log(L[2, 2])) - alpha
    s12 = float(L[1, 0]) / e_alpha
    s13 = float(L[2, 0]) / e_alpha
    s23 = float(L[2, 1]) / e_alpha
    t1 = float(L[3, 0]) / e_alpha
    t2 = float(L[3, 1]) / e_alpha
    t3 = float(L[3, 2]) / e_alpha

    return np.array(
        [alpha, d1, d2, d3, s12, s13, s23, t1, t2, t3],
        dtype=np.float64,
    )


def decode_phi_to_physical_dict(phi: np.ndarray) -> dict:
    """Decode a single ``(10,)`` ``φ`` vector to a physical-quantity dict.

    Used for logging and saving results; external consumers see ``(m, r, I)``
    directly rather than the internal optimization coordinates.

    Returns:
        Dictionary with keys ``"mass"`` (float), ``"com"`` (``(3,)`` array),
        and ``"inertia"`` (``(3, 3)`` symmetric array).
    """
    phi_t = torch.as_tensor(phi, dtype=torch.float64).reshape(1, 10)
    mass, com, inertia = decode_phi_batched(phi_t)
    return {
        "mass": float(mass[0].item()),
        "com": com[0].cpu().numpy().astype(np.float64),
        "inertia": inertia[0].cpu().numpy().astype(np.float64),
    }


# ---------------------------------------------------------------------------
# MuJoCo/Newton interop: convert principal-axis (diaginertia, quat) to the
# full 3x3 body-frame symmetric tensor that Newton stores in body_inertia.
# ---------------------------------------------------------------------------


def mujoco_inertial_to_body_inertia(
    diaginertia: np.ndarray,
    quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Rebuild ``I_body`` from MuJoCo ``<inertial>`` attributes.

    MuJoCo's ``<inertial>`` element stores the inertia tensor in a
    principal-axis frame: ``diaginertia`` gives the three principal moments
    and ``quat`` rotates the principal frame into the body frame. This
    helper produces the equivalent 3x3 symmetric tensor in body coordinates
    so it can be fed to :func:`encode_physical_to_phi`.

    Args:
        diaginertia: ``(3,)`` principal moments [kg·m²], strictly positive.
        quat_wxyz: ``(4,)`` unit quaternion in MuJoCo's ``(w, x, y, z)``
            convention rotating the principal frame into the body frame.

    Returns:
        ``(3, 3)`` symmetric positive-definite inertia tensor expressed in
        the body frame at the CoM.
    """
    diag = np.asarray(diaginertia, dtype=np.float64).reshape(3)
    if np.any(diag <= 0.0):
        raise ValueError(f"diaginertia must be positive, got {diag}")

    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    # Standard wxyz → rotation matrix.
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )

    I_body = R @ np.diag(diag) @ R.T
    return 0.5 * (I_body + I_body.T)
