import numpy as np
import cma
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
THETA_TRUE = 2.0  # compliance: force -> displacement
PHI_TRUE = 0.25  # damping: velocity -> displacement reduction
SIGMA = 0.1  # noise std
T = 200  # trajectory length
N_ITER = 20  # alternating SysID iterations
THETA_INIT = 0.5  # wrong initial theta
PHI_INIT = 3.0  # wrong initial phi
BATCH_STEP = 10  # timesteps added per iteration
RNG = np.random.default_rng(42)


# ─────────────────────────────────────────────
# Dynamics
# x_{t+1} = x_t + theta * u_t - phi * xdot_t + w_t
# where xdot_t = x_t - x_{t-1}  (actual velocity from state)
# ─────────────────────────────────────────────
def simulate(u_seq, theta, phi, sigma=0.0, rng=None):
    """
    Simulate dynamics with xdot computed from actual state difference.
    Returns x trajectory and xdot sequence.
    """
    x = np.zeros(T + 1)
    xdot = np.zeros(T)
    noise = rng.normal(0, sigma, T) if (sigma > 0 and rng is not None) else np.zeros(T)

    for t in range(T):
        xdot[t] = x[t] - x[t - 1] if t > 0 else 0.0
        x[t + 1] = x[t] + theta * u_seq[t] - phi * xdot[t] + noise[t]

    return x, xdot


# ─────────────────────────────────────────────
# Real world data collection
# ─────────────────────────────────────────────
def generate_data(u_seq):
    """Collect real-world trajectory with true parameters and noise."""
    return simulate(u_seq, THETA_TRUE, PHI_TRUE, sigma=SIGMA, rng=RNG)


# ─────────────────────────────────────────────
# Simulator (black-box, deterministic)
# Uses same structure but with given (theta, phi)
# ─────────────────────────────────────────────
def simulator_loss(theta, phi, x_real, xdot_real, t_end):
    """
    Compute sum of squared prediction errors up to t_end.
    xdot_real is the actual velocity computed from real trajectory.
    """
    total = 0.0
    for t in range(t_end):
        pred = x_real[t] + theta * u_seq_used[t] - phi * xdot_real[t]
        total += (x_real[t + 1] - pred) ** 2
    return total


# ─────────────────────────────────────────────
# SysID loss (batch) — uses precomputed xdot from real trajectory
# ─────────────────────────────────────────────
def sysid_loss_batch(param, fixed_param, x, xdot, u_seq, optimize, t_end):
    total = 0.0
    for t in range(t_end):
        if optimize == 'theta':
            theta, phi = param, fixed_param
        else:
            theta, phi = fixed_param, param
        pred = x[t] + theta * u_seq[t] - phi * xdot[t]
        total += (x[t + 1] - pred) ** 2
    return total


# ─────────────────────────────────────────────
# CMA-ES 1D optimizer
# ─────────────────────────────────────────────
def cmaes_1d(loss_fn, x0, sigma0, lo, hi):
    def wrapped(x):
        val = float(np.clip(x[0], lo, hi))
        return loss_fn(val)

    res = cma.fmin(
        wrapped, [x0, 0.0], sigma0,
        options={'verbose': -9, 'tolx': 1e-6, 'tolfun': 1e-6, 'maxiter': 300}
    )
    return float(np.clip(res[0][0], lo, hi))


# ─────────────────────────────────────────────
# Analytic alternating SysID (closed-form)
# ─────────────────────────────────────────────
def analytic_theta(x, xdot, u_seq, phi_hat):
    """Fix phi, solve theta in closed form."""
    r = (x[1:] - x[:-1]) + phi_hat * xdot
    denom = np.sum(u_seq ** 2)
    return np.sum(u_seq * r) / denom if denom > 1e-10 else 0.0


def analytic_phi(x, xdot, u_seq, theta_hat):
    """Fix theta, solve phi in closed form."""
    r = (x[1:] - x[:-1]) - theta_hat * u_seq
    denom = np.sum(xdot ** 2)
    return -np.sum(xdot * r) / denom if denom > 1e-10 else 0.0


def run_analytic(x, xdot, u_seq):
    phi_hat = PHI_INIT
    theta_errs, phi_errs = [], []
    for _ in range(N_ITER):
        theta_hat = analytic_theta(x, xdot, u_seq, phi_hat)
        phi_hat = analytic_phi(x, xdot, u_seq, theta_hat)
        theta_errs.append(abs(theta_hat - THETA_TRUE))
        phi_errs.append(abs(phi_hat - PHI_TRUE))
    return np.array(theta_errs), np.array(phi_errs)


# ─────────────────────────────────────────────
# CMA-ES alternating SysID
# ─────────────────────────────────────────────
def run_cmaes(x, xdot, u_seq, label=""):
    theta_hat = THETA_INIT
    phi_hat = PHI_INIT
    theta_errs = [abs(THETA_INIT - THETA_TRUE)]
    phi_errs = [abs(PHI_INIT - PHI_TRUE)]
    # Per-step minimized cost (sum-of-squares / t_end); iteration 0 is
    # the initial guess on the first data window.
    costs = [sysid_loss_batch(PHI_INIT, THETA_INIT, x, xdot, u_seq,
                              'phi', BATCH_STEP) / BATCH_STEP]

    print(f"  iter=0 | theta_err={abs(THETA_INIT - THETA_TRUE):.4f}, phi_err={abs(PHI_INIT - PHI_TRUE):.4f}")

    for k in range(N_ITER):
        t_end = min((k + 1) * BATCH_STEP, T)

        loss_theta = lambda theta: sysid_loss_batch(
            theta, phi_hat, x, xdot, u_seq, 'theta', t_end
        )
        theta_hat = cmaes_1d(loss_theta, x0=theta_hat,
                             sigma0=0.5, lo=0.0, hi=10.0)

        loss_phi = lambda phi: sysid_loss_batch(
            phi, theta_hat, x, xdot, u_seq, 'phi', t_end
        )
        phi_hat = cmaes_1d(loss_phi, x0=phi_hat,
                           sigma0=0.5, lo=0.0, hi=10.0)

        t_err = abs(theta_hat - THETA_TRUE)
        p_err = abs(phi_hat - PHI_TRUE)
        theta_errs.append(t_err)
        phi_errs.append(p_err)
        cost = sysid_loss_batch(phi_hat, theta_hat, x, xdot, u_seq,
                                'phi', t_end) / t_end
        costs.append(cost)
        print(
            f"  iter={k + 1:2d} | theta_hat={theta_hat:.4f} (err={t_err:.4f}), phi_hat={phi_hat:.4f} (err={p_err:.4f}), cost/step={cost:.4f}, t_end={t_end}")

    return np.array(theta_errs), np.array(phi_errs), np.array(costs)


# ─────────────────────────────────────────────
# Fisher Information computation
# I(theta,theta) = (1/sigma^2) * sum(u_t^2)
# I(phi,phi)     = (1/sigma^2) * sum(xdot_t^2)
# I(theta,phi)   = -(1/sigma^2) * sum(u_t * xdot_t)
# ─────────────────────────────────────────────
def compute_fim(u_seq, xdot_seq, sigma):
    Itt = np.sum(u_seq ** 2) / sigma ** 2 * 0.01
    Ipp = np.sum(xdot_seq ** 2) / sigma ** 2 * 0.01
    Itp = -np.sum(u_seq * xdot_seq) / sigma ** 2
    return Itt, Ipp, Itp


# ─────────────────────────────────────────────
# Trajectory design
# Trajectory A: constant force (coupled)
# Trajectory B: alternating force / free motion (decoupled)
# ─────────────────────────────────────────────

# Trajectory A: constant force applied
# -> u_t = 1 always -> xdot grows -> u * xdot large -> coupled
u_A = RNG.uniform(1.0, 1.8, T) + RNG.normal(0, 0.05, T)
x_A, xdot_A = generate_data(u_A)

# Trajectory B: alternating
# odd t: apply force (u=1), even t: no force (u=0) -> free damping
# -> u_t * xdot_t alternates between nonzero and zero -> decoupled
u_B = np.array([1.0 if t % 2 == 0 else 0.6 for t in range(T)]) + RNG.normal(0, 0.05, T)
x_B, xdot_B = generate_data(u_B)

# ─────────────────────────────────────────────
# Print FIM values
# ─────────────────────────────────────────────
Itt_A, Ipp_A, Itp_A = compute_fim(u_A, xdot_A, SIGMA)
Itt_B, Ipp_B, Itp_B = compute_fim(u_B, xdot_B, SIGMA)

print("\n" + "=" * 60)
print("FIM Analysis")
print("=" * 60)
print(f"Traj A | I(θ,θ)={Itt_A:.1f}, I(φ,φ)={Ipp_A:.1f}, I(θ,φ)={Itp_A:.1f}, ratio={abs(Itp_A / Itt_A):.3f}")
print(f"Traj B | I(θ,θ)={Itt_B:.1f}, I(φ,φ)={Ipp_B:.1f}, I(θ,φ)={Itp_B:.1f}, ratio={abs(Itp_B / Itt_B):.3f}")

# ─────────────────────────────────────────────
# Run SysID
# ─────────────────────────────────────────────
print("CMA-ES SysID on Trajectory A...")
c_theta_A, c_phi_A, c_cost_A = run_cmaes(x_A, xdot_A, u_A)

print("CMA-ES SysID on Trajectory B...")
c_theta_B, c_phi_B, c_cost_B = run_cmaes(x_B, xdot_B, u_B)

print("\n" + "=" * 60)
print("Final Summary")
print("=" * 60)
print(f"Traj A | I(θ,θ)={Itt_A:.1f}, I(φ,φ)={Ipp_A:.1f}, I(θ,φ)={Itp_A:.1f}, ratio={abs(Itp_A / Itt_A):.3f}")
print(f"       | final theta_err={c_theta_A[-1]:.4f}, final phi_err={c_phi_A[-1]:.4f}")
print(f"Traj B | I(θ,θ)={Itt_B:.1f}, I(φ,φ)={Ipp_B:.1f}, I(θ,φ)={Itp_B:.1f}, ratio={abs(Itp_B / Itt_B):.3f}")
print(f"       | final theta_err={c_theta_B[-1]:.4f}, final phi_err={c_phi_B[-1]:.4f}")
print("=" * 60)

# ─────────────────────────────────────────────
# Plot (house pastel style, same as the go2/g1 result figures)
# ─────────────────────────────────────────────
OURS_COLOR = "#FFB5BA"      # soft pink — IDTD (decoupled excitation)
BASELINE_COLOR = "#A0C8E8"  # soft blue — tr(FIM) (coupled excitation)


def _darken(hex_color: str, factor: float = 0.7) -> str:
    """Multiply RGB by ``factor`` (0..1) to get a darker edge tone."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r * factor), int(g * factor), int(b * factor),
    )


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

iters_with_init = np.arange(N_ITER + 1)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
fig.patch.set_facecolor("white")

# Color = method, linestyle/marker = parameter.
METHODS = [
    ("tr(FIM)", (c_theta_A, c_phi_A, c_cost_A), BASELINE_COLOR),
    ("IDTD",    (c_theta_B, c_phi_B, c_cost_B), OURS_COLOR),
]
PARAMS = [(r"$\theta$", 0, "-", "o"), (r"$\phi$", 1, "--", "s")]

for mlabel, curves, col in METHODS:
    for plabel, idx, ls, mk in PARAMS:
        axL.plot(
            iters_with_init, curves[idx], linestyle=ls, marker=mk,
            markersize=8, linewidth=2.6, color=col,
            markeredgecolor=_darken(col, 0.6), markeredgewidth=1.0,
            label=f"{mlabel}, {plabel}",
        )
axL.set_xlabel("Iteration")
axL.set_ylabel("Estimation error")
axL.set_title("Parameter estimation error")
axL.legend(ncol=2)

COST_STRIDE = 2                    # plot every Nth iteration only
_sl = slice(None, None, COST_STRIDE)
for mlabel, curves, col in METHODS:
    axR.plot(
        iters_with_init[_sl], curves[2][_sl], linestyle="-", marker="s",
        markersize=8, linewidth=2.6, color=col,
        markeredgecolor=_darken(col, 0.6), markeredgewidth=1.0,
        label=mlabel,
    )
axR.set_yscale("log")
axR.axhline(SIGMA ** 2, color="gray", linestyle=":", linewidth=1.5)
axR.text(0.98, SIGMA ** 2 * 1.15, r"$\sigma^2$ (noise floor)",
         transform=axR.get_yaxis_transform(), ha="right",
         va="bottom", fontsize=10, color="gray")
axR.set_xlabel("Iteration")
axR.set_ylabel("per-step cost")
axR.set_title("Minimized SysID cost")
axR.legend()

plt.tight_layout()
plt.savefig('./SysID/scripts/examples/toy_sysid_physical.png', dpi=150)
plt.close()
print("\nDone. Saved to toy_sysid_physical.png")