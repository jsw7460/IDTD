import numpy as np
import cma
import matplotlib.pyplot as plt
import matplotlib

# Embed fonts as TrueType (Type 42) instead of Type 3 for IEEE compliance
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
THETA_TRUE  = 2.0
PHI_TRUE    = 1.0
SIGMA       = 1.5
T           = 100
N_ITER      = 20
THETA_INIT  = 0.0    # wrong initial theta
PHI_INIT    = 5.0   # wrong initial phi
BATCH_STEP  = 1      # add 5 timesteps per iteration
N_SEEDS     = 5     # independent repetitions for the 95% CI band
BASE_SEED   = 43

# ─────────────────────────────────────────────
# Simulator (black-box, deterministic)
# ─────────────────────────────────────────────
def simulator(x_t, u_t, xdot_t, theta, phi):
    return x_t + theta * u_t - phi * xdot_t

# ─────────────────────────────────────────────
# Real world (with noise)
# ─────────────────────────────────────────────
def generate_data(u_seq, xdot_seq, rng):
    x = np.zeros(T + 1)
    w = rng.normal(0, SIGMA, T)
    for t in range(T):
        x[t+1] = x[t] + THETA_TRUE * u_seq[t] - PHI_TRUE * xdot_seq[t] + w[t]
    return x

# ─────────────────────────────────────────────
# SysID loss (batch)
# ─────────────────────────────────────────────
def sysid_loss_batch(param, fixed_param, x, u_seq, xdot_seq, optimize, t_end):
    total = 0.0
    for t in range(t_end):
        if optimize == 'theta':
            theta, phi = param, fixed_param
        else:
            theta, phi = fixed_param, param
        pred = simulator(x[t], u_seq[t], xdot_seq[t], theta, phi)
        total += (x[t+1] - pred) ** 2
    return total

# ─────────────────────────────────────────────
# CMA-ES 1D optimizer
# ─────────────────────────────────────────────
def cmaes_1d(loss_fn, x0, sigma0, lo, hi):
    """CMA-ES for 1D parameter using 2D workaround."""
    def wrapped(x):
        val = float(np.clip(x[0], lo, hi))
        return loss_fn(val)

    res = cma.fmin(
        wrapped, [x0, 0.0], sigma0,
        options={
            'verbose': -9,
            'tolx':    1e-6,
            'tolfun':  1e-6,
            'maxiter': 300,
        }
    )
    return float(np.clip(res[0][0], lo, hi))

# ─────────────────────────────────────────────
# Analytic alternating SysID
# ─────────────────────────────────────────────
def analytic_theta(x, u_seq, xdot_seq, phi_hat):
    r = (x[1:] - x[:-1]) + phi_hat * xdot_seq
    denom = np.sum(u_seq ** 2)
    return np.sum(u_seq * r) / denom if denom > 1e-10 else 0.0

def analytic_phi(x, u_seq, xdot_seq, theta_hat):
    r = (x[1:] - x[:-1]) - theta_hat * u_seq
    denom = np.sum(xdot_seq ** 2)
    return -np.sum(xdot_seq * r) / denom if denom > 1e-10 else 0.0

def run_analytic(x, u_seq, xdot_seq):
    phi_hat = PHI_INIT
    theta_errs, phi_errs = [], []
    for _ in range(N_ITER):
        theta_hat = analytic_theta(x, u_seq, xdot_seq, phi_hat)
        phi_hat   = analytic_phi(x, u_seq, xdot_seq, theta_hat)
        theta_errs.append(abs(theta_hat - THETA_TRUE))
        phi_errs.append(abs(phi_hat - PHI_TRUE))
    return np.array(theta_errs), np.array(phi_errs)

# ─────────────────────────────────────────────
# CMA-ES alternating SysID
# ─────────────────────────────────────────────
def run_cmaes(x, u_seq, xdot_seq, heldout):
    """Alternating 1-D CMA-ES identification on (x, u, xdot).

    ``heldout = (x_H, u_H, xdot_H)`` is a trajectory NEVER seen by the
    optimizer; each iteration's estimate is scored on it (full length)
    to measure whether the fit captured the SYSTEM or just the
    identification trajectory — a coupled excitation can drive the
    in-sample loss to the noise floor with a wrong (theta, phi) pair
    whose combination cancels on that data, and only the held-out loss
    exposes it.
    """
    x_H, u_H, xdot_H = heldout
    theta_hat = THETA_INIT
    phi_hat   = PHI_INIT
    theta_errs = [abs(THETA_INIT - THETA_TRUE)]  # iteration 0
    phi_errs = [abs(PHI_INIT - PHI_TRUE)]  # iteration 0
    # Per-step minimized cost (sum-of-squares / t_end): iteration 0 is
    # the initial guess evaluated on the first data window.
    costs = [sysid_loss_batch(PHI_INIT, THETA_INIT, x, u_seq, xdot_seq,
                              'phi', BATCH_STEP) / BATCH_STEP]
    ho_costs = [sysid_loss_batch(PHI_INIT, THETA_INIT, x_H, u_H, xdot_H,
                                 'phi', T) / T]

    for k in range(N_ITER):
        t_end = min((k + 1) * BATCH_STEP, T)

        # Step 1: fix phi, optimize theta via CMA-ES
        loss_theta = lambda theta: sysid_loss_batch(
            theta, phi_hat, x, u_seq, xdot_seq, 'theta', t_end
        )
        theta_hat = cmaes_1d(loss_theta, x0=theta_hat,
                             sigma0=0.3, lo=THETA_TRUE - 10, hi=THETA_TRUE + 10)

        # Step 2: fix theta, optimize phi via CMA-ES
        loss_phi = lambda phi: sysid_loss_batch(
            phi, theta_hat, x, u_seq, xdot_seq, 'phi', t_end
        )
        phi_hat = cmaes_1d(loss_phi, x0=phi_hat,
                           sigma0=0.3, lo=PHI_TRUE - 10, hi=PHI_TRUE + 10)

        theta_errs.append(abs(theta_hat - THETA_TRUE))
        phi_errs.append(abs(phi_hat - PHI_TRUE))
        costs.append(sysid_loss_batch(phi_hat, theta_hat, x, u_seq,
                                      xdot_seq, 'phi', t_end) / t_end)
        ho_costs.append(sysid_loss_batch(phi_hat, theta_hat, x_H, u_H,
                                         xdot_H, 'phi', T) / T)

    return (np.array(theta_errs), np.array(phi_errs), np.array(costs),
            np.array(ho_costs))

# ─────────────────────────────────────────────
# Trajectories (per-seed excitation noise + process noise)
# ─────────────────────────────────────────────
def make_trajectories(rng):
    u_A    = np.ones(T) + rng.normal(0, 0.1, T)
    xdot_A = np.ones(T) + rng.normal(0, 0.0, T)
    x_A    = generate_data(u_A, xdot_A, rng)

    u_B    = np.array([1.0 if t % 2 == 0 else 0.0 for t in range(T)]) + rng.normal(0, 0.1, T)
    xdot_B = np.array([0.0 if t % 2 == 0 else 1.0 for t in range(T)]) + rng.normal(0, 0.1, T)
    x_B    = generate_data(u_B, xdot_B, rng)

    # Held-out trajectory: independent random excitation on both
    # channels — unlike A the (u, xdot) pair is uncorrelated, unlike B
    # it is not alternating, so it favors neither method. Never used by
    # the optimizer; evaluation only.
    u_H    = rng.uniform(-1.0, 1.0, T)
    xdot_H = rng.uniform(-1.0, 1.0, T)
    x_H    = generate_data(u_H, xdot_H, rng)
    return (x_A, u_A, xdot_A), (x_B, u_B, xdot_B), (x_H, u_H, xdot_H)

# ─────────────────────────────────────────────
# Run: N_SEEDS independent repetitions for the 95% CI
# ─────────────────────────────────────────────
runs = {"theta_A": [], "phi_A": [], "cost_A": [], "ho_A": [],
        "theta_B": [], "phi_B": [], "cost_B": [], "ho_B": []}
for s in range(N_SEEDS):
    rng = np.random.default_rng(BASE_SEED + s)
    traj_A, traj_B, traj_H = make_trajectories(rng)
    print(f"[seed {s + 1}/{N_SEEDS}] CMA-ES on Trajectory A...")
    th_A, ph_A, co_A, ho_A = run_cmaes(*traj_A, heldout=traj_H)
    print(f"[seed {s + 1}/{N_SEEDS}] CMA-ES on Trajectory B...")
    th_B, ph_B, co_B, ho_B = run_cmaes(*traj_B, heldout=traj_H)
    runs["theta_A"].append(th_A)
    runs["phi_A"].append(ph_A)
    runs["cost_A"].append(co_A)
    runs["ho_A"].append(ho_A)
    runs["theta_B"].append(th_B)
    runs["phi_B"].append(ph_B)
    runs["cost_B"].append(co_B)
    runs["ho_B"].append(ho_B)

# (N_SEEDS, N_ITER + 1) per curve.
curves = {k: np.stack(v) for k, v in runs.items()}

# ─────────────────────────────────────────────
# Plot (2-panel publication style)
# ─────────────────────────────────────────────
OURS_COLOR = "#A0C8E8"      # soft pink - IDTD (decoupled excitation)
BASELINE_COLOR = "#FFB5BA"  # soft blue - tr(FIM) (coupled excitation)

FIGSIZE = (12, 4.4)
DPI = 300
LINEWIDTH = 3.0
MARKER_SIZE = 8

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

iters = np.arange(N_ITER + 1)
# Panel order: the loss panel (axR) LEFT, the estimation-error panel
# (axL) RIGHT — the variable names keep their original roles.
fig, (axR, axL) = plt.subplots(1, 2, figsize=FIGSIZE, dpi=150)

# Color = method, linestyle/marker = parameter.
METHODS = [
    ("Coupled", "A", BASELINE_COLOR),
    ("Decoupled",    "B", OURS_COLOR),
]
PARAMS = [
    (r"$\theta$", "theta", "-",  "o"),
    (r"$\phi$",   "phi",   "--", "s"),
]

# ── left: theta + phi estimation error in one panel ──
for mlabel, mkey, col in METHODS:
    for plabel, pkey, ls, mk in PARAMS:
        arr = curves[f"{pkey}_{mkey}"]            # (N_SEEDS, N_ITER + 1)
        mean = arr.mean(axis=0)
        ci = 1.96 * arr.std(axis=0) / np.sqrt(arr.shape[0])
        axL.plot(
            iters, mean, linestyle=ls, marker=mk,
            color=col, linewidth=LINEWIDTH, markersize=MARKER_SIZE,
            markerfacecolor="white", markeredgewidth=1.5,
            markevery=2, label=f"{mlabel}, {plabel}",
        )
        axL.fill_between(iters, mean - ci, mean + ci,
                         color=col, alpha=0.16, linewidth=0)
axL.set_xlabel("Iteration", fontsize=16)
axL.set_ylabel("Estimation error", fontsize=16)
axL.set_title("Parameter estimation error", fontsize=17)
axL.grid(True, alpha=0.4, linestyle="--")
axL.tick_params(labelsize=16)
# In-panel legend (center-right sits in the empty band between the
# coupled plateau and the decoupled curves), matching the loss panel.
axL.legend(fontsize=11, frameon=True, ncol=1, loc="center right")

# ── right: in-sample matching loss AND held-out prediction loss in
# ONE panel — color = method, linestyle/marker = which data. The y
# range is clamped around the plateau region so the held-out gap
# (coupled stuck ~2x above the noise floor) stays visually large; the
# early-iteration transient dips below the bottom are clipped on
# purpose.
COST_STRIDE = 2                    # plot every Nth iteration only
_sl = slice(None, None, COST_STRIDE)
_DATASETS = [
    ("matching", "cost", "--", "o"),
    ("held-out", "ho", "-", "s"),
]
for mlabel, mkey, col in METHODS:
    for dlabel, dkey, ls, mk in _DATASETS:
        arr = curves[f"{dkey}_{mkey}"]
        mean = arr.mean(axis=0)
        ci = 1.96 * arr.std(axis=0) / np.sqrt(arr.shape[0])
        axR.plot(
            iters[_sl], mean[_sl], linestyle=ls, marker=mk, color=col,
            linewidth=LINEWIDTH, markersize=MARKER_SIZE,
            markerfacecolor="white", markeredgewidth=1.5,
            label=f"{mlabel}, {dlabel}",
        )
        axR.fill_between(iters[_sl], np.maximum(mean - ci, 1e-12)[_sl],
                         (mean + ci)[_sl], color=col, alpha=0.16, linewidth=0)
axR.set_yscale("log")
axR.set_ylim(1.0, 30.0)
axR.axhline(SIGMA ** 2, color="gray", linestyle=":", linewidth=1.5)
axR.text(0.98, SIGMA ** 2 * 1.05, r"$\sigma^2$ (noise floor)",
         transform=axR.get_yaxis_transform(), ha="right",
         va="bottom", fontsize=11, color="gray")
axR.set_xlabel("Iteration", fontsize=16)
axR.set_ylabel("per-step cost", fontsize=16)
axR.set_title("Trajectory matching loss", fontsize=17)
axR.grid(True, alpha=0.4, linestyle="--")
axR.tick_params(labelsize=16)
axR.legend(fontsize=11, frameon=True, ncol=1, loc="upper right")

plt.tight_layout()
plt.savefig("./SysID/scripts/examples/toy_sysid_cmaes.png",
            dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.savefig("./SysID/scripts/examples/toy_sysid_cmaes.pdf",
            bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()
print("Done.")