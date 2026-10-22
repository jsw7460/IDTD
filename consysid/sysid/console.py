"""Pretty console logging for SysID optimization.

Follows JaxRLWorld's ConsoleWriter layout:
    ═══ header ═══
    section with key-value rows
    multi-column breakdown
    ═══ footer ═══
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

RST = Style.RESET_ALL
PAD = 35
ANSI_EXTRA = 9  # invisible chars added by a single Fore.X + RST pair
WIDTH = 100


# ── helpers ────────────────────────────────────────────────────────────

def _kv(key: str, val: str, val_color: str = Fore.GREEN) -> str:
    label = f"  {Fore.WHITE}{key}{RST}"
    return f"{label.ljust(PAD + ANSI_EXTRA)}{val_color}{val}{RST}"


def _header(width: int, title: str) -> list[str]:
    return [
        f"{Fore.CYAN}{'=' * width}{RST}",
        f"{Fore.CYAN}|{RST}{title.center(width - 2)}{Fore.CYAN}|{RST}",
        f"{Fore.CYAN}{'=' * width}{RST}",
        "",
    ]


def _footer(width: int) -> str:
    return f"{Fore.CYAN}{'=' * width}{RST}"


def _sign_color(v: float) -> str:
    return Fore.GREEN if v > 0 else Fore.RED if v < 0 else Fore.WHITE


def _format_time(seconds: float) -> str:
    d = int(seconds // 86400)
    seconds %= 86400
    h = int(seconds // 3600)
    seconds %= 3600
    m = int(seconds // 60)
    s = int(seconds % 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# ── public API ─────────────────────────────────────────────────────────

_start_time: float | None = None


def print_header(
    sim_type: str,
    sysid_config_name: str,
    trajectory: str,
    dim: int,
    popsize: int,
    num_clips: int,
    num_envs: int,
    horizon_range: tuple[int, int],
    sigma0: float,
    term_names: list[str],
    term_slices: dict[str, tuple[int, int]],
    theta_0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    wandb_url: str | None = None,
    log_dir: str = "",
) -> None:
    global _start_time
    _start_time = _time.time()

    lines = _header(WIDTH, f" {Fore.GREEN}CMA-ES System Identification{RST} ")

    # ── Run Info ──
    lines.append(f"{Fore.MAGENTA}Run Config:{RST}")
    if log_dir:
        lines.append(_kv("Log Dir", log_dir, Fore.WHITE))
    if wandb_url:
        lines.append(_kv("WandB", wandb_url, Fore.BLUE))
    lines.append(_kv("Simulator", sim_type, Fore.YELLOW))
    lines.append(_kv("SysID Config", sysid_config_name, Fore.YELLOW))
    lines.append(_kv("Trajectory", Path(trajectory).name, Fore.BLUE))
    lines.append(_kv("Horizon Range", f"{horizon_range[0]}-{horizon_range[1]} steps"))
    lines.append(_kv("Clips / Iter", f"{num_clips:,}"))
    lines.append(_kv("Parallel Envs", f"{num_envs:,}"))
    lines.append(_kv("Population Size", str(popsize)))
    lines.append(_kv("Sigma0", str(sigma0)))
    lines.append(_kv("Total Dim", str(dim)))

    # ── Parameter Space ──
    lines.append("")
    lines.append(f"{Fore.MAGENTA}Parameter Space:{RST}")
    for name in term_names:
        s, e = term_slices[name]
        d = e - s
        defaults = theta_0[s:e]
        lo, hi = lower[s:e], upper[s:e]
        label = f"  {Fore.YELLOW}{name}{RST}"
        if d == 1:
            detail = (
                f"dim={d}  "
                f"default={Fore.GREEN}{defaults[0]:.4f}{RST}  "
                f"bounds=[{lo[0]:.2f}, {hi[0]:.2f}]"
            )
        else:
            def_str = ", ".join(f"{v:.3f}" for v in defaults)
            detail = (
                f"dim={d}  "
                f"default=[{Fore.GREEN}{def_str}{RST}]  "
                f"bounds=[{lo[0]:.2f}, {hi[0]:.2f}]"
            )
        lines.append(f"{label.ljust(PAD + ANSI_EXTRA)}{detail}")

    lines.append("")
    lines.append(_footer(WIDTH))
    print("\n".join(lines))


def print_iteration(
    iteration: int,
    costs_arr: np.ndarray,
    pred_costs_arr: np.ndarray,
    reg_costs_arr: np.ndarray,
    best_idx: int,
    best_theta: np.ndarray,
    global_best_cost: float,
    sigma: float,
    term_names: list[str],
    term_slices: dict[str, tuple[int, int]],
    theta_0: np.ndarray,
    trajectory_name: str,
    wandb_url: str | None = None,
    term_costs: dict[str, np.ndarray] | None = None,
    max_iterations: int = 0,
    theta_gt: np.ndarray | None = None,
    num_episodes: int = 0,
    term_decoders: dict | None = None,
    log_dir: str = "",
) -> None:
    best_cost = costs_arr[best_idx]
    mean_cost = costs_arr.mean()
    best_pred = pred_costs_arr[best_idx]
    best_reg = reg_costs_arr[best_idx]

    # ═══ header ═══
    iter_label = f" Iteration {Fore.GREEN}{iteration}/{max_iterations}{RST} "
    lines = _header(WIDTH, iter_label)

    # ── Cost ──
    lines.append(f"{Fore.MAGENTA}Cost:{RST}")
    lines.append(_kv("Best", f"{best_cost:.4f}", Fore.GREEN))
    lines.append(_kv("Mean", f"{mean_cost:.4f}", Fore.YELLOW))
    lines.append(_kv("Global Best", f"{global_best_cost:.4f}", Fore.MAGENTA))
    lines.append(_kv("Pred / Reg", f"{best_pred:.4f}  /  {best_reg:.4f}", Fore.CYAN))
    lines.append(_kv("Sigma", f"{sigma:.4f}", Fore.CYAN))

    # ── Loss Breakdown (multi-column) ──
    if term_costs is not None:
        lines.append("")
        lines.append(f"{Fore.MAGENTA}Loss Breakdown:{RST}")
        items = [(name, per_cand[best_idx]) for name, per_cand in term_costs.items()]
        num_columns = 3
        name_width = 18
        value_width = 10
        for i in range(0, len(items), num_columns):
            row_items = items[i:i + num_columns]
            segments = []
            for name, val in row_items:
                color = Fore.GREEN if val < 1.0 else Fore.YELLOW if val < 10.0 else Fore.RED
                segment = (
                    f"{Fore.CYAN}{name:<{name_width}}{RST} "
                    f"{color}{val:>{value_width}.4f}{RST}"
                )
                segments.append(segment)
            lines.append("  " + "   ".join(segments))

    # ── Theta ──
    lines.append("")
    lines.append(f"{Fore.GREEN}Theta:{RST}")
    for name in term_names:
        s, e = term_slices[name]
        vals = best_theta[s:e]
        defs = theta_0[s:e]
        label = f"  {Fore.YELLOW}{name}{RST}"

        # Format predicted values
        if e - s == 1:
            val_str = f"{Fore.GREEN}{vals[0]:.4f}{RST}"
            def_str = f"{Fore.WHITE}{defs[0]:.4f}{RST}"
        else:
            val_str = "[" + ", ".join(f"{Fore.GREEN}{v:.4f}{RST}" for v in vals) + "]"
            def_str = "[" + ", ".join(f"{Fore.WHITE}{d:.4f}{RST}" for d in defs) + "]"

        lines.append(f"{label.ljust(PAD + ANSI_EXTRA)}predicted  {val_str}")
        lines.append(f"{''.ljust(PAD)}default    {def_str}")

        # Ground truth + error (if available)
        if theta_gt is not None:
            gt_vals = theta_gt[s:e]
            if not np.any(np.isnan(gt_vals)):
                if e - s == 1:
                    err = abs(vals[0] - gt_vals[0])
                    gt_str = f"{Fore.CYAN}{gt_vals[0]:.4f}{RST}"
                    err_str = f"{Fore.RED}{err:.4f}{RST}" if err > 0.1 else f"{Fore.GREEN}{err:.4f}{RST}"
                else:
                    gt_str = "[" + ", ".join(f"{Fore.CYAN}{g:.4f}{RST}" for g in gt_vals) + "]"
                    err = np.linalg.norm(vals - gt_vals)
                    err_str = f"{Fore.RED}{err:.4f}{RST}" if err > 0.1 else f"{Fore.GREEN}{err:.4f}{RST}"
                lines.append(f"{''.ljust(PAD)}GT         {gt_str}  |err| = {err_str}")

        # Physical-space summary for reparameterized terms (e.g. log-Cholesky).
        if term_decoders is not None and term_decoders.get(name) is not None:
            def _fmt_physical(d: dict, color: str) -> str:
                bits = []
                if "mass" in d:
                    bits.append(f"m={color}{float(d['mass']):.4f}{RST}")
                if "com" in d:
                    c = np.asarray(d["com"]).flatten()
                    bits.append(f"r=[{color}{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}{RST}]")
                if "inertia" in d:
                    I = np.asarray(d["inertia"])
                    bits.append(
                        f"diag(I)=[{color}{I[0,0]:.4f}, {I[1,1]:.4f}, {I[2,2]:.4f}{RST}]"
                    )
                return "  ".join(bits)

            try:
                dec = term_decoders[name](vals)
            except Exception:
                dec = None
            try:
                dec_def = term_decoders[name](defs)
            except Exception:
                dec_def = None
            if dec is not None:
                lines.append(f"{''.ljust(PAD)}phys pred  " + _fmt_physical(dec, Fore.GREEN))
                if dec_def is not None:
                    lines.append(
                        f"{''.ljust(PAD)}phys def   " + _fmt_physical(dec_def, Fore.WHITE)
                    )
                # Physical-space GT error, if GT is available
                if theta_gt is not None and not np.any(np.isnan(theta_gt[s:e])):
                    try:
                        gt_dec = term_decoders[name](theta_gt[s:e])
                        err_parts = []
                        if "mass" in dec and "mass" in gt_dec:
                            err_parts.append(
                                f"|dm|={abs(float(dec['mass']) - float(gt_dec['mass'])):.4f}"
                            )
                        if "com" in dec and "com" in gt_dec:
                            dr = np.linalg.norm(
                                np.asarray(dec["com"]) - np.asarray(gt_dec["com"])
                            )
                            err_parts.append(f"|dr|={dr:.4f}")
                        if "inertia" in dec and "inertia" in gt_dec:
                            dI = np.linalg.norm(
                                np.asarray(dec["inertia"]) - np.asarray(gt_dec["inertia"])
                            )
                            err_parts.append(f"|dI|_F={dI:.4f}")
                        if err_parts:
                            lines.append(
                                f"{''.ljust(PAD)}phys GT err "
                                + f"{Fore.CYAN}" + "  ".join(err_parts) + RST
                            )
                    except Exception:
                        pass
            else:
                lines.append(f"{''.ljust(PAD)}GT         {Fore.WHITE}unknown{RST}")

    # ── Summary ──
    lines.append("")
    lines.append(f"{Fore.MAGENTA}Summary:{RST}")
    traj_name = Path(trajectory_name).name
    lines.append(_kv("Dataset", traj_name, Fore.GREEN))
    if num_episodes > 0:
        lines.append(_kv("Dataset Episodes", f"{num_episodes:,}", Fore.GREEN))
    if log_dir:
        lines.append(_kv("Log Dir", log_dir, Fore.WHITE))
    if wandb_url:
        lines.append(_kv("WandB", wandb_url, Fore.CYAN))
    if _start_time is not None:
        elapsed = _time.time() - _start_time
        eta = ""
        if max_iterations > 0 and iteration > 0:
            eta_s = elapsed / iteration * (max_iterations - iteration)
            eta = f" | ETA: {Fore.YELLOW}{_format_time(eta_s)}{RST}"
        lines.append(_kv("Elapsed", f"{_format_time(elapsed)}{eta}", Fore.GREEN))

    lines.append("")
    lines.append(_footer(WIDTH))
    print("\n".join(lines))


def print_convergence(
    iteration: int,
    theta_best: np.ndarray,
    cost_best: float,
    term_names: list[str],
    term_slices: dict[str, tuple[int, int]],
    theta_0: np.ndarray,
    stop_reason: dict,
) -> None:
    lines = _header(WIDTH, f" {Fore.GREEN}Converged after {iteration} iterations{RST} ")

    lines.append(_kv("Best Cost", f"{cost_best:.6f}"))
    lines.append(_kv("Stop Reason", str(stop_reason), Fore.WHITE))

    lines.append("")
    lines.append(f"{Fore.MAGENTA}Final Parameters:{RST}")
    for name in term_names:
        s, e = term_slices[name]
        vals = theta_best[s:e]
        defs = theta_0[s:e]
        deltas = vals - defs

        label = f"  {Fore.YELLOW}{name}{RST}"
        val_str = ", ".join(f"{Fore.GREEN}{v:.4f}{RST}" for v in vals)
        def_str = ", ".join(f"{v:.4f}" for v in defs)
        delta_str = ", ".join(f"{_sign_color(d)}{d:+.4f}{RST}" for d in deltas)

        lines.append(f"{label.ljust(PAD + ANSI_EXTRA)}Theta   = [{val_str}]")
        lines.append(f"{''.ljust(PAD)}Default = [{Fore.WHITE}{def_str}{RST}]")
        lines.append(f"{''.ljust(PAD)}Delta   = [{delta_str}]")

    if _start_time is not None:
        elapsed = _time.time() - _start_time
        lines.append("")
        lines.append(_kv("Total Time", _format_time(elapsed), Fore.CYAN))

    lines.append("")
    lines.append(_footer(WIDTH))
    print("\n".join(lines))