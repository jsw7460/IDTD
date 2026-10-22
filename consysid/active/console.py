"""Pretty console logging for active exploration.

Follows JaxRLWorld's ConsoleWriter layout with sections for
FIM metrics, rollout stats, and command optimization progress.
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

RST = Style.RESET_ALL
PAD = 35
ANSI_EXTRA = 9
WIDTH = 100


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


_start_time: float | None = None


def print_active_header(
    sim_type: str,
    sysid_config_name: str,
    bezier_dim: int,
    popsize: int,
    num_groups: int,
    param_dim: int,
    total_envs: int,
    num_segments: int,
    segment_duration: float,
    bezier_degree: int,
    optimized_commands: dict[str, tuple[float, float]],
    delta_param: float,
    ksync_steps: int,
    sigma0: float,
    theta_surrogate: np.ndarray,
    wandb_url: str | None = None,
    log_dir: str = "",
) -> None:
    global _start_time
    _start_time = _time.time()

    lines = _header(WIDTH, f" {Fore.GREEN}Active Exploration (active exploration){RST} ")

    lines.append(f"{Fore.MAGENTA}Run Config:{RST}")
    if log_dir:
        lines.append(_kv("Log Dir", log_dir, Fore.WHITE))
    if wandb_url:
        lines.append(_kv("WandB", wandb_url, Fore.BLUE))
    lines.append(_kv("Simulator", sim_type, Fore.YELLOW))
    lines.append(_kv("SysID Config", sysid_config_name, Fore.YELLOW))
    lines.append(_kv("Total Envs", f"{total_envs} ({num_groups} groups x {param_dim + 1} envs/group)"))
    lines.append(_kv("Population Size", str(popsize)))
    lines.append(_kv("Sigma0", str(sigma0)))
    lines.append(_kv("CMA-ES Dim", str(bezier_dim)))

    lines.append("")
    lines.append(f"{Fore.MAGENTA}Bezier Config:{RST}")
    lines.append(_kv("Segments", f"{num_segments} x {segment_duration}s = {num_segments * segment_duration:.0f}s"))
    lines.append(_kv("Bezier Degree", f"{bezier_degree} ({bezier_degree + 1} control points)"))
    lines.append(_kv("Delta Param", str(delta_param)))
    lines.append(_kv("KSync Steps", str(ksync_steps)))

    lines.append("")
    lines.append(f"{Fore.MAGENTA}Optimized Commands:{RST}")
    for name, (lo, hi) in optimized_commands.items():
        label = f"  {Fore.YELLOW}{name}{RST}"
        lines.append(f"{label.ljust(PAD + ANSI_EXTRA)}bounds=[{lo:.2f}, {hi:.2f}]")

    lines.append("")
    lines.append(f"{Fore.MAGENTA}Surrogate Theta:{RST}")
    theta_str = ", ".join(f"{v:.4f}" for v in theta_surrogate)
    lines.append(f"  {Fore.GREEN}[{theta_str}]{RST}")

    lines.append("")
    lines.append(_footer(WIDTH))
    print("\n".join(lines))


def print_active_iteration(
    iteration: int,
    max_iterations: int,
    fim_best: float,
    fim_mean: float,
    fim_global_best: float,
    sigma: float,
    num_terminated: int,
    num_groups: int,
    rollout_time: float,
    mean_terminated: float = 0.0,
    best_terminated: int = 0,
    wandb_url: str | None = None,
    log_dir: str = "",
) -> None:
    iter_label = f" Iteration {Fore.GREEN}{iteration}/{max_iterations}{RST} "
    lines = _header(WIDTH, iter_label)

    # FIM section
    lines.append(f"{Fore.GREEN}Fisher Information:{RST}")
    fim_color = Fore.GREEN if fim_best > 0 else Fore.RED
    lines.append(_kv("Best tr(FIM)", f"{fim_best:.6f}", fim_color))
    lines.append(_kv("Mean tr(FIM)", f"{fim_mean:.6f}", Fore.YELLOW))
    lines.append(_kv("Global Best", f"{fim_global_best:.6f}", Fore.MAGENTA))

    # CMA-ES section
    lines.append("")
    lines.append(f"{Fore.CYAN}CMA-ES:{RST}")
    lines.append(_kv("Sigma", f"{sigma:.6f}", Fore.CYAN))

    # Rollout stats — terminated groups from three angles. ``best`` is the
    # converging solution (the one to watch); ``worst`` is a CMA-ES tail
    # sample and stays near num_groups regardless, so it's de-emphasised.
    lines.append("")
    lines.append(f"{Fore.YELLOW}Rollout Stats:{RST}")
    lines.append(_kv("Groups Term. (best)", f"{best_terminated}/{num_groups}",
                      Fore.GREEN if best_terminated == 0 else Fore.RED))
    lines.append(_kv("Groups Term. (mean)", f"{mean_terminated:.1f}/{num_groups}",
                      Fore.YELLOW))
    lines.append(_kv("Groups Term. (worst)", f"{num_terminated}/{num_groups}",
                      Fore.WHITE))
    lines.append(_kv("Rollout Time", f"{rollout_time:.2f}s", Fore.CYAN))

    # Summary
    lines.append("")
    lines.append(f"{Fore.WHITE}Summary:{RST}")
    if log_dir:
        lines.append(_kv("Log Dir", log_dir, Fore.WHITE))
    if wandb_url:
        lines.append(_kv("WandB", wandb_url, Fore.BLUE))
    if _start_time is not None:
        elapsed = _time.time() - _start_time
        eta = ""
        if max_iterations > 0 and iteration > 0:
            eta_s = elapsed / iteration * (max_iterations - iteration)
            eta = f" | ETA: {Fore.YELLOW}{_format_time(eta_s)}{RST}"
        lines.append(_kv("Elapsed", f"{_format_time(elapsed)}{eta}", Fore.CYAN))

    lines.append("")
    lines.append(_footer(WIDTH))
    print("\n".join(lines))


def print_active_rollout_step(
    step: int,
    total_steps: int,
    num_terminated: int,
    num_groups: int,
) -> None:
    """Inline progress indicator during rollout (overwrites same line)."""
    alive = num_groups - num_terminated
    pct = (step + 1) / total_steps * 100
    print(
        f"\r  rollout step {step + 1}/{total_steps} ({pct:.0f}%) "
        f"| alive: {alive}/{num_groups}",
        end="", flush=True,
    )


def print_active_rollout_done() -> None:
    """Clear the inline progress line."""
    print("\r" + " " * 80 + "\r", end="", flush=True)


def print_active_convergence(
    iteration: int,
    best_fim: float,
    stop_reason: dict,
) -> None:
    lines = _header(WIDTH, f" {Fore.GREEN}Active Exploration Complete ({iteration} iterations){RST} ")

    lines.append(_kv("Best tr(FIM)", f"{best_fim:.6f}"))
    lines.append(_kv("Stop Reason", str(stop_reason), Fore.WHITE))

    if _start_time is not None:
        elapsed = _time.time() - _start_time
        lines.append(_kv("Total Time", _format_time(elapsed), Fore.CYAN))

    lines.append("")
    lines.append(_footer(WIDTH))
    print("\n".join(lines))
