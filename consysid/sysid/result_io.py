"""Load-side helpers for ``sysid_result.npz`` — handles term-order remapping.

The saved ``theta_best`` vector is laid out in the term order produced
by ``iter_terms`` on the ``SysIDConfig`` used for that run. That order
depends on where in the MRO each term was declared (``iter_terms``
walks leaf→base and preserves dict insertion order), so two configs
with the same terms can produce different orderings if they declare
them at different positions in the inheritance chain. A raw-vector
load would then silently assign values to the wrong slots.

Saves include ``term_names`` and ``term_dims``; loads look those up
and rebuild the vector term-by-term into the *current* config's slots.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def atomic_savez(path: Path, data: dict) -> None:
    """Write an npz file atomically so live readers never see half-written state.

    Standard tmp-file + ``os.replace`` dance — ``os.replace`` is an atomic
    rename on POSIX, so a consumer opening the ``*_latest.npz`` always
    sees either the previous complete snapshot or the new one, never a
    torn intermediate.

    ``np.savez`` auto-appends ``".npz"`` when given a string path, which
    would turn ``"foo.npz.tmp"`` into ``"foo.npz.tmp.npz"`` and break
    the rename — passing an open file object suppresses that.

    Shared by identification (``sysid_result_latest.npz``) and active exploration
    (``active_commands_latest.npz``) checkpointers so both pipelines
    use the same crash-safe write convention.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **data)
    os.replace(tmp, path)


def load_theta_best(path: str, params: Any) -> np.ndarray:
    """Load ``theta_best`` from a sysid result npz, remapped to ``params``.

    Args:
        path: Path to a ``sysid_result.npz`` (or ``_latest``/``_iter*``).
        params: The current run's ``SysIDParameters`` — defines the
            target term ordering via ``params.terms`` / ``params._term_slices``.

    Returns:
        ``(params.dim,)`` float32 vector with each term pulled from the
        saved vector by name.

    Raises:
        ValueError: term sets disagree, per-term dim disagrees, or
            (legacy files without metadata) the raw vector length is
            inconsistent with the current config.
    """
    data = np.load(path, allow_pickle=True)
    theta_raw = data["theta_best"].astype(np.float32)

    # Legacy files saved before term metadata was added. We can't
    # remap without a layout, so we at least assert total-dim
    # consistency and return the vector verbatim.
    if "term_names" not in data.files:
        if theta_raw.shape != (params.dim,):
            raise ValueError(
                f"legacy sysid_result at {path} has shape {theta_raw.shape} "
                f"but current config dim is {params.dim} — cannot infer layout"
            )
        return theta_raw

    saved_names = [str(n) for n in data["term_names"]]
    if "term_dims" in data.files:
        saved_dims = [int(d) for d in data["term_dims"]]
    else:
        # term_names was saved but term_dims wasn't (intermediate schema).
        # Fall back to the current config's dims keyed by name — still
        # validates the total-length consistency below.
        saved_dims = [params.terms[n].dim for n in saved_names]

    saved_total = sum(saved_dims)
    if saved_total != theta_raw.shape[0]:
        raise ValueError(
            f"sysid_result at {path}: term_dims sum {saved_total} != "
            f"theta_best length {theta_raw.shape[0]}"
        )

    current_names = list(params.terms.keys())
    if set(saved_names) != set(current_names):
        only_saved = sorted(set(saved_names) - set(current_names))
        only_current = sorted(set(current_names) - set(saved_names))
        raise ValueError(
            f"sysid_result at {path} has different terms than current config:\n"
            f"  only in saved:   {only_saved}\n"
            f"  only in current: {only_current}"
        )

    saved_slice: dict[str, tuple[int, int]] = {}
    off = 0
    for n, d in zip(saved_names, saved_dims):
        saved_slice[n] = (off, off + d)
        off += d

    remapped = np.empty(params.dim, dtype=np.float32)
    for n, (c_lo, c_hi) in params._term_slices.items():
        s_lo, s_hi = saved_slice[n]
        saved_d = s_hi - s_lo
        current_d = c_hi - c_lo
        if saved_d != current_d:
            raise ValueError(
                f"term {n!r} dim mismatch: saved={saved_d} vs current={current_d}"
            )
        remapped[c_lo:c_hi] = theta_raw[s_lo:s_hi]

    if saved_names != current_names:
        print(
            f"[load_theta_best] term order differs between saved and current "
            f"config — remapping:\n"
            f"  saved:   {saved_names}\n"
            f"  current: {current_names}"
        )

    return remapped
