"""Generic, term-driven SysID parameter manager.

Discovers ``SysIDTermConfig`` attributes on a ``SysIDConfig`` dataclass,
auto-computes DIM / theta_0 / bounds, and delegates ``apply_to_env`` to
each term's apply function.  No simulator-specific subclasses needed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from jaxrlworld.rl.configs.base_config import iter_terms

from consysid.configs.sysid_term_config import SysIDConfig, SysIDTermConfig


def build_per_param_vector(
    config: SysIDConfig, per_term: dict[str, float], fallback: float
) -> np.ndarray:
    """Expand a ``{term_name: value}`` dict into a ``(param_dim,)`` vector.

    The vector is laid out in the canonical term order (identical to the
    one :class:`SysIDParameters` uses for ``theta`` / the FIM axes, since
    both come from ``iter_terms``), so it can index parameter axes
    directly. Terms absent from ``per_term`` receive ``fallback``.

    Env-free: only reads the config's static term dims. Used e.g. to
    build the per-parameter ``decoupling_power`` (α) vector for the
    decoupling-coverage objective from a per-term α dict.
    """
    terms = iter_terms(config, SysIDTermConfig)
    if not terms:
        raise ValueError(f"{type(config).__name__} declares no SysID terms")
    unknown = set(per_term) - set(terms)
    if unknown:
        raise ValueError(
            f"unknown term(s) {sorted(unknown)}; "
            f"valid term names: {sorted(terms)}"
        )
    dim = sum(term.dim for term in terms.values())
    vec = np.full(dim, float(fallback), dtype=np.float64)
    offset = 0
    for name, term in terms.items():
        if name in per_term:
            vec[offset:offset + term.dim] = float(per_term[name])
        offset += term.dim
    return vec


class SysIDParameters:
    """Term-driven parameter manager for system identification.

    Usage::

        from consysid.configs.presets.go2_genesis import Go2GenesisMassComConfig
        params = SysIDParameters(env, Go2GenesisMassComConfig())
        params.dim          # 4 (mass=1 + com=3)
        params.theta_0      # [6.921, 0, 0, 0]
        params.apply_to_env(theta_batch)
    """

    def __init__(self, env: Any, config: SysIDConfig):
        self.env = env
        self.config = config
        self.device = env.device

        self.terms: dict[str, SysIDTermConfig] = iter_terms(config, SysIDTermConfig)
        if not self.terms:
            raise ValueError("SysIDConfig has no SysIDTermConfig terms defined.")

        self._term_slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for name, term in self.terms.items():
            self._term_slices[name] = (offset, offset + term.dim)
            offset += term.dim

        self.dim = offset
        self.theta_0 = self._build_theta_0()
        self.theta_gt = self._build_theta_gt()

    def _build_theta_0(self) -> np.ndarray:
        parts = []
        for term in self.terms.values():
            if isinstance(term.default, (int, float)):
                parts.append(np.full(term.dim, term.default, dtype=np.float32))
            else:
                arr = np.array(term.default, dtype=np.float32)
                assert len(arr) == term.dim, (
                    f"default length {len(arr)} != dim {term.dim}"
                )
                parts.append(arr)
        return np.concatenate(parts)

    def build_theta(self, spec: Any) -> np.ndarray:
        """Build a flat ``(dim,)`` theta vector from heterogeneous sources.

        Single dispatch point used by every stage of the pipeline that
        needs a starting / linearisation θ — identification CMA-ES init,
        active exploration linearisation point, etc. Lets every config
        field that accepts a "theta source" take any of:

        * ``None``                                 — return ``theta_0``
            (each term's ``default`` field, the URDF nominal).
        * ``np.ndarray`` of shape ``(self.dim,)``  — used as-is after
            shape validation. ``float32`` cast applied.
        * ``dict[str, float | sequence[float]]``   — partial dict
            keyed by term names. Scalar values broadcast across the
            term's ``dim``; sequences must match ``term.dim`` exactly.
            Missing terms inherit ``theta_0`` (term default), so the
            user only fills the dims they care about.
        * ``str``                                  — treated as an
            artifact spec (local file path or wandb reference like
            ``"entity/project/sysid-result:vN"``) and loaded via
            :func:`consysid.sysid.result_io.load_theta_best`. The
            wandb form is auto-downloaded through
            :func:`consysid.io.resolve_artifact_path`.

        Returns
        -------
        np.ndarray
            New ``(dim,)`` ``float32`` array. Mutating the result does
            not affect ``params.theta_0`` (always copied).
        """
        if spec is None:
            return self.theta_0.copy()

        if isinstance(spec, np.ndarray):
            if spec.shape != (self.dim,):
                raise ValueError(
                    f"build_theta: ndarray spec shape {tuple(spec.shape)} "
                    f"!= (dim={self.dim},)"
                )
            return spec.astype(np.float32, copy=True)

        if isinstance(spec, dict):
            return self._build_theta_from_dict(spec)

        if isinstance(spec, str):
            from consysid.io import resolve_artifact_path
            from consysid.sysid.result_io import load_theta_best

            local = resolve_artifact_path(spec)
            return load_theta_best(local, self).astype(np.float32, copy=False)

        raise TypeError(
            f"build_theta: spec must be None / np.ndarray / dict / str, "
            f"got {type(spec).__name__}"
        )

    def _build_theta_from_dict(self, spec: dict) -> np.ndarray:
        """Assemble a flat ``(dim,)`` vector from a partial term dict.

        Missing terms inherit ``theta_0`` (term defaults) so the user
        only fills the dims they want to override.
        """
        unknown = set(spec.keys()) - set(self.terms.keys())
        if unknown:
            raise ValueError(
                f"build_theta: unknown term names in dict spec: "
                f"{sorted(unknown)}. "
                f"Valid term names: {sorted(self.terms.keys())}"
            )

        result = self.theta_0.copy()
        for name, val in spec.items():
            start, end = self._term_slices[name]
            term_dim = end - start
            if isinstance(val, (int, float, np.floating, np.integer)):
                result[start:end] = float(val)
            else:
                arr = np.asarray(val, dtype=np.float32)
                if arr.shape != (term_dim,):
                    raise ValueError(
                        f"build_theta: spec[{name!r}] expected shape "
                        f"({term_dim},) (or a scalar), got "
                        f"{tuple(arr.shape)}"
                    )
                result[start:end] = arr
        return result

    def _build_theta_gt(self) -> np.ndarray | None:
        """Build ground-truth vector. Returns None if no term has GT."""
        has_any_gt = any(
            t.ground_truth is not None or t.ground_truth_physical is not None
            for t in self.terms.values()
        )
        if not has_any_gt:
            return None
        parts = []
        for term in self.terms.values():
            if term.ground_truth_physical is not None:
                if term.gt_encoder is None:
                    raise ValueError(
                        f"term has ground_truth_physical set but no gt_encoder"
                    )
                gt_arr = np.asarray(
                    term.gt_encoder(term.ground_truth_physical), dtype=np.float32,
                )
                if gt_arr.shape != (term.dim,):
                    raise ValueError(
                        f"gt_encoder returned shape {gt_arr.shape}, expected ({term.dim},)"
                    )
                parts.append(gt_arr)
            elif term.ground_truth is None:
                parts.append(np.full(term.dim, np.nan, dtype=np.float32))
            elif isinstance(term.ground_truth, (int, float)):
                parts.append(np.full(term.dim, term.ground_truth, dtype=np.float32))
            else:
                parts.append(np.array(term.ground_truth, dtype=np.float32))
        return np.concatenate(parts)

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower_parts, upper_parts = [], []
        for term in self.terms.values():
            lo, hi = term.bounds
            if isinstance(lo, (int, float)):
                lower_parts.append(np.full(term.dim, lo, dtype=np.float32))
            else:
                lower_parts.append(np.array(lo, dtype=np.float32))
            if isinstance(hi, (int, float)):
                upper_parts.append(np.full(term.dim, hi, dtype=np.float32))
            else:
                upper_parts.append(np.array(hi, dtype=np.float32))
        return np.concatenate(lower_parts), np.concatenate(upper_parts)

    def apply_to_env(self, theta_batch: np.ndarray) -> None:
        """Apply parameter vector to all parallel environments.

        Args:
            theta_batch: shape ``(B, dim)`` where B = num_envs.
        """
        env_ids = torch.arange(theta_batch.shape[0], device=self.device)
        for name, term in self.terms.items():
            start, end = self._term_slices[name]
            values = theta_batch[:, start:end]
            term.func(env=self.env, values=values, env_ids=env_ids, **term.params)

        # First-time post-apply state dump — verifies that every PD /
        # friction term's ``joint_patterns`` regex resolved to the
        # expected DOF indices. The dump reads back the model arrays
        # the apply functions just wrote, joins them with the env's
        # ``view.joint_dof_names`` so the user can spot a swapped /
        # off-by-one mapping at a glance, then sets a flag so we
        # don't spam the log on every subsequent apply (the apply
        # function is called O(num_iter × num_candidates) times
        # during CMA-ES; a one-shot verification at startup is enough).
        if not getattr(self.env, "_sysid_apply_verify_logged", False):
            try:
                self._print_apply_verification()
            except Exception as exc:  # noqa: BLE001
                # Best-effort — never break the apply path on a
                # readback hiccup. Most likely cause: non-Newton sim
                # without ``robot_view`` / ``scene_manager`` API.
                print(
                    f"[SysIDParameters.apply_to_env] verification print "
                    f"skipped: {exc}"
                )
            self.env._sysid_apply_verify_logged = True

    def _print_apply_verification(self) -> None:
        """Dump post-apply PD / friction state for human inspection.

        Reads back ``joint_target_ke`` / ``joint_target_kd`` /
        ``joint_friction`` (and foot ``shape_material_mu`` when
        applicable) and prints per-DOF / per-foot rows so the user
        can directly verify regex → DOF mappings are correct *after*
        the writes have happened. Newton-specific (relies on
        ``env.scene_manager.robot_view`` and ``env.scene_manager.model``);
        the caller wraps in try/except to keep other sims silent.
        """
        import warp as wp

        view = self.env.scene_manager.robot_view
        model = self.env.scene_manager.model

        # Per-DOF columns. ``view.get_attribute`` returns a 3-D tensor
        # ``(num_envs, count_per_world, joint_dof_count)`` — count_per_world
        # is 1 for SysID setups (one articulation per world), so we peel
        # that middle dim off before scalar indexing per dof. Mirrors
        # the same flattening done in
        # ``param_terms/newton.py::_apply_joint_dof_attribute`` (the
        # write side), keeping read & write layouts consistent.
        # Skip an attribute silently if the sim variant doesn't expose
        # it (e.g. a non-PD actuator path).
        attrs = ["joint_target_ke", "joint_target_kd", "joint_friction"]
        columns: dict[str, "torch.Tensor"] = {}
        for attr in attrs:
            try:
                t = wp.to_torch(view.get_attribute(attr, model))
                if t.ndim == 3:
                    # (num_envs, count_per_world, joint_dof_count) — peel.
                    t = t[:, 0, :]
                # Now expect (num_envs, joint_dof_count); leave 1-D
                # tensors as-is (some attrs may be flat).
                columns[attr] = t
            except Exception:
                pass

        if columns:
            print(
                "\n[SysIDParameters.apply_to_env] post-apply state "
                "(env 0) — verify regex→DOF mapping below:"
            )
            header_attrs = list(columns.keys())
            header_widths = {a: max(len(a), 14) for a in header_attrs}
            print(
                f"  {'joint dof name':<35s} "
                + " ".join(f"{a:>{header_widths[a]}s}" for a in header_attrs)
            )
            for dof_idx, dof_name in enumerate(view.joint_dof_names):
                vals = " ".join(
                    f"{(columns[a][0, dof_idx] if columns[a].ndim >= 2 else columns[a][dof_idx]).item():>{header_widths[a]}.6f}"
                    for a in header_attrs
                )
                print(f"  {dof_name:<35s} {vals}")

        # Foot contact mu (per-foot shape) — same readback pattern as
        # ``apply_contact_friction`` / ``set_foot_friction`` so the
        # printed value is exactly what the solver will see at the
        # next contact pair.
        try:
            from jaxrlworld.rl.envs.utils.newton.body_cache import get_cache

            cache = get_cache(self.env)
            foot_body_idx = cache.get_body_indices(".*foot$")
            if hasattr(foot_body_idx, "tolist"):
                foot_body_idx = foot_body_idx.tolist()

            n_per_env = model.shape_count // self.env.num_envs
            n_robot = self.env.num_envs * n_per_env
            flat_mu = wp.to_torch(model.shape_material_mu)
            shape_mu = flat_mu[:n_robot].reshape(self.env.num_envs, n_per_env)

            print("\n  foot shape contact mu (env 0):")
            for bi in foot_body_idx:
                for si in model.body_shapes[int(bi)]:
                    print(
                        f"    body {int(bi):3d} shape {int(si):3d}: "
                        f"mu = {shape_mu[0, int(si)].item():.4f}"
                    )
        except Exception:
            # Silent skip: contact mu read-back depends on Newton
            # internals that may differ across versions. Joint table
            # above is the primary verification artifact.
            pass

    def build_delta_vec(self, fallback: float) -> np.ndarray:
        """Return a ``(dim,)`` float array of per-dim finite-difference
        step sizes, with any term-level ``delta_param`` override applied
        and ``fallback`` used elsewhere.

        Ensures the partitioner and FIM evaluator agree on the exact δ
        used for each parameter dim — critical when θ and φ have
        different natural scales (mass vs friction, etc.). Index layout
        matches ``theta_0``.
        """
        delta_vec = np.full(self.dim, fallback, dtype=np.float32)
        for name, term in self.terms.items():
            start, end = self._term_slices[name]
            override = getattr(term, "delta_param", None)
            if override is not None:
                if override <= 0.0:
                    raise ValueError(
                        f"term {name!r}: delta_param must be positive, "
                        f"got {override}"
                    )
                delta_vec[start:end] = float(override)
        return delta_vec

    def regularization(self, theta_batch: np.ndarray) -> np.ndarray:
        """L2 regularization toward defaults, weighted per term."""
        reg = np.zeros(theta_batch.shape[0], dtype=np.float64)
        for name, term in self.terms.items():
            start, end = self._term_slices[name]
            dev = theta_batch[:, start:end] - self.theta_0[start:end]
            reg += term.reg_weight * np.sum(dev ** 2, axis=1)
        return reg

    def describe(self) -> str:
        """Human-readable summary of the parameter space."""
        lines = [f"SysIDParameters: {self.dim}D"]
        for name, term in self.terms.items():
            start, end = self._term_slices[name]
            lo, hi = self.get_bounds()
            lines.append(
                f"  [{start}:{end}] {name} (dim={term.dim}) "
                f"default={self.theta_0[start:end]} "
                f"bounds=[{lo[start:end]}, {hi[start:end]}] "
                f"reg={term.reg_weight}"
            )
        return "\n".join(lines)
