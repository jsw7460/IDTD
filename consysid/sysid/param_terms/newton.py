"""Newton simulator apply functions for SysID parameters.

All functions follow the signature:
    func(env, values: np.ndarray, env_ids: torch.Tensor, **params) -> None

where ``values`` has shape ``(B, dim)`` and ``env_ids`` shape ``(B,)``.

Two patterns are used for writing to Newton's Model:

1. Body-level attributes (``body_mass``, ``body_com``, ``body_inertia``) —
   direct ``wp.to_torch(model.X).reshape(num_envs, bodies_per_env, ...)``
   via ``body_cache``. Name matching runs against the full hierarchical
   body labels.

2. Joint-DOF attributes (``joint_target_ke``, ``joint_target_kd``,
   ``joint_armature``, ``joint_friction``) — ``robot_view.get_attribute``
   / ``set_attribute`` scoped to the robot articulation. DOF selection
   uses ``joint_patterns`` regex matched against the view's leaf-name
   ``joint_dof_names`` (with ``:N`` suffix stripped for multi-DOF joints).
   This removes the need for ``free_joint_dofs=6`` offset hardcoding and
   supports non-contiguous DOF subsets.
"""
from __future__ import annotations

import re

import numpy as np
import torch


def _resolve_joint_dof_indices(
    env,
    joint_patterns: str | tuple[str, ...] | list[str],
) -> list[int]:
    """Robot-view DOF column indices in CANONICAL (act-manager) order.

    Every per-joint value vector in SysID (term defaults, bounds, theta
    slots, dataset joint channels) is declared in the act-manager's
    canonical joint order, so pattern matching runs over
    ``act_manager.actuated_joint_names`` and the i-th returned index is
    the ``view.joint_dof_names`` column of the i-th matched CANONICAL
    joint. The two orders differ whenever the MJCF parse order deviates
    from the canonical DFS walk (K1: the XML declares the right arm
    before the left leg) — resolving patterns directly against the view,
    as this helper originally did, scattered canonical-ordered vectors
    onto the wrong physical joints for such robots.

    Args:
        env: Newton env exposing ``act_manager`` and
            ``scene_manager.robot_view``.
        joint_patterns: Single regex or iterable of regexes. Uses
            ``re.search`` against bare (prefix-stripped) joint names.

    Returns:
        Robot-view DOF column indices, ordered canonically.

    Raises:
        ValueError: if no canonical joint matches, or a matched joint
            does not map to exactly one view DOF column (multi-DOF
            joints are not supported by the per-joint apply path).
    """
    view = env.scene_manager.robot_view
    if isinstance(joint_patterns, str):
        joint_patterns = (joint_patterns,)
    compiled = [re.compile(p) for p in joint_patterns]

    canonical_bare = [
        n.rsplit("/", 1)[-1] for n in env.act_manager.actuated_joint_names
    ]
    matched_names = [
        n for n in canonical_bare if any(rx.search(n) for rx in compiled)
    ]
    if not matched_names:
        raise ValueError(
            f"No actuated joints match joint_patterns={joint_patterns!r}. "
            f"Available joints: {canonical_bare}"
        )

    view_bare = [
        n.rsplit(":", 1)[0].rsplit("/", 1)[-1] for n in view.joint_dof_names
    ]
    matched: list[int] = []
    for name in matched_names:
        cols = [i for i, v in enumerate(view_bare) if v == name]
        if len(cols) != 1:
            raise ValueError(
                f"Joint {name!r} maps to {len(cols)} view DOF columns "
                f"(expected exactly 1). View DOF names: {view_bare}"
            )
        matched.append(cols[0])
    return matched


def apply_mass(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    body_pattern: str = ".*trunk$",
) -> None:
    """Set absolute mass on a body. values: (B, 1) in kg."""
    import warp as wp
    from newton import ModelFlags
    from jaxrlworld.rl.envs.utils.newton.body_cache import get_cache

    cache = get_cache(env)
    model = env.scene_manager.model
    body_idx = cache.get_body_indices(body_pattern)[0]

    body_mass = wp.to_torch(model.body_mass).reshape(env.num_envs, cache.bodies_per_env)
    body_mass[env_ids, body_idx] = torch.tensor(
        values[:, 0], dtype=torch.float32, device=env.device,
    )
    wp.copy(model.body_mass, wp.from_torch(body_mass.flatten(), dtype=wp.float32))

    env.scene_manager.solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)


_default_com_cache: dict[int, torch.Tensor] = {}


def apply_com(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    body_pattern: str = ".*trunk$",
) -> None:
    """Set CoM offset on a body. values: (B, 3) in metres (shift from default)."""
    import warp as wp
    from newton import ModelFlags
    from jaxrlworld.rl.envs.utils.newton.body_cache import get_cache

    cache = get_cache(env)
    model = env.scene_manager.model
    body_idx = cache.get_body_indices(body_pattern)[0]

    body_com = wp.to_torch(model.body_com).reshape(env.num_envs, cache.bodies_per_env, 3)

    # Cache default CoM on first call (before any modifications)
    cache_key = (id(env), body_idx)
    if cache_key not in _default_com_cache:
        _default_com_cache[cache_key] = body_com[0, body_idx].clone()
    default_com = _default_com_cache[cache_key]

    shift = torch.tensor(values, dtype=torch.float32, device=env.device)
    body_com[env_ids, body_idx, :] = default_com + shift

    wp.copy(
        model.body_com,
        wp.from_torch(body_com.reshape(-1, 3).contiguous(), dtype=wp.vec3),
    )

    env.scene_manager.solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)


def apply_log_cholesky_inertial(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    body_pattern: str = ".*trunk$",
) -> None:
    """Set mass + CoM + full inertia via log-Cholesky pseudo-inertia.

    values: (B, 10) — log-Cholesky coordinates phi. Each row decodes to
    a physically valid (mass, com, inertia) triple which is written to
    Newton's body_mass, body_com, and body_inertia respectively for the
    matched body. See consysid.sysid.log_cholesky for the parameterization.
    """
    import warp as wp
    from newton import ModelFlags
    from jaxrlworld.rl.envs.utils.newton.body_cache import get_cache

    from consysid.sysid.log_cholesky import decode_phi_batched

    cache = get_cache(env)
    model = env.scene_manager.model
    body_idx = cache.get_body_indices(body_pattern)[0]

    phi = torch.as_tensor(values, dtype=torch.float32, device=env.device)
    mass, com, inertia = decode_phi_batched(phi)

    body_mass = wp.to_torch(model.body_mass).reshape(
        env.num_envs, cache.bodies_per_env,
    )
    body_mass[env_ids, body_idx] = mass
    wp.copy(model.body_mass, wp.from_torch(body_mass.flatten(), dtype=wp.float32))

    body_com = wp.to_torch(model.body_com).reshape(
        env.num_envs, cache.bodies_per_env, 3,
    )
    body_com[env_ids, body_idx, :] = com
    wp.copy(
        model.body_com,
        wp.from_torch(body_com.reshape(-1, 3).contiguous(), dtype=wp.vec3),
    )

    body_inertia = wp.to_torch(model.body_inertia).reshape(
        env.num_envs, cache.bodies_per_env, 3, 3,
    )
    body_inertia[env_ids, body_idx, :, :] = inertia
    wp.copy(
        model.body_inertia,
        wp.from_torch(
            body_inertia.reshape(-1, 3, 3).contiguous(), dtype=wp.mat33,
        ),
    )

    env.scene_manager.solver.notify_model_changed(
        ModelFlags.BODY_INERTIAL_PROPERTIES,
    )


def _apply_joint_dof_attribute(
    env,
    attr_name: str,
    values: np.ndarray,
    env_ids: torch.Tensor,
    joint_patterns: str | tuple[str, ...] | list[str],
) -> None:
    """Shared plumbing for joint-DOF writes via ArticulationView.

    Resolves ``joint_patterns`` to articulation-local DOF indices, fetches
    the attribute's staging tensor through ``robot_view``, scatters
    ``values`` into the matched ``(env_ids, dof_indices)`` slots, and
    pushes the update back with a ``JOINT_DOF_PROPERTIES`` notify.

    ``values`` must have shape ``(len(env_ids), len(dof_indices))``.
    """
    import warp as wp
    from newton import ModelFlags

    view = env.scene_manager.robot_view
    model = env.scene_manager.model

    dof_indices = _resolve_joint_dof_indices(env, joint_patterns)
    dof_idx_t = torch.as_tensor(dof_indices, dtype=torch.long, device=env.device)

    attrib = wp.to_torch(view.get_attribute(attr_name, model))
    # attrib shape: (num_envs, count_per_world, joint_dof_count). SysID
    # setups have exactly one articulation per world, so squeeze that
    # dim as a live view before scattering.
    attrib_flat = attrib[:, 0, :]

    vals_t = torch.as_tensor(values, dtype=torch.float32, device=env.device)
    if vals_t.shape != (env_ids.shape[0], dof_idx_t.shape[0]):
        raise ValueError(
            f"{attr_name}: expected values shape "
            f"({env_ids.shape[0]}, {dof_idx_t.shape[0]}), got {tuple(vals_t.shape)}"
        )

    attrib_flat[env_ids.unsqueeze(1), dof_idx_t] = vals_t

    view.set_attribute(attr_name, model, attrib)
    env.scene_manager.solver.notify_model_changed(
        ModelFlags.JOINT_DOF_PROPERTIES,
    )

    # JaxRLWorld's explicit-PD path (rl/actuators/actuator_pd.py)
    # computes torques as ``stiffness * error_pos - damping * joint_vel``
    # reading the actuator's own ``stiffness`` / ``damping`` tensors —
    # NOT ``model.joint_target_ke`` / ``joint_target_kd`` (which are
    # only consumed by Newton's implicit-PD solver path, unused here).
    # Without this mirror write, ``apply_kp*`` / ``apply_kd*`` are
    # silently dropped from the live PD computation: SysID would tune
    # ``model.joint_target_ke`` while the actuator keeps cfg-default
    # gains, producing a constant bias on every torque-bearing cost
    # term. Mirror the write here so a single apply call updates both
    # surfaces atomically.
    if attr_name in ("joint_target_ke", "joint_target_kd"):
        _sync_actuator_pd_buffer(
            env, attr_name, vals_t, env_ids, dof_indices,
        )


def _sync_actuator_pd_buffer(
    env,
    attr_name: str,
    vals_t: torch.Tensor,
    env_ids: torch.Tensor,
    dof_indices: list[int] | tuple[int, ...] | "torch.Tensor",
) -> None:
    """Mirror a model-side PD-gain write into every matching actuator's
    ``stiffness`` / ``damping`` tensor.

    Maps each robot-view DOF index to its canonical (act-manager) joint
    name, then locates that name inside each actuator group's local
    joint list. Caller-supplied ``vals_t`` already has its columns
    aligned to ``dof_indices`` (post-broadcast for ``*_scalar`` apply
    variants), so we only need a column-permutation to go from
    "robot-view DOF order" to "actuator-local joint order".

    Args:
        attr_name: ``"joint_target_ke"`` → ``stiffness``; ``"joint_target_kd"``
            → ``damping``. Other names are no-ops (caller already gates
            on the PD pair, but the dispatch is kept here for safety).
        vals_t: ``(len(env_ids), len(dof_indices))`` already on device.
        env_ids: env subset to write.
        dof_indices: robot-view DOF indices that ``vals_t``'s columns
            correspond to.
    """
    if attr_name == "joint_target_ke":
        buf_attr = "stiffness"
    elif attr_name == "joint_target_kd":
        buf_attr = "damping"
    else:
        return

    view = env.scene_manager.robot_view
    if hasattr(dof_indices, "tolist"):
        dof_idx_list = list(dof_indices.tolist())
    else:
        dof_idx_list = list(dof_indices)
    matched_dof_names = [view.joint_dof_names[d] for d in dof_idx_list]
    name_to_value_col = {n: c for c, n in enumerate(matched_dof_names)}

    canonical_names = list(env.act_manager.actuated_joint_names)

    for actuator, joint_idx in env.act_manager._actuators:
        if not hasattr(actuator, buf_attr):
            continue
        if hasattr(joint_idx, "tolist"):
            j_list = joint_idx.tolist()
        else:
            j_list = list(joint_idx)

        local_buf_cols: list[int] = []
        local_value_cols: list[int] = []
        for k, can_idx in enumerate(j_list):
            name = canonical_names[can_idx]
            if name in name_to_value_col:
                local_buf_cols.append(k)
                local_value_cols.append(name_to_value_col[name])

        if not local_buf_cols:
            continue

        buf = getattr(actuator, buf_attr)
        buf_idx_t = torch.as_tensor(
            local_buf_cols, dtype=torch.long, device=env.device,
        )
        val_idx_t = torch.as_tensor(
            local_value_cols, dtype=torch.long, device=env.device,
        )
        buf[env_ids.unsqueeze(1), buf_idx_t.unsqueeze(0)] = vals_t[:, val_idx_t]


def apply_kp(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    joint_patterns: str | tuple[str, ...] | list[str] = (r".*_joint$",),
) -> None:
    """Set PD proportional gains (``joint_target_ke``).

    ``values`` shape ``(B, n_matched_dofs)``, columns in CANONICAL
    (act-manager) joint order — the same order every SysID term config
    declares its defaults in. ``joint_patterns`` selects joints by regex
    on bare joint names. Default ``(r".*_joint$",)`` targets every joint
    whose name ends in ``_joint`` — this excludes Newton's
    ``floating_base`` DOFs and picks up the 12 actuated leg joints on a
    Go2.
    """
    _apply_joint_dof_attribute(
        env, "joint_target_ke", values, env_ids, joint_patterns,
    )


def apply_kd(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    joint_patterns: str | tuple[str, ...] | list[str] = (r".*_joint$",),
) -> None:
    """Set PD derivative gains (``joint_target_kd``). See :func:`apply_kp`."""
    _apply_joint_dof_attribute(
        env, "joint_target_kd", values, env_ids, joint_patterns,
    )


def apply_joint_friction(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    joint_patterns: str | tuple[str, ...] | list[str] = (r".*_joint$",),
) -> None:
    """Set Coulomb joint friction torque (``joint_friction``) [N·m].

    Static load-independent friction on the DOFs matched by
    ``joint_patterns``. See :func:`apply_kp` for the pattern convention.
    """
    _apply_joint_dof_attribute(
        env, "joint_friction", values, env_ids, joint_patterns,
    )


def _apply_joint_dof_attribute_scalar(
    env,
    attr_name: str,
    values: np.ndarray,
    env_ids: torch.Tensor,
    joint_patterns: str | tuple[str, ...] | list[str],
) -> None:
    """Shared plumbing for scalar→DOF-broadcast joint writes.

    Expects ``values`` shape ``(B, 1)`` and broadcasts it to every DOF
    matched by ``joint_patterns`` before dispatching to the per-DOF
    writer. Used by every ``*_scalar`` apply variant so the broadcast
    rule lives in one place.
    """
    dof_indices = _resolve_joint_dof_indices(env, joint_patterns)
    num_dofs = len(dof_indices)

    if values.shape[1] != 1:
        raise ValueError(
            f"apply_*_scalar (attr={attr_name!r}) expects values of shape "
            f"(B, 1), got {tuple(values.shape)}"
        )

    # Broadcast (B, 1) → (B, num_dofs). ``broadcast_to`` returns a
    # read-only view; ``.copy()`` materialises it so the downstream
    # ``torch.as_tensor`` produces a writable tensor with real strides.
    values_broadcast = np.broadcast_to(
        values, (values.shape[0], num_dofs),
    ).copy()
    _apply_joint_dof_attribute(
        env, attr_name, values_broadcast, env_ids, joint_patterns,
    )


def apply_joint_friction_scalar(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    joint_patterns: str | tuple[str, ...] | list[str] = (r".*_joint$",),
) -> None:
    """Set a **single scalar** Coulomb joint friction shared across all
    matched DOFs.

    Expects ``values`` shape ``(B, 1)`` — a single coefficient per env.
    Useful for low-dim ablation runs where per-joint friction is
    collapsed to one parameter to sharpen objective-specific effects
    without changing the apply plumbing.
    """
    _apply_joint_dof_attribute_scalar(
        env, "joint_friction", values, env_ids, joint_patterns,
    )


def apply_kp_scalar(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    joint_patterns: str | tuple[str, ...] | list[str] = (r".*_joint$",),
) -> None:
    """Set a single scalar PD proportional gain (``joint_target_ke``)
    shared across all matched DOFs.

    Expects ``values`` shape ``(B, 1)``. Pair with two distinct
    ``joint_patterns`` (e.g. hip/thigh group vs. calf group) when the
    "real-world" PD config groups joints by motor type.
    """
    _apply_joint_dof_attribute_scalar(
        env, "joint_target_ke", values, env_ids, joint_patterns,
    )


def apply_kd_scalar(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    joint_patterns: str | tuple[str, ...] | list[str] = (r".*_joint$",),
) -> None:
    """Set a single scalar PD derivative gain (``joint_target_kd``)
    shared across all matched DOFs. See :func:`apply_kp_scalar`."""
    _apply_joint_dof_attribute_scalar(
        env, "joint_target_kd", values, env_ids, joint_patterns,
    )


def _resolve_shape_label_indices(model, shapes_per_env: int, pattern: str) -> list[int]:
    """Per-env-local shape indices whose ``model.shape_label`` matches *pattern*.

    Matches against the raw ``model.shape_label`` (world-0 slice) and
    returns indices directly into ``model.shape_material_mu``'s per-env
    layout. This deliberately bypasses ``resolve_selector`` /
    ``ArticulationView.shape_names``: the view's shape indexing is offset
    by one shape from the raw model array, so writing through it lands on
    the neighbouring shape (the foot collision geom at raw index N gets
    written at N+1). Matching the raw labels keeps the indices aligned
    with the array we actually write.
    """
    compiled = re.compile(pattern)
    matched = [si for si in range(shapes_per_env) if compiled.search(model.shape_label[si])]
    if not matched:
        raise ValueError(
            f"apply_contact_friction matched no shapes for pattern={pattern!r}."
        )
    return matched


def apply_contact_friction(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    geom_label_pattern: str = r".*foot_collision$",
) -> None:
    """Set per-env contact friction on the foot collision shapes.

    ``values`` is ``(B, 1)`` (one mu shared by all feet — legacy scalar
    foot_friction) OR ``(B, n_foot)`` (per-foot mu, in the order the foot
    collision shapes are matched, i.e. ``_resolve_shape_label_indices``'s
    shape-label order). Both the collect env and the Stage-1 sim env call
    this same helper, so the per-foot column order is self-consistent
    between ground truth and identification.

    Foot collision shapes are matched by name against the raw
    ``model.shape_label`` and written directly into
    ``model.shape_material_mu``. The Newton MuJoCo solver resolves a
    contact pair's friction by geom *priority*: Go2's foot collision geoms
    (priority 1) win over the ground plane (priority 0), so only the foot
    geoms need writing (ground is left untouched).
    """
    import warp as wp
    from newton import ModelFlags

    model = env.scene_manager.model

    # shape_material_mu may contain trailing global shapes (e.g. ground) that
    # don't divide evenly by num_envs. Slice off those trailing entries before
    # reshaping into per-env layout.
    shapes_per_env = model.shape_count // env.num_envs
    n_robot_shapes = env.num_envs * shapes_per_env
    flat_mu = wp.to_torch(model.shape_material_mu)
    shape_mu = flat_mu[:n_robot_shapes].reshape(env.num_envs, shapes_per_env)

    target_shapes = _resolve_shape_label_indices(model, shapes_per_env, geom_label_pattern)
    n_foot = len(target_shapes)

    vals = np.asarray(values, dtype=np.float32)
    if vals.ndim == 1:
        vals = vals[:, None]
    if vals.shape[1] == 1:
        cols = [vals[:, 0]] * n_foot                     # scalar → all feet
    elif vals.shape[1] == n_foot:
        cols = [vals[:, i] for i in range(n_foot)]       # per-foot
    else:
        raise ValueError(
            f"apply_contact_friction: values has {vals.shape[1]} columns; "
            f"expected 1 (scalar) or {n_foot} (per-foot).",
        )

    for i, si in enumerate(target_shapes):
        shape_mu[env_ids, si] = torch.tensor(
            cols[i], dtype=torch.float32, device=env.device,
        )

    wp.copy(model.shape_material_mu, wp.from_torch(flat_mu, dtype=wp.float32))
    env.scene_manager.solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)


def apply_armature(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    joint_patterns: str | tuple[str, ...] | list[str] = (r".*_joint$",),
    unit: float = 1.0,
) -> None:
    """Set joint reflected-rotor inertia (``joint_armature``).

    ``unit`` scales θ to SI (armature spans ~1e-3..1e-1 kg·m² across a
    humanoid — pass e.g. 1e-2 so the raw-θ CMA sees an O(0.1–10) axis).
    See :func:`apply_kp` for the pattern convention.
    """
    _apply_joint_dof_attribute(
        env, "joint_armature", np.asarray(values) * unit, env_ids,
        joint_patterns,
    )


def apply_tau_scale(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    group_patterns: tuple[tuple[str, ...], ...] = (),
    group_scales: tuple[float, ...] = (),
) -> None:
    """Set the tanh torque-saturation scale κ per joint GROUP.

    The actuator model is ``τ = κ·tanh(τ_PD/κ)`` (JaxRLWorld explicit PD
    actuator, ``tau_scale``). κ is an actuator-TYPE property (motor +
    gearbox design), so it is identified per group with left/right tied
    — per-unit variation stays in the per-joint friction/armature terms.

    ``values`` shape ``(B, n_groups)``, θ as a RATIO of the group's
    nominal scale: the written κ is ``θ_g · group_scales[g]`` [N·m],
    broadcast to every actuated joint matched by ``group_patterns[g]``.
    Groups must cover every actuated joint exactly once (crash
    otherwise). Writes the explicit actuator's ``tau_scale`` buffer —
    the same surface the training DR (``randomize_tau_scale``) mutates,
    in actuator-local column order mapped from canonical indices.

    Args:
        values: ``(B, n_groups)`` ratio θ (nominal 1.0 = group_scales).
        group_patterns: One regex tuple per group, matched with
            ``re.search`` against bare canonical joint names.
        group_scales: Per-group nominal κ [N·m] (typically the effort
            limit), same length as ``group_patterns``.
    """
    if len(group_patterns) == 0 or len(group_patterns) != len(group_scales):
        raise ValueError(
            f"apply_tau_scale: group_patterns ({len(group_patterns)}) and "
            f"group_scales ({len(group_scales)}) must be non-empty and equal-length"
        )
    canonical = [
        n.rsplit("/", 1)[-1] for n in env.act_manager.actuated_joint_names
    ]
    vals_t = torch.as_tensor(values, dtype=torch.float32, device=env.device)
    if vals_t.shape != (env_ids.shape[0], len(group_patterns)):
        raise ValueError(
            f"apply_tau_scale: expected values shape "
            f"({env_ids.shape[0]}, {len(group_patterns)}), got {tuple(vals_t.shape)}"
        )

    full = torch.full(
        (env_ids.shape[0], len(canonical)), float("nan"), device=env.device,
    )
    for gi, pats in enumerate(group_patterns):
        compiled = [re.compile(p) for p in pats]
        cols = [
            j for j, n in enumerate(canonical)
            if any(rx.search(n) for rx in compiled)
        ]
        if not cols:
            raise ValueError(
                f"apply_tau_scale: group {gi} patterns {pats!r} match no "
                f"actuated joint. Available: {canonical}"
            )
        if not torch.isnan(full[0, cols]).all():
            raise ValueError(
                f"apply_tau_scale: group {gi} patterns {pats!r} overlap a "
                f"previously covered joint"
            )
        full[:, cols] = (vals_t[:, gi] * float(group_scales[gi])).unsqueeze(1)
    if torch.isnan(full).any():
        uncovered = [n for j, n in enumerate(canonical)
                     if bool(torch.isnan(full[0, j]))]
        raise ValueError(
            f"apply_tau_scale: joints not covered by any group: {uncovered}"
        )

    wrote = False
    for actuator, joint_idx in env.act_manager.actuators:
        if not getattr(actuator, "_use_tau_scale", False):
            raise ValueError(
                "apply_tau_scale: actuator has no tau_scale configured — "
                "set the robot cfg's tau_scale so the tanh model is active"
            )
        j = torch.as_tensor(joint_idx, dtype=torch.long, device=env.device)
        actuator.tau_scale[env_ids] = full[:, j]
        wrote = True
    if not wrote:
        raise ValueError("apply_tau_scale: env has no explicit actuators")


def _expand_groups_full(env, values, env_ids, group_patterns, fname: str):
    """Expand per-group values to a per-joint (B, n_actuated) tensor.

    Shared plumbing for the grouped actuator-buffer setters: patterns
    are matched against bare canonical joint names; groups must cover
    every actuated joint exactly once.
    """
    canonical = [
        n.rsplit("/", 1)[-1] for n in env.act_manager.actuated_joint_names
    ]
    vals_t = torch.as_tensor(values, dtype=torch.float32, device=env.device)
    if vals_t.shape != (env_ids.shape[0], len(group_patterns)):
        raise ValueError(
            f"{fname}: expected values shape "
            f"({env_ids.shape[0]}, {len(group_patterns)}), got {tuple(vals_t.shape)}"
        )
    full = torch.full(
        (env_ids.shape[0], len(canonical)), float("nan"), device=env.device,
    )
    for gi, pats in enumerate(group_patterns):
        compiled = [re.compile(p) for p in pats]
        cols = [j for j, n in enumerate(canonical)
                if any(rx.search(n) for rx in compiled)]
        if not cols:
            raise ValueError(f"{fname}: group {gi} patterns {pats!r} match "
                             f"no actuated joint. Available: {canonical}")
        if not torch.isnan(full[0, cols]).all():
            raise ValueError(f"{fname}: group {gi} patterns {pats!r} overlap "
                             f"a previously covered joint")
        full[:, cols] = vals_t[:, gi].unsqueeze(1)
    if torch.isnan(full).any():
        uncovered = [n for j, n in enumerate(canonical)
                     if bool(torch.isnan(full[0, j]))]
        raise ValueError(f"{fname}: joints not covered by any group: {uncovered}")
    return full


def _write_actuator_buffer(env, env_ids, full, attr: str, flag: str,
                           fname: str) -> None:
    wrote = False
    for actuator, joint_idx in env.act_manager.actuators:
        if not getattr(actuator, flag, False):
            raise ValueError(
                f"{fname}: actuator feature {flag!r} is not active — set the "
                f"robot cfg field so the buffer exists (neutral default is fine)"
            )
        j = torch.as_tensor(joint_idx, dtype=torch.long, device=env.device)
        getattr(actuator, attr)[env_ids] = full[:, j]
        wrote = True
    if not wrote:
        raise ValueError(f"{fname}: env has no explicit actuators")


def apply_tau_lpf_tc(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    group_patterns: tuple[tuple[str, ...], ...] = (),
    unit: float = 1.0,
) -> None:
    """Set the first-order torque-lag time constant per joint GROUP.

    ``values`` shape ``(B, n_groups)``; the written time constant is
    ``theta * unit`` seconds (pass unit=1e-2 so theta rides in units of
    10 ms for CMA scaling). Writes the explicit PD actuator's
    ``tau_lpf_tc`` buffer — the robot cfg must have
    ``tau_lpf_time_constant`` set (0.0 neutral default suffices).
    """
    full = _expand_groups_full(env, values, env_ids, group_patterns,
                               "apply_tau_lpf_tc") * float(unit)
    if bool((full < 0.0).any()):
        raise ValueError("apply_tau_lpf_tc: time constants must be >= 0")
    _write_actuator_buffer(env, env_ids, full, "tau_lpf_tc", "_use_lpf",
                           "apply_tau_lpf_tc")


def apply_dyn_gain(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    group_patterns: tuple[tuple[str, ...], ...] = (),
) -> None:
    """Set the velocity-gated transmission efficiency per joint GROUP.

    ``values`` shape ``(B, n_groups)``, the moving-regime torque
    fraction (1.0 = lossless). Writes the actuator's ``dyn_gain``
    buffer — the robot cfg must have ``dyn_gain``/``dyn_gain_velocity``
    set (1.0 neutral default suffices).
    """
    full = _expand_groups_full(env, values, env_ids, group_patterns,
                               "apply_dyn_gain")
    if bool((full <= 0.0).any()):
        raise ValueError("apply_dyn_gain: gains must be > 0")
    _write_actuator_buffer(env, env_ids, full, "dyn_gain", "_use_dyn_gain",
                           "apply_dyn_gain")


# ═══════════════════════════════════════════════════════════════════════════
# Crazyflie (thrust-motor) terms
#
# The propeller actuator carries per-env/per-motor parameter BUFFERS
# (kf/km/tau_inc/tau_dec, shape (num_envs, num_motors)) — the thrust
# analogue of the PD actuators' stiffness/damping buffers. These setters
# write θ slices into those buffers / the base body's inertial fields.
#
# UNIT CONVENTION: θ carries SCALED units so every axis is O(0.01–10)
# for the raw-θ CMA (kf in 1e-10 N/rpm², km in 1e-12 N·m/rpm², inertia
# in 1e-5 kg·m², com in mm). Each setter converts to SI via ``unit``.
# ═══════════════════════════════════════════════════════════════════════════

_CF_ACTION_TERM = "propeller_thrust"


def _propeller_actuator(env):
    return env.act_manager._terms[_CF_ACTION_TERM]._actuator


def apply_kf_per_rotor(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    unit: float = 1e-10,
) -> None:
    """Per-rotor thrust coefficient. ``values``: (B, num_motors), in
    ``unit`` [N/rpm²] (default 1e-10 → θ≈3.16 at nominal)."""
    act = _propeller_actuator(env)
    v = torch.as_tensor(values, dtype=torch.float32, device=act.device) * unit
    act.kf[env_ids] = v


def apply_km(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    unit: float = 1e-12,
) -> None:
    """Yaw-torque coefficient. ``values``: (B, num_motors) per-rotor, or
    (B, 1) shared (broadcasts over motors); in ``unit`` [N·m/rpm²]
    (default 1e-12 → θ≈7.94 at nominal)."""
    act = _propeller_actuator(env)
    v = torch.as_tensor(values, dtype=torch.float32, device=act.device) * unit
    act.km[env_ids] = v


def apply_motor_tau(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    unit: float = 1e-2,
) -> None:
    """First-order motor time constant, symmetric (inc = dec).
    ``values``: (B, num_motors) per-rotor, or (B, 1) shared; in ``unit``
    [s] (default 1e-2 → θ≈5 at the nominal 0.05 s; keeps this axis'
    bound width comparable to the other terms' for the raw-θ CMA)."""
    act = _propeller_actuator(env)
    v = torch.as_tensor(values, dtype=torch.float32, device=act.device) * unit
    act.tau_inc[env_ids] = v
    act.tau_dec[env_ids] = v


def apply_inertia_diag(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    body_pattern: str = ".*base_link$",
    unit: float = 1e-5,
) -> None:
    """Diagonal body inertia of the base. ``values``: (B, 3) in ``unit``
    [kg·m²] (default 1e-5 → θ≈[1.40, 1.40, 2.17] at cf2x nominal).
    Off-diagonals are zeroed (the cf2x inertia is principal-axis
    diagonal); mass and com are left untouched."""
    import warp as wp
    from newton import ModelFlags
    from jaxrlworld.rl.envs.utils.newton.body_cache import get_cache

    cache = get_cache(env)
    model = env.scene_manager.model
    body_idx = cache.get_body_indices(body_pattern)[0]

    diag = torch.as_tensor(values, dtype=torch.float32, device=env.device) * unit
    inertia = torch.zeros(
        (diag.shape[0], 3, 3), dtype=torch.float32, device=env.device,
    )
    inertia[:, 0, 0] = diag[:, 0]
    inertia[:, 1, 1] = diag[:, 1]
    inertia[:, 2, 2] = diag[:, 2]

    body_inertia = wp.to_torch(model.body_inertia).reshape(
        env.num_envs, cache.bodies_per_env, 3, 3,
    )
    body_inertia[env_ids, body_idx, :, :] = inertia
    wp.copy(
        model.body_inertia,
        wp.from_torch(
            body_inertia.reshape(-1, 3, 3).contiguous(), dtype=wp.mat33,
        ),
    )
    env.scene_manager.solver.notify_model_changed(
        ModelFlags.BODY_INERTIAL_PROPERTIES,
    )


def apply_com_xy(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    body_pattern: str = ".*base_link$",
    unit: float = 1e-3,
) -> None:
    """Base CoM x/y position. ``values``: (B, 2) in ``unit`` [m]
    (default 1e-3 = millimetres; cf2x nominal com is the body origin so
    θ = the offset itself). z is left untouched."""
    import warp as wp
    from newton import ModelFlags
    from jaxrlworld.rl.envs.utils.newton.body_cache import get_cache

    cache = get_cache(env)
    model = env.scene_manager.model
    body_idx = cache.get_body_indices(body_pattern)[0]

    xy = torch.as_tensor(values, dtype=torch.float32, device=env.device) * unit
    body_com = wp.to_torch(model.body_com).reshape(
        env.num_envs, cache.bodies_per_env, 3,
    )
    body_com[env_ids, body_idx, 0] = xy[:, 0]
    body_com[env_ids, body_idx, 1] = xy[:, 1]
    wp.copy(
        model.body_com,
        wp.from_torch(body_com.reshape(-1, 3).contiguous(), dtype=wp.vec3),
    )
    env.scene_manager.solver.notify_model_changed(
        ModelFlags.BODY_INERTIAL_PROPERTIES,
    )
