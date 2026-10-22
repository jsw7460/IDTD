"""Wandb artifact spec resolution + upload helpers.

Pipeline-stage outputs (datasets, sysid_result.npz, active_commands.npz)
all share one portable interface:

  * **spec** — a string that the user / config carries. Either a local
    file path or a wandb artifact reference (``entity/project/name:vN``
    or ``entity/project/name:alias``). Paths are detected by checking
    whether the string is an existing file; otherwise treated as a
    wandb ref. So users can mix-and-match in the same config field
    without thinking about it.
  * ``resolve_artifact_path(spec)`` — spec → local file path. Wandb
    refs auto-download with digest-based caching. Used by every
    consumer that needs the actual bytes (clip dataset loader, theta
    init loader, command-trajectory replay).
  * ``upload_artifact(local, ...)`` — local file → wandb artifact.
    Returns the canonical reference string (``entity/project/name:
    vN``) for the next stage to embed in its own config or metadata.
    Used by every producer that wants its output to be portable
    across servers (identification, active exploration, data collection).

Together these two functions are *the* interface for cross-server
SysID workflows. Pipeline-stage propagation chains (e.g.
``theta_init_path`` embedded in ``active_commands.npz``) carry the
spec string verbatim — when the upstream stage uploaded its result to
wandb, the embedded spec is a wandb ref; when it stayed local, the
embedded spec is a local path. Both cases are handled transparently
by ``resolve_artifact_path`` at the next stage.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import wandb


# Default cache root for resolved wandb artifacts. Each artifact lands
# under ``<cache_root>/<safe_name>/`` (slashes/colons replaced with
# underscores) and a sibling ``.wandb_digest`` file records the remote
# artifact's digest at fetch time so subsequent calls can skip the
# download when cache is fresh.
DEFAULT_ARTIFACT_CACHE = "wandb_artifacts"


def is_wandb_ref(spec: str) -> bool:
    """Heuristic spec-type detection.

    A spec is a *local path* if it points at an existing file on disk;
    otherwise it is treated as a *wandb artifact reference*. This is
    intentionally permissive — the only failure mode is "user wrote a
    bad path that happens to look like a wandb ref", which surfaces as
    a clean ``wandb.Api().artifact(...)`` 404 with the offending spec
    in the error message.

    Note: returns ``False`` for *non-existing* local-looking paths too
    (so a typo in a local path is reported by the wandb side, with
    enough context to fix). To probe definitively, pair with
    ``os.path.isfile`` / ``Path.exists`` upstream.
    """
    return not os.path.isfile(spec)


def resolve_artifact_path(
    spec: str,
    cache_dir: str = DEFAULT_ARTIFACT_CACHE,
    expected_suffix: str = ".npz",
) -> str:
    """Resolve a spec string to a local file path.

    Local paths return unchanged (verified to exist). Wandb refs are
    auto-downloaded and cached under ``cache_dir`` keyed on the spec
    string. Cache freshness is verified against the remote artifact's
    digest, so re-runs after a wandb-side update see the new bytes
    without manual cache eviction.

    Args:
        spec: Local file path or wandb artifact reference string
            (e.g. ``"<entity>/SysID/SysID-Dataset:v45"``).
        cache_dir: Local cache root for downloaded artifacts. Defaults
            to ``./wandb_artifacts/``; each artifact gets its own
            subdirectory keyed on ``spec``.
        expected_suffix: File extension to pick out of the downloaded
            artifact directory. Default ``.npz`` matches every
            SysID-pipeline artifact (datasets, sysid_results, command
            trajectories). Pass ``""`` to disable suffix filtering.

    Returns:
        Absolute or repo-relative local file path that callers can pass
        to ``np.load`` directly.

    Raises:
        FileNotFoundError: ``spec`` looks local but the file doesn't
            exist; or wandb artifact downloaded but contained no file
            matching ``expected_suffix``.
    """
    # Local file → return as-is, but verify it actually exists so that
    # a typo in a local path doesn't silently hand off a phantom path
    # to downstream callers.
    if os.path.isfile(spec):
        return spec

    if not is_wandb_ref(spec):
        # Looked local-ish but doesn't exist. Surface the exact
        # missing path so the user can fix the typo.
        raise FileNotFoundError(
            f"resolve_artifact_path: {spec!r} is not an existing local "
            f"file and does not look like a wandb reference (no '/' "
            f"separator). Did you mean a wandb artifact reference like "
            f"'entity/project/name:version'?"
        )

    api = wandb.Api()
    artifact = api.artifact(spec)
    remote_digest = artifact.digest

    safe_name = spec.replace("/", "_").replace(":", "_")
    download_root = os.path.join(cache_dir, safe_name)
    digest_file = os.path.join(download_root, ".wandb_digest")

    # Cache hit? Compare digest to remote so a wandb-side bump
    # auto-evicts the stale cache.
    if os.path.isdir(download_root) and os.path.isfile(digest_file):
        with open(digest_file, "r") as f:
            cached_digest = f.read().strip()
        if cached_digest == remote_digest:
            cached = _pick_file(download_root, expected_suffix)
            if cached is not None:
                print(f"[resolve_artifact_path] cache hit: {cached}")
                return cached

    # Cache miss / stale → download.
    artifact.download(root=download_root)
    with open(digest_file, "w") as f:
        f.write(remote_digest)

    downloaded = _pick_file(download_root, expected_suffix)
    if downloaded is None:
        raise FileNotFoundError(
            f"Wandb artifact {spec!r} downloaded to {download_root} but "
            f"contains no file matching suffix {expected_suffix!r}."
        )
    print(f"[resolve_artifact_path] downloaded: {downloaded}")
    return downloaded


def _pick_file(directory: str, expected_suffix: str) -> str | None:
    """Return the first file in ``directory`` matching ``expected_suffix``.

    Suffix ``""`` disables filtering and returns the first regular file
    (used when the artifact may carry a non-npz payload).
    """
    if not os.path.isdir(directory):
        return None
    for entry in sorted(os.listdir(directory)):
        full = os.path.join(directory, entry)
        if not os.path.isfile(full):
            continue
        if expected_suffix and not entry.endswith(expected_suffix):
            continue
        if entry.startswith("."):  # skip the digest file etc.
            continue
        return full
    return None


def upload_artifact(
    local_path: str,
    artifact_name: str,
    artifact_type: str,
    wandb_run: Any,
    aliases: list[str] | None = None,
    metadata: dict | None = None,
) -> str:
    """Upload a local file to wandb as an artifact.

    Returns the canonical reference string ``entity/project/name:vN``
    (using the auto-assigned numeric version) so the next stage's
    config field can carry it directly. The caller is responsible for
    persisting this ref string somewhere durable (printed to stdout
    for the user; embedded in a downstream artifact's metadata; etc.).

    Args:
        local_path: File to upload. Must exist.
        artifact_name: Name of the artifact in the wandb registry
            (e.g. ``"sysid-result"``, ``"active-commands"``).
        artifact_type: Wandb artifact type (free-form string —
            grouping label in the wandb UI). Convention here:
            ``"sysid-result"``, ``"active-commands"``, ``"dataset"``.
        wandb_run: Active wandb run handle (the artifact is logged
            under this run, so it appears in the run's lineage view).
        aliases: Optional human-readable aliases (in addition to the
            auto-assigned ``vN`` version). Useful for tagging a result
            with a descriptive name like ``"pd-gains-friction"``.
        metadata: Optional dict — recorded as the artifact's metadata
            in wandb (visible in the UI, queryable via the API). Use
            this to record the produce-time config so the artifact is
            self-describing without needing to crack open the npz.

    Returns:
        Canonical reference string ``entity/project/name:vN``. Stable
        across re-runs of the producing stage (each call produces a
        fresh version).
    """
    if wandb_run is None:
        raise ValueError(
            "upload_artifact requires an active wandb run "
            "(``run = wandb.init(...)``); got None."
        )

    if not os.path.isfile(local_path):
        raise FileNotFoundError(
            f"upload_artifact: source file does not exist: {local_path!r}"
        )

    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata=metadata or {},
    )
    artifact.add_file(str(local_path))

    logged = wandb_run.log_artifact(artifact, aliases=list(aliases or []))
    # ``wait`` blocks until the upload finishes so that the version
    # number is filled in before we read it.
    logged.wait()

    # Construct canonical reference. ``logged.qualified_name`` is the
    # public API in newer wandb versions (returns ``entity/project/
    # name:vN``); fall back to manual assembly for older versions.
    ref = getattr(logged, "qualified_name", None)
    if ref is None:
        entity = logged.entity if hasattr(logged, "entity") else wandb_run.entity
        project = logged.project if hasattr(logged, "project") else wandb_run.project
        version = logged.version if hasattr(logged, "version") else "latest"
        # ``version`` may already include the ``v`` prefix or be a bare integer.
        ver_str = str(version)
        if ver_str.isdigit():
            ver_str = f"v{ver_str}"
        ref = f"{entity}/{project}/{logged.name}:{ver_str}"

    print(
        f"[upload_artifact] uploaded {local_path} → wandb ref: {ref}\n"
        f"  type={artifact_type!r}, aliases={list(aliases or [])}"
    )
    return ref


def refs_from_group(
    group: str | list[str],
    *,
    entity: str,
    project: str = "SysID",
    artifact_type: str = "sysid-result",
    state: str = "finished",
    artifact_alias: str | None = None,
) -> list[str]:
    """Collect artifact refs from one or more wandb GROUPS.

    Lets a comparison / visualization target a wandb group instead of
    pasting each run's artifact version by hand. ``group`` may be a single
    name OR a LIST of names — in the list case every group's finished runs
    are POOLED (e.g. several seed-batches / re-runs of the same condition
    logged under different groups).

    Every run whose ``run.state == state`` (default ``"finished"``)
    contributes the latest ``artifact_type`` artifact it logged; runs in
    any other state (crashed, killed, running, ...) are dropped.
    ``artifact_alias`` (e.g. a stage tag) further restricts to artifacts
    carrying that alias when a run logged several versions.

    Returns qualified refs ("entity/project/name:vXXX"). Raises if the
    group(s) yield nothing usable (wrong name / entity / project, or no
    run finished), rather than silently producing an empty method.
    """
    groups = [group] if isinstance(group, str) else list(group)
    api = wandb.Api()
    refs: list[str] = []
    dropped_state: list[str] = []
    dropped_no_art: list[str] = []
    for g in groups:
        runs = api.runs(f"{entity}/{project}", filters={"group": g})
        for run in runs:
            if run.state != state:
                dropped_state.append(f"{g}/{run.name}[{run.state}]")
                continue
            arts = [a for a in run.logged_artifacts() if a.type == artifact_type]
            if artifact_alias is not None:
                arts = [a for a in arts if artifact_alias in a.aliases]
            if not arts:
                dropped_no_art.append(f"{g}/{run.name}")
                continue
            # logged_artifacts() is chronological → last is this run's newest.
            refs.append(arts[-1].qualified_name)
    print(
        f"[refs_from_group] groups={groups}: {len(refs)} {state} run(s) "
        f"with a {artifact_type} artifact"
    )
    if dropped_state:
        print(f"  dropped {len(dropped_state)} non-{state}: {dropped_state}")
    if dropped_no_art:
        print(
            f"  dropped {len(dropped_no_art)} {state} run(s) w/o "
            f"{artifact_type}: {dropped_no_art}"
        )
    if not refs:
        raise RuntimeError(
            f"refs_from_group: groups {groups} yielded no {state} runs with a "
            f"{artifact_type} artifact — check the group name(s) / entity / "
            f"project / alias."
        )
    return refs
