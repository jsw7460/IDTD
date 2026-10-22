"""IO utilities for SysID artifacts (datasets, results, commands).

Single abstraction for any pipeline-stage output that may live locally
*or* on wandb — datasets, sysid_result.npz, active_commands.npz, etc.
``resolve_artifact_path`` consumes a spec string (local file or wandb
artifact reference) and returns a local file path; ``upload_artifact``
goes the other way (local file → wandb, returns canonical reference
string for downstream consumers).

Together they let any pipeline stage be run on any server without
manual rsync — the previous stage's output goes to wandb, the next
stage's config takes a wandb reference, and the artifact propagation
chain (e.g. ``theta_init_path`` embedded in ``active_commands.npz``)
travels as portable refs rather than server-local paths.
"""

from consysid.io.artifacts import (
    is_wandb_ref,
    refs_from_group,
    resolve_artifact_path,
    upload_artifact,
)

__all__ = [
    "is_wandb_ref",
    "refs_from_group",
    "resolve_artifact_path",
    "upload_artifact",
]
