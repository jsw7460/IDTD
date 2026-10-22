"""Trajectory clip extraction for SysID cost evaluation.

Segments collected trajectory(s) (npz) into fixed-horizon clips,
respecting episode boundaries so that no clip spans a reset.

Supports loading a *single* trajectory (legacy path) or a *list* of
trajectories (refinement combining passive identification data with
active active-collect data). When multiple datasets are provided two sampling
modes are exposed:

  * ``"pooled"``     — every per-iteration clip slot is filled from the
                       dataset that owns it in proportion to its sample
                       volume ``T_i * N_i``. Approximates the asymptotic
                       MLE estimator over the union of datasets (each
                       clip equally weighted in expectation).
  * ``"stratified"`` — equal number of clips drawn from each dataset
                       per iteration (last dataset absorbs any
                       remainder). Same asymptotic estimator as
                       ``"pooled"`` once iter-count is large, but
                       gives small datasets guaranteed representation
                       per iteration so CMA-ES sees their gradient
                       signal early.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Clip:
    """A single H-step trajectory clip for open-loop evaluation.

    All arrays have time as the first dimension.
    """
    root_pos: np.ndarray        # [H+1, 3]
    root_quat: np.ndarray       # [H+1, 4]
    root_lin_vel: np.ndarray    # [H+1, 3]
    root_ang_vel: np.ndarray    # [H+1, 3]
    joint_pos: np.ndarray       # [H+1, num_joints]
    joint_vel: np.ndarray       # [H+1, num_joints]
    joint_torques: np.ndarray   # [H+1, num_joints]
    actions: np.ndarray         # [H, num_actions]  (raw policy outputs)
    horizon: int
    motor_rpm: np.ndarray | None = None
    """[H+1, num_motors] hidden motor-RPM state (thrust robots only —
    recorded by the collect loop when the robot has a propeller
    actuator). The Newton open-loop evaluator restores ``motor_rpm[0]``
    into the actuator at clip start; without it the RPM state would
    start at whatever the previous clip left behind and the whole 1-5
    step horizon sits inside the resulting transient. ``None`` for
    joint-actuated robots."""


_ARRAY_KEYS = (
    "root_pos", "root_quat", "root_lin_vel", "root_ang_vel",
    "joint_pos", "joint_vel", "joint_torques",
    "actions", "dones",
)


class ClipDataset:
    """Extracts H-horizon clips from one or more collected trajectory npz files.

    Handles episode boundaries via the ``dones`` mask: clips never span
    a reset boundary. With multiple datasets, ``sample_clips()``
    distributes the per-iteration clip budget across datasets according
    to ``sampling``.
    """

    def __init__(
        self,
        npz_paths: str | list[str],
        horizon_range: tuple[int, int] = (25, 75),
        num_clips: int = 50,
        seed: int = 42,
        sampling: str = "pooled",
    ):
        # Normalize to a list. ``str`` and ``list[str]`` both flow
        # through the same code path so single-trajectory callers stay
        # bit-identical to the legacy single-npz behaviour.
        if isinstance(npz_paths, (str, bytes)):
            npz_paths = [str(npz_paths)]
        else:
            npz_paths = [str(p) for p in npz_paths]
        if len(npz_paths) == 0:
            raise ValueError("ClipDataset requires at least one npz path.")

        if sampling not in ("pooled", "stratified"):
            raise ValueError(
                f"sampling must be 'pooled' or 'stratified', got {sampling!r}"
            )

        self.npz_paths = npz_paths
        self.horizon_range = horizon_range
        self.num_clips = num_clips
        self.sampling = sampling
        self.rng = np.random.RandomState(seed)

        # Load every dataset into a list of dicts. Per-dataset arrays
        # stay separate (different ``T`` is allowed across datasets, so
        # we cannot stack along the time axis).
        self._datasets: list[dict] = []
        ref_dt: float | None = None
        ref_nj: int | None = None
        ref_na: int | None = None
        for path in npz_paths:
            data = np.load(path, allow_pickle=True)
            ds = {k: data[k] for k in _ARRAY_KEYS}
            # Optional hidden-actuator-state channel (thrust robots).
            ds["motor_rpm"] = (
                data["motor_rpm"] if "motor_rpm" in data.files else None
            )
            ds["dt"] = float(data["dt"])
            ds["T"], ds["N"] = ds["root_pos"].shape[:2]
            ds["num_episodes"] = (
                int(data["num_episodes"])
                if "num_episodes" in data.files else 0
            )
            ds["path"] = path

            # Cross-dataset consistency — same dt / num_joints /
            # num_actions are required for the clips to be drop-in
            # interchangeable in the open-loop evaluator.
            if ref_dt is None:
                ref_dt = ds["dt"]
                ref_nj = int(ds["joint_pos"].shape[-1])
                ref_na = int(ds["actions"].shape[-1])
            else:
                if not np.isclose(ds["dt"], ref_dt):
                    raise ValueError(
                        f"dataset {path}: dt={ds['dt']} does not match "
                        f"first dataset's dt={ref_dt}"
                    )
                nj = int(ds["joint_pos"].shape[-1])
                na = int(ds["actions"].shape[-1])
                if nj != ref_nj or na != ref_na:
                    raise ValueError(
                        f"dataset {path}: (num_joints={nj}, num_actions={na}) "
                        f"does not match first dataset's "
                        f"(num_joints={ref_nj}, num_actions={ref_na})"
                    )
            self._datasets.append(ds)

        self.dt = ref_dt
        # Aggregate stats (legacy attributes mirror the single-dataset
        # case; for multi-dataset they sum across).
        self.T = sum(d["T"] for d in self._datasets)         # total step-count
        self.N = sum(d["N"] for d in self._datasets)         # total env-count
        self.num_episodes = sum(d["num_episodes"] for d in self._datasets)

        # Per-iteration clip allocation across datasets, computed once
        # at init since neither sampling mode nor dataset sizes change
        # mid-run. ``_per_ds_count`` sums to ``num_clips``.
        self._per_ds_count = self._allocate_clip_budget()

        log_lines = [
            f"ClipDataset: {len(self._datasets)} dataset(s), "
            f"sampling={sampling!r}, num_clips={num_clips}",
        ]
        for i, (ds, count) in enumerate(zip(self._datasets, self._per_ds_count)):
            log_lines.append(
                f"  [{i}] T={ds['T']:5d}  N={ds['N']:4d}  "
                f"episodes={ds['num_episodes']:4d}  per-iter clips={count:4d}  "
                f"path={ds['path']}"
            )
        print("\n".join(log_lines))

    # ── Per-iteration sampling ────────────────────────────────────────

    def _allocate_clip_budget(self) -> list[int]:
        """Decide how many clips to draw from each dataset per iteration.

        ``"pooled"``     — proportional to ``T_i * N_i`` (sample volume),
                           so each (env, t) cell has equal probability
                           across the union.
        ``"stratified"`` — equal share, last dataset absorbs remainder.
        """
        n_ds = len(self._datasets)
        if n_ds == 1:
            return [self.num_clips]

        if self.sampling == "pooled":
            volumes = np.array(
                [ds["T"] * ds["N"] for ds in self._datasets], dtype=np.float64,
            )
            shares = volumes / volumes.sum()
            counts = [int(round(self.num_clips * s)) for s in shares]
        else:  # "stratified"
            base = self.num_clips // n_ds
            counts = [base] * n_ds

        # Reconcile rounding so total == num_clips. Distribute any
        # difference onto the largest dataset (pooled) or the first
        # dataset (stratified) — both are deterministic and keep the
        # allocation close to its intended ratio.
        diff = self.num_clips - sum(counts)
        if diff != 0:
            target_idx = (
                int(np.argmax([ds["T"] * ds["N"] for ds in self._datasets]))
                if self.sampling == "pooled" else 0
            )
            counts[target_idx] += diff
        return counts

    def sample_clips(self) -> list[Clip]:
        """Draw a fresh per-iteration batch of clips from every dataset.

        Each dataset contributes ``self._per_ds_count[i]`` clips,
        sampled independently with the standard episode-boundary-
        respecting logic. The returned list is the concatenation in
        dataset order — downstream consumers iterate over it
        positionally and don't need to know which dataset a clip came
        from.
        """
        clips: list[Clip] = []
        for ds_idx, count in enumerate(self._per_ds_count):
            if count > 0:
                clips.extend(self._sample_from_dataset(ds_idx, count))
        return clips

    def _sample_from_dataset(self, ds_idx: int, count: int) -> list[Clip]:
        """Sample ``count`` clips from a single dataset, respecting
        episode boundaries (no clip may span a ``done==True`` step).
        """
        ds = self._datasets[ds_idx]
        h_min, h_max = self.horizon_range
        clips: list[Clip] = []

        # Oversample tries in case dones force us to discard candidates.
        for _ in range(count * 10):
            if len(clips) >= count:
                break

            env_idx = self.rng.randint(0, ds["N"])
            h = self.rng.randint(h_min, h_max + 1)

            # Need H+1 states and H actions, so max start = T - H - 1.
            max_start = ds["T"] - h - 1
            if max_start < 0:
                continue

            t0 = self.rng.randint(0, max_start + 1)

            # Reject if a reset falls inside [t0, t0+H): step t+1 after
            # ``done[t]`` is a reset state, so the t→t+1 transition is
            # invalid for open-loop replay.
            if ds["dones"][t0:t0 + h, env_idx].any():
                continue

            clips.append(Clip(
                root_pos=ds["root_pos"][t0:t0 + h + 1, env_idx],
                root_quat=ds["root_quat"][t0:t0 + h + 1, env_idx],
                root_lin_vel=ds["root_lin_vel"][t0:t0 + h + 1, env_idx],
                root_ang_vel=ds["root_ang_vel"][t0:t0 + h + 1, env_idx],
                joint_pos=ds["joint_pos"][t0:t0 + h + 1, env_idx],
                joint_vel=ds["joint_vel"][t0:t0 + h + 1, env_idx],
                joint_torques=ds["joint_torques"][t0:t0 + h + 1, env_idx],
                actions=ds["actions"][t0:t0 + h, env_idx],
                horizon=h,
                motor_rpm=(
                    ds["motor_rpm"][t0:t0 + h + 1, env_idx]
                    if ds["motor_rpm"] is not None else None
                ),
            ))

        if len(clips) < count:
            print(
                f"WARNING: dataset {ds_idx} ({ds['path']}) only yielded "
                f"{len(clips)}/{count} clips. Consider lowering horizon "
                f"or collecting more episodes for this dataset."
            )

        return clips

    # ── Legacy convenience accessors ──────────────────────────────────

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> Clip:
        return self.sample_clips()[idx]
