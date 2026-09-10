"""Load paired per-task episode outcomes from the RoboTwin paired runs.

Each run directory holds one subdirectory per task with `results.json`, a
JSON array of `{seed, success, steps}` records. We intersect seeds across
the two variants so STEP sees true paired binary sequences sorted by seed.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import PairSpec


def _load_task_results(run_dir: Path, task: str) -> dict[int, bool]:
    path = run_dir / task / "results.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, bool] = {}
    for rec in records:
        seed = int(rec["seed"])
        if seed in out:
            raise ValueError(f"Duplicate seed {seed} in {path}")
        out[seed] = bool(int(rec["success"]))
    return out


def _collect_per_task(run_dirs: tuple[Path, ...]) -> dict[str, dict[int, bool]]:
    """Walk all run dirs and merge per-task seed→success maps. A task is
    expected to appear in at most one dir per side; if the same task has
    `results.json` in two dirs we raise to flag the ambiguity rather than
    silently choosing one."""
    out: dict[str, dict[int, bool]] = {}
    source: dict[str, Path] = {}
    for run_dir in run_dirs:
        for task_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            task = task_dir.name
            if not (task_dir / "results.json").exists():
                continue
            if task in out:
                raise ValueError(
                    f"Task {task!r} has results.json in both "
                    f"{source[task]} and {run_dir}; aborting to avoid "
                    "silent merge of conflicting paired runs."
                )
            out[task] = _load_task_results(run_dir, task)
            source[task] = run_dir
    return out


def load_paired_episodes(
    spec: PairSpec,
) -> tuple[dict[str, tuple[list[bool], list[bool]]], list[str]]:
    """Return (`{task: (seq_low, seq_high)}`, skipped_tasks).

    Walks every dir in `spec.low_run_dirs` and `spec.high_run_dirs`, taking
    the per-task `results.json` from whichever dir holds it. Skips any task
    that is missing on either side.
    """
    lo_all = _collect_per_task(spec.low_run_dirs)
    hi_all = _collect_per_task(spec.high_run_dirs)
    all_tasks = sorted(set(lo_all) | set(hi_all))
    skipped: set[str] = set()

    out: dict[str, tuple[list[bool], list[bool]]] = {}
    for task in all_tasks:
        lo = lo_all.get(task)
        hi = hi_all.get(task)
        if lo is None or hi is None:
            skipped.add(task)
            continue
        common = sorted(set(lo) & set(hi))
        if not common:
            skipped.add(task)
            continue
        out[task] = (
            [lo[s] for s in common],
            [hi[s] for s in common],
        )
    return out, sorted(skipped)


def concat_overall(per_task: dict[str, tuple[list[bool], list[bool]]]) -> tuple[list[bool], list[bool]]:
    """Concatenate per-task paired sequences in (task, seed) order — deterministic."""
    seq_lo: list[bool] = []
    seq_hi: list[bool] = []
    for task in sorted(per_task):
        lo, hi = per_task[task]
        seq_lo.extend(lo)
        seq_hi.extend(hi)
    return seq_lo, seq_hi
