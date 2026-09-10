"""Load per-task seed-level outcomes from the paired RoboTwin runs.

Each run directory holds one subdirectory per task. Inside, `results.json` is a
JSON array of records `{seed, success, steps}` — one per evaluated episode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import RunSpec


@dataclass(frozen=True)
class TaskOutcomes:
    task: str
    seeds: tuple[int, ...]
    success_low: tuple[int, ...]   # 0/1, indexed by seeds
    success_high: tuple[int, ...]
    steps_low: tuple[int, ...]     # env steps, indexed by seeds
    steps_high: tuple[int, ...]


@dataclass(frozen=True)
class EpisodeResult:
    success: int
    steps: int


def _load_task_results(run_dir: Path, task: str) -> dict[int, EpisodeResult]:
    path = run_dir / task / "results.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, EpisodeResult] = {}
    for rec in records:
        seed = int(rec["seed"])
        if seed in out:
            raise ValueError(f"Duplicate seed {seed} in {path}")
        out[seed] = EpisodeResult(
            success=1 if int(rec["success"]) else 0,
            steps=int(rec["steps"]),
        )
    return out


def _collect_per_task(run_dirs: tuple[Path, ...]) -> dict[str, dict[int, EpisodeResult]]:
    """Merge per-task seed→success maps across a tuple of run dirs.

    A task is expected to have `results.json` in at most one of the dirs on
    a given side; if two dirs both contain it we raise to flag the ambiguity.
    """
    out: dict[str, dict[int, EpisodeResult]] = {}
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


def load_paired_outcomes(
    low: RunSpec, high: RunSpec,
) -> tuple[list[TaskOutcomes], list[str]]:
    """Return (per-task paired outcomes, skipped tasks).

    Walks every dir in `low.paths` and `high.paths` and merges per-task
    seed→success maps. Skips any task missing on either side.
    """
    lo_all = _collect_per_task(low.paths)
    hi_all = _collect_per_task(high.paths)
    all_tasks = sorted(set(lo_all) | set(hi_all))
    skipped: set[str] = set()

    out: list[TaskOutcomes] = []
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
        out.append(
            TaskOutcomes(
                task=task,
                seeds=tuple(common),
                success_low=tuple(lo[s].success for s in common),
                success_high=tuple(hi[s].success for s in common),
                steps_low=tuple(lo[s].steps for s in common),
                steps_high=tuple(hi[s].steps for s in common),
            )
        )
    return out, sorted(skipped)
