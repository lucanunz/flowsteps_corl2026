"""Loader for real-world paired_results.json files.

Several on-disk layouts are supported per (model, task) — different policies
came with different folder conventions. The loader probes the candidate
intermediate paths below and uses the first that exists:

  - <REAL_WORLD_ROOT>/<model>/ab_rollouts/paired_num_steps_eval/<task>/
  - <REAL_WORLD_ROOT>/<model>/ab_rollouts/<task>/                       (xvla)
  - <REAL_WORLD_ROOT>/<model>/paired_num_steps_eval/<task>/             (dit, lap, pi05)
  - <REAL_WORLD_ROOT>/<model>/paired_num_steps/<task>/                  (flower, mibot)
  - <REAL_WORLD_ROOT>/<model>/<task>/

The canonical model key ``openpi`` maps to the on-disk folder ``pi05`` via
``_MODEL_DIR`` below.

Each layout then has one or more `paired_<timestamp>/paired_results.json` files
(when an experiment is interrupted and restarted). We concatenate all
completed pairs across timestamps; pair_ids reset within each timestamp
so a (timestamp, pair_id) tuple is the unique key.

A "completed pair" has both `run_number=1` and `run_number=2` rows, with
condition IDs `{a, b}` mapping to `num_flow_steps == variant_low / variant_high`.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import PairSpec


REAL_WORLD_ROOT = Path(__file__).resolve().parent.parent  # corl2026_logs/real_world/


_LAYOUT_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("ab_rollouts", "paired_num_steps_eval"),
    ("ab_rollouts",),
    ("paired_num_steps_eval",),
    ("paired_num_steps",),
    (),
)

# Canonical model key -> on-disk folder name, when they differ.
_MODEL_DIR: dict[str, str] = {"openpi": "pi05"}


def _task_root(model: str, task: str) -> Path:
    """Locate the task root under whichever intermediate layout exists on disk."""
    base = REAL_WORLD_ROOT / _MODEL_DIR.get(model, model)
    for prefix in _LAYOUT_PREFIXES:
        candidate = base.joinpath(*prefix, task)
        if candidate.is_dir():
            return candidate
    # No layout matched — return the canonical (lap-style) path so callers
    # raise a meaningful FileNotFoundError when they try to read it.
    return base / "ab_rollouts" / "paired_num_steps_eval" / task


def iter_paired_results_files(model: str, task: str) -> list[Path]:
    root = _task_root(model, task)
    if not root.is_dir():
        return []
    return sorted(root.glob("paired_*/paired_results.json"))


def _completed_pairs(rows: list[dict]) -> dict[int, dict[str, dict]]:
    """Group rows by pair_id, keep only pairs with both conditions a and b present."""
    by_pair: dict[int, dict[str, dict]] = {}
    for row in rows:
        pid = row["pair_id"]
        cond = row["condition_id"]
        by_pair.setdefault(pid, {})[cond] = row
    return {pid: recs for pid, recs in by_pair.items() if {"a", "b"} <= recs.keys()}


def load_paired_outcomes(spec: PairSpec) -> tuple[list[bool], list[bool]]:
    """Return aligned per-pair (low, high) binary success outcomes.

    Concatenates completed pairs across every `paired_<timestamp>/` directory
    found for this (model, task). Raises FileNotFoundError if no
    `paired_results.json` exists; returns ``([], [])`` if files exist but
    no pair has both run_number records.
    """
    files = iter_paired_results_files(spec.model, spec.task)
    if not files:
        raise FileNotFoundError(_task_root(spec.model, spec.task))

    seq_low: list[bool] = []
    seq_high: list[bool] = []
    for path in files:
        with path.open() as h:
            data = json.load(h)
        rows = data.get("results", [])
        for pid, recs in sorted(_completed_pairs(rows).items()):
            rec_low = recs["a"]
            rec_high = recs["b"]
            if rec_low["num_flow_steps"] != spec.variant_low:
                raise ValueError(
                    f"{path}: pair_id {pid} condition a has "
                    f"num_flow_steps={rec_low['num_flow_steps']}, "
                    f"expected {spec.variant_low}"
                )
            if rec_high["num_flow_steps"] != spec.variant_high:
                raise ValueError(
                    f"{path}: pair_id {pid} condition b has "
                    f"num_flow_steps={rec_high['num_flow_steps']}, "
                    f"expected {spec.variant_high}"
                )
            seq_low.append(bool(rec_low["manual_success"]))
            seq_high.append(bool(rec_high["manual_success"]))
    return seq_low, seq_high
