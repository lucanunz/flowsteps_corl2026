"""Loaders for CALVIN ABC-D runs.

For XR-0 we reuse the per-sequence pickles via `analyze_calvin_steps`.
For X-VLA we recover per-rollout 0-5 completion scores from the rank-style
`log.txt` (one line per rollout, with running averages of chain SR@k and
mean completion). Summing the differenced cumulative counts across all 5
chain levels yields an exact integer score per rollout (rounding errors in
individual columns cancel out).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent          # tri_statistics_STEP/
_CALVIN_DIR = _PKG_DIR.parent                       # calvin/
_ROOT_DIR = _CALVIN_DIR.parent                      # corl2026_logs/

for _p in (_CALVIN_DIR, _ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analyze_calvin_steps import (  # type: ignore  # noqa: E402
    DEFAULT_RUNS,
    RunResult,
    align_paired_sequence_results,
    load_results,
)

from .config import ANALYZE_KEY, PairSpec


CALVIN_DATA_ROOT = _CALVIN_DIR / "abc_d"


# ---------------------------------------------------------------------------
# Run loaders
# ---------------------------------------------------------------------------

def _find_run_spec(analyze_key: str, step_label: str) -> dict:
    for entry in DEFAULT_RUNS[analyze_key]:
        if entry["label"] == step_label:
            return entry
    raise KeyError(f"No DEFAULT_RUNS entry for model={analyze_key!r} step={step_label!r}")


def load_run(model: str, steps: int) -> RunResult:
    analyze_key = ANALYZE_KEY[model]
    entry = _find_run_spec(analyze_key, str(steps))
    result_path = CALVIN_DATA_ROOT / entry["relative_path"]
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    return load_results(
        result_path=result_path,
        model_key=analyze_key,
        step_label=entry["label"],
        is_baseline=bool(entry["is_baseline"]),
    )


def load_pair(spec: PairSpec) -> tuple[RunResult, RunResult]:
    return load_run(spec.model, spec.variant_low), load_run(spec.model, spec.variant_high)


# ---------------------------------------------------------------------------
# X-VLA log.txt parser
# ---------------------------------------------------------------------------

_XVLA_LINE_RE = re.compile(r"^(\d+):\s*(.*?)\s*\|\|\s*$")
_XVLA_COL_RE = re.compile(r"(\d+)/5\s*:\s*([\d.]+)%")


def parse_xvla_log(log_path: Path, use_json: bool = True) -> list[int]:
    """Per-rollout 0-5 completion scores for an X-VLA CALVIN run.

    Prefers the `completion_scores.json` sidecar beside the log; the
    reconstruction below is the fallback, and is what produced that sidecar.

    Each line has the form:
        t: 1/5 : a% | 2/5 : b% | 3/5 : c% | 4/5 : d% | 5/5 : e% | 6/5 : f% ||
    where a..e are running fractions (out of t+1 rollouts) of rollouts that
    reached at least k subtasks (k = 1..5), and f is the running mean
    completion expressed as a percentage. We use the 5 chain-level columns,
    not the mean column: per-rollout score(t) = Σ_k=1..5 (n_k(t) − n_k(t−1))
    with n_k(t) = round(a_k(t) / 100 × (t+1)). Per-column rounding errors
    cancel in the sum, giving exact integer scores.
    """
    if use_json:
        sidecar = log_path.parent / "completion_scores.json"
        if sidecar.exists():
            scores = json.loads(sidecar.read_text(encoding="utf-8")).get("completion_scores")
            if scores:
                return [int(v) for v in scores]
    rows: dict[int, dict[int, float]] = {}
    with log_path.open() as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = _XVLA_LINE_RE.match(line)
            if not m:
                continue
            idx = int(m.group(1))
            cols = {int(a): float(b) for a, b in _XVLA_COL_RE.findall(m.group(2))}
            rows[idx] = cols
    if not rows:
        raise ValueError(f"No rollout lines parsed from {log_path}")
    n = max(rows.keys()) + 1
    if set(rows.keys()) != set(range(n)):
        missing = sorted(set(range(n)) - set(rows.keys()))
        raise ValueError(f"Missing rollout indices in {log_path}: {missing[:10]}")

    cumcount = {
        k: [round(rows[t][k] / 100.0 * (t + 1)) for t in range(n)]
        for k in range(1, 6)
    }
    scores: list[int] = []
    for t in range(n):
        s = 0
        for k in range(1, 6):
            prev = cumcount[k][t - 1] if t > 0 else 0
            s += cumcount[k][t] - prev
        if s < 0 or s > 5:
            raise ValueError(
                f"Reconstructed out-of-range score {s} at rollout {t} in {log_path}"
            )
        scores.append(s)
    return scores


def _xvla_log_path(steps: int) -> Path:
    return CALVIN_DATA_ROOT / "X-VLA" / f"{steps}step" / "log.txt"


def _xvla_completion_scores(spec: PairSpec) -> tuple[list[int], list[int]]:
    low = parse_xvla_log(_xvla_log_path(spec.variant_low))
    high = parse_xvla_log(_xvla_log_path(spec.variant_high))
    return low, high


# ---------------------------------------------------------------------------
# Flower per-episode JSON parser
# ---------------------------------------------------------------------------

_FLOWER_PER_EPISODE_ROOT = CALVIN_DATA_ROOT / "flower" / "per-episode-logs"


def _flower_per_episode_path(steps: int) -> Path:
    """Locate Flower's per-episode JSON for a given step count.

    Layout: flower/per-episode-logs/abc_d_{N}step/logs/<timestamp>/per_episode_results.json
    The timestamp folder is opaque; we glob for the JSON file.
    """
    run_root = _FLOWER_PER_EPISODE_ROOT / f"abc_d_{steps}step" / "logs"
    matches = sorted(run_root.rglob("per_episode_results.json"))
    if not matches:
        raise FileNotFoundError(run_root / "*" / "per_episode_results.json")
    if len(matches) > 1:
        # Multiple timestamped runs — pick the most recent (lexicographic sort
        # on timestamp directory names works for the ISO-like format used).
        matches.sort()
    return matches[-1]


def _canonical_flower_key(episode: dict) -> str:
    payload = [episode["initial_state"], episode["eval_sequence"]]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _load_flower_per_episode(steps: int) -> list[dict]:
    path = _flower_per_episode_path(steps)
    data = json.load(path.open())
    # Single top-level key (typically "0") wrapping a list of episode dicts.
    if isinstance(data, dict):
        payload = next(iter(data.values()))
    else:
        payload = data
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected Flower per_episode_results schema in {path}")
    return payload


def _flower_paired_scores(spec: PairSpec) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    low_eps = _load_flower_per_episode(spec.variant_low)
    high_eps = _load_flower_per_episode(spec.variant_high)
    low_map = {_canonical_flower_key(e): int(e["result"]) for e in low_eps}
    high_map = {_canonical_flower_key(e): int(e["result"]) for e in high_eps}
    shared = sorted(set(low_map) & set(high_map))
    if not shared:
        return None
    low_scores = tuple(low_map[k] for k in shared)
    high_scores = tuple(high_map[k] for k in shared)
    return low_scores, high_scores


# ---------------------------------------------------------------------------
# Paired-data dispatch
# ---------------------------------------------------------------------------

def _mibot_paired_scores(
    low: RunResult, high: RunResult
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    paired = align_paired_sequence_results(high, low)
    if paired is None:
        return None
    high_results, low_results = paired
    return low_results, high_results


def paired_completion_scores(
    spec: PairSpec,
    low: RunResult,
    high: RunResult,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return aligned per-rollout (low, high) 0-5 completion scores, or None."""
    if spec.model == "mibot":
        return _mibot_paired_scores(low, high)
    if spec.model == "xvla":
        try:
            low_scores, high_scores = _xvla_completion_scores(spec)
        except (FileNotFoundError, ValueError):
            return None
        if len(low_scores) != len(high_scores):
            return None
        return tuple(low_scores), tuple(high_scores)
    if spec.model == "flower":
        try:
            return _flower_paired_scores(spec)
        except (FileNotFoundError, ValueError):
            return None
    return None


def paired_sr5_sequences(
    spec: PairSpec,
    low: RunResult,
    high: RunResult,
) -> tuple[list[bool], list[bool]] | None:
    """Return aligned (seq_low, seq_high) of SR@5 binary outcomes, or None."""
    paired_scores = paired_completion_scores(spec, low, high)
    if paired_scores is None:
        return None
    low_scores, high_scores = paired_scores
    return [int(s) >= 5 for s in low_scores], [int(s) >= 5 for s in high_scores]
