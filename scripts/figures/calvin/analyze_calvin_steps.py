"""Shared analyze_calvin_steps data loaders and statistical primitives; no table output."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path



DEFAULT_RUNS = {
    "flower": [
        {"label": "1", "relative_path": "flower/abc_d_1step/results.json", "is_baseline": True},
        {"label": "4", "relative_path": "flower/abc_d_4step/results.json", "is_baseline": False},
    ],
    "mibot": [
        {"label": "1", "relative_path": "mibot/rerun_abc_d_numsteps1/results.json", "is_baseline": True},
        {"label": "5", "relative_path": "mibot/rerun_abc_d_numsteps5/results.json", "is_baseline": False},
    ],
    "x-vla": [
        {"label": "1", "relative_path": "X-VLA/1step/results.json", "is_baseline": True},
        {"label": "10", "relative_path": "X-VLA/10step/results.json", "is_baseline": False},
    ],
}

DISPLAY_NAMES = {
    "flower": "Flower",
    "mibot": "XR-0",
    "x-vla": "X-VLA",
}

CHAIN_LEVELS = ["1", "2", "3", "4", "5"]


@dataclass(frozen=True)
class RunResult:
    model_key: str
    model_name: str
    step_label: str
    result_path: Path
    num_sequences: int
    avg_seq_len: float
    chain_sr: dict[str, float]
    task_success_rate: dict[str, float]
    task_counts: dict[str, tuple[int, int]]
    is_baseline: bool
    raw_sequence_keys: tuple[str, ...] | None
    raw_sequence_results: tuple[int, ...] | None


@dataclass(frozen=True)
class PairedSequenceData:
    sequence_keys: tuple[str, ...]
    results: tuple[int, ...]


def load_results(result_path: Path, model_key: str, step_label: str, is_baseline: bool) -> RunResult:
    with result_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not raw:
        raise ValueError(f"Empty results file: {result_path}")

    payload = next(iter(raw.values()))
    task_info = payload["task_info"]
    raw_sequence_data = load_raw_sequence_data(result_path)

    task_counts = {
        task_name: (int(task_result["success"]), int(task_result["total"]))
        for task_name, task_result in task_info.items()
    }
    task_success_rate = {
        task_name: success / total if total else 0.0
        for task_name, (success, total) in task_counts.items()
    }

    return RunResult(
        model_key=model_key,
        model_name=DISPLAY_NAMES[model_key],
        step_label=step_label,
        result_path=result_path,
        num_sequences=int(payload.get("num_sequences", 1000)),
        avg_seq_len=float(payload["avg_seq_len"]),
        chain_sr={level: float(payload["chain_sr"][level]) for level in CHAIN_LEVELS},
        task_success_rate=task_success_rate,
        task_counts=task_counts,
        is_baseline=is_baseline,
        raw_sequence_keys=raw_sequence_data.sequence_keys if raw_sequence_data else None,
        raw_sequence_results=raw_sequence_data.results if raw_sequence_data else None,
    )


def canonical_sequence_key(sequence: object) -> str:
    return json.dumps(sequence, sort_keys=True, separators=(",", ":"))


def _load_rank_payloads(run_dir: Path) -> list[tuple[str, dict]]:
    """Return each rank's `{results, sequences}` payload for one CALVIN run.

    Prefers `rank_results.json`, which holds every rank keyed by its original
    file stem; falls back to the `rank_*_results.pkl` pickles so the loader
    still works against untidied copies of the logs.
    """
    merged = run_dir / "rank_results.json"
    if merged.exists():
        return list(json.loads(merged.read_text(encoding="utf-8")).items())
    return [(path.stem, pickle.loads(path.read_bytes()))
            for path in sorted(run_dir.glob("rank_*_results.pkl"))]


def load_raw_sequence_data(result_path: Path) -> PairedSequenceData | None:
    raw_paths = _load_rank_payloads(result_path.parent)
    if not raw_paths:
        return None

    sequence_keys: list[str] = []
    results: list[int] = []
    for raw_path, payload in raw_paths:
        shard_sequences = payload.get("sequences")
        shard_results = payload.get("results")
        if not isinstance(shard_sequences, list) or not isinstance(shard_results, list):
            raise ValueError(f"Unexpected raw result schema in {raw_path}")
        if len(shard_sequences) != len(shard_results):
            raise ValueError(f"Mismatched sequences/results lengths in {raw_path}")
        sequence_keys.extend(canonical_sequence_key(sequence) for sequence in shard_sequences)
        results.extend(int(result) for result in shard_results)

    return PairedSequenceData(sequence_keys=tuple(sequence_keys), results=tuple(results))


def sequence_length_distribution(run: RunResult) -> dict[int, float]:
    p_ge_1 = run.chain_sr["1"]
    p_ge_2 = run.chain_sr["2"]
    p_ge_3 = run.chain_sr["3"]
    p_ge_4 = run.chain_sr["4"]
    p_ge_5 = run.chain_sr["5"]

    distribution = {
        0: 1.0 - p_ge_1,
        1: p_ge_1 - p_ge_2,
        2: p_ge_2 - p_ge_3,
        3: p_ge_3 - p_ge_4,
        4: p_ge_4 - p_ge_5,
        5: p_ge_5,
    }
    clipped_distribution = {length: max(probability, 0.0) for length, probability in distribution.items()}
    total_probability = sum(clipped_distribution.values())
    if total_probability <= 0.0:
        raise ValueError(f"Invalid chain success rates for {run.result_path}")
    return {
        length: probability / total_probability
        for length, probability in clipped_distribution.items()
    }


def align_paired_sequence_results(
    reference: RunResult,
    baseline: RunResult,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if reference.raw_sequence_keys is None or reference.raw_sequence_results is None:
        return None
    if baseline.raw_sequence_keys is None or baseline.raw_sequence_results is None:
        return None

    reference_pairs = list(zip(reference.raw_sequence_keys, reference.raw_sequence_results, strict=True))
    baseline_pairs = list(zip(baseline.raw_sequence_keys, baseline.raw_sequence_results, strict=True))

    if len(reference_pairs) != len(baseline_pairs):
        return None

    if reference.raw_sequence_keys == baseline.raw_sequence_keys:
        return reference.raw_sequence_results, baseline.raw_sequence_results

    reference_pairs.sort(key=lambda item: item[0])
    baseline_pairs.sort(key=lambda item: item[0])
    if [key for key, _ in reference_pairs] != [key for key, _ in baseline_pairs]:
        return None

    return (
        tuple(result for _, result in reference_pairs),
        tuple(result for _, result in baseline_pairs),
    )


