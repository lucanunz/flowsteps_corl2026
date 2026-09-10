"""Shared analyze_calvin_task_completion data loaders and statistical primitives; no table output."""

from __future__ import annotations

import math
from dataclasses import dataclass


import numpy as np
from scipy import stats

from analyze_calvin_steps import RunResult, sequence_length_distribution


COMPLETION_LEVELS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
DIRICHLET_NUM_SAMPLES = 20000
DIRICHLET_SEED = 0
ALPHA = 0.05


@dataclass(frozen=True)
class CompletionStats:
    counts: tuple[int, int, int, int, int, int]
    n: int
    mean: float
    variance: float
    is_paired_capable: bool


@dataclass(frozen=True)
class WelchTestResult:
    delta: float
    ci_low: float
    ci_high: float
    t_statistic: float
    df: float
    p_value: float


def completion_counts_from_raw(raw_results: tuple[int, ...]) -> tuple[int, int, int, int, int, int]:
    counts = [0] * len(COMPLETION_LEVELS)
    for value in raw_results:
        if value < 0 or value > 5:
            raise ValueError(f"Raw completion value out of range: {value}")
        counts[value] += 1
    return tuple(counts)  # type: ignore[return-value]


def completion_counts_from_chain_sr(run: RunResult) -> tuple[int, int, int, int, int, int]:
    distribution = sequence_length_distribution(run)
    raw = [distribution[level] * run.num_sequences for level in COMPLETION_LEVELS]
    floored = [int(math.floor(value)) for value in raw]
    remainders = [value - floor_value for value, floor_value in zip(raw, floored)]
    deficit = run.num_sequences - sum(floored)
    if deficit < 0:
        raise ValueError(f"Reconstruction over-counted for {run.result_path}")
    order = sorted(range(len(remainders)), key=lambda idx: remainders[idx], reverse=True)
    counts = list(floored)
    for idx in order[:deficit]:
        counts[idx] += 1
    if sum(counts) != run.num_sequences:
        raise ValueError(f"Failed to reconcile reconstructed counts for {run.result_path}")
    return tuple(counts)  # type: ignore[return-value]


def build_completion_stats(run: RunResult) -> CompletionStats:
    if run.raw_sequence_results is not None:
        counts = completion_counts_from_raw(run.raw_sequence_results)
        is_paired_capable = True
    else:
        counts = completion_counts_from_chain_sr(run)
        is_paired_capable = False

    n = sum(counts)
    if n == 0:
        raise ValueError(f"Empty completion distribution for {run.result_path}")
    mean = sum(level * count for level, count in zip(COMPLETION_LEVELS, counts)) / n
    if n > 1:
        sum_sq = sum((level ** 2) * count for level, count in zip(COMPLETION_LEVELS, counts))
        variance = (sum_sq - n * mean * mean) / (n - 1)
    else:
        variance = 0.0
    variance = max(variance, 0.0)

    if not is_paired_capable:
        reconstructed_mean = mean
        reported_mean = float(run.avg_seq_len)
        if abs(reconstructed_mean - reported_mean) > 1e-2:
            raise ValueError(
                "Reconstruction sanity failed for "
                f"{run.result_path}: reconstructed mean {reconstructed_mean:.4f} "
                f"vs reported avg_seq_len {reported_mean:.4f}"
            )

    return CompletionStats(
        counts=counts,
        n=n,
        mean=mean,
        variance=variance,
        is_paired_capable=is_paired_capable,
    )


def welch_t_test(reference: CompletionStats, baseline: CompletionStats) -> WelchTestResult:
    delta = reference.mean - baseline.mean
    se_squared = reference.variance / reference.n + baseline.variance / baseline.n
    if se_squared <= 0.0:
        return WelchTestResult(
            delta=delta,
            ci_low=delta,
            ci_high=delta,
            t_statistic=0.0 if delta == 0.0 else math.inf,
            df=float(max(reference.n + baseline.n - 2, 1)),
            p_value=1.0 if delta == 0.0 else 0.0,
        )

    se = math.sqrt(se_squared)
    t_statistic = delta / se
    numerator = se_squared ** 2
    denominator = (
        (reference.variance / reference.n) ** 2 / max(reference.n - 1, 1)
        + (baseline.variance / baseline.n) ** 2 / max(baseline.n - 1, 1)
    )
    df = numerator / denominator if denominator > 0 else float(reference.n + baseline.n - 2)
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), df))
    critical_value = float(stats.t.ppf(1.0 - ALPHA / 2.0, df))
    margin = critical_value * se

    return WelchTestResult(
        delta=delta,
        ci_low=delta - margin,
        ci_high=delta + margin,
        t_statistic=t_statistic,
        df=df,
        p_value=p_value,
    )


def dirichlet_posterior_mean_samples(
    counts: tuple[int, int, int, int, int, int],
    num_samples: int = DIRICHLET_NUM_SAMPLES,
    seed: int = DIRICHLET_SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    alpha = np.array([count + 1 for count in counts], dtype=float)
    samples = rng.dirichlet(alpha, size=num_samples)
    levels = np.array(COMPLETION_LEVELS, dtype=float)
    return samples @ levels


