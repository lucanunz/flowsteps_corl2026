"""Build per-task and overall result records from paired outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_runs import TaskOutcomes
from .stats import (
    McNemarResult,
    benjamini_hochberg,
    bonferroni,
    cluster_bootstrap_delta_ci,
    holm,
    mcnemar,
    paired_delta_ci,
    wilcoxon_signed_rank,
    wilson_interval,
)


@dataclass(frozen=True)
class TaskResult:
    task: str
    n: int
    successes_low: int
    successes_high: int
    steps_low: int
    steps_high: int
    rate_low: float
    rate_high: float
    throughput_low: float
    throughput_high: float
    wilson_low: tuple[float, float]
    wilson_high: tuple[float, float]
    delta: float
    mcnemar: McNemarResult
    q_bh: float = 1.0
    q_holm: float = 1.0
    q_bonf: float = 1.0


@dataclass(frozen=True)
class OverallResult:
    n_tasks: int
    n_episodes: int
    successes_low: int
    successes_high: int
    rate_low: float
    rate_high: float
    delta_pooled: float
    delta_pooled_ci: tuple[float, float]
    mcnemar_pooled: McNemarResult
    # Task-mean view (each task weighted equally).
    mean_rate_low: float
    mean_rate_high: float
    mean_delta: float
    mean_delta_ci: tuple[float, float]
    wilcoxon_stat: float
    wilcoxon_p: float


def build_task_result(outcome: TaskOutcomes) -> TaskResult:
    low = np.asarray(outcome.success_low, dtype=int)
    high = np.asarray(outcome.success_high, dtype=int)
    n = int(low.size)
    s_low = int(low.sum())
    s_high = int(high.sum())
    steps_low = int(np.asarray(outcome.steps_low, dtype=int).sum())
    steps_high = int(np.asarray(outcome.steps_high, dtype=int).sum())
    rate_low = s_low / n
    rate_high = s_high / n
    return TaskResult(
        task=outcome.task,
        n=n,
        successes_low=s_low,
        successes_high=s_high,
        steps_low=steps_low,
        steps_high=steps_high,
        rate_low=rate_low,
        rate_high=rate_high,
        throughput_low=1000.0 * s_low / steps_low if steps_low else 0.0,
        throughput_high=1000.0 * s_high / steps_high if steps_high else 0.0,
        wilson_low=wilson_interval(s_low, n),
        wilson_high=wilson_interval(s_high, n),
        delta=rate_high - rate_low,
        mcnemar=mcnemar(low, high),
    )


def apply_corrections(tasks: list[TaskResult]) -> list[TaskResult]:
    pvalues = [t.mcnemar.p_exact for t in tasks]
    q_bh = benjamini_hochberg(pvalues)
    q_h = holm(pvalues)
    q_b = bonferroni(pvalues)
    out: list[TaskResult] = []
    for t, qbh, qh, qb in zip(tasks, q_bh, q_h, q_b, strict=True):
        out.append(TaskResult(
            task=t.task, n=t.n,
            successes_low=t.successes_low, successes_high=t.successes_high,
            steps_low=t.steps_low, steps_high=t.steps_high,
            rate_low=t.rate_low, rate_high=t.rate_high,
            throughput_low=t.throughput_low, throughput_high=t.throughput_high,
            wilson_low=t.wilson_low, wilson_high=t.wilson_high,
            delta=t.delta, mcnemar=t.mcnemar,
            q_bh=qbh, q_holm=qh, q_bonf=qb,
        ))
    return out


def build_overall(
    outcomes: list[TaskOutcomes],
    tasks: list[TaskResult],
    *,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
) -> OverallResult:
    per_low = [np.asarray(o.success_low, dtype=int) for o in outcomes]
    per_high = [np.asarray(o.success_high, dtype=int) for o in outcomes]
    pooled_low = np.concatenate(per_low)
    pooled_high = np.concatenate(per_high)
    n_episodes = int(pooled_low.size)
    s_low = int(pooled_low.sum())
    s_high = int(pooled_high.sum())
    pooled_mcnemar = mcnemar(pooled_low, pooled_high)

    pooled_delta, pooled_lo, pooled_hi = cluster_bootstrap_delta_ci(
        per_low, per_high,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )

    rates_low = np.array([t.rate_low for t in tasks])
    rates_high = np.array([t.rate_high for t in tasks])
    mean_delta, mean_lo, mean_hi = paired_delta_ci(rates_low, rates_high)
    wstat, wp = wilcoxon_signed_rank(rates_high, rates_low)

    return OverallResult(
        n_tasks=len(tasks),
        n_episodes=n_episodes,
        successes_low=s_low,
        successes_high=s_high,
        rate_low=s_low / n_episodes,
        rate_high=s_high / n_episodes,
        delta_pooled=pooled_delta,
        delta_pooled_ci=(pooled_lo, pooled_hi),
        mcnemar_pooled=pooled_mcnemar,
        mean_rate_low=float(rates_low.mean()),
        mean_rate_high=float(rates_high.mean()),
        mean_delta=mean_delta,
        mean_delta_ci=(mean_lo, mean_hi),
        wilcoxon_stat=wstat,
        wilcoxon_p=wp,
    )
