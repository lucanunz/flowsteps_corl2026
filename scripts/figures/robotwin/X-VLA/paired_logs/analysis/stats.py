"""Stats primitives for the paired-step analysis.

We have true paired binary outcomes (seed-matched), so McNemar is the
primary test. Aggregates across many tasks use a cluster bootstrap (by
seed within task) for confidence intervals and Benjamini-Hochberg /
Holm corrections for per-task multiple-testing control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class McNemarResult:
    n: int            # number of paired episodes
    b: int            # low success, high failure
    c: int            # low failure, high success
    both_success: int
    both_failure: int
    p_exact: float    # two-sided exact binomial p (preferred when b+c small)
    p_chi2_cc: float  # chi-square with continuity correction (asymptotic)


def mcnemar(low: np.ndarray, high: np.ndarray) -> McNemarResult:
    low = np.asarray(low, dtype=int)
    high = np.asarray(high, dtype=int)
    assert low.shape == high.shape
    n = int(low.size)
    both_success = int(np.sum((low == 1) & (high == 1)))
    both_failure = int(np.sum((low == 0) & (high == 0)))
    b = int(np.sum((low == 1) & (high == 0)))
    c = int(np.sum((low == 0) & (high == 1)))

    discordant = b + c
    if discordant == 0:
        p_exact = 1.0
        p_chi2 = 1.0
    else:
        p_exact = float(sps.binomtest(min(b, c), n=discordant, p=0.5,
                                       alternative="two-sided").pvalue)
        chi2 = (abs(b - c) - 1.0) ** 2 / discordant
        p_chi2 = float(sps.chi2.sf(chi2, df=1))
    return McNemarResult(
        n=n, b=b, c=c,
        both_success=both_success, both_failure=both_failure,
        p_exact=p_exact, p_chi2_cc=p_chi2,
    )


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    z = sps.norm.ppf(0.5 + confidence / 2.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_delta_ci(
    low: np.ndarray,
    high: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Wald CI for paired difference (high - low). Inputs may be 0/1 episode
    outcomes or per-task rates in [0, 1]."""
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    diffs = high - low
    n = diffs.size
    delta = float(diffs.mean())
    if n <= 1:
        return delta, delta, delta
    var = float(diffs.var(ddof=1)) / n
    z = sps.norm.ppf(0.5 + confidence / 2.0)
    margin = z * math.sqrt(max(var, 0.0))
    return delta, delta - margin, delta + margin


def cluster_bootstrap_delta_ci(
    per_task_low: list[np.ndarray],
    per_task_high: list[np.ndarray],
    confidence: float = 0.95,
    n_resamples: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for the pooled high-minus-low rate, resampling tasks then
    episodes within task. Conservative when tasks have very different sizes;
    here every task has the same N so it's a clean two-stage scheme.
    """
    rng = np.random.default_rng(seed)
    n_tasks = len(per_task_low)
    assert n_tasks == len(per_task_high)
    sizes = np.array([len(x) for x in per_task_low])

    pooled_low = np.concatenate(per_task_low) if per_task_low else np.array([])
    pooled_high = np.concatenate(per_task_high) if per_task_high else np.array([])
    point = float(pooled_high.mean() - pooled_low.mean())

    deltas = np.empty(n_resamples, dtype=float)
    task_idx_pool = np.arange(n_tasks)
    for k in range(n_resamples):
        task_choice = rng.choice(task_idx_pool, size=n_tasks, replace=True)
        s_low = 0
        s_high = 0
        n_total = 0
        for t in task_choice:
            m = sizes[t]
            ep = rng.integers(0, m, size=m)
            s_low += int(per_task_low[t][ep].sum())
            s_high += int(per_task_high[t][ep].sum())
            n_total += m
        deltas[k] = s_high / n_total - s_low / n_total
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(deltas, alpha))
    hi = float(np.quantile(deltas, 1.0 - alpha))
    return point, lo, hi


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    qvals = [1.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        q = min(running, pvalues[i] * m / (rank + 1))
        running = q
        qvals[i] = q
    return qvals


def holm(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, pvalues[i] * (m - rank))
        running = max(running, a)
        adj[i] = running
    return adj


def bonferroni(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    return [min(1.0, p * m) for p in pvalues]


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon on paired samples (e.g. per-task rates). Returns (W, p)."""
    if len(x) < 2:
        return 0.0, 1.0
    res = sps.wilcoxon(x, y, zero_method="wilcox", correction=False,
                       alternative="two-sided", method="approx")
    return float(res.statistic), float(res.pvalue)
