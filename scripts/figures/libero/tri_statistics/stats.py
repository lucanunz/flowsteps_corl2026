"""TRI-style statistics helpers used by the tri_statistics package."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import stats as scipy_stats


def beta_posterior_samples(
    successes: int,
    trials: int,
    n_samples: int = 2000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw samples from Beta(s+1, n-s+1) posterior with uniform Beta(1,1) prior."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"invalid (s={successes}, n={trials})")
    rng = rng if rng is not None else np.random.default_rng(0)
    a = successes + 1
    b = trials - successes + 1
    return scipy_stats.beta(a, b).rvs(size=n_samples, random_state=rng)


def beta_posterior_mean(successes: int, trials: int) -> float:
    return (successes + 1) / (trials + 2)


def fisher_exact_2x2(s_low: int, n_low: int, s_high: int, n_high: int) -> float:
    """Two-sided Fisher's exact p-value for the 2x2 table of successes/failures."""
    if n_low <= 0 or n_high <= 0:
        return 1.0
    table = [[s_low, n_low - s_low], [s_high, n_high - s_high]]
    _, p = scipy_stats.fisher_exact(table, alternative="two-sided")
    return float(p)


def mcnemar_midp(b: int, c: int) -> float:
    """Exact two-sided mid-p for paired binary outcomes; b, c are discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    binom = scipy_stats.binom(n, 0.5)
    cdf_lower = float(binom.cdf(k))
    sf_upper = float(binom.sf(k - 1))
    pmf_k = float(binom.pmf(k))
    one_sided = min(cdf_lower, sf_upper)
    p = 2.0 * one_sided - pmf_k
    return min(1.0, max(0.0, p))


def benjamini_hochberg_qvalues(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted q-values, preserving input order."""
    if not pvalues:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda item: item[1])
    qvalues = [1.0] * len(pvalues)
    running_min = 1.0
    total = len(pvalues)
    for rank_from_end, (original_index, pvalue) in enumerate(reversed(indexed), start=1):
        rank = total - rank_from_end + 1
        adjusted = min(1.0, pvalue * total / rank)
        running_min = min(running_min, adjusted)
        qvalues[original_index] = running_min
    return qvalues


def bonferroni(pvalues: Iterable[float]) -> list[float]:
    pv = list(pvalues)
    m = len(pv)
    if m == 0:
        return []
    return [min(1.0, p * m) for p in pv]


def holm(pvalues: Iterable[float]) -> list[float]:
    """Holm step-down adjusted p-values, preserving input order."""
    pv = list(pvalues)
    m = len(pv)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pv[i])
    adjusted_sorted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = pv[idx] * (m - rank)
        adj = min(1.0, adj)
        if adj < running_max:
            adj = running_max
        else:
            running_max = adj
        adjusted_sorted[rank] = adj
    out = [0.0] * m
    for rank, idx in enumerate(order):
        out[idx] = adjusted_sorted[rank]
    return out


def cld_from_pmatrix(
    labels: list[str],
    pmat: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, str]:
    """Compact Letter Display from a square pairwise p-value matrix.

    Methods sharing a letter are not significantly different at level alpha.
    Implements a simple greedy sweep: each maximal clique of non-significant
    methods becomes one letter group. For 2 methods this reduces to "a"/"a"
    when not significant, "a"/"b" when significant.
    """
    k = len(labels)
    if k == 0:
        return {}
    if pmat.shape != (k, k):
        raise ValueError(f"pmat shape {pmat.shape} != ({k}, {k})")

    not_sig = np.ones((k, k), dtype=bool)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            p = pmat[i, j]
            if math.isnan(p):
                not_sig[i, j] = True
            else:
                not_sig[i, j] = p > alpha

    cliques: list[set[int]] = []
    indices = list(range(k))
    for i in indices:
        # Build a maximal clique containing i (greedy).
        clique = {i}
        for j in indices:
            if j == i:
                continue
            if all(not_sig[j, m] for m in clique):
                clique.add(j)
        cliques.append(clique)

    # Deduplicate cliques (subset removal).
    unique: list[set[int]] = []
    for c in cliques:
        if any(c <= other and c != other for other in cliques):
            continue
        if c not in unique:
            unique.append(c)

    # Order cliques by smallest index for stable letter assignment.
    unique.sort(key=lambda s: min(s))
    letters = "abcdefghijklmnopqrstuvwxyz"
    assignment: dict[int, list[str]] = {i: [] for i in range(k)}
    for ci, clique in enumerate(unique):
        if ci >= len(letters):
            raise ValueError("too many CLD groups for the alphabet")
        for idx in clique:
            assignment[idx].append(letters[ci])

    return {labels[i]: "".join(assignment[i]) or "a" for i in range(k)}


def cld_two_methods(label_low: str, label_high: str, pvalue: float, alpha: float = 0.05) -> dict[str, str]:
    """Convenience for the 2-method case used throughout the report."""
    if math.isnan(pvalue) or pvalue > alpha:
        return {label_low: "a", label_high: "a"}
    return {label_low: "a", label_high: "b"}
