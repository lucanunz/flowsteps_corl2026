"""McNemar paired-difference primitives.

The McNemar test conditions on the discordant pairs (b, c) of the paired 2×2
table and tests H₀: the two policies have equal marginal success rates, i.e.
b ~ Binomial(b + c, ½).

Two methods, selected by the number of discordant pairs n_d = b + c:

* **Exact binomial** (n_d < ``EXACT_MAX_DISCORDANT``). Two-sided p-value

      p = 2 · Σ_{i=0}^{min(b,c)} C(n_d, i) · 0.5^{n_d},          clamped to ≤ 1.

* **χ² with Yates' continuity correction** (otherwise)

      χ² = (|b − c| − 1)² / (b + c),     p = 1 − F_{χ²₁}(χ²) = erfc(√(χ²/2)).

  The corrected numerator is clamped at 0 (a |b − c| < 1 table is perfectly
  symmetric and must not yield χ² > 0); this only matters in the degenerate
  b = c case, which in practice falls in the exact-test regime anyway.

The estimand reported alongside the p-value is Δ̂ = (c − b)/n = p_reference −
p_1step, matching the sign convention of the TOST fork (positive ⇒ reference
better). The test itself uses only b and c; Δ̂ is descriptive.

This module is self-contained (stdlib ``math`` only) so it can be unit-tested
without the per-benchmark loader stack; multiplicity correctors (Bonferroni,
Holm) for the per-model task family live here too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import comb

# Method tags.
EXACT = "exact-binomial"
YATES = "yates-chi2"

# Direction tags (which policy wins the discordant pairs).
DIR_REFERENCE = "reference"   # c > b  → multi-step reference better
DIR_ONESTEP = "1-step"        # b > c  → 1-step better
DIR_NONE = "none"             # b == c → no directional signal


@dataclass
class McNemarResult:
    """Outcome of one paired McNemar test on the discordant pairs (b, c)."""
    b: int                    # 1-step succeeds, reference fails
    c: int                    # reference succeeds, 1-step fails
    n_discordant: int         # b + c
    n_paired: int             # a + b + c + d (for the descriptive Δ̂ / rates)
    delta_hat: float          # Δ̂ = (c − b)/n_paired (signed proportion)
    method: str               # EXACT | YATES
    statistic: float | None   # Yates χ² (None for the exact test)
    p_value: float
    alpha: float
    significant: bool         # p_value < alpha (uncorrected)
    direction: str            # DIR_REFERENCE | DIR_ONESTEP | DIR_NONE
    # Filled in by the family multiplicity step (per-model task family).
    q_bonferroni: float | None = None
    q_holm: float | None = None


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact-binomial McNemar p-value on discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_yates_chi2(b: int, c: int) -> tuple[float, float]:
    """Yates-corrected χ² statistic and p-value on discordant counts (b, c).

    Returns (χ², p). The χ² has 1 degree of freedom, so the upper-tail
    probability is ``erfc(√(χ²/2))`` (no SciPy needed).
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    corrected = max(0.0, abs(b - c) - 1.0)
    chi2 = (corrected * corrected) / n
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return chi2, p


def mcnemar_test(
    b: int,
    c: int,
    *,
    n_paired: int,
    alpha: float = 0.05,
    exact_max_discordant: int = 25,
) -> McNemarResult:
    """Run the McNemar test, choosing exact vs Yates by the discordant count.

    ``n_paired`` is used only for the descriptive Δ̂ = (c − b)/n_paired.
    """
    n_disc = b + c
    if n_disc < exact_max_discordant:
        method, statistic, p = EXACT, None, mcnemar_exact_p(b, c)
    else:
        method = YATES
        statistic, p = mcnemar_yates_chi2(b, c)

    if c > b:
        direction = DIR_REFERENCE
    elif b > c:
        direction = DIR_ONESTEP
    else:
        direction = DIR_NONE

    delta_hat = (c - b) / n_paired if n_paired else 0.0
    return McNemarResult(
        b=b,
        c=c,
        n_discordant=n_disc,
        n_paired=n_paired,
        delta_hat=delta_hat,
        method=method,
        statistic=statistic,
        p_value=p,
        alpha=alpha,
        significant=p < alpha,
        direction=direction,
    )


# ── Multiplicity correctors (per-model task family) ──────────────────────────
#
# Unlike the equivalence (TOST) fork — whose "equivalent on all tasks" claim is
# an intersection–union test that needs no correction — a difference test over
# several tasks DOES inflate the family-wise error rate, so we adjust the
# per-task p-values across the K tasks of each model.

def bonferroni(pvals: list[float]) -> list[float]:
    """Bonferroni-adjusted p-values (q_i = min(1, m · p_i))."""
    m = len(pvals)
    return [min(1.0, p * m) for p in pvals]


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (enforced monotone non-decreasing)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        q[idx] = min(1.0, running)
    return q
