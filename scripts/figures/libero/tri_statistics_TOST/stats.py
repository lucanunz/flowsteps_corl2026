"""Paired equivalence (TOST) primitives for the tri_statistics_TOST variant.

We test equivalence of the multi-step *reference* policy and the *1-step* policy
on their paired per-rollout success outcomes, against a pre-registered margin δ.

Estimand: Δ = p_reference − p_1step  (positive ⇒ reference better). On the paired
2×2 table

        a = both succeed     b = 1-step only (low wins)
        c = reference only   d = both fail        n = a + b + c + d

the marginals are p_high = (a + c)/n, p_low = (a + b)/n, and Δ̂ = (c − b)/n.

Confidence interval: Newcombe's score-interval method for the difference between
two *paired* proportions — Newcombe (1998), *Statistics in Medicine*
17:2635–2650, **Method 10**. It combines two Wilson score intervals for the
marginal proportions with a continuity-corrected estimate of the within-pair
association (the φ-coefficient of the 2×2 table). We reuse the project's existing
`wilson_score_interval`. (This is *not* the MOVER interval, which is the
square-and-add construction for two *independent* groups; Method 10 adds the φ
correlation term that MOVER lacks.)

TOST at level α ⇔ the two-sided (1 − 2α) CI ⊂ (−δ, +δ). With the default
α = 0.05 this is the 90% CI. The verdict is trichotomous:

    Equivalent          CI ⊂ (−δ, +δ)        equivalence demonstrated within δ
    RelevantDifference  CI ∩ (−δ, +δ) = ∅     a ≥ δ difference demonstrated
    Inconclusive        otherwise            underpowered — neither shown

Paired data only: there is no unpaired / two-proportion fallback. Cells without
a paired table are simply not assessed (handled by the caller in equivalence.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from wilson_score_interval import wilson_score_interval  # type: ignore


EQUIVALENT = "Equivalent"
RELEVANT_DIFFERENCE = "RelevantDifference"
INCONCLUSIVE = "Inconclusive"

# Human-readable labels for report tables.
VERDICT_LABEL: dict[str, str] = {
    EQUIVALENT: "equivalent",
    RELEVANT_DIFFERENCE: "difference > δ",
    INCONCLUSIVE: "inconclusive",
}

# Compact codes / colors for plot annotations and heatmaps.
VERDICT_CODE: dict[str, str] = {
    EQUIVALENT: "≡",
    RELEVANT_DIFFERENCE: "Δ",
    INCONCLUSIVE: "?",
}
VERDICT_COLOR: dict[str, str] = {
    EQUIVALENT: "#009E73",          # green — equivalence shown
    RELEVANT_DIFFERENCE: "#D55E00",  # vermillion — real gap
    INCONCLUSIVE: "#6B7280",        # grey — undecided
}


@dataclass
class TostResult:
    """Outcome of one paired TOST against margin δ (proportion units)."""
    delta_hat: float        # Δ̂ = p_high − p_low (signed proportion)
    ci_low: float           # lower limit of the (1 − 2α) CI for Δ (proportion)
    ci_high: float          # upper limit of the (1 − 2α) CI for Δ (proportion)
    alpha: float            # one-sided TOST level (CI coverage is 1 − 2α)
    delta_margin: float     # δ (proportion units, e.g. 0.03 for 3pp)
    verdict: str            # EQUIVALENT | RELEVANT_DIFFERENCE | INCONCLUSIVE
    n_paired: int           # n = a + b + c + d

    @property
    def ci_confidence(self) -> float:
        return 1.0 - 2.0 * self.alpha


def paired_newcombe_ci(
    a: int,
    b: int,
    c: int,
    d: int,
    *,
    confidence: float = 0.90,
) -> tuple[float, float, float] | None:
    """Newcombe (1998) Method 10 CI for Δ = p_high − p_low on a paired 2×2 table.

    Score-interval method for the difference between two paired proportions: the
    two Wilson marginal intervals combined with a continuity-corrected φ (the
    2×2 φ-coefficient). Not the MOVER interval — Method 10 adds the φ term.

    `a` both succeed, `b` 1-step only, `c` reference only, `d` both fail.
    `confidence` is the two-sided coverage of the returned interval (pass
    1 − 2α to obtain the interval a TOST at level α inverts).

    Returns (Δ̂, L, U) in proportion units, or None when n == 0.
    """
    n = a + b + c + d
    if n <= 0:
        return None

    m1 = a + c          # reference successes
    m2 = a + b          # 1-step successes
    p1 = m1 / n
    p2 = m2 / n
    delta_hat = p1 - p2

    # Wilson score intervals for the two marginal proportions at the target
    # coverage. Method 10 combines these two per-arm score intervals.
    l1, u1 = wilson_score_interval(m1, n, confidence=confidence)
    l2, u2 = wilson_score_interval(m2, n, confidence=confidence)

    # Within-pair association correction φ (Newcombe 1998, Method 10).
    e = a * d - b * c
    denom_sq = m1 * m2 * (n - m1) * (n - m2)
    if denom_sq <= 0:
        phi = 0.0
    else:
        denom = math.sqrt(denom_sq)
        if e > n / 2.0:
            phi = (e - n / 2.0) / denom   # positive association, continuity-shrunk
        elif e >= 0.0:
            phi = 0.0                      # weak positive association → no correction
        else:
            phi = e / denom               # negative association

    def _root(x: float, y: float) -> float:
        # √(x² − 2φ·x·y + y²), clamped at 0 to guard tiny negative radicands.
        val = x * x - 2.0 * phi * x * y + y * y
        return math.sqrt(val) if val > 0.0 else 0.0

    lower = delta_hat - _root(p1 - l1, u2 - p2)
    upper = delta_hat + _root(u1 - p1, p2 - l2)
    return delta_hat, lower, upper


# Backwards-compatible alias (the function was previously, and incorrectly,
# named after MOVER). Prefer `paired_newcombe_ci`.
paired_mover_ci = paired_newcombe_ci


def tost_equivalence(
    ci_low: float,
    ci_high: float,
    *,
    delta: float,
) -> str:
    """Trichotomous verdict from a (1 − 2α) CI for Δ against margin δ.

    `delta` is in proportion units (e.g. 0.03). Boundary-touching CIs fall
    through to Inconclusive (the conservative choice).
    """
    if ci_low > -delta and ci_high < delta:
        return EQUIVALENT
    if ci_low >= delta or ci_high <= -delta:
        return RELEVANT_DIFFERENCE
    return INCONCLUSIVE


def paired_tost(
    a: int,
    b: int,
    c: int,
    d: int,
    *,
    delta: float,
    alpha: float = 0.05,
) -> TostResult | None:
    """Run the paired TOST on a 2×2 table. `delta` in proportion units.

    Returns None when the table is empty (n == 0) — the caller reports such
    cells as "not assessed".
    """
    ci = paired_newcombe_ci(a, b, c, d, confidence=1.0 - 2.0 * alpha)
    if ci is None:
        return None
    delta_hat, lo, hi = ci
    return TostResult(
        delta_hat=delta_hat,
        ci_low=lo,
        ci_high=hi,
        alpha=alpha,
        delta_margin=delta,
        verdict=tost_equivalence(lo, hi, delta=delta),
        n_paired=a + b + c + d,
    )
