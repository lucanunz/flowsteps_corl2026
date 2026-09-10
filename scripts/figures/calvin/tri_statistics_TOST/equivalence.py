"""Equivalence (TOST) helpers shared across the benchmark forks.

`cell_tost` computes the paired TOST from a reused STEP `CellStats` (a, b, c, d),
returning None for cells without a paired table. `combined_equiv_verdict` builds
the intersection–union "equivalent on all" summary over a group of cells — with
**no Bonferroni**: the conjunction null is the union of per-cell "differs by ≥ δ"
nulls, so testing each at the full level α is already a level-α conjunction test.
"""

from __future__ import annotations

from dataclasses import dataclass

from .stats import (
    EQUIVALENT,
    RELEVANT_DIFFERENCE,
    TostResult,
    paired_t_tost,
    paired_tost,
)


def cell_tost(cell, *, delta: float, alpha: float = 0.05) -> TostResult | None:
    """Paired TOST for one reused CellStats, or None when it has no paired table."""
    if (
        getattr(cell, "paired_low", None) is None
        or not cell.n_paired
        or cell.n_paired <= 0
        or cell.both_success is None
    ):
        return None
    return paired_tost(
        cell.both_success,
        cell.discordant_b,
        cell.discordant_c,
        cell.both_failure,
        delta=delta,
        alpha=alpha,
    )


def chain_length_tost(
    low_scores,
    high_scores,
    *,
    delta: float,
    alpha: float = 0.05,
) -> TostResult | None:
    """Paired-t TOST on the mean chain length (0–5), or None without paired scores.

    `low_scores`/`high_scores` are aligned per-rollout completion scores for the
    1-step and reference policies (e.g. from `paired_completion_scores`). δ is in
    subtask units.
    """
    if not low_scores or not high_scores:
        return None
    return paired_t_tost(low_scores, high_scores, delta=delta, alpha=alpha)


@dataclass
class CombinedEquivalenceVerdict:
    label: str                # group label (e.g. model or setting)
    family_size: int          # K — number of cells in the group
    alpha: float              # per-cell TOST level (no α-split)
    delta_margin: float       # δ (proportion units)
    n_equivalent: int
    n_relevant: int
    n_inconclusive: int
    n_unassessed: int
    combined_decision: str    # EquivalentOnAll | DifferenceOnSome | NotEstablished


def combined_equiv_verdict(
    label: str,
    tosts: list,
    *,
    alpha: float,
    delta: float,
) -> CombinedEquivalenceVerdict:
    """Intersection–union summary over per-cell TostResults (None ⇒ unassessed)."""
    n_eq = n_rel = n_inc = n_un = 0
    for tost in tosts:
        if tost is None:
            n_un += 1
        elif tost.verdict == EQUIVALENT:
            n_eq += 1
        elif tost.verdict == RELEVANT_DIFFERENCE:
            n_rel += 1
        else:
            n_inc += 1

    K = len(tosts)
    if K > 0 and n_eq == K:
        combined = "EquivalentOnAll"
    elif n_rel > 0:
        combined = "DifferenceOnSome"
    else:
        combined = "NotEstablished"

    return CombinedEquivalenceVerdict(
        label=label,
        family_size=K,
        alpha=alpha,
        delta_margin=delta,
        n_equivalent=n_eq,
        n_relevant=n_rel,
        n_inconclusive=n_inc,
        n_unassessed=n_un,
        combined_decision=combined,
    )
