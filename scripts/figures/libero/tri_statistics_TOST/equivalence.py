"""Attach paired equivalence (TOST) results to reused PairResult / per-task records.

Data loading and cell construction are reused verbatim from
`tri_statistics_STEP.pairs` (those builders fill the paired 2×2 table — a, b, c,
d, n_paired — without running STEP). Here we only compute the TOST per cell and,
for the per-task analysis, the per-suite combined equivalence verdict.

Multiplicity: the combined "equivalent on *all* tasks" claim is an
intersection–union test. Its null is the *union* of the per-task "differs by ≥ δ"
nulls, so each per-task TOST runs at the **full** level α with **no Bonferroni**
and the conjunction is still a level-α test — the opposite of STEP's α-splitting.
"""

from __future__ import annotations

from dataclasses import dataclass

from .stats import (
    EQUIVALENT,
    INCONCLUSIVE,
    RELEVANT_DIFFERENCE,
    TostResult,
    paired_tost,
)


def _cell_tost(cell, *, delta: float, alpha: float) -> TostResult | None:
    """Paired TOST for one cell, or None when it has no paired 2×2 table."""
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


def apply_equivalence(pair, *, delta: float, alpha: float = 0.05) -> None:
    """Attach `cell.tost` (a TostResult or None) to every cell of a PairResult.

    `delta` is in proportion units (e.g. 0.03 for a 3pp margin).
    """
    for cell in pair.cells.values():
        cell.tost = _cell_tost(cell, delta=delta, alpha=alpha)
    # Stash the settings on the pair for the reporter/plotter.
    pair.delta_margin = delta
    pair.tost_alpha = alpha


# ── Per-task combined verdict (intersection–union; no correction) ────────────

@dataclass
class CombinedEquivalenceVerdict:
    """Intersection–union summary over the per-task TOSTs in one suite.

    `EquivalentOnAll` requires *every* task in the suite to be individually
    equivalent within δ (each at the full level α — no Bonferroni). Any task that
    is a relevant difference, inconclusive, or unassessed blocks that claim.
    """
    suite: str
    family_size: int          # K — number of tasks in the suite
    alpha: float              # per-task TOST level (no α-split)
    delta_margin: float       # δ (proportion units)
    n_equivalent: int
    n_relevant: int
    n_inconclusive: int
    n_unassessed: int         # tasks with no paired data
    combined_decision: str    # EquivalentOnAll | DifferenceOnSome | NotEstablished


def _combined_equiv_verdict(
    suite: str,
    cells: list,
    *,
    alpha: float,
    delta: float,
) -> CombinedEquivalenceVerdict:
    n_eq = n_rel = n_inc = n_un = 0
    for cell in cells:
        tost = getattr(cell, "tost", None)
        if tost is None:
            n_un += 1
        elif tost.verdict == EQUIVALENT:
            n_eq += 1
        elif tost.verdict == RELEVANT_DIFFERENCE:
            n_rel += 1
        else:
            n_inc += 1

    K = len(cells)
    if K > 0 and n_eq == K:
        combined = "EquivalentOnAll"
    elif n_rel > 0:
        combined = "DifferenceOnSome"
    else:
        combined = "NotEstablished"

    return CombinedEquivalenceVerdict(
        suite=suite,
        family_size=K,
        alpha=alpha,
        delta_margin=delta,
        n_equivalent=n_eq,
        n_relevant=n_rel,
        n_inconclusive=n_inc,
        n_unassessed=n_un,
        combined_decision=combined,
    )


def apply_equivalence_per_task(result, *, delta: float, alpha: float = 0.05) -> None:
    """Attach per-task `cell.tost` and per-suite combined equivalence verdicts."""
    result.delta_margin = delta
    result.tost_alpha = alpha
    result.combined_equiv = {}
    for suite, cells in result.suite_tasks.items():
        for cell in cells:
            cell.tost = _cell_tost(cell, delta=delta, alpha=alpha)
        if cells:
            result.combined_equiv[suite] = _combined_equiv_verdict(
                suite, cells, alpha=alpha, delta=delta
            )
