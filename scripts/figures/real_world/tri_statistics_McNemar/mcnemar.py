"""McNemar helpers over reused STEP cells, plus the per-model family verdict.

``cell_mcnemar`` computes the paired McNemar test from a reused STEP ``CellStats``
(its b, c, n_paired), returning None for cells without a paired table.
``apply_family_multiplicity`` adjusts a model's per-task p-values across its task
family (Bonferroni + Holm), and ``combined_difference_verdict`` summarises the
family into a single "difference on some / no difference detected" decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from .stats import (
    McNemarResult,
    bonferroni,
    holm,
    mcnemar_test,
)


def cell_mcnemar(
    cell,
    *,
    alpha: float = 0.05,
    exact_max_discordant: int = 25,
) -> McNemarResult | None:
    """Paired McNemar for one reused CellStats, or None without a paired table."""
    if (
        getattr(cell, "paired_low", None) is None
        or not cell.n_paired
        or cell.n_paired <= 0
        or cell.both_success is None
    ):
        return None
    return mcnemar_test(
        cell.discordant_b,
        cell.discordant_c,
        n_paired=cell.n_paired,
        alpha=alpha,
        exact_max_discordant=exact_max_discordant,
    )


def apply_family_multiplicity(results: list[McNemarResult | None]) -> None:
    """Fill q_bonferroni / q_holm on the assessed results of one model's family.

    Unassessed (None) cells are skipped and do not count toward the family size,
    matching how the STEP/TOST forks treat missing paired data.
    """
    assessed = [r for r in results if r is not None]
    if not assessed:
        return
    pvals = [r.p_value for r in assessed]
    for r, qb, qh in zip(assessed, bonferroni(pvals), holm(pvals)):
        r.q_bonferroni = qb
        r.q_holm = qh


@dataclass
class CombinedDifferenceVerdict:
    label: str                 # group label (e.g. model)
    family_size: int           # K — number of assessed tasks in the family
    alpha: float               # family-wise level
    n_significant: int         # tasks with q_holm < alpha
    n_not_significant: int
    n_unassessed: int
    combined_decision: str     # DifferenceOnSome | NoDifferenceDetected | NotAssessed


def combined_difference_verdict(
    label: str,
    results: list[McNemarResult | None],
    *,
    alpha: float,
) -> CombinedDifferenceVerdict:
    """Family summary using Holm-adjusted p-values (FWER-controlled at α).

    A single Holm-significant task is enough to reject the family null that
    "1-step and the reference agree on every task", so the family decision is
    ``DifferenceOnSome`` iff any assessed task has q_holm < α.
    """
    assessed = [r for r in results if r is not None]
    K = len(assessed)
    n_unassessed = len(results) - K
    n_sig = sum(1 for r in assessed if r.q_holm is not None and r.q_holm < alpha)

    if K == 0:
        decision = "NotAssessed"
    elif n_sig > 0:
        decision = "DifferenceOnSome"
    else:
        decision = "NoDifferenceDetected"

    return CombinedDifferenceVerdict(
        label=label,
        family_size=K,
        alpha=alpha,
        n_significant=n_sig,
        n_not_significant=K - n_sig,
        n_unassessed=n_unassessed,
        combined_decision=decision,
    )
