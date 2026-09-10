"""Apply STEP / z-test + multiplicity correction to CALVIN PairResults.

Family per (model, dataset) pair = 1 cell (SR@5), so α_split = α_global.
The two STEP columns are therefore identical, but we keep both for parity
with the LIBERO pipeline.

Also fills the Welch CLD on each cell (independent of the binary CLD).
"""

from __future__ import annotations

from .config import CELL_KEYS
from .pairs import CellStats, PairResult
from .stats import (
    benjamini_hochberg_qvalues,
    bonferroni,
    cld_from_step_decision,
    cld_two_methods,
    holm,
    step_test,
)


def _label_low_high(pair: PairResult) -> tuple[str, str]:
    return (
        f"{pair.spec.variant_low}-step",
        f"{pair.spec.variant_high}-step",
    )


def _fill_cell_step_or_z(
    cell: CellStats,
    *,
    alpha_global: float,
    alpha_split: float,
    label_low: str,
    label_high: str,
) -> None:
    if cell.paired_low is not None and cell.paired_high is not None and len(cell.paired_low) > 0:
        cell.primary_test = "step"
        n_max = min(len(cell.paired_low), len(cell.paired_high))
        cell.step_n_max = n_max

        res_un = step_test(cell.paired_low, cell.paired_high, alpha=alpha_global, n_max=n_max)
        cell.step_alpha_uncorrected = alpha_global
        cell.step_decision_uncorrected = res_un.decision_name
        cell.step_stop_time_uncorrected = res_un.stop_time
        cld_un = cld_from_step_decision(label_low, label_high, res_un.decision)
        cell.cld_low_uncorrected = cld_un[label_low]
        cell.cld_high_uncorrected = cld_un[label_high]

        res_co = step_test(cell.paired_low, cell.paired_high, alpha=alpha_split, n_max=n_max)
        cell.step_alpha_corrected = alpha_split
        cell.step_decision_corrected = res_co.decision_name
        cell.step_stop_time_corrected = res_co.stop_time
        cld_co = cld_from_step_decision(label_low, label_high, res_co.decision)
        cell.cld_low_corrected = cld_co[label_low]
        cell.cld_high_corrected = cld_co[label_high]
    else:
        cell.primary_test = "z"


def _fill_z_family(
    cells: list[CellStats],
    *,
    alpha: float,
    label_low: str,
    label_high: str,
) -> None:
    """Compute z-test Bonferroni/Holm/BH for the family and set CLDs.

    With family size 1 the q-values equal the raw p_z; we still compute them
    via the same helpers so the column layout matches the LIBERO report.
    """
    p_z = [c.p_z for c in cells]
    q_bonf_z = bonferroni(p_z)
    q_holm_z = holm(p_z)
    q_bh_z = benjamini_hochberg_qvalues(p_z)

    for cell, qbz, qhz, qbhz in zip(cells, q_bonf_z, q_holm_z, q_bh_z):
        cell.q_bonf_z = qbz
        cell.q_holm_z = qhz
        cell.q_bh_z = qbhz
        if cell.primary_test != "step":
            cld_un = cld_two_methods(label_low, label_high, pvalue=cell.p_z, alpha=alpha)
            cell.cld_low_uncorrected = cld_un[label_low]
            cell.cld_high_uncorrected = cld_un[label_high]
            cld_co = cld_two_methods(label_low, label_high, pvalue=qhz, alpha=alpha)
            cell.cld_low_corrected = cld_co[label_low]
            cell.cld_high_corrected = cld_co[label_high]


def _fill_welch_cld(
    cell: CellStats,
    *,
    alpha: float,
    label_low: str,
    label_high: str,
) -> None:
    if cell.welch is None:
        return
    letters = cld_two_methods(
        label_low, label_high, pvalue=cell.welch.p_value, alpha=alpha
    )
    cell.welch_cld_low = letters[label_low]
    cell.welch_cld_high = letters[label_high]


def apply_corrections(pair: PairResult, alpha: float = 0.05) -> None:
    """Run STEP at α_global / α_split, fill q_*_z, and set CLDs (binary + Welch).

    Family = all cells in the pair (here always 1: SR@5). α_split = α / 1 = α.
    """
    label_low, label_high = _label_low_high(pair)
    ordered_keys = [k for k in CELL_KEYS if k in pair.cells]
    cells = [pair.cells[k] for k in ordered_keys]
    family_size = max(len(cells), 1)
    alpha_split = alpha / family_size

    for cell in cells:
        _fill_cell_step_or_z(
            cell,
            alpha_global=alpha,
            alpha_split=alpha_split,
            label_low=label_low,
            label_high=label_high,
        )
    _fill_z_family(cells, alpha=alpha, label_low=label_low, label_high=label_high)

    for cell in cells:
        _fill_welch_cld(cell, alpha=alpha, label_low=label_low, label_high=label_high)
