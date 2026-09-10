"""Apply multiplicity correction to STEP-based PairResults.

STEP returns a decision (AcceptNull / AcceptAlternative / FailToDecide) at a
pre-set α; it does not produce a p-value. The natural family-wise correction
for STEP is therefore **α-splitting**: divide the global α by the number of
hypotheses in the family, then run STEP at the split α. We report decisions at
both α_global and α_split, so the reader can see which separations survive
the correction and which only held at the looser threshold.

For cells where no paired episode data exists, the primary test falls back to the two-proportion z-test.
Those cells get the usual Bonferroni / Holm / BH q-columns and Holm-q drives their CLD.
"""

from __future__ import annotations

from typing import Iterable

from sequentialized_barnard_tests import Decision

from .config import ALL_CELLS, SUITE_ORDER
from .pairs import CellStats, CombinedSuiteVerdict, PairResult, PerTaskPairResult
from .stats import (
    benjamini_hochberg_qvalues,
    bonferroni,
    cld_from_step_decision,
    cld_two_methods,
    holm,
    step_test,
)


def _label_low_high(pair_or_result) -> tuple[str, str]:
    return (
        f"{pair_or_result.spec.variant_low}-step",
        f"{pair_or_result.spec.variant_high}-step",
    )


def _decision_from_name(name: str | None) -> Decision | None:
    if name is None:
        return None
    return Decision[name]


def _fill_cell_step_or_z(
    cell: CellStats,
    *,
    alpha_global: float,
    alpha_split: float,
    label_low: str,
    label_high: str,
) -> None:
    """Run STEP on a single cell (twice — at α_global and α_split) if paired
    data exists. Otherwise mark `primary_test = "z"` and leave CLD blank;
    the caller will set the z-based CLD after computing q-values family-wide.
    """
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
        # CLDs for z-fallback cells are set after the family-wide q-values are computed.


def _fill_z_family(
    cells: list[CellStats],
    *,
    alpha: float,
    label_low: str,
    label_high: str,
    apply_correction: bool = True,
) -> None:
    """Compute z-test Bonferroni/Holm/BH for the family and set CLDs.

    - The "uncorrected" CLD comes from the raw z p-value vs α.
    - The "corrected" CLD comes from Holm-q vs α (or from the raw p-value vs α
      when `apply_correction=False`, in which case it matches the uncorrected CLD).

    Both STEP-primary and z-primary cells are part of the *same* z-test family
    (so the q-columns in the unpaired table stay sensible), but only z-primary
    cells have their CLDs overwritten by this function. STEP-primary cells'
    CLDs were already set by `_fill_cell_step_or_z` above.
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
            pvalue_for_corrected = qhz if apply_correction else cell.p_z
            cld_co = cld_two_methods(label_low, label_high, pvalue=pvalue_for_corrected, alpha=alpha)
            cell.cld_low_corrected = cld_co[label_low]
            cell.cld_high_corrected = cld_co[label_high]


def apply_corrections(
    pair: PairResult,
    alpha: float = 0.05,
    correction_family: str = "pair",
) -> None:
    """Fill q_*_z, run STEP at α_global and α_split, set CLD letters.

    `correction_family`:
      * "pair" — family = all cells in this (model, dataset) pair (4 suites +
        overall = 5 hypotheses); α_split = α / family_size (default).
      * "none" — no multiplicity correction; α_split = α_global, so the
        "corrected" decisions and CLDs equal the uncorrected ones.
    """
    pair.correction_family = correction_family
    label_low, label_high = _label_low_high(pair)
    ordered_keys = [k for k in ALL_CELLS if k in pair.cells]
    cells = [pair.cells[k] for k in ordered_keys]

    if correction_family == "pair":
        family_size = max(len(cells), 1)
        alpha_split = alpha / family_size
        apply_z_correction = True
    elif correction_family == "none":
        alpha_split = alpha
        apply_z_correction = False
    else:
        raise ValueError(f"Unknown correction_family: {correction_family!r}")

    for cell in cells:
        _fill_cell_step_or_z(
            cell,
            alpha_global=alpha,
            alpha_split=alpha_split,
            label_low=label_low,
            label_high=label_high,
        )
    _fill_z_family(
        cells,
        alpha=alpha,
        label_low=label_low,
        label_high=label_high,
        apply_correction=apply_z_correction,
    )


def _combined_verdict_for_suite(
    suite: str,
    suite_cells: list[CellStats],
    *,
    alpha_global: float,
    alpha_split: float,
    correction_applied: bool,
) -> CombinedSuiteVerdict:
    """Aggregate per-task STEP decisions at α_split into one multitask verdict.

    Reads `cell.step_decision_corrected` (already set by `_fill_cell_step_or_z`).
    Implements the intersection-union test: the multitask H1 in one direction
    is confirmed iff every per-task subtest rejected H0 in that direction.
    """
    n_alt = n_null = n_undecided = n_z = 0
    for cell in suite_cells:
        if cell.primary_test != "step":
            n_z += 1
            continue
        dec = cell.step_decision_corrected
        if dec == "AcceptAlternative":
            n_alt += 1
        elif dec == "AcceptNull":
            n_null += 1
        else:
            n_undecided += 1

    K = len(suite_cells)
    if n_z > 0:
        combined = "NotConfirmed"
    elif n_alt == K:
        combined = "AlternativeOnAll"
    elif n_null == K:
        combined = "NullOnAll"
    elif n_alt > 0 and n_null > 0:
        combined = "Mixed"
    else:
        combined = "NotConfirmed"

    return CombinedSuiteVerdict(
        suite=suite,
        family_size=K,
        alpha_global=alpha_global,
        alpha_split=alpha_split,
        n_alt_alpha_s=n_alt,
        n_null_alpha_s=n_null,
        n_undecided_alpha_s=n_undecided,
        n_z_fallback=n_z,
        combined_decision=combined,
        correction_applied=correction_applied,
    )


def apply_corrections_per_task(
    result: PerTaskPairResult,
    correction_family: str = "within_suite",
    alpha: float = 0.05,
) -> None:
    """Fill q_*_z, run STEP at α_global and α_split, set CLDs on every per-task cell.

    `correction_family`:
      * "within_suite" — family = ≤10 tasks within the same suite (default).
      * "dataset"      — family = all tasks across all suites (more conservative).
      * "none"         — no multiplicity correction; α_split = α_global,
        so "corrected" decisions and CLDs equal the uncorrected ones.

    Also populates `result.combined_verdicts[suite]` with the per-suite
    intersection-union multitask verdict (see `CombinedSuiteVerdict`). The
    family used for the combined verdict is always the LIBERO suite (K = number
    of tasks in that suite), regardless of `correction_family` — the verdict
    only changes its α_split (and `correction_applied` flag) accordingly.
    """
    result.correction_family = correction_family
    result.combined_verdicts = {}
    label_low, label_high = _label_low_high(result)
    apply_z_correction = correction_family != "none"

    def _process(family_cells: list[CellStats]) -> None:
        if correction_family == "none":
            alpha_split = alpha
        else:
            family_size = max(len(family_cells), 1)
            alpha_split = alpha / family_size
        for cell in family_cells:
            _fill_cell_step_or_z(
                cell,
                alpha_global=alpha,
                alpha_split=alpha_split,
                label_low=label_low,
                label_high=label_high,
            )
        _fill_z_family(
            family_cells,
            alpha=alpha,
            label_low=label_low,
            label_high=label_high,
            apply_correction=apply_z_correction,
        )

    if correction_family == "within_suite":
        for suite_cells in result.suite_tasks.values():
            _process(suite_cells)
    elif correction_family in ("dataset", "none"):
        all_cells = [
            cell
            for suite in SUITE_ORDER
            for cell in result.suite_tasks.get(suite, [])
        ]
        _process(all_cells)
    else:
        raise ValueError(f"Unknown correction_family: {correction_family!r}")

    # Combined multitask verdict per LIBERO suite, regardless of correction_family.
    # The α_split used per task depends on what each cell actually saw — read it
    # off the cell so the verdict text matches the per-task STEP α_s column.
    for suite, suite_cells in result.suite_tasks.items():
        if not suite_cells:
            continue
        per_cell_alpha = next(
            (c.step_alpha_corrected for c in suite_cells if c.step_alpha_corrected is not None),
            alpha,
        )
        result.combined_verdicts[suite] = _combined_verdict_for_suite(
            suite,
            suite_cells,
            alpha_global=alpha,
            alpha_split=per_cell_alpha,
            correction_applied=apply_z_correction,
        )
