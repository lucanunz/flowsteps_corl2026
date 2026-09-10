"""Build STEP-style PairResult records for the RoboTwin paired runs.

Mirrors the dataclass shapes used in `libero/tri_statistics_STEP/pairs.py`
field-for-field so report.py / plotting.py can be adapted mechanically.

A `PairResult` carries a single `Overall` cell (per-pair mode). A
`PerTaskPairResult` carries one cell per task (50 cells for RoboTwin Clean),
plus a `CombinedVerdict` summarising the intersection-union outcome across
all tasks once STEP has run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import OVERALL_KEY, PairSpec
from .io_runs import concat_overall, load_paired_episodes
from .stats import (
    Counts,
    fisher_exact_2x2,
    two_proportion_z_pvalue,
    wilson_score_interval,
)


@dataclass
class CellStats:
    """One row of the analysis: holds Counts, paired sequence, and the
    fields that `corrections.apply_corrections` later fills in."""
    suite: str                                    # "Overall" or task name
    counts_low: Counts
    counts_high: Counts
    wilson_low: tuple[float, float]
    wilson_high: tuple[float, float]
    p_z: float
    p_fisher: float
    # Paired-data diagnostics. All filled for RoboTwin (paired by construction).
    n_paired: int | None = None
    discordant_b: int | None = None               # low success, high failure
    discordant_c: int | None = None               # low failure, high success
    both_success: int | None = None
    both_failure: int | None = None
    paired_low: list[bool] | None = None
    paired_high: list[bool] | None = None
    # z-test multiplicity (filled by apply_corrections).
    q_bonf_z: float | None = None
    q_holm_z: float | None = None
    q_bh_z: float | None = None
    # STEP outcomes (filled by apply_corrections).
    step_n_max: int | None = None
    step_alpha_uncorrected: float | None = None
    step_alpha_corrected: float | None = None
    step_decision_uncorrected: str | None = None  # Decision enum .name
    step_decision_corrected: str | None = None
    step_stop_time_uncorrected: int | None = None
    step_stop_time_corrected: int | None = None
    # Always "step" for RoboTwin (paired by construction); kept for parity with libero.
    primary_test: str = "step"
    cld_low_uncorrected: str = "a"
    cld_high_uncorrected: str = "a"
    cld_low_corrected: str = "a"
    cld_high_corrected: str = "a"


@dataclass
class PairResult:
    spec: PairSpec
    cells: dict[str, CellStats]
    notes: list[str] = field(default_factory=list)
    correction_family: str = "pair"

    @property
    def uses_paired_cld(self) -> bool:
        return all(c.primary_test == "step" for c in self.cells.values())

    @property
    def cld_basis(self) -> str:
        kinds = {c.primary_test for c in self.cells.values()}
        if kinds == {"step"}:
            return "paired STEP"
        if kinds == {"z"}:
            return "unpaired z"
        return "mixed paired/unpaired"


@dataclass
class CombinedVerdict:
    """Intersection-union summary over per-task STEP decisions.

    Multitask H1 in one direction is confirmed at α iff every per-task STEP
    decided that direction at α_split. Any FailToDecide leaves the combined
    claim unconfirmed; both directions appearing leaves it Mixed.
    """
    family_size: int
    alpha_global: float
    alpha_split: float
    n_alt_alpha_s: int
    n_null_alpha_s: int
    n_undecided_alpha_s: int
    n_z_fallback: int
    combined_decision: str    # AlternativeOnAll | NullOnAll | Mixed | NotConfirmed
    correction_applied: bool


@dataclass
class PerTaskPairResult:
    spec: PairSpec
    tasks: list[CellStats]
    notes: list[str] = field(default_factory=list)
    correction_family: str = "within_suite"
    combined_verdict: CombinedVerdict | None = None

    @property
    def cld_basis(self) -> str:
        kinds = {c.primary_test for c in self.tasks}
        if kinds == {"step"}:
            return "paired STEP"
        if kinds == {"z"}:
            return "unpaired z"
        return "mixed paired/unpaired"


def _wilson(c: Counts, confidence: float = 0.95) -> tuple[float, float]:
    return wilson_score_interval(c.successes, c.trials, confidence=confidence)


def _make_cell_from_paired(
    suite: str,
    seq_low: list[bool],
    seq_high: list[bool],
) -> CellStats:
    assert len(seq_low) == len(seq_high)
    n = len(seq_low)
    s_lo = sum(1 for x in seq_low if x)
    s_hi = sum(1 for x in seq_high if x)
    both_s = sum(1 for a, b in zip(seq_low, seq_high) if a and b)
    both_f = sum(1 for a, b in zip(seq_low, seq_high) if (not a) and (not b))
    b = sum(1 for a, c in zip(seq_low, seq_high) if a and (not c))
    c = sum(1 for a, d in zip(seq_low, seq_high) if (not a) and d)
    counts_low = Counts(successes=s_lo, trials=n)
    counts_high = Counts(successes=s_hi, trials=n)
    return CellStats(
        suite=suite,
        counts_low=counts_low,
        counts_high=counts_high,
        wilson_low=_wilson(counts_low),
        wilson_high=_wilson(counts_high),
        p_z=two_proportion_z_pvalue(counts_low, counts_high),
        p_fisher=fisher_exact_2x2(s_lo, n, s_hi, n),
        n_paired=n,
        discordant_b=b,
        discordant_c=c,
        both_success=both_s,
        both_failure=both_f,
        paired_low=list(seq_low),
        paired_high=list(seq_high),
    )


def _skipped_note(skipped: list[str]) -> str | None:
    if not skipped:
        return None
    return (
        f"skipped {len(skipped)} task(s) with no `results.json` in one of "
        f"the runs (cannot be paired): {', '.join(skipped)}"
    )


def build_pair_result(spec: PairSpec) -> PairResult:
    """One Overall cell aggregating all paired episodes across all tasks."""
    per_task, skipped = load_paired_episodes(spec)
    seq_lo, seq_hi = concat_overall(per_task)
    overall = _make_cell_from_paired(OVERALL_KEY, seq_lo, seq_hi)
    notes = [n for n in (_skipped_note(skipped),) if n]
    return PairResult(spec=spec, cells={OVERALL_KEY: overall}, notes=notes)


def build_per_task_pair_result(spec: PairSpec) -> PerTaskPairResult:
    """One cell per task; tasks listed in alphabetical name order with stable
    task indices for plot labels."""
    per_task, skipped = load_paired_episodes(spec)
    cells: list[CellStats] = []
    for task_idx, task in enumerate(sorted(per_task)):
        seq_lo, seq_hi = per_task[task]
        cell = _make_cell_from_paired(task, seq_lo, seq_hi)
        # Stash plot-only metadata exactly the way libero does, so the
        # plotting code (mostly copy-paste) keeps working.
        cell.__dict__["task_idx"] = task_idx
        cell.__dict__["parent_suite"] = spec.setting
        cells.append(cell)
    notes = [n for n in (_skipped_note(skipped),) if n]
    return PerTaskPairResult(spec=spec, tasks=cells, notes=notes)


def cells_in_order(pair: PairResult) -> list[CellStats]:
    return [pair.cells[OVERALL_KEY]]
