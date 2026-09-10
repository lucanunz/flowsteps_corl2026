"""Build PairResult records for the real-world STEP pipeline.

One pair per (model, task), each carrying a single binary cell on
`manual_success`. STEP itself is deferred to
`corrections.apply_corrections_for_model`, where the Bonferroni α-split
across the 4 tasks of the model can be applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import PairSpec
from .io_runs import load_paired_outcomes
from .stats import (
    Counts,
    two_proportion_z_pvalue,
    wilson_score_interval,
)


@dataclass
class CellStats:
    task: str
    counts_low: Counts
    counts_high: Counts
    wilson_low: tuple[float, float]
    wilson_high: tuple[float, float]
    p_z: float
    # Paired-binary diagnostics.
    n_paired: int
    discordant_b: int
    discordant_c: int
    both_success: int
    both_failure: int
    paired_low: list[bool]
    paired_high: list[bool]
    # z-test multiplicity (filled in by apply_corrections_for_model).
    q_bonf_z: float | None = None
    q_holm_z: float | None = None
    q_bh_z: float | None = None
    # STEP outputs (filled in by apply_corrections_for_model).
    step_n_max: int | None = None
    step_alpha_uncorrected: float | None = None
    step_alpha_corrected: float | None = None
    step_decision_uncorrected: str | None = None
    step_decision_corrected: str | None = None
    step_stop_time_uncorrected: int | None = None
    step_stop_time_corrected: int | None = None
    # Always "step" for real-world (paired data always available).
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

    @property
    def cld_basis(self) -> str:
        return "paired STEP"


def _wilson(c: Counts, confidence: float = 0.95) -> tuple[float, float]:
    return wilson_score_interval(c.successes, c.trials, confidence=confidence)


def build_pair_result(spec: PairSpec) -> PairResult:
    seq_low, seq_high = load_paired_outcomes(spec)
    n = len(seq_low)
    assert len(seq_high) == n, "loader must return equal-length sequences"
    if n == 0:
        raise ValueError(f"No completed pairs for {spec.model}/{spec.task}")

    successes_low = sum(seq_low)
    successes_high = sum(seq_high)
    c_low = Counts(successes=successes_low, trials=n)
    c_high = Counts(successes=successes_high, trials=n)

    both_success = sum(1 for l, h in zip(seq_low, seq_high) if l and h)
    both_failure = sum(1 for l, h in zip(seq_low, seq_high) if (not l) and (not h))
    b = sum(1 for l, h in zip(seq_low, seq_high) if l and (not h))
    c = sum(1 for l, h in zip(seq_low, seq_high) if (not l) and h)

    cell = CellStats(
        task=spec.task,
        counts_low=c_low,
        counts_high=c_high,
        wilson_low=_wilson(c_low),
        wilson_high=_wilson(c_high),
        p_z=two_proportion_z_pvalue(c_low, c_high),
        n_paired=n,
        discordant_b=b,
        discordant_c=c,
        both_success=both_success,
        both_failure=both_failure,
        paired_low=list(seq_low),
        paired_high=list(seq_high),
    )

    return PairResult(spec=spec, cells={spec.task: cell})


def cells_in_order(pair: PairResult) -> list[CellStats]:
    return [pair.cells[pair.spec.task]]
