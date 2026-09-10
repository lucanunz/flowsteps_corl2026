"""Build PairResult records for the CALVIN STEP pipeline.

One pair per (model, dataset). The pair carries a single binary cell —
`sr5` — for the STEP / z-fallback binary analysis, plus the per-cell
completion statistics (0-5 score) for the Welch / Dirichlet analysis.

STEP itself is not run here; it is deferred to `corrections.apply_corrections`
so the family α-split is applied consistently with the LIBERO pipeline
(family size = 1 here, so α_split = α_global, but the API stays the same).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import CELL_KEYS, PairSpec
from .io_runs import (
    RunResult,
    load_pair,
    paired_completion_scores,
    paired_sr5_sequences,
)
from .stats import (
    CompletionStats,
    Counts,
    WelchTestResult,
    build_completion_stats,
    completion_counts_from_raw,
    two_proportion_z_pvalue,
    welch_t_test,
    wilson_score_interval,
)


@dataclass
class CellStats:
    suite: str  # cell key, e.g. "sr5"
    counts_low: Counts
    counts_high: Counts
    wilson_low: tuple[float, float]
    wilson_high: tuple[float, float]
    p_z: float
    # Paired-binary diagnostics (populated when paired pickle data exists).
    n_paired: int | None = None
    discordant_b: int | None = None
    discordant_c: int | None = None
    both_success: int | None = None
    both_failure: int | None = None
    paired_low: list[bool] | None = None
    paired_high: list[bool] | None = None
    # z-test multiplicity (family of 1 → q = p).
    q_bonf_z: float | None = None
    q_holm_z: float | None = None
    q_bh_z: float | None = None
    # STEP outcomes (filled in by apply_corrections when paired data exists).
    step_n_max: int | None = None
    step_alpha_uncorrected: float | None = None
    step_alpha_corrected: float | None = None
    step_decision_uncorrected: str | None = None
    step_decision_corrected: str | None = None
    step_stop_time_uncorrected: int | None = None
    step_stop_time_corrected: int | None = None
    # "Primary" test for the binary cell.
    primary_test: str = "z"            # "step" or "z"
    cld_low_uncorrected: str = "a"
    cld_high_uncorrected: str = "a"
    cld_low_corrected: str = "a"
    cld_high_corrected: str = "a"
    # Completion-score (0-5) per-cell statistics.
    completion_low: CompletionStats | None = None
    completion_high: CompletionStats | None = None
    welch: WelchTestResult | None = None
    welch_cld_low: str = "a"
    welch_cld_high: str = "a"
    # Whether `completion_*` were derived from per-rollout data (raw pickles
    # or parsed log.txt) on both sides — True iff both sides have per-rollout
    # integer scores. Kept for back-compat; `completion_source_*` describe the
    # actual source.
    completion_from_raw: bool = False
    completion_source_low: str = "reconstructed"
    completion_source_high: str = "reconstructed"


@dataclass
class PairResult:
    spec: PairSpec
    low_run: RunResult
    high_run: RunResult
    cells: dict[str, CellStats]
    notes: list[str] = field(default_factory=list)

    @property
    def cld_basis(self) -> str:
        kinds = {c.primary_test for c in self.cells.values()}
        if kinds == {"step"}:
            return "paired STEP"
        if kinds == {"z"}:
            return "unpaired z"
        return "mixed paired/unpaired"


def _wilson(c: Counts, confidence: float = 0.95) -> tuple[float, float]:
    return wilson_score_interval(c.successes, c.trials, confidence=confidence)


def _sr5_counts(run: RunResult) -> Counts:
    n = run.num_sequences
    s = int(round(run.chain_sr["5"] * n))
    return Counts(successes=s, trials=n)


def build_pair_result(spec: PairSpec) -> PairResult:
    """Load both variants for a (model, dataset) pair and build the SR@5 cell."""
    low_run, high_run = load_pair(spec)

    notes: list[str] = []

    # Binary SR@5 counts from aggregate chain_sr.
    c_low = _sr5_counts(low_run)
    c_high = _sr5_counts(high_run)
    p_z = two_proportion_z_pvalue(c_low, c_high)

    # Paired SR@5 sequences (only when a per-rollout source exists).
    paired_seqs = paired_sr5_sequences(spec, low_run, high_run) if spec.has_paired_data else None
    n_paired = b = c_disc = both_s = both_f = None
    seq_lo: list[bool] | None = None
    seq_hi: list[bool] | None = None
    if paired_seqs is not None:
        seq_lo, seq_hi = paired_seqs
        n_paired = len(seq_lo)
        both_s = sum(1 for l, h in zip(seq_lo, seq_hi) if l and h)
        both_f = sum(1 for l, h in zip(seq_lo, seq_hi) if (not l) and (not h))
        b = sum(1 for l, h in zip(seq_lo, seq_hi) if l and (not h))
        c_disc = sum(1 for l, h in zip(seq_lo, seq_hi) if (not l) and h)
        # Sanity check: paired SR@5 count must match aggregate chain_sr count.
        s_low_paired = sum(seq_lo)
        s_high_paired = sum(seq_hi)
        if s_low_paired != c_low.successes or s_high_paired != c_high.successes:
            notes.append(
                f"paired SR@5 counts ({s_low_paired}/{s_high_paired}) disagree with "
                f"aggregate chain_sr ({c_low.successes}/{c_high.successes}); using paired"
            )
            c_low = Counts(successes=s_low_paired, trials=n_paired)
            c_high = Counts(successes=s_high_paired, trials=n_paired)
            p_z = two_proportion_z_pvalue(c_low, c_high)

    # Completion stats (0-5) — per-rollout source preferred (raw pickles for
    # XR-0, parsed log.txt for X-VLA), chain_sr reconstruction otherwise.
    completion_low = build_completion_stats(low_run)
    completion_high = build_completion_stats(high_run)
    source_low = "raw pickles" if completion_low.is_paired_capable else "reconstructed"
    source_high = "raw pickles" if completion_high.is_paired_capable else "reconstructed"

    paired_completion = paired_completion_scores(spec, low_run, high_run) if spec.has_paired_data else None
    if paired_completion is not None:
        low_scores, high_scores = paired_completion
        completion_low = CompletionStats(
            counts=completion_counts_from_raw(low_scores),
            n=len(low_scores),
            mean=_mean(low_scores),
            variance=_variance(low_scores),
            is_paired_capable=True,
        )
        completion_high = CompletionStats(
            counts=completion_counts_from_raw(high_scores),
            n=len(high_scores),
            mean=_mean(high_scores),
            variance=_variance(high_scores),
            is_paired_capable=True,
        )
        if spec.model == "xvla":
            source_low = source_high = "X-VLA log.txt"
        elif spec.model == "flower":
            source_low = source_high = "per-episode JSON"
        else:
            source_low = source_high = "raw pickles"

    completion_from_raw = (
        completion_low.is_paired_capable and completion_high.is_paired_capable
    )

    welch = welch_t_test(completion_high, completion_low)

    cell = CellStats(
        suite="sr5",
        counts_low=c_low,
        counts_high=c_high,
        wilson_low=_wilson(c_low),
        wilson_high=_wilson(c_high),
        p_z=p_z,
        n_paired=n_paired,
        discordant_b=b,
        discordant_c=c_disc,
        both_success=both_s,
        both_failure=both_f,
        paired_low=seq_lo,
        paired_high=seq_hi,
        completion_low=completion_low,
        completion_high=completion_high,
        welch=welch,
        completion_from_raw=completion_from_raw,
        completion_source_low=source_low,
        completion_source_high=source_high,
    )

    return PairResult(
        spec=spec,
        low_run=low_run,
        high_run=high_run,
        cells={"sr5": cell},
        notes=notes,
    )


def _mean(values) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    m = _mean(values)
    return sum((v - m) ** 2 for v in values) / (n - 1)


def cells_in_order(pair: PairResult) -> list[CellStats]:
    return [pair.cells[k] for k in CELL_KEYS if k in pair.cells]
