"""Build PairResult records for the STEP variant.

Mirrors `tri_statistics.pairs` but, instead of computing the McNemar mid-p at
cell-construction time, captures the *aligned paired sequences* of per-rollout
binary outcomes. The STEP test itself is run later, in
`corrections.apply_corrections`, where the family size (and therefore the
Bonferroni-split α) is known.

Cells without paired episode data (Flower / SmolVLA on LIBERO base) fall back
to the two-proportion z-test as primary, exactly like the original pipeline.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the existing aggregate loaders without duplicating them.
_LIBERO_DIR = (Path(__file__).resolve().parents[2] / 'libero')
sys.path.insert(0, str(_LIBERO_DIR.parent))  # for wilson_score_interval
sys.path.insert(0, str(_LIBERO_DIR))         # for analyze_libero_steps

from analyze_libero_steps import (  # type: ignore  # noqa: E402
    Counts,
    RUN_SPECS,
    load_run,
    make_overall,
    two_proportion_z_pvalue,
)
from wilson_score_interval import wilson_score_interval  # type: ignore  # noqa: E402

from .config import (  # noqa: E402
    ALL_CELLS,
    OVERALL_KEY,
    PairSpec,
    SUITE_ORDER,
)
from .io_episodes import (  # noqa: E402
    discordant_counts,
    group_episodes_by_task,
    load_dit_log_episodes,
    load_flower_libero_episodes,
    load_flower_liberoplus_episodes,
    load_lap_episodes,
    load_mibot_episodes,
    load_mibot_liberoplus_episodes,
    load_pi05_episodes,
    load_pi05_liberoplus_episodes,
    load_smolvla_paired_episodes,
    load_xvla_episodes,
    map_liberoplus_to_task_id,
    pair_flower_liberoplus_episodes,
    pair_pi05_liberoplus_episodes,
    paired_sequences,
    paired_sequences_per_task,
    resolve_dit_task_descriptions,
    resolve_lap_task_descriptions,
    resolve_mibot_task_descriptions,
    resolve_smolvla_task_descriptions,
)
from .stats import (  # noqa: E402
    fisher_exact_2x2,
)


@dataclass
class CellStats:
    suite: str
    counts_low: Counts
    counts_high: Counts
    wilson_low: tuple[float, float]
    wilson_high: tuple[float, float]
    p_z: float
    p_fisher: float
    # Paired-data diagnostics (kept for reporting parity with the original pipeline).
    n_paired: int | None = None
    discordant_b: int | None = None
    discordant_c: int | None = None
    both_success: int | None = None
    both_failure: int | None = None
    # Aligned per-rollout outcomes (low, high). Populated when paired data exists;
    # consumed by `corrections.apply_corrections` to run STEP at α_global and α_split.
    paired_low: list[bool] | None = None
    paired_high: list[bool] | None = None
    # z-test multiplicity (always filled in by apply_corrections).
    q_bonf_z: float | None = None
    q_holm_z: float | None = None
    q_bh_z: float | None = None
    # STEP outcomes (filled in by apply_corrections when paired data exists).
    step_n_max: int | None = None
    step_alpha_uncorrected: float | None = None
    step_alpha_corrected: float | None = None
    step_decision_uncorrected: str | None = None     # Decision enum .name
    step_decision_corrected:   str | None = None
    step_stop_time_uncorrected: int | None = None
    step_stop_time_corrected:   int | None = None
    # "Primary" test = STEP when paired sequences exist, else z.
    primary_test: str = "z"            # "step" or "z"
    # CLD letters under two thresholds:
    # - uncorrected: at α_global (no family correction)
    # - corrected:   at α_split = α_global / family_size  (STEP), or Holm-z (z fallback)
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
        """True iff every cell's CLD derives from STEP (paired) tests."""
        return all(c.primary_test == "step" for c in self.cells.values())

    @property
    def cld_basis(self) -> str:
        """Human-readable description of the test driving the CLD letters."""
        kinds = {c.primary_test for c in self.cells.values()}
        if kinds == {"step"}:
            return "paired STEP"
        if kinds == {"z"}:
            return "unpaired z"
        return "mixed paired/unpaired"


def _find_run_spec(model: str, dataset: str, steps: int):
    for spec in RUN_SPECS:
        if spec.model_key == model and spec.dataset == dataset and spec.steps == steps:
            return spec
    raise KeyError(f"No RunSpec for {model=} {dataset=} {steps=}")


def _wilson(c: Counts, confidence: float = 0.95) -> tuple[float, float]:
    return wilson_score_interval(c.successes, c.trials, confidence=confidence)


# XR-0 LIBERO+ uses singular "1step" for nsteps=1 but plural "5steps" otherwise.
def _mibot_liberoplus_dir(steps: int) -> Path:
    suffix = "step" if steps == 1 else "steps"
    return _LIBERO_DIR / f"mibot/logs_liberoplus_{steps}{suffix}"


def _dit_shards_dir(dataset: str, steps: int) -> Path:
    bench = "libero_eval" if dataset == "libero" else "liberoplus_eval"
    return _LIBERO_DIR / f"dit/pretrained/{bench}/final_model/sharded_n{steps}"


def _episode_maps(spec: PairSpec) -> tuple[dict | None, dict | None]:
    """Return per-suite episode dicts for (low, high) variants, or (None, None)."""
    if not spec.has_episode_data:
        return None, None
    low: dict[str, dict] = {}
    high: dict[str, dict] = {}
    if spec.model == "dit":
        low_dir = _dit_shards_dir(spec.dataset, spec.variant_low)
        high_dir = _dit_shards_dir(spec.dataset, spec.variant_high)
        for suite in SUITE_ORDER:
            low[suite] = load_dit_log_episodes(low_dir, suite)
            high[suite] = load_dit_log_episodes(high_dir, suite)
        return low, high
    if spec.model == "xvla":
        ds_prefix = "libero" if spec.dataset == "libero" else "liberoplus"
        low_dir = _LIBERO_DIR / f"X-VLA/{ds_prefix}_{spec.variant_low}steps"
        high_dir = _LIBERO_DIR / f"X-VLA/{ds_prefix}_{spec.variant_high}steps"
        for suite in SUITE_ORDER:
            lo = load_xvla_episodes(low_dir, suite)
            hi = load_xvla_episodes(high_dir, suite)
            low[suite] = lo or {}
            high[suite] = hi or {}
        return low, high
    if spec.dataset == "libero":
        if spec.model == "pi05":
            low_dir = _LIBERO_DIR / f"pi05/libero/numstep{spec.variant_low}"
            high_dir = _LIBERO_DIR / f"pi05/libero/numstep{spec.variant_high}"
            for suite in SUITE_ORDER:
                low[suite] = load_pi05_episodes(low_dir, suite)
                high[suite] = load_pi05_episodes(high_dir, suite)
            return low, high
        if spec.model == "lap":
            root = _LIBERO_DIR / "lap/lap_libero/results"
            for suite in SUITE_ORDER:
                low[suite] = load_lap_episodes(root / f"{suite}_ns{spec.variant_low}")
                high[suite] = load_lap_episodes(root / f"{suite}_ns{spec.variant_high}")
            return low, high
        if spec.model == "mibot":
            low_dir = _LIBERO_DIR / f"mibot/logs_{spec.variant_low}step"
            high_dir = _LIBERO_DIR / f"mibot/logs_{spec.variant_high}step"
            for suite in SUITE_ORDER:
                low[suite] = load_mibot_episodes(low_dir, suite) or {}
                high[suite] = load_mibot_episodes(high_dir, suite) or {}
            return low, high
        if spec.model == "flower":
            root = _LIBERO_DIR / "flower/libero_logs"
            for suite in SUITE_ORDER:
                low[suite] = load_flower_libero_episodes(root, spec.variant_low, suite)
                high[suite] = load_flower_libero_episodes(root, spec.variant_high, suite)
            return low, high
        if spec.model == "smolvla":
            low_path = _LIBERO_DIR / f"smolvla/libero_logs/paired/videos_smolvla_flowstep{spec.variant_low}_actionstep10_episodes.json"
            high_path = _LIBERO_DIR / f"smolvla/libero_logs/paired/videos_smolvla_flowstep{spec.variant_high}_actionstep10_episodes.json"
            for suite in SUITE_ORDER:
                low[suite] = load_smolvla_paired_episodes(low_path, suite)
                high[suite] = load_smolvla_paired_episodes(high_path, suite)
            return low, high

    if spec.dataset == "liberoplus":
        if spec.model == "smolvla":
            low_path = _LIBERO_DIR / f"smolvla/liberoplus_logs/paired/videos_smolvla_flowstep{spec.variant_low}_actionstep10_episodes.json"
            high_path = _LIBERO_DIR / f"smolvla/liberoplus_logs/paired/videos_smolvla_flowstep{spec.variant_high}_actionstep10_episodes.json"
            for suite in SUITE_ORDER:
                low[suite] = load_smolvla_paired_episodes(low_path, suite)
                high[suite] = load_smolvla_paired_episodes(high_path, suite)
            return low, high
        if spec.model == "lap":
            root = _LIBERO_DIR / "lap/libero_to_liberoplus"
            for suite in SUITE_ORDER:
                low[suite] = load_lap_episodes(root / f"{suite}_ns{spec.variant_low}")
                high[suite] = load_lap_episodes(root / f"{suite}_ns{spec.variant_high}")
            return low, high
        if spec.model == "pi05":
            low_dir = _LIBERO_DIR / f"pi05/libero_to_liberoplus/videos_pi05_flowstep{spec.variant_low}"
            high_dir = _LIBERO_DIR / f"pi05/libero_to_liberoplus/videos_pi05_flowstep{spec.variant_high}"
            for suite in SUITE_ORDER:
                _, lo_counts = load_pi05_liberoplus_episodes(low_dir, suite)
                _, hi_counts = load_pi05_liberoplus_episodes(high_dir, suite)
                lo, hi, _ = pair_pi05_liberoplus_episodes(lo_counts, hi_counts)
                low[suite] = lo
                high[suite] = hi
            return low, high
        if spec.model == "mibot":
            low_dir = _mibot_liberoplus_dir(spec.variant_low)
            high_dir = _mibot_liberoplus_dir(spec.variant_high)
            for suite in SUITE_ORDER:
                _, lo_counts = load_mibot_liberoplus_episodes(low_dir, suite)
                _, hi_counts = load_mibot_liberoplus_episodes(high_dir, suite)
                lo, hi, _ = pair_pi05_liberoplus_episodes(lo_counts, hi_counts)
                low[suite] = lo
                high[suite] = hi
            return low, high
        if spec.model == "flower":
            root = _LIBERO_DIR / "flower/liberoplus_logs"
            for suite in SUITE_ORDER:
                lo_counts = load_flower_liberoplus_episodes(root / suite, spec.variant_low)
                hi_counts = load_flower_liberoplus_episodes(root / suite, spec.variant_high)
                lo, hi, _ = pair_flower_liberoplus_episodes(lo_counts, hi_counts)
                low[suite] = lo
                high[suite] = hi
            return low, high
    return None, None


def build_pair_result(spec: PairSpec) -> PairResult:
    """Load both variants for a (model, dataset) pair and compute per-cell stats.

    STEP is NOT run here — it's deferred to `apply_corrections` so the family
    size is known when α_split is computed. This function only captures the
    paired sequences and the unpaired tests.
    """
    low_spec = _find_run_spec(spec.model, spec.dataset, spec.variant_low)
    high_spec = _find_run_spec(spec.model, spec.dataset, spec.variant_high)
    low_run = load_run(_LIBERO_DIR, low_spec)
    high_run = load_run(_LIBERO_DIR, high_spec)

    notes: list[str] = []

    suite_low_keys = set(low_run.suite_counts.keys())
    suite_high_keys = set(high_run.suite_counts.keys())
    suites_present = sorted(suite_low_keys & suite_high_keys, key=SUITE_ORDER.index)
    missing_low = suite_low_keys - suite_high_keys
    missing_high = suite_high_keys - suite_low_keys
    if missing_low or missing_high:
        notes.append(
            f"suite mismatch: low_only={sorted(missing_low)} high_only={sorted(missing_high)}"
        )

    eps_low_by_suite, eps_high_by_suite = _episode_maps(spec)

    cells: dict[str, CellStats] = {}
    for suite in suites_present:
        c_low = low_run.suite_counts[suite]
        c_high = high_run.suite_counts[suite]
        p_z = two_proportion_z_pvalue(c_low, c_high)
        p_fisher = fisher_exact_2x2(c_low.successes, c_low.trials, c_high.successes, c_high.trials)

        n_paired: int | None = None
        b = c = both_s = both_f = None
        seq_lo: list[bool] | None = None
        seq_hi: list[bool] | None = None

        if eps_low_by_suite is not None and eps_high_by_suite is not None:
            ep_low = eps_low_by_suite.get(suite, {})
            ep_high = eps_high_by_suite.get(suite, {})
            if ep_low and ep_high:
                n_paired, both_s, b, c, both_f = discordant_counts(ep_low, ep_high)
                seq_lo, seq_hi = paired_sequences(ep_low, ep_high)
                excluded = c_low.trials - n_paired
                if excluded > 0:
                    notes.append(
                        f"{suite}: paired set excludes {excluded} trial(s) "
                        f"(n_paired={n_paired}, aggregate n={c_low.trials})"
                    )

        cells[suite] = CellStats(
            suite=suite,
            counts_low=c_low,
            counts_high=c_high,
            wilson_low=_wilson(c_low),
            wilson_high=_wilson(c_high),
            p_z=p_z,
            p_fisher=p_fisher,
            n_paired=n_paired,
            discordant_b=b,
            discordant_c=c,
            both_success=both_s,
            both_failure=both_f,
            paired_low=seq_lo,
            paired_high=seq_hi,
        )

    # OVERALL row aggregates over the suites we actually have.
    suite_low = {k: v.counts_low for k, v in cells.items()}
    suite_high = {k: v.counts_high for k, v in cells.items()}
    overall_low = make_overall(suite_low)
    overall_high = make_overall(suite_high)
    p_z = two_proportion_z_pvalue(overall_low, overall_high)
    p_fisher = fisher_exact_2x2(
        overall_low.successes, overall_low.trials,
        overall_high.successes, overall_high.trials,
    )

    n_paired = b = c = both_s = both_f = None
    seq_lo = seq_hi = None
    if eps_low_by_suite is not None and eps_high_by_suite is not None:
        joined_low: dict = {}
        joined_high: dict = {}
        for suite in suites_present:
            for k, v in eps_low_by_suite.get(suite, {}).items():
                joined_low[(suite, *k)] = v
            for k, v in eps_high_by_suite.get(suite, {}).items():
                joined_high[(suite, *k)] = v
        if joined_low and joined_high:
            n_paired, both_s, b, c, both_f = discordant_counts(joined_low, joined_high)
            seq_lo, seq_hi = paired_sequences(joined_low, joined_high)

    cells[OVERALL_KEY] = CellStats(
        suite=OVERALL_KEY,
        counts_low=overall_low,
        counts_high=overall_high,
        wilson_low=_wilson(overall_low),
        wilson_high=_wilson(overall_high),
        p_z=p_z,
        p_fisher=p_fisher,
        n_paired=n_paired,
        discordant_b=b,
        discordant_c=c,
        both_success=both_s,
        both_failure=both_f,
        paired_low=seq_lo,
        paired_high=seq_hi,
    )
    return PairResult(spec=spec, cells=cells, notes=notes)


def cells_in_order(pair: PairResult) -> list[CellStats]:
    return [pair.cells[k] for k in ALL_CELLS if k in pair.cells]


# ---------------------------------------------------------------------------
# Per-task pair results
# ---------------------------------------------------------------------------

@dataclass
class CombinedSuiteVerdict:
    """Intersection-union summary over the per-task STEP decisions in one suite.

    Each per-task STEP runs at α_split = α_global / K (Bonferroni). The combined
    multitask claim H1: ∀τ p_high,τ > p_low,τ is *confirmed at α_global* iff
    every task's α_split STEP decided AcceptAlternative; symmetrically for the
    opposite direction. Any per-task FailToDecide leaves the combined claim
    unconfirmed (`NotConfirmed`); both directions appearing leaves it `Mixed`.
    """
    suite: str
    family_size: int                      # K — number of per-task subtests in this suite
    alpha_global: float                   # α (the level at which the combined claim is confirmed)
    alpha_split: float                    # α_s = α / K (per-task subtest level)
    n_alt_alpha_s: int                    # tasks rejecting H0 toward max-step at α_s
    n_null_alpha_s: int                   # tasks rejecting H0 toward 1-step at α_s
    n_undecided_alpha_s: int              # tasks with FailToDecide at α_s
    n_z_fallback: int                     # tasks without paired data (unexpected in current data)
    combined_decision: str                # AlternativeOnAll | NullOnAll | Mixed | NotConfirmed
    correction_applied: bool              # False when correction_family == "none"


@dataclass
class PerTaskPairResult:
    spec: PairSpec
    # {suite: [CellStats per task]}; CellStats.suite holds the task name,
    # CellStats.parent_suite holds the LIBERO suite it came from.
    suite_tasks: dict[str, list[CellStats]]
    notes: list[str] = field(default_factory=list)
    correction_family: str = "within_suite"
    # Filled in by apply_corrections_per_task after STEP runs.
    combined_verdicts: dict[str, CombinedSuiteVerdict] = field(default_factory=dict)

    @property
    def cld_basis(self) -> str:
        kinds: set[str] = set()
        for cells in self.suite_tasks.values():
            for c in cells:
                kinds.add(c.primary_test)
        if kinds == {"step"}:
            return "paired STEP"
        if kinds == {"z"}:
            return "unpaired z"
        return "mixed paired/unpaired"


def _make_task_cell(
    task_name: str,
    parent_suite: str,
    s_low: int, n_low: int,
    s_high: int, n_high: int,
    ep_low_task: dict[int, bool] | None,
    ep_high_task: dict[int, bool] | None,
) -> CellStats:
    """Build one CellStats for a single task, capturing paired sequences if available."""
    c_low = Counts(successes=s_low, trials=n_low)
    c_high = Counts(successes=s_high, trials=n_high)
    p_z = two_proportion_z_pvalue(c_low, c_high)
    p_fisher = fisher_exact_2x2(s_low, n_low, s_high, n_high)

    n_paired = b = c_ = both_s = both_f = None
    seq_lo: list[bool] | None = None
    seq_hi: list[bool] | None = None
    if ep_low_task is not None and ep_high_task is not None and ep_low_task and ep_high_task:
        shared_eps = set(ep_low_task) & set(ep_high_task)
        n_paired = len(shared_eps)
        both_s = both_f = b = c_ = 0
        for eid in shared_eps:
            sl, sh = ep_low_task[eid], ep_high_task[eid]
            if sl and sh:
                both_s += 1
            elif not sl and not sh:
                both_f += 1
            elif sl and not sh:
                b += 1
            else:
                c_ += 1
        seq_lo, seq_hi = paired_sequences_per_task(ep_low_task, ep_high_task)

    cell = CellStats(
        suite=task_name,
        counts_low=c_low,
        counts_high=c_high,
        wilson_low=_wilson(c_low),
        wilson_high=_wilson(c_high),
        p_z=p_z,
        p_fisher=p_fisher,
        n_paired=n_paired,
        discordant_b=b,
        discordant_c=c_,
        both_success=both_s,
        both_failure=both_f,
        paired_low=seq_lo,
        paired_high=seq_hi,
    )
    cell.__dict__["parent_suite"] = parent_suite
    return cell


def _regroup_liberoplus(
    task_map: dict[str, dict[int, bool]],
    suite: str,
    desc_map: dict[str, str] | None = None,
) -> tuple[dict[str, tuple[int, int]], dict[str, dict[int, bool]]]:
    """Remap LIBERO+ per-variant episode maps to 10 canonical task groups."""
    agg_s: dict[int, int] = defaultdict(int)
    agg_n: dict[int, int] = defaultdict(int)
    agg_eps: dict[int, dict[int, bool]] = defaultdict(dict)
    for variant_key, eps in task_map.items():
        lookup_key = desc_map.get(variant_key, variant_key) if desc_map else variant_key
        idx = map_liberoplus_to_task_id(lookup_key, suite)
        if idx is None:
            continue
        s = sum(eps.values())
        agg_s[idx] += s
        agg_n[idx] += len(eps)
        offset = len(agg_eps[idx])
        for sub_ep_id, success in eps.items():
            agg_eps[idx][offset + sub_ep_id] = success
    counts = {str(i): (agg_s[i], agg_n[i]) for i in range(10) if agg_n[i] > 0}
    eps_out = {str(i): dict(agg_eps[i]) for i in range(10) if agg_eps[i]}
    return counts, eps_out


def _regroup_liberoplus_counts(
    raw_counts: dict[str, tuple[int, int]],
    suite: str,
) -> dict[str, tuple[int, int]]:
    agg_s: dict[int, int] = defaultdict(int)
    agg_n: dict[int, int] = defaultdict(int)
    for variant_key, (s, n) in raw_counts.items():
        idx = map_liberoplus_to_task_id(variant_key, suite)
        if idx is None:
            continue
        agg_s[idx] += s
        agg_n[idx] += n
    return {str(i): (agg_s[i], agg_n[i]) for i in range(10) if agg_n[i] > 0}


def _get_per_task_counts_low_high(
    spec: PairSpec,
    suite: str,
    eps_low_by_suite: dict | None,
    eps_high_by_suite: dict | None,
) -> tuple[
    dict[str, tuple[int, int]],
    dict[str, tuple[int, int]],
    dict[str, dict[int, bool]] | None,
    dict[str, dict[int, bool]] | None,
    dict[str, str],
]:
    low_counts: dict[str, tuple[int, int]] = {}
    high_counts: dict[str, tuple[int, int]] = {}
    low_tasks: dict[str, dict[int, bool]] | None = None
    high_tasks: dict[str, dict[int, bool]] | None = None
    desc_map: dict[str, str] = {}

    if eps_low_by_suite is not None and eps_high_by_suite is not None:
        ep_low = eps_low_by_suite.get(suite, {})
        ep_high = eps_high_by_suite.get(suite, {})
        low_tasks_raw = group_episodes_by_task(ep_low)
        high_tasks_raw = group_episodes_by_task(ep_high)

        if spec.dataset == "liberoplus":
            id_desc: dict[str, str] | None = None
            if spec.model == "lap":
                root = _LIBERO_DIR / "lap/libero_to_liberoplus"
                id_desc = resolve_lap_task_descriptions(root / f"{suite}_ns{spec.variant_low}")
            elif spec.model == "smolvla":
                paired_path = _LIBERO_DIR / f"smolvla/liberoplus_logs/paired/videos_smolvla_flowstep{spec.variant_low}_actionstep10_episodes.json"
                id_desc = resolve_smolvla_task_descriptions(paired_path, suite)
            elif spec.model == "dit":
                low_results = _dit_shards_dir(spec.dataset, spec.variant_low) / suite / f"final_results_{suite}.json"
                id_desc = resolve_dit_task_descriptions(low_results)
            low_counts, low_tasks = _regroup_liberoplus(low_tasks_raw, suite, id_desc)
            high_counts, high_tasks = _regroup_liberoplus(high_tasks_raw, suite, id_desc)
        else:
            low_tasks = low_tasks_raw
            high_tasks = high_tasks_raw
            for task_key, episodes in low_tasks.items():
                s = sum(episodes.values())
                low_counts[task_key] = (s, len(episodes))
            for task_key, episodes in high_tasks.items():
                s = sum(episodes.values())
                high_counts[task_key] = (s, len(episodes))
            if spec.model == "lap":
                root = _LIBERO_DIR / "lap/lap_libero/results"
                desc_map = resolve_lap_task_descriptions(root / f"{suite}_ns{spec.variant_low}")
            elif spec.model == "mibot":
                low_dir = _LIBERO_DIR / f"mibot/logs_{spec.variant_low}step"
                desc_map = resolve_mibot_task_descriptions(low_dir, suite)
            elif spec.model == "smolvla":
                paired_path = _LIBERO_DIR / f"smolvla/libero_logs/paired/videos_smolvla_flowstep{spec.variant_low}_actionstep10_episodes.json"
                desc_map = resolve_smolvla_task_descriptions(paired_path, suite)
            elif spec.model == "dit":
                low_results = _dit_shards_dir(spec.dataset, spec.variant_low) / suite / f"final_results_{suite}.json"
                desc_map = resolve_dit_task_descriptions(low_results)
        return low_counts, high_counts, low_tasks, high_tasks, desc_map

    if spec.dataset == "liberoplus":
        if spec.model == "flower":
            root = _LIBERO_DIR / "flower/liberoplus_logs"
            lo_raw = load_flower_liberoplus_episodes(root / suite, spec.variant_low)
            hi_raw = load_flower_liberoplus_episodes(root / suite, spec.variant_high)
            low_counts = _regroup_liberoplus_counts(lo_raw, suite)
            high_counts = _regroup_liberoplus_counts(hi_raw, suite)
            return low_counts, high_counts, None, None, {}
        if spec.model in ("pi05", "mibot"):
            if spec.model == "pi05":
                low_dir = _LIBERO_DIR / f"pi05/libero_to_liberoplus/videos_pi05_flowstep{spec.variant_low}"
                high_dir = _LIBERO_DIR / f"pi05/libero_to_liberoplus/videos_pi05_flowstep{spec.variant_high}"
            else:
                low_dir = _mibot_liberoplus_dir(spec.variant_low)
                high_dir = _mibot_liberoplus_dir(spec.variant_high)
            _, lo_raw = load_pi05_liberoplus_episodes(low_dir, suite)
            _, hi_raw = load_pi05_liberoplus_episodes(high_dir, suite)
            low_counts = _regroup_liberoplus_counts(lo_raw, suite)
            high_counts = _regroup_liberoplus_counts(hi_raw, suite)
            return low_counts, high_counts, None, None, {}

    return low_counts, high_counts, None, None, {}


def build_per_task_pair_result(spec: PairSpec) -> PerTaskPairResult:
    """Build per-task CellStats for every suite of a (model, dataset) pair."""
    from .io_episodes import _get_base_task_names  # noqa: PLC0415
    eps_low_by_suite, eps_high_by_suite = _episode_maps(spec)
    notes: list[str] = []
    suite_tasks: dict[str, list[CellStats]] = {}

    for suite in SUITE_ORDER:
        low_counts, high_counts, low_tasks, high_tasks, desc_map = \
            _get_per_task_counts_low_high(spec, suite, eps_low_by_suite, eps_high_by_suite)

        if not low_counts or not high_counts:
            notes.append(f"{suite}: no per-task data available")
            continue

        if spec.dataset == "liberoplus":
            base_names = _get_base_task_names(suite)
            common_tasks = sorted(
                set(low_counts) & set(high_counts),
                key=lambda k: int(k) if k.isdigit() else k,
            )
            liberoplus_desc = {str(i): base_names[i] for i in range(len(base_names)) if str(i) in low_counts}
        else:
            base_names = []
            liberoplus_desc = {}
            common_tasks = sorted(
                set(low_counts) & set(high_counts),
                key=lambda k: desc_map.get(k, k),
            )

        if not common_tasks:
            notes.append(f"{suite}: no overlapping tasks between variants")
            continue

        task_cells: list[CellStats] = []
        for task_idx, task_key in enumerate(common_tasks):
            s_lo, n_lo = low_counts[task_key]
            s_hi, n_hi = high_counts[task_key]
            ep_lo = low_tasks.get(task_key) if low_tasks else None
            ep_hi = high_tasks.get(task_key) if high_tasks else None
            if spec.dataset == "liberoplus":
                display_name = liberoplus_desc.get(task_key, task_key)
                task_idx_display = int(task_key) if task_key.isdigit() else task_idx
            else:
                display_name = desc_map.get(task_key, task_key)
                task_idx_display = task_idx
            cell = _make_task_cell(
                task_name=display_name,
                parent_suite=suite,
                s_low=s_lo, n_low=n_lo,
                s_high=s_hi, n_high=n_hi,
                ep_low_task=ep_lo,
                ep_high_task=ep_hi,
            )
            cell.__dict__["task_idx"] = task_idx_display
            task_cells.append(cell)

        suite_tasks[suite] = task_cells

    return PerTaskPairResult(spec=spec, suite_tasks=suite_tasks, notes=notes)
