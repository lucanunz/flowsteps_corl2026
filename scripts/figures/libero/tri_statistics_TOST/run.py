#!/usr/bin/env python3
"""Paired equivalence (TOST) statistics for LIBERO / LIBERO+.

Mirrors tri_statistics_STEP/run.py but runs an equivalence test instead of a
difference test. Data loading is reused from tri_statistics_STEP.

Usage (from the libero/ directory or anywhere):
    python tri_statistics_TOST/run.py
    python tri_statistics_TOST/run.py --dataset libero --delta-pp 3
    python tri_statistics_TOST/run.py --per-task
    python tri_statistics_TOST/run.py --group-suites   # Spatial+Object+Goal vs LIBERO-10
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent     # tri_statistics_TOST/
_LIBERO_DIR = _PKG_DIR.parent                  # libero/
_ROOT_DIR = _LIBERO_DIR.parent                 # corl2026_logs/

for _p in (_ROOT_DIR, _LIBERO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Reuse the STEP package's data loaders (they fill the paired 2×2 table without
# running STEP).
from tri_statistics_STEP.pairs import (  # noqa: E402
    CellStats,
    PairResult,
    PerTaskPairResult,
    build_pair_result,
    build_per_task_pair_result,
)
from wilson_score_interval import wilson_score_interval  # noqa: E402

# LIBERO suite keys: the three short-horizon suites grouped by --group-suites,
# and the long-horizon suite kept separate.
_SHORT_SUITE_KEYS = ("libero_spatial", "libero_object", "libero_goal")
_LONG_SUITE_KEY = "libero_10"
_SHORT_GROUP_LABEL = "Spatial+Object+Goal"

from tri_statistics_TOST.config import (  # noqa: E402
    ALL_CELLS,
    DATASETS,
    DELTA_MARGIN_PP,
    MODEL_DISPLAY,
    PAIR_SPECS,
    SUITE_DISPLAY,
    TOST_ALPHA,
)
from tri_statistics_TOST.equivalence import (  # noqa: E402
    apply_equivalence,
    apply_equivalence_per_task,
)
from tri_statistics_TOST.plotting import (  # noqa: E402
    plot_equivalence_forest,
    plot_per_task_equivalence_forest,
)
from tri_statistics_TOST.report import (  # noqa: E402
    write_pair_report,
    write_per_task_report,
)
from tri_statistics_TOST.stats import (  # noqa: E402
    EQUIVALENT,
    INCONCLUSIVE,
    RELEVANT_DIFFERENCE,
    VERDICT_CODE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate paired equivalence (TOST) statistics for LIBERO / LIBERO+."
    )
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Root output directory (default: tri_statistics_TOST/outputs/).")
    p.add_argument("--confidence", type=float, default=0.95,
                   help="Confidence for the Wilson per-arm CIs in the SR columns "
                        "(default: 0.95). The TOST level is set by --alpha.")
    p.add_argument("--alpha", type=float, default=TOST_ALPHA,
                   help=f"One-sided TOST level; equivalence CI is 1-2α two-sided "
                        f"(default: {TOST_ALPHA}).")
    p.add_argument("--delta-pp", type=float, default=DELTA_MARGIN_PP,
                   help=f"Pre-registered equivalence margin in percentage points "
                        f"(default: {DELTA_MARGIN_PP}).")
    p.add_argument("--dataset", choices=["libero", "liberoplus", "all"], default="all",
                   help="Which dataset(s) to process (default: all).")
    p.add_argument("--per-task", action="store_true",
                   help="Run per-task analysis instead of per-suite.")
    p.add_argument("--group-suites", action="store_true",
                   help="Pool Spatial+Object+Goal into one cell and keep LIBERO-10 "
                        "separate (one decision each). Writes to outputs_grouped/ by "
                        "default so the standard outputs are not overwritten.")
    return p.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.group_suites:
        return _PKG_DIR / "outputs_grouped"
    return _PKG_DIR / "outputs"


def _aggregate_cells(cells: list, label: str, confidence: float) -> CellStats:
    """Pool several suite cells into one cell (success rates + paired 2×2).

    Mirrors how the Overall cell is built, but over an arbitrary subset of
    suites. The TOST reads only the pooled paired 2×2; the SR columns use the
    pooled aggregate counts.
    """
    counts_cls = type(cells[0].counts_low)
    s_lo = sum(c.counts_low.successes for c in cells)
    n_lo = sum(c.counts_low.trials for c in cells)
    s_hi = sum(c.counts_high.successes for c in cells)
    n_hi = sum(c.counts_high.trials for c in cells)

    paired = [c for c in cells if c.both_success is not None and c.n_paired]
    if paired:
        a = sum(c.both_success for c in paired)
        b = sum(c.discordant_b for c in paired)
        cc = sum(c.discordant_c for c in paired)
        d = sum(c.both_failure for c in paired)
        n_paired = a + b + cc + d
        seq_lo = [x for c in paired for x in (c.paired_low or [])]
        seq_hi = [x for c in paired for x in (c.paired_high or [])]
    else:
        a = b = cc = d = n_paired = None
        seq_lo = seq_hi = None

    return CellStats(
        suite=label,
        counts_low=counts_cls(successes=s_lo, trials=n_lo),
        counts_high=counts_cls(successes=s_hi, trials=n_hi),
        wilson_low=wilson_score_interval(s_lo, n_lo, confidence=confidence),
        wilson_high=wilson_score_interval(s_hi, n_hi, confidence=confidence),
        p_z=1.0, p_fisher=1.0,
        n_paired=n_paired, discordant_b=b, discordant_c=cc,
        both_success=a, both_failure=d,
        paired_low=seq_lo, paired_high=seq_hi,
    )


def _regroup_short_vs_long(pair: PairResult, confidence: float) -> None:
    """Rewrite pair.cells to {short-group, LIBERO-10} for --group-suites mode."""
    short_cells = [pair.cells[k] for k in _SHORT_SUITE_KEYS if k in pair.cells]
    new_cells: dict = {}
    if short_cells:
        new_cells["short"] = _aggregate_cells(short_cells, _SHORT_GROUP_LABEL, confidence)
    if _LONG_SUITE_KEY in pair.cells:
        new_cells[_LONG_SUITE_KEY] = pair.cells[_LONG_SUITE_KEY]
    pair.cells = new_cells
    pair.__dict__["cell_order"] = list(new_cells.keys())


def main() -> int:
    args = parse_args()
    out_root = _resolve_output_dir(args)
    delta = args.delta_pp / 100.0
    datasets_to_run = list(DATASETS) if args.dataset == "all" else [args.dataset]

    if args.per_task:
        if args.group_suites:
            raise SystemExit("--group-suites is a per-suite option; drop --per-task.")
        return _run_per_task(args, out_root, delta, datasets_to_run)

    print(f"Equivalence margin δ = {args.delta_pp:.1f}pp  |  TOST α = {args.alpha}  "
          f"(⇒ {1 - 2 * args.alpha:.0%} CI)")
    if args.group_suites:
        print(f"Suite grouping: '{_SHORT_GROUP_LABEL}' (one cell) + LIBERO-10 (separate)")
    print(f"Output root: {out_root}")
    print("Loading pair results …")
    pairs: list[PairResult] = []
    skipped = 0
    for spec in PAIR_SPECS:
        if spec.dataset not in datasets_to_run:
            continue
        try:
            pair = build_pair_result(spec)
        except FileNotFoundError as exc:
            print(f"  [SKIP]  {spec.model:8s} {spec.dataset}: file not found — {exc.filename}")
            skipped += 1
            continue
        except Exception as exc:
            print(f"  [ERROR] {spec.model:8s} {spec.dataset}: {exc}")
            traceback.print_exc()
            skipped += 1
            continue
        if args.group_suites:
            _regroup_short_vs_long(pair, args.confidence)
        apply_equivalence(pair, delta=delta, alpha=args.alpha)
        pairs.append(pair)
        for note in pair.notes:
            print(f"  [NOTE]  {spec.model} {spec.dataset}: {note}")

    if not pairs:
        print("No pair results loaded — nothing to do.")
        return 1
    print(f"Loaded {len(pairs)} pair(s), skipped {skipped}.\n")

    print("Generating per-pair figures and reports …")
    for pair in pairs:
        fig_dir = out_root / pair.spec.dataset / "figures"
        forest_path = fig_dir / f"{pair.spec.model}_equivalence_forest.png"
        try:
            plot_equivalence_forest(pair, forest_path, delta_pp=args.delta_pp)
            print(f"  [PLOT]   {forest_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  forest {pair.spec.model} {pair.spec.dataset}: {exc}")
            traceback.print_exc()
        report_path = out_root / pair.spec.dataset / f"{pair.spec.model}_report.md"
        try:
            write_pair_report(pair, report_path, confidence=args.confidence,
                              delta_pp=args.delta_pp)
            print(f"  [REPORT] {report_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  report {pair.spec.model} {pair.spec.dataset}: {exc}")
            traceback.print_exc()

    _print_summary(pairs, args.delta_pp)
    return 0


def _run_per_task(args, out_root: Path, delta: float, datasets_to_run: list[str]) -> int:
    print(f"Per-task equivalence  (δ = {args.delta_pp:.1f}pp, α = {args.alpha}) …")
    print(f"Output root: {out_root}")
    results: list[PerTaskPairResult] = []
    skipped = 0
    for spec in PAIR_SPECS:
        if spec.dataset not in datasets_to_run:
            continue
        try:
            result = build_per_task_pair_result(spec)
        except FileNotFoundError as exc:
            print(f"  [SKIP]  {spec.model:8s} {spec.dataset}: file not found — {exc.filename}")
            skipped += 1
            continue
        except Exception as exc:
            print(f"  [ERROR] {spec.model:8s} {spec.dataset}: {exc}")
            traceback.print_exc()
            skipped += 1
            continue
        apply_equivalence_per_task(result, delta=delta, alpha=args.alpha)
        results.append(result)
        for note in result.notes:
            print(f"  [NOTE]  {spec.model} {spec.dataset}: {note}")

    if not results:
        print("No per-task results loaded — nothing to do.")
        return 1
    print(f"Loaded {len(results)} result(s), skipped {skipped}.\n")

    print("Generating per-task figures and reports …")
    for result in results:
        fig_dir = out_root / result.spec.dataset / "per_task_figures"
        forest_path = fig_dir / f"{result.spec.model}_task_equivalence_forest.png"
        try:
            plot_per_task_equivalence_forest(result, forest_path, delta_pp=args.delta_pp)
            print(f"  [PLOT]   {forest_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  task_forest {result.spec.model} {result.spec.dataset}: {exc}")
            traceback.print_exc()
        report_path = out_root / result.spec.dataset / f"{result.spec.model}_per_task_report.md"
        try:
            write_per_task_report(result, report_path, confidence=args.confidence,
                                  delta_pp=args.delta_pp)
            print(f"  [REPORT] {report_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  report {result.spec.model} {result.spec.dataset}: {exc}")
            traceback.print_exc()

    _print_per_task_summary(results, args.delta_pp)
    return 0


def _print_summary(pairs: list[PairResult], delta_pp: float) -> None:
    W = 120
    print()
    print("=" * W)
    print(f"{'Model':<10} {'Dataset':<12} {'Suite':<16} {'1-step':>8} "
          f"{'ref':>8} {'Δ pp':>7} {'90% CI (pp)':>20}  verdict")
    print("-" * W)
    for pair in sorted(pairs, key=lambda p: (p.spec.dataset, p.spec.model)):
        order = getattr(pair, "cell_order", None) or [k for k in ALL_CELLS if k in pair.cells]
        for key in order:
            if key not in pair.cells:
                continue
            cell = pair.cells[key]
            model_disp = MODEL_DISPLAY.get(pair.spec.model, pair.spec.model)
            suite_disp = SUITE_DISPLAY.get(cell.suite, cell.suite)
            tost = getattr(cell, "tost", None)
            if tost is None:
                verdict = "not assessed"
                ci = "—"
                d = "  —  "
            else:
                verdict = VERDICT_CODE[tost.verdict] + " " + tost.verdict
                ci = f"[{tost.ci_low*100:+.1f}, {tost.ci_high*100:+.1f}]"
                d = f"{tost.delta_hat*100:+.1f}"
            print(f"{model_disp:<10} {pair.spec.dataset:<12} {suite_disp:<16} "
                  f"{cell.counts_low.rate*100:>7.1f}% {cell.counts_high.rate*100:>7.1f}% "
                  f"{d:>7} {ci:>20}  {verdict}")
    print("=" * W)
    print(f"δ = ±{delta_pp:.0f}pp. equivalent = 90% CI ⊂ ±δ; difference>δ = CI outside ±δ; "
          "inconclusive = CI straddles a boundary.")


def _print_per_task_summary(results: list[PerTaskPairResult], delta_pp: float) -> None:
    W = 132
    print()
    print("=" * W)
    print("Combined equivalence verdict per suite (intersection–union, no Bonferroni)")
    print("-" * W)
    print(f"{'Model':<10} {'Dataset':<12} {'Suite':<14} {'K':>3} "
          f"{'equiv':>6} {'diff':>5} {'incon':>6} {'n/a':>4}  Combined")
    print("-" * W)
    for result in sorted(results, key=lambda r: (r.spec.dataset, r.spec.model)):
        model_disp = MODEL_DISPLAY.get(result.spec.model, result.spec.model)
        for suite, v in result.combined_equiv.items():
            suite_disp = SUITE_DISPLAY.get(suite, suite)
            print(f"{model_disp:<10} {result.spec.dataset:<12} {suite_disp:<14} "
                  f"{v.family_size:>3} {v.n_equivalent:>6} {v.n_relevant:>5} "
                  f"{v.n_inconclusive:>6} {v.n_unassessed:>4}  {v.combined_decision}")
    print("=" * W)
    print("EquivalentOnAll = every task equivalent within ±δ at α (no Bonferroni; "
          "IUT). Small per-task n ⇒ mostly inconclusive — expected.")


if __name__ == "__main__":
    raise SystemExit(main())
