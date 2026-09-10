#!/usr/bin/env python3
"""STEP-based within-model 1-step vs max-step statistics for LIBERO / LIBERO+.

Usage (from the libero/ directory):
    python tri_statistics_STEP/run.py
    python tri_statistics_STEP/run.py --dataset libero
    python tri_statistics_STEP/run.py --output-dir /path/to/outputs --confidence 0.95
    python tri_statistics_STEP/run.py --per-task --correction-family within_suite

Usage (from inside tri_statistics_STEP/):
    python run.py
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent    # tri_statistics_STEP/
_LIBERO_DIR = _PKG_DIR.parent                 # libero/
_ROOT_DIR = _LIBERO_DIR.parent                # corl2026_logs/

for _p in (_ROOT_DIR, _LIBERO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tri_statistics_STEP.config import (  # noqa: E402
    ALL_CELLS,
    DATASETS,
    MODEL_DISPLAY,
    PAIR_SPECS,
    SUITE_DISPLAY,
)
from tri_statistics_STEP.corrections import (  # noqa: E402
    apply_corrections,
    apply_corrections_per_task,
)
from tri_statistics_STEP.pairs import (  # noqa: E402
    PairResult,
    PerTaskPairResult,
    build_pair_result,
    build_per_task_pair_result,
)
from tri_statistics_STEP.plotting import (  # noqa: E402
    plot_dataset_overall_comparison,
    plot_pair_bars,
    plot_pair_violins,
    plot_per_task_appendix_combined,
    plot_per_task_bars,
    plot_per_task_paper_violins,
    plot_per_task_paper_violins_appendix,
    plot_per_task_violins,
    plot_task_heatmap,
)
from tri_statistics_STEP.report import write_pair_report, write_per_task_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate STEP-based within-model 1-step vs max-step statistics "
            "for LIBERO and LIBERO+."
        )
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root output directory (default: tri_statistics_STEP/outputs/).",
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for Wilson intervals and FWER control (default: 0.95).",
    )
    p.add_argument(
        "--dataset",
        choices=["libero", "liberoplus", "all"],
        default="all",
        help="Which dataset(s) to process (default: all).",
    )
    p.add_argument(
        "--per-task",
        action="store_true",
        help="Run per-task analysis instead of per-suite.",
    )
    p.add_argument(
        "--appendix-layout",
        choices=["2x2", "1x4"],
        default="2x2",
        help=(
            "Layout for the per-task appendix violin PDFs: '2x2' (two rows of "
            "two suites; default) or '1x4' (all four suites in one wide row)."
        ),
    )
    p.add_argument(
        "--appendix-width",
        type=float,
        default=5.5,
        help=(
            "Target text/column width in inches for the appendix violin PDFs. "
            "The figure is authored at this width so \\includegraphics[width=\\linewidth] "
            "applies no scaling (default: 5.5)."
        ),
    )
    p.add_argument(
        "--correction-family",
        choices=["within_suite", "dataset", "pair"],
        default=None,
        help=(
            "Multiplicity correction family. "
            "Per-suite mode accepts 'pair' (default; α_split = α / (4 suites + Overall)). "
            "Per-task mode accepts 'within_suite' (default; ≤10 tasks per suite) "
            "or 'dataset' (all tasks across suites)."
        ),
    )
    return p.parse_args()


def _resolve_correction_family(args: argparse.Namespace) -> str:
    """Resolve the correction family, applying mode-specific defaults and
    rejecting per-task-only values in per-suite mode."""
    cf = args.correction_family
    if args.per_task:
        if cf is None:
            return "within_suite"
        if cf == "pair":
            raise SystemExit(
                "--correction-family pair is invalid in --per-task mode "
                "(use within_suite or dataset)."
            )
        return cf
    # per-suite mode
    if cf is None:
        return "pair"
    if cf in ("within_suite", "dataset"):
        raise SystemExit(
            f"--correction-family {cf} is only valid with --per-task "
            "(per-suite mode accepts: pair)."
        )
    return cf


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return _PKG_DIR / "outputs"


def main() -> int:
    args = parse_args()
    correction_family = _resolve_correction_family(args)
    out_root: Path = _resolve_output_dir(args)
    alpha = 1.0 - args.confidence
    datasets_to_run = list(DATASETS) if args.dataset == "all" else [args.dataset]

    if args.per_task:
        return _run_per_task(args, out_root, alpha, datasets_to_run, correction_family)

    print(f"Correction family: {correction_family}")
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

        apply_corrections(pair, alpha=alpha, correction_family=correction_family)
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
        fig_dir.mkdir(parents=True, exist_ok=True)

        violins_path = fig_dir / f"{pair.spec.model}_violins.png"
        try:
            plot_pair_violins(pair, violins_path)
            print(f"  [PLOT]   {violins_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  violin {pair.spec.model} {pair.spec.dataset}: {exc}")

        bars_path = fig_dir / f"{pair.spec.model}_bars.png"
        try:
            plot_pair_bars(pair, bars_path)
            print(f"  [PLOT]   {bars_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  bars   {pair.spec.model} {pair.spec.dataset}: {exc}")

        report_path = out_root / pair.spec.dataset / f"{pair.spec.model}_report.md"
        try:
            write_pair_report(pair, report_path, confidence=args.confidence,
                              correction_family=correction_family)
            print(f"  [REPORT] {report_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  report {pair.spec.model} {pair.spec.dataset}: {exc}")

    print("\nGenerating per-dataset overview plots …")
    for dataset in datasets_to_run:
        ds_pairs = [p for p in pairs if p.spec.dataset == dataset]
        if not ds_pairs:
            continue
        overview_path = out_root / dataset / "figures" / "overview_bars.png"
        try:
            plot_dataset_overall_comparison(ds_pairs, overview_path, dataset=dataset)
            print(f"  [PLOT]   {overview_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  overview {dataset}: {exc}")

    _print_summary(pairs)
    return 0


def _run_per_task(
    args: argparse.Namespace,
    out_root: Path,
    alpha: float,
    datasets_to_run: list[str],
    correction_family: str,
) -> int:
    print(f"Per-task mode  (correction family: {correction_family}) …")
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

        apply_corrections_per_task(result, correction_family=correction_family, alpha=alpha)
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
        fig_dir.mkdir(parents=True, exist_ok=True)

        violins_path = fig_dir / f"{result.spec.model}_task_violins.png"
        try:
            plot_per_task_violins(result, violins_path)
            print(f"  [PLOT]   {violins_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  task_violins {result.spec.model} {result.spec.dataset}: {exc}")

        paper_violins_path = fig_dir / f"{result.spec.model}_task_paper_violins.png"
        try:
            plot_per_task_paper_violins(result, paper_violins_path)
            print(f"  [PLOT]   {paper_violins_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  task_paper_violins {result.spec.model} {result.spec.dataset}: {exc}")

        appendix_dir = out_root / result.spec.dataset / "appendix_figures"
        appendix_dir.mkdir(parents=True, exist_ok=True)
        appendix_pdf = appendix_dir / f"{result.spec.model}_task_paper_violins.pdf"
        try:
            plot_per_task_paper_violins_appendix(
                result, appendix_pdf,
                layout=args.appendix_layout,
                target_width_in=args.appendix_width,
            )
            print(f"  [PLOT]   {appendix_pdf.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  appendix_violins {result.spec.model} {result.spec.dataset}: {exc}")

        bars_path = fig_dir / f"{result.spec.model}_task_bars.png"
        try:
            plot_per_task_bars(result, bars_path)
            print(f"  [PLOT]   {bars_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  task_bars {result.spec.model} {result.spec.dataset}: {exc}")

        heat_path = fig_dir / f"{result.spec.model}_task_heatmap.png"
        try:
            plot_task_heatmap(result, heat_path)
            print(f"  [PLOT]   {heat_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  heatmap {result.spec.model} {result.spec.dataset}: {exc}")

        report_path = out_root / result.spec.dataset / f"{result.spec.model}_per_task_report.md"
        try:
            write_per_task_report(result, report_path,
                                  confidence=args.confidence,
                                  correction_family=correction_family)
            print(f"  [REPORT] {report_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  report {result.spec.model} {result.spec.dataset}: {exc}")

    # Compact per-model appendix figures: both datasets in one canvas
    # (suites as rows, datasets as columns).
    print("\nGenerating combined per-model appendix figures …")
    by_model: dict[str, list[PerTaskPairResult]] = {}
    for result in results:
        by_model.setdefault(result.spec.model, []).append(result)
    combined_dir = out_root / "appendix_combined"
    for model, model_results in by_model.items():
        combined_pdf = combined_dir / f"{model}_task_paper_violins.pdf"
        try:
            plot_per_task_appendix_combined(
                model_results, combined_pdf,
                target_width_in=args.appendix_width,
            )
            print(f"  [PLOT]   {combined_pdf.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  combined_appendix {model}: {exc}")

    _print_per_task_summary(results)
    return 0


def _fmt_dec(name: str | None) -> str:
    if name is None:
        return "    —"
    return {
        "AcceptAlternative": "high>low",
        "AcceptNull":        "low>high",
        "FailToDecide":      "  —    ",
    }.get(name, name)


def _print_per_task_summary(results: list[PerTaskPairResult]) -> None:
    W = 130
    print()
    print("=" * W)
    print(
        f"{'Model':<10} {'Dataset':<12} {'Suite':<12} {'Task':<38}"
        f" {'Δ pp':>7} {'STEP α_g':>9} {'STEP α_s':>9}  CLD"
    )
    print("-" * W)
    for result in sorted(results, key=lambda r: (r.spec.dataset, r.spec.model)):
        for suite, cells in result.suite_tasks.items():
            suite_disp = SUITE_DISPLAY.get(suite, suite)
            model_disp = MODEL_DISPLAY.get(result.spec.model, result.spec.model)
            for cell in cells:
                delta = (cell.counts_high.rate - cell.counts_low.rate) * 100
                task_short = cell.suite[:36]
                dec_g = _fmt_dec(cell.step_decision_uncorrected) if cell.primary_test == "step" else "    z"
                dec_s = _fmt_dec(cell.step_decision_corrected) if cell.primary_test == "step" else "    z"
                print(
                    f"{model_disp:<10} {result.spec.dataset:<12} {suite_disp:<12} {task_short:<38}"
                    f" {delta:>+6.1f}pp {dec_g:>9} {dec_s:>9}  "
                    f"{cell.cld_low_corrected}/{cell.cld_high_corrected}"
                )
    print("=" * W)
    print("STEP α_g = decision at α_global; STEP α_s = decision at α_split (Bonferroni). 'z' = no paired data, z-test used.")

    # Per-suite combined multitask verdicts (intersection-union over per-task α_s decisions).
    print()
    Wv = 140
    print("=" * Wv)
    print("Combined multitask verdict per suite (Bonferroni intersection-union over per-task α_s decisions)")
    print("-" * Wv)
    print(
        f"{'Model':<10} {'Dataset':<12} {'Suite':<12} {'K':>3} "
        f"{'α_s':>7} {'alt/K':>7} {'null/K':>7} {'und/K':>7} {'z/K':>5}  Combined"
    )
    print("-" * Wv)
    for result in sorted(results, key=lambda r: (r.spec.dataset, r.spec.model)):
        model_disp = MODEL_DISPLAY.get(result.spec.model, result.spec.model)
        for suite, verdict in result.combined_verdicts.items():
            suite_disp = SUITE_DISPLAY.get(suite, suite)
            print(
                f"{model_disp:<10} {result.spec.dataset:<12} {suite_disp:<12} "
                f"{verdict.family_size:>3} {verdict.alpha_split:>7.4f} "
                f"{verdict.n_alt_alpha_s:>3}/{verdict.family_size:<3} "
                f"{verdict.n_null_alpha_s:>3}/{verdict.family_size:<3} "
                f"{verdict.n_undecided_alpha_s:>3}/{verdict.family_size:<3} "
                f"{verdict.n_z_fallback:>2}/{verdict.family_size:<2}  "
                f"{verdict.combined_decision}"
            )
    print("=" * Wv)
    print(
        "Combined decision: AlternativeOnAll = high-step uniformly better at α=K·α_s; "
        "NullOnAll = low-step uniformly better; Mixed = per-task contradictions; "
        "NotConfirmed = at least one undecided."
    )


def _print_summary(pairs: list[PairResult]) -> None:
    W = 122
    print()
    print("=" * W)
    print(
        f"{'Model':<10} {'Dataset':<12} {'Suite':<16}"
        f" {'1-step':>8} {'max-step':>10} {'Δ pp':>7}"
        f" {'p_z':>8} {'STEP α_g':>9} {'STEP α_s':>9}  CLD  basis"
    )
    print("-" * W)
    for pair in sorted(pairs, key=lambda p: (p.spec.dataset, p.spec.model)):
        for key in ALL_CELLS:
            if key not in pair.cells:
                continue
            cell = pair.cells[key]
            model_disp = MODEL_DISPLAY.get(pair.spec.model, pair.spec.model)
            suite_disp = SUITE_DISPLAY.get(key, key)
            delta = (cell.counts_high.rate - cell.counts_low.rate) * 100

            def _fmtp(v: float | None) -> str:
                if v is None:
                    return "    n/a"
                return "< 0.001" if v < 0.001 else f"{v:7.3f}"

            if cell.primary_test == "step":
                dec_g = _fmt_dec(cell.step_decision_uncorrected)
                dec_s = _fmt_dec(cell.step_decision_corrected)
            else:
                dec_g = dec_s = "    z"
            print(
                f"{model_disp:<10} {pair.spec.dataset:<12} {suite_disp:<16}"
                f" {cell.counts_low.rate * 100:>7.1f}%"
                f" {cell.counts_high.rate * 100:>9.1f}%"
                f" {delta:>+6.1f}pp"
                f" {_fmtp(cell.p_z):>8}"
                f" {dec_g:>9} {dec_s:>9}"
                f"  {cell.cld_low_corrected}/{cell.cld_high_corrected}"
                f"  {cell.primary_test}"
            )
    print("=" * W)
    print("STEP α_g = decision at α_global; STEP α_s = decision at α_split (Bonferroni). basis 'z' = no paired data, z-test used.")


if __name__ == "__main__":
    raise SystemExit(main())
