#!/usr/bin/env python3
"""STEP-based within-model 1-step vs max-step statistics for CALVIN ABC-D.

Usage (from the calvin/ directory):
    python tri_statistics_STEP/run.py
    python tri_statistics_STEP/run.py --output-dir /path/to/outputs --confidence 0.95

Usage (from inside tri_statistics_STEP/):
    python run.py
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent          # tri_statistics_STEP/
_CALVIN_DIR = _PKG_DIR.parent                       # calvin/
_ROOT_DIR = _CALVIN_DIR.parent                      # corl2026_logs/

for _p in (_ROOT_DIR, _CALVIN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tri_statistics_STEP.config import (  # noqa: E402
    CELL_KEYS,
    DATASETS,
    DATASET_DISPLAY,
    MODEL_DISPLAY,
    PAIR_SPECS,
)
from tri_statistics_STEP.corrections import apply_corrections  # noqa: E402
from tri_statistics_STEP.pairs import PairResult, build_pair_result  # noqa: E402
from tri_statistics_STEP.plotting import (  # noqa: E402
    plot_combined_completion_violin,
    plot_combined_sr5_violin,
    plot_completion_histogram,
    plot_completion_violin,
    plot_dataset_overall_comparison,
    plot_sr5_pair_violin,
)
from tri_statistics_STEP.report import write_pair_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate STEP-based within-model 1-step vs max-step statistics "
            "for CALVIN ABC-D (SR@5 binary STEP/z + 0-5 completion Welch)."
        )
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_PKG_DIR / "outputs",
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
        choices=list(DATASETS) + ["all"],
        default="all",
        help="Which dataset(s) to process (default: all).",
    )
    return p.parse_args()


def _fmt_dec(name: str | None) -> str:
    if name is None:
        return "    —"
    return {
        "AcceptAlternative": "high>low",
        "AcceptNull":        "low>high",
        "FailToDecide":      "  —    ",
    }.get(name, name)


def main() -> int:
    args = parse_args()
    out_root: Path = args.output_dir
    alpha = 1.0 - args.confidence
    datasets_to_run = list(DATASETS) if args.dataset == "all" else [args.dataset]

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

        apply_corrections(pair, alpha=alpha)
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

        try:
            sr5_path = fig_dir / f"{pair.spec.model}_sr5_violin.png"
            plot_sr5_pair_violin(pair, sr5_path)
            print(f"  [PLOT]   {sr5_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  sr5 violin {pair.spec.model}: {exc}")
            traceback.print_exc()

        try:
            comp_path = fig_dir / f"{pair.spec.model}_completion_violin.png"
            plot_completion_violin(pair, comp_path)
            print(f"  [PLOT]   {comp_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  completion violin {pair.spec.model}: {exc}")
            traceback.print_exc()

        try:
            hist_path = fig_dir / f"{pair.spec.model}_completion_histogram.png"
            plot_completion_histogram(pair, hist_path)
            print(f"  [PLOT]   {hist_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  completion histogram {pair.spec.model}: {exc}")
            traceback.print_exc()

        report_path = out_root / pair.spec.dataset / f"{pair.spec.model}_report.md"
        try:
            write_pair_report(pair, report_path, confidence=args.confidence)
            print(f"  [REPORT] {report_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  report {pair.spec.model}: {exc}")
            traceback.print_exc()

    print("\nGenerating per-dataset overview plots …")
    for dataset in datasets_to_run:
        ds_pairs = [p for p in pairs if p.spec.dataset == dataset]
        if not ds_pairs:
            continue
        overview_path = out_root / dataset / "figures" / "overview_sr5_bars.png"
        try:
            plot_dataset_overall_comparison(ds_pairs, overview_path, dataset=dataset)
            print(f"  [PLOT]   {overview_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  overview {dataset}: {exc}")
            traceback.print_exc()

    print("\nGenerating appendix combined plots …")
    for dataset in datasets_to_run:
        ds_pairs = [p for p in pairs if p.spec.dataset == dataset]
        if not ds_pairs:
            continue
        appendix_dir = out_root / dataset / "figures" / "appendix"
        sr5_path = appendix_dir / "combined_sr5_violin.png"
        try:
            plot_combined_sr5_violin(ds_pairs, sr5_path, dataset=dataset)
            print(f"  [PLOT]   {sr5_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  combined sr5 violin {dataset}: {exc}")
            traceback.print_exc()
        comp_path = appendix_dir / "combined_completion_violin.png"
        try:
            plot_combined_completion_violin(ds_pairs, comp_path, dataset=dataset)
            print(f"  [PLOT]   {comp_path.relative_to(out_root)}")
        except Exception as exc:
            print(f"  [ERROR]  combined completion violin {dataset}: {exc}")
            traceback.print_exc()

    _print_summary(pairs)
    return 0


def _print_summary(pairs: list[PairResult]) -> None:
    W = 134
    print()
    print("=" * W)
    print(
        f"{'Model':<10} {'Dataset':<14} {'Cell':<6}"
        f" {'1-step':>8} {'N-step':>8} {'Δ pp':>7}"
        f" {'p_z':>8} {'STEP α_g':>9} {'STEP α_s':>9}"
        f" {'Welch Δ':>9} {'Welch p':>9}  CLD  basis"
    )
    print("-" * W)
    for pair in sorted(pairs, key=lambda p: (p.spec.dataset, p.spec.model)):
        for key in CELL_KEYS:
            if key not in pair.cells:
                continue
            cell = pair.cells[key]
            model_disp = MODEL_DISPLAY.get(pair.spec.model, pair.spec.model)
            ds_disp = DATASET_DISPLAY.get(pair.spec.dataset, pair.spec.dataset)
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

            welch_delta = cell.welch.delta if cell.welch else float("nan")
            welch_p = cell.welch.p_value if cell.welch else float("nan")
            print(
                f"{model_disp:<10} {ds_disp:<14} {key:<6}"
                f" {cell.counts_low.rate * 100:>7.1f}%"
                f" {cell.counts_high.rate * 100:>7.1f}%"
                f" {delta:>+6.1f}pp"
                f" {_fmtp(cell.p_z):>8}"
                f" {dec_g:>9} {dec_s:>9}"
                f" {welch_delta:>+9.3f} {_fmtp(welch_p):>9}"
                f"  {cell.cld_low_corrected}/{cell.cld_high_corrected}"
                f"  {cell.primary_test}"
            )
    print("=" * W)
    print(
        "STEP α_g = decision at α_global; STEP α_s = decision at α_split (Bonferroni, "
        "family of 1 here). basis 'z' = no per-rollout data, two-proportion z used."
    )


if __name__ == "__main__":
    raise SystemExit(main())
