#!/usr/bin/env python3
"""Paired equivalence (TOST) statistics for CALVIN ABC-D (SR@5).

The binary success metric is SR@5 (full 5-task chain completion). For each model
we run a paired TOST of Δ = p_reference − p_1step against δ = 3pp. The continuous
0–5 completion score (analysed by Welch's t-test in the STEP pipeline) is not a
proportion and is out of scope here.

Usage (from corl2026_logs/calvin/):
    python tri_statistics_TOST/run.py
    python tri_statistics_TOST/run.py --delta-pp 3
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent          # tri_statistics_TOST/
_CALVIN_DIR = _PKG_DIR.parent                       # calvin/
_ROOT_DIR = _CALVIN_DIR.parent                      # corl2026_logs/
for _p in (_ROOT_DIR, _CALVIN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Import THIS fork's tri_statistics_TOST first, while calvin/ is at the front of
# sys.path. The sibling tri_statistics_STEP package inserts libero/ at the front
# on import, which would otherwise shadow this benchmark's tri_statistics_TOST
# with libero's identically-named package.
from tri_statistics_TOST.config import (  # noqa: E402
    DELTA_MARGIN_CHAIN_SUBTASKS,
    DELTA_MARGIN_PP,
    TOST_ALPHA,
)
from tri_statistics_TOST.equivalence import (  # noqa: E402
    cell_tost,
    chain_length_tost,
    combined_equiv_verdict,
)
from tri_statistics_TOST.plotting import plot_forest  # noqa: E402
from tri_statistics_TOST.report import (  # noqa: E402
    RowData,
    Section,
    fmt_chain,
    fmt_combined_line,
    fmt_sr,
    write_equivalence_report,
)
from tri_statistics_TOST.stats import VERDICT_CODE  # noqa: E402

# Sibling STEP loaders (these add libero/ to sys.path).
from tri_statistics_STEP.config import (  # noqa: E402
    CELL_DISPLAY,
    CELL_KEYS,
    DATASET_DISPLAY,
    DATASETS,
    MODEL_DISPLAY,
    PAIR_SPECS,
)
from tri_statistics_STEP.io_runs import paired_completion_scores  # noqa: E402
from tri_statistics_STEP.pairs import build_pair_result  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=_PKG_DIR / "outputs")
    p.add_argument("--confidence", type=float, default=0.95,
                   help="Confidence for the Wilson per-arm SR CIs (default 0.95).")
    p.add_argument("--alpha", type=float, default=TOST_ALPHA)
    p.add_argument("--delta-pp", type=float, default=DELTA_MARGIN_PP)
    p.add_argument("--chain-delta", type=float, default=DELTA_MARGIN_CHAIN_SUBTASKS,
                   help="Equivalence margin δ for the mean chain length, in subtask "
                        f"units (default {DELTA_MARGIN_CHAIN_SUBTASKS}).")
    p.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_root: Path = args.output_dir
    delta = args.delta_pp / 100.0
    chain_delta = args.chain_delta
    chain_margin_str = f"±{chain_delta:g} subtasks"
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    cell_key = CELL_KEYS[0]  # "sr5"
    cell_disp = CELL_DISPLAY.get(cell_key, cell_key)

    print(f"Equivalence δ = {args.delta_pp:.1f}pp ({cell_disp}) / {chain_margin_str} "
          f"(chain length)  |  TOST α = {args.alpha}  (⇒ {1 - 2 * args.alpha:.0%} CI)")

    for dataset in datasets:
        rows: list[RowData] = []
        forest_rows: list = []
        tosts: list = []
        chain_rows: list[RowData] = []
        chain_forest_rows: list = []
        chain_tosts: list = []
        for spec in PAIR_SPECS:
            if spec.dataset != dataset:
                continue
            try:
                pair = build_pair_result(spec)
            except Exception as exc:
                print(f"  [ERROR] {spec.model} {dataset}: {exc}")
                traceback.print_exc()
                continue
            cell = pair.cells.get(cell_key)
            if cell is None:
                print(f"  [SKIP]  {spec.model} {dataset}: no {cell_key} cell")
                continue
            tost = cell_tost(cell, delta=delta, alpha=args.alpha)
            label = MODEL_DISPLAY.get(spec.model, spec.model)
            rows.append(RowData(label, fmt_sr(cell.counts_low, cell.wilson_low),
                                fmt_sr(cell.counts_high, cell.wilson_high), tost))
            forest_rows.append((label, tost))
            tosts.append(tost)

            # Mean chain-length (0–5) paired-t TOST on the same paired rollouts.
            paired_scores = (
                paired_completion_scores(spec, pair.low_run, pair.high_run)
                if spec.has_paired_data else None
            )
            if paired_scores is None:
                chain_tost = None
                chain_low_cell = chain_high_cell = "— (no paired data)"
            else:
                low_scores, high_scores = paired_scores
                chain_tost = chain_length_tost(low_scores, high_scores,
                                                delta=chain_delta, alpha=args.alpha)
                chain_low_cell = fmt_chain(cell.completion_low)
                chain_high_cell = fmt_chain(cell.completion_high)
            chain_rows.append(RowData(label, chain_low_cell, chain_high_cell, chain_tost))
            chain_forest_rows.append((label, chain_tost))
            chain_tosts.append(chain_tost)

            for note in pair.notes:
                print(f"  [NOTE]  {spec.model} {dataset}: {note}")

        if not rows:
            print(f"  No pairs for {dataset}.")
            continue

        ds_disp = DATASET_DISPLAY.get(dataset, dataset)

        # --- SR@5 (binary, paired-proportion) section -----------------------
        verdict = combined_equiv_verdict("all models", tosts, alpha=args.alpha, delta=delta)
        combined = fmt_combined_line(verdict, low_label="1-step",
                                     high_label="the reference", delta_pp=args.delta_pp)
        intro = (
            f"**Test.** Per-model paired TOST of Δ = reference − 1-step on "
            f"**{cell_disp}** (δ = {args.delta_pp:.0f}pp) and on the **mean chain length** "
            f"(0–5 subtasks; δ = {chain_margin_str}), both at α = {args.alpha:.2f} "
            f"(equivalence ⇔ the {1 - 2 * args.alpha:.0%} CI ⊂ ±δ). SR@5 CI = "
            "correlation-corrected paired MOVER (Newcombe 1998, Method 10); chain-length "
            "CI = paired t-test on the per-rollout score differences. The reference step "
            "count is per model (Flower 4, XR-0 5, X-VLA 10). **Paired data only.**"
        )
        sr_section = Section(title=f"{cell_disp}", combined_line=combined, rows=rows)

        # --- Mean chain length (0–5, paired-t) section ----------------------
        chain_verdict = combined_equiv_verdict("all models", chain_tosts,
                                               alpha=args.alpha, delta=chain_delta)
        chain_combined = fmt_combined_line(chain_verdict, low_label="1-step",
                                           high_label="the reference",
                                           delta_pp=args.delta_pp,
                                           margin_str=chain_margin_str)
        chain_section = Section(
            title="Average chain length (0–5 subtasks)",
            combined_line=chain_combined,
            rows=chain_rows,
            value_scale=1.0,
            value_decimals=2,
            low_hdr="1-step mean (95% CI)",
            high_hdr="reference mean (95% CI)",
            delta_hdr="Δ subtasks",
            ci_hdr="90% CI (subtasks)",
            margin_display=f"±{chain_delta:g}",
        )

        report_path = out_root / dataset / "equivalence_report.md"
        write_equivalence_report(
            report_path,
            title=f"{ds_disp}: equivalence of 1-step vs reference ({cell_disp} & chain length)",
            intro=intro, sections=[sr_section, chain_section],
            item_hdr="Model", low_hdr="1-step (95% CI)", high_hdr="reference (95% CI)",
            delta_pp=args.delta_pp,
        )
        print(f"  [REPORT] {report_path}")

        forest_path = out_root / dataset / "figures" / "equivalence_forest.png"
        plot_forest(forest_rows, forest_path,
                    title=f"{ds_disp} — {cell_disp} equivalence (δ=±{args.delta_pp:.0f}pp, 90% CI)",
                    delta_pp=args.delta_pp,
                    band_label=f"±{args.delta_pp:.0f}pp equivalence band")
        print(f"  [PLOT]   {forest_path}")

        chain_forest_path = out_root / dataset / "figures" / "equivalence_forest_chain_length.png"
        plot_forest(chain_forest_rows, chain_forest_path,
                    title=f"{ds_disp} — mean chain length equivalence ({chain_margin_str}, 90% CI)",
                    delta_pp=chain_delta, value_scale=1.0,
                    x_label="Δ = reference − 1-step (subtasks)",
                    band_label=f"{chain_margin_str} equivalence band")
        print(f"  [PLOT]   {chain_forest_path}")

        _print_console_block(cell_disp, "pp", 100.0, 1, rows, verdict)
        _print_console_block("chain length", "subtasks", 1.0, 2, chain_rows, chain_verdict)
    return 0


def _print_console_block(metric, unit, scale, decimals, rows, verdict) -> None:
    print(f"\n  [{metric}]  {'Model':<10} {'Δ ' + unit:>9} "
          f"{'90% CI (' + unit + ')':>22}  verdict")
    for r in rows:
        if r.tost is None:
            print(f"  {r.label:<10} {'  —  ':>9} {'—':>22}  not assessed")
        else:
            t = r.tost
            ci = f"[{t.ci_low*scale:+.{decimals}f}, {t.ci_high*scale:+.{decimals}f}]"
            print(f"  {r.label:<10} {t.delta_hat*scale:>+8.{decimals}f} "
                  + ci.rjust(22) + f"  {VERDICT_CODE[t.verdict]} {t.verdict}")
    print(f"  Combined ({verdict.combined_decision}): "
          f"{verdict.n_equivalent} equiv / {verdict.n_relevant} diff / "
          f"{verdict.n_inconclusive} incon of {verdict.family_size}")


if __name__ == "__main__":
    raise SystemExit(main())
