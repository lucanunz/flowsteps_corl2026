#!/usr/bin/env python3
"""Paired equivalence (TOST) statistics for the RoboTwin paired runs.

Per-pair mode (default): the pooled Overall success rate per setting (clean,
randomized). Per-task mode (--per-task): the 50 tasks per setting, with a
combined intersection–union "equivalent on all tasks" verdict (no Bonferroni).

Usage (from corl2026_logs/robotwin/):
    python tri_statistics_TOST/run.py
    python tri_statistics_TOST/run.py --per-task
    python tri_statistics_TOST/run.py --setting clean
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_ROBOTWIN_DIR = _PKG_DIR.parent
_REPO_ROOT = _ROBOTWIN_DIR.parent
for _p in (_REPO_ROOT, _ROBOTWIN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Import THIS fork's tri_statistics_TOST first (see CALVIN run.py for why).
from tri_statistics_TOST.config import DELTA_MARGIN_PP, TOST_ALPHA  # noqa: E402
from tri_statistics_TOST.equivalence import (  # noqa: E402
    cell_tost,
    combined_equiv_verdict,
)
from tri_statistics_TOST.plotting import plot_forest  # noqa: E402
from tri_statistics_TOST.report import (  # noqa: E402
    RowData,
    Section,
    fmt_combined_line,
    fmt_sr,
    write_equivalence_report,
)
from tri_statistics_TOST.stats import VERDICT_CODE  # noqa: E402

# Sibling STEP loaders.
from tri_statistics_STEP.config import (  # noqa: E402
    MODEL_DISPLAY,
    SETTING_DISPLAY,
    select_specs,
)
from tri_statistics_STEP.pairs import (  # noqa: E402
    build_pair_result,
    build_per_task_pair_result,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=_PKG_DIR / "outputs")
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--alpha", type=float, default=TOST_ALPHA)
    ap.add_argument("--delta-pp", type=float, default=DELTA_MARGIN_PP)
    ap.add_argument("--setting", default="all", choices=("all", *sorted(SETTING_DISPLAY)))
    ap.add_argument("--model", default="all", choices=("all", *sorted(MODEL_DISPLAY)))
    ap.add_argument("--per-task", action="store_true")
    return ap.parse_args()


def _intro(delta_pp: float, alpha: float, per_task: bool) -> str:
    base = (
        f"**Test.** Paired TOST of Δ = p_reference − p_1step against the "
        f"pre-registered margin **δ = {delta_pp:.0f}pp** at α = {alpha:.2f} "
        f"(equivalence ⇔ the {1 - 2 * alpha:.0%} CI ⊂ ±{delta_pp:.0f}pp). CI = "
        "correlation-corrected paired MOVER (Newcombe 1998, Method 10). "
        "**Paired data only.**"
    )
    if per_task:
        base += (
            " The combined 'equivalent on all tasks' claim is an intersection–union "
            "test, so each per-task TOST runs at the full α with **no Bonferroni**. "
            f"Per-task n is small, so most tasks are inconclusive at ±{delta_pp:.0f}pp."
        )
    return base


def main() -> int:
    args = parse_args()
    delta = args.delta_pp / 100.0
    out_root: Path = args.output_dir
    specs = select_specs(model=args.model, setting=args.setting)
    if not specs:
        print(f"No PairSpec matches --model {args.model!r} --setting {args.setting!r}")
        return 1
    print(f"Equivalence δ = {args.delta_pp:.1f}pp | TOST α = {args.alpha} "
          f"(⇒ {1 - 2 * args.alpha:.0%} CI) | mode: {'per-task' if args.per_task else 'per-pair'}")

    if args.per_task:
        return _run_per_task(args, specs, out_root, delta)

    # Per-pair: one Overall cell per setting → a single report with one row each.
    rows: list[RowData] = []
    forest_rows: list = []
    tosts: list = []
    for spec in specs:
        try:
            pair = build_pair_result(spec)
        except Exception as exc:
            print(f"  [ERROR] {spec.model} {spec.setting}: {exc}")
            traceback.print_exc()
            continue
        cell = next(iter(pair.cells.values()))
        tost = cell_tost(cell, delta=delta, alpha=args.alpha)
        label = SETTING_DISPLAY.get(spec.setting, spec.setting)
        rows.append(RowData(label, fmt_sr(cell.counts_low, cell.wilson_low),
                            fmt_sr(cell.counts_high, cell.wilson_high), tost))
        forest_rows.append((label, tost))
        tosts.append(tost)
        for note in pair.notes:
            print(f"  [NOTE]  {spec.model} {spec.setting}: {note}")

    if not rows:
        print("No pairs loaded.")
        return 1

    verdict = combined_equiv_verdict("all settings", tosts, alpha=args.alpha, delta=delta)
    combined = fmt_combined_line(verdict, low_label="1-step", high_label="10-step",
                                 delta_pp=args.delta_pp)
    report_path = out_root / "equivalence_report.md"
    write_equivalence_report(
        report_path,
        title="RoboTwin: equivalence of 1-step vs 10-step (Overall success rate)",
        intro=_intro(args.delta_pp, args.alpha, per_task=False),
        sections=[Section(title="Overall success rate", combined_line=combined, rows=rows)],
        item_hdr="Setting", low_hdr="1-step (95% CI)", high_hdr="10-step (95% CI)",
        delta_pp=args.delta_pp,
    )
    print(f"  [REPORT] {report_path}")
    forest_path = out_root / "figures" / "equivalence_forest.png"
    plot_forest(forest_rows, forest_path,
                title=f"RoboTwin — Overall equivalence (δ=±{args.delta_pp:.0f}pp, 90% CI)",
                delta_pp=args.delta_pp)
    print(f"  [PLOT]   {forest_path}")
    _print_rows(rows)
    return 0


def _run_per_task(args, specs, out_root: Path, delta: float) -> int:
    for spec in specs:
        try:
            pt = build_per_task_pair_result(spec)
        except Exception as exc:
            print(f"  [ERROR] {spec.model} {spec.setting}: {exc}")
            traceback.print_exc()
            continue
        rows: list[RowData] = []
        forest_rows: list = []
        tosts: list = []
        for cell in pt.tasks:
            tost = cell_tost(cell, delta=delta, alpha=args.alpha)
            label = cell.suite
            rows.append(RowData(label, fmt_sr(cell.counts_low, cell.wilson_low),
                                fmt_sr(cell.counts_high, cell.wilson_high), tost))
            forest_rows.append((label if len(label) <= 34 else label[:31] + "…", tost))
            tosts.append(tost)
        for note in pt.notes:
            print(f"  [NOTE]  {spec.model} {spec.setting}: {note}")
        if not rows:
            print(f"  No tasks for {spec.setting}.")
            continue

        setting_disp = SETTING_DISPLAY.get(spec.setting, spec.setting)
        verdict = combined_equiv_verdict(setting_disp, tosts, alpha=args.alpha, delta=delta)
        combined = fmt_combined_line(verdict, low_label="1-step", high_label="10-step",
                                     delta_pp=args.delta_pp)
        report_path = out_root / spec.setting / "equivalence_per_task_report.md"
        write_equivalence_report(
            report_path,
            title=f"RoboTwin {setting_disp}: per-task equivalence (1-step vs 10-step)",
            intro=_intro(args.delta_pp, args.alpha, per_task=True),
            sections=[Section(title=setting_disp, combined_line=combined, rows=rows)],
            item_hdr="Task", low_hdr="1-step (95% CI)", high_hdr="10-step (95% CI)",
            delta_pp=args.delta_pp,
        )
        print(f"  [REPORT] {report_path}")
        forest_path = out_root / spec.setting / "figures" / "equivalence_task_forest.png"
        plot_forest(forest_rows, forest_path,
                    title=f"RoboTwin {setting_disp} — per-task equivalence "
                          f"(δ=±{args.delta_pp:.0f}pp, 90% CI)",
                    delta_pp=args.delta_pp)
        print(f"  [PLOT]   {forest_path}")
        print(f"  {setting_disp}: combined ({verdict.combined_decision}) — "
              f"{verdict.n_equivalent} equiv / {verdict.n_relevant} diff / "
              f"{verdict.n_inconclusive} incon / {verdict.n_unassessed} n/a of {verdict.family_size}")
    return 0


def _print_rows(rows: list) -> None:
    print(f"\n  {'Setting':<22} {'Δ pp':>7} {'90% CI (pp)':>20}  verdict")
    for r in rows:
        if r.tost is None:
            print(f"  {r.label:<22} {'  —  ':>7} {'—':>20}  not assessed")
        else:
            t = r.tost
            print(f"  {r.label:<22} {t.delta_hat*100:>+6.1f} "
                  + f"[{t.ci_low*100:+.1f}, {t.ci_high*100:+.1f}]".rjust(20)
                  + f"  {VERDICT_CODE[t.verdict]} {t.verdict}")


if __name__ == "__main__":
    raise SystemExit(main())
