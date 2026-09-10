#!/usr/bin/env python3
"""Glyph-only variant of make_libero_latex_table.

For each 1-step delta cell, render only a verdict glyph after the success rate:
- approx (gray): no significant separation
- blacktriangle (OliveGreen): 1-step significantly better
- blacktriangledown (OrangeRed): 1-step significantly worse

No inline (pp) delta numbers. Significance comes from the same Bonferroni-corrected
STEP reports used by the dagger-marked table; this script does not modify the
underlying analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analyze_libero_steps import (
    DISPLAY_NAMES,
    MODEL_ORDER,
    SUITES,
    Counts,
    RunResult,
    RUN_SPECS,
    load_run,
)
from table_helpers import (
    SUITE_LABELS,
    format_counts,
    format_percent,
    latex_escape,
    latex_model_name,
    load_step_significance,
    strip_trailing_period,
)


DEFAULT_CAPTION = "LIBERO success rates for 1-step and max-step variants."
DEFAULT_LABEL = "tab:libero_results_glyph"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a glyph-only LaTeX table for LIBERO 1-step and max-step results."
    )
    parser.add_argument(
        "--libero-dir",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / 'libero'),
        help="Directory containing LIBERO experiment logs.",
    )
    parser.add_argument("--dataset", choices=["libero", "liberoplus"], default="libero")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--precision", type=int, default=1)
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--cell-format",
        choices=["counts", "percent"],
        default="counts",
    )
    parser.add_argument("--also-percent-only", action="store_true")
    return parser.parse_args()


def _format_base(counts: Counts, precision: int, cell_format: str, *, bold: bool) -> str:
    base = (
        format_percent(counts, precision)
        if cell_format == "percent"
        else format_counts(counts, precision)
    )
    return rf"\textbf{{{base}}}" if bold else base


def format_glyph_cell(
    counts: Counts,
    reference: Counts,
    *,
    precision: int,
    cell_format: str,
    verdict: str,
    bold: bool = False,
) -> str:
    """verdict: 'up', 'down', 'mixed', or 'none'."""
    base = _format_base(counts, precision, cell_format, bold=bold)
    if verdict == "up":
        return f"{base} {{\\scriptsize $\\textcolor{{OliveGreen}}{{\\blacktriangle}}$}}"
    elif verdict == "down":
        return f"{base} {{\\scriptsize $\\textcolor{{OrangeRed}}{{\\blacktriangledown}}$}}"
    elif verdict == "mixed":
        return f"{base} {{\\scriptsize $\\textcolor{{OliveGreen}}{{\\blacktriangle}}\\textcolor{{OrangeRed}}{{\\blacktriangledown}}$}}"
    return base


def _get_verdict(
    markers: dict[tuple[str, str], set[int]],
    model_key: str,
    cell_key: str,
    delta_pp: float,
) -> str:
    """Return 'up', 'down', 'mixed', or 'none'."""
    cell_markers = markers.get((model_key, cell_key), set())
    if 1 in cell_markers and -1 in cell_markers:
        return "mixed"
    rounded_sign = 1 if delta_pp > 0.0 else -1 if delta_pp < 0.0 else 0
    if rounded_sign == 0:
        return "none"
    if rounded_sign in cell_markers:
        return "up" if rounded_sign == 1 else "down"
    return "none"


def build_latex_glyph_table(
    results: list[RunResult],
    precision: int,
    caption: str,
    label: str,
    cell_format: str,
    significance_markers: dict[tuple[str, str], set[int]],
    caption_suffix: str,
) -> str:
    by_model: dict[str, list[RunResult]] = {}
    for run in results:
        by_model.setdefault(run.spec.model_key, []).append(run)

    lines = [
        r"\begin{table}[htp]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}{caption_suffix}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lllllll}",
        r"\midrule",
        "Model & Steps & "
        + " & ".join(SUITE_LABELS[suite] for suite in SUITES)
        + r" & Average \\",
        r"\midrule",
    ]

    for model_index, model_key in enumerate(MODEL_ORDER):
        model_runs = sorted(by_model.get(model_key, []), key=lambda run: run.spec.steps)
        if not model_runs:
            continue

        model_name = latex_model_name(model_key)
        multirow = rf"\multirow{{{len(model_runs)}}}{{*}}{{{model_name}}}"
        reference_run = model_runs[-1]
        display_runs = [reference_run] + model_runs[:-1]

        # Per-suite + Average: bold the higher rate between 1-step and max-step.
        def _bold_low(counts_low: Counts, counts_high: Counts) -> tuple[bool, bool]:
            if counts_low.rate > counts_high.rate:
                return True, False
            if counts_high.rate > counts_low.rate:
                return False, True
            return False, False

        low_run = model_runs[0] if model_runs[0].spec.steps == 1 else None

        for run_index, run in enumerate(display_runs):
            model_cell = multirow if run_index == 0 else ""
            if run.spec.steps == 1 and reference_run.spec.steps != 1:
                suite_cells = []
                for suite in SUITES:
                    delta_pp = (
                        run.suite_counts[suite].rate
                        - reference_run.suite_counts[suite].rate
                    ) * 100.0
                    bold_low, _ = _bold_low(
                        run.suite_counts[suite], reference_run.suite_counts[suite]
                    )
                    suite_cells.append(
                        format_glyph_cell(
                            run.suite_counts[suite],
                            reference_run.suite_counts[suite],
                            precision=precision,
                            cell_format=cell_format,
                            verdict=_get_verdict(
                                significance_markers, model_key, suite, delta_pp
                            ),
                            bold=bold_low,
                        )
                    )
                bold_low_avg, _ = _bold_low(run.overall, reference_run.overall)
                avg_cell = _format_base(run.overall, precision, cell_format, bold=bold_low_avg)
                cells = [model_cell, str(run.spec.steps), *suite_cells, avg_cell]
            else:
                bold_per_suite = {}
                if low_run is not None:
                    for suite in SUITES:
                        _, bold_high = _bold_low(
                            low_run.suite_counts[suite], run.suite_counts[suite]
                        )
                        bold_per_suite[suite] = bold_high
                    _, bold_high_avg = _bold_low(low_run.overall, run.overall)
                else:
                    bold_high_avg = False
                cells = [
                    model_cell,
                    str(run.spec.steps),
                    *(
                        _format_base(
                            run.suite_counts[suite], precision, cell_format,
                            bold=bold_per_suite.get(suite, False),
                        )
                        for suite in SUITES
                    ),
                    _format_base(run.overall, precision, cell_format, bold=bold_high_avg),
                ]
            lines.append(" & ".join(cells) + r" \\")

        if model_index != len(MODEL_ORDER) - 1:
            lines.append("")
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


CAPTION_SUFFIX = (
    r" Glyphs encode the 1-step vs max-step paired STEP comparison, "
    r"controlling the family-wise error at $\alpha = 0.05$ via Bonferroni "
    r"(per-test $\alpha_s = \alpha / K$; $K$ = number of base tasks per suite "
    r"for suite columns, $K = 5$ for the Average column): "
    r"$\textcolor{OliveGreen}{\blacktriangle}$ 1-step significantly better, "
    r"$\textcolor{OrangeRed}{\blacktriangledown}$ 1-step significantly worse, "
    r"$\textcolor{OliveGreen}{\blacktriangle}\textcolor{OrangeRed}{\blacktriangledown}$"
    r" mixed task-level results."
)


def main() -> None:
    args = parse_args()
    label = "LIBERO" if args.dataset == "libero" else "LIBERO+"
    args.caption = args.caption or f"{label} success rates for 1-step and max-step variants."
    args.label = args.label or f"tab:{args.dataset}_results_glyph"
    results = [load_run(args.libero_dir, spec) for spec in RUN_SPECS
               if spec.dataset == args.dataset]
    significance_markers = load_step_significance(args.libero_dir, args.dataset)
    table = build_latex_glyph_table(
        results=results,
        precision=args.precision,
        caption=args.caption,
        label=args.label,
        cell_format=args.cell_format,
        significance_markers=significance_markers,
        caption_suffix=CAPTION_SUFFIX,
    )
    if args.also_percent_only and args.cell_format != "percent":
        table += "\n" + build_latex_glyph_table(
            results=results,
            precision=args.precision,
            caption=strip_trailing_period(args.caption) + " (percentages only).",
            label=args.label + "_percent_only",
            cell_format="percent",
            significance_markers=significance_markers,
            caption_suffix=CAPTION_SUFFIX,
        )

    if args.output is None:
        print(table, end="")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
