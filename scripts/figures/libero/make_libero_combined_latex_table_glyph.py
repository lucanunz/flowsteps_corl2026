#!/usr/bin/env python3
"""Compact combined LIBERO/LIBERO+ glyph table.

The default layout shows each model as a max-step row followed by a 1-step row.
The paired layout shows one row per model with max-step / 1-step cells. Glyphs
keep the same meaning as the existing glyph tables: significant STEP separation
of 1-step vs max-step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analyze_libero_steps import (
    DISPLAY_NAMES,
    MODEL_ORDER,
    RUN_SPECS,
    Counts,
    RunResult,
    load_run,
)
from table_helpers import (
    format_counts,
    format_percent,
    latex_escape,
    latex_model_name,
    load_step_significance,
    strip_trailing_period,
)


SGO_SUITES = ("libero_spatial", "libero_goal", "libero_object")
DATASETS = ("libero", "liberoplus")
DATASET_LABELS = {
    "libero": "LIBERO",
    "liberoplus": "LIBERO+",
}
# Header for each dataset's LIBERO-10 column group. Spelled out rather than
# derived, because the base suite reads "LIBERO-10" while the LIBERO+ one takes
# a space to avoid the "LIBERO+-10" double hyphen.
TEN_SUITE_LABELS = {
    "libero": "LIBERO-10",
    "liberoplus": "LIBERO+ 10",
}
DEFAULT_CAPTION = (
    "LIBERO and LIBERO+ success rates. SGO is an average over Spatial, Goal, "
    "and Object suites."
)
DEFAULT_LABEL = "tab:libero_liberoplus_compact_glyph"
# Emitted after the caption text but commented out, so the legend travels with
# the table without printing; uncomment the line in the paper to show it.
CAPTION_SUFFIX = (
    " %Glyphs encode the 1-step vs max-step paired STEP comparison, "
    r"controlling the family-wise error at $\alpha = 0.05$ via Bonferroni: "
    r"$\textcolor{OliveGreen}{\blacktriangle}$ and "
    r"$\textcolor{OrangeRed}{\blacktriangledown}$ respectively highlight when "
    "there is at least one task where 1-step is significantly better and worse."
    "\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a compact combined LIBERO/LIBERO+ glyph LaTeX table."
    )
    parser.add_argument(
        "--libero-dir",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / 'libero'),
        help="Directory containing LIBERO experiment logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the LaTeX table. Defaults to stdout.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=1,
        help="Number of decimal places for percentages.",
    )
    parser.add_argument(
        "--caption",
        default=DEFAULT_CAPTION,
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help="LaTeX table label.",
    )
    parser.add_argument(
        "--cell-format",
        choices=["counts", "percent"],
        default="counts",
        help="Whether cells include successes/trials counts or percentages only.",
    )
    parser.add_argument(
        "--also-percent-only",
        action="store_true",
        help="Append a second version of the table with percentages only.",
    )
    parser.add_argument(
        "--bold-best",
        action="store_true",
        help="Bold the higher rate between 1-step and max-step within each metric.",
    )
    parser.add_argument(
        "--layout",
        choices=["multirow", "pair"],
        default="multirow",
        help=(
            "Table layout: multirow uses separate max-step and 1-step rows; "
            "pair uses one row per model with max-step / 1-step cells."
        ),
    )
    return parser.parse_args()


def load_combined_results(libero_dir: Path) -> list[RunResult]:
    specs = [spec for spec in RUN_SPECS if spec.dataset in DATASETS]
    return [load_run(libero_dir, spec) for spec in specs]


def pooled_counts(run: RunResult, suites: tuple[str, ...]) -> Counts:
    return Counts(
        successes=sum(run.suite_counts[suite].successes for suite in suites),
        trials=sum(run.suite_counts[suite].trials for suite in suites),
    )


def _format_base(
    counts: Counts,
    precision: int,
    cell_format: str,
    *,
    bold: bool = False,
) -> str:
    base = (
        format_percent(counts, precision)
        if cell_format == "percent"
        else format_counts(counts, precision)
    )
    return rf"\textbf{{{base}}}" if bold else base


def _format_glyph(verdict: str) -> str:
    if verdict == "up":
        return r" {\scriptsize $\textcolor{OliveGreen}{\blacktriangle}$}"
    if verdict == "down":
        return r" {\scriptsize $\textcolor{OrangeRed}{\blacktriangledown}$}"
    if verdict == "mixed":
        return (
            r" {\scriptsize $\textcolor{OliveGreen}{\blacktriangle}"
            r"\textcolor{OrangeRed}{\blacktriangledown}$}"
        )
    return ""


def _verdict_from_directions(directions: set[int]) -> str:
    has_up = 1 in directions
    has_down = -1 in directions
    if has_up and has_down:
        return "mixed"
    if has_up:
        return "up"
    if has_down:
        return "down"
    return "none"


def _suite_verdict(
    markers: dict[tuple[str, str], set[int]],
    model_key: str,
    suite: str,
    low: Counts,
    high: Counts,
) -> str:
    delta_pp = (low.rate - high.rate) * 100.0
    rounded_sign = 1 if delta_pp > 0.0 else -1 if delta_pp < 0.0 else 0
    if rounded_sign == 0:
        return "none"
    directions = markers.get((model_key, suite), set())
    if rounded_sign in directions:
        return "up" if rounded_sign == 1 else "down"
    return "none"


def _sgo_verdict(
    markers: dict[tuple[str, str], set[int]],
    model_key: str,
) -> str:
    directions: set[int] = set()
    for suite in SGO_SUITES:
        directions.update(markers.get((model_key, suite), set()))
    return _verdict_from_directions(directions)


def format_metric_cell(
    counts: Counts,
    *,
    precision: int,
    cell_format: str,
    verdict: str = "none",
    bold: bool = False,
) -> str:
    return f"{_format_base(counts, precision, cell_format, bold=bold)}{_format_glyph(verdict)}"


def format_pair_cells(
    low: Counts,
    high: Counts,
    *,
    precision: int,
    cell_format: str,
    verdict: str,
    bold_best: bool,
) -> tuple[str, str]:
    """Return the (Reference, 1 Step) cells; the glyph rides on the 1-step cell."""
    low_bold, high_bold = _best_bold_flags(low, high)
    low_text = _format_base(
        low,
        precision,
        cell_format,
        bold=bold_best and low_bold,
    )
    high_text = _format_base(
        high,
        precision,
        cell_format,
        bold=bold_best and high_bold,
    )
    return high_text, f"{low_text}{_format_glyph(verdict)}"


def _best_bold_flags(low: Counts, high: Counts) -> tuple[bool, bool]:
    if low.rate > high.rate:
        return True, False
    if high.rate > low.rate:
        return False, True
    return False, False


def _runs_by_dataset_model(
    results: list[RunResult],
) -> dict[tuple[str, str], tuple[RunResult, RunResult]]:
    grouped: dict[tuple[str, str], list[RunResult]] = {}
    for run in results:
        grouped.setdefault((run.spec.dataset, run.spec.model_key), []).append(run)

    paired: dict[tuple[str, str], tuple[RunResult, RunResult]] = {}
    for key, runs in grouped.items():
        ordered = sorted(runs, key=lambda run: run.spec.steps)
        low_runs = [run for run in ordered if run.spec.steps == 1]
        if not low_runs or len(ordered) < 2:
            continue
        paired[key] = (low_runs[0], ordered[-1])
    return paired


def build_latex_combined_multirow_table(
    results: list[RunResult],
    precision: int,
    caption: str,
    label: str,
    cell_format: str,
    significance_by_dataset: dict[str, dict[tuple[str, str], set[int]]],
    caption_suffix: str,
    bold_best: bool = False,
) -> str:
    paired = _runs_by_dataset_model(results)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}{caption_suffix}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"Model & Steps & LIBERO S/G/O Avg & LIBERO-10 & LIBERO+ S/G/O Avg & LIBERO+-10 \\",
        r"\midrule",
    ]

    model_rows: list[str] = []
    for model_index, model_key in enumerate(MODEL_ORDER):
        high_cells: list[str] = []
        low_cells: list[str] = []
        high_steps: set[int] = set()
        for dataset in DATASETS:
            run_pair = paired.get((dataset, model_key))
            markers = significance_by_dataset[dataset]
            if run_pair is None:
                high_cells.extend(["--", "--"])
                low_cells.extend(["--", "--"])
                continue

            low_run, high_run = run_pair
            high_steps.add(high_run.spec.steps)
            low_sgo = pooled_counts(low_run, SGO_SUITES)
            high_sgo = pooled_counts(high_run, SGO_SUITES)
            low_sgo_bold, high_sgo_bold = _best_bold_flags(low_sgo, high_sgo)
            low_10_bold, high_10_bold = _best_bold_flags(
                low_run.suite_counts["libero_10"],
                high_run.suite_counts["libero_10"],
            )
            high_cells.extend(
                [
                    format_metric_cell(
                        high_sgo,
                        precision=precision,
                        cell_format=cell_format,
                        bold=bold_best and high_sgo_bold,
                    ),
                    format_metric_cell(
                        high_run.suite_counts["libero_10"],
                        precision=precision,
                        cell_format=cell_format,
                        bold=bold_best and high_10_bold,
                    ),
                ]
            )
            low_cells.extend(
                [
                    format_metric_cell(
                        low_sgo,
                        precision=precision,
                        cell_format=cell_format,
                        verdict=_sgo_verdict(markers, model_key),
                        bold=bold_best and low_sgo_bold,
                    ),
                    format_metric_cell(
                        low_run.suite_counts["libero_10"],
                        precision=precision,
                        cell_format=cell_format,
                        verdict=_suite_verdict(
                            markers,
                            model_key,
                            "libero_10",
                            low_run.suite_counts["libero_10"],
                            high_run.suite_counts["libero_10"],
                        ),
                        bold=bold_best and low_10_bold,
                    ),
                ]
            )

        if not high_steps:
            continue

        high_step_label = (
            str(next(iter(high_steps)))
            if len(high_steps) == 1
            else "/".join(str(step) for step in sorted(high_steps))
        )
        model_name = latex_escape(DISPLAY_NAMES[model_key])
        model_rows.append(
            " & ".join(
                [
                    rf"\multirow{{2}}{{*}}{{{model_name}}}",
                    high_step_label,
                    *high_cells,
                ]
            )
            + r" \\"
        )
        model_rows.append(
            " & ".join(
                [
                    "",
                    "1",
                    *low_cells,
                ]
            )
            + r" \\"
        )
        if model_index != len(MODEL_ORDER) - 1:
            model_rows.append(r"\midrule")

    lines.extend(model_rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_latex_combined_pair_table(
    results: list[RunResult],
    precision: int,
    caption: str,
    label: str,
    cell_format: str,
    significance_by_dataset: dict[str, dict[tuple[str, str], set[int]]],
    caption_suffix: str,
    bold_best: bool = False,
) -> str:
    paired = _runs_by_dataset_model(results)

    # One column group per (dataset, metric), each split into Reference / 1 Step.
    groups = [
        label
        for dataset in DATASETS
        for label in (f"{DATASET_LABELS[dataset]} SGO", TEN_SUITE_LABELS[dataset])
    ]
    header_groups = "\n".join(
        rf"& \multicolumn{{2}}{{c}}{{{group}}}" for group in groups
    )
    cmidrules = "\n".join(
        rf"\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}" for i in range(len(groups))
    )
    subheader = " ".join(r"& Reference & 1 Step" for _ in groups)

    lines = [
        r"\begin{table}[htp]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}{caption_suffix}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{l{'c' * 2 * len(groups)}}}",
        r"\midrule",
        r"\multirow{2}{*}{Model} ",
        header_groups + r" \\",
        cmidrules,
        subheader + r" \\",
        r"\midrule",
    ]

    model_rows: list[str] = []
    for model_key in MODEL_ORDER:
        cells: list[str] = []
        has_any_dataset = False
        for dataset in DATASETS:
            run_pair = paired.get((dataset, model_key))
            markers = significance_by_dataset[dataset]
            if run_pair is None:
                cells.extend(["--", "--", "--", "--"])
                continue

            has_any_dataset = True
            low_run, high_run = run_pair
            cells.extend(
                format_pair_cells(
                    pooled_counts(low_run, SGO_SUITES),
                    pooled_counts(high_run, SGO_SUITES),
                    precision=precision,
                    cell_format=cell_format,
                    verdict=_sgo_verdict(markers, model_key),
                    bold_best=bold_best,
                )
            )
            cells.extend(
                format_pair_cells(
                    low_run.suite_counts["libero_10"],
                    high_run.suite_counts["libero_10"],
                    precision=precision,
                    cell_format=cell_format,
                    verdict=_suite_verdict(
                        markers,
                        model_key,
                        "libero_10",
                        low_run.suite_counts["libero_10"],
                        high_run.suite_counts["libero_10"],
                    ),
                    bold_best=bold_best,
                )
            )

        if not has_any_dataset:
            continue
        # Model name on its own line, then one line per dataset/metric group.
        row = [f"{latex_model_name(model_key)} "]
        for i in range(0, len(cells), 2):
            row.append(f"& {cells[i]} & {cells[i + 1]}")
        model_rows.append("\n".join(row) + r" \\")

    lines.append("\n\n".join(model_rows))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def make_table(args: argparse.Namespace, *, cell_format: str, caption: str, label: str) -> str:
    results = load_combined_results(args.libero_dir)
    significance_by_dataset = {
        dataset: load_step_significance(args.libero_dir, dataset)
        for dataset in DATASETS
    }
    builder = (
        build_latex_combined_pair_table
        if args.layout == "pair"
        else build_latex_combined_multirow_table
    )
    return builder(
        results=results,
        precision=args.precision,
        caption=caption,
        label=label,
        cell_format=cell_format,
        significance_by_dataset=significance_by_dataset,
        caption_suffix=CAPTION_SUFFIX,
        bold_best=args.bold_best,
    )


def main() -> None:
    args = parse_args()
    table = make_table(
        args,
        cell_format=args.cell_format,
        caption=args.caption,
        label=args.label,
    )
    if args.also_percent_only and args.cell_format != "percent":
        table += "\n" + make_table(
            args,
            cell_format="percent",
            caption=strip_trailing_period(args.caption) + " (percentages only).",
            label=args.label + "_percent_only",
        )

    if args.output is None:
        print(table, end="")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
