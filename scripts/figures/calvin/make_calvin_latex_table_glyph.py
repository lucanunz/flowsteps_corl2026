#!/usr/bin/env python3
"""Glyph variant of make_calvin_latex_table with layout and bold-best options.

For each 1-step vs max-step comparison, render a verdict glyph after the
success rate (or avg-length value):
- approx (gray): no significant separation
- blacktriangle (OliveGreen): 1-step significantly better
- blacktriangledown (OrangeRed): 1-step significantly worse

CALVIN's pipeline only tests SR@5 (paired STEP) and Avg. Len. (Welch t-test);
SR@1-4 cells get no glyph since they were not formally tested.

Two layouts are supported (mirroring make_libero_combined_latex_table_glyph.py):
- multirow (default): one max-step row and one 1-step row per model. The
  glyph sits next to the 1-step value.
- pair: one row per model, each metric cell is "max-step / 1-step" with the
  glyph after the pair.

Optionally `--bold-best` bolds the higher value within each (1-step, max-step)
comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from table_helpers import (  # noqa: E402
    CHAIN_LEVELS,
    RunResult,
    _load_calvin_significance,
    format_number,
    format_percent,
    latex_escape,
    load_calvin_results,
)
from tri_statistics_STEP.config import MODEL_DISPLAY, MODEL_HIGH_STEPS  # noqa: E402


DEFAULT_CAPTION = "CALVIN ABC-D success rates for 1-step and max-step variants."
DEFAULT_LABEL = "tab:calvin_results_glyph"

CAPTION_SUFFIX = (
    r" Glyphs on SR@5 and Avg. Len. encode the 1-step vs max-step comparison: "
    r"$\approx$ no separation, $\textcolor{OliveGreen}{\blacktriangle}$ 1-step "
    r"significantly better, $\textcolor{OrangeRed}{\blacktriangledown}$ 1-step "
    r"significantly worse. SR@5 uses paired STEP after Bonferroni alpha-splitting "
    r"with a single CALVIN binary family ($m=1$, $\alpha_{\mathrm{split}}=0.05$); "
    r"Avg. Len. uses a Welch t-test at $p < 0.05$. SR@1-4 cells carry no glyph "
    r"since they were not formally tested."
)
PAIR_LAYOUT_NOTE = r" Pair-layout cells are formatted as ``max-step / 1-step''."

# The grouped layout drops the gray $\approx$ (a glyph appears only on a
# significant separation) and splits each metric into Reference / 1 Step columns.
CAPTION_SUFFIX_GROUPED = (
    r" Each metric is split into the max-step (Reference) and 1-step columns, with "
    r"the higher value in bold. Glyphs on SR@5 and Avg. Len. mark a significant "
    r"1-step vs max-step difference: $\textcolor{OliveGreen}{\blacktriangle}$ 1-step "
    r"better, $\textcolor{OrangeRed}{\blacktriangledown}$ 1-step worse; no glyph means "
    r"no separation (or, for SR@1-4, not formally tested). SR@5 uses paired STEP after "
    r"Bonferroni alpha-splitting with a single CALVIN binary family ($m=1$, "
    r"$\alpha_{\mathrm{split}}=0.05$); Avg. Len. uses a Welch t-test at $p < 0.05$."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a glyph LaTeX table for CALVIN ABC-D 1-step and max-step "
            "results. Supports multirow / pair layouts and optional bold-best."
        )
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--precision", type=int, default=2)
    parser.add_argument("--caption", default=DEFAULT_CAPTION)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument(
        "--bold-best",
        action="store_true",
        help="Bold the higher value between 1-step and max-step within each metric.",
    )
    parser.add_argument(
        "--layout",
        choices=["multirow", "pair", "grouped"],
        default="multirow",
        help=(
            "Table layout: multirow uses separate max-step and 1-step rows; "
            "pair uses one row per model with 'max-step / 1-step' cells; "
            "grouped uses one row per model with each metric split into "
            "Reference / 1 Step sub-columns under a \\multicolumn header."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _glyph_suffix(delta: float, separated: bool, *, show_approx: bool = True) -> str:
    if separated and delta > 0:
        glyph, color = r"\blacktriangle", "OliveGreen"
    elif separated and delta < 0:
        glyph, color = r"\blacktriangledown", "OrangeRed"
    elif show_approx:
        glyph, color = r"\approx", "gray"
    else:
        return ""  # grouped layout shows a glyph only on a significant separation
    return f" {{\\scriptsize $\\textcolor{{{color}}}{{{glyph}}}$}}"


def _bold(text: str, *, bold: bool) -> str:
    return rf"\textbf{{{text}}}" if bold else text


def _chain_glyph(
    one: RunResult,
    ref: RunResult,
    level: str,
    markers: dict[tuple[str, str], int],
    *,
    show_approx: bool = True,
) -> str:
    delta = (one.chain_sr[level] - ref.chain_sr[level]) * 100.0
    rounded_sign = 1 if delta > 0.0 else -1 if delta < 0.0 else 0
    separated = (
        rounded_sign != 0
        and markers.get((one.model_key, level)) == rounded_sign
    )
    return _glyph_suffix(delta, separated, show_approx=show_approx)


def _avg_glyph(
    one: RunResult,
    ref: RunResult,
    markers: dict[tuple[str, str], int],
    *,
    show_approx: bool = True,
) -> str:
    delta = one.avg_seq_len - ref.avg_seq_len
    rounded_sign = 1 if delta > 0.0 else -1 if delta < 0.0 else 0
    separated = (
        rounded_sign != 0
        and markers.get((one.model_key, "avg_len")) == rounded_sign
    )
    return _glyph_suffix(delta, separated, show_approx=show_approx)


# ---------------------------------------------------------------------------
# multirow layout (separate max-step and 1-step rows)
# ---------------------------------------------------------------------------

def _multirow_chain_cell(
    value: float,
    other_value: float,
    precision: int,
    *,
    bold_best: bool,
    glyph: str = "",
) -> str:
    base = format_percent(value, precision)
    base = _bold(base, bold=bold_best and value > other_value)
    return f"{base}{glyph}"


def _multirow_avg_cell(
    value: float,
    other_value: float,
    precision: int,
    *,
    bold_best: bool,
    glyph: str = "",
) -> str:
    base = format_number(value, precision)
    base = _bold(base, bold=bold_best and value > other_value)
    return f"{base}{glyph}"


def build_latex_multirow_table(
    results: dict[str, list[RunResult]],
    precision: int,
    caption: str,
    label: str,
    markers: dict[tuple[str, str], int],
    caption_suffix: str,
    bold_best: bool = False,
) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}{caption_suffix}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llllllll}",
        r"\toprule",
        r"Model & Steps & 1 & 2 & 3 & 4 & 5 & Avg. Len. \\",
        r"\midrule",
    ]

    model_keys = list(MODEL_HIGH_STEPS)
    for model_index, model_key in enumerate(model_keys):
        model_runs = results[model_key]
        ref_run = model_runs[0]    # max-step
        one_run = model_runs[1]    # 1-step
        model_name = latex_escape(MODEL_DISPLAY[model_key])
        multirow = rf"\multirow{{{len(model_runs)}}}{{*}}{{{model_name}}}"

        # max-step row
        lines.append(" & ".join([
            multirow,
            ref_run.step_label,
            *(
                _multirow_chain_cell(
                    ref_run.chain_sr[level],
                    one_run.chain_sr[level],
                    precision,
                    bold_best=bold_best,
                )
                for level in CHAIN_LEVELS
            ),
            _multirow_avg_cell(
                ref_run.avg_seq_len,
                one_run.avg_seq_len,
                precision,
                bold_best=bold_best,
            ),
        ]) + r" \\")

        # 1-step row (glyphs on SR@5 and Avg. Len.)
        chain_cells = []
        for level in CHAIN_LEVELS:
            glyph = _chain_glyph(one_run, ref_run, level, markers) if level == "5" else ""
            chain_cells.append(_multirow_chain_cell(
                one_run.chain_sr[level],
                ref_run.chain_sr[level],
                precision,
                bold_best=bold_best,
                glyph=glyph,
            ))
        avg_cell = _multirow_avg_cell(
            one_run.avg_seq_len,
            ref_run.avg_seq_len,
            precision,
            bold_best=bold_best,
            glyph=_avg_glyph(one_run, ref_run, markers),
        )
        lines.append(" & ".join([
            "",
            one_run.step_label,
            *chain_cells,
            avg_cell,
        ]) + r" \\")

        if model_index != len(model_keys) - 1:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# pair layout (one row per model; "max-step / 1-step" cells)
# ---------------------------------------------------------------------------

def _pair_chain_cell(
    one: RunResult,
    ref: RunResult,
    level: str,
    precision: int,
    markers: dict[tuple[str, str], int],
    *,
    bold_best: bool,
) -> str:
    one_v = one.chain_sr[level]
    ref_v = ref.chain_sr[level]
    one_text = _bold(format_percent(one_v, precision), bold=bold_best and one_v > ref_v)
    ref_text = _bold(format_percent(ref_v, precision), bold=bold_best and ref_v > one_v)
    glyph = _chain_glyph(one, ref, level, markers) if level == "5" else ""
    return f"{ref_text} / {one_text}{glyph}"


def _pair_avg_cell(
    one: RunResult,
    ref: RunResult,
    precision: int,
    markers: dict[tuple[str, str], int],
    *,
    bold_best: bool,
) -> str:
    one_v = one.avg_seq_len
    ref_v = ref.avg_seq_len
    one_text = _bold(format_number(one_v, precision), bold=bold_best and one_v > ref_v)
    ref_text = _bold(format_number(ref_v, precision), bold=bold_best and ref_v > one_v)
    return f"{ref_text} / {one_text}{_avg_glyph(one, ref, markers)}"


def build_latex_pair_table(
    results: dict[str, list[RunResult]],
    precision: int,
    caption: str,
    label: str,
    markers: dict[tuple[str, str], int],
    caption_suffix: str,
    bold_best: bool = False,
) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}{caption_suffix}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lllllll}",
        r"\toprule",
        r"Model & 1 & 2 & 3 & 4 & 5 & Avg. Len. \\",
        r"\midrule",
    ]

    for model_key in MODEL_HIGH_STEPS:
        model_runs = results[model_key]
        ref_run = model_runs[0]    # max-step
        one_run = model_runs[1]    # 1-step
        cells = [latex_escape(MODEL_DISPLAY[model_key])]
        for level in CHAIN_LEVELS:
            cells.append(_pair_chain_cell(
                one_run, ref_run, level, precision, markers,
                bold_best=bold_best,
            ))
        cells.append(_pair_avg_cell(
            one_run, ref_run, precision, markers,
            bold_best=bold_best,
        ))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# grouped layout (one row per model; each metric split into Reference / 1 Step
# sub-columns under a \multicolumn header, mirroring the LIBERO combined table)
# ---------------------------------------------------------------------------

# Column header label per chain level / metric.
_GROUP_HEADERS = [(level, f"SR@{level}") for level in CHAIN_LEVELS] + [("avg_len", "Avg. Len.")]


def _grouped_chain_cells(
    one: RunResult, ref: RunResult, level: str, precision: int,
    markers: dict[tuple[str, str], int],
) -> tuple[str, str]:
    """Return (Reference cell, 1-step cell) for one chain level; bold the higher,
    glyph (no approx) after the 1-step value when the level is formally tested."""
    ref_v, one_v = ref.chain_sr[level], one.chain_sr[level]
    ref_text = _bold(format_percent(ref_v, precision), bold=ref_v > one_v)
    one_text = _bold(format_percent(one_v, precision), bold=one_v > ref_v)
    glyph = _chain_glyph(one, ref, level, markers, show_approx=False) if level == "5" else ""
    return ref_text, f"{one_text}{glyph}"


def _grouped_avg_cells(
    one: RunResult, ref: RunResult, precision: int,
    markers: dict[tuple[str, str], int],
) -> tuple[str, str]:
    ref_v, one_v = ref.avg_seq_len, one.avg_seq_len
    ref_text = _bold(format_number(ref_v, precision), bold=ref_v > one_v)
    one_text = _bold(format_number(one_v, precision), bold=one_v > ref_v)
    glyph = _avg_glyph(one, ref, markers, show_approx=False)
    return ref_text, f"{one_text}{glyph}"


def build_latex_grouped_table(
    results: dict[str, list[RunResult]],
    precision: int,
    caption: str,
    label: str,
    markers: dict[tuple[str, str], int],
    caption_suffix: str,
    bold_best: bool = False,  # unused: the grouped layout always bolds the higher value
) -> str:
    n_groups = len(_GROUP_HEADERS)
    colspec = "l" + "cc" * n_groups
    header_groups = " ".join(
        rf"& \multicolumn{{2}}{{c}}{{{title}}}" for _, title in _GROUP_HEADERS
    )
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n_groups)
    )
    subheader = "".join(r" & Reference & 1 Step" for _ in _GROUP_HEADERS)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}{caption_suffix}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        rf"\multirow{{2}}{{*}}{{Model}} {header_groups} \\",
        cmidrules,
        rf"{subheader} \\",
        r"\midrule",
    ]

    for model_key in MODEL_HIGH_STEPS:
        model_runs = results[model_key]
        ref_run = model_runs[0]    # max-step (Reference)
        one_run = model_runs[1]    # 1-step
        cells: list[str] = [latex_escape(MODEL_DISPLAY[model_key])]
        for level in CHAIN_LEVELS:
            cells.extend(_grouped_chain_cells(one_run, ref_run, level, precision, markers))
        cells.extend(_grouped_avg_cells(one_run, ref_run, precision, markers))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    markers = _load_calvin_significance()
    if args.layout == "grouped":
        caption_suffix = CAPTION_SUFFIX_GROUPED
        builder = build_latex_grouped_table
    elif args.layout == "pair":
        caption_suffix = CAPTION_SUFFIX + PAIR_LAYOUT_NOTE
        builder = build_latex_pair_table
    else:
        caption_suffix = CAPTION_SUFFIX
        builder = build_latex_multirow_table
    table = builder(
        results=load_calvin_results(),
        precision=args.precision,
        caption=args.caption,
        label=args.label,
        markers=markers,
        caption_suffix=caption_suffix,
        bold_best=args.bold_best,
    )

    if args.output is None:
        print(table, end="")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
