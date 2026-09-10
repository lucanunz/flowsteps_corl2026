#!/usr/bin/env python3
"""Generate grouped compact TOST tables at a default margin of 3 percentage points.

Outputs cover LIBERO/LIBERO+, CALVIN, RoboTwin, and real-world experiments.
Simulation cells are coloured by paired TOST; the real-world table includes
paired confidence intervals and McNemar significance markers. Extraction runs
in separate processes because the benchmark packages share module names.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# This script lives in corl2026_logs/TOST_analysis/; the benchmarks (libero,
# calvin, …) and wilson_score_interval live one level up, in corl2026_logs/.
SCRIPT_DIR = Path(__file__).resolve().parent   # TOST_analysis/
ROOT = SCRIPT_DIR.parent                        # corl2026_logs/

# Canonical model keys / display labels (match the reference compact_glyph .tex).
MODEL_ORDER = ["dit", "smolvla", "xvla", "flower", "pi05", "lap", "mibot"]
MODEL_LABEL = {
    "dit": "DiT", "smolvla": "SmolVLA", "xvla": "X-VLA", "flower": "Flower",
    "pi05": r"Pi0.5", "lap": "LAP", "mibot": "XR-0",
}
# Map each benchmark's raw model key onto the canonical key above.
MODEL_NORM = {
    "dit": "dit", "fmt": "dit",
    "smolvla": "smolvla",
    "xvla": "xvla", "x-vla": "xvla",
    "flower": "flower",
    "pi05": "pi05", "pi0.5": "pi05", "openpi": "pi05",
    "lap": "lap",
    "mibot": "mibot",
}

# Real-world task layout (hard-coded so the parent needs no cross-package import).
RW_TASK_ORDER = ["spoon_drawer", "open_drawer", "open_oven", "close_oven"]
RW_TASK_LABEL = {
    "spoon_drawer": r"Spoon $\to$ Drawer", "open_drawer": "Open Drawer",
    "open_oven": "Open Oven", "close_oven": "Close Oven",
}

VERDICT_BG = {
    "Equivalent": "tostEqBg",
    "RelevantDifference": "tostDiffBg",
    "Inconclusive": "tostIncBg",
}


def _norm(model: str) -> str:
    return MODEL_NORM.get(model.lower().strip(), model.lower().strip())


# ─────────────────────────── extraction (subprocess) ────────────────────────

def _row(
    bench, dataset, model, col, *, t, a, b, c, d, high_steps,
    mcnemar_significant: bool = False,
):
    """One serialisable cell record for a *rate* metric (counts + TOST verdict)."""
    n = a + b + c + d
    return {
        "bench": bench, "dataset": dataset, "model": model, "col": col,
        "kind": "rate",
        "low_succ": a + b,      # 1-step successes (p_low marginal)
        "high_succ": a + c,     # reference successes (p_high marginal)
        "n": n,
        "high_steps": high_steps,   # reference (multi-step) variant's step count
        "verdict": (t.verdict if t else None),
        "delta": (t.delta_hat if t else None),
        "ci_low": (t.ci_low if t else None),
        "ci_high": (t.ci_high if t else None),
        "mcnemar_significant": mcnemar_significant,
    }


def _row_mean(bench, dataset, model, col, *, t, low_mean, high_mean, n, high_steps):
    """One serialisable cell record for a *mean* metric (e.g. CALVIN chain length).

    Cells show ``reference / 1-step`` mean values (0–5 subtasks) rather than
    success rates; the TOST verdict is the paired-t result on the per-rollout
    score differences.
    """
    return {
        "bench": bench, "dataset": dataset, "model": model, "col": col,
        "kind": "mean",
        "low_val": low_mean,    # 1-step mean (0–5)
        "high_val": high_mean,  # reference mean (0–5)
        "n": n,
        "high_steps": high_steps,
        "verdict": (t.verdict if t else None),
        "delta": (t.delta_hat if t else None),
        "ci_low": (t.ci_low if t else None),
        "ci_high": (t.ci_high if t else None),
    }


def _extract(bench: str, delta: float, alpha: float, chain_delta: float) -> list[dict]:
    benchdir = ROOT / bench
    for p in (str(ROOT), str(benchdir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    rows: list[dict] = []

    if bench == "libero":
        from tri_statistics_TOST.config import PAIR_SPECS  # type: ignore
        from tri_statistics_TOST.stats import paired_tost  # type: ignore
        from tri_statistics_STEP.pairs import build_pair_result, cells_in_order  # type: ignore

        sgo = {"libero_spatial", "libero_object", "libero_goal"}

        def pooled(cells):
            a = sum(c.both_success for c in cells)
            b = sum(c.discordant_b for c in cells)
            cc = sum(c.discordant_c for c in cells)
            d = sum(c.both_failure for c in cells)
            return a, b, cc, d, paired_tost(a, b, cc, d, delta=delta, alpha=alpha)

        for spec in PAIR_SPECS:
            pair = build_pair_result(spec)
            by_suite = {c.suite: c for c in cells_in_order(pair)}
            blabel = "libero" if spec.dataset == "libero" else "liberoplus"
            sgo_cells = [by_suite[s] for s in sgo if s in by_suite]
            if sgo_cells:
                a, b, cc, d, t = pooled(sgo_cells)
                rows.append(_row("libero", blabel, spec.model, "SGO",
                                 t=t, a=a, b=b, c=cc, d=d,
                                 high_steps=spec.variant_high))
            if "libero_10" in by_suite:
                a, b, cc, d, t = pooled([by_suite["libero_10"]])
                rows.append(_row("libero", blabel, spec.model, "L10",
                                 t=t, a=a, b=b, c=cc, d=d,
                                 high_steps=spec.variant_high))

    elif bench == "calvin":
        from tri_statistics_TOST.equivalence import chain_length_tost  # type: ignore
        from tri_statistics_TOST.stats import paired_tost  # type: ignore
        from tri_statistics_STEP.config import CELL_KEYS, PAIR_SPECS  # type: ignore
        from tri_statistics_STEP.io_runs import paired_completion_scores  # type: ignore
        from tri_statistics_STEP.pairs import build_pair_result  # type: ignore
        for spec in PAIR_SPECS:
            pair = build_pair_result(spec)
            cell = pair.cells[CELL_KEYS[0]]
            a, b = cell.both_success, cell.discordant_b
            cc, d = cell.discordant_c, cell.both_failure
            t = paired_tost(a, b, cc, d, delta=delta, alpha=alpha)
            rows.append(_row("calvin", None, spec.model, "SR@5",
                             t=t, a=a, b=b, c=cc, d=d,
                             high_steps=spec.variant_high))

            # Mean chain length (0–5): paired-t TOST on the same paired rollouts.
            ps = (paired_completion_scores(spec, pair.low_run, pair.high_run)
                  if spec.has_paired_data else None)
            if ps is None:
                ct, low_mean, high_mean, n_chain = None, 0.0, 0.0, 0
            else:
                low_scores, high_scores = ps
                ct = chain_length_tost(low_scores, high_scores,
                                       delta=chain_delta, alpha=alpha)
                n_chain = len(low_scores)
                low_mean = sum(low_scores) / n_chain
                high_mean = sum(high_scores) / n_chain
            rows.append(_row_mean("calvin", None, spec.model, "Chain",
                                  t=ct, low_mean=low_mean, high_mean=high_mean,
                                  n=n_chain, high_steps=spec.variant_high))

    elif bench == "robotwin":
        from tri_statistics_TOST.stats import paired_tost  # type: ignore
        from tri_statistics_STEP.config import select_specs  # type: ignore
        from tri_statistics_STEP.pairs import build_pair_result  # type: ignore
        setting_key = {"clean": "Clean", "rand": "Rand"}
        for spec in select_specs():
            cell = next(iter(build_pair_result(spec).cells.values()))
            a, b = cell.both_success, cell.discordant_b
            cc, d = cell.discordant_c, cell.both_failure
            t = paired_tost(a, b, cc, d, delta=delta, alpha=alpha)
            rows.append(_row("robotwin", None, spec.model,
                             setting_key.get(spec.setting, spec.setting.capitalize()),
                             t=t, a=a, b=b, c=cc, d=d,
                             high_steps=spec.variant_high))

    elif bench == "real_world":
        from tri_statistics_McNemar.config import EXACT_MAX_DISCORDANT  # type: ignore
        from tri_statistics_McNemar.mcnemar import (  # type: ignore
            apply_family_multiplicity,
            cell_mcnemar,
        )
        from tri_statistics_McNemar.stats import mcnemar_test  # type: ignore
        from tri_statistics_TOST.stats import paired_tost  # type: ignore
        from tri_statistics_STEP.config import MODELS, PAIR_SPECS  # type: ignore
        from tri_statistics_STEP.pairs import build_pair_result  # type: ignore
        by_model: dict[str, list] = {}
        high_by_model: dict[str, int] = {}
        for spec in PAIR_SPECS:
            try:
                cell = build_pair_result(spec).cells[spec.task]
            except Exception:
                continue
            by_model.setdefault(spec.model, []).append((spec.task, cell))
            high_by_model[spec.model] = spec.variant_high
        for model in MODELS:
            cells = by_model.get(model, [])
            hs = high_by_model.get(model)
            tot = [0, 0, 0, 0]
            mcnemar_results = [
                cell_mcnemar(
                    cell, alpha=alpha,
                    exact_max_discordant=EXACT_MAX_DISCORDANT,
                )
                for _, cell in cells
            ]
            apply_family_multiplicity(mcnemar_results)
            for (task, cell), mcnemar in zip(cells, mcnemar_results):
                a, b = cell.both_success, cell.discordant_b
                cc, d = cell.discordant_c, cell.both_failure
                t = paired_tost(a, b, cc, d, delta=delta, alpha=alpha)
                rows.append(_row("real_world", None, model, task,
                                 t=t, a=a, b=b, c=cc, d=d, high_steps=hs,
                                 mcnemar_significant=(
                                     mcnemar.q_holm is not None
                                     and mcnemar.q_holm < alpha
                                 )))
                tot[0] += a; tot[1] += b; tot[2] += cc; tot[3] += d
            if cells:
                a, b, cc, d = tot
                t = paired_tost(a, b, cc, d, delta=delta, alpha=alpha)
                pooled_mcnemar = mcnemar_test(
                    b, cc, n_paired=a + b + cc + d, alpha=alpha,
                    exact_max_discordant=EXACT_MAX_DISCORDANT,
                )
                rows.append(_row("real_world", None, model, "Pooled",
                                 t=t, a=a, b=b, c=cc, d=d, high_steps=hs,
                                 mcnemar_significant=pooled_mcnemar.significant))
    else:
        raise SystemExit(f"unknown benchmark {bench!r}")

    return rows


# ─────────────────────────────── LaTeX assembly ─────────────────────────────

PREAMBLE = r"""% --- move these to your preamble (requires xcolor with the [table] option) ---
% \usepackage[table]{xcolor}
% \usepackage{booktabs, multirow}
\definecolor{tostEqBg}{HTML}{C8E6C9}
\definecolor{tostDiffBg}{HTML}{FFCDD2}
\definecolor{tostIncBg}{HTML}{FFE0B2}
\newcommand{\cNA}{\textcolor{black!30}{--}}
% -----------------------------------------------------------------------------
"""

# Legend snippet reused across captions. The "cells show…" clause depends on the
# layout; the verdict-colour clause is shared.
_CELL_DESC = {
    "pair": (r"Cells show \emph{reference\,/\,1-step} success rates (higher in "
             r"bold). "),
    "multirow": (r"Each model spans two rows — the multi-step reference and 1-step "
                 r"(the \emph{Steps} column gives the reference step count) — with "
                 r"the higher rate of each pair in bold. "),
    "grouped": (r"Each metric is split into \emph{Reference} (multi-step) and "
                r"\emph{1 Step} sub-columns, with the higher of the two in bold. "),
}
def _fmt(x: float) -> str:
    """Trim trailing zeros for captions: 3.0 -> '3', 2.5 -> '2.5'."""
    return f"{x:g}"


def _colour_desc(delta_pp: float, alpha: float) -> str:
    return (
        r"Cells are coloured by the paired-TOST verdict of 1-step vs the multi-step "
        rf"reference (margin $\delta={_fmt(delta_pp)}$\,pp, $\alpha={_fmt(alpha)}$): "
        r"\colorbox{tostEqBg}{equivalent} (90\% CI $\subset\pm\delta$), "
        r"\colorbox{tostDiffBg}{relevant difference} (CI outside $\pm\delta$), "
        r"\colorbox{tostIncBg}{inconclusive} (CI straddles $\pm\delta$); "
        r"\cNA~not assessed."
    )


def _legend(layout: str, delta_pp: float, alpha: float) -> str:
    return _CELL_DESC[layout] + _colour_desc(delta_pp, alpha)


def _pct(succ: int, n: int) -> str:
    return f"{100.0 * succ / n:.1f}\\%"


def _bold(text: str, *, on: bool) -> str:
    return rf"\textbf{{{text}}}" if on else text


def _cell_values(rec: dict) -> tuple[str, float, str, float]:
    """(low_display, low_compare, high_display, high_compare) for a record.

    ``rate`` records render percentages; ``mean`` records render the raw 0–5
    mean. The *compare* values drive which side is bold (higher is better in
    both cases).
    """
    if rec.get("kind") == "mean":
        lo, hi = rec["low_val"], rec["high_val"]
        return f"{lo:.2f}", lo, f"{hi:.2f}", hi
    n, lo, hi = rec["n"], rec["low_succ"], rec["high_succ"]
    return _pct(lo, n), lo, _pct(hi, n), hi


def _cell_single(rec: dict | None, which: str, *, colour: bool = True) -> str:
    """Multirow layout: one coloured cell with a single variant's value.

    ``which`` is 'ref' (reference / multi-step row) or 'low' (1-step row). Both
    rows of a metric carry the same verdict colour so the pair reads at a glance;
    the higher of the two values is bold.
    """
    if rec is None or rec.get("verdict") is None:
        return r"\cNA"
    lo_d, lo_c, hi_d, hi_c = _cell_values(rec)
    # Bold the strictly-higher value; on an exact tie neither is bold (matches the
    # reference table's _best_bold_flags).
    disp, is_high = (hi_d, hi_c > lo_c) if which == "ref" else (lo_d, lo_c > hi_c)
    value = _bold(disp, on=is_high)
    if which == "low" and rec.get("mcnemar_significant", False):
        value += r"\textsuperscript{*}"
    if not colour:
        return value
    return rf"\cellcolor{{{VERDICT_BG[rec['verdict']]}}}{value}"


def _ci_pair(rec: dict | None) -> str:
    """Small paired-difference CI spanning a Reference / 1 Step pair."""
    if rec is None or rec.get("ci_low") is None or rec.get("ci_high") is None:
        return r"\textcolor{black!30}{\scriptsize $[--,\,--]$}"

    def signed_pp(value: float) -> str:
        return f"{value * 100:+.1f}"

    lo = signed_pp(rec["ci_low"])
    hi = signed_pp(rec["ci_high"])
    return rf"\textcolor{{black!55}}{{\scriptsize $[{lo},\,{hi}]$}}"


def _grid(rows: list[dict]) -> dict[tuple, dict]:
    """grid[(bench, dataset, model_canonical, col)] = record."""
    g: dict[tuple, dict] = {}
    for r in rows:
        g[(r["bench"], r["dataset"], _norm(r["model"]), r["col"])] = r
    return g


def _wrap(caption: str, label: str, colspec: str, header_lines: list[str],
          body_rows: list[str]) -> str:
    out = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
        *header_lines, r"\midrule",
    ]
    out.extend(body_rows)
    out.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    return "\n".join(out) + "\n"


def _grouped_table(caption, label, base_colspec, col_titles, key_lists, grid) -> str:
    """Grouped layout: one row per model; each metric is a Reference / 1 Step pair
    of sub-columns under a \\multicolumn header (mirrors the reference grouped
    glyph table). ``base_colspec`` is ignored — the spec is rebuilt as l + cc·k."""
    n = len(col_titles)
    colspec = "l" + "cc" * n
    header_groups = " ".join(rf"& \multicolumn{{2}}{{c}}{{{t}}}" for t in col_titles)
    cmids = " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n))
    subheader = "".join(r" & Reference & 1 Step" for _ in col_titles) + r" \\"
    header_lines = [rf"\multirow{{2}}{{*}}{{Model}} {header_groups} \\", cmids, subheader]

    show_pair_cis = label == "tab:tost_realworld_compact"
    body = []
    for m in MODEL_ORDER:
        recs = [grid.get((b, d, m, c)) for (b, d, c) in key_lists]
        if all(r is None or r.get("verdict") is None for r in recs):
            continue
        model_cell = MODEL_LABEL[m]
        if show_pair_cis:
            model_cell = rf"\multirow{{2}}{{*}}{{{model_cell}}}"
        cells: list[str] = [model_cell]
        for r in recs:
            cells.append(_cell_single(r, "ref", colour=not show_pair_cis))
            cells.append(_cell_single(r, "low", colour=not show_pair_cis))
        row_end = r" \\[-1pt]" if show_pair_cis else r" \\"
        body.append(" & ".join(cells) + row_end)
        if show_pair_cis:
            ci_cells = " ".join(
                rf"& \multicolumn{{2}}{{c}}{{{_ci_pair(r)}}}" for r in recs
            )
            body.append(ci_cells + r" \\[3pt]")
    return _wrap(caption, label, colspec, header_lines, body)


# Per-benchmark table specs: (caption, label, column-c-spec, column titles, keys).
# All accept (grid, delta_pp, chain_delta, alpha); most only need ``grid``.
def _libero_spec(grid, delta_pp, chain_delta, alpha):
    return (
        r"Compact LIBERO and LIBERO+ success rates with TOST equivalence "
        r"colouring. S/G/O Avg pools Spatial, Goal and Object (the verdict is the "
        r"paired TOST on the pooled 2$\times$2 table); LIBERO-10 is its own suite. ",
        "tab:tost_libero_compact", "llll",
        ["LIBERO S/G/O Avg", "LIBERO-10", "LIBERO+ S/G/O Avg", "LIBERO+-10"],
        [("libero", "libero", "SGO"), ("libero", "libero", "L10"),
         ("libero", "liberoplus", "SGO"), ("libero", "liberoplus", "L10")],
    )


def _calvin_spec(grid, delta_pp, chain_delta, alpha):
    return (
        r"CALVIN ABC-D 5-task chain success (SR@5) and mean chain length "
        r"(0--5 completed subtasks) with TOST equivalence colouring. The "
        r"\emph{Mean chain} cells show \emph{reference\,/\,1-step} mean subtasks "
        r"completed (higher in bold); its verdict is a paired \emph{t}-test TOST "
        rf"on the per-rollout score differences against $\delta={_fmt(chain_delta)}$ subtasks "
        rf"(the SR@5 column keeps $\delta={_fmt(delta_pp)}$\,pp). ",
        "tab:tost_calvin_compact", "ll", ["CALVIN SR@5", "Mean chain"],
        [("calvin", None, "SR@5"), ("calvin", None, "Chain")],
    )


def _robotwin_spec(grid, delta_pp, chain_delta, alpha):
    return (
        r"RoboTwin success rates (pooled over 50 tasks $\times$ 100 seeds, "
        r"X-VLA) with TOST equivalence colouring. ",
        "tab:tost_robotwin_compact", "ll", ["Clean", "Randomized"],
        [("robotwin", None, "Clean"), ("robotwin", None, "Rand")],
    )


def _real_world_spec(grid, delta_pp, chain_delta, alpha):
    cols = RW_TASK_ORDER + ["Pooled"]
    return (
        r"Real-world manual-success rates per task and pooled (n=100 per task, "
        r"400 pooled). ",
        "tab:tost_realworld_compact", "c" * len(cols),
        [RW_TASK_LABEL[t] for t in RW_TASK_ORDER] + ["Pooled"],
        [("real_world", None, c) for c in cols],
    )




def build_table(spec_fn, grid, layout: str,
                delta_pp: float, chain_delta: float, alpha: float) -> str:
    caption, label, cspec, titles, keys = spec_fn(grid, delta_pp, chain_delta, alpha)
    if layout == "grouped" and label == "tab:tost_realworld_compact":
        caption += (
            _CELL_DESC["grouped"]
            + r"The small gray line below each pair reports its paired 90\% CI for "
            r"$\Delta=p_{\mathrm{Reference}}-p_{\mathrm{1\ Step}}$. "
            r"A superscript * marks a significant paired McNemar difference "
            r"($\alpha=0.05$; per-task $p$-values Holm-adjusted within each model; "
            r"pooled $p$-values unadjusted)."
        )
    else:
        caption = caption + _legend(layout, delta_pp, alpha)
    return _grouped_table(caption, label, cspec, titles, keys, grid)


# ──────────────────────────────────── CLI ───────────────────────────────────

BUILDERS = {
    "libero": ("tost_libero_compact.tex", _libero_spec),
    "calvin": ("tost_calvin_compact.tex", _calvin_spec),
    "robotwin": ("tost_robotwin_compact.tex", _robotwin_spec),
    "real_world": ("tost_realworld_compact.tex", _real_world_spec),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "verdict_tables")
    p.add_argument("--delta-pp", type=float, default=3.0)
    p.add_argument("--chain-delta", type=float, default=None,
                   help="Equivalence margin δ for the CALVIN mean chain length, in "
                        "subtask units. If omitted it scales with --delta-pp at the "
                        "pre-registered ratio (0.15 subtasks at δ=3pp), e.g. 0.05 at "
                        "δ=1pp; pass a value to override.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--layout", choices=["grouped"],
                   default="grouped",
                   help="Grouped Reference / 1 Step columns (the retained compact layout).")
    p.add_argument("--extract", choices=list(BUILDERS), default=None,
                   help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    delta = args.delta_pp / 100.0
    # Chain-length margin scales with --delta-pp unless explicitly overridden,
    # keeping the pre-registered ratio (0.15 subtasks at δ=3pp).
    chain_delta = (args.chain_delta if args.chain_delta is not None
                   else args.delta_pp * (0.15 / 3.0))

    if args.extract:
        print(json.dumps(_extract(args.extract, delta, args.alpha, chain_delta)))
        return 0

    # Extract every benchmark in its own subprocess (import-collision isolation).
    rows: list[dict] = []
    for bench in BUILDERS:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--extract", bench,
             "--delta-pp", str(args.delta_pp), "--chain-delta", str(chain_delta),
             "--alpha", str(args.alpha)],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(f"[ERROR] extraction for {bench} failed:\n{proc.stderr}\n")
            return 1
        rows.extend(json.loads(proc.stdout.strip().splitlines()[-1]))

    grid = _grid(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fname, spec_fn in BUILDERS.values():
        latex = PREAMBLE + "\n" + build_table(spec_fn, grid, args.layout,
                                              args.delta_pp, chain_delta, args.alpha)
        (args.out_dir / fname).write_text(latex, encoding="utf-8")
        print(f"% wrote {args.out_dir / fname}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
