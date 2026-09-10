"""Generic section-based markdown report for the equivalence (TOST) forks.

A report is a title + intro followed by one or more sections; each section has an
optional combined-verdict line and a table of rows. This shape covers CALVIN
(one section, SR@5 rows), RoboTwin (a section per setting) and real-world (a
section per model). Cells without paired data render "not assessed".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .stats import EQUIVALENT, INCONCLUSIVE, RELEVANT_DIFFERENCE

_VERDICT_DISPLAY = {
    EQUIVALENT: "**equivalent**",
    RELEVANT_DIFFERENCE: "difference > δ",
    INCONCLUSIVE: "inconclusive",
}
_NOT_ASSESSED = "— (no paired data; not assessed)"


@dataclass
class RowData:
    label: str
    sr_low: str   # generic "low" descriptive cell (SR% for SR@5, mean[CI] for chain length)
    sr_high: str  # generic "high" descriptive cell
    tost: object  # TostResult | None


@dataclass
class Section:
    title: str | None
    combined_line: str | None
    rows: list = field(default_factory=list)
    # Per-section unit formatting (defaults reproduce the SR@5 / pp output).
    value_scale: float = 100.0      # ×100 for proportions→pp, ×1 for subtasks
    value_decimals: int = 1         # decimals for Δ / CI columns
    low_hdr: str | None = None      # overrides the report-level low/high headers
    high_hdr: str | None = None
    delta_hdr: str = "Δ pp"
    ci_hdr: str = "90% CI (pp)"
    margin_display: str = "±3pp"    # rendered in the per-row margin column


def fmt_sr(counts, wilson) -> str:
    return (
        f"{counts.rate * 100:.1f}%"
        f" [{wilson[0] * 100:.1f}, {wilson[1] * 100:.1f}]"
        f" ({counts.successes}/{counts.trials})"
    )


def fmt_chain(stats) -> str:
    """Mean chain length with a 95% normal CI from a CompletionStats (mean, variance, n)."""
    from math import sqrt

    n = stats.n
    mean = stats.mean
    if n > 1 and stats.variance > 0.0:
        se = sqrt(stats.variance / n)
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
    else:
        lo = hi = mean
    return f"{mean:.2f} [{lo:.2f}, {hi:.2f}] (n={n})"


def fmt_ci(tost, *, scale: float = 100.0, decimals: int = 1) -> str:
    return f"[{tost.ci_low * scale:+.{decimals}f}, {tost.ci_high * scale:+.{decimals}f}]"


def write_equivalence_report(
    out_path: Path,
    *,
    title: str,
    intro: str,
    sections: list,
    item_hdr: str,
    low_hdr: str,
    high_hdr: str,
    delta_pp: float,
    footer: str | None = None,
) -> None:
    lines: list[str] = [f"# {title}", ""]
    lines.append(
        "> **Single-seed caveat**: results come from a single training run. "
        "Uncertainty estimates characterise *evaluation* stochasticity only."
    )
    lines.append("")
    lines.append(intro)

    for section in sections:
        lines.append("")
        if section.title:
            lines.append(f"## {section.title}")
            lines.append("")
        if section.combined_line:
            lines.append(section.combined_line)
            lines.append("")
        sec_low_hdr = section.low_hdr or low_hdr
        sec_high_hdr = section.high_hdr or high_hdr
        scale = section.value_scale
        dec = section.value_decimals
        lines.append(
            f"| {item_hdr} | {sec_low_hdr} | {sec_high_hdr} "
            f"| {section.delta_hdr} | {section.ci_hdr} | margin | n_paired | verdict |"
        )
        lines.append("| --- | --- | --- | ---: | :---: | :---: | ---: | :---: |")
        for row in section.rows:
            if row.tost is None:
                lines.append(
                    f"| {row.label} | {row.sr_low} | {row.sr_high} "
                    f"| {_NOT_ASSESSED} | — | — | — | — |"
                )
                continue
            t = row.tost
            lines.append(
                f"| {row.label} | {row.sr_low} | {row.sr_high} "
                f"| {t.delta_hat * scale:+.{dec}f} | {fmt_ci(t, scale=scale, decimals=dec)} "
                f"| {section.margin_display} | {t.n_paired} | {_VERDICT_DISPLAY[t.verdict]} |"
            )

    lines.append("")
    lines.append(
        "_Verdicts: **equivalent** = 90% CI ⊂ ±δ (equivalence shown); "
        "difference > δ = CI lies entirely outside ±δ; inconclusive = CI straddles "
        "a margin boundary (underpowered). Paired TOST; CI by Newcombe's (1998) "
        "Method 10 for the SR@5 paired-proportion difference, and a paired t-test for "
        "the mean chain-length difference. Paired data only._"
    )
    if footer:
        lines.append("")
        lines.append(footer)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_combined_line(
    verdict,
    *,
    low_label: str,
    high_label: str,
    delta_pp: float,
    margin_str: str | None = None,
) -> str:
    """One-line intersection–union combined verdict (no Bonferroni).

    `margin_str` overrides the rendered margin (e.g. "±0.15 subtasks"); it
    defaults to "±{delta_pp}pp" for the SR@5 metric.
    """
    margin = margin_str or f"±{delta_pp:.0f}pp"
    K = verdict.family_size
    counts = (
        f"{verdict.n_equivalent}/{K} equivalent, "
        f"{verdict.n_relevant}/{K} difference>δ, "
        f"{verdict.n_inconclusive}/{K} inconclusive"
    )
    if verdict.n_unassessed:
        counts += f", {verdict.n_unassessed}/{K} not assessed"
    head = (
        f"Combined equivalence verdict (intersection–union, **no Bonferroni**; "
        f"K={K}, α={verdict.alpha:.2f} per cell, δ={margin}):"
    )
    if verdict.combined_decision == "EquivalentOnAll":
        body = (
            f"**{high_label} is equivalent to {low_label} within {margin} "
            f"on every cell** ({counts})."
        )
    elif verdict.combined_decision == "DifferenceOnSome":
        body = (
            f"**not equivalent on all** — at least one cell shows a "
            f"≥ {margin.lstrip('±')} difference ({counts})."
        )
    else:
        body = f"**not established** — {counts}."
    return f"{head} {body}"
