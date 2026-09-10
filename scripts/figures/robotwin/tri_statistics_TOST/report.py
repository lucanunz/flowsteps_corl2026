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
    sr_low: str
    sr_high: str
    tost: object  # TostResult | None


@dataclass
class Section:
    title: str | None
    combined_line: str | None
    rows: list = field(default_factory=list)


def fmt_sr(counts, wilson) -> str:
    return (
        f"{counts.rate * 100:.1f}%"
        f" [{wilson[0] * 100:.1f}, {wilson[1] * 100:.1f}]"
        f" ({counts.successes}/{counts.trials})"
    )


def fmt_ci(tost) -> str:
    return f"[{tost.ci_low * 100:+.1f}, {tost.ci_high * 100:+.1f}]"


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
        lines.append(
            f"| {item_hdr} | {low_hdr} | {high_hdr} "
            f"| Δ pp | 90% CI (pp) | margin | n_paired | verdict |"
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
                f"| {t.delta_hat * 100:+.1f} | {fmt_ci(t)} | ±{delta_pp:.0f}pp "
                f"| {t.n_paired} | {_VERDICT_DISPLAY[t.verdict]} |"
            )

    lines.append("")
    lines.append(
        "_Verdicts: **equivalent** = 90% CI ⊂ ±δ (equivalence shown); "
        "difference > δ = CI lies entirely outside ±δ; inconclusive = CI straddles "
        "a margin boundary (underpowered). Paired TOST, Newcombe (1998) Method 10 "
        "MOVER interval. Paired data only._"
    )
    if footer:
        lines.append("")
        lines.append(footer)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_combined_line(verdict, *, low_label: str, high_label: str, delta_pp: float) -> str:
    """One-line intersection–union combined verdict (no Bonferroni)."""
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
        f"K={K}, α={verdict.alpha:.2f} per cell, δ=±{delta_pp:.0f}pp):"
    )
    if verdict.combined_decision == "EquivalentOnAll":
        body = (
            f"**{high_label} is equivalent to {low_label} within ±{delta_pp:.0f}pp "
            f"on every cell** ({counts})."
        )
    elif verdict.combined_decision == "DifferenceOnSome":
        body = (
            f"**not equivalent on all** — at least one cell shows a "
            f"≥ {delta_pp:.0f}pp difference ({counts})."
        )
    else:
        body = f"**not established** — {counts}."
    return f"{head} {body}"
