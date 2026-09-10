"""Markdown reports for the equivalence (TOST) variant.

Per-pair report: a Wilson-CI success-rate table plus the paired TOST verdict
(Δ̂, 90% CI, margin ±δ) for every cell that has paired data; cells without paired
data are flagged "not assessed". Per-task report mirrors the STEP layout but
reports per-task equivalence and opens each suite with the intersection–union
combined verdict (explicitly noting that no Bonferroni is applied).
"""

from __future__ import annotations

from pathlib import Path

from tri_statistics_STEP.pairs import cells_in_order  # reuse cell ordering

from .config import MODEL_DISPLAY, OVERALL_KEY, SUITE_DISPLAY, SUITE_ORDER
from .stats import EQUIVALENT, INCONCLUSIVE, RELEVANT_DIFFERENCE


def _ordered_cells(pair):
    """Cells in report order, honouring a custom `pair.cell_order` if present
    (set by --group-suites); otherwise the standard suite order."""
    order = getattr(pair, "cell_order", None)
    if order is not None:
        return [pair.cells[k] for k in order if k in pair.cells]
    return cells_in_order(pair)

_NOT_ASSESSED = "— (no paired data; not assessed)"

_VERDICT_DISPLAY = {
    EQUIVALENT: "**equivalent**",
    RELEVANT_DIFFERENCE: "difference > δ",
    INCONCLUSIVE: "inconclusive",
}


def _fmt_sr(counts, wilson) -> str:
    return (
        f"{counts.rate * 100:.1f}%"
        f" [{wilson[0] * 100:.1f}, {wilson[1] * 100:.1f}]"
        f" ({counts.successes}/{counts.trials})"
    )


def _fmt_ci(tost) -> str:
    return f"[{tost.ci_low * 100:+.1f}, {tost.ci_high * 100:+.1f}]"


def _headline(overall_tost, variant_low: int, variant_high: int, delta_pp: float) -> str:
    """One-line plain-language summary for the Overall row."""
    if overall_tost is None:
        return (
            f"**Overall: equivalence not assessed** — no paired episode data for "
            f"this model/dataset."
        )
    ci = _fmt_ci(overall_tost)
    d = f"{overall_tost.delta_hat * 100:+.1f}"
    if overall_tost.verdict == EQUIVALENT:
        return (
            f"**Overall: {variant_high}-step is equivalent to {variant_low}-step "
            f"within ±{delta_pp:.0f}pp** (Δ̂ = {d}pp, 90% CI {ci} pp ⊂ "
            f"±{delta_pp:.0f}pp)."
        )
    if overall_tost.verdict == RELEVANT_DIFFERENCE:
        return (
            f"**Overall: a relevant difference (≥ {delta_pp:.0f}pp) is demonstrated** "
            f"(Δ̂ = {d}pp, 90% CI {ci} pp lies outside ±{delta_pp:.0f}pp)."
        )
    return (
        f"**Overall: inconclusive at ±{delta_pp:.0f}pp** — neither equivalence nor "
        f"a ≥ {delta_pp:.0f}pp difference is established (Δ̂ = {d}pp, 90% CI {ci} pp "
        f"straddles a margin boundary)."
    )


def write_pair_report(
    pair,
    out_path: Path,
    confidence: float = 0.95,
    delta_pp: float = 3.0,
) -> None:
    """Write the per-(model, dataset) equivalence report."""
    model_name = MODEL_DISPLAY.get(pair.spec.model, pair.spec.model)
    dataset_label = "LIBERO+" if pair.spec.dataset == "liberoplus" else "LIBERO"
    ci_pct = f"{confidence:.0%}"
    alpha = getattr(pair, "tost_alpha", 0.05)
    cells = _ordered_cells(pair)
    overall_cell = pair.cells.get(OVERALL_KEY)
    overall_tost = getattr(overall_cell, "tost", None) if overall_cell else None

    lines: list[str] = []
    lines.append(
        f"# {model_name} — {dataset_label}: equivalence of "
        f"{pair.spec.variant_low}-step vs {pair.spec.variant_high}-step"
    )
    lines.append("")
    lines.append(
        "> **Single-seed caveat**: results come from a single training run. "
        "Uncertainty estimates characterise *evaluation* stochasticity only, "
        "not training stochasticity."
    )
    lines.append("")
    lines.append(
        f"**Test.** Paired TOST (two one-sided tests) of Δ = "
        f"p_{{{pair.spec.variant_high}-step}} − p_{{{pair.spec.variant_low}-step}} "
        f"against the pre-registered margin **δ = {delta_pp:.0f}pp**, at "
        f"α = {alpha:.2f}. Equivalence is declared when the two-sided "
        f"{1 - 2 * alpha:.0%} CI for Δ lies entirely within ±{delta_pp:.0f}pp. "
        "The CI is Newcombe's score-interval method for the difference between "
        "paired proportions (Newcombe 1998, Method 10) on the paired per-rollout "
        "2×2 table. **Paired data only** — cells without per-episode data are not "
        "assessed."
    )

    if getattr(pair, "notes", None):
        lines.append("")
        lines.append("> **Data-loading notes:**")
        for note in pair.notes:
            lines.append(f">  - {note}")

    if overall_cell is not None:
        lines.append("")
        lines.append(_headline(overall_tost, pair.spec.variant_low,
                               pair.spec.variant_high, delta_pp))

    # ── Equivalence table ────────────────────────────────────────────────────
    lines.append("")
    lines.append("## Equivalence (paired TOST)")
    lines.append("")
    low_hdr = f"{pair.spec.variant_low}-step ({ci_pct} CI)"
    high_hdr = f"{pair.spec.variant_high}-step ({ci_pct} CI)"
    lines.append(
        f"| Suite | {low_hdr} | {high_hdr} | Δ pp (paired) "
        f"| 90% CI (pp) | margin | n_paired | verdict |"
    )
    lines.append(
        "| --- | --- | --- | ---: | :---: | :---: | ---: | :---: |"
    )
    for cell in cells:
        suite_label = SUITE_DISPLAY.get(cell.suite, cell.suite)
        sr_low = _fmt_sr(cell.counts_low, cell.wilson_low)
        sr_high = _fmt_sr(cell.counts_high, cell.wilson_high)
        tost = getattr(cell, "tost", None)
        if tost is None:
            lines.append(
                f"| {suite_label} | {sr_low} | {sr_high} | {_NOT_ASSESSED} "
                f"| — | — | — | — |"
            )
            continue
        lines.append(
            f"| {suite_label} | {sr_low} | {sr_high} "
            f"| {tost.delta_hat * 100:+.1f} | {_fmt_ci(tost)} "
            f"| ±{delta_pp:.0f}pp | {tost.n_paired} "
            f"| {_VERDICT_DISPLAY[tost.verdict]} |"
        )

    lines.append("")
    lines.append(
        "_Verdicts: **equivalent** = 90% CI ⊂ ±δ (equivalence shown); "
        "difference > δ = CI lies entirely outside ±δ; inconclusive = CI straddles "
        "a margin boundary (underpowered)._"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_combined(verdict, variant_low: int, variant_high: int, delta_pp: float) -> str:
    K = verdict.family_size
    counts = (
        f"{verdict.n_equivalent}/{K} equivalent, "
        f"{verdict.n_relevant}/{K} difference>δ, "
        f"{verdict.n_inconclusive}/{K} inconclusive"
    )
    if verdict.n_unassessed:
        counts += f", {verdict.n_unassessed}/{K} not assessed (no paired data)"

    head = (
        f"Combined equivalence verdict (intersection–union, **no Bonferroni**; "
        f"K={K}, α={verdict.alpha:.2f} per task, δ=±{delta_pp:.0f}pp):"
    )
    if verdict.combined_decision == "EquivalentOnAll":
        body = (
            f"**every task is equivalent within ±{delta_pp:.0f}pp** at α="
            f"{verdict.alpha:.2f} ({counts})."
        )
    elif verdict.combined_decision == "DifferenceOnSome":
        body = (
            f"**not equivalent on all tasks** — at least one task shows a "
            f"≥ {delta_pp:.0f}pp difference ({counts})."
        )
    else:
        body = f"**not established** — {counts}."
    return f"{head} {body}"


def write_per_task_report(
    result,
    out_path: Path,
    confidence: float = 0.95,
    delta_pp: float = 3.0,
) -> None:
    """Write the per-task equivalence report for a (model, dataset) pair."""
    model_name = MODEL_DISPLAY.get(result.spec.model, result.spec.model)
    dataset_label = "LIBERO+" if result.spec.dataset == "liberoplus" else "LIBERO"
    ci_pct = f"{confidence:.0%}"
    alpha = getattr(result, "tost_alpha", 0.05)

    lines: list[str] = []
    lines.append(
        f"# {model_name} — {dataset_label}: per-task equivalence "
        f"{result.spec.variant_low}-step vs {result.spec.variant_high}-step"
    )
    lines.append("")
    lines.append(
        "> **Single-seed caveat**: results come from a single training run. "
        "Uncertainty estimates characterise *evaluation* stochasticity only."
    )
    lines.append("")
    lines.append(
        f"**Test.** Per-task paired TOST against δ = {delta_pp:.0f}pp at α="
        f"{alpha:.2f} (90% CI ⊂ ±δ ⇒ equivalent). **Multitask framing:** the "
        f"combined claim 'every task is equivalent within ±{delta_pp:.0f}pp' is an "
        "intersection–union test, so each per-task TOST runs at the full α with "
        "**no Bonferroni** and the conjunction is still a level-α test. Per-task "
        "samples are small, so most tasks are expected to be *inconclusive* at "
        f"±{delta_pp:.0f}pp; the pooled suite / Overall rows in the per-suite "
        "report are where equivalence is demonstrable. **Paired data only.**"
    )

    if getattr(result, "notes", None):
        lines.append("")
        lines.append("> **Data-loading notes:**")
        for note in result.notes:
            lines.append(f">  - {note}")

    low_hdr = f"{result.spec.variant_low}-step ({ci_pct} CI)"
    high_hdr = f"{result.spec.variant_high}-step ({ci_pct} CI)"

    for suite in SUITE_ORDER:
        if suite not in result.suite_tasks:
            continue
        cells = result.suite_tasks[suite]
        lines.append("")
        lines.append(f"## {SUITE_DISPLAY.get(suite, suite)}")
        lines.append("")
        verdict = result.combined_equiv.get(suite)
        if verdict is not None:
            lines.append(_fmt_combined(
                verdict, result.spec.variant_low, result.spec.variant_high, delta_pp
            ))
            lines.append("")
        lines.append(
            f"| Task | {low_hdr} | {high_hdr} | Δ pp | 90% CI (pp) "
            "| n_paired | verdict |"
        )
        lines.append("| --- | --- | --- | ---: | :---: | ---: | :---: |")
        for cell in cells:
            sr_lo = _fmt_sr(cell.counts_low, cell.wilson_low)
            sr_hi = _fmt_sr(cell.counts_high, cell.wilson_high)
            tost = getattr(cell, "tost", None)
            if tost is None:
                lines.append(
                    f"| {cell.suite} | {sr_lo} | {sr_hi} | {_NOT_ASSESSED} "
                    f"| — | — | — |"
                )
                continue
            lines.append(
                f"| {cell.suite} | {sr_lo} | {sr_hi} "
                f"| {tost.delta_hat * 100:+.1f} | {_fmt_ci(tost)} "
                f"| {tost.n_paired} | {_VERDICT_DISPLAY[tost.verdict]} |"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
