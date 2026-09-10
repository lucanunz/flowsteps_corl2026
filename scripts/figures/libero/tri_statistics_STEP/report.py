"""Markdown reports for the STEP variant.

Per-pair report keeps the unpaired z/Fisher table (so the file is directly
comparable to the McNemar report) and adds a STEP paired-test section that
shows the decision at α_global and at α_split, plus the early-stopping time.
"""

from __future__ import annotations

from pathlib import Path

from .config import MODEL_DISPLAY, SUITE_DISPLAY, SUITE_ORDER
from .pairs import PairResult, PerTaskPairResult, cells_in_order


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _fmt_alpha(a: float | None) -> str:
    if a is None:
        return "—"
    return f"{a:.4f}"


def _fmt_sr(counts, wilson) -> str:
    return (
        f"{counts.rate * 100:.1f}%"
        f" [{wilson[0] * 100:.1f}, {wilson[1] * 100:.1f}]"
        f" ({counts.successes}/{counts.trials})"
    )


_DECISION_DISPLAY = {
    "AcceptAlternative": "high > low",
    "AcceptNull":        "low > high",
    "FailToDecide":      "no separation",
}


def _fmt_decision(name: str | None) -> str:
    if name is None:
        return "—"
    return _DECISION_DISPLAY.get(name, name)


def _fmt_combined_verdict(verdict, variant_low: int, variant_high: int) -> str:
    """One-line summary line for the per-suite combined multitask verdict."""
    K = verdict.family_size
    a_s = verdict.alpha_split
    a_g = verdict.alpha_global
    counts = (
        f"{verdict.n_alt_alpha_s}/{K} toward {variant_high}-step, "
        f"{verdict.n_null_alpha_s}/{K} toward {variant_low}-step, "
        f"{verdict.n_undecided_alpha_s}/{K} undecided"
    )
    if verdict.n_z_fallback:
        counts += f", {verdict.n_z_fallback}/{K} z-fallback (no paired data)"

    if not verdict.correction_applied:
        head = (
            f"Combined multitask verdict (no correction; K={K}, α_s=α_g={a_g:.4f}):"
        )
    else:
        head = (
            f"Combined multitask verdict (Bonferroni, K={K}, "
            f"α_s={a_s:.4f}, α_g={a_g:.4f}):"
        )

    decision = verdict.combined_decision
    if decision == "AlternativeOnAll":
        body = (
            f"**{variant_high}-step uniformly better at α={a_g:.4f}** — every "
            f"task rejected H0 toward {variant_high}-step ({counts})."
        )
    elif decision == "NullOnAll":
        body = (
            f"**{variant_low}-step uniformly better at α={a_g:.4f}** — every "
            f"task rejected H0 toward {variant_low}-step ({counts})."
        )
    elif decision == "Mixed":
        body = (
            f"**not confirmed (mixed direction)** — per-task subtests "
            f"contradict each other ({counts})."
        )
    else:  # NotConfirmed
        body = f"**not confirmed** — {counts}."

    return f"{head} {body}"


def write_pair_report(
    pair: PairResult,
    out_path: Path,
    confidence: float = 0.95,
    correction_family: str = "pair",
) -> None:
    """Write a markdown report for a single (model, dataset) 1-step vs max-step comparison."""
    model_name = MODEL_DISPLAY.get(pair.spec.model, pair.spec.model)
    dataset_label = "LIBERO+" if pair.spec.dataset == "liberoplus" else "LIBERO"
    ci_pct = f"{confidence:.0%}"
    alpha = 1.0 - confidence
    cells = cells_in_order(pair)
    family_size = max(len(cells), 1)
    no_correction = correction_family == "none"
    alpha_split = alpha if no_correction else alpha / family_size

    lines: list[str] = []
    lines.append(
        f"# {model_name} — {dataset_label}: "
        f"{pair.spec.variant_low}-step vs {pair.spec.variant_high}-step"
    )
    lines.append("")
    lines.append(
        "> **Single-seed caveat**: results come from a single training run. "
        "Uncertainty estimates characterise *evaluation* stochasticity only, "
        "not training stochasticity."
    )

    if pair.notes:
        lines.append("")
        lines.append("> **Data-loading notes:**")
        for note in pair.notes:
            lines.append(f">  - {note}")

    lines.append("")
    if no_correction:
        lines.append(
            f"**CLD basis:** {pair.cld_basis}. "
            "**No multiplicity correction** applied: α_split = α_global = "
            f"{alpha:.4f}, so 'corrected' decisions equal uncorrected ones. "
            "Same CLD letter = methods not distinguishable at α_global."
        )
    else:
        lines.append(
            f"**CLD basis:** {pair.cld_basis}. "
            f"STEP cells use the α-split decision at α = α_global / {family_size} = "
            f"{alpha_split:.4f}; z-fallback cells use Holm-corrected p-values. "
            "Same CLD letter = methods not distinguishable at the corrected threshold."
        )

    # ── Unpaired tests table ──────────────────────────────────────────────────
    lines.append("")
    lines.append("## Unpaired tests")
    lines.append("")
    if no_correction:
        lines.append(
            "Correction family: **none** (α_split = α_global). "
            f"Confidence intervals: {ci_pct} Wilson score. "
            "The CLD column reflects the *primary* test (paired STEP where "
            "per-episode data permits, unpaired z otherwise); the q_* columns in "
            "this table are from the unpaired z-test and are shown for diagnostic "
            "purposes only — they do not drive the CLD."
        )
    else:
        lines.append(
            f"Correction family: {family_size} hypotheses (4 suites + Overall). "
            f"Confidence intervals: {ci_pct} Wilson score. "
            "The CLD column reflects the *primary* test (paired STEP where "
            "per-episode data permits, unpaired z otherwise); the q_* columns in "
            "this table are from the unpaired z-test."
        )
    lines.append("")

    low_hdr = f"{pair.spec.variant_low}-step ({ci_pct} CI)"
    high_hdr = f"{pair.spec.variant_high}-step ({ci_pct} CI)"
    lines.append(
        f"| Suite | {low_hdr} | {high_hdr}"
        " | Δ pp | p_z | p_Fisher | q_Bonf | q_Holm | q_BH | CLD |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")

    for cell in cells:
        suite_label = SUITE_DISPLAY.get(cell.suite, cell.suite)
        delta = (cell.counts_high.rate - cell.counts_low.rate) * 100
        sr_low = _fmt_sr(cell.counts_low, cell.wilson_low)
        sr_high = _fmt_sr(cell.counts_high, cell.wilson_high)
        q_bonf_str = _fmt_p(cell.q_bonf_z)
        q_holm_str = _fmt_p(cell.q_holm_z)
        q_bh_str = _fmt_p(cell.q_bh_z)
        cld_str = f"{cell.cld_low_corrected} / {cell.cld_high_corrected}"
        lines.append(
            f"| {suite_label} | {sr_low} | {sr_high} | {delta:+.1f}"
            f" | {_fmt_p(cell.p_z)} | {_fmt_p(cell.p_fisher)}"
            f" | {q_bonf_str} | {q_holm_str} | {q_bh_str} | {cld_str} |"
        )

    # ── Paired STEP table ─────────────────────────────────────────────────────
    has_paired = any(c.primary_test == "step" for c in cells)
    lines.append("")

    if has_paired:
        lines.append("## Paired tests (STEP)")
        lines.append("")
        if no_correction:
            lines.append(
                "Mirrored STEP (`sequentialized_barnard_tests`) on aligned per-rollout "
                "outcomes. Policy 0 = 1-step, Policy 1 = max-step. AcceptAlternative ⇒ "
                "max-step is significantly better; AcceptNull ⇒ 1-step is significantly "
                "better; FailToDecide ⇒ no separation under the rollout budget. "
                f"α_global = α_split = {alpha:.4f} (**no multiplicity correction**). "
                "n_max is the matched rollout count; stop_n is the step at which STEP "
                "made its decision (≤ n_max). The α_global and α_split columns are "
                "identical by construction."
            )
        else:
            lines.append(
                "Mirrored STEP (`sequentialized_barnard_tests`) on aligned per-rollout "
                "outcomes. Policy 0 = 1-step, Policy 1 = max-step. AcceptAlternative ⇒ "
                "max-step is significantly better; AcceptNull ⇒ 1-step is significantly "
                "better; FailToDecide ⇒ no separation under the rollout budget. "
                f"α_global = {alpha:.4f}; α_split = α_global / {family_size} = "
                f"{alpha_split:.4f} (Bonferroni). n_max is the matched rollout count; "
                "stop_n is the step at which STEP made its decision (≤ n_max). "
                "The CLD column reflects α_split."
            )
        lines.append("")
        lines.append(
            "| Suite | n_paired | b | c | n_max | decision (α_global) | stop_n | decision (α_split) | stop_n (split) | CLD (corr.) |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | :---: | ---: | :---: | ---: | :---: |"
        )
        for cell in cells:
            if cell.primary_test != "step":
                continue
            suite_label = SUITE_DISPLAY.get(cell.suite, cell.suite)
            cld_str = f"{cell.cld_low_corrected} / {cell.cld_high_corrected}"
            lines.append(
                f"| {suite_label} | {cell.n_paired}"
                f" | {cell.discordant_b} | {cell.discordant_c}"
                f" | {cell.step_n_max}"
                f" | {_fmt_decision(cell.step_decision_uncorrected)}"
                f" | {cell.step_stop_time_uncorrected}"
                f" | {_fmt_decision(cell.step_decision_corrected)}"
                f" | {cell.step_stop_time_corrected}"
                f" | {cld_str} |"
            )
    else:
        lines.append("## Paired tests")
        lines.append("")
        lines.append(
            "_Per-episode data not available for this model/dataset. "
            "Unpaired z-test results are reported above; STEP was not run._"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_task_report(
    result: PerTaskPairResult,
    out_path: Path,
    confidence: float = 0.95,
    correction_family: str = "within_suite",
) -> None:
    """Write a per-task markdown report for a (model, dataset) pair."""
    model_name = MODEL_DISPLAY.get(result.spec.model, result.spec.model)
    dataset_label = "LIBERO+" if result.spec.dataset == "liberoplus" else "LIBERO"
    ci_pct = f"{confidence:.0%}"
    alpha = 1.0 - confidence

    lines: list[str] = []
    lines.append(
        f"# {model_name} — {dataset_label}: per-task "
        f"{result.spec.variant_low}-step vs {result.spec.variant_high}-step"
    )
    lines.append("")
    lines.append(
        "> **Single-seed caveat**: results come from a single training run. "
        "Uncertainty estimates characterise *evaluation* stochasticity only."
    )

    if correction_family == "within_suite":
        fam_desc = "within each suite (10 tasks per family)"
    elif correction_family == "dataset":
        fam_desc = "across all suites in the dataset (up to 40 tasks per family)"
    elif correction_family == "none":
        fam_desc = "none (α_split = α_global)"
    else:
        fam_desc = correction_family
    lines.append("")
    if correction_family == "none":
        lines.append(
            f"**CLD basis:** {result.cld_basis}. "
            "**No multiplicity correction** applied: α_split = α_global = "
            f"{alpha:.4f}, so 'corrected' decisions and CLDs equal the "
            "uncorrected ones. Same CLD letter = tasks not distinguishable at α_global."
        )
    else:
        lines.append(
            f"**CLD basis:** {result.cld_basis}. "
            f"STEP cells use α-split = α_global / family_size; z-fallback cells "
            f"use Holm-corrected p-values. Correction family: {fam_desc}. "
            "Same CLD letter = methods not distinguishable at the corrected threshold."
        )

    # Multitask framing: explain how the per-suite combined verdict is built.
    lines.append("")
    lines.append(
        "**Multitask framing.** Per suite (K = number of tasks), we test "
        f"K hypotheses, one per task τ: H0,τ: p_{{{result.spec.variant_high}-step,τ}} ≤ "
        f"p_{{{result.spec.variant_low}-step,τ}} vs H1,τ: the reverse. By "
        "Bonferroni (union bound) we run each per-task STEP at α_s = α/K; the "
        "*combined* multitask claim H1: ∀τ p_high,τ > p_low,τ is confirmed at α "
        "iff every per-task STEP at α_s rejects H0,τ in the same direction. "
        "Each suite section below opens with this combined verdict and is then "
        "followed by the per-task table that drives it."
    )

    if result.notes:
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
        has_paired = any(c.primary_test == "step" for c in cells)
        lines.append("")
        lines.append(f"## {SUITE_DISPLAY.get(suite, suite)}")
        lines.append("")
        verdict = result.combined_verdicts.get(suite)
        if verdict is not None:
            lines.append(_fmt_combined_verdict(
                verdict,
                variant_low=result.spec.variant_low,
                variant_high=result.spec.variant_high,
            ))
            lines.append("")
        lines.append(
            f"| Task | {low_hdr} | {high_hdr}"
            " | Δ pp | p_z | p_Fisher | q_Holm | q_BH | CLD |"
            + (
                " STEP α_g | stop_n α_g | STEP α_s | stop_n α_s | b | c |"
                if has_paired else ""
            )
        )
        sep = "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |"
        sep += (" :---: | ---: | :---: | ---: | ---: | ---: |" if has_paired else "")
        lines.append(sep)

        for cell in cells:
            delta = (cell.counts_high.rate - cell.counts_low.rate) * 100
            sr_lo = _fmt_sr(cell.counts_low, cell.wilson_low)
            sr_hi = _fmt_sr(cell.counts_high, cell.wilson_high)
            q_holm = _fmt_p(cell.q_holm_z)
            q_bh = _fmt_p(cell.q_bh_z)
            cld = f"{cell.cld_low_corrected} / {cell.cld_high_corrected}"
            row = (
                f"| {cell.suite} | {sr_lo} | {sr_hi} | {delta:+.1f}"
                f" | {_fmt_p(cell.p_z)} | {_fmt_p(cell.p_fisher)}"
                f" | {q_holm} | {q_bh} | {cld} |"
            )
            if has_paired:
                if cell.primary_test == "step":
                    dec_g = _fmt_decision(cell.step_decision_uncorrected)
                    stop_g = (
                        str(cell.step_stop_time_uncorrected)
                        if cell.step_stop_time_uncorrected is not None else "—"
                    )
                    dec_s = _fmt_decision(cell.step_decision_corrected)
                    stop_s = (
                        str(cell.step_stop_time_corrected)
                        if cell.step_stop_time_corrected is not None else "—"
                    )
                else:
                    dec_g = dec_s = "—"
                    stop_g = stop_s = "—"
                b = str(cell.discordant_b) if cell.discordant_b is not None else "—"
                c = str(cell.discordant_c) if cell.discordant_c is not None else "—"
                row += f" {dec_g} | {stop_g} | {dec_s} | {stop_s} | {b} | {c} |"
            lines.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
