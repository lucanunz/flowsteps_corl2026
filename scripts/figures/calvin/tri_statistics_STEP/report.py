"""Markdown reports for the CALVIN STEP pipeline.

One report per (model, dataset) pair with three sections:
  1. Unpaired SR@5 z-test table (with Bonferroni / Holm / BH columns — these
     degenerate to p_z under family size 1 but stay for layout parity).
  2. Paired STEP table (only when paired pickle data is available).
  3. Completion-score (0-5) section: per-cell counts, Welch's t, Dirichlet
     posterior summary.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import CELL_DISPLAY, DATASET_DISPLAY, MODEL_DISPLAY
from .pairs import PairResult, cells_in_order
from .stats import dirichlet_posterior_mean_samples


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


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


def _stdev(variance: float) -> float:
    return math.sqrt(max(variance, 0.0))


def write_pair_report(pair: PairResult, out_path: Path, confidence: float = 0.95) -> None:
    """Write a per-model markdown report (SR@5 STEP/z + completion Welch/Dirichlet)."""
    model_name = MODEL_DISPLAY.get(pair.spec.model, pair.spec.model)
    dataset_label = DATASET_DISPLAY.get(pair.spec.dataset, pair.spec.dataset)
    ci_pct = f"{confidence:.0%}"
    alpha = 1.0 - confidence
    cells = cells_in_order(pair)
    family_size = max(len(cells), 1)
    alpha_split = alpha / family_size

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
    lines.append(
        f"**CLD basis (binary SR@5):** {pair.cld_basis}. "
        f"STEP cells use the α-split decision at α = α_global / {family_size} = "
        f"{alpha_split:.4f}; z-fallback cells use Holm-corrected p-values. "
        "Same CLD letter = methods not distinguishable at the corrected threshold."
    )

    # ── Unpaired tests table (SR@5) ───────────────────────────────────────────
    lines.append("")
    lines.append("## Binary SR@5 — unpaired tests")
    lines.append("")
    lines.append(
        f"Correction family: {family_size} hypothesis (SR@5 only). "
        f"Confidence intervals: {ci_pct} Wilson score. "
        "The CLD column reflects the *primary* test (paired STEP where "
        "per-sequence pickle data permits, unpaired z otherwise); the q_* "
        "columns are from the unpaired z-test."
    )
    lines.append("")

    low_hdr = f"{pair.spec.variant_low}-step ({ci_pct} CI)"
    high_hdr = f"{pair.spec.variant_high}-step ({ci_pct} CI)"
    lines.append(
        f"| Cell | {low_hdr} | {high_hdr}"
        " | Δ pp | p_z | q_Bonf | q_Holm | q_BH | CLD |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |")
    for cell in cells:
        cell_label = CELL_DISPLAY.get(cell.suite, cell.suite)
        delta = (cell.counts_high.rate - cell.counts_low.rate) * 100
        sr_low = _fmt_sr(cell.counts_low, cell.wilson_low)
        sr_high = _fmt_sr(cell.counts_high, cell.wilson_high)
        cld_str = f"{cell.cld_low_corrected} / {cell.cld_high_corrected}"
        lines.append(
            f"| {cell_label} | {sr_low} | {sr_high} | {delta:+.1f}"
            f" | {_fmt_p(cell.p_z)}"
            f" | {_fmt_p(cell.q_bonf_z)} | {_fmt_p(cell.q_holm_z)} | {_fmt_p(cell.q_bh_z)}"
            f" | {cld_str} |"
        )

    # ── Paired STEP table ─────────────────────────────────────────────────────
    has_paired = any(c.primary_test == "step" for c in cells)
    lines.append("")
    if has_paired:
        lines.append("## Binary SR@5 — paired tests (STEP)")
        lines.append("")
        lines.append(
            "Mirrored STEP (`sequentialized_barnard_tests`) on aligned per-sequence "
            "outcomes. Policy 0 = 1-step, Policy 1 = N-step. AcceptAlternative ⇒ "
            "N-step is significantly better; AcceptNull ⇒ 1-step is significantly "
            "better; FailToDecide ⇒ no separation under the rollout budget. "
            f"α_global = {alpha:.4f}; α_split = α_global / {family_size} = "
            f"{alpha_split:.4f} (Bonferroni). n_max is the matched rollout count; "
            "stop_n is the step at which STEP made its decision (≤ n_max). "
            "The CLD column reflects α_split."
        )
        lines.append("")
        lines.append(
            "| Cell | n_paired | b | c | n_max | decision (α_global) | stop_n "
            "| decision (α_split) | stop_n (split) | CLD (corr.) |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | :---: | ---: | :---: | ---: | :---: |"
        )
        for cell in cells:
            if cell.primary_test != "step":
                continue
            cell_label = CELL_DISPLAY.get(cell.suite, cell.suite)
            cld_str = f"{cell.cld_low_corrected} / {cell.cld_high_corrected}"
            lines.append(
                f"| {cell_label} | {cell.n_paired}"
                f" | {cell.discordant_b} | {cell.discordant_c}"
                f" | {cell.step_n_max}"
                f" | {_fmt_decision(cell.step_decision_uncorrected)}"
                f" | {cell.step_stop_time_uncorrected}"
                f" | {_fmt_decision(cell.step_decision_corrected)}"
                f" | {cell.step_stop_time_corrected}"
                f" | {cld_str} |"
            )
    else:
        lines.append("## Binary SR@5 — paired tests")
        lines.append("")
        lines.append(
            "_Per-rollout data not available for this model. "
            "Unpaired z-test results are reported above; STEP was not run._"
        )

    # ── Completion-score (0-5) section ────────────────────────────────────────
    lines.append("")
    lines.append("## Task completion score (0-5)")
    lines.append("")
    lines.append(
        "Per-rollout completed-subtask score is treated as a discrete random "
        "variable. We compare the 1-step baseline against the N-step reference "
        "using Welch's t-test on the per-rollout score and translate the result "
        "into a CLD per model at the 0.05 level. For models that expose "
        "per-sequence pickles we use the raw counts; otherwise the count vector "
        "(n_0, …, n_5) is reconstructed from `chain_sr` via "
        "`chain_sr[k] - chain_sr[k+1]`."
    )
    lines.append("")
    lines.append("### Completion distribution per cell")
    lines.append("")
    lines.append(
        "| Steps | n | Mean | SD | n_0 | n_1 | n_2 | n_3 | n_4 | n_5 | Source |"
    )
    lines.append(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    cell = cells[0]
    assert cell.completion_low is not None and cell.completion_high is not None
    for step_label, cs, source in (
        (str(pair.spec.variant_low), cell.completion_low, cell.completion_source_low),
        (str(pair.spec.variant_high), cell.completion_high, cell.completion_source_high),
    ):
        counts = cs.counts
        lines.append(
            f"| {step_label} | {cs.n} | {cs.mean:.3f} | {_stdev(cs.variance):.3f}"
            f" | {counts[0]} | {counts[1]} | {counts[2]} | {counts[3]} | {counts[4]} | {counts[5]}"
            f" | {source} |"
        )

    lines.append("")
    lines.append("### Welch's t-test (1-step vs N-step)")
    lines.append("")
    lines.append(
        "| Δ mean | 95% CI | t | df | p | CLD (1-step / N-step) |"
    )
    lines.append("| ---: | --- | ---: | ---: | ---: | :---: |")
    welch = cell.welch
    if welch is not None:
        cld_text = f"{cell.welch_cld_low} / {cell.welch_cld_high}"
        ci_text = f"[{welch.ci_low:+.3f}, {welch.ci_high:+.3f}]"
        lines.append(
            f"| {welch.delta:+.3f} | {ci_text} | {welch.t_statistic:+.2f}"
            f" | {welch.df:.1f} | {_fmt_p(welch.p_value)} | {cld_text} |"
        )

    lines.append("")
    lines.append("### Dirichlet posterior on mean completion")
    lines.append("")
    lines.append(
        "For each cell, we draw 20,000 samples from the Dirichlet(1+n_0, …, 1+n_5) "
        "posterior over the categorical distribution and compute the implied "
        "mean completion E[k] = Σ k · p_k. Posterior median and 95% credible "
        "interval reported below."
    )
    lines.append("")
    lines.append(
        "| Steps | Empirical mean | Posterior median | 95% credible interval |"
    )
    lines.append("| ---: | ---: | ---: | --- |")
    for step_label, cs, seed in (
        (str(pair.spec.variant_low), cell.completion_low, 0),
        (str(pair.spec.variant_high), cell.completion_high, 1),
    ):
        samples = dirichlet_posterior_mean_samples(cs.counts, seed=seed)
        median = float(np.median(samples))
        lower = float(np.quantile(samples, 0.025))
        upper = float(np.quantile(samples, 0.975))
        lines.append(
            f"| {step_label} | {cs.mean:.3f} | {median:.3f}"
            f" | [{lower:.3f}, {upper:.3f}] |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
