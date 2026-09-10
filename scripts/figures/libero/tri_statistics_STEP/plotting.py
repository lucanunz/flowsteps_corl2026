"""Matplotlib plotting for the STEP variant.

Mirrors `tri_statistics.plotting` but:
  - CLD letters above each pair come from `cell.cld_*_corrected` (STEP α-split
    decision when paired data is available, Holm-corrected z otherwise).
  - Significance markers reflect a two-tier STEP outcome: "**" when STEP
    separated the policies at α_split (survives Bonferroni); "*" when STEP
    only separated them at α_global; "" otherwise. z-fallback cells keep the
    classic asterisk tiers from Holm-q.
  - Per-task heatmaps annotate corrected STEP separations with "**".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from .config import ALL_CELLS, MODEL_DISPLAY, OVERALL_KEY, SUITE_DISPLAY, SUITE_ORDER
from .pairs import CellStats, PairResult, PerTaskPairResult
from .stats import beta_posterior_samples


_VIOLIN_COLORS = ("#4C72B0", "#DD8452")  # low, high
_SEPARATION_DECISIONS = {"AcceptAlternative", "AcceptNull"}


def _cell_marker(cell: CellStats, *, no_correction: bool = False) -> str:
    """Two-tier significance marker for plot annotations.

    For STEP cells:
        "**" if α_split decision separates the policies (Bonferroni-corrected)
        "*"  if only α_global decision separates them (uncorrected)
        ""   otherwise

    For z-fallback cells: the classic */**/*** tiers from Holm-q on the z-test.

    When `no_correction=True`, asterisk markers are suppressed (they would be
    meaningless because α_split = α_global and Holm-q is not the basis for the CLD).
    """
    if no_correction:
        return ""
    if cell.primary_test == "step":
        if cell.step_decision_corrected in _SEPARATION_DECISIONS:
            return "**"
        if cell.step_decision_uncorrected in _SEPARATION_DECISIONS:
            return "*"
        return ""
    q = cell.q_holm_z
    if q is None:
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def _is_no_corr(obj) -> bool:
    return getattr(obj, "correction_family", "pair") == "none"


def _fmt_variant(prefix: str, steps: int) -> str:
    return f"{prefix}{steps}"


def plot_pair_violins(pair: PairResult, out_path: Path, n_samples: int = 4000) -> None:
    cells = [pair.cells[k] for k in ALL_CELLS if k in pair.cells]
    n = len(cells)
    no_corr = _is_no_corr(pair)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 4.6), sharey=True)
    if n == 1:
        axes = [axes]
    rng = np.random.default_rng(0)

    label_low = _fmt_variant("step ", pair.spec.variant_low)
    label_high = _fmt_variant("step ", pair.spec.variant_high)

    for ax, cell in zip(axes, cells):
        s_low = beta_posterior_samples(cell.counts_low.successes, cell.counts_low.trials, n_samples, rng)
        s_high = beta_posterior_samples(cell.counts_high.successes, cell.counts_high.trials, n_samples, rng)

        parts = ax.violinplot(
            [s_low, s_high],
            positions=[0, 1],
            widths=0.7,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )
        for body, color in zip(parts["bodies"], _VIOLIN_COLORS):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.55)

        for pos, c in ((0, cell.counts_low), (1, cell.counts_high)):
            ax.plot(
                pos,
                c.rate,
                marker="_",
                linestyle="None",
                color="black",
                markeredgewidth=0.8,
                markersize=6,
                zorder=3,
            )

        ax.text(0, 1.03, cell.cld_low_corrected, ha="center", va="bottom",
                fontsize=11, fontweight="bold", transform=ax.get_xaxis_transform())
        ax.text(1, 1.03, cell.cld_high_corrected, ha="center", va="bottom",
                fontsize=11, fontweight="bold", transform=ax.get_xaxis_transform())

        marker = _cell_marker(cell, no_correction=no_corr)
        title_suffix = f"  {marker}" if marker else ""
        ax.set_title(f"{SUITE_DISPLAY.get(cell.suite, cell.suite)}{title_suffix}", fontsize=10)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([label_low, label_high], fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if cell is cells[0]:
            ax.set_ylabel("Success rate (Beta posterior)")

    if no_corr:
        subtitle = f"CLD from {pair.cld_basis} (no multiplicity correction)"
    else:
        subtitle = (
            f"CLD from {pair.cld_basis} (α-split = 0.05 / family).  "
            "** = STEP at α_split,  * = STEP at α_global only"
        )
    fig.suptitle(
        f"{MODEL_DISPLAY.get(pair.spec.model, pair.spec.model)} — "
        f"{pair.spec.dataset.upper()}: "
        f"{pair.spec.variant_low}-step vs {pair.spec.variant_high}-step\n"
        f"{subtitle}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pair_bars(pair: PairResult, out_path: Path) -> None:
    cells = [pair.cells[k] for k in ALL_CELLS if k in pair.cells]
    n = len(cells)
    no_corr = _is_no_corr(pair)
    fig, ax = plt.subplots(figsize=(1.6 * n + 1.2, 4.4))

    labels = [SUITE_DISPLAY.get(c.suite, c.suite) for c in cells]
    x = np.arange(n)
    width = 0.36
    rates_low = [c.counts_low.rate for c in cells]
    rates_high = [c.counts_high.rate for c in cells]
    err_low = np.clip(np.array([
        [c.counts_low.rate - c.wilson_low[0] for c in cells],
        [c.wilson_low[1] - c.counts_low.rate for c in cells],
    ]), 0.0, None)
    err_high = np.clip(np.array([
        [c.counts_high.rate - c.wilson_high[0] for c in cells],
        [c.wilson_high[1] - c.counts_high.rate for c in cells],
    ]), 0.0, None)

    label_low = f"{pair.spec.variant_low}-step"
    label_high = f"{pair.spec.variant_high}-step"
    ax.bar(x - width / 2, rates_low, width, color=_VIOLIN_COLORS[0],
           edgecolor="black", label=label_low,
           yerr=err_low, capsize=3, error_kw={"lw": 0.8})
    ax.bar(x + width / 2, rates_high, width, color=_VIOLIN_COLORS[1],
           edgecolor="black", label=label_high,
           yerr=err_high, capsize=3, error_kw={"lw": 0.8})

    for i, cell in enumerate(cells):
        marker = _cell_marker(cell, no_correction=no_corr)
        if marker:
            top = max(cell.wilson_low[1], cell.wilson_high[1])
            ax.text(i, min(1.02, top + 0.03), marker, ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Success rate")
    if no_corr:
        ax_subtitle = f"(no multiplicity correction;  {pair.cld_basis})"
    else:
        ax_subtitle = f"(** = STEP α_split,  * = STEP α_global only;  {pair.cld_basis})"
    ax.set_title(
        f"{MODEL_DISPLAY.get(pair.spec.model, pair.spec.model)} — "
        f"{pair.spec.dataset.upper()}\n"
        f"{ax_subtitle}"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dataset_overall_comparison(
    results: list[PairResult], out_path: Path, dataset: str
) -> None:
    """Side-by-side bar chart of all models' overall success rate (1-step vs max-step)."""
    fig, ax = plt.subplots(figsize=(1.4 * len(results) + 1.5, 4.4))
    x = np.arange(len(results))
    width = 0.36

    rates_low = []
    rates_high = []
    err_low_lo, err_low_hi = [], []
    err_high_lo, err_high_hi = [], []
    labels = []
    markers = []
    test_kinds: set[str] = set()
    no_corr = any(_is_no_corr(p) for p in results)
    for pair in results:
        cell = pair.cells[OVERALL_KEY]
        rates_low.append(cell.counts_low.rate)
        rates_high.append(cell.counts_high.rate)
        err_low_lo.append(max(0.0, cell.counts_low.rate - cell.wilson_low[0]))
        err_low_hi.append(max(0.0, cell.wilson_low[1] - cell.counts_low.rate))
        test_kinds.add(cell.primary_test)
        err_high_lo.append(max(0.0, cell.counts_high.rate - cell.wilson_high[0]))
        err_high_hi.append(max(0.0, cell.wilson_high[1] - cell.counts_high.rate))
        labels.append(MODEL_DISPLAY.get(pair.spec.model, pair.spec.model))
        markers.append(_cell_marker(cell, no_correction=_is_no_corr(pair)))

    ax.bar(x - width / 2, rates_low, width, color=_VIOLIN_COLORS[0],
           edgecolor="black", label="1-step",
           yerr=[err_low_lo, err_low_hi], capsize=3, error_kw={"lw": 0.8})
    ax.bar(x + width / 2, rates_high, width, color=_VIOLIN_COLORS[1],
           edgecolor="black", label="max-step",
           yerr=[err_high_lo, err_high_hi], capsize=3, error_kw={"lw": 0.8})

    for i, mark in enumerate(markers):
        if mark:
            top = max(rates_low[i] + err_low_hi[i], rates_high[i] + err_high_hi[i])
            ax.text(i, min(1.02, top + 0.03), mark, ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Overall success rate")
    if test_kinds == {"step"}:
        basis = "paired STEP"
    elif test_kinds == {"z"}:
        basis = "unpaired z"
    else:
        basis = "STEP where available, z otherwise"
    if no_corr:
        title_suffix = f"(no multiplicity correction; basis: {basis})"
    else:
        title_suffix = f"(** = α_split,  * = α_global only; basis: {basis})"
    ax.set_title(
        f"Overall success — {dataset.upper()} (1-step vs max-step per model)\n"
        f"{title_suffix}"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-task figures
# ---------------------------------------------------------------------------

def _wrap_task_label(name: str, max_chars: int = 28) -> str:
    words = name.split()
    lines: list[str] = []
    current = ""
    for w in words:
        candidate = f"{current} {w}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = w
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def plot_per_task_violins(result: PerTaskPairResult, out_path: Path, n_samples: int = 2000) -> None:
    suites = [s for s in SUITE_ORDER if s in result.suite_tasks]
    if not suites:
        return
    no_corr = _is_no_corr(result)
    n_rows = max(len(result.suite_tasks[s]) for s in suites)
    n_cols = len(suites)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols, 2.2 * n_rows),
        squeeze=False,
    )
    rng = np.random.default_rng(0)
    label_low = f"{result.spec.variant_low}-step"
    label_high = f"{result.spec.variant_high}-step"

    for col_idx, suite in enumerate(suites):
        cells = result.suite_tasks[suite]
        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]
            if row_idx >= len(cells):
                ax.set_visible(False)
                continue
            cell = cells[row_idx]
            task_idx = cell.__dict__.get("task_idx", row_idx)

            s_lo = beta_posterior_samples(cell.counts_low.successes, cell.counts_low.trials, n_samples, rng)
            s_hi = beta_posterior_samples(cell.counts_high.successes, cell.counts_high.trials, n_samples, rng)
            parts = ax.violinplot([s_lo, s_hi], positions=[0, 1], widths=0.7,
                                  showmeans=False, showextrema=False)
            for body, color in zip(parts["bodies"], _VIOLIN_COLORS):
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_alpha(0.55)

            for pos, c in ((0, cell.counts_low), (1, cell.counts_high)):
                ax.plot(pos, c.rate, marker="_", linestyle="None",
                        color="black", markeredgewidth=0.6,
                        markersize=4, zorder=3)

            marker = _cell_marker(cell, no_correction=no_corr)
            cld_str = f"{cell.cld_low_corrected}/{cell.cld_high_corrected}"
            if marker:
                cld_str = f"{cell.cld_low_corrected}/{cell.cld_high_corrected} {marker}"
            ax.set_title(f"task {task_idx}  {cld_str}", fontsize=7, pad=2)
            ax.set_ylim(0, 1.05)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["1-step", f"{result.spec.variant_high}-step"], fontsize=6)
            ax.tick_params(axis="y", labelsize=6)
            ax.grid(axis="y", linestyle=":", alpha=0.3)

            if col_idx == 0:
                ax.set_ylabel("SR", fontsize=6)

        first = cells[0]
        first_marker = _cell_marker(first, no_correction=no_corr)
        axes[0, col_idx].set_title(
            f"{SUITE_DISPLAY.get(suite, suite)}\ntask {first.__dict__.get('task_idx', 0)}  "
            f"{first.cld_low_corrected}/{first.cld_high_corrected}"
            + (f" {first_marker}" if first_marker else ""),
            fontsize=7, pad=2,
        )

    cld_descriptor = (
        f"CLD from {result.cld_basis} (no multiplicity correction)."
        if no_corr
        else f"CLD from {result.cld_basis} (α-split per family). "
    )
    fig.suptitle(
        f"{MODEL_DISPLAY.get(result.spec.model, result.spec.model)} — "
        f"{result.spec.dataset.upper()}: per-task Beta posteriors\n"
        f"{cld_descriptor} "
        f"| blue = {label_low}  orange = {label_high}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_task_paper_violins(
    result: PerTaskPairResult,
    out_path: Path,
    n_samples: int = 2000,
) -> None:
    """Paper-style grouped per-task posterior violins (STEP-aware annotations)."""
    suites = [s for s in SUITE_ORDER if s in result.suite_tasks]
    if not suites:
        return
    no_corr = _is_no_corr(result)

    n_suites = len(suites)
    fig, axes = plt.subplots(
        1, n_suites,
        figsize=(4.6 * n_suites, 4.4),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    rng = np.random.default_rng(0)
    label_low = f"{result.spec.variant_low}-step"
    label_high = f"{result.spec.variant_high}-step"
    offsets = (-0.18, 0.18)
    width = 0.32

    for col_idx, (ax, suite) in enumerate(zip(axes, suites)):
        cells = result.suite_tasks[suite]
        x = np.arange(len(cells), dtype=float)

        for task_pos, cell in zip(x, cells):
            s_low = beta_posterior_samples(
                cell.counts_low.successes, cell.counts_low.trials, n_samples, rng,
            )
            s_high = beta_posterior_samples(
                cell.counts_high.successes, cell.counts_high.trials, n_samples, rng,
            )
            positions = [task_pos + offsets[0], task_pos + offsets[1]]
            parts = ax.violinplot(
                [s_low, s_high], positions=positions, widths=width,
                showmeans=False, showextrema=False, showmedians=False,
            )
            for body, color in zip(parts["bodies"], _VIOLIN_COLORS):
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_linewidth(0.7)
                body.set_alpha(0.65)

            top_low = float(parts["bodies"][0].get_paths()[0].vertices[:, 1].max())
            top_high = float(parts["bodies"][1].get_paths()[0].vertices[:, 1].max())

            for pos, c in ((positions[0], cell.counts_low), (positions[1], cell.counts_high)):
                ax.plot(
                    pos, c.rate, marker="_", linestyle="None",
                    color="black", markeredgewidth=0.6,
                    markersize=4, zorder=3,
                )

            ax.text(
                positions[0], min(top_low + 0.035, 1.13), cell.cld_low_corrected,
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )
            ax.text(
                positions[1], min(top_high + 0.035, 1.13), cell.cld_high_corrected,
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )
            marker = _cell_marker(cell, no_correction=no_corr)
            if marker:
                ax.text(
                    task_pos, min(max(top_low, top_high) + 0.085, 1.16), marker,
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

        task_labels = [str(c.__dict__.get("task_idx", i)) for i, c in enumerate(cells)]
        ax.set_title(SUITE_DISPLAY.get(suite, suite), fontsize=11, pad=8)
        ax.set_xlim(-0.7, len(cells) - 0.3)
        ax.set_ylim(0, 1.18)
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, fontsize=8)
        ax.set_xlabel("Task ID", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_axisbelow(True)
        for boundary in np.arange(0.5, len(cells) - 0.5, 1.0):
            ax.axvline(boundary, color="0.85", lw=0.6, ls=":")
        if col_idx == 0:
            ax.set_ylabel("Success rate", fontsize=10)
        else:
            ax.tick_params(axis="y", labelleft=False)

    dataset_label = "LIBERO+" if result.spec.dataset == "liberoplus" else result.spec.dataset.upper()
    model_label = MODEL_DISPLAY.get(result.spec.model, result.spec.model)
    fig.legend(
        handles=[
            Patch(facecolor=_VIOLIN_COLORS[0], edgecolor="black", alpha=0.65, label=label_low),
            Patch(facecolor=_VIOLIN_COLORS[1], edgecolor="black", alpha=0.65, label=label_high),
        ],
        loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.91),
    )
    fig.suptitle(
        f"{model_label} — {dataset_label}: per-task Beta posteriors", fontsize=13, y=0.99,
    )
    if no_corr:
        caption = (
            f"CLD from {result.cld_basis} (no multiplicity correction). "
            "Horizontal lines mark the observed success rate."
        )
    else:
        caption = (
            f"CLD from {result.cld_basis} (α-split per family). "
            "** = STEP separated at α_split; * = STEP separated at α_global only. "
            "Horizontal lines mark the observed success rate."
        )
    fig.text(
        0.5, 0.84,
        caption,
        ha="center", va="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- Appendix per-task violin canvas (width-driven, inch-locked) -----------
# The figure is authored at the user's column width (``target_width_in``) so a
# plain \includegraphics[width=\linewidth] applies no scaling and every label
# renders at its true point size. Each suite panel is then placed at an explicit
# inch position derived from that width, so all panels are exactly equal-sized
# and every (model, dataset) PDF is byte-identical, scaling consistently in
# LaTeX. Mirrors the locked-canvas approach in
# analyse_flow_ablation_perfamily.py:panel_combined.
_APX_LEFT_PAD = 0.55   # y-label + y-tick labels (left column)
_APX_RIGHT_PAD = 0.10
_APX_COL_GAP = 0.30    # gap between adjacent columns
_APX_TOP_PAD = 0.28    # top-row title clearance
_APX_ROW_GAP = 0.55    # upper-row xlabel + lower-row title
_APX_BOTTOM_PAD = 0.72  # bottom-row xlabel + shared legend strip

# (n_cols, panel aspect ratio ax_w/ax_h). Panel width is solved from the figure
# width; height follows from the aspect so panels keep a fixed shape per layout.
_APX_LAYOUTS = {
    "2x2": (2, 1.45),   # wide-ish panels, two rows
    "1x4": (4, 0.92),   # all four suites in a single wide row
}

# On-page font sizes (points). Because the figure is authored at the column
# width, these are the sizes the reader actually sees.
_APX_FS_TITLE = 10
_APX_FS_LABEL = 9
_APX_FS_TICK = 8
_APX_FS_ANNOT = 8     # CLD letters + significance markers
_APX_FS_LEGEND = 8


def _draw_appendix_suite_panel(ax, cells, *, no_corr, rng, n_samples) -> None:
    """Draw one suite's per-task 1-step vs max-step violins on ``ax``.

    Renders the violin pair, the observed-rate lines, CLD letters, and the two-tier
    */** STEP marker per task, then fixes the x/y limits, task-ID ticks, grid,
    and inter-task separators. Titles, axis labels, and tick-label visibility
    are left to the caller — they differ between the single-dataset and
    combined (suites×datasets) layouts.
    """
    offsets = (-0.18, 0.18)
    width = 0.32
    x = np.arange(len(cells), dtype=float)

    for task_pos, cell in zip(x, cells):
        s_low = beta_posterior_samples(
            cell.counts_low.successes, cell.counts_low.trials, n_samples, rng,
        )
        s_high = beta_posterior_samples(
            cell.counts_high.successes, cell.counts_high.trials, n_samples, rng,
        )
        positions = [task_pos + offsets[0], task_pos + offsets[1]]
        parts = ax.violinplot(
            [s_low, s_high], positions=positions, widths=width,
            showmeans=False, showextrema=False, showmedians=False,
        )
        for body, color in zip(parts["bodies"], _VIOLIN_COLORS):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_linewidth(0.7)
            body.set_alpha(0.65)

        top_low = float(parts["bodies"][0].get_paths()[0].vertices[:, 1].max())
        top_high = float(parts["bodies"][1].get_paths()[0].vertices[:, 1].max())

        for pos, c in ((positions[0], cell.counts_low), (positions[1], cell.counts_high)):
            ax.plot(
                pos, c.rate, marker="_", linestyle="None",
                color="black", markeredgewidth=0.6,
                markersize=4, zorder=3,
            )

        ax.text(
            positions[0], min(top_low + 0.04, 1.13), cell.cld_low_corrected,
            ha="center", va="bottom", fontsize=_APX_FS_ANNOT, fontweight="bold",
        )
        ax.text(
            positions[1], min(top_high + 0.04, 1.13), cell.cld_high_corrected,
            ha="center", va="bottom", fontsize=_APX_FS_ANNOT, fontweight="bold",
        )
        marker = _cell_marker(cell, no_correction=no_corr)
        if marker:
            ax.text(
                task_pos, min(max(top_low, top_high) + 0.095, 1.16), marker,
                ha="center", va="bottom", fontsize=_APX_FS_ANNOT, fontweight="bold",
            )

    task_labels = [str(c.__dict__.get("task_idx", i)) for i, c in enumerate(cells)]
    ax.set_xlim(-0.7, len(cells) - 0.3)
    ax.set_ylim(0, 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=_APX_FS_TICK)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    for boundary in np.arange(0.5, len(cells) - 0.5, 1.0):
        ax.axvline(boundary, color="0.85", lw=0.6, ls=":")


def plot_per_task_paper_violins_appendix(
    result: PerTaskPairResult,
    out_path: Path,
    n_samples: int = 2000,
    layout: str = "2x2",
    target_width_in: float = 5.5,
) -> None:
    """Per-task paper violins for the appendix, sized to the column width.

    Same content as `plot_per_task_paper_violins` (per-task 1-step vs max-step
    Beta posteriors, CLD letters, and the two-tier */** STEP markers) but with
    no figure title/caption and a single shared legend.

    The canvas is authored at ``target_width_in`` inches wide so that
    ``\\includegraphics[width=\\linewidth]`` performs no scaling; panel widths
    are solved from that width and laid out per ``layout`` ('2x2' = two rows of
    two suites, '1x4' = one wide row of four). Every (model, dataset) PDF is
    byte-identical in size for a given (layout, width).
    """
    suites = [s for s in SUITE_ORDER if s in result.suite_tasks]
    if not suites:
        return
    no_corr = _is_no_corr(result)

    if layout not in _APX_LAYOUTS:
        raise ValueError(f"unknown appendix layout {layout!r}; choose from {sorted(_APX_LAYOUTS)}")
    n_cols_cfg, aspect = _APX_LAYOUTS[layout]
    n_suites = len(suites)
    n_cols = min(n_cols_cfg, n_suites)
    n_rows = -(-n_suites // n_cols)  # ceil

    # Solve panel geometry from the target column width.
    ax_w = (target_width_in - _APX_LEFT_PAD - _APX_RIGHT_PAD
            - (n_cols - 1) * _APX_COL_GAP) / n_cols
    ax_h = ax_w / aspect
    fig_w = target_width_in
    fig_h = (_APX_TOP_PAD + n_rows * ax_h
             + (n_rows - 1) * _APX_ROW_GAP + _APX_BOTTOM_PAD)
    fig = plt.figure(figsize=(fig_w, fig_h))

    rng = np.random.default_rng(0)
    label_low = f"{result.spec.variant_low}-step"
    label_high = f"{result.spec.variant_high}-step"

    for idx, suite in enumerate(suites):
        row, col = divmod(idx, n_cols)
        left_in = _APX_LEFT_PAD + col * (ax_w + _APX_COL_GAP)
        top_in = fig_h - _APX_TOP_PAD - row * (ax_h + _APX_ROW_GAP)
        bottom_in = top_in - ax_h
        ax = fig.add_axes((left_in / fig_w, bottom_in / fig_h,
                           ax_w / fig_w, ax_h / fig_h))

        _draw_appendix_suite_panel(
            ax, result.suite_tasks[suite],
            no_corr=no_corr, rng=rng, n_samples=n_samples,
        )
        ax.set_title(SUITE_DISPLAY.get(suite, suite), fontsize=_APX_FS_TITLE, pad=5)
        # Only the bottom row carries the "Task ID" label (the columns share it).
        if row == n_rows - 1:
            ax.set_xlabel("Task ID", fontsize=_APX_FS_LABEL)
        if col == 0:
            ax.set_ylabel("Success rate", fontsize=_APX_FS_LABEL)
            ax.tick_params(axis="y", labelsize=_APX_FS_TICK)
        else:
            ax.tick_params(axis="y", labelleft=False)

    # Single shared legend, centered in the bottom strip below the panels.
    fig.legend(
        handles=[
            Patch(facecolor=_VIOLIN_COLORS[0], edgecolor="black", alpha=0.65, label=label_low),
            Patch(facecolor=_VIOLIN_COLORS[1], edgecolor="black", alpha=0.65, label=label_high),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.10 / fig_h),
        ncol=2, frameon=False, fontsize=_APX_FS_LEGEND,
        handlelength=1.4, handletextpad=0.5, columnspacing=1.6,
    )

    # No bbox_inches='tight' so the canvas stays byte-identical across figures.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), dpi=200)  # companion for quick inspection
    plt.close(fig)


# --- Combined per-model appendix canvas (suites × datasets) ----------------
# One figure per model showing both datasets compactly: suites are rows,
# datasets are columns. Same width-driven, inch-locked authoring as the
# single-dataset appendix so it scales 1:1 at \linewidth.
# Left band, outer→inner: suite name (outer) | "Success rate" y-label | y-ticks.
_APX_CMB_LEFT_PAD = 1.48    # total left band width (in)
_APX_CMB_SUITE_LABEL_X = 0.55  # x of the (horizontal) suite row labels, in inches
                               # from the left edge (text is centered here) —
                               # lower = further out (left). Must be wide enough
                               # for the longest suite name ("LIBERO-10").
_APX_CMB_RIGHT_PAD = 0.10
_APX_CMB_COL_GAP = 0.3     # gap between dataset columns
_APX_CMB_TOP_PAD = 0.30     # top band height = column-header text + header gap
_APX_CMB_HEADER_GAP = 0.04  # distance from the LIBERO/LIBERO+ headers to the
                            # panels — decrease to push the titles closer down.
_APX_CMB_ROW_GAP = 0.30     # gap between suite rows
_APX_CMB_BOTTOM_PAD = 0.80  # bottom-row xlabel + shared legend strip
_APX_CMB_ASPECT = 1.95      # panel ax_w/ax_h (shorter panels → fits 4 rows)

_DATASET_DISPLAY = {"libero": "LIBERO", "liberoplus": "LIBERO+"}
_DATASET_ORDER = ("libero", "liberoplus")


def plot_per_task_appendix_combined(
    results: list[PerTaskPairResult],
    out_path: Path,
    n_samples: int = 2000,
    target_width_in: float = 5.5,
) -> None:
    """Compact per-model appendix figure with both datasets in one canvas.

    ``results`` are the per-dataset `PerTaskPairResult`s for a single model
    (typically LIBERO and LIBERO+). Suites become rows and datasets become
    columns, so all per-task detail is preserved while halving the appendix
    figure count. Datasets are labelled by column headers; suites by outer,
    horizontal row labels; "Success rate" is the inner left-column y-label.
    Authored at ``target_width_in`` inches wide like the single-dataset appendix.
    """
    if not results:
        return
    by_ds = {r.spec.dataset: r for r in results}
    datasets = ([d for d in _DATASET_ORDER if d in by_ds]
                + [d for d in by_ds if d not in _DATASET_ORDER])
    suites = [s for s in SUITE_ORDER
              if any(s in by_ds[d].suite_tasks for d in datasets)]
    if not suites or not datasets:
        return

    n_rows = len(suites)
    n_cols = len(datasets)
    ax_w = (target_width_in - _APX_CMB_LEFT_PAD - _APX_CMB_RIGHT_PAD
            - (n_cols - 1) * _APX_CMB_COL_GAP) / n_cols
    ax_h = ax_w / _APX_CMB_ASPECT
    fig_w = target_width_in
    fig_h = (_APX_CMB_TOP_PAD + n_rows * ax_h
             + (n_rows - 1) * _APX_CMB_ROW_GAP + _APX_CMB_BOTTOM_PAD)
    fig = plt.figure(figsize=(fig_w, fig_h))

    rng = np.random.default_rng(0)
    first = by_ds[datasets[0]]
    label_low = f"{first.spec.variant_low}-step"
    label_high = f"{first.spec.variant_high}-step"

    col_centers: list[float] = []
    for r_i, suite in enumerate(suites):
        for c_i, ds in enumerate(datasets):
            result = by_ds[ds]
            left_in = _APX_CMB_LEFT_PAD + c_i * (ax_w + _APX_CMB_COL_GAP)
            top_in = fig_h - _APX_CMB_TOP_PAD - r_i * (ax_h + _APX_CMB_ROW_GAP)
            bottom_in = top_in - ax_h
            ax = fig.add_axes((left_in / fig_w, bottom_in / fig_h,
                               ax_w / fig_w, ax_h / fig_h))
            if r_i == 0:
                col_centers.append((left_in + ax_w / 2) / fig_w)

            cells = result.suite_tasks.get(suite)
            if not cells:
                ax.set_axis_off()
                continue

            _draw_appendix_suite_panel(
                ax, cells, no_corr=_is_no_corr(result), rng=rng, n_samples=n_samples,
            )
            # x labels only on the bottom row (all rows share task IDs 0–9).
            if r_i == n_rows - 1:
                ax.set_xlabel("Task ID", fontsize=_APX_FS_LABEL)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            # "Success rate" sits on the inner side (left-column y-label); y-ticks
            # show there too. Suite names go on the outer side (see below).
            if c_i == 0:
                ax.set_ylabel("Success rate", fontsize=_APX_FS_LABEL)
                ax.tick_params(axis="y", labelsize=_APX_FS_TICK)
            else:
                ax.tick_params(axis="y", labelleft=False)

        # Outer, horizontal suite label centered on this row.
        row_mid_y = (top_in - ax_h / 2) / fig_h
        fig.text(_APX_CMB_SUITE_LABEL_X / fig_w, row_mid_y,
                 SUITE_DISPLAY.get(suite, suite),
                 ha="center", va="center",
                 fontsize=_APX_FS_LABEL, fontweight="bold")

    # Dataset column headers, just above the panels.
    header_y = (fig_h - _APX_CMB_TOP_PAD + _APX_CMB_HEADER_GAP) / fig_h
    for c_i, ds in enumerate(datasets):
        fig.text(col_centers[c_i], header_y, _DATASET_DISPLAY.get(ds, ds.upper()),
                 ha="center", va="bottom",
                 fontsize=_APX_FS_TITLE + 1, fontweight="bold")

    fig.legend(
        handles=[
            Patch(facecolor=_VIOLIN_COLORS[0], edgecolor="black", alpha=0.65, label=label_low),
            Patch(facecolor=_VIOLIN_COLORS[1], edgecolor="black", alpha=0.65, label=label_high),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.10 / fig_h),
        ncol=2, frameon=False, fontsize=_APX_FS_LEGEND,
        handlelength=1.4, handletextpad=0.5, columnspacing=1.6,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), dpi=200)  # companion for quick inspection
    plt.close(fig)


def plot_per_task_bars(result: PerTaskPairResult, out_path: Path) -> None:
    """Horizontal bar chart per suite: per-task delta pp (max-step − 1-step)."""
    suites = [s for s in SUITE_ORDER if s in result.suite_tasks]
    if not suites:
        return
    no_corr = _is_no_corr(result)
    n_suites = len(suites)
    fig, axes = plt.subplots(1, n_suites, figsize=(4.0 * n_suites, 5.5), sharey=False)
    if n_suites == 1:
        axes = [axes]

    label_low = f"{result.spec.variant_low}-step"
    label_high = f"{result.spec.variant_high}-step"

    for ax, suite in zip(axes, suites):
        cells = result.suite_tasks[suite]
        n_tasks = len(cells)
        y = np.arange(n_tasks)
        deltas = [(c.counts_high.rate - c.counts_low.rate) * 100 for c in cells]
        err = [
            np.sqrt(
                ((c.counts_low.rate - c.wilson_low[0]) ** 2 +
                 (c.wilson_high[1] - c.counts_high.rate) ** 2)
            ) * 100
            for c in cells
        ]
        colors = [
            _VIOLIN_COLORS[1] if d >= 0 else _VIOLIN_COLORS[0]
            for d in deltas
        ]
        ax.barh(y, deltas, xerr=err, color=colors, edgecolor="black",
                height=0.6, capsize=3, error_kw={"lw": 0.8})
        ax.axvline(0, color="black", lw=0.8, ls="--")

        for i, cell in enumerate(cells):
            marker = _cell_marker(cell, no_correction=no_corr)
            if marker:
                x_pos = deltas[i] + (err[i] + 1.0) * (1 if deltas[i] >= 0 else -1)
                ha = "left" if deltas[i] >= 0 else "right"
                ax.text(x_pos, i, marker, va="center", ha=ha,
                        fontsize=9, fontweight="bold")

        task_labels = [str(c.__dict__.get("task_idx", i)) for i, c in enumerate(cells)]
        ax.set_yticks(y)
        ax.set_yticklabels(task_labels, fontsize=9)
        ax.set_xlabel("Δ success rate (pp)")
        ax.set_title(
            f"{SUITE_DISPLAY.get(suite, suite)}\n({label_high} − {label_low})",
            fontsize=9,
        )
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    if no_corr:
        bars_suffix = f"({result.cld_basis}; no multiplicity correction)"
    else:
        bars_suffix = f"({result.cld_basis}; ** = STEP α_split, * = STEP α_global only)"
    fig.suptitle(
        f"{MODEL_DISPLAY.get(result.spec.model, result.spec.model)} — "
        f"{result.spec.dataset.upper()}: per-task Δ success rate\n"
        f"{bars_suffix}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_task_heatmap(result: PerTaskPairResult, out_path: Path) -> None:
    """Heatmap of Δ pp (max-step − 1-step) across tasks × suites; ** for corrected STEP wins."""
    suites = [s for s in SUITE_ORDER if s in result.suite_tasks]
    if not suites:
        return
    no_corr = _is_no_corr(result)
    n_tasks = max(len(result.suite_tasks[s]) for s in suites)

    delta_mat = np.full((n_tasks, len(suites)), np.nan)
    sig_mat = np.zeros((n_tasks, len(suites)), dtype=int)   # 0 none, 1 uncorrected only, 2 corrected
    row_labels_per_suite: dict[str, list[str]] = {}

    for col_idx, suite in enumerate(suites):
        cells = result.suite_tasks.get(suite, [])
        row_labels_per_suite[suite] = [
            str(c.__dict__.get("task_idx", i)) for i, c in enumerate(cells)
        ]
        for row_idx, cell in enumerate(cells):
            delta_mat[row_idx, col_idx] = (cell.counts_high.rate - cell.counts_low.rate) * 100
            if cell.primary_test == "step":
                if cell.step_decision_corrected in _SEPARATION_DECISIONS:
                    sig_mat[row_idx, col_idx] = 2
                elif cell.step_decision_uncorrected in _SEPARATION_DECISIONS:
                    sig_mat[row_idx, col_idx] = 1
            else:
                q = cell.q_holm_z
                if q is not None and q < 0.05:
                    sig_mat[row_idx, col_idx] = 2

    vmax = max(float(np.nanmax(np.abs(delta_mat))), 1.0)

    fig, axes = plt.subplots(
        1, len(suites),
        figsize=(3.8 * len(suites), 0.55 * n_tasks + 1.8),
        sharey=False,
    )
    if len(suites) == 1:
        axes = [axes]

    for col_idx, (ax, suite) in enumerate(zip(axes, suites)):
        data = delta_mat[:, col_idx: col_idx + 1]
        im = ax.imshow(
            data, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax,
            extent=[-0.5, 0.5, n_tasks - 0.5, -0.5],
        )
        labels = row_labels_per_suite[suite]
        for row_idx in range(len(labels)):
            tier = sig_mat[row_idx, col_idx]
            if no_corr:
                prefix = "  "
                fw = "normal"
            else:
                prefix = "**" if tier == 2 else (" *" if tier == 1 else "  ")
                fw = "bold" if tier == 2 else "normal"
            val = delta_mat[row_idx, col_idx]
            if not np.isnan(val):
                ax.text(0, row_idx, f"{prefix}{val:+.1f}", ha="center", va="center",
                        fontsize=7, color="black", fontweight=fw)

        ax.set_xticks([])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(SUITE_DISPLAY.get(suite, suite), fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ pp")

    if no_corr:
        heat_suffix = f"no multiplicity correction;  basis: {result.cld_basis}"
    else:
        heat_suffix = (
            f"** = STEP at α_split,  * = STEP at α_global only;  basis: {result.cld_basis}"
        )
    fig.suptitle(
        f"{MODEL_DISPLAY.get(result.spec.model, result.spec.model)} — "
        f"{result.spec.dataset.upper()}: per-task Δ success rate "
        f"({result.spec.variant_high}-step − {result.spec.variant_low}-step)\n"
        f"{heat_suffix}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
