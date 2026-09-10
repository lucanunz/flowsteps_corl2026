"""Matplotlib plotting for the CALVIN STEP pipeline.

Three plots per (model, dataset) pair:
  - `plot_sr5_pair_violin`: Beta-posterior violin for SR@5 with CLD letters,
    significance markers identical to the LIBERO STEP pipeline.
  - `plot_completion_violin`: Dirichlet-posterior violin of the mean
    completion score (0-5) with Welch CLD letters.
  - `plot_completion_histogram`: stacked bar chart of n_0..n_5 fractions.

Also `plot_dataset_overall_comparison`: side-by-side SR@5 bars for all
models in a dataset (mirrors the LIBERO overview plot).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .config import CELL_DISPLAY, DATASET_DISPLAY, MODEL_DISPLAY
from .pairs import CellStats, PairResult
from .stats import beta_posterior_samples, dirichlet_posterior_mean_samples


_MODEL_ORDER = ("flower", "mibot", "xvla")


def _sort_pairs_by_model(results: list[PairResult]) -> list[PairResult]:
    rank = {m: i for i, m in enumerate(_MODEL_ORDER)}
    return sorted(results, key=lambda p: rank.get(p.spec.model, 99))


_VIOLIN_COLORS = ("#4C72B0", "#DD8452")  # low, high
_SEPARATION_DECISIONS = {"AcceptAlternative", "AcceptNull"}


def _cell_marker(cell: CellStats) -> str:
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


def _welch_marker(cell: CellStats) -> str:
    if cell.welch is None:
        return ""
    p = cell.welch.p_value
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_sr5_pair_violin(pair: PairResult, out_path: Path, n_samples: int = 4000) -> None:
    cell = pair.cells["sr5"]
    fig, ax = plt.subplots(figsize=(4.0, 4.6))
    rng = np.random.default_rng(0)

    label_low = f"{pair.spec.variant_low}-step"
    label_high = f"{pair.spec.variant_high}-step"

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
        ax.plot(pos, c.rate, marker="_", linestyle="None",
                color="black", markeredgewidth=0.8, markersize=6, zorder=3)

    ax.text(0, 1.03, cell.cld_low_corrected, ha="center", va="bottom",
            fontsize=11, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(1, 1.03, cell.cld_high_corrected, ha="center", va="bottom",
            fontsize=11, fontweight="bold", transform=ax.get_xaxis_transform())

    marker = _cell_marker(cell)
    title_suffix = f"  {marker}" if marker else ""
    ax.set_title(f"{CELL_DISPLAY['sr5']}{title_suffix}", fontsize=11)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_low, label_high], fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("SR@5 (Beta posterior)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        f"{MODEL_DISPLAY.get(pair.spec.model, pair.spec.model)} — "
        f"{DATASET_DISPLAY.get(pair.spec.dataset, pair.spec.dataset)}: "
        f"{pair.spec.variant_low}-step vs {pair.spec.variant_high}-step\n"
        f"CLD from {pair.cld_basis}. ** = STEP at α_split, * = STEP at α_global only",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_completion_violin(pair: PairResult, out_path: Path) -> None:
    cell = pair.cells["sr5"]
    assert cell.completion_low is not None and cell.completion_high is not None

    samples_low = dirichlet_posterior_mean_samples(cell.completion_low.counts, seed=0)
    samples_high = dirichlet_posterior_mean_samples(cell.completion_high.counts, seed=1)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    positions = [0, 1]
    parts = ax.violinplot(
        [samples_low, samples_high],
        positions=positions,
        widths=0.7,
        showextrema=False,
        showmeans=False,
        showmedians=False,
    )
    for body, color in zip(parts["bodies"], _VIOLIN_COLORS):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)

    for pos, cs in (
        (positions[0], cell.completion_low),
        (positions[1], cell.completion_high),
    ):
        ax.plot(pos, cs.mean, marker="_", linestyle="None", color="#111827",
                markeredgewidth=0.9, markersize=11, zorder=5)

    y_high = max(float(samples_low.max()), float(samples_high.max()))
    y_low = min(float(samples_low.min()), float(samples_high.min()))
    span = y_high - y_low
    cld_y = y_high + max(0.02, span * 0.15)
    ax.text(positions[0], cld_y, cell.welch_cld_low, ha="center", va="bottom",
            fontsize=14, fontweight="bold")
    ax.text(positions[1], cld_y, cell.welch_cld_high, ha="center", va="bottom",
            fontsize=14, fontweight="bold")

    welch_marker = _welch_marker(cell)
    suffix = f"  {welch_marker}" if welch_marker else ""
    ax.set_xticks(positions)
    ax.set_xticklabels([
        f"{pair.spec.variant_low} step",
        f"{pair.spec.variant_high} step",
    ])
    ax.set_ylabel("Posterior mean completion score (0-5)")
    welch_p = cell.welch.p_value if cell.welch else float("nan")
    p_text = "<0.001" if welch_p < 0.001 else f"{welch_p:.3f}"
    ax.set_title(
        f"{MODEL_DISPLAY.get(pair.spec.model, pair.spec.model)} task completion "
        f"— Welch p={p_text}{suffix}",
        fontsize=11,
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_ylim(y_low - span * 0.1, cld_y + span * 0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_completion_histogram(pair: PairResult, out_path: Path) -> None:
    cell = pair.cells["sr5"]
    assert cell.completion_low is not None and cell.completion_high is not None
    levels = list(range(6))
    n_low = cell.completion_low.n
    n_high = cell.completion_high.n
    frac_low = [c / n_low for c in cell.completion_low.counts]
    frac_high = [c / n_high for c in cell.completion_high.counts]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    width = 0.4
    indices = np.arange(len(levels))
    ax.bar(
        indices - width / 2, frac_low, width=width,
        color=_VIOLIN_COLORS[0],
        label=f"{pair.spec.variant_low} step (n={n_low})",
    )
    ax.bar(
        indices + width / 2, frac_high, width=width,
        color=_VIOLIN_COLORS[1],
        label=f"{pair.spec.variant_high} step (n={n_high})",
    )
    for x, fraction, count in zip(indices - width / 2, frac_low, cell.completion_low.counts):
        ax.text(x, fraction, f"{count}", ha="center", va="bottom", fontsize=8, color="#374151")
    for x, fraction, count in zip(indices + width / 2, frac_high, cell.completion_high.counts):
        ax.text(x, fraction, f"{count}", ha="center", va="bottom", fontsize=8, color="#1d4ed8")

    ax.set_xticks(indices)
    ax.set_xticklabels([str(level) for level in levels])
    ax.set_xlabel("Completed subtasks per rollout")
    ax.set_ylabel("Fraction of rollouts")
    ax.set_title(
        f"{MODEL_DISPLAY.get(pair.spec.model, pair.spec.model)} "
        f"{DATASET_DISPLAY.get(pair.spec.dataset, pair.spec.dataset)} task-completion distribution"
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dataset_overall_comparison(
    results: list[PairResult], out_path: Path, dataset: str
) -> None:
    """Side-by-side SR@5 bars for all models in a dataset."""
    fig, ax = plt.subplots(figsize=(1.6 * len(results) + 1.5, 4.4))
    x = np.arange(len(results))
    width = 0.36

    rates_low: list[float] = []
    rates_high: list[float] = []
    err_low_lo, err_low_hi = [], []
    err_high_lo, err_high_hi = [], []
    labels: list[str] = []
    markers: list[str] = []
    test_kinds: set[str] = set()
    for pair in results:
        cell = pair.cells["sr5"]
        rates_low.append(cell.counts_low.rate)
        rates_high.append(cell.counts_high.rate)
        err_low_lo.append(cell.counts_low.rate - cell.wilson_low[0])
        err_low_hi.append(cell.wilson_low[1] - cell.counts_low.rate)
        err_high_lo.append(cell.counts_high.rate - cell.wilson_high[0])
        err_high_hi.append(cell.wilson_high[1] - cell.counts_high.rate)
        labels.append(
            f"{MODEL_DISPLAY.get(pair.spec.model, pair.spec.model)}\n"
            f"1 vs {pair.spec.variant_high}"
        )
        markers.append(_cell_marker(cell))
        test_kinds.add(cell.primary_test)

    ax.bar(x - width / 2, rates_low, width, color=_VIOLIN_COLORS[0],
           edgecolor="black", label="1-step",
           yerr=[err_low_lo, err_low_hi], capsize=3, error_kw={"lw": 0.8})
    ax.bar(x + width / 2, rates_high, width, color=_VIOLIN_COLORS[1],
           edgecolor="black", label="N-step",
           yerr=[err_high_lo, err_high_hi], capsize=3, error_kw={"lw": 0.8})

    for i, (rate_lo, rate_hi, marker) in enumerate(zip(rates_low, rates_high, markers)):
        if marker:
            top = max(rate_lo, rate_hi)
            ax.text(i, min(1.02, top + 0.04), marker, ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("SR@5")
    basis = "paired STEP" if test_kinds == {"step"} else (
        "unpaired z" if test_kinds == {"z"} else "mixed paired/unpaired"
    )
    ax.set_title(
        f"{DATASET_DISPLAY.get(dataset, dataset)}: SR@5 by model\n"
        f"(** = STEP α_split,  * = STEP α_global only;  {basis})"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_combined_sr5_violin(
    results: list[PairResult],
    out_path: Path,
    dataset: str,
    n_samples: int = 4000,
) -> None:
    """Combined SR@5 Beta-posterior violin plot for all models in a dataset.

    Three grouped pairs (low / high variant per model), CLD letters above each
    violin and STEP/z significance markers above each pair, intended for the
    paper appendix.
    """
    pairs = _sort_pairs_by_model(results)
    rng = np.random.default_rng(0)

    positions: list[int] = []
    for i in range(len(pairs)):
        base = 3 * i
        positions.extend([base, base + 1])

    sample_arrays: list[np.ndarray] = []
    for pair in pairs:
        cell = pair.cells["sr5"]
        sample_arrays.append(
            beta_posterior_samples(
                cell.counts_low.successes, cell.counts_low.trials, n_samples, rng
            )
        )
        sample_arrays.append(
            beta_posterior_samples(
                cell.counts_high.successes, cell.counts_high.trials, n_samples, rng
            )
        )

    fig, ax = plt.subplots(figsize=(5.5, 3.7))
    parts = ax.violinplot(
        sample_arrays,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    for idx, body in enumerate(parts["bodies"]):
        color = _VIOLIN_COLORS[idx % 2]
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.55)

    for i, pair in enumerate(pairs):
        cell = pair.cells["sr5"]
        pos_low = positions[2 * i]
        pos_high = positions[2 * i + 1]
        for pos, c in ((pos_low, cell.counts_low), (pos_high, cell.counts_high)):
            ax.plot(pos, c.rate, marker="_", linestyle="None",
                    color="black", markeredgewidth=0.8, markersize=6, zorder=3)
        top_low = float(parts["bodies"][2 * i].get_paths()[0].vertices[:, 1].max())
        top_high = float(parts["bodies"][2 * i + 1].get_paths()[0].vertices[:, 1].max())
        cld_y_data = max(top_low, top_high) + 0.03
        ax.text(pos_low, cld_y_data, cell.cld_low_corrected, ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.text(pos_high, cld_y_data, cell.cld_high_corrected, ha="center", va="bottom",
                fontsize=9, fontweight="bold")

        marker = _cell_marker(cell)
        if marker:
            center = (pos_low + pos_high) / 2.0
            ax.text(center, cld_y_data + 0.06, marker, ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    variant_labels: list[str] = []
    for pair in pairs:
        variant_labels.append(f"{pair.spec.variant_low}-step")
        variant_labels.append(f"{pair.spec.variant_high}-step")
    ax.set_xticks(positions)
    ax.set_xticklabels(variant_labels, fontsize=8)

    for i, pair in enumerate(pairs):
        center = (positions[2 * i] + positions[2 * i + 1]) / 2.0
        ax.text(
            center, -0.13,
            MODEL_DISPLAY.get(pair.spec.model, pair.spec.model),
            ha="center", va="top",
            fontsize=9, fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xlim(positions[0] - 1.0, positions[-1] + 1.0)
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel(f"{CELL_DISPLAY['sr5']} (Beta posterior)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    legend_handles = [
        Patch(facecolor=_VIOLIN_COLORS[0], edgecolor="black", alpha=0.55, label="1-step"),
        Patch(facecolor=_VIOLIN_COLORS[1], edgecolor="black", alpha=0.55, label="N-step"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)

    ax.set_title(
        f"{DATASET_DISPLAY.get(dataset, dataset)}: SR@5 posterior, 1-step vs N-step "
        f"(** = STEP α_split, * = STEP α_global)",
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")  # PNG keeps the title
    ax.set_title("")  # paper PDF: no title (use the LaTeX caption instead)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_combined_completion_violin(
    results: list[PairResult],
    out_path: Path,
    dataset: str,
) -> None:
    """Combined Dirichlet-posterior mean-completion violin plot for all models."""
    pairs = _sort_pairs_by_model(results)

    positions: list[int] = []
    for i in range(len(pairs)):
        base = 3 * i
        positions.extend([base, base + 1])

    sample_arrays: list[np.ndarray] = []
    for pair in pairs:
        cell = pair.cells["sr5"]
        assert cell.completion_low is not None and cell.completion_high is not None
        sample_arrays.append(
            dirichlet_posterior_mean_samples(cell.completion_low.counts, seed=0)
        )
        sample_arrays.append(
            dirichlet_posterior_mean_samples(cell.completion_high.counts, seed=1)
        )

    y_high = max(float(s.max()) for s in sample_arrays)
    y_low = min(float(s.min()) for s in sample_arrays)
    span = y_high - y_low

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    parts = ax.violinplot(
        sample_arrays,
        positions=positions,
        widths=0.75,
        showextrema=False,
        showmeans=False,
        showmedians=False,
    )
    for idx, body in enumerate(parts["bodies"]):
        color = _VIOLIN_COLORS[idx % 2]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)

    for i, pair in enumerate(pairs):
        cell = pair.cells["sr5"]
        pos_low = positions[2 * i]
        pos_high = positions[2 * i + 1]
        samples_low = sample_arrays[2 * i]
        samples_high = sample_arrays[2 * i + 1]
        for pos, cs in (
            (pos_low, cell.completion_low),
            (pos_high, cell.completion_high),
        ):
            ax.plot(pos, cs.mean, marker="_", linestyle="None", color="#111827",
                    markeredgewidth=0.9, markersize=11, zorder=5)

        pair_top = max(float(samples_low.max()), float(samples_high.max()))
        cld_y_pair = pair_top + max(0.01, span * 0.04)
        ax.text(pos_low, cld_y_pair, cell.welch_cld_low, ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.text(pos_high, cld_y_pair, cell.welch_cld_high, ha="center", va="bottom",
                fontsize=9, fontweight="bold")

        wmark = "*" if cell.welch is not None and cell.welch.p_value < 0.05 else ""
        if wmark:
            center = (pos_low + pos_high) / 2.0
            ax.text(center, cld_y_pair + span * 0.07, wmark, ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    variant_labels: list[str] = []
    for pair in pairs:
        variant_labels.append(f"{pair.spec.variant_low}-step")
        variant_labels.append(f"{pair.spec.variant_high}-step")
    ax.set_xticks(positions)
    ax.set_xticklabels(variant_labels, fontsize=8)

    for i, pair in enumerate(pairs):
        center = (positions[2 * i] + positions[2 * i + 1]) / 2.0
        ax.text(
            center, -0.13,
            MODEL_DISPLAY.get(pair.spec.model, pair.spec.model),
            ha="center", va="top",
            fontsize=9, fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xlim(positions[0] - 1.0, positions[-1] + 1.0)
    ax.set_ylim(y_low - span * 0.1, y_high + span * 0.25)
    ax.set_ylabel("Avg. Len.", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    legend_handles = [
        Patch(facecolor=_VIOLIN_COLORS[0], edgecolor=_VIOLIN_COLORS[0], alpha=0.55, label="1-step"),
        Patch(facecolor=_VIOLIN_COLORS[1], edgecolor=_VIOLIN_COLORS[1], alpha=0.55, label="N-step"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)

    ax.set_title(
        f"{DATASET_DISPLAY.get(dataset, dataset)}: posterior mean task completion, "
        f"1-step vs N-step (Welch * = p<0.05)",
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")  # PNG keeps the title
    ax.set_title("")  # paper PDF: no title (use the LaTeX caption instead)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
