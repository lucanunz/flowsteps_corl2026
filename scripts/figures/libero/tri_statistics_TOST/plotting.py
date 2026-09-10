"""Equivalence forest plots for the TOST variant.

The equivalence-native view: per cell, the paired Δ̂ with its 90% CI as a
horizontal bar, a shaded ±δ band, and a vertical line at 0. A CI inside the band
is equivalence; a CI entirely outside is a relevant difference; otherwise
inconclusive. Cells without paired data are drawn as a greyed "not assessed" row.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tri_statistics_STEP.pairs import cells_in_order  # noqa: E402

from .config import MODEL_DISPLAY, SUITE_DISPLAY, SUITE_ORDER  # noqa: E402
from .stats import VERDICT_COLOR, VERDICT_LABEL  # noqa: E402


def _ordered_cells(pair):
    """Cells in plot order, honouring a custom `pair.cell_order` if present."""
    order = getattr(pair, "cell_order", None)
    if order is not None:
        return [pair.cells[k] for k in order if k in pair.cells]
    return cells_in_order(pair)


def _forest(ax, rows: list[tuple[str, object]], delta_pp: float, title: str) -> None:
    """rows: list of (label, TostResult|None), drawn top-to-bottom."""
    n = len(rows)
    ys = list(range(n))[::-1]  # first row at top

    ax.axvspan(-delta_pp, delta_pp, color="#9CA3AF", alpha=0.20, lw=0,
               label=f"±{delta_pp:.0f}pp equivalence band", zorder=1)
    ax.axvline(0.0, color="#111827", lw=1.2, zorder=2)
    ax.axvline(-delta_pp, color="#6B7280", lw=0.8, ls="--", zorder=2)
    ax.axvline(delta_pp, color="#6B7280", lw=0.8, ls="--", zorder=2)

    seen: set[str] = set()
    for y, (label, tost) in zip(ys, rows):
        if tost is None:
            ax.text(0.0, y, "not assessed (no paired data)", va="center",
                    ha="center", fontsize=8, color="#9CA3AF", style="italic",
                    zorder=3)
            continue
        d = tost.delta_hat * 100.0
        lo = tost.ci_low * 100.0
        hi = tost.ci_high * 100.0
        color = VERDICT_COLOR[tost.verdict]
        legend_label = VERDICT_LABEL[tost.verdict] if tost.verdict not in seen else None
        seen.add(tost.verdict)
        ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="round",
                zorder=3)
        ax.plot([lo, lo], [y - 0.16, y + 0.16], color=color, lw=2.2, zorder=3)
        ax.plot([hi, hi], [y - 0.16, y + 0.16], color=color, lw=2.2, zorder=3)
        ax.scatter([d], [y], s=42, color=color, edgecolor="#111827", lw=0.7,
                   zorder=4, label=legend_label)

    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _ in rows])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Δ = reference − 1-step (pp)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(axis="x", color="#E5E7EB", lw=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=8, frameon=False)


def plot_equivalence_forest(pair, out_path: Path, delta_pp: float = 3.0) -> None:
    cells = _ordered_cells(pair)
    rows = [
        (SUITE_DISPLAY.get(c.suite, c.suite), getattr(c, "tost", None))
        for c in cells
    ]
    model_name = MODEL_DISPLAY.get(pair.spec.model, pair.spec.model)
    dataset_label = "LIBERO+" if pair.spec.dataset == "liberoplus" else "LIBERO"
    title = (
        f"{model_name} — {dataset_label}: "
        f"{pair.spec.variant_high}-step vs {pair.spec.variant_low}-step "
        f"equivalence (δ=±{delta_pp:.0f}pp, 90% CI)"
    )
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(rows) + 1.6))
    _forest(ax, rows, delta_pp, title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_task_equivalence_forest(result, out_path: Path, delta_pp: float = 3.0) -> None:
    """One forest per (model, dataset) with every task, grouped by suite."""
    rows: list[tuple[str, object]] = []
    for suite in SUITE_ORDER:
        cells = result.suite_tasks.get(suite)
        if not cells:
            continue
        rows.append((f"— {SUITE_DISPLAY.get(suite, suite)} —", None))
        for cell in cells:
            label = cell.suite if len(cell.suite) <= 34 else cell.suite[:31] + "…"
            rows.append((label, getattr(cell, "tost", None)))
    if not rows:
        return
    model_name = MODEL_DISPLAY.get(result.spec.model, result.spec.model)
    dataset_label = "LIBERO+" if result.spec.dataset == "liberoplus" else "LIBERO"
    title = (
        f"{model_name} — {dataset_label}: per-task equivalence "
        f"(δ=±{delta_pp:.0f}pp, 90% CI)"
    )
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(rows) + 1.6))
    _forest(ax, rows, delta_pp, title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
