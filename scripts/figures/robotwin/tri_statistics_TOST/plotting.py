"""Generic equivalence forest plot for the TOST forks.

`plot_forest(rows, ...)` draws one horizontal row per cell: the paired Δ̂ with its
90% CI, a shaded ±δ band and a 0 line. CI inside the band ⇒ equivalent; CI
outside ⇒ relevant difference; otherwise inconclusive. `rows` is a list of
(label, TostResult|None); None rows render as a greyed "not assessed" marker.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .stats import VERDICT_COLOR, VERDICT_LABEL  # noqa: E402


def plot_forest(rows: list, out_path: Path, *, title: str, delta_pp: float = 3.0) -> None:
    if not rows:
        return
    n = len(rows)
    ys = list(range(n))[::-1]

    fig, ax = plt.subplots(figsize=(7.4, 0.42 * n + 1.7))
    ax.axvspan(-delta_pp, delta_pp, color="#9CA3AF", alpha=0.20, lw=0,
               label=f"±{delta_pp:.0f}pp equivalence band", zorder=1)
    ax.axvline(0.0, color="#111827", lw=1.2, zorder=2)
    ax.axvline(-delta_pp, color="#6B7280", lw=0.8, ls="--", zorder=2)
    ax.axvline(delta_pp, color="#6B7280", lw=0.8, ls="--", zorder=2)

    seen: set[str] = set()
    for y, (label, tost) in zip(ys, rows):
        if tost is None:
            ax.text(0.0, y, "not assessed (no paired data)", va="center",
                    ha="center", fontsize=8, color="#9CA3AF", style="italic", zorder=3)
            continue
        d = tost.delta_hat * 100.0
        lo, hi = tost.ci_low * 100.0, tost.ci_high * 100.0
        color = VERDICT_COLOR[tost.verdict]
        legend_label = VERDICT_LABEL[tost.verdict] if tost.verdict not in seen else None
        seen.add(tost.verdict)
        ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="round", zorder=3)
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
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
