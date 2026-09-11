#!/usr/bin/env python3
"""Generate a 2x3 grid of step-diff plots (LIBERO-10 on top, spatial/goal/object on bottom)."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_paper_training_curves import REPO_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 25,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 12,
    "legend.fontsize": 16,
})

from plot_paper_training_curves import (
    COLORS,
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    LINESTYLES,
    MARKERS,
    _series,
    add_training_percent,
    apply_results_selection,
    load_rows,
)


COLUMN_SPECS = [
    [
        {"model": "DiT", "variant": "scratch", "reference_inference_steps": 10, "label": "DiT"},
        {
            "model": "Pi0.5",
            "variant": "pretrainedVLM",
            "reference_inference_steps": 10,
            "label": "Pi0.5 pretrainedVLM",
            "legend_label": r"Paligemma VLM",
        },
    ],
    [
        {
            "model": "Pi0.5",
            "variant": "pretrained",
            "reference_inference_steps": 10,
            "label": "Pi0.5 pretrained",
            "legend_label": r"$\pi_{0.5}$",
        },
        {
            "model": "Pi0.5",
            "variant": "scratchActionExpOnly",
            "reference_inference_steps": 10,
            "label": "Pi0.5 scratchActionExpOnly",
            "legend_label": r"$\pi_{0.5}$ VLM",
        },
        {
            "model": "Pi0.5",
            "variant": "pretrainedVLM",
            "reference_inference_steps": 10,
            "label": "Pi0.5 pretrainedVLM",
            "legend_label": r"Paligemma VLM",
        },
        {
            "model": "Pi0.5",
            "variant": "scratchEverything",
            "reference_inference_steps": 10,
            "label": "Pi0.5 scratchEverything",
            "legend_label": r"Scratch",
        },
    ],
    [
        {
            "model": "Pi0.5",
            "variant": "pretrained",
            "reference_inference_steps": 10,
            "label": "Pi0.5 pretrained",
            "legend_label": r"$\pi_{0.5}$",
        },
        {"model": "FLOWER", "variant": "pretrained", "reference_inference_steps": 4, "label": "FLOWER"},
        {"model": "X-VLA", "variant": "pretrained", "reference_inference_steps": 10, "label": "X-VLA"},
    ],
]

COLUMN_TITLES = [
    "VLM Initialization",
    r"$\pi_{0.5}$ Initialization Variants",
    "Pretrained VLAs",
]

ROW_SUITES = [
    ("libero_10", "LIBERO-10"),
    (("spatial", "goal", "object"), "Spatial/Goal/Object"),
]


def _row_success(row, suite_selector):
    if isinstance(suite_selector, str):
        return row["suite_success"][suite_selector]
    return sum(row["suite_success"][s] for s in suite_selector) / len(suite_selector)


def _apply_row_metric(rows, suite_selector):
    for row in rows:
        row["success"] = _row_success(row, suite_selector)


def _plot_diff_on_ax(ax, rows, specs):
    diff_values = []
    for spec in specs:
        baseline = {row["train_step"]: row for row in _series(rows, spec["model"], spec["variant"], 1)}
        reference_steps = spec["reference_inference_steps"]
        reference = {
            row["train_step"]: row
            for row in _series(rows, spec["model"], spec["variant"], reference_steps)
        }
        shared_steps = sorted(set(baseline) & set(reference))
        if not shared_steps:
            raise ValueError(f"No paired 1-step/reference-step data for plot series: {spec}")

        display_name = spec["label"]
        values = [
            reference[train_step]["success"] - baseline[train_step]["success"]
            for train_step in shared_steps
        ]
        diff_values.extend(values)
        ax.plot(
            [baseline[train_step]["training_pct"] for train_step in shared_steps],
            values,
            label=spec.get("legend_label", display_name),
            color=COLORS[display_name],
            marker=MARKERS[reference_steps],
            linestyle=LINESTYLES[reference_steps],
            linewidth=2.4,
            markersize=5.5,
        )

    ax.axhline(0.0, color="0.35", linewidth=1.0, alpha=0.8)
    ax.set_xlim(0, 102)
    ax.set_ylim(-20, 15)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(alpha=0.25, linewidth=0.8)
    return diff_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory of normalized setting JSON files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for generated PNG/PDF figures.",
    )
    parser.add_argument(
        "--output_name",
        default="plot_grid_step_diff",
        help="Base filename (without extension) for the output figure.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.data)
    add_training_percent(rows)
    # Initialize 'success' field; will be overridden per-row below.
    apply_results_selection(rows, None)

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17.5, 9), sharey=True, sharex=True)

    for row_idx, (suite_selector, row_label) in enumerate(ROW_SUITES):
        _apply_row_metric(rows, suite_selector)
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            specs = COLUMN_SPECS[col_idx]
            _plot_diff_on_ax(ax, rows, specs)

            if row_idx == 0:
                # ax.set_title(f"{COLUMN_TITLES[col_idx]}", pad=30)
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
                if col_idx == 0:
                    # ax.set_ylabel(f"{row_label}\n\nSuccess rate gap (pp)")
                    # ax.set_ylabel(f"Success rate gap (pp)")
                    ax.set_xlabel("VLM initialization", fontsize=22, labelpad=8)
                    ax.xaxis.set_label_position('top')
                    ax.set_ylabel("LIBERO-10", fontsize=22, labelpad=38)
                elif col_idx == 1:
                    ax.set_xlabel(r"$\pi_{0.5}$ initialization variants", fontsize=22, labelpad=8)
                    ax.xaxis.set_label_position('top')
                elif col_idx == 2:
                    ax.set_xlabel(r"Pretrained VLAs", fontsize=22, labelpad=8)
                    ax.xaxis.set_label_position('top')
            else:
                # ax.set_xlabel("% of training")
                if col_idx == 0:
                    ax.set_ylabel("LIBERO SGO", fontsize=22, labelpad=38)
                pass

    fig.supxlabel("% of training", fontsize=18, y=0.13, x=0.528)
    fig.supylabel("Success rate gap (pp)", fontsize=18, x=0.0655, y=0.55)

    fig.tight_layout(rect=(0, 0.10, 1, 1))

    for col_idx in range(n_cols):
        ax = axes[1][col_idx]
        ax.set_ylim([-20, 15])
        handles, labels = ax.get_legend_handles_labels()
        bbox = ax.get_position()
        fig.legend(
            handles,
            labels,
            handler_map={type(handles[0]): HandlerLine2D(numpoints=1)},
            handlelength=0,
            markerscale=2,
            loc="upper center",
            bbox_to_anchor=(bbox.x0 + bbox.width / 2.0, 0.55),
            frameon=False,
            # ncol=min(2, len(labels)),
            # ncol=1 if col_idx in [0] else 2,
            ncol=2,
            fontsize=19.0,
        )



    for ext in ("pdf", "png"):
        path = os.path.join(str(args.out_dir), f"{args.output_name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
