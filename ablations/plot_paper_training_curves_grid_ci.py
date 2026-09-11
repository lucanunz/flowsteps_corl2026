#!/usr/bin/env python3
"""2x3 grid of step-diff plots with 95% confidence intervals.

Same panels and series as `plot_paper_training_curves_grid.py`, plus a shaded
95% interval around every curve.

The plotted quantity is the paired gap Delta = success(n=reference) -
success(n=1) at one checkpoint, so its interval is the *paired* one the rest of
the repository uses: Newcombe (1998) Method 10 on the 2x2 table of the two
policies' per-episode outcomes (`scripts/equivalence.py`), at 95% two-sided
coverage. Episodes are joined by (suite, task, episode_id) through
`scripts/paired.py`, never by aggregate rate.

Two caveats, both surfaced on stdout when the figure is written:

* X-VLA's LIBERO evaluations carry `pairing_basis == "assumed_order"`, so their
  tables need the evaluation-order assumption (`align(..., allow_order=True)`,
  the library's `--allow-order`). Pass `--no_allow_order` to drop those points
  instead. Its five *checkpoint* sweeps additionally do not key-join at all:
  each task sits in exactly one evaluation shard, 16 of the 40 tasks landed in a
  differently-named shard in the n=1 and n=10 runs, and `episode_id` is
  `[shard, index]` — so 800 of 2,000 keys differ by shard label alone while the
  index sets 0..49 match task for task. Those are joined within (suite, task)
  by episode index instead — one level down from the order assumption the rows
  already declare, and it recovers all 2,000 pairs with marginals identical to
  the plotted rates. `--no_order_join` refuses that and drops the five points'
  intervals.
A point whose two settings cannot be paired at all keeps its estimate but is
drawn with a hollow marker and no interval. No point in the current export does.
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_paper_training_curves import REPO_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.legend_handler import HandlerLine2D

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 25,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 12,
    "legend.fontsize": 16,
})

from plot_paper_training_curves import (  # noqa: E402
    COLORS,
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    LINESTYLES,
    MARKERS,
    PRIMARY_SUBSTITUTES,
    _series,
    _series_identity,
    add_training_percent,
    apply_results_selection,
    load_rows,
)

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from paired import align, load  # noqa: E402
from equivalence import paired_newcombe_ci  # noqa: E402

# Reuse the grid's panel definitions verbatim so the two figures cannot drift.
from plot_paper_training_curves_grid import COLUMN_SPECS, ROW_SUITES  # noqa: E402


CONFIDENCE = 0.95

# `suite_success` keys (ROW_SUITES) -> the `episodes[*].suite` values they pool.
EPISODE_SUITES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "libero_10": "libero_10",
}


def setting_paths(data_dir):
    """(model, variant, train_step, nsteps) -> setting file, same rows load_rows reads."""
    data_dir = Path(data_dir)
    paths = {}
    for path in sorted(data_dir.glob("libero__*__checkpoint__*.json")):
        setting = load(path)["setting"]
        if setting["benchmark"] != "libero" or setting["evaluation"] != "checkpoint":
            continue
        identity = _series_identity(setting)
        if identity is None:
            continue
        substitute = PRIMARY_SUBSTITUTES.get(identity)
        if substitute is not None:
            path = data_dir / substitute.format(nsteps=setting["nsteps"])
        paths[(*identity, setting["nsteps"])] = path
    return paths


def _order_join(low, high):
    """Positional join within (suite, task), by episode index.

    Only for cohorts whose rows already declare `pairing_basis ==
    "assumed_order"`: the recorded `episode_id` is not a trial identity there,
    so joining on it buys nothing the order assumption does not already grant.
    Returns aligned (low_rows, high_rows), or None when the two sides do not
    present the same tasks with the same per-task episode counts — in which case
    there is no order to assume and the points are dropped.
    """
    def grouped(data):
        bins = defaultdict(list)
        for row in data["episodes"]:
            if row["pairing_basis"] != "assumed_order" or row["success"] is None:
                return None
            bins[(row["suite"], row["task"])].append(row)
        return bins

    low_bins, high_bins = grouped(low), grouped(high)
    if low_bins is None or high_bins is None or set(low_bins) != set(high_bins):
        return None
    if any(len(low_bins[k]) != len(high_bins[k]) for k in low_bins):
        return None

    def order(rows):
        # episode_id is [shard, index] here; the shard label is the run-layout
        # artifact that differs, the index is the within-task rollout order.
        return sorted(rows, key=lambda row: row["episode_id"][-1])

    low_rows, high_rows = [], []
    for bin_key in sorted(low_bins):
        lows, highs = order(low_bins[bin_key]), order(high_bins[bin_key])
        if [r["episode_id"][-1] for r in lows] != [r["episode_id"][-1] for r in highs]:
            return None
        low_rows.extend(lows)
        high_rows.extend(highs)
    return low_rows, high_rows


def _paired_table(low_rows, high_rows, episode_suites):
    """2x2 counts over the selected suites: (both, 1-step only, reference only, neither)."""
    a = b = c = d = 0
    for low, high in zip(low_rows, high_rows):
        if low["suite"] not in episode_suites:
            continue
        if low["success"] and high["success"]:
            a += 1
        elif low["success"]:
            b += 1
        elif high["success"]:
            c += 1
        else:
            d += 1
    return a, b, c, d


def diff_ci(paths, model, variant, train_step, reference_steps, suite_selector, *,
            allow_order, order_join, notes):
    """95% paired CI for the step gap, in percentage points, or None if unpairable."""
    selector = (suite_selector,) if isinstance(suite_selector, str) else suite_selector
    episode_suites = {EPISODE_SUITES[suite] for suite in selector}

    low_path = paths.get((model, variant, train_step, 1))
    high_path = paths.get((model, variant, train_step, reference_steps))
    if low_path is None or high_path is None:
        return None

    low, high = load(low_path), load(high_path)
    series = f"{model} {variant} @ {train_step}"
    try:
        low_rows, high_rows, _ = align(low, high, allow_order=allow_order)
    except ValueError as error:
        joined = _order_join(low, high) if (order_join and allow_order) else None
        if joined is None:
            notes.append(f"  no paired CI for {series}: {error}")
            return None
        low_rows, high_rows = joined
        notes.append(f"  {series}: order-joined within (suite, task) by episode index "
                     f"({len(low_rows)} pairs); keys do not join ({error.args[0].split(';')[0]})")

    a, b, c, d = _paired_table(low_rows, high_rows, episode_suites)
    interval = paired_newcombe_ci(a, b, c, d, confidence=CONFIDENCE)
    if interval is None:
        return None
    delta, lower, upper = interval
    return 100.0 * delta, 100.0 * lower, 100.0 * upper, a + b + c + d


def _plot_diff_on_ax(ax, rows, specs, paths, suite_selector, *, allow_order, order_join,
                     notes, extremes):
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
        color = COLORS[display_name]
        xs = [baseline[step]["training_pct"] for step in shared_steps]
        values = [reference[step]["success"] - baseline[step]["success"] for step in shared_steps]

        lows, highs, uncertain_x, uncertain_y = [], [], [], []
        for step, x, value in zip(shared_steps, xs, values):
            result = diff_ci(
                paths, spec["model"], spec["variant"], step, reference_steps, suite_selector,
                allow_order=allow_order, order_join=order_join, notes=notes,
            )
            if result is None:
                # Keep the point (it is what the original figure plots) but say so.
                lows.append(value)
                highs.append(value)
                uncertain_x.append(x)
                uncertain_y.append(value)
                continue
            delta, lower, upper, _ = result
            if abs(delta - value) > 1e-6:
                raise ValueError(
                    f"paired table disagrees with the plotted rate for {spec['label']} @ {step}: "
                    f"{delta:.4f} vs {value:.4f} pp"
                )
            lows.append(lower)
            highs.append(upper)

        extremes.extend(lows + highs)
        ax.fill_between(xs, lows, highs, color=color, alpha=0.15, linewidth=0)
        ax.plot(
            xs,
            values,
            label=spec.get("legend_label", display_name),
            color=color,
            marker=MARKERS[reference_steps],
            linestyle=LINESTYLES[reference_steps],
            linewidth=2.4,
            markersize=5.5,
        )
        if uncertain_x:
            # Hollow marker: point estimate from task_counts, no pairable episodes.
            ax.plot(
                uncertain_x, uncertain_y, linestyle="none", marker=MARKERS[reference_steps],
                markersize=7.0, markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.8, zorder=5,
            )

    ax.axhline(0.0, color="0.35", linewidth=1.0, alpha=0.8)
    ax.set_xlim(0, 102)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(alpha=0.25, linewidth=0.8)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory of normalized setting JSON files.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Directory for generated PNG/PDF figures.")
    parser.add_argument("--output_name", default="plot_grid_step_diff_ci",
                        help="Base filename (without extension) for the output figure.")
    parser.add_argument("--no_allow_order", dest="allow_order", action="store_false",
                        help="Refuse order-assumed pairing (drops X-VLA's intervals).")
    parser.add_argument("--no_order_join", dest="order_join", action="store_false",
                        help="Do not fall back to a within-(suite, task) join by episode index "
                             "for assumed-order cohorts whose keys do not join; X-VLA's five "
                             "checkpoint points then get no interval.")
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                        help="Shared y limits in pp; default fits the widest interval.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    notes, extremes = [], []

    rows = load_rows(args.data)
    add_training_percent(rows)

    apply_results_selection(rows, None)
    paths = setting_paths(args.data)

    from plot_paper_training_curves_grid import _apply_row_metric

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17.5, 9), sharey=True, sharex=True)

    for row_idx, (suite_selector, _row_label) in enumerate(ROW_SUITES):
        _apply_row_metric(rows, suite_selector)
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            _plot_diff_on_ax(
                ax, rows, COLUMN_SPECS[col_idx], paths, suite_selector,
                allow_order=args.allow_order, order_join=args.order_join,
                notes=notes, extremes=extremes,
            )

            if row_idx == 0:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
                if col_idx == 0:
                    ax.set_xlabel("VLM initialization", fontsize=22, labelpad=8)
                    ax.xaxis.set_label_position("top")
                    ax.set_ylabel("LIBERO-10", fontsize=22, labelpad=38)
                elif col_idx == 1:
                    ax.set_xlabel(r"$\pi_{0.5}$ initialization variants", fontsize=22, labelpad=8)
                    ax.xaxis.set_label_position("top")
                elif col_idx == 2:
                    ax.set_xlabel(r"Pretrained VLAs", fontsize=22, labelpad=8)
                    ax.xaxis.set_label_position("top")
            elif col_idx == 0:
                ax.set_ylabel("LIBERO SGO", fontsize=22, labelpad=38)

    if args.ylim is not None:
        ylim = tuple(args.ylim)
    else:
        span = max(extremes) - min(extremes)
        ylim = (min(extremes) - 0.04 * span, max(extremes) + 0.04 * span)

    fig.supxlabel("% of training", fontsize=18, y=0.13, x=0.528)
    fig.supylabel("Success rate gap (pp)", fontsize=18, x=0.0655, y=0.55)
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    for col_idx in range(n_cols):
        for row_idx in range(n_rows):
            axes[row_idx][col_idx].set_ylim(*ylim)
        ax = axes[1][col_idx]
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
            ncol=2,
            fontsize=19.0,
        )

    for ext in ("pdf", "png"):
        path = os.path.join(str(args.out_dir), f"{args.output_name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)

    print(f"bands: {int(CONFIDENCE * 100)}% paired Newcombe Method 10 intervals, "
          f"y limits {ylim[0]:.1f}..{ylim[1]:.1f} pp")
    if args.allow_order:
        print("X-VLA's LIBERO tables use the evaluation-order pairing assumption "
              "(--no_allow_order drops them)")
    for note in dict.fromkeys(notes):
        print(note)


if __name__ == "__main__":
    main()
