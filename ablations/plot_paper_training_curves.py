#!/usr/bin/env python3
"""Generate paper figures from the LIBERO checkpoint sweeps in `data/`."""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUT_DIR = REPO_ROOT / "figures" / "ablations"

# Same setting-file reader the rest of the repository uses.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from paired import load  # noqa: E402


OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}


def _load_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    return plt, PercentFormatter


COLORBLIND_SAFE_EXTRA = {
    "purple": "#882E72",
    "teal": "#1965B0",
    "cyan": "#4EB265",
    "rose": "#DC050C",
    "sand": "#F7F056",
    "brown": "#A6761D",
    "slate": "#666666",
    "lavender": "#9999CC",
    "magenta": "#EE6677",
    "indigo": "#4477AA",
    "mint": "#228833",
    "wine": "#AA3377",
}

COLORS = {
    "DiT": OKABE_ITO["blue"],
    "Pi0.5 scratchEverything": "cornflowerblue",
    "Pi0.5 pretrained": OKABE_ITO["orange"],
    "Pi0.5 pretrainedVLM": "m",
    "Pi0.5 scratchActionExpOnly": "lightseagreen",
    "FLOWER": OKABE_ITO["blue"],
    "X-VLA": OKABE_ITO["vermillion"],
}
MARKERS = {
    1: "o",
    4: "o",
    10: "o",
}
LINESTYLES = {
    1: "-",
    4: "-",
    10: "-",
}
SUITES = {
    "spatial": "Spatial",
    "object": "Object",
    "goal": "Goal",
    "libero_10": "LIBERO-10",
}
SUITE_ALIASES = {
    "spatial": "spatial",
    "libero_spatial": "spatial",
    "object": "object",
    "libero_object": "object",
    "goal": "goal",
    "libero_goal": "goal",
    "libero10": "libero_10",
    "libero_10": "libero_10",
    "libero-10": "libero_10",
    "10": "libero_10",
}


def _normal_variant(value):
    """Fold the misspelled scratchEveryting checkpoint into one series."""
    if value in {"scratchEverything", "scratchEveryting"}:
        return "scratchEverything"
    return value


def _normalize_suite_name(value):
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    key = SUITE_ALIASES.get(key)
    if key is None:
        valid = ", ".join(["all"] + sorted(SUITE_ALIASES))
        raise argparse.ArgumentTypeError(f"unknown suite '{value}'. Valid values include: {valid}")
    return key


def parse_results_arg(value):
    value = value.strip()
    if value.lower() == "all":
        return None

    suites = []
    seen = set()
    for part in value.split(","):
        if not part.strip():
            continue
        suite = _normalize_suite_name(part)
        if suite not in seen:
            suites.append(suite)
            seen.add(suite)

    if not suites:
        raise argparse.ArgumentTypeError("--results must be 'all' or a comma-separated suite list")
    return tuple(suites)


def _selected_success(row, selected_suites):
    if selected_suites is None:
        return row["overall_success"]
    suite_success = row["suite_success"]
    missing = [suite for suite in selected_suites if suite not in suite_success]
    if missing:
        labels = ", ".join(SUITES[suite] for suite in missing)
        raise ValueError(f"missing suite values for {row['model']} {row['variant']}: {labels}")
    return sum(suite_success[suite] for suite in selected_suites) / len(selected_suites)


def apply_results_selection(rows, selected_suites):
    for row in rows:
        row["success"] = _selected_success(row, selected_suites)


def result_suffix(selected_suites):
    if selected_suites is None:
        return ""
    return "_" + "_".join(selected_suites)


# ---------------------------------------------------------------------------
# Data loading
#
# Rows come from the normalized export in `data/` — the same setting files
# `scripts/plot.py` and `scripts/analyze.py` read — restricted to the
# `evaluation == "checkpoint"` sweeps on LIBERO. Per-suite rates use
# `task_counts` when the file carries them (the two aggregate-only Pi0.5
# settings have complete task counts but an incomplete episode transcript) and
# fall back to the episode rows otherwise.
# ---------------------------------------------------------------------------

DATA_SUITES = {
    "libero_spatial": "spatial",
    "libero_object": "object",
    "libero_goal": "goal",
    "libero_10": "libero_10",
}

MODEL_DISPLAY = {
    "pi05": "Pi0.5",
    "xvla": "X-VLA",
    "flower": "FLOWER",
    "dit": "DiT",
}

# DiT's last checkpoint is recorded as "final_model"; it is checkpoint_30000.
DIT_FINAL_TRAIN_STEP = 30000

# One curve point is not read from the checkpoint cohort. The X-VLA sweep's
# `ckpt-60000` rerun (xvla_indepth/runs/libero_rerun/…) scores ~12pp lower on
# LIBERO-10 than the primary evaluation of the same released checkpoint, and it
# is the primary evaluation the paper tables and figures report. Substituted
# explicitly so the mixed cohort is deliberate rather than accidental.
PRIMARY_SUBSTITUTES = {
    ("X-VLA", "pretrained", 60000): "libero__xvla__n{nsteps}.json",
}

_PI05_CHECKPOINT = re.compile(r"(?P<thousands>\d+)k(?:_(?P<variant>\w+))?$")
_DIT_CHECKPOINT = re.compile(r"(?P<variant>[A-Za-z]+)-(?P<train_step>\d+|final_model)$")


def _suite_success(data):
    """Per-suite success percentages, or None when a suite is missing."""
    successes = defaultdict(int)
    trials = defaultdict(int)
    counts = data.get("task_counts")
    if counts:
        for count in counts:
            suite = DATA_SUITES.get(count["suite"])
            if suite is None:
                continue
            successes[suite] += count["successes"]
            trials[suite] += count["trials"]
    else:
        for episode in data["episodes"]:
            suite = DATA_SUITES.get(episode["suite"])
            if suite is None or episode["success"] is None:
                continue
            successes[suite] += episode["success"]
            trials[suite] += 1
    if set(trials) != set(DATA_SUITES.values()) or not all(trials.values()):
        return None
    return {suite: 100.0 * successes[suite] / trials[suite] for suite in trials}


def _series_identity(setting):
    """Map a setting onto (display model, variant, train step), None to skip."""
    model = setting["model"]
    checkpoint = setting.get("checkpoint") or ""
    if model == "pi05":
        match = _PI05_CHECKPOINT.fullmatch(checkpoint)
        if match is None:
            return None
        variant = match.group("variant") or "pretrained"
        if variant == "libero10only":  # LIBERO-10 re-runs, not a training curve
            return None
        return "Pi0.5", _normal_variant(variant), 1000 * int(match.group("thousands"))
    if model == "dit":
        match = _DIT_CHECKPOINT.fullmatch(checkpoint)
        if match is None:
            return None
        train_step = match.group("train_step")
        return (
            "DiT",
            # "pretrained" here means the CLIP-initialized encoder, not robot
            # pretraining; the plots call this DiT series "scratch".
            "scratch" if match.group("variant") == "pretrained" else match.group("variant"),
            DIT_FINAL_TRAIN_STEP if train_step == "final_model" else int(train_step),
        )
    if model in ("xvla", "flower"):
        if not checkpoint.isdigit():
            return None
        return MODEL_DISPLAY[model], "pretrained", int(checkpoint)
    return None


def load_rows(data_dir):
    """Read the LIBERO checkpoint sweeps out of the normalized setting files."""
    data_dir = Path(data_dir)
    rows = []
    for path in sorted(data_dir.glob("libero__*__checkpoint__*.json")):
        data = load(path)
        setting = data["setting"]
        if setting["benchmark"] != "libero" or setting["evaluation"] != "checkpoint":
            continue
        identity = _series_identity(setting)
        if identity is None:
            continue
        substitute = PRIMARY_SUBSTITUTES.get(identity)
        if substitute is not None:
            data = load(data_dir / substitute.format(nsteps=setting["nsteps"]))
        suite_success = _suite_success(data)
        if suite_success is None:
            continue
        model, variant, train_step = identity
        rows.append(
            {
                "model": model,
                "variant": variant,
                "train_step": train_step,
                "inference_steps": setting["nsteps"],
                "overall_success": sum(suite_success.values()) / len(suite_success),
                "suite_success": suite_success,
            }
        )
    if not rows:
        raise ValueError(f"No LIBERO checkpoint settings found in {data_dir}")
    return rows


def add_training_percent(rows):
    max_steps = defaultdict(int)
    for row in rows:
        series_key = (row["model"], row["variant"])
        max_steps[series_key] = max(max_steps[series_key], row["train_step"])

    for row in rows:
        series_key = (row["model"], row["variant"])
        row["training_pct"] = 100.0 * row["train_step"] / max_steps[series_key]


def _series(rows, model, variant, inference_steps):
    return sorted(
        (
            row
            for row in rows
            if row["model"] == model
            and row["variant"] == variant
            and row["inference_steps"] == inference_steps
        ),
        key=lambda row: row["training_pct"],
    )


def _line_label(display_name, inference_steps, legend_label=None):
    if legend_label is not None:
        return legend_label
    return f"{display_name}, n={inference_steps}"


def plot_series(rows, specs, output_base):
    plt, PercentFormatter = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    plotted = 0
    for spec in specs:
        subset = _series(rows, spec["model"], spec["variant"], spec["inference_steps"])
        if not subset:
            raise ValueError(f"No data for plot series: {spec}")
        display_name = spec["label"]
        ax.plot(
            [row["training_pct"] for row in subset],
            [row["success"] for row in subset],
            label=_line_label(display_name, spec["inference_steps"], spec.get("legend_label")),
            color=COLORS[display_name],
            marker=MARKERS[spec["inference_steps"]],
            linestyle=LINESTYLES[spec["inference_steps"]],
            linewidth=2.4,
            markersize=5.5,
        )
        plotted += 1

    if plotted == 0:
        raise ValueError(f"No series plotted for {output_base}")

    ax.set_xlabel("% of training")
    ax.set_ylabel("Success rate (%)")
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(alpha=0.25, linewidth=0.8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        # fontsize=8.0,
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))

    for ext in ("pdf", "png"):
        path = f"{output_base}.{ext}"
        fig.savefig(path, dpi=300)
        print(f"wrote {path}")
    plt.close(fig)


def plot_step_diff(rows, specs, output_base, legend_args=None):
    plt, PercentFormatter = _load_matplotlib()
    if legend_args is None:
        legend_args = {}
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    plotted = 0
    diff_values = []
    for spec in specs:
        baseline = {
            row["train_step"]: row
            for row in _series(rows, spec["model"], spec["variant"], 1)
        }
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
            linestyle="-",
            linewidth=2.4,
            markersize=5.5,
        )
        plotted += 1

    if plotted == 0:
        raise ValueError(f"No series plotted for {output_base}")

    ax.axhline(0.0, color="0.35", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("% of training")
    if legend_args.get("show_y_label", True):
        ax.set_ylabel("Success rate gap (pp)")
    ax.set_xlim(0, 102)
    ymin = min(diff_values + [0.0])
    ymax = max(diff_values + [0.0])
    padding = max(2.0, 0.12 * (ymax - ymin))
    # ax.set_ylim(ymin - padding, ymax + padding)
    ax.set_ylim(-20, 15)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(alpha=0.25, linewidth=0.8)
    handles, labels = ax.get_legend_handles_labels()
    if legend_args.get("show_legend", True):
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_args.get("legend_y", 0.0)),
            frameon=False,
            ncol=legend_args.get("ncol", 2),
            fontsize=legend_args.get("fontsize", 8.0),
        )
        fig.tight_layout(rect=(0, 0.18, 1, 1))

    for ext in ("pdf", "png"):
        path = f"{output_base}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


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
        "--results",
        default=None,
        type=parse_results_arg,
        help=(
            "Result metric to plot: 'all' for current aggregate values, one suite "
            "(spatial, object, goal, libero_10), or a comma-separated suite subset."
        ),
    )
    parser.add_argument(
        "--show_legend",
        action="store_true",
        help="Whether to include the legend in the generated figure.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.data)
    add_training_percent(rows)
    apply_results_selection(rows, args.results)
    suffix = result_suffix(args.results)

    plot_series(
        rows,
        [
            {"model": "DiT", "variant": "scratch", "inference_steps": 1, "label": "DiT"},
            {"model": "DiT", "variant": "scratch", "inference_steps": 10, "label": "DiT"},
            # {
            #     "model": "Pi0.5",
            #     "variant": "scratchEverything",
            #     "inference_steps": 1,
            #     "label": "Pi0.5 scratchEverything",
            #     "legend_label": r"$\pi_{0.5}$ n=1",
            # },
            # {
            #     "model": "Pi0.5",
            #     "variant": "scratchEverything",
            #     "inference_steps": 10,
            #     "label": "Pi0.5 scratchEverything",
            #     "legend_label": r"$\pi_{0.5}$ n=10",
            # },
            {"model": "Pi0.5", "variant": "pretrainedVLM", "inference_steps": 1, "label": "Pi0.5 pretrainedVLM", "legend_label": r"Paligemma VLM n=1"},
            {"model": "Pi0.5", "variant": "pretrainedVLM", "inference_steps": 10, "label": "Pi0.5 pretrainedVLM", "legend_label": r"Paligemma VLM n=10"},
        ],
        os.path.join(str(args.out_dir), f"plot1_scratch_dit_pi05_success{suffix}"),
    )
    plot_step_diff(
        rows,
        [
            {"model": "DiT", "variant": "scratch", "reference_inference_steps": 10, "label": "DiT"},
            # {
            #     "model": "Pi0.5",
            #     "variant": "scratchEverything",
            #     "reference_inference_steps": 10,
            #     "label": "Pi0.5 scratchEverything",
            #     "legend_label": r"$\pi_{0.5}$ Scratch",
            # },
            {
                "model": "Pi0.5",
                "variant": "pretrainedVLM",
                "reference_inference_steps": 10,
                "label": "Pi0.5 pretrainedVLM",
                "legend_label": r"Paligemma VLM",
            },
        ],
        os.path.join(str(args.out_dir), f"plot1_scratch_dit_pi05_success_diff{suffix}"),
        legend_args={
            "ncol": 2, "legend_y": 0.1, "fontsize": 11.0,
            "show_y_label": True, "show_legend": args.show_legend,
            },
    )

    plot_series(
        rows,
        [
            {"model": "Pi0.5", "variant": "pretrained", "inference_steps": 1, "label": "Pi0.5 pretrained", "legend_label": r"$\pi_{0.5}$ base n=1"},
            {"model": "Pi0.5", "variant": "pretrained", "inference_steps": 10, "label": "Pi0.5 pretrained", "legend_label": r"$\pi_{0.5}$ base n=10"},
            {"model": "Pi0.5", "variant": "pretrainedVLM", "inference_steps": 1, "label": "Pi0.5 pretrainedVLM", "legend_label": r"Paligemma VLM n=1"},
            {"model": "Pi0.5", "variant": "pretrainedVLM", "inference_steps": 10, "label": "Pi0.5 pretrainedVLM", "legend_label": r"Paligemma VLM n=10"},
            {
                "model": "Pi0.5",
                "variant": "scratchActionExpOnly",
                "inference_steps": 1,
                "label": "Pi0.5 scratchActionExpOnly",
                "legend_label": r"$\pi_{0.5}$ VLM n=1",
            },
            {
                "model": "Pi0.5",
                "variant": "scratchActionExpOnly",
                "inference_steps": 10,
                "label": "Pi0.5 scratchActionExpOnly",
                "legend_label": r"$\pi_{0.5}$ VLM n=10",
            },
        ],
        os.path.join(str(args.out_dir), f"plot2_pi05_initialization_success{suffix}"),
    )
    plot_step_diff(
        rows,
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
        os.path.join(str(args.out_dir), f"plot2_pi05_initialization_success_diff{suffix}"),
        legend_args={
            "ncol": 2, "legend_y": 0.0, "fontsize": 11,
            "show_y_label": False, "show_legend": args.show_legend,
            },
    )

    plot_series(
        rows,
        [
            {"model": "Pi0.5", "variant": "pretrained", "inference_steps": 1, "label": "Pi0.5 pretrained", "legend_label": r"$\pi_{0.5}$ n=1"},
            {"model": "Pi0.5", "variant": "pretrained", "inference_steps": 10, "label": "Pi0.5 pretrained", "legend_label": r"$\pi_{0.5}$ n=10"},
            {"model": "FLOWER", "variant": "pretrained", "inference_steps": 1, "label": "FLOWER"},
            {"model": "FLOWER", "variant": "pretrained", "inference_steps": 4, "label": "FLOWER"},
            {"model": "X-VLA", "variant": "pretrained", "inference_steps": 1, "label": "X-VLA"},
            {"model": "X-VLA", "variant": "pretrained", "inference_steps": 10, "label": "X-VLA"},
        ],
        os.path.join(str(args.out_dir), f"plot3_pretrained_model_success{suffix}"),
    )
    plot_step_diff(
        rows,
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
        os.path.join(str(args.out_dir), f"plot3_pretrained_model_success_diff{suffix}"),
        legend_args={
            "ncol": 3, "legend_y": 0.1, "fontsize": 11,
            "show_y_label": False, "show_legend": args.show_legend,
        },
    )


if __name__ == "__main__":
    main()
