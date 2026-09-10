#!/usr/bin/env python3
"""Generate the per-suite and pooled S/G/O flow-ablation scatter panels.

By default both retained directories are regenerated. Use --mode per-suite or
--mode sgo-avg to render one. Shared loaders are kept with the plotting code.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Optional

ROOT = (Path(__file__).resolve().parents[0])
sys.path.append(str(ROOT))
from wilson_score_interval import wilson_score_interval  # noqa: E402

# --- visual encoding -----------------------------------------------------
MODEL_ORDER = ["DiT", "SmolVLA", "X-VLA", "Flower", "Pi0.5", "LAP", "XR-0"]
MODEL_SIZE_B = {"DiT":0.21, "SmolVLA": 0.45, "X-VLA": 0.90, "Flower": 0.95,
                "Pi0.5": 2.30, "LAP": 3.00, "XR-0": 4.70}
MODEL_MARKERS = {"DiT": "X", "SmolVLA": "v", "X-VLA": "p", "Flower": "s",
                 "Pi0.5": "D", "LAP": "^", "XR-0": "P"}
MODEL_LEGEND = {"DiT": "DiT", "SmolVLA": "SmolVLA", "X-VLA": "X-VLA", "Flower": "FLOWER",
                "Pi0.5": r"$\pi_{0.5}$", "LAP": "LAP", "XR-0": "XR-0"}
BENCH_ORDER = ["CALVIN", "LIBERO", "LIBERO+",
               "RoboTwin Clean", "RoboTwin Randomized", "Real-world"]
BENCH_COLORS = {"CALVIN": "#0072B2", "LIBERO": "#D55E00", "LIBERO+": "#009E73",
                "RoboTwin Clean": "#56B4E9",
                "RoboTwin Randomized": "#E69F00",
                "Real-world": "#CC79A7"}


@dataclass
class PairedPoint:
    benchmark: str
    model: str
    sub: str          # "" for aggregate, otherwise suite/task name
    one_steps: int
    max_steps: int
    succ_one: int
    n_one: int
    succ_max: int
    n_max: int
    metric: str
    # paired episode-level counts (only for real-world); else None.
    paired_b_one_wins: Optional[int] = None
    paired_c_max_wins: Optional[int] = None
    paired_both_succ: Optional[int] = None
    paired_both_fail: Optional[int] = None

    @property
    def is_aggregate(self) -> bool:
        return self.sub == ""

    @property
    def rate_one(self) -> float:
        return self.succ_one / self.n_one if self.n_one else 0.0

    @property
    def rate_max(self) -> float:
        return self.succ_max / self.n_max if self.n_max else 0.0

    @property
    def delta_pp(self) -> float:
        return (self.rate_one - self.rate_max) * 100.0

    def delta_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        """95% CI on (p1 - pmax) in percentage points."""
        # Use paired contingency when available (real-world).
        if self.paired_b_one_wins is not None:
            n = (self.paired_both_succ + self.paired_both_fail
                 + self.paired_b_one_wins + self.paired_c_max_wins)
            if n == 0:
                return 0.0, 0.0
            b, c = self.paired_b_one_wins, self.paired_c_max_wins
            d = (b - c) / n
            var = (b + c) / n**2 - (b - c) ** 2 / n**3
            z = NormalDist().inv_cdf(0.5 + confidence / 2)
            half = z * math.sqrt(max(var, 0.0))
            return (d - half) * 100, (d + half) * 100
        # Unpaired two-proportion Wald (per-point CI).
        p1, n1 = self.rate_one, self.n_one
        p2, n2 = self.rate_max, self.n_max
        if not (n1 and n2):
            return 0.0, 0.0
        z = NormalDist().inv_cdf(0.5 + confidence / 2)
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        half = z * se * 100
        return (p1 - p2) * 100 - half, (p1 - p2) * 100 + half

    def pvalue(self) -> float:
        """McNemar p when paired, else two-proportion z."""
        if self.paired_b_one_wins is not None:
            b, c = self.paired_b_one_wins, self.paired_c_max_wins
            if b + c == 0:
                return 1.0
            # exact two-sided binomial (McNemar).
            k = min(b, c)
            n = b + c
            # P(X<=k) * 2  (mid-p style not used; classic exact).
            p = 0.0
            for i in range(0, k + 1):
                p += math.comb(n, i) * 0.5**n
            return min(1.0, 2 * p)
        p1, n1 = self.rate_one, self.n_one
        p2, n2 = self.rate_max, self.n_max
        if not (n1 and n2):
            return 1.0
        p_pool = (self.succ_one + self.succ_max) / (n1 + n2)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        if se == 0:
            return 1.0
        z = (p1 - p2) / se
        return 2 * (1 - NormalDist().cdf(abs(z)))


# --- loaders -------------------------------------------------------------
def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def model_key(model: str) -> tuple[int, str]:
    return (MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER),
            model)


def model_size_b(model: str) -> Optional[float]:
    return MODEL_SIZE_B.get(model)


def model_size_text(model: str) -> str:
    size = model_size_b(model)
    return f"{size:.2f}" if size is not None else "n/a"


def model_legend(model: str) -> str:
    return MODEL_LEGEND.get(model, model)


def load_libero_family() -> tuple[list[PairedPoint], list[PairedPoint]]:
    """Returns (aggregate, per-suite) paired points for LIBERO + LIBERO+."""
    mod = _import(ROOT / "libero" / "analyze_libero_steps.py", "lab_libero")
    libero_dir = ROOT / "libero"
    runs = [mod.load_run(libero_dir, s) for s in mod.RUN_SPECS]
    bench_label = {"libero": "LIBERO", "liberoplus": "LIBERO+"}
    SUITE_LABEL = {"libero_spatial": "Spatial", "libero_object": "Object",
                   "libero_goal": "Goal", "libero_10": "LIBERO-10"}

    agg, sub = [], []
    for dataset in ("libero", "liberoplus"):
        for mk in mod.MODEL_ORDER:
            mruns = sorted([r for r in runs
                            if r.spec.dataset == dataset and r.spec.model_key == mk],
                           key=lambda r: r.spec.steps)
            if len(mruns) < 2 or mruns[0].spec.steps != 1:
                continue
            one, mx = mruns[0], mruns[-1]
            agg.append(PairedPoint(
                benchmark=bench_label[dataset],
                model=mod.DISPLAY_NAMES[mk], sub="",
                one_steps=one.spec.steps, max_steps=mx.spec.steps,
                succ_one=one.overall.successes, n_one=one.overall.trials,
                succ_max=mx.overall.successes, n_max=mx.overall.trials,
                metric="Overall success rate",
            ))
            for suite_key, suite_lbl in SUITE_LABEL.items():
                c1 = one.suite_counts[suite_key]
                cm = mx.suite_counts[suite_key]
                sub.append(PairedPoint(
                    benchmark=bench_label[dataset],
                    model=mod.DISPLAY_NAMES[mk], sub=suite_lbl,
                    one_steps=one.spec.steps, max_steps=mx.spec.steps,
                    succ_one=c1.successes, n_one=c1.trials,
                    succ_max=cm.successes, n_max=cm.trials,
                    metric=f"Suite success rate ({suite_lbl})",
                ))
    return agg, sub


def load_real_world() -> tuple[list[PairedPoint], list[PairedPoint]]:
    """(aggregate per model, per-task) paired points for real-world rollouts.

    The real-world logs are organized slightly differently by model, so discover
    every summary below real_world/<model>/.../<task>/paired_*/summary.json and
    pool all batches for the same model/task.
    """
    from collections import defaultdict

    MODEL_LABEL = {
        "dit": "DiT",
        "flower": "Flower",
        "lap": "LAP",
        "openpi": "Pi0.5",
        "pi05": "Pi0.5",
        "xvla": "X-VLA",
        "mibot": "XR-0",
    }
    TASK_LABEL = {"spoon_drawer": "Spoon→Drawer", "open_drawer": "Open Drawer",
                  "open_oven": "Open Oven", "close_oven": "Close Oven"}
    bench = "Real-world"
    agg, sub = [], []
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    real_root = ROOT / "real_world"
    for summary_path in sorted(real_root.glob("*/**/paired_*/summary.json")):
        rel = summary_path.relative_to(real_root)
        model_key_raw = rel.parts[0]
        # Only the per-model directories are valid; skip anything else swept up
        # by the recursive glob (e.g. the archived ``old_logs/`` tree).
        if model_key_raw not in MODEL_LABEL:
            continue
        task = summary_path.parent.parent.name
        grouped[(MODEL_LABEL.get(model_key_raw, model_key_raw), task)].append(summary_path)

    for model in sorted({m for m, _ in grouped}, key=model_key):
        ts1 = ns1 = tsm = nsm = 0
        b_tot = c_tot = bs_tot = bf_tot = 0
        model_max_steps: set[int] = set()
        for _, task in sorted(k for k in grouped if k[0] == model):
            s1 = n1 = sM = nM = 0
            b = c = bs = bf = 0
            task_max_steps: set[int] = set()
            for summary_path in grouped[(model, task)]:
                summary = json.load(open(summary_path))
                # Skip interrupted/aborted batches that completed no pairs
                # (empty ``conditions``); these are restart artifacts.
                if not summary.get("num_completed_pairs") or not summary["conditions"]:
                    continue
                step_keys = sorted(int(k) for k in summary["conditions"])
                if 1 not in step_keys or len(step_keys) < 2:
                    raise ValueError(f"Expected paired 1/max-step conditions in {summary_path}")
                max_step = max(k for k in step_keys if k != 1)
                task_max_steps.add(max_step)
                model_max_steps.add(max_step)
                c1 = summary["conditions"]["1"]
                cM = summary["conditions"][str(max_step)]
                pr = summary["paired"]
                s1 += int(c1["num_successes"])
                n1 += int(c1["num_runs"])
                sM += int(cM["num_successes"])
                nM += int(cM["num_runs"])
                wins = pr.get("wins_by_num_flow_steps", {})
                b += int(wins.get("1", 0))
                c += int(wins.get(str(max_step), 0))
                bs += int(pr["ties_success"])
                bf += int(pr["ties_failure"])
            if len(task_max_steps) != 1:
                raise ValueError(f"Mixed max-step conditions for {model}/{task}: {task_max_steps}")
            max_steps = task_max_steps.pop()
            sub.append(PairedPoint(
                benchmark=bench, model=model, sub=TASK_LABEL.get(task, task),
                one_steps=1, max_steps=max_steps,
                succ_one=s1, n_one=n1, succ_max=sM, n_max=nM,
                metric="Manual success",
                paired_b_one_wins=b, paired_c_max_wins=c,
                paired_both_succ=bs, paired_both_fail=bf,
            ))
            ts1 += s1; ns1 += n1; tsm += sM; nsm += nM
            b_tot += b; c_tot += c; bs_tot += bs; bf_tot += bf
        if not ns1:
            continue
        if len(model_max_steps) != 1:
            raise ValueError(f"Mixed max-step conditions for {model}: {model_max_steps}")
        agg.append(PairedPoint(
            benchmark=bench, model=model, sub="",
            one_steps=1, max_steps=model_max_steps.pop(),
            succ_one=ts1, n_one=ns1, succ_max=tsm, n_max=nsm,
            metric="Manual success (pooled across tasks)",
            paired_b_one_wins=b_tot, paired_c_max_wins=c_tot,
            paired_both_succ=bs_tot, paired_both_fail=bf_tot,
        ))
    return agg, sub


def load_robotwin() -> tuple[list[PairedPoint], list[PairedPoint]]:
    """RoboTwin X-VLA paired 1-flow-step vs 10-flow-step results."""
    paired_logs = ROOT / "robotwin" / "X-VLA" / "paired_logs"
    if str(paired_logs) not in sys.path:
        sys.path.insert(0, str(paired_logs))
    from analysis.config import SETTING_DISPLAY, default_setting_specs  # type: ignore
    from analysis.io_runs import load_paired_outcomes  # type: ignore
    from analysis.pairs import apply_corrections, build_task_result  # type: ignore

    agg, sub = [], []
    for setting, (low_spec, high_spec) in default_setting_specs().items():
        bench = SETTING_DISPLAY[setting]
        s1 = sm = n = b = c = bs = bf = 0
        outcomes, skipped = load_paired_outcomes(low_spec, high_spec)
        tasks = apply_corrections([build_task_result(o) for o in outcomes])
        if skipped:
            print(f"RoboTwin {setting}: skipped {len(skipped)} unpaired tasks: {', '.join(skipped)}")
        for task in tasks:
            one_only = task.mcnemar.b
            max_only = task.mcnemar.c
            both_succ = task.mcnemar.both_success
            both_fail = task.mcnemar.both_failure
            sub.append(PairedPoint(
                benchmark=bench, model="X-VLA",
                sub=task.task.replace("_", " "),
                one_steps=1, max_steps=10,
                succ_one=task.successes_low, n_one=task.n,
                succ_max=task.successes_high, n_max=task.n,
                metric="RoboTwin paired success",
                paired_b_one_wins=one_only,
                paired_c_max_wins=max_only,
                paired_both_succ=both_succ,
                paired_both_fail=both_fail,
            ))
            s1 += task.successes_low
            sm += task.successes_high
            n += task.n
            b += one_only
            c += max_only
            bs += both_succ
            bf += both_fail
        agg.append(PairedPoint(
            benchmark=bench, model="X-VLA", sub="",
            one_steps=1, max_steps=10,
            succ_one=s1, n_one=n, succ_max=sm, n_max=n,
            metric="RoboTwin paired success (pooled across tasks)",
            paired_b_one_wins=b,
            paired_c_max_wins=c,
            paired_both_succ=bs,
            paired_both_fail=bf,
        ))
    return agg, sub




import argparse
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "flow_ablation_crossbench" / "perfamily"  # rebound in main()


MODEL_MARKERS["Pi0.5"] = "*"


def model_marker(model: str) -> str:
    return MODEL_MARKERS.get(model, "X")


# Palette: drawn from Paul Tol's "muted" (deuteranopia/protanopia-safe) plus
# one strong orange. The 10 colors are kept maximally distinct in hue,
# luminance, or both, so each panel's points and the unified legend stay
# readable for the common colorblind types.

# Real-world tasks.
TASK_COLORS = {
    "Spoon→Drawer": "#332288",  # indigo
    "Open Drawer":      "#44AA99",  # teal
    "Open Oven":        "#AA4499",  # purple
    "Close Oven":       "#999933",  # olive
}

# RoboTwin suites (local override; does not mutate the imported module).
BENCH_COLORS = {
    "Clean":      "#88CCEE",  # light cyan
    "Randomized": "#FE6100",  # vivid orange
}

# LIBERO suites. "SGO avg" reuses Spatial's indigo since it never co-occurs
# with Spatial/Object/Goal in a panel.
SUITE_COLORS = {
    "Spatial":    "#882255",  # wine
    "Object":     "#117733",  # dark green
    "Goal":       "#CC6677",  # rose
    "LIBERO-10":  "#DDCC77",  # sand
    "SGO avg":    "#882255",
}
SUITE_ORDER = ["Spatial", "Object", "Goal", "LIBERO-10"]
SGO_SUITES = {"Spatial", "Object", "Goal"}

# CALVIN ABC->D chain position palette. Sampled from viridis (perceptually
# uniform + colorblind-safe). SR1 = dark purple ... SR5 = bright yellow.
CALVIN_POS_COLORS = {
    "SR1": "#440154",
    "SR2": "#3B528B",
    "SR3": "#21918C",
    "SR4": "#5EC962",
    "SR5": "#FDE725",
}
CALVIN_POS_ORDER = ["SR1", "SR2", "SR3", "SR4", "SR5"]


plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "axes.linewidth": 0.7,
    "lines.markersize": 5,
})


PANEL_SIZE = (2.15, 2.15)
MARKER_S = 22
EDGE = "#111827"

# Uniform canvas for every panel PDF/PNG so they scale identically in LaTeX.
# AX_POS keeps the scatter axes in the exact same spot across all four panels;
# the legend lives in the lower band [0, ~0.40] of the figure.
PANEL_FIG_SIZE_NO_YLABEL = (2.4, 2.8)
# LIBERO carries the y-label + y-tick margin (~0.4 in), so its canvas is
# proportionally wider — when the user gives every panel the same LaTeX
# width, the actual scatter plots render at the same size across panels.
PANEL_FIG_SIZE_WITH_YLABEL = (2.85, 2.8)
# Both AX_POS values share the same axes height (0.55) and axes-top
# (fig y = 0.88) so the scatters are the same physical size in inches and
# the titles align vertically.
AX_POS_WITH_YLABEL = (0.14, 0.33, 0.80, 0.55)
AX_POS_NO_YLABEL   = (0.05, 0.33, 0.90, 0.55)


def _save_png_pdf(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")


def _new_panel_fig(title: str = "", show_ylabel: bool = True):
    """Create a panel figure with the locked-down canvas + axes position."""
    figsize = PANEL_FIG_SIZE_WITH_YLABEL if show_ylabel else PANEL_FIG_SIZE_NO_YLABEL
    pos = AX_POS_WITH_YLABEL if show_ylabel else AX_POS_NO_YLABEL
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(pos)
    _setup_axes(ax, title=title, show_ylabel=show_ylabel)
    return fig, ax


def _save_panel(fig, name: str) -> None:
    """Save without bbox_inches='tight' so every panel PDF is byte-identical
    in dimensions — guaranteeing equal sizes when scaled in LaTeX."""
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def _setup_axes(ax, title: str = "", show_ylabel: bool = True) -> None:
    lim = [-3, 103]
    if NOTE_BAND and NOTE_STYLE in ("symmetric", "tost"):
        # Shade the ±NOTE_PP band around the y = x diagonal.
        lo, hi = lim
        ax.fill_between([lo, hi],
                        [lo - NOTE_PP, hi - NOTE_PP],
                        [lo + NOTE_PP, hi + NOTE_PP],
                        color="#9CA3AF", alpha=0.22, linewidth=0,
                        zorder=0.5)
    ax.plot(lim, lim, "k--", lw=0.8, zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("ref-step SR (%)", labelpad=1)
    if show_ylabel:
        ax.set_ylabel("1-step SR (%)", labelpad=1)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    if title:
        ax.set_title(title, pad=4)
    ax.tick_params(length=2.5, width=0.6)


def _scatter_bench(ax, points) -> None:
    for p in points:
        ax.scatter(p.rate_max * 100, p.rate_one * 100,
                   s=MARKER_S,
                   marker=model_marker(p.model),
                   color=BENCH_COLORS[p.benchmark],
                   edgecolor=EDGE, linewidth=0.4,
                   alpha=0.92, zorder=3)


# Diagonal-note settings, set by main() from --note and --note-pp CLI flags.
#   style="no-worse"  -> P(ref-step - 1-step <= k pp)  (1-step within k pp
#                        worse, ties and 1-step wins included)
#   style="symmetric" -> P(|1-step - ref-step| <= k pp)
NOTE_STYLE = "no-worse"
NOTE_PP = 2.0
NOTE_BAND = False   # only honored when NOTE_STYLE == "symmetric"


def _add_diagonal_note(ax, points) -> None:
    """Annotate the panel with one cumulative fraction over the paired
    (ref-step, 1-step) SRs. Box sits in the lower-right of the scatter."""
    n = len(points)
    if not n:
        return
    k = NOTE_PP
    k_label = f"{k:g}"
    if NOTE_STYLE == "tost":
        # Equivalence (two one-sided tests): a point counts only if its whole
        # 95% CI on Δ lies within [-k, +k] -- i.e. the trials are precise
        # enough to certify equivalence at margin k. Small-n points with wide
        # CIs are (honestly) not counted.
        n_eq = 0
        for p in points:
            lo, hi = p.delta_ci()
            if lo >= -k and hi <= k:
                n_eq += 1
        text = f"|Δ|≤{k_label} pp (95% CI): {n_eq/n*100:.0f}%"
    elif NOTE_STYLE == "symmetric":
        vals = [abs(p.rate_one - p.rate_max) * 100 for p in points]
        text = f"|Δ|≤{k_label} pp: {sum(v <= k for v in vals)/n*100:.0f}%"
    else:  # "no-worse"
        vals = [(p.rate_one - p.rate_max) * 100 for p in points]
        text = (f"Δ≥-{k_label} pp: "
                f"{sum(v >= -k for v in vals)/n*100:.0f}%")
    ax.text(0.96, 0.04, text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#9CA3AF", linewidth=0.5, alpha=0.85),
            zorder=4)


def load_calvin_positions() -> list:
    """One paired point per (model, chain position) for CALVIN ABC->D.

    Returns 3 models x 5 positions = 15 PairedPoint objects. The trial counts
    are reconstructed from `task_info["total"]` and the chain SR ladder (same
    logic as `load_calvin`).
    """
    base = ROOT / "calvin" / "abc_d"
    specs = [
        ("Flower", 1, 4, base / "flower" / "abc_d_1step" / "results.json",
                       base / "flower" / "abc_d_4step" / "results.json"),
        ("X-VLA",  1, 10, base / "X-VLA" / "1step" / "results.json",
                        base / "X-VLA" / "10step" / "results.json"),
        ("XR-0",  1, 5, base / "mibot" / "rerun_abc_d_numsteps1" / "results.json",
                       base / "mibot" / "rerun_abc_d_numsteps5" / "results.json"),
    ]

    def n_chains(d):
        k = next(iter(d))
        chain = d[k]["chain_sr"]
        tot = sum(v["total"] for v in d[k]["task_info"].values())
        denom = 1 + sum(float(chain[str(i)]) for i in (1, 2, 3, 4))
        return int(round(tot / denom))

    pts: list = []
    for model, s1, sm, p1, pm in specs:
        d1 = json.load(open(p1))
        dm = json.load(open(pm))
        k1, km = next(iter(d1)), next(iter(dm))
        n1, nm = n_chains(d1), n_chains(dm)
        for pos in range(1, 6):
            sr1 = float(d1[k1]["chain_sr"][str(pos)])
            srm = float(dm[km]["chain_sr"][str(pos)])
            pts.append(PairedPoint(
                benchmark="CALVIN", model=model, sub=f"SR{pos}",
                one_steps=s1, max_steps=sm,
                succ_one=int(round(sr1 * n1)), n_one=n1,
                succ_max=int(round(srm * nm)), n_max=nm,
                metric=f"Chain SR@{pos}",
            ))
    return pts


def pool_realworld_by_model(points: list) -> list:
    """Pool real-world task trials per model into one point.

    Returns 6 PairedPoints (one per model), with the paired McNemar counts
    summed across tasks so the CI/p-value stay paired.
    """
    pooled: dict[str, dict] = defaultdict(
        lambda: {"succ_one": 0, "n_one": 0, "succ_max": 0, "n_max": 0,
                 "b": 0, "c": 0, "bs": 0, "bf": 0, "template": None})
    for p in points:
        d = pooled[p.model]
        d["succ_one"] += p.succ_one
        d["n_one"] += p.n_one
        d["succ_max"] += p.succ_max
        d["n_max"] += p.n_max
        if p.paired_b_one_wins is not None:
            d["b"] += p.paired_b_one_wins
            d["c"] += p.paired_c_max_wins
            d["bs"] += p.paired_both_succ
            d["bf"] += p.paired_both_fail
        d["template"] = p
    out: list = []
    for model, d in pooled.items():
        t = d["template"]
        out.append(PairedPoint(
            benchmark=t.benchmark, model=model, sub="all tasks",
            one_steps=t.one_steps, max_steps=t.max_steps,
            succ_one=d["succ_one"], n_one=d["n_one"],
            succ_max=d["succ_max"], n_max=d["n_max"],
            metric="Real-world success rate pooled across tasks",
            paired_b_one_wins=d["b"], paired_c_max_wins=d["c"],
            paired_both_succ=d["bs"], paired_both_fail=d["bf"],
        ))
    return out


def pool_sgo(points: list) -> list:
    """Pool Spatial+Object+Goal trials per (benchmark, model) into one point.

    Keeps LIBERO-10 points untouched. Returns 7 models x 2 datasets x 2
    sub-categories = 28 points.
    """
    pooled: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"succ_one": 0, "n_one": 0, "succ_max": 0, "n_max": 0,
                 "template": None})
    out: list = []
    for p in points:
        if p.sub == "LIBERO-10":
            out.append(p)
            continue
        if p.sub in SGO_SUITES:
            d = pooled[(p.benchmark, p.model)]
            d["succ_one"] += p.succ_one
            d["n_one"] += p.n_one
            d["succ_max"] += p.succ_max
            d["n_max"] += p.n_max
            d["template"] = p
    for (bench, model), d in pooled.items():
        t = d["template"]
        out.append(PairedPoint(
            benchmark=bench, model=model, sub="SGO avg",
            one_steps=t.one_steps, max_steps=t.max_steps,
            succ_one=d["succ_one"], n_one=d["n_one"],
            succ_max=d["succ_max"], n_max=d["n_max"],
            metric="Pooled Spatial+Object+Goal success rate",
        ))
    return out


def panel_libero(points, mode: str) -> None:
    expected = 28 if mode == "sgo-avg" else 56
    assert len(points) == expected, (
        f"expected {expected} LIBERO points in {mode} mode, got {len(points)}")
    fig, ax = _new_panel_fig(title="LIBERO / LIBERO+", show_ylabel=True)
    for p in points:
        is_libero = p.benchmark == "LIBERO"
        ax.scatter(p.rate_max * 100, p.rate_one * 100,
                   s=MARKER_S,
                   marker=model_marker(p.model),
                   color=SUITE_COLORS[p.sub],
                   edgecolor=EDGE if is_libero else "none",
                   linewidth=0.5 if is_libero else 0.0,
                   alpha=0.92, zorder=3)
    suite_labels = (["SGO avg", "LIBERO-10"] if mode == "sgo-avg"
                    else SUITE_ORDER)
    handles = [_swatch(SUITE_COLORS[s], s) for s in suite_labels]
    handles.append(_swatch("#9CA3AF", "LIBERO", with_edge=True))
    handles.append(_swatch("#9CA3AF", "LIBERO+", with_edge=False))
    _legend_below(ax, handles, ncol=2)
    _add_diagonal_note(ax, points)
    _save_panel(fig, "panel_libero")


def panel_robotwin(points) -> None:
    assert len(points) == 100, f"expected 100 RoboTwin points, got {len(points)}"
    fig, ax = _new_panel_fig(title="RoboTwin", show_ylabel=False)
    _scatter_bench(ax, points)
    handles = [
        _swatch(BENCH_COLORS["Clean"], "Clean"),
        _swatch(BENCH_COLORS["Randomized"], "Randomized"),
    ]
    _legend_below(ax, handles, ncol=1)
    _add_diagonal_note(ax, points)
    _save_panel(fig, "panel_robotwin")


def panel_calvin(points) -> None:
    assert len(points) == 15, f"expected 15 CALVIN points, got {len(points)}"
    fig, ax = _new_panel_fig(title="CALVIN ABC→D", show_ylabel=False)
    for p in points:
        ax.scatter(p.rate_max * 100, p.rate_one * 100,
                   s=MARKER_S,
                   marker=model_marker(p.model),
                   color=CALVIN_POS_COLORS[p.sub],
                   edgecolor=EDGE, linewidth=0.4,
                   alpha=0.92, zorder=3)
    handles = [_swatch(CALVIN_POS_COLORS[p], p) for p in CALVIN_POS_ORDER]
    _legend_below(ax, handles, ncol=3)
    _add_diagonal_note(ax, points)
    _save_panel(fig, "panel_calvin")


def panel_realworld(points) -> None:
    assert len(points) == 24, f"expected 24 real-world points, got {len(points)}"
    fig, ax = _new_panel_fig(title="Real world", show_ylabel=False)
    for p in points:
        ax.scatter(p.rate_max * 100, p.rate_one * 100,
                   s=MARKER_S,
                   marker=model_marker(p.model),
                   color=TASK_COLORS[p.sub],
                   edgecolor=EDGE, linewidth=0.4,
                   alpha=0.92, zorder=3)
    handles = [_swatch(TASK_COLORS[t], t) for t in TASK_COLORS]
    _legend_below(ax, handles, ncol=2)
    _add_diagonal_note(ax, points)
    _save_panel(fig, "panel_realworld")


def panel_realworld_avg(points) -> None:
    """One point per model, averaged across the 4 real-world tasks. Less
    crowded companion to `panel_realworld`. Model markers come from the
    shared `legend_models` strip; the single in-panel legend entry just
    names the suite that's being pooled."""
    pooled = pool_realworld_by_model(points)
    assert len(pooled) == 6, f"expected 6 model points, got {len(pooled)}"
    fig, ax = _new_panel_fig(title="Real world", show_ylabel=False)
    for p in pooled:
        ax.scatter(p.rate_max * 100, p.rate_one * 100,
                   s=MARKER_S,
                   marker=model_marker(p.model),
                   color="#4477AA",
                   edgecolor=EDGE, linewidth=0.4,
                   alpha=0.92, zorder=3)
    _legend_below(ax, [_swatch("#4477AA", "Real world suite")],
                  ncol=1)
    _add_diagonal_note(ax, pooled)
    _save_panel(fig, "panel_realworld_avg")


def panel_combined(libero_points, robotwin_points, calvin_points,
                   rw_points, mode: str, *,
                   include_model_legend: bool = False,
                   legends_inside: bool = False) -> None:
    """Single composite PDF: all four panels in one row, optionally followed
    by the model-legend strip.

    Drop-in replacement for the four \\includegraphics + legend subfigures in
    LaTeX. Axes are explicitly placed in inches with square dimensions (so
    `set_aspect('equal')` doesn't leave dead space left/right of the scatter
    inside each axes box, which is what made the previous version look like
    the panels were swimming in white space).
    """
    LEFT_PAD = 0.45      # LIBERO y-label + y-tick labels
    RIGHT_PAD = 0.06
    INTER_GAP = 0.12     # between adjacent axes (tick marks still draw)
    AX_SIDE = 2.00       # square axes -> bigger scatters than the old 2.28x1.54
    TOP_PAD = 0.18       # title clearance
    # When legends sit inside the axes, the band below only holds the xlabel.
    BELOW_AX_PAD = 0.13 if legends_inside else 0.75
    MODEL_LEGEND_H = 0.49 if include_model_legend else 0.0

    fig_W = LEFT_PAD + 4 * AX_SIDE + 3 * INTER_GAP + RIGHT_PAD
    fig_H = TOP_PAD + AX_SIDE + BELOW_AX_PAD + MODEL_LEGEND_H
    fig = plt.figure(figsize=(fig_W, fig_H))

    panel_specs = [
        ("LIBERO / LIBERO+", True),
        ("RoboTwin",         False),
        ("CALVIN ABC→D",     False),
        ("Real world",       False),
    ]
    axes = []
    centers_frac = []
    cum_x_in = LEFT_PAD
    b_in = MODEL_LEGEND_H + BELOW_AX_PAD
    for title, show_y in panel_specs:
        l_in = cum_x_in
        ax = fig.add_axes((l_in/fig_W, b_in/fig_H,
                           AX_SIDE/fig_W, AX_SIDE/fig_H))
        _setup_axes(ax, title=title, show_ylabel=show_y)
        axes.append(ax)
        centers_frac.append((l_in + AX_SIDE/2) / fig_W)
        cum_x_in += AX_SIDE + INTER_GAP
    libero_ax, robotwin_ax, calvin_ax, rw_ax = axes

    # LIBERO / LIBERO+ (same scatter logic as panel_libero).
    for p in libero_points:
        is_libero = p.benchmark == "LIBERO"
        libero_ax.scatter(p.rate_max * 100, p.rate_one * 100,
                          s=MARKER_S,
                          marker=model_marker(p.model),
                          color=SUITE_COLORS[p.sub],
                          edgecolor=EDGE if is_libero else "none",
                          linewidth=0.5 if is_libero else 0.0,
                          alpha=0.92, zorder=3)
    suite_labels = (["SGO avg", "LIBERO-10"] if mode == "sgo-avg"
                    else SUITE_ORDER)
    libero_handles = [_swatch(SUITE_COLORS[s], s) for s in suite_labels]
    libero_handles.append(_swatch("#9CA3AF", "LIBERO", with_edge=True))
    libero_handles.append(_swatch("#9CA3AF", "LIBERO+", with_edge=False))
    _add_diagonal_note(libero_ax, libero_points)

    # RoboTwin.
    _scatter_bench(robotwin_ax, robotwin_points)
    robotwin_handles = [
        _swatch(BENCH_COLORS["Clean"], "Clean"),
        _swatch(BENCH_COLORS["Randomized"], "Randomized"),
    ]
    _add_diagonal_note(robotwin_ax, robotwin_points)

    # CALVIN ABC->D.
    for p in calvin_points:
        calvin_ax.scatter(p.rate_max * 100, p.rate_one * 100,
                          s=MARKER_S,
                          marker=model_marker(p.model),
                          color=CALVIN_POS_COLORS[p.sub],
                          edgecolor=EDGE, linewidth=0.4,
                          alpha=0.92, zorder=3)
    calvin_handles = [_swatch(CALVIN_POS_COLORS[p], p)
                      for p in CALVIN_POS_ORDER]
    _add_diagonal_note(calvin_ax, calvin_points)

    # Real-world averaged across the 4 tasks.
    rw_pooled = pool_realworld_by_model(rw_points)
    for p in rw_pooled:
        rw_ax.scatter(p.rate_max * 100, p.rate_one * 100,
                      s=MARKER_S,
                      marker=model_marker(p.model),
                      color="#4477AA",
                      edgecolor=EDGE, linewidth=0.4,
                      alpha=0.92, zorder=3)
    rw_handles = [_swatch("#4477AA", "Real world suite")]
    _add_diagonal_note(rw_ax, rw_pooled)

    # Per-panel color legends pinned at a common baseline in figure coords so
    # variable row counts don't push the model legend around.
    # Baseline = top of the per-panel legend band (just below the xlabel).
    legend_top_y_frac = (MODEL_LEGEND_H + BELOW_AX_PAD - 0.37) / fig_H
    # (ax, x_frac, handles, ncol_below, ncol_inside)
    # Inside-mode ncols are tuned to keep each legend box within its axes
    # width: LIBERO collapses to one tall column, CALVIN goes 2x(2+3 entries).
    legend_specs = [
        (libero_ax,   centers_frac[0], libero_handles,   2, 1),
        (robotwin_ax, centers_frac[1], robotwin_handles, 1, 1),
        (calvin_ax,   centers_frac[2], calvin_handles,   3, 1),
        (rw_ax,       centers_frac[3], rw_handles,       1, 1),
    ]
    for ax, x_frac, handles, ncol_below, ncol_inside in legend_specs:
        if legends_inside:
            ax.legend(handles=handles, loc="upper left",
                      ncol=ncol_inside, frameon=True, framealpha=0.85,
                      edgecolor="#9CA3AF", fancybox=False,
                      borderpad=0.3, labelspacing=0.25,
                      handletextpad=0.35, columnspacing=0.9, fontsize=7.5)
        else:
            ax.legend(handles=handles, loc="upper center",
                      bbox_to_anchor=(x_frac, legend_top_y_frac),
                      bbox_transform=fig.transFigure,
                      ncol=ncol_below, frameon=False,
                      handletextpad=0.35, columnspacing=0.9, fontsize=7.5)

    # Single model-legend strip at the bottom (full width, one row). Opt-in.
    if include_model_legend:
        all_points = (list(libero_points) + list(robotwin_points)
                      + list(calvin_points) + list(rw_pooled))
        models = sorted({p.model for p in all_points}, key=model_key)
        model_handles = [_model_handle(m) for m in models]
        fig.legend(handles=model_handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.005),
                   bbox_transform=fig.transFigure,
                   ncol=len(model_handles), frameon=False,
                   handletextpad=0.35, columnspacing=0.9, fontsize=8)

    fig.savefig(OUT / "panel_combined.png")
    fig.savefig(OUT / "panel_combined.pdf")
    plt.close(fig)


def _swatch(color: str, label: str, with_edge: bool = True):
    return plt.Line2D([], [], color=color, marker="o", linestyle="None",
                      markeredgecolor=EDGE if with_edge else "none",
                      markeredgewidth=0.5 if with_edge else 0.0,
                      markersize=7, label=label)


def _model_handle(model: str):
    return plt.Line2D([], [], color="#D1D5DB", marker=model_marker(model),
                      linestyle="None", markeredgecolor=EDGE,
                      markeredgewidth=0.6, markersize=7,
                      label=f"{model_legend(model)} ({model_size_text(model)} B)")


def _legend_below(ax, handles, ncol: int | None = None,
                  fig_y: float = 0.03) -> None:
    """Place a legend with its bottom edge at fig y=`fig_y` (figure-fraction
    coordinates). Using figure coords keeps the legend at the same vertical
    spot across panels even when their axes positions differ."""
    if ncol is None:
        ncol = -(len(handles) // -2)  # ceildiv for 2 rows
    fig = ax.figure
    ax.legend(
        handles=handles, loc="lower center",
        bbox_to_anchor=(0.5, fig_y),
        bbox_transform=fig.transFigure,
        ncol=ncol, frameon=False,
        handletextpad=0.35, columnspacing=0.9,
        fontsize=7.5,
    )


def _save_single_row_legend(handles, name: str,
                            width_hint: float = 20.0) -> None:
    """Render `handles` as a single horizontal row; `bbox_inches='tight'`
    crops the canvas down to the legend strip on save."""
    fig = plt.figure(figsize=(width_hint, 0.5))
    if name == "legend_models":
        # For the model legend, we want a single row with no wrapping, so we
        # set ncol to the number of handles. For the other legends, we allow
        # wrapping into two rows if needed, so we set ncol to 2.
        fig.legend(handles=handles, loc="center", ncol=len(handles),
                frameon=False, handletextpad=0.35, columnspacing=0.9,
                fontsize=8)
    else:
        def ceildiv(a, b):
            return -(a // -b)
        ncol=ceildiv(len(handles), 2)
        fig.legend(handles=handles, loc="center", ncol=ncol,
                frameon=False, handletextpad=0.35, columnspacing=0.9,
                fontsize=8)
    _save_png_pdf(fig, name)
    plt.close(fig)


def render_legend(libero_points, robotwin_points, rw_points,
                  calvin_points, mode: str) -> None:
    """Only the model legend is rendered as a standalone file now; per-
    benchmark color legends live inside each panel PDF (below the xlabel)."""
    del mode  # unused — kept for API symmetry with the panels.
    all_points = (list(libero_points) + list(robotwin_points)
                  + list(rw_points) + list(calvin_points))
    models = sorted({p.model for p in all_points}, key=model_key)
    model_handles = [_model_handle(m) for m in models]
    _save_single_row_legend(model_handles, "legend_models")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["all", "per-suite", "sgo-avg"],
                        default="all",
                        help="LIBERO panel: per-suite (56 points) or sgo-avg "
                             "(pool Spatial+Object+Goal -> 28 points).")
    parser.add_argument("--note", choices=["no-worse", "symmetric", "tost"],
                        default="symmetric",
                        help="Diagonal note style: 'no-worse' "
                             "reports the fraction of points where 1-step is "
                             "worse than ref-step by at most --note-pp pp "
                             "(ties and 1-step wins included); 'symmetric' "
                             "uses the point estimate |Δ| ≤ --note-pp pp; "
                             "'tost' is the CI-based equivalence version of "
                             "'symmetric' -- a point counts only if its whole "
                             "95% CI on Δ lies within ±--note-pp pp, so "
                             "small-n points are not certified.")
    parser.add_argument("--note-pp", type=float, default=3.0,
                        help="Threshold in percentage points used by the "
                             "diagonal note. Default: 3.")
    parser.add_argument("--note-band", action="store_true",
                        help="When --note symmetric, shade a ±note-pp band "
                             "around the y = x diagonal. Ignored for other "
                             "--note styles.")
    parser.add_argument("--combined-model-legend", action=argparse.BooleanOptionalAction, default=True,
                        help="Include the model legend strip at the bottom "
                             "of panel_combined.{png,pdf}. On by default; "
                             "the standalone legend_models.{png,pdf} is "
                             "always written and can be included separately "
                             "in LaTeX.")
    parser.add_argument("--combined-legends-inside", action=argparse.BooleanOptionalAction, default=True,
                        help="In panel_combined, draw per-panel color "
                             "legends inside each axes (upper-left corner) "
                             "instead of below the xlabel. Shrinks figure "
                             "height. On by default.")
    args = parser.parse_args()
    modes = ("per-suite", "sgo-avg") if args.mode == "all" else (args.mode,)
    for mode in modes:
        args.mode = mode
        render(args)


def render(args) -> None:
    global OUT, NOTE_STYLE, NOTE_PP, NOTE_BAND
    NOTE_STYLE = args.note
    NOTE_PP = args.note_pp
    NOTE_BAND = args.note_band

    OUT = ROOT / "flow_ablation_crossbench" / (
        "per_family_SGOavg" if args.mode == "sgo-avg" else "perfamily")
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading CALVIN ...")
    calvin_points = load_calvin_positions()
    print("Loading LIBERO / LIBERO+ ...")
    _, libero_sub = load_libero_family()
    print("Loading RoboTwin ...")
    _, robotwin_sub = load_robotwin()
    print("Loading real-world ...")
    _, rw_sub = load_real_world()

    # RoboTwin loader emits long benchmark labels; the panel title already
    # carries "RoboTwin", so strip the prefix to keep the in-panel legend
    # readable and match BENCH_COLORS keys ("Clean" / "Randomized").
    for p in robotwin_sub:
        if p.benchmark == "RoboTwin Clean":
            p.benchmark = "Clean"
        elif p.benchmark == "RoboTwin Randomized":
            p.benchmark = "Randomized"

    libero_points = [p for p in libero_sub
                     if p.benchmark in {"LIBERO", "LIBERO+"}]
    if args.mode == "sgo-avg":
        libero_points = pool_sgo(libero_points)
    robotwin_points = [p for p in robotwin_sub
                       if p.benchmark in {"Clean",
                                          "Randomized"}]
    rw_points = [p for p in rw_sub if p.benchmark == "Real-world"]

    panel_calvin(calvin_points)
    panel_libero(libero_points, args.mode)
    panel_robotwin(robotwin_points)
    panel_realworld(rw_points)
    panel_realworld_avg(rw_points)
    panel_combined(libero_points, robotwin_points, calvin_points,
                   rw_points, args.mode,
                   include_model_legend=args.combined_model_legend,
                   legends_inside=args.combined_legends_inside)
    render_legend(libero_points, robotwin_points, rw_points,
                  calvin_points, args.mode)

    print(f"Done. Outputs in {OUT}")


if __name__ == "__main__":
    main()
