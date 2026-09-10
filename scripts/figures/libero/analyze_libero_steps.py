"""Shared analyze_libero_steps data loaders and statistical primitives; no table output."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any


import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

MODEL_ORDER = [ "dit", "smolvla", "xvla", "flower", "pi05", "lap", "mibot"]
DISPLAY_NAMES = {
    "flower": "Flower",
    "lap": "LAP",
    "mibot": "XR-0",
    "smolvla": "SmolVLA",
    "xvla": "X-VLA",
    "pi05": "Pi0.5",
    "dit": "DiT",
}

PLOT_METRICS = [
    ("libero_spatial", "Spatial"),
    ("libero_object", "Object"),
    ("libero_goal", "Goal"),
    ("libero_10", "LIBERO-10"),
    ("overall", "Overall"),
]


@dataclass(frozen=True)
class Counts:
    successes: int
    trials: int

    @property
    def rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0


@dataclass(frozen=True)
class RunSpec:
    model_key: str
    steps: int
    dataset: str
    relative_path: str
    loader: str


@dataclass(frozen=True)
class RunResult:
    spec: RunSpec
    suite_counts: dict[str, Counts]
    overall: Counts


RUN_SPECS = [
    # LIBERO.
    RunSpec("flower", 1, "libero", "flower/libero_logs", "flower_summary"),
    RunSpec("flower", 4, "libero", "flower/libero_logs", "flower_summary"),
    RunSpec("lap", 1, "libero", "lap/lap_libero/results", "lap_results"),
    RunSpec("lap", 10, "libero", "lap/lap_libero/results", "lap_results"),
    RunSpec("mibot", 1, "libero", "mibot/logs_1step", "mibot_libero"),
    RunSpec("mibot", 5, "libero", "mibot/logs_5step", "mibot_libero"),
    RunSpec("pi05", 1, "libero", "pi05/libero/numstep1", "pi05_libero"),
    RunSpec("pi05", 10, "libero", "pi05/libero/numstep10", "pi05_libero"),
    RunSpec("smolvla", 1, "libero", "smolvla/libero_logs/paired/videos_smolvla_flowstep1_actionstep10_episodes.json", "smolvla_paired"),
    RunSpec("smolvla", 10, "libero", "smolvla/libero_logs/paired/videos_smolvla_flowstep10_actionstep10_episodes.json", "smolvla_paired"),
    RunSpec("xvla", 1, "libero", "X-VLA/libero_1steps/results.json", "xvla_concat"),
    RunSpec("xvla", 10, "libero", "X-VLA/libero_10steps/results.json", "xvla_concat"),
    # LIBERO+.
    RunSpec("flower", 1, "liberoplus", "flower/liberoplus_logs", "flower_summary"),
    RunSpec("flower", 4, "liberoplus", "flower/liberoplus_logs", "flower_summary"),
    RunSpec("lap", 1, "liberoplus", "lap/libero_to_liberoplus", "lap_results"),
    RunSpec("lap", 10, "liberoplus", "lap/libero_to_liberoplus", "lap_results"),
    RunSpec("mibot", 1, "liberoplus", "mibot/logs_liberoplus_1step/aggregate_results.json", "mibot_liberoplus"),
    RunSpec("mibot", 5, "liberoplus", "mibot/logs_liberoplus_5steps", "mibot_liberoplus"),
    RunSpec("pi05", 1, "liberoplus", "pi05/libero_to_liberoplus/videos_pi05_flowstep1/summary_by_suite.json", "pi05_liberoplus"),
    RunSpec("pi05", 10, "liberoplus", "pi05/libero_to_liberoplus/videos_pi05_flowstep10/summary_by_suite.json", "pi05_liberoplus"),
    RunSpec("xvla", 1, "liberoplus", "X-VLA/liberoplus_1steps/results.json", "xvla_concat"),
    RunSpec("xvla", 10, "liberoplus", "X-VLA/liberoplus_10steps/results.json", "xvla_concat"),
    RunSpec("smolvla", 1, "liberoplus", "smolvla/liberoplus_logs/paired/videos_smolvla_flowstep1_actionstep10_episodes.json", "smolvla_paired"),
    RunSpec("smolvla", 10, "liberoplus", "smolvla/liberoplus_logs/paired/videos_smolvla_flowstep10_actionstep10_episodes.json", "smolvla_paired"),
    # DiT.
    RunSpec("dit", 1, "libero", "dit/pretrained/libero_eval/final_model/sharded_n1/aggregated_results.json", "dit_aggregated"),
    RunSpec("dit", 10, "libero", "dit/pretrained/libero_eval/final_model/sharded_n10/aggregated_results.json", "dit_aggregated"),
    RunSpec("dit", 1, "liberoplus", "dit/pretrained/liberoplus_eval/final_model/sharded_n1/aggregated_results.json", "dit_aggregated"),
    RunSpec("dit", 10, "liberoplus", "dit/pretrained/liberoplus_eval/final_model/sharded_n10/aggregated_results.json", "dit_aggregated"),
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_rollout_records(path: Path) -> list[Any]:
    """Read an X-VLA results file as a list of records, in order.

    These files are plain JSON arrays. The fallback reads the legacy layout, a
    bare concatenation of JSON objects, so the loader also works against
    untidied copies of the logs.
    """
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except ValueError:
        decoder = json.JSONDecoder()
        objects: list[Any] = []
        index = 0
        while index < len(text):
            while index < len(text) and text[index].isspace():
                index += 1
            if index >= len(text):
                break
            record, index = decoder.raw_decode(text, index)
            objects.append(record)
        return objects
    if not isinstance(obj, list):
        raise ValueError(f"{path}: expected a JSON array of rollout records")
    return obj


def make_overall(suite_counts: dict[str, Counts]) -> Counts:
    return Counts(
        successes=sum(counts.successes for counts in suite_counts.values()),
        trials=sum(counts.trials for counts in suite_counts.values()),
    )


def load_xvla_concat(path: Path) -> tuple[dict[str, Counts], Counts]:
    suite_counts: dict[str, Counts] = {}
    for obj in load_rollout_records(path):
        for suite, payload in obj.items():
            suite_counts[suite] = Counts(
                successes=int(payload["num_successes"]),
                trials=int(payload["num_rollouts"]),
            )
    return suite_counts, make_overall(suite_counts)


def load_summary_json(path: Path) -> tuple[dict[str, Counts], Counts]:
    raw = read_json(path)
    suite_counts = {
        suite: Counts(successes=int(payload["success"]), trials=int(payload["total"]))
        for suite, payload in raw["suites"].items()
    }
    overall = Counts(
        successes=int(raw["overall"]["success"]),
        trials=int(raw["overall"]["total"]),
    )
    return suite_counts, overall


def load_smolvla_paired(path: Path) -> tuple[dict[str, Counts], Counts]:
    raw = read_json(path)
    suite_counts: dict[str, Counts] = {suite: Counts(successes=0, trials=0) for suite in SUITES}
    for ep in raw:
        suite = ep.get("task_suite_name")
        if suite not in suite_counts:
            continue
        prev = suite_counts[suite]
        suite_counts[suite] = Counts(
            successes=prev.successes + (1 if ep["success"] else 0),
            trials=prev.trials + 1,
        )
    return suite_counts, make_overall(suite_counts)


def load_pi05_libero(path: Path) -> tuple[dict[str, Counts], Counts]:
    suite_counts: dict[str, Counts] = {}
    for suite in SUITES:
        suite_path = path / f"libero_all_LIBEROBenchmark_{suite}.json"
        raw = read_json(suite_path)
        successes = 0
        trials = 0
        for task in raw["tasks"]:
            episodes = task.get("episodes", [])
            successes += sum(1 for episode in episodes if episode["metrics"]["success"])
            trials += len(episodes)
        if not trials:
            trials = int(raw["merge_info"]["total_episodes"])
            successes = round(float(raw["mean_success"]) * trials)
        suite_counts[suite] = Counts(successes=successes, trials=trials)
    return suite_counts, make_overall(suite_counts)


def load_mibot_libero(path: Path) -> tuple[dict[str, Counts], Counts]:
    suite_counts: dict[str, Counts] = {}
    for suite in SUITES:
        raw = read_json(path / suite / "merged_results.json")
        task_rates = [float(row["success_rate"]) for row in raw["individual_results"]]
        if len(task_rates) != 10:
            raise ValueError(f"Expected 10 task rates in {path / suite / 'merged_results.json'}")
        suite_counts[suite] = Counts(
            successes=sum(round(rate * 50) for rate in task_rates),
            trials=50 * len(task_rates),
        )
    return suite_counts, make_overall(suite_counts)


def load_flower_summary(path: Path, steps: int) -> tuple[dict[str, Counts], Counts]:
    suite_counts: dict[str, Counts] = {}
    for suite in SUITES:
        raw = read_json(path / suite / f"nsteps_{steps}" / "summary.json")
        successes = sum(int(task["num_success"]) for task in raw["per_task"].values())
        trials = sum(int(task["num_rollouts"]) for task in raw["per_task"].values())
        suite_counts[suite] = Counts(successes=successes, trials=trials)
    return suite_counts, make_overall(suite_counts)


def load_lap_results(path: Path, steps: int) -> tuple[dict[str, Counts], Counts]:
    suite_counts: dict[str, Counts] = {}
    for suite in SUITES:
        raw = read_json(path / f"{suite}_ns{steps}" / "results.json")
        suite_counts[suite] = Counts(
            successes=int(raw["summary"]["total_successes"]),
            trials=int(raw["summary"]["total_episodes"]),
        )
    return suite_counts, make_overall(suite_counts)


def load_mibot_liberoplus(path: Path, steps: int) -> tuple[dict[str, Counts], Counts]:
    if steps == 1:
        raw = read_json(path)
        suite_counts = {
            suite: Counts(successes=int(payload["successes"]), trials=int(payload["rollouts"]))
            for suite, payload in raw["suites"].items()
        }
        overall = Counts(successes=int(raw["total"]["successes"]), trials=int(raw["total"]["rollouts"]))
        return suite_counts, overall

    suite_counts: dict[str, Counts] = {}
    for suite in SUITES:
        raw = read_json(path / suite / "summary.json")
        suite_counts[suite] = Counts(successes=int(raw["successes"]), trials=int(raw["rollouts"]))
    return suite_counts, make_overall(suite_counts)


def load_dit_aggregated(path: Path) -> tuple[dict[str, Counts], Counts]:
    raw = read_json(path)
    suite_counts: dict[str, Counts] = {}
    for suite in SUITES:
        payload = raw[suite]
        suite_counts[suite] = Counts(
            successes=int(payload["total_successes"]),
            trials=int(payload["total_episodes"]),
        )
    agg = raw.get("_aggregate")
    if agg is not None:
        overall = Counts(successes=int(agg["total_successes"]), trials=int(agg["total_episodes"]))
    else:
        overall = make_overall(suite_counts)
    return suite_counts, overall


def load_pi05_liberoplus(path: Path) -> tuple[dict[str, Counts], Counts]:
    raw = read_json(path)
    suite_counts = {
        suite: Counts(successes=int(payload["success"]), trials=int(payload["total_episodes"]))
        for suite, payload in raw["suites"].items()
    }
    overall = Counts(
        successes=int(raw["overall"]["success"]),
        trials=int(raw["overall"]["total_episodes"]),
    )
    return suite_counts, overall


def load_run(libero_dir: Path, spec: RunSpec) -> RunResult:
    path = libero_dir / spec.relative_path
    if not path.exists():
        raise FileNotFoundError(path)

    if spec.loader == "xvla_concat":
        suite_counts, overall = load_xvla_concat(path)
    elif spec.loader == "summary_json":
        suite_counts, overall = load_summary_json(path)
    elif spec.loader == "smolvla_paired":
        suite_counts, overall = load_smolvla_paired(path)
    elif spec.loader == "pi05_libero":
        suite_counts, overall = load_pi05_libero(path)
    elif spec.loader == "mibot_libero":
        suite_counts, overall = load_mibot_libero(path)
    elif spec.loader == "flower_summary":
        suite_counts, overall = load_flower_summary(path, spec.steps)
    elif spec.loader == "lap_results":
        suite_counts, overall = load_lap_results(path, spec.steps)
    elif spec.loader == "mibot_liberoplus":
        suite_counts, overall = load_mibot_liberoplus(path, spec.steps)
    elif spec.loader == "pi05_liberoplus":
        suite_counts, overall = load_pi05_liberoplus(path)
    elif spec.loader == "dit_aggregated":
        suite_counts, overall = load_dit_aggregated(path)
    else:
        raise ValueError(f"Unknown loader: {spec.loader}")

    missing_suites = sorted(set(SUITES) - set(suite_counts))
    if missing_suites:
        raise ValueError(f"Missing suites for {spec}: {missing_suites}")
    return RunResult(spec=spec, suite_counts=suite_counts, overall=overall)


def two_proportion_z_pvalue(baseline: Counts, reference: Counts) -> float:
    if baseline.trials <= 0 or reference.trials <= 0:
        return 1.0

    pooled_rate = (baseline.successes + reference.successes) / (
        baseline.trials + reference.trials
    )
    standard_error = math.sqrt(
        pooled_rate
        * (1.0 - pooled_rate)
        * ((1.0 / baseline.trials) + (1.0 / reference.trials))
    )
    if standard_error <= 0.0:
        return 1.0

    z_score = (reference.rate - baseline.rate) / standard_error
    return 2.0 * (1.0 - NormalDist().cdf(abs(z_score)))


