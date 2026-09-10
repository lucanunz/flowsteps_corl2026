"""Configuration for the within-model 1-step vs max-step pairs.

For each model and dataset (libero / liberoplus) we look up the matching
RunSpec entries from analyze_libero_steps.py and pair them.
"""

from __future__ import annotations

from dataclasses import dataclass

# Within each model: variant_low (1) vs variant_high (max).
MODEL_HIGH_STEPS: dict[str, int] = {
    "flower": 4,
    "lap": 10,
    "mibot": 5,
    "pi05": 10,
    "smolvla": 10,
    "xvla": 10,
    "dit": 10,
}

# Models evaluated on each dataset. Every model now has both a LIBERO-base and
# a LIBERO+ pair.
DATASET_MODELS: dict[str, tuple[str, ...]] = {
    "libero": tuple(MODEL_HIGH_STEPS),
    "liberoplus": tuple(MODEL_HIGH_STEPS),
}

# Models that have per-episode data we can pair (LIBERO base).
MODELS_WITH_EPISODES_LIBERO: set[str] = {"flower", "lap", "mibot", "pi05", "smolvla", "xvla", "dit"}
# LIBERO+ paired data: every model exposes per-rollout outcomes either as
# episodes[*] (LAP), as a concatenated stream (X-VLA), as a flat list of
# `{task_suite_name, task_id, episode_idx, success}` records (SmolVLA), or as
# per-task {success, failure} counts with globally unique task keys
# (Flower/XR-0/Pi0.5). For the
# count-based models we pair only the single-rollout tasks; multi-rollout tasks
# are skipped from the paired test and recorded as a diagnostic.
MODELS_WITH_EPISODES_LIBEROPLUS: set[str] = {"flower", "lap", "mibot", "pi05", "smolvla", "xvla", "dit"}


DATASETS = ("libero", "liberoplus")


@dataclass(frozen=True)
class PairSpec:
    model: str
    dataset: str
    variant_low: int
    variant_high: int
    has_episode_data: bool


def build_pair_specs() -> list[PairSpec]:
    specs: list[PairSpec] = []
    for dataset in DATASETS:
        for model in DATASET_MODELS[dataset]:
            high = MODEL_HIGH_STEPS[model]
            if dataset == "libero":
                has_eps = model in MODELS_WITH_EPISODES_LIBERO
            else:
                has_eps = model in MODELS_WITH_EPISODES_LIBEROPLUS
            specs.append(PairSpec(
                model=model,
                dataset=dataset,
                variant_low=1,
                variant_high=high,
                has_episode_data=has_eps,
            ))
    return specs


PAIR_SPECS: list[PairSpec] = build_pair_specs()


# Suite ordering for tables and figures (always show OVERALL last).
SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
OVERALL_KEY = "overall"
ALL_CELLS = SUITE_ORDER + [OVERALL_KEY]


SUITE_DISPLAY = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "LIBERO-10",
    "overall": "Overall",
}


MODEL_DISPLAY = {
    "flower": "Flower",
    "lap": "LAP",
    "mibot": "XR-0",
    "pi05": "Pi0.5",
    "smolvla": "SmolVLA",
    "xvla": "X-VLA",
    "dit": "DiT",
}
