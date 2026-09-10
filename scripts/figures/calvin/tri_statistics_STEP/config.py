"""Configuration for the CALVIN ABC-D 1-step vs max-step STEP pipeline.

Mirrors `corl2026_logs/libero/tri_statistics_STEP/config.py` but for CALVIN.
CALVIN has a single dataset (abc_d) and a single binary cell (SR@5 — full
sequence completion), so the family size for STEP α-splitting is 1.
"""

from __future__ import annotations

from dataclasses import dataclass


# Within each model: variant_low (1) vs variant_high (max).
MODEL_HIGH_STEPS: dict[str, int] = {
    "xvla": 10,
    "flower": 4,
    "mibot": 5,
}

# Models that expose per-rollout outcomes we can pair across variants:
#   - mibot:  rank_*_results.pkl with full per-sequence metadata.
#   - xvla:   log.txt with running averages from which we reconstruct exact
#             per-rollout 0-5 completion scores by summing the chain-level
#             cumulative counts.
#   - flower: per-episode-logs/abc_d_{N}step/logs/*/per_episode_results.json
#             with explicit `result` (0-5) and sr1..sr5 fields per episode.
MODELS_WITH_PAIRED_DATA: set[str] = {"flower", "mibot", "xvla"}


DATASETS = ("abc_d",)


@dataclass(frozen=True)
class PairSpec:
    model: str
    dataset: str
    variant_low: int
    variant_high: int
    has_paired_data: bool


def build_pair_specs() -> list[PairSpec]:
    specs: list[PairSpec] = []
    for dataset in DATASETS:
        for model, high in MODEL_HIGH_STEPS.items():
            specs.append(
                PairSpec(
                    model=model,
                    dataset=dataset,
                    variant_low=1,
                    variant_high=high,
                    has_paired_data=model in MODELS_WITH_PAIRED_DATA,
                )
            )
    return specs


PAIR_SPECS: list[PairSpec] = build_pair_specs()


# Cell layout: one binary cell per (model, dataset), keyed "sr5".
CELL_KEYS = ("sr5",)
CELL_DISPLAY = {"sr5": "SR@5"}


MODEL_DISPLAY = {
    "flower": "Flower",
    "mibot": "XR-0",
    "xvla": "X-VLA",
}

DATASET_DISPLAY = {
    "abc_d": "CALVIN ABC-D",
}


# `analyze_calvin_steps.DEFAULT_RUNS` keys use "x-vla" (hyphen); our internal
# model keys use "xvla" (matching the LIBERO pipeline). This map bridges them.
ANALYZE_KEY: dict[str, str] = {
    "flower": "flower",
    "mibot": "mibot",
    "xvla": "x-vla",
}
