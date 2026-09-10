"""Configuration for the real-world 1-step vs 10-step STEP pipeline.

Mirrors `corl2026_logs/calvin/tri_statistics_STEP/config.py` but for the
manual real-world rollouts. Each (model, task) is a paired binary cell
(manual_success), and the Bonferroni correction family is the 4 tasks
within each model.
"""

from __future__ import annotations

from dataclasses import dataclass


MODELS: tuple[str, ...] = ("lap", "openpi", "xvla", "dit", "flower", "mibot")

TASKS: tuple[str, ...] = (
    "spoon_drawer",
    "open_drawer",
    "open_oven",
    "close_oven",
)

# Per-model task list. All models include every task by default; override
# here if a model's data is missing or partial enough to warrant exclusion.
MODEL_TASKS: dict[str, tuple[str, ...]] = {
    "lap": TASKS,
    "openpi": TASKS,
    "xvla": TASKS,
    "dit": TASKS,
    "flower": TASKS,
    "mibot": TASKS,
}

VARIANT_LOW: int = 1
# Per-model "high" variant — flow models with fewer denoising steps reach
# their reference performance at different step counts.
MODEL_HIGH_STEPS: dict[str, int] = {
    "lap":    10,
    "openpi": 10,
    "xvla":   10,
    "dit":    10,
    "flower":  5,
    "mibot":   5,
}


MODEL_DISPLAY: dict[str, str] = {
    "lap": "LAP",
    "openpi": r"$\pi_{0.5}$",
    "xvla": "X-VLA",
    "dit": "DiT",
    "flower": "FLOWER",
    "mibot": "XR-0",
}

TASK_DISPLAY: dict[str, str] = {
    "spoon_drawer": "Spoon → Drawer",
    "open_drawer": "Open Drawer",
    "open_oven": "Open Oven",
    "close_oven": "Close Oven",
}


@dataclass(frozen=True)
class PairSpec:
    model: str
    task: str
    variant_low: int
    variant_high: int


def build_pair_specs() -> list[PairSpec]:
    specs: list[PairSpec] = []
    for model in MODELS:
        for task in MODEL_TASKS.get(model, TASKS):
            specs.append(
                PairSpec(
                    model=model,
                    task=task,
                    variant_low=VARIANT_LOW,
                    variant_high=MODEL_HIGH_STEPS[model],
                )
            )
    return specs


PAIR_SPECS: list[PairSpec] = build_pair_specs()
