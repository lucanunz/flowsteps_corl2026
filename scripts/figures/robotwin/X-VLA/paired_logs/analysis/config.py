"""Paths and constants for the RoboTwin X-VLA paired step analysis.

The paired runs live under `paired_logs/runs/` and compare 1-flow-step vs
10-flow-step inference on two RoboTwin settings (Clean and Randomized),
with seeds matched 1:1 across the two runs of each setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PKG_DIR = Path(__file__).resolve().parent              # .../paired_logs/analysis/
PAIRED_LOGS_DIR = PKG_DIR.parent                       # .../paired_logs/
RUNS_DIR = PAIRED_LOGS_DIR / "runs"

CLEAN_LOW_RUN = "20260518_112400_robotwin_1flow_100ep_demo_clean_seed0"
CLEAN_HIGH_RUN = "20260514_115552_robotwin_10flow_100ep_demo_clean_seed0"

# Randomized: primary May-14/May-18 runs cover 44 of the 50 tasks; supplements
# (May-20 for 1-flow, May-21 for 10-flow) cover the remaining 6 hardest
# tasks. Loader merges per-task across the dirs.
RAND_LOW_RUN = "20260518_103303_robotwin_1flow_100ep_demo_randomized_seed0"
RAND_LOW_SUPP = "20260520_125548_robotwin_1flow_100ep_demo_randomized_seed0"
RAND_HIGH_RUN = "20260514_122856_robotwin_10flow_100ep_demo_randomized_seed0"
RAND_HIGH_SUPP = "20260521_111337_robotwin_10flow_100ep_demo_randomized_seed0"

LOW_LABEL = "1 flow step"
HIGH_LABEL = "10 flow steps"

SETTING_DISPLAY: dict[str, str] = {
    "clean": "RoboTwin Clean",
    "rand": "RoboTwin Randomized",
}
MODEL_DISPLAY = "X-VLA"

DEFAULT_OUTPUT_DIR = PKG_DIR / "outputs"


@dataclass(frozen=True)
class RunSpec:
    label: str
    paths: tuple[Path, ...]   # primary + any supplement dirs


def default_setting_specs() -> dict[str, tuple[RunSpec, RunSpec]]:
    """Return {setting: (low_spec, high_spec)} for every paired setting."""
    return {
        "clean": (
            RunSpec(label=LOW_LABEL, paths=(RUNS_DIR / CLEAN_LOW_RUN,)),
            RunSpec(label=HIGH_LABEL, paths=(RUNS_DIR / CLEAN_HIGH_RUN,)),
        ),
        "rand": (
            RunSpec(label=LOW_LABEL, paths=(
                RUNS_DIR / RAND_LOW_RUN,
                RUNS_DIR / RAND_LOW_SUPP,
            )),
            RunSpec(label=HIGH_LABEL, paths=(
                RUNS_DIR / RAND_HIGH_RUN,
                RUNS_DIR / RAND_HIGH_SUPP,
            )),
        ),
    }
