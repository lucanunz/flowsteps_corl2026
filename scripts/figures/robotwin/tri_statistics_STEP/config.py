"""Paths, run specs, and display names for the RoboTwin STEP analysis.

Mirrors `libero/tri_statistics_STEP/config.py` in spirit: declares the
(model, setting, variant_low, variant_high) pairs we know how to load, plus
display strings used in reports and figures.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


PKG_DIR = Path(__file__).resolve().parent                   # robotwin/tri_statistics_STEP/
ROBOTWIN_DIR = PKG_DIR.parent                               # robotwin/
REPO_ROOT = ROBOTWIN_DIR.parent                             # corl2026_logs/
LIBERO_DIR = REPO_ROOT / "libero"
LIBERO_STEP_DIR = LIBERO_DIR / "tri_statistics_STEP"

# Make libero's primitives importable. We do this once at config import time
# so every downstream module in this package can `from .stats import …`
# without each one re-doing sys.path surgery.
for _p in (REPO_ROOT, LIBERO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# Where the paired RoboTwin runs live.
PAIRED_RUNS_DIR = ROBOTWIN_DIR / "X-VLA" / "paired_logs" / "runs"

# X-VLA seed-paired runs for both RoboTwin settings.
CLEAN_LOW_RUN = "20260518_112400_robotwin_1flow_100ep_demo_clean_seed0"
CLEAN_HIGH_RUN = "20260514_115552_robotwin_10flow_100ep_demo_clean_seed0"

# Randomized: 6 of the 50 task dirs (the hardest ones) are absent from the
# May-14 (10-flow) / May-18 (1-flow) primary runs. Two supplements fill them:
#   - May-20 (1-flow) covers exactly those 6 tasks.
#   - May-21 (10-flow) covers exactly the same 6 tasks at 10-flow.
# So per side we declare a tuple `(primary, supplement)`; the loader walks
# every dir and merges results per task. No task may appear in more than one
# dir on the same side.
RAND_LOW_RUN = "20260518_103303_robotwin_1flow_100ep_demo_randomized_seed0"
RAND_LOW_SUPP = "20260520_125548_robotwin_1flow_100ep_demo_randomized_seed0"
RAND_HIGH_RUN = "20260514_122856_robotwin_10flow_100ep_demo_randomized_seed0"
RAND_HIGH_SUPP = "20260521_111337_robotwin_10flow_100ep_demo_randomized_seed0"


@dataclass(frozen=True)
class PairSpec:
    model: str                     # e.g. "xvla"
    setting: str                   # e.g. "clean"
    variant_low: int               # nominal # of flow steps for the "low" run
    variant_high: int
    low_run_dirs: tuple[Path, ...] # primary + any supplements
    high_run_dirs: tuple[Path, ...]

    @property
    def dataset(self) -> str:
        """Alias used by report templates copied from the libero package."""
        return self.setting


PAIR_SPECS: tuple[PairSpec, ...] = (
    PairSpec(
        model="xvla",
        setting="clean",
        variant_low=1,
        variant_high=10,
        low_run_dirs=(PAIRED_RUNS_DIR / CLEAN_LOW_RUN,),
        high_run_dirs=(PAIRED_RUNS_DIR / CLEAN_HIGH_RUN,),
    ),
    PairSpec(
        model="xvla",
        setting="rand",
        variant_low=1,
        variant_high=10,
        low_run_dirs=(
            PAIRED_RUNS_DIR / RAND_LOW_RUN,
            PAIRED_RUNS_DIR / RAND_LOW_SUPP,
        ),
        high_run_dirs=(
            PAIRED_RUNS_DIR / RAND_HIGH_RUN,
            PAIRED_RUNS_DIR / RAND_HIGH_SUPP,
        ),
    ),
)


MODEL_DISPLAY: dict[str, str] = {
    "xvla": "X-VLA",
}
SETTING_DISPLAY: dict[str, str] = {
    "clean": "RoboTwin Clean",
    "rand": "RoboTwin Randomized",
}
DATASET_DISPLAY = SETTING_DISPLAY  # alias for libero-style report code

# In the per-pair report there is exactly one cell per pair (the Overall
# aggregate). We name it consistently so report/plot code can address it.
OVERALL_KEY = "Overall"
ALL_CELLS: tuple[str, ...] = (OVERALL_KEY,)

DEFAULT_OUTPUT_DIR = PKG_DIR / "outputs"


def select_specs(model: str | None = None, setting: str | None = None) -> list[PairSpec]:
    out = list(PAIR_SPECS)
    if model is not None and model != "all":
        out = [s for s in out if s.model == model]
    if setting is not None and setting != "all":
        out = [s for s in out if s.setting == setting]
    return out
