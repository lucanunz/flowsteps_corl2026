"""Configuration for the STEP-based 1-step vs max-step pipeline.

Pure re-export of `tri_statistics.config` — same models, datasets, suites and
display names. Kept as a thin shim so callers can `from tri_statistics_STEP.config
import ...` and stay decoupled from the original package.
"""

from __future__ import annotations

from tri_statistics.config import (  # noqa: F401
    ALL_CELLS,
    DATASETS,
    MODEL_DISPLAY,
    MODEL_HIGH_STEPS,
    MODELS_WITH_EPISODES_LIBERO,
    MODELS_WITH_EPISODES_LIBEROPLUS,
    OVERALL_KEY,
    PAIR_SPECS,
    PairSpec,
    SUITE_DISPLAY,
    SUITE_ORDER,
    build_pair_specs,
)
