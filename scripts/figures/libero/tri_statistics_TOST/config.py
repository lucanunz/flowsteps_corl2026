"""Configuration for the paired equivalence (TOST) pipeline.

Re-export of `tri_statistics.config` — same models, datasets, suites and display
names as the STEP pipeline — plus the pre-registered equivalence constants.
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

# ── Pre-registered equivalence settings ──────────────────────────────────────
# δ: the equivalence margin in percentage points. A 1-step vs reference
# difference whose CI lies entirely within ±δ is declared *equivalent*. This is
# the same 3pp band drawn (implicitly) in the cross-benchmark step-gap figure.
DELTA_MARGIN_PP: float = 3.0

# TOST level. Two one-sided tests, each at α; declaring equivalence at α is
# equivalent to checking that the two-sided (1 − 2α) CI ⊂ (−δ, +δ).
# With α = 0.05 the relevant interval is the 90% CI.
TOST_ALPHA: float = 0.05
