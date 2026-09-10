"""Statistical primitives shim for the real-world STEP pipeline.

Re-exports the STEP wrapper / CLD helpers / multiplicity correctors from the
LIBERO STEP package. The LIBERO STEP package shares our package name
(`tri_statistics_STEP`), so we load its `stats.py` by file path via
`importlib.util` to avoid the name clash.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent          # tri_statistics_STEP/
_REAL_WORLD_DIR = _PKG_DIR.parent                   # real_world/
_ROOT_DIR = _REAL_WORLD_DIR.parent                  # corl2026_logs/
_LIBERO_DIR = _ROOT_DIR / "libero"

for _p in (_REAL_WORLD_DIR, _ROOT_DIR, _LIBERO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analyze_libero_steps import (  # type: ignore  # noqa: E402,F401
    Counts,
    two_proportion_z_pvalue,
)

from tri_statistics.stats import (  # type: ignore  # noqa: E402,F401
    benjamini_hochberg_qvalues,
    beta_posterior_samples,
    bonferroni,
    cld_two_methods,
    holm,
)

_LIBERO_STEP_STATS_PATH = _LIBERO_DIR / "tri_statistics_STEP" / "stats.py"
_spec = importlib.util.spec_from_file_location(
    "_libero_tri_statistics_STEP_stats", _LIBERO_STEP_STATS_PATH
)
assert _spec is not None and _spec.loader is not None
_libero_step_stats = importlib.util.module_from_spec(_spec)
sys.modules["_libero_tri_statistics_STEP_stats"] = _libero_step_stats
_spec.loader.exec_module(_libero_step_stats)

Decision = _libero_step_stats.Decision
DECISION_LABEL = _libero_step_stats.DECISION_LABEL
DECISION_CODE = _libero_step_stats.DECISION_CODE
StepResult = _libero_step_stats.StepResult
step_test = _libero_step_stats.step_test
cld_from_step_decision = _libero_step_stats.cld_from_step_decision

from wilson_score_interval import wilson_score_interval  # type: ignore  # noqa: E402,F401
