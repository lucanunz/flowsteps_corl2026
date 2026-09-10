"""Stats shim. Re-exports primitives from libero's packages and from
libero/tri_statistics_STEP/stats.py (loaded by file path to avoid the
package-name clash between libero/, real_world/, and robotwin/ STEP packages).
"""

from __future__ import annotations

import importlib.util
import sys

from .config import LIBERO_DIR, LIBERO_STEP_DIR, REPO_ROOT


for _p in (REPO_ROOT, LIBERO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


from analyze_libero_steps import (  # type: ignore  # noqa: E402,F401
    Counts,
    two_proportion_z_pvalue,
)
from tri_statistics.stats import (  # type: ignore  # noqa: E402,F401
    benjamini_hochberg_qvalues,
    beta_posterior_mean,
    beta_posterior_samples,
    bonferroni,
    cld_from_pmatrix,
    cld_two_methods,
    fisher_exact_2x2,
    holm,
)
from wilson_score_interval import wilson_score_interval  # type: ignore  # noqa: E402,F401

# libero/tri_statistics_STEP/stats.py — load by file path to dodge the
# `tri_statistics_STEP` name collision with this package.
_LIBERO_STEP_STATS_PATH = LIBERO_STEP_DIR / "stats.py"
_spec = importlib.util.spec_from_file_location(
    "_libero_tri_statistics_STEP_stats", _LIBERO_STEP_STATS_PATH
)
assert _spec is not None and _spec.loader is not None
_libero_step_stats = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_libero_tri_statistics_STEP_stats", _libero_step_stats)
_spec.loader.exec_module(_libero_step_stats)

Decision = _libero_step_stats.Decision
DECISION_LABEL = _libero_step_stats.DECISION_LABEL
DECISION_CODE = _libero_step_stats.DECISION_CODE
StepResult = _libero_step_stats.StepResult
step_test = _libero_step_stats.step_test
cld_from_step_decision = _libero_step_stats.cld_from_step_decision
