"""Pre-registered equivalence constants for the TOST pipeline.

Display names, pair specs and data loaders are pulled from the sibling
`tri_statistics_STEP` package per benchmark; only the equivalence settings live
here so they are identical across all benchmarks.
"""

from __future__ import annotations

# δ: equivalence margin in percentage points (the 3pp band).
DELTA_MARGIN_PP: float = 3.0

# δ for the mean chain-length (0–5 subtask) equivalence test, in subtask units.
# 3% of the 0–5 scale — the chain-length analogue of DELTA_MARGIN_PP (3pp on [0,1]).
DELTA_MARGIN_CHAIN_SUBTASKS: float = 0.15

# TOST level. Equivalence at α ⇔ the two-sided (1 − 2α) CI ⊂ ±δ. α = 0.05 ⇒ 90% CI.
TOST_ALPHA: float = 0.05
