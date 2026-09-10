"""Pre-registered equivalence constants for the TOST pipeline.

Display names, pair specs and data loaders are pulled from the sibling
`tri_statistics_STEP` package per benchmark; only the equivalence settings live
here so they are identical across all benchmarks.
"""

from __future__ import annotations

# δ: equivalence margin in percentage points (the 3pp band).
DELTA_MARGIN_PP: float = 3.0

# TOST level. Equivalence at α ⇔ the two-sided (1 − 2α) CI ⊂ ±δ. α = 0.05 ⇒ 90% CI.
TOST_ALPHA: float = 0.05
