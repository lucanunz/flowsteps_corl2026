"""Pre-registered constants for the McNemar difference-test pipeline.

Display names, pair specs and data loaders are pulled from the sibling
``tri_statistics_STEP`` package per benchmark; only the McNemar settings live here
so they are identical across any benchmark that grows a McNemar fork.
"""

from __future__ import annotations

# Significance level for the (two-sided) McNemar test.
MCNEMAR_ALPHA: float = 0.05

# Method-selection threshold on the number of *discordant* pairs (b + c).
# Strictly below this we use the exact binomial test; at or above it we use the
# χ² approximation with Yates' continuity correction.
EXACT_MAX_DISCORDANT: int = 25
