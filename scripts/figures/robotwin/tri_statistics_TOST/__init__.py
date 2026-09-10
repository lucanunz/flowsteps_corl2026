"""Equivalence (TOST) variant of the 1-step vs max-step pipeline.

Where `tri_statistics_STEP/` runs a *difference* test (mirrored STEP) and reports
`FailToDecide` when the two policies do not separate, this package runs a paired
*equivalence* test (TOST) of the success-rate difference against a pre-registered
margin δ = 3pp. A `FailToDecide` is absence of evidence; TOST instead positively
demonstrates equivalence within δ (or flags a genuine ≥ δ difference, or declares
the data inconclusive).

Paired data only — cells without a paired 2×2 table are reported "not assessed".
Data loading is reused from the sibling `tri_statistics_STEP` package.
"""
