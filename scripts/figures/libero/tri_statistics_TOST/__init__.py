"""Equivalence (TOST) variant of the 1-step vs max-step pipeline.

Where `tri_statistics_STEP/` runs a *difference* test (mirrored STEP) and reports
`FailToDecide` when the two policies do not separate, this package runs an
*equivalence* test: a two-one-sided-test (TOST) of the paired success-rate
difference against a pre-registered margin δ. A `FailToDecide` is absence of
evidence; TOST instead lets us *positively demonstrate* equivalence within δ
(or flag a genuine ≥ δ difference, or declare the data inconclusive).

Paired data only: the test is computed on the paired per-rollout 2×2 table. Cells
without paired episode data are reported as "not assessed" — there is no
unpaired fallback. Data loading is reused from the sibling `tri_statistics_STEP`
package; only the statistics, reporting and figures are equivalence-specific.
"""
