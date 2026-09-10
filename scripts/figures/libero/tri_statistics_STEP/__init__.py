"""STEP-based variant of the LIBERO / LIBERO+ 1-step vs max-step pipeline.

This package mirrors `tri_statistics/` but swaps the primary paired test from
McNemar mid-p to STEP (Sequential Testing for Efficient Policy comparison;
TRI-ML's `sequentialized_barnard_tests` package). For LIBERO/LIBERO+ rollout
budgets McNemar is underpowered (discordant counts are tiny) and STEP recovers
power on the same data.
"""
