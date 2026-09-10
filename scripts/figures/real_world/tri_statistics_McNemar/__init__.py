"""McNemar paired-difference test for the real-world rollouts.

This is the *difference*-test counterpart to the equivalence (TOST) fork in the
sibling ``tri_statistics_TOST`` package. Where TOST asks "can we demonstrate that
1-step and the multi-step reference are equivalent within δ?", McNemar asks the
complementary question: "is there a detectable difference between them at all?".

The test operates on the *discordant* pairs of the paired 2×2 table only:

        a = both succeed     b = 1-step only          (1-step "wins")
        c = reference only   d = both fail            n = a + b + c + d

Under H₀ (the two policies have equal marginal success rates) the discordant
pairs split symmetrically, so b ~ Binomial(b + c, ½). Following the
pre-registered rule we use the **exact binomial** test when the number of
discordant pairs is small (b + c < 25) and the **χ² approximation with Yates'
continuity correction** otherwise.

Data loading and the paired 2×2 tables are reused verbatim from the sibling
``tri_statistics_STEP`` package. Paired data only — cells without a paired table
are reported "not assessed".
"""
