"""Statistical primitives for the STEP variant.

Re-exports the legacy primitives (Wilson, Fisher, Beta-posterior samples,
Bonferroni / Holm / BH) from `tri_statistics.stats`, drops `mcnemar_midp`, and
adds a thin wrapper around STEP from `sequentialized_barnard_tests`.

STEP is decision-based, not p-value-based: at a pre-set α, it returns one of
{AcceptNull, AcceptAlternative, FailToDecide}. We use the mirrored one-sided
test, which interprets AcceptAlternative as `policy 1 > policy 0` and
AcceptNull as the reverse direction. FailToDecide means the data does not
separate the two policies under the chosen rollout budget.
"""

from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass
from typing import Sequence

from sequentialized_barnard_tests import (
    Decision,
    Hypothesis,
    get_mirrored_test,
)

from tri_statistics.stats import (  # noqa: F401
    benjamini_hochberg_qvalues,
    beta_posterior_mean,
    beta_posterior_samples,
    bonferroni,
    cld_from_pmatrix,
    cld_two_methods,
    fisher_exact_2x2,
    holm,
)


DECISION_LABEL: dict[Decision, str] = {
    Decision.AcceptAlternative: "high > low",
    Decision.AcceptNull:        "low > high",
    Decision.FailToDecide:      "no separation",
}

# Short codes for plot annotations / heatmaps.
DECISION_CODE: dict[Decision, str] = {
    Decision.AcceptAlternative: "H",   # max-step better
    Decision.AcceptNull:        "L",   # 1-step better
    Decision.FailToDecide:      "·",
}


@dataclass
class StepResult:
    """Outcome of a single STEP comparison (one α, one n_max)."""
    decision: Decision
    decision_name: str          # enum name, e.g. "AcceptAlternative"
    decision_label: str         # human-readable, e.g. "high > low"
    stop_time: int              # info["Time"] — number of paired rollouts examined
    n_max: int                  # n_max budget passed to STEP
    alpha: float                # α level the test was run at
    n_paired: int               # actual number of paired rollouts available


def _silence_synthesis_output():
    """Library's first-run policy synthesis prints to stdout via tqdm; mute it."""
    return contextlib.redirect_stdout(io.StringIO())


def step_test(
    seq_low: Sequence[bool],
    seq_high: Sequence[bool],
    *,
    alpha: float,
    n_max: int | None = None,
) -> StepResult:
    """Run the mirrored STEP test treating `seq_low` as policy 0 and `seq_high` as policy 1.

    `alternative = Hypothesis.P0LessThanP1`, so the decisions translate as:
      * AcceptAlternative → policy 1 (max-step) significantly outperforms policy 0 (1-step)
      * AcceptNull        → policy 0 (1-step) significantly outperforms policy 1 (max-step)
      * FailToDecide      → no statistical separation at this α and n_max

    `n_max` defaults to the matched sample size (`min(len(seq_low), len(seq_high))`).
    For retrospective analysis this is the natural choice: STEP examines exactly
    the data we collected.
    """
    n = min(len(seq_low), len(seq_high))
    if n_max is None:
        n_max = n
    if n_max <= 0:
        return StepResult(
            decision=Decision.FailToDecide,
            decision_name=Decision.FailToDecide.name,
            decision_label=DECISION_LABEL[Decision.FailToDecide],
            stop_time=0,
            n_max=0,
            alpha=alpha,
            n_paired=n,
        )

    with _silence_synthesis_output():
        test = get_mirrored_test(
            n_max=n_max,
            alternative=Hypothesis.P0LessThanP1,
            alpha=alpha,
        )
        result = test.run_on_sequence(
            [bool(x) for x in seq_low[:n_max]],
            [bool(x) for x in seq_high[:n_max]],
        )

    return StepResult(
        decision=result.decision,
        decision_name=result.decision.name,
        decision_label=DECISION_LABEL[result.decision],
        stop_time=int(result.info.get("Time", n_max)),
        n_max=n_max,
        alpha=alpha,
        n_paired=n,
    )


def cld_from_step_decision(
    label_low: str,
    label_high: str,
    decision: Decision | None,
) -> dict[str, str]:
    """CLD for the two-method case driven by a STEP decision.

    Same letter unless STEP separated the methods (in *either* direction).
    `None` (no STEP run — e.g. the no-pair fallback path) yields ('a', 'a').
    """
    if decision is None or decision == Decision.FailToDecide:
        return {label_low: "a", label_high: "a"}
    return {label_low: "a", label_high: "b"}
