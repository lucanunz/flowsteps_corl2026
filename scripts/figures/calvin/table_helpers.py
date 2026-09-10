"""Data loading and formatting shared by the CALVIN glyph tables."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from tri_statistics_STEP.config import MODEL_DISPLAY, MODEL_HIGH_STEPS  # noqa: E402


CALVIN_DATA_ROOT = _THIS_DIR / "abc_d"
CHAIN_LEVELS = ("1", "2", "3", "4", "5")
STEP_REPORT_ROOT = _THIS_DIR / "tri_statistics_STEP" / "outputs" / "abc_d"

_STEP_DIRECTION_TO_DELTA_SIGN = {
    "low > high": 1,
    "high > low": -1,
}


@dataclass(frozen=True)
class RunResult:
    model_key: str
    step_label: str
    avg_seq_len: float
    chain_sr: dict[str, float]


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def format_percent(value: float, precision: int) -> str:
    return f"{value * 100.0:.{precision}f}\\%"


def format_number(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


def _split_markdown_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def _decision_delta_sign(decision: str) -> int | None:
    return _STEP_DIRECTION_TO_DELTA_SIGN.get(decision)


def _parse_p_value(value: str) -> float:
    value = value.strip()
    if value.startswith("<"):
        return 0.0
    return float(value)


def _load_calvin_significance() -> dict[tuple[str, str], int]:
    """Load corrected SR@5 STEP and Avg. Len. Welch marker directions."""
    markers: dict[tuple[str, str], int] = {}
    for model_key in MODEL_HIGH_STEPS:
        report_path = STEP_REPORT_ROOT / f"{model_key}_report.md"
        if not report_path.exists():
            continue
        in_step_table = False
        in_welch_table = False
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("| Cell | n_paired |"):
                in_step_table = True
                in_welch_table = False
                continue
            if line.startswith("| Δ mean |"):
                in_welch_table = True
                in_step_table = False
                continue
            if not line.startswith("|"):
                if line.strip():
                    in_step_table = False
                    in_welch_table = False
                continue
            if line.startswith("| ---"):
                continue

            parts = _split_markdown_row(line)
            if in_step_table and len(parts) >= 8 and parts[0] == "SR@5":
                direction = _decision_delta_sign(parts[7])
                if direction is not None:
                    markers[(model_key, "5")] = direction
            elif in_welch_table and len(parts) >= 6:
                try:
                    delta_high_minus_low = float(parts[0])
                    p_value = _parse_p_value(parts[4])
                except ValueError:
                    continue
                if p_value < 0.05 and " / " in parts[5]:
                    direction = -1 if delta_high_minus_low > 0 else 1 if delta_high_minus_low < 0 else 0
                    if direction != 0:
                        markers[(model_key, "avg_len")] = direction
    return markers


def result_path(model_key: str, steps: int) -> Path:
    if model_key == "flower":
        return CALVIN_DATA_ROOT / "flower" / f"abc_d_{steps}step" / "results.json"
    if model_key == "mibot":
        return CALVIN_DATA_ROOT / "mibot" / f"rerun_abc_d_numsteps{steps}" / "results.json"
    if model_key == "xvla":
        return CALVIN_DATA_ROOT / "X-VLA" / f"{steps}step" / "results.json"
    raise KeyError(f"Unknown CALVIN model key: {model_key}")


def load_run_result(model_key: str, steps: int) -> RunResult:
    path = result_path(model_key, steps)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not raw:
        raise ValueError(f"Empty results file: {path}")
    payload = next(iter(raw.values()))
    return RunResult(
        model_key=model_key,
        step_label=str(steps),
        avg_seq_len=float(payload["avg_seq_len"]),
        chain_sr={level: float(payload["chain_sr"][level]) for level in CHAIN_LEVELS},
    )


def load_calvin_results() -> dict[str, list[RunResult]]:
    by_model: dict[str, list[RunResult]] = {}
    for model_key, high_steps in MODEL_HIGH_STEPS.items():
        by_model[model_key] = [
            load_run_result(model_key, high_steps),
            load_run_result(model_key, 1),
        ]
    return by_model


