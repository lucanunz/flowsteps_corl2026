"""Data loading and formatting shared by the LIBERO glyph tables."""

from __future__ import annotations

from pathlib import Path

from analyze_libero_steps import (
    DISPLAY_NAMES,
    MODEL_ORDER,
    RUN_SPECS,
    SUITES,
    Counts,
    RunResult,
    load_run,
)


SUITE_LABELS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "10",
}

STEP_OUTPUT_DIR = Path("tri_statistics_STEP") / "outputs"

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


# Typeset model names for the LIBERO LaTeX tables. These carry math mode and
# small-caps styling, so they bypass `latex_escape`; every other model falls
# back to the plain `DISPLAY_NAMES` entry. Kept separate from `DISPLAY_NAMES`
# because that mapping also feeds the plots and the cross-benchmark scatter,
# which must stay plain text.
LATEX_DISPLAY_NAMES = {
    "flower": r"FLOWER",
    "pi05": r"$\pi_{0.5}$",
}


def latex_model_name(model_key: str) -> str:
    """Table-ready model name, escaped unless it is already LaTeX."""
    if model_key in LATEX_DISPLAY_NAMES:
        return LATEX_DISPLAY_NAMES[model_key]
    return latex_escape(DISPLAY_NAMES[model_key])


_STEP_DIRECTION_TO_DELTA_SIGN = {
    "low > high": 1,
    "high > low": -1,
}


def strip_trailing_period(text: str) -> str:
    return text[:-1] if text.endswith(".") else text


def format_counts(counts: Counts, precision: int) -> str:
    return f"{counts.rate * 100:.{precision}f}\\% ({counts.successes}/{counts.trials})"


def format_percent(counts: Counts, precision: int) -> str:
    return f"{counts.rate * 100:.{precision}f}\\%"


def _split_markdown_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def _suite_label_to_key() -> dict[str, str]:
    labels = {label: suite for suite, label in SUITE_LABELS.items()}
    labels["LIBERO-10"] = "libero_10"
    labels["Overall"] = "overall"
    return labels


def _decision_delta_sign(decision: str) -> int | None:
    return _STEP_DIRECTION_TO_DELTA_SIGN.get(decision)


def _parse_aggregate_step_directions(report_path: Path) -> dict[str, int]:
    """Return corrected aggregate STEP directions keyed by suite/overall key."""
    if not report_path.exists():
        return {}
    label_to_key = _suite_label_to_key()
    directions: dict[str, int] = {}
    in_step_table = False
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Suite | n_paired |"):
            in_step_table = True
            continue
        if not in_step_table:
            continue
        if not line.startswith("|"):
            if line.strip():
                in_step_table = False
            continue
        if line.startswith("| ---"):
            continue
        parts = _split_markdown_row(line)
        if len(parts) < 8:
            continue
        suite_key = label_to_key.get(parts[0])
        direction = _decision_delta_sign(parts[7])
        if suite_key is not None and direction is not None:
            directions[suite_key] = direction
    return directions


def _parse_per_task_step_directions(report_path: Path) -> dict[str, set[int]]:
    """Return corrected per-task STEP directions grouped by suite key."""
    if not report_path.exists():
        return {}
    label_to_key = _suite_label_to_key()
    directions: dict[str, set[int]] = {}
    current_suite: str | None = None
    alpha_split_index: int | None = None

    for line in report_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current_suite = label_to_key.get(line[3:].strip())
            alpha_split_index = None
            continue
        if current_suite is None or not line.startswith("|"):
            continue
        parts = _split_markdown_row(line)
        if not parts:
            continue
        if parts[0] == "Task":
            alpha_split_index = parts.index("STEP α_s") if "STEP α_s" in parts else None
            continue
        if parts[0] == "---" or alpha_split_index is None or len(parts) <= alpha_split_index:
            continue
        direction = _decision_delta_sign(parts[alpha_split_index])
        if direction is not None:
            directions.setdefault(current_suite, set()).add(direction)
    return directions


def load_step_significance(libero_dir: Path, dataset: str) -> dict[tuple[str, str], set[int]]:
    """Load compact marker directions from the STEP reports.

    Suite columns use corrected per-task STEP decisions within that suite.
    The Average column uses the corrected Overall STEP decision from the
    aggregate report.
    """
    out_dir = libero_dir / STEP_OUTPUT_DIR / dataset
    markers: dict[tuple[str, str], set[int]] = {}
    for model_key in MODEL_ORDER:
        per_task = _parse_per_task_step_directions(out_dir / f"{model_key}_per_task_report.md")
        for suite, directions in per_task.items():
            markers[(model_key, suite)] = set(directions)

        aggregate = _parse_aggregate_step_directions(out_dir / f"{model_key}_report.md")
        if "overall" in aggregate:
            markers[(model_key, "overall")] = {aggregate["overall"]}
    return markers


def load_libero_results(libero_dir: Path) -> list[RunResult]:
    specs = [spec for spec in RUN_SPECS if spec.dataset == "libero"]
    return [load_run(libero_dir, spec) for spec in specs]


