#!/usr/bin/env python3
"""Regenerate the retained figures, simulation analyses, and tables."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "scatter": [("analyse_flow_ablation_crossbench.py",)],
    "tost": [("TOST_analysis/gen_tost_verdict_tables.py", "--delta-pp", "3", "--layout", "grouped")],
    "simulation": [
        ("libero/tri_statistics_TOST/run.py", "--delta-pp", "3"),
        ("libero/tri_statistics_TOST/run.py", "--delta-pp", "3", "--per-task"),
        ("calvin/tri_statistics_TOST/run.py", "--delta-pp", "3", "--chain-delta", "0.15"),
        ("robotwin/tri_statistics_TOST/run.py", "--delta-pp", "3"),
        ("robotwin/tri_statistics_TOST/run.py", "--delta-pp", "3", "--per-task"),
    ],
    "violins": [
        ("libero/tri_statistics_STEP/run.py",),
        ("libero/tri_statistics_STEP/run.py", "--per-task"),
    ],
    "glyphs": [
        ("calvin/tri_statistics_STEP/run.py",),
        ("calvin/make_calvin_latex_table_glyph.py", "--layout", "grouped", "--precision", "1", "--bold-best", "--output", "calvin/calvin_table_glyph.tex"),
        ("libero/make_libero_combined_latex_table_glyph.py", "--layout", "pair", "--cell-format", "percent", "--bold-best", "--output", "libero/libero_liberoplus_compact_glyph.tex"),
        ("libero/make_libero_latex_table_glyph.py", "--cell-format", "percent", "--output", "libero/libero_table_glyph.tex"),
        ("libero/make_libero_latex_table_glyph.py", "--dataset", "liberoplus", "--cell-format", "percent", "--output", "libero/liberoplus_table_glyph.tex"),
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["all", *COMMANDS], default="all", nargs="?")
    args = parser.parse_args()
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for target in COMMANDS if args.target == "all" else [args.target]:
        for command in COMMANDS[target]:
            print(f"[{target}] python {' '.join(command)}", flush=True)
            subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
