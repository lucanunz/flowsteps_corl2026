# CoRL 2026 statistical data

Paired evaluation logs for seven policies across LIBERO, LIBERO+, CALVIN,
RoboTwin and a real-robot setup, normalized to one JSON per
model / benchmark / inference-step setting in `data/` (193 settings,
444,329 episodes, seven policies). Everything under `figures/` is generated and
untracked.

## Setup

Use python 3.11 or newer.

```bash
micromamba create -n flowsteps python=3.11 -y
python -m pip install -r requirements.txt          # validation + overview plots
python -m pip install -r requirements-figures.txt  # paper figures and tables
```

## Generate the assets

```bash
python scripts/validate.py                 # re-checks the export, writes validation.json
python scripts/plot.py                     # overview figures
python reproduce.py all                    # all paper figures, reports and tables
python reproduce.py ablations              # training-progress ablation figures
```

`reproduce.py` also takes a single target: `scatter`, `tost`, `simulation`,
`violins`, `glyphs`, or `ablations`. Run `all` when regenerating `glyphs` — the LIBERO glyph
tables consume the STEP reports written by `violins`. Use `--output DIR` to
write somewhere other than `figures/`.

One-off comparison of two settings:

```bash
python scripts/analyze.py data/robotwin_clean__xvla__n1.json \
                          data/robotwin_clean__xvla__n10.json
```

It prints the paired 2×2 table, an exact two-sided McNemar p-value and a paired
equivalence test (default margin 3pp, alpha 0.05; CALVIN chains use a
0.15-subtask margin). See `--suite`, `--task`, `--margin`, `--chain-margin`,
`--alpha`, `--allow-order`, `--allow-partial`, `--output`.

## Where the assets go

| Command | Output | What it is |
| --- | --- | --- |
| `scripts/plot.py` | `figures/overview/<benchmark>.{png,pdf}` | Success rate versus inference steps, one panel set per benchmark |
| `reproduce.py scatter` | `figures/flow_ablation_crossbench/perfamily/`, `.../per_family_SGOavg/` | Cross-benchmark flow-ablation scatter plots, per model family and family-averaged |
| `reproduce.py tost` | `figures/TOST_analysis/verdict_tables/tost_<benchmark>_compact.tex` | Compact LaTeX equivalence-verdict tables (δ = 3pp) for LIBERO, CALVIN, RoboTwin and real world |
| `reproduce.py simulation` | `figures/{libero,calvin,robotwin}/tri_statistics_TOST/outputs/` | TOST equivalence figures plus `equivalence_report.md` / `equivalence_per_task_report.md`, pooled and per task |
| `reproduce.py violins` | `figures/libero/tri_statistics_STEP/outputs/` | LIBERO / LIBERO+ per-step violin figures, appendix figures and per-model `*_report.md` STEP reports |
| `reproduce.py glyphs` | `figures/calvin/tri_statistics_STEP/outputs/`, `figures/calvin/calvin_table_glyph.tex`, `figures/libero/*_glyph.tex` | CALVIN STEP figures and reports, plus the glyph LaTeX result tables used in the paper |
| `scripts/validate.py` | `validation.json` (tracked) | Provenance and pairing-coverage report |
| `reproduce.py ablations` | `figures/ablations/plot_grid_step_diff.{png,pdf}`, `.../plot{1,2,3}_*.{png,pdf}` | LIBERO step gap versus % of training — the 2×3 grid, plus the same ablations as individual success-rate and step-gap panels |

A full `reproduce.py all` plus `plot.py` writes roughly 33 MB of PNG, PDF, TeX
and Markdown into `figures/`. The `ablations` target is the exception to the
`reproduce.py` model: its scripts read `data/` directly rather than the packed
source documents, so they run in place instead of in the temporary tree. See
`ablations/README.md`.

## Data

Each setting file carries `setting` (model, benchmark, `nsteps`, checkpoint,
evaluation cohort, horizon where recorded), `episodes` with pairing identities,
reported marginal counts, and `source_documents` — the original statistical
payloads with SHA-256 provenance.

Join within the same model / checkpoint / evaluation by
**(suite, task, episode_id)**, never by row position or aggregate success rate.
Different models use different task identifiers; this export does not establish
cross-model pairing.

| Data | Trial identity |
| --- | --- |
| LIBERO | Task plus recorded episode index |
| LIBERO+ | Exact task/perturbation variant plus episode index |
| RoboTwin | Task plus simulation seed |
| Real world | Task plus session timestamp and pair ID |
| CALVIN | Initial state and full task chain, or recorded sequence index |

Supporting files: `task_metadata.json` (frozen LIBERO/LIBERO+ task tables),
`shared_inputs.json` (source documents shared across settings),
`source_inventory.json` and `validation.json` (provenance and coverage).

Two settings are **aggregate-only**, both on the Pi0.5 `29k_pretrainedVLM`
checkpoint. Their evaluations ran in full — 40 tasks, 2,000 trials each, with
complete per-task success/failure counts in `task_counts`. What is missing is
only the per-episode console record, which survives for some suites and not
others: at 10 steps, 37 of 2,000 episodes (`libero_goal` and `libero_spatial`
only); at 1 step, 1,000 of 2,000 (`libero_10` and `libero_object` only). No
trials are unaccounted for; the episode-level transcript that pairing needs was
simply never captured for the remaining suites.

Use their `task_counts` for marginal success rates. They cannot be paired
against each other under any flag — the surviving suites are disjoint, so the
intersection is empty and `analyze.py` fails on them with or without
`--allow-partial`. See `VALIDATION.md`.

`AGENTS.md` documents the pairing rules, statistical conventions and repository
layout in full.
