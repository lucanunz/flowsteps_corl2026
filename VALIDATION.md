# Migration validation — 2026-09-10

> Updated 2026-09-09: the Polaris benchmark and the GR00T model were removed from
> this export (data, figures, loaders, pair specs and inventories).
>
> Updated 2026-09-10: the superseded `legacy` evaluation cohort was removed —
> 22 settings, 3,605 episodes, including the `flower_old` model. 20 of the 22 had
> a direct non-legacy replacement at the same model/benchmark/step setting and
> six carried no episodes at all. 281 source documents became unreferenced and
> were pruned from `source_inventory.json`; `shared_inputs.json` was unaffected.
> No figure output changed.
>
> Updated 2026-09-10: the RoboTwin Pi0.5 per-run exports and the
> `robomimic_tool_hang` settings were removed — 32 settings, 17,400 episodes
> (28 `run-<timestamp>_<config>` RoboTwin Pi0.5 settings carrying all 17,400;
> four aggregate-only robomimic checkpoint settings carrying none). Neither
> group was read by any figure or table: `scripts/figures/robotwin/` declares
> X-VLA pair specs only, and no retained script loads a robomimic path.
> 317 source documents became unreferenced and were pruned from
> `source_inventory.json`; `shared_inputs.json` was unaffected. RoboTwin now
> covers X-VLA only, and `robomimic_tool_hang` is no longer a benchmark in this
> export. No figure output changed. The removed data is unchanged in the
> original log repository. Counts below reflect the 193 remaining settings and
> seven models.

## Preserved data

- 193 setting JSON files; 444,329 normalized episode records.
- 7,736 statistical JSON/JSONL source documents retained with their values and
  dictionary/list order unchanged. SHA-256 and parsed-payload comparisons passed
  against the previous normalized checkout; all 7,736 were re-verified after the
  RoboTwin Pi0.5 / robomimic removal.
- 1,160 Pi0.5 console logs hashed and parsed. They contain 107,037 explicit
  task/episode/outcome observations before deduplication. Trial-sharded logs use
  their recorded episode-range offset; shard-local episode counters are not
  conflated across shards.
- DiT checkpoint and XR-0 extra-step text outcomes were converted into JSON
  episode sidecars.
- CALVIN retains completion scores and sequence identities; real-world data
  retains session-qualified pair IDs, condition order, policy seeds, initial
  positions and scalar rollout metrics. Source payloads retain additional fields.
- Repeated/copy/supplemental source documents remain explicit instead of being
  silently pooled into the primary selection.

A complete fresh migration into a separate directory produced byte-identical
JSON for all settings, shared inputs, task metadata and source inventories.

`validation.json` lists every setting, trial count, pairing basis and primary
pairing coverage. `source_inventory.json` lists source JSON/JSONL hashes.

## Statistical checks

All **197 joint outcome distributions** checked against the pre-migration loaders
match: 70 LIBERO/LIBERO+ aggregate/suite comparisons, 100 RoboTwin task comparisons,
24 real-world task comparisons, and three CALVIN completion-score comparisons.
This checks paired joint outcomes, not just marginal success rates.

Primary 1/max-step pairing coverage:

- CALVIN: 1,000 sequences per model.
- LIBERO: 2,000 episodes per model.
- LIBERO+: 10,030 paired variants per model, except Pi0.5 and XR-0 with 9,976
  identifiable single-trial variants. Their additional multi-trial marginal
  counts are preserved; no individual pairing is inferred from those totals.
- RoboTwin: 5,000 seeds per setting, including randomized-run supplements.
- Real world: 400 pairs per model, with session identity preserved.

18 automated tests pass, covering identity-based alignment, reordered input,
missing pairs, null outcomes, duplicate keys, session/suite separation, explicit
ordering assumptions, refusal to fabricate paired data from marginals, exact
McNemar reference values, paired-t confidence limits, condition-swap symmetry,
source-disagreement handling and text-log episode/shard parsing.

## Existing figure reproduction

`python reproduce.py all` completed using only the packed JSON and retained
scripts in a temporary directory. No simulator/task-package installation is
needed; task metadata is frozen in `task_metadata.json`.

Compared the generated artifacts with the previous checkout:

- 185 PNGs: byte-identical.
- 71 statistical reports: byte-identical.
- 8 LaTeX tables: byte-identical.
- 40 PDFs generated successfully; byte identity is not claimed because their
  metadata can include generation timestamps.

The new overview plots are separate from this comparison. Existing paper plot
algorithms and statistical defaults were preserved.

## Source limitations retained explicitly

The Pi0.5 `29k_pretrainedVLM` checkpoint has only 1,000 logged outcomes at one
inference step and 37 at ten steps, while its aggregate JSONs contain complete
2,000-trial marginal counts. Missing task logs are recorded in each file's
`normalization.log_aggregate_mismatches`. Full marginal counts and the observed
trial subset are both preserved. The paired analysis refuses an implicit
complete-run comparison here.

One Pi0.5 horizon log contains an extra success/failure message without intact
trial context. It is retained in `unidentified_log_outcomes` and is not assigned
to an episode. Complete identifiable outcomes still reconcile with that setting's
aggregate counts.

Some older aggregate-only runs cannot establish individual pairing. Their
original statistical payloads are retained, and the paired CLI refuses
comparisons without observed paired rows. Positional identities in older X-VLA
evaluations remain labeled as an evaluation-order assumption, requiring
`--allow-order`.

The previous cleaned checkout and the raw log tree are preserved outside this
repository and were not modified. Reproducing the statistics and figures from the
committed JSON does not require either of them.
