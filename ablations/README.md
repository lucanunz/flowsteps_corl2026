# Ablation figures

Training-progress ablations for the 1-step vs multi-step comparison: success
rate and the step gap as a function of training progress, on LIBERO.

```bash
python reproduce.py ablations                         # both scripts, into figures/ablations/
python ablations/plot_paper_training_curves_grid.py   # 2x3 grid of step-gap panels
python ablations/plot_paper_training_curves.py        # the individual panels
```

`reproduce.py ablations` runs them in place rather than in the temporary source
tree the other targets use: these scripts read the normalized rows in `data/`,
not the packed `source_documents`.

Both take `--data` (default: the repo's `data/`) and `--out_dir` (default:
`figures/ablations/`, untracked like the rest of `figures/`).
`plot_paper_training_curves.py`
also takes `--results` to plot one suite or a comma-separated subset instead of
the four-suite mean, and `--show_legend`.

## What they read

The `evaluation == "checkpoint"` settings on LIBERO, loaded through
`scripts/paired.py` like the rest of the repository. Per-suite rates come from
`task_counts` when a file carries them — needed for the two aggregate-only Pi0.5
`29k_pretrainedVLM` settings — and from the episode rows otherwise.

| Series | Setting files | Train steps |
| --- | --- | --- |
| Pi0.5, four init variants | `libero__pi05__n{1,10}__checkpoint__ckpt-{5,10,15,20,25,29}k[_variant]` | 5k–29k |
| X-VLA | `libero__xvla__n{1,10}__checkpoint__ckpt-{10000…60000}` | 10k–60k |
| FLOWER | `libero__flower__n{1,4}__checkpoint__ckpt-{8000…40000}` | 8k–40k |
| DiT | `libero__dit__n{1,10}__checkpoint__ckpt-pretrained-{10000,20000,final_model}` | 10k–30k |

`final_model` is checkpoint_30000. The `libero10only`, `0noise` and
`unspecified` Pi0.5 settings are skipped. The DiT `scratchALL` and
`scratchVisOnly` sweeps load but no plot spec uses them.

**One substitution.** The X-VLA curve's 60k point is read from the primary
evaluation (`libero__xvla__n{1,10}.json`), not from the checkpoint cohort's
`ckpt-60000` rerun — see `PRIMARY_SUBSTITUTES` in
`plot_paper_training_curves.py`. The rerun scores ~12pp lower on LIBERO-10
(84.4 vs 96.0 at n=1); the primary evaluation is what the paper tables report.

## Model variants

Every model here is a VLA with a VLM backbone and a flow-matching action
expert. The Pi0.5 checkpoint suffixes name how the two halves were initialized:

- **pretrained** (no suffix) — both components pretrained on robot data.
- **pretrainedVLM** — VLM from generic (non-robot) pretrained weights, action
  expert from scratch.
- **scratchActionExpOnly** — VLM pretrained on robot data, action expert from
  scratch.
- **scratchEverything** — both from scratch. Spelled `scratchEveryting` in the
  25k checkpoint name; the loader folds the two spellings into one series.

The DiT baseline exists to check whether step-count invariance shows up without
large-scale robot pretraining: a CLIP ViT encoder with a from-scratch action
expert, no robot pretraining. `outputs/libero_dit_bs256/checkpoint_30000`
(hidden_dim 512, 6 layers, 8 heads) is 211.0M parameters:

| Component | Params | Notes |
| --- | --- | --- |
| observation_encoder | 149.2M | CLIP text encoder 63.4M (frozen) + CLIP ViT-B/16 vision encoder and projections 85.8M (fine-tuned at 0.1x LR) |
| noise_predictor (DiT) | 61.8M | trained from scratch |
| **Total** | **211.0M** | of which ~147.6M trainable |

Its checkpoint sweep re-ran the same eval on checkpoint_10000 and
checkpoint_20000 at both integration-step settings.
