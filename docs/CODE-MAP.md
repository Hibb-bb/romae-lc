# Code map

One line per file: what it is and its status. Written 2026-10-19 as step 1
of the clean-up, next to `EXPERIMENTS.md`. The statuses are a proposal, not
a change: nothing has been moved yet.

Statuses: **core** = the encoder and its training, needed by everything;
**paper** = evaluation that makes the encoder paper's tables and figures;
**decoder** = the phase rope / forecast work; **tool** = cluster and
tracking helpers; **archive** = the world-model line (September 26 to
October 1) and rejected ideas; nothing alive imports it except where noted;
**drop** = leftovers nothing references.

Sizes: library 11 files / 4.8k lines; `project/` 85 files / 25k lines;
44 job scripts; 36 test files.

## Top level

| path | what | status |
|---|---|---|
| `romae_lc/` | the library: tokens, rotary ladder, transformer, spectral layer, data | core |
| `project/` | the research code on top of it | see below |
| `tests/` | library tests, 10 files | core |
| `docs/` | design doc, plan, this map, the ledger, the rope and spectral notes, two dated session notes | keep; the two session notes can fold into the ledger |
| `examples/` | the library's original examples (`train_mae.py`, `train_lejepa.py`, `train_lewm.py`, `common.py`, `analyze_pc_periods.py`, `analyze_time_encoding.py`, `jobs/`) from 09-23 | drop (`analyze_pc_periods.py` chose the ladder; its output `results/pc_period/recommendation.json` is worth a copy into docs) |
| `lomb_scargle.py` | a loose Lomb-Scargle script at the root, 10-06; nothing imports it | drop (check first whether `project/fold.py` already has what it does) |
| `runs/`, `results/`, `logs/` (top level) | three September checkpoints, the pc_period analysis, six old job logs | drop after copying the recommendation |
| `README.md` | library README plus a pointer to `project/` | keep; becomes the entry page in step 4 |
| `pyproject.toml`, `uv.lock`, `.python-version`, `LICENSE`, `.gitignore` | packaging | keep |

## `romae_lc/`

| file | what | status |
|---|---|---|
| `model.py` | `RoMAE`, `RoMAEForPreTraining`, the MAE forward with mask tokens, `spectral_frequency`, `phase_rope_layout`, `PhasorFeatures` | core (phasor is dead weight once the decoder paper is written) |
| `rope.py` | axial p-RoPE, nD-RoPE, the dense quantile ladder | core |
| `spectral.py` | the spectral layer (learned periodogram over the tokens, reader, CLS summary) | core |
| `transformer.py` | blocks, attention (softmax and linear) | core (linear attention is a dead branch, see `maew_lin`) |
| `tokenize.py` | one token per observation, (time, wavelength) positions | core |
| `data.py` | light-curve records, the toy simulator, PHOEBE loader, views, collation | core (PHOEBE loader unused by the project) |
| `analysis.py` | reads the rotary time unit off a dataset; used by `examples/analyze_pc_periods.py` only | tool (keep small, or move its ladder-choice function next to `rope.py`) |
| `frames.py` | LeWorldModel frames and actions; nothing in `project/` imports it | archive |
| `lewm.py` | LeWorldModel; imported by `project/train_wm.py` only | archive (goes with `train_wm.py`) |
| `lejepa.py` | LeJEPA; nothing in `project/` imports it | archive |
| `__init__.py` | exports | core (trim with the above) |

## `project/` top-level scripts

| file | lines | what | status |
|---|---|---|---|
| `common.py` | 1480 | the hub: data loading (ZTFxPC and the Chen cache), model build and checkpoint loading, the in-training probe, warm start; imported by 53 files | core (split in step 5) |
| `pretrain_mae.py` | 926 | stage-1 training: masks, targets, spectral aux, cycle queries, phase rope, warm start, JEPA and GRU-ODE switches | core (imports `decoder`, `diagnostics`, `bottleneck`, `gruode`, `jepa`, `tracking`; those stay until these imports are cut) |
| `mae_data.py` | 245 | windows, masks, targets for stage 1 | core |
| `templates.py` | 354 | whole-record Fourier templates, cycle queries | core / decoder |
| `chen_data.py` | 124 | the 800k-star cache: build and load, hash splits, our period rule | core |
| `fold.py` | 257 | fold R2 (6-harmonic weighted fit), fine search, candidates; imported by 11 files | paper |
| `period_probe.py` | 1252 | the period probe: read-outs on frozen latents, Lomb-Scargle next to it, figures | paper |
| `cache_latents.py` | 403 | encode every window once (`latents.pt`); 7 importers | paper |
| `plot_period_examples.py` | 232 | shared figure helpers; 11 importers in `eval/` | paper |
| `baselines.py` | 93 | per-band imputation baselines (constant, linear, GP); used by `forecast`, `train_decoder`, `infer` | archive with them |
| `tracking.py` | 163 | Weights & Biases helper | tool |
| `bench.py` | 224 | loader and encoder throughput benchmark | tool (optional) |
| `sine_test.py` | 281 | the sine test of the decoder | decoder |
| `gruode.py` | 134 | GRU-ODE encoder baseline (triton deregistered at import) | paper (one ablation row); could move under `baselines/` later |
| `jepa.py` | 264 | token-level JEPA objective | paper (ablation rows) |
| `bottleneck.py` | 322 | the bottleneck autoencoder (`--bottleneck`); imported by `common` and `pretrain_mae` | archive (dead result) once the two imports are cut |
| `decoder.py` | 271 | stage-3 decoder on frozen latents; imported by `pretrain_mae` and the archive scripts | archive once `pretrain_mae` stops importing it |
| `diagnostics.py` | 108 | time-shuffle score, surprise along window grids; imported by `pretrain_mae`, `gate`, `inject`, `train_wm` | archive once `pretrain_mae` stops importing it (the shuffle score is printed in the probe line) |
| `phase_probe.py` | 227 | is the window's phase in the latent? (always chance) | paper (negative result) |
| `phase_decoder.py` | 815 | the phase decoder ladder (ceiling, context) | archive (its ceiling result is in the ledger) |
| `pretrain_phase.py` | 338 | the phase bottleneck stage 1 | archive (dead) |
| `period_head.py` | 506 | conv net on a Lomb-Scargle spectrum | archive (rejected by the user) |
| `token_head.py` | 352 | per-token period read-out | archive (dead) |
| `train_predictor.py` | 1659 | stage-2 flow / sequence predictor on cached latents | archive (world model) |
| `train_wm.py` | 765 | joint LeWorldModel training | archive (world model) |
| `train_decoder.py` | 530 | stage-3 decoders | archive (world model) |
| `lookback.py` | 572 | the lookback decoder on the forecast task | archive (world model) |
| `forecast.py` | 646 | end-to-end brightness forecast with predictor + decoder | archive (world model; the forecast scoring code may be reused for the decoder paper) |
| `residual.py` | 730 | residual model (M4) | archive (world model) |
| `energy.py`, `flow.py`, `infer.py` | 248, 129, 606 | energies, flow matching, energy-based inference (M6, M7) | archive (world model; `flow.py` is imported by `decoder.py`) |
| `gate.py` | 531 | the gate on frozen latents | archive (world model) |
| `inject.py`, `anomaly.py` | 348, 414 | injected anomalies (M2) and the anomaly test | archive (world model) |
| `check_nll.py` | 155 | is the flow's NLL real? | archive (world model) |
| `README.md` | 751 | decisions log and command book; mixes alive and archived commands | keep; rewrite in step 4 (the decisions move to the ledger) |

## `project/eval/`

| file | what | status |
|---|---|---|
| `period_table.py` | a period for every star of every split | paper |
| `period_report.py` | the all-star report (main table) | paper |
| `hit_curves.py` | hit-rate curves strict / alias | paper |
| `ls_benchmark.py` | search efficiency against Lomb-Scargle | paper |
| `ls_allstars.py` | the collaborator's astropy search at several budgets | paper |
| `budget_sweep.py` | model-seeded search over its own budget | paper |
| `period_sensitivity.py` | smallest resolvable period change | paper |
| `runtime.py` | ms per star, batched, compiled, quantised | paper |
| `classify_rf.py` | random forest on frozen features | paper |
| `period_chen.py` | periods for the 800k Chen set | paper |
| `period_page.py`, `method_page.py` | the two artifact pages | paper |
| `fold_examples.py`, `scatter_by_class.py`, `scatter_per_class.py`, `slides_figs.py`, `diagrams.py` | talk and paper figures | paper |
| `recon_demo.py` | masked-reconstruction pictures | paper |
| `forecast_demo.py`, `sine_demo.py` | decoder pictures | decoder |
| `starembed_feats.py` | StarEmbed features (never ran; needs x86) | archive unless the comparison is revived |

## `project/jobs/` (Slurm)

| script | status |
|---|---|
| `pretrain_mae.sh` (the chain), `submit_queue.sh`, `queue.txt` (ignored by git), `cache_latents.sh`, `period_probe.sh`, `encoder_ablation.sh`, `chen_cache.sh`, `period_chen.sh`, `sine_test.sh` | core / decoder |
| `period_table.sh`, `period_report.sh`, `period_report_inner.sh`, `eval_bundle.sh`, `ls_benchmark.sh`, `ls_allstars.sh`, `budget_sweep.sh`, `period_sensitivity.sh`, `runtime.sh`, `classify_rf.sh`, `phase_probe.sh`, `rope_ablation.sh` | paper |
| `smoke_wandb.sh`, `bench.sh`, `delta_cpu_starembed.sh`, `starembed_feats.sh` | tool / archive |
| `pretrain_mae_wide.sh`, `pretrain_mae_bn.sh`, `pretrain_phase.sh`, `phase_decoder.sh`, `period_head.sh`, `token_head.sh` | archive |
| `train_wm.sh`, `train_wm_w60.sh`, `train_predictor.sh`, `train_decoder.sh`, `train_lookback.sh`, `forecast.sh`, `gate.sh`, `inject.sh`, `anomaly.sh`, `residual.sh`, `infer.sh`, `pipeline.sh`, `submit_pipeline.sh` | archive (world model) |

## Tests

`tests/` (library, 10 files) stays. In `project/tests/`, the tests follow
their module: `test_project.py`, `test_mae_data.py`, `test_cycles.py`,
`test_encoder_loading.py`, `test_period_probe.py`, `test_period_table.py`,
`test_period_report.py`, `test_period_sensitivity.py`, `test_runtime.py`,
`test_ls_benchmark.py`, `test_budget_sweep.py`, `test_classify_rf.py`,
`test_phase_probe.py`, `test_gruode.py`, `test_jepa.py` stay; the rest
(`test_anomaly`, `test_bottleneck`, `test_forecast`, `test_gate`,
`test_lookback`, `test_period_head`, `test_phase_decoder`, `test_predictor`,
`test_pretrain_phase`, `test_token_head`) move with their modules.

## Machine-specific paths (step 4 turns these into one config)

`/projects/bfrf/data/PC_matches/ZTFxPC` (the default `--data`),
`/work/hdd/bfrf/hibb/chen_cache` and `chen20_ztf_dr24`, `project/runs` and
`project/results` as relative output roots, the `.venv` activation in the
job scripts, and the partition names in `project/jobs/*.sh`.

## Order of removal, when step 3 starts

1. Cut the imports of `decoder`, `diagnostics`, `bottleneck` from
   `pretrain_mae.py` and `common.py` (behind flags nobody uses now).
2. Move the world-model scripts, their jobs and tests to `project/archive/`
   together with `romae_lc/lewm.py`, `frames.py`, `lejepa.py`.
3. Move the rejected period and phase scripts the same way.
4. Drop the top-level leftovers.
5. Run both test suites after each move.
