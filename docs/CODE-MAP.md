# Code map

The main names in every file, for the tree after the clean-up (2026-11-18, branch `cleanup`).
The root `README.md` has the short version: one line per file. This page adds the function and
class names, and a table from the old file names to the new ones.

Run scripts from the repo root as modules: `python -m project.train`, not `python project/train.py`.

## The pipeline in one picture

```
records            project/data.py        load_data -> Data (train / validation / test records)
   |
windows, tokens    project/windows.py     frame_config, frame_loader, TokenSpec, WindowDataset
   |
hidden tokens      project/masks.py       make_mask, phase_band_mask, forecast_tail_mask
(or views)         project/views.py       make_views
   |
model              project/backbone.py    sizes, rotary time ladder
                   project/jepa.py        TokenJEPA, forward_views, sigreg
   |
training           project/train.py       flags (from project/recipes/*.toml), loop, validation
   |
checkpoint         project/checkpoint.py  mae.pt / last.pt: save, load_mae, load_encoder
   |
evaluation         project/eval/          encode -> card_class, card_period, card_phase,
                                          card_forecast -> test_card -> summary
```

## `romae_lc/`: the library

It does not import `project/`. `docs/LIBRARY.md` shows how to use it.

| file | lines | main names |
|---|---|---|
| `data.py` | 685 | `Record`, `simulate`, `load_pc`, `load_phoebe`, `normalize`, `ViewConfig`, `LightCurveDataset`, `collate`, `PC_BANDS`, `CLASSES` |
| `frames.py` | 385 | `FrameConfig`, `sample_frames`, `frame_grid`, `FrameDataset`, `collate_frames` |
| `tokenize.py` | 206 | `tokenize`, `Tokens`, `wavelengths_for` |
| `rope.py` | 538 | `AxialRope`, `SimplexRope`, `BlockRope`, `layout`, `ladder_rows`, `resample_ladder` |
| `transformer.py` | 261 | `TransformerConfig`, `Transformer`, `Attention`, `Block`, `attention_mask` |
| `spectral.py` | 205 | `SpectralLayer`, `soft_bin_target`, `spectral_aux_loss` |
| `model.py` | 828 | `RoMAEBase`, `RoMAE`, `RoMAEForPreTraining`, `RoMAEForClassification`, `MaskedDecoder`, `gen_mask`, `phase_rope_layout` |
| `analysis.py` | 654 | `suggest_time_encoding`, `rotary_ladder`, `timescales_for`, `periodogram`, `time_shuffle_score` |

## `project/`: data to checkpoint

| file | lines | main names |
|---|---|---|
| `data.py` | 152 | `load_data`, `Data`, `add_data_args`, `data_args_from`, `subset`, `class_vocabulary`, `superclass`, `fine_class`, `ANOMALY_CLASSES`, `DATA_ROOT` |
| `windows.py` | 314 | `TokenSpec`, `err_stats`, `err_channel`, `add_frame_args`, `frame_config`, `frame_loader`, `fuse_frames`, `collate_err_frames`, `grid_starts`, `object_windows`, `WindowDataset`, `make_collate` |
| `masks.py` | 200 | `MASK_MODES`, `block_mask`, `make_mask`, `phase_band_mask`, `forecast_tail_mask`, `longest_gap` |
| `views.py` | 65 | `make_views`, `pack`, `take` |
| `backbone.py` | 410 | `PRESETS`, `add_model_args`, `resolve_model_args`, `encoder_config`, `Ladder`, `get_ladder`, `measure_ladder`, `dense_ladder`, `deal_ladder`, `rope_geometry`, `rope_layouts` |
| `jepa.py` | 396 | `TokenJEPA`, `forward_views`, `sigreg` |
| `recipe.py` | 85 | `apply_recipe`, `read_recipe`, `recipe_files`, `recipe_text` |
| `train.py` | 781 | in file order: `RUN_CONTROL`, `check_resumable`, `add_train_args`, `build_parser`, `read_flags` / `build_tokens`, `build_loaders` / `build_model` / `Run`, `window_loss`, `views_loss`, `train_step`, `train_loop` / `validation_losses`, `evaluate`, `baseline` / `save`, `restore`, `finish` / `main` |
| `probes.py` | 267 | `probe`, `probe_metrics`, `linear_probe`, `ridge_r2`, `shuffle_score`, `baseline_probe`, `hand_features`, `embed_records` |
| `tracking.py` | 163 | `Tracker`, `add_wandb_args`, `StepTimer`, `gpu_stats` |
| `utils.py` | 68 | `cosine_schedule`, `seed_all`, `add_device_arg`, `get_device`, `JsonlLog`, `dump_json` |
| `checkpoint.py` | 229 | `mae_state`, `jepa_state`, `save_atomic`, `load_mae`, `load_encoder`, `EncoderMeta`, `FrameEncoder`, `PooledEncoder`, `LatentEncoder` |

### Recipes, `project/recipes/`

| file | what |
|---|---|
| `base.toml` | what every run shares. Not a run |
| `enc_j1.toml` | the first JEPA run. Extends `base.toml` |
| `from_j1.toml` | `enc_j1` continued for 20k steps from its checkpoint. Not a run. Extends `enc_j1.toml` |
| `enc_o2_noadd.toml`, `enc_o2_fc.toml`, `enc_o2_fcbig.toml`, `enc_o2_fcbig_ph.toml` | the option 2 runs. Each extends the one before; the first extends `from_j1.toml` |
| `enc_o3_ph.toml`, `enc_o3_fcph.toml`, `enc_o3_fcph80.toml`, `enc_o3_fcph_pband.toml` | the option 3 runs. `enc_o3_ph` extends `from_j1.toml`, `enc_o3_fcph` extends `enc_o3_ph`, the last two extend `enc_o3_fcph` |
| `enc_le1.toml` | the no-EMA run with views and SIGReg. Extends `base.toml` |

### What a checkpoint holds (`mae.pt`, `last.pt`)

One dict. `kind` (`"jepa"` or absent for a plain masked autoencoder), `state_dict`, `backbone`, `mae`,
`jepa` (how to rebuild the model), `spec`, `frames`, `ladder` (how to make tokens), `classes`, `args`
(every flag), `recipe` (the recipe text), `step`, `metrics`, `epoch`, `elapsed`, `wandb_id`, `aux`
(always empty now). `last.pt` also holds `opt` and `sched`.

`project.checkpoint.load_encoder(path)` gives the frozen encoder and its `EncoderMeta`.

## `project/eval/`: evaluation

| file | lines | main names | command line |
|---|---|---|---|
| `common_eval.py` | 122 | `eval_device`, `run_name`, `window_dataset`, `window_loader`, `spectral_layers`, `fourier_forecast`, `zstats`, `period_hits`, `star_tables`, `write_json`, `git_commit` | no |
| `encode.py` | 231 | `main` | yes |
| `card_class.py` | 218 | `StarProbe`, `star_probe`, `probe_inputs`, `batches`, `scores`, `run` | yes |
| `card_period.py` | 133 | `logp_probe`, `peak_stars`, `peak_periods`, `peak_hits` | yes |
| `card_phase.py` | 247 | `Stars`, `pick_stars`, `pick_windows`, `embed`, `probes`, `run` | yes |
| `card_forecast.py` | 337 | `Mode`, `TEST_CARD`, `CARD_MODES`, `gap_mask`, `phase_gap_mask`, `collect`, `score`, `pick_stars`, `run` | yes |
| `card_demo.py` | 148 | `main` | yes |
| `classify_rf.py` | 189 | `run`, `plot_confusion`, `PARAM_GRID` | yes |
| `test_card.py` | 147 | `CARDS`, `RF_PARAMS`, `make_latents`, `main` | yes |
| `summary.py` | 133 | `headline`, `rows`, `table`, `COLUMNS`, `LEGEND` | yes |
| `style.py` | 29 | `tidy`, `save`, the colours | no |

Extra read-outs in `project/`, not part of the test card:

| file | lines | main names |
|---|---|---|
| `period_probe.py` | 1,177 | `object_table`, `pool_latents`, `fit_ridge`, `fit_mlp`, `fit_bins`, `fit_joint`, `fit_winjoint`, `score`, `gls_power`, `ls_reference`, `failure_tables`, `run` |
| `phase_probe.py` | 264 | `fold_template`, `band_templates`, `reference_phase`, `window_targets`, `fit_ridge`, `fit_mlp`, `run` |
| `fold.py` | 257 | `fold_r2`, `fold_compare`, `refine_period`, `refine_all`, `fold_tables`, `plot_fold` |

Who uses them: `common_eval.star_tables` uses `period_probe.object_table`. `card_phase.py` uses
`phase_probe.reference_phase`. `tests/test_spectral.py` uses `period_probe.gls_power`.

## `project/jobs/`

| file | what |
|---|---|
| `train.sh RECIPE [flags]` | trains a recipe, as linked jobs if `MAX_LINKS` > 1, then calls `test_card.sh` |
| `test_card.sh RUN ...` | `project.eval.test_card` for each run. `RF=1`, `PERIOD_PROBE=1`, `CARD_ARGS` |
| `demo.sh RUN ...` | `project.eval.card_demo` for each run. `HIDE`, `DEMO_ARGS` |
| `after_done.sh FILE sbatch ...` | waits for a file, then submits |
| `in_session.sh CMD ...` | runs a command inside the running Slurm job named `debug`, or here if there is none |

## Tests

| file | what it tests |
|---|---|
| `project/tests/golden.py` | the before/after check of the clean-up. Not a pytest file: `python -m project.tests.golden check` |
| `project/tests/_args.py` | the small training flags the tests share (`MAE_ARGS`) |
| `project/tests/test_common.py` | the error channel, window grids, the ladder, the probes |
| `project/tests/test_masks.py` | the masks |
| `project/tests/test_jepa.py` | `TokenJEPA`, the views, short training runs |
| `project/tests/test_cycles.py` | the phase rope and the frequency estimate of the spectral layer |
| `project/tests/test_recipe.py` | the recipe loader and every recipe file |
| `project/tests/test_encoder_loading.py` | `load_encoder` |
| `project/tests/test_cache_latents.py` | `project.eval.encode` |
| `project/tests/test_eval.py` | the shared evaluation helpers, the forecast modes, the summary table |
| `project/tests/test_classify_rf.py` | the random-forest script |
| `project/tests/test_period_probe.py`, `test_phase_probe.py` | the two extra probes |
| `tests/test_*.py` | the library: data, frames, tokens, rope, transformer, spectral layer, model, analysis |

## Old name -> new name

The tag `pre-cleanup` holds the old tree. There are no shims: the old import paths do not work.

| old | new |
|---|---|
| `project/pretrain_mae.py` | `project/train.py`. `forecast_tail_mask` went to `project/masks.py` |
| `project/common.py` | split into `project/data.py`, `windows.py`, `backbone.py`, `checkpoint.py`, `probes.py`, `utils.py`. `WMMeta` is now `EncoderMeta` |
| `project/mae_data.py` | `project/masks.py` |
| `project/jepa.py`: `make_views`, `_pack`, `_take` | `project/views.py` (`make_views`, `pack`, `take`) |
| `project/jepa.py`: `jepa_state` | `project/checkpoint.py` |
| `project/dyn.py`: `sigreg` | `project/jepa.py`. The rest of `dyn.py` is in `legacy/` |
| `project/diagnostics.py`: `shuffle_score` | `project/probes.py`. The rest is in `legacy/` |
| `project/inject.py`: `fold_template`, `band_templates` | `project/phase_probe.py`. The rest is in `legacy/` |
| `project/cache_latents.py` | `project/eval/encode.py`. `object_windows`, `WindowDataset`, `make_collate` went to `project/windows.py` |
| `project/eval/attn_probe.py` | `project/eval/card_class.py` (class) and `card_period.py` (log period) |
| `project/eval/spectral_peak_probe.py` | `project/eval/card_period.py` |
| `project/eval/token_phase_probe.py` | `project/eval/card_phase.py` |
| `project/eval/jepa_forecast_probe.py` | `project/eval/card_forecast.py` |
| `project/eval/jepa_forecast_demo.py` | `project/eval/card_demo.py` |
| `project/eval/dataset_figs_style.py` | `project/eval/style.py` |
| `project/jobs/pretrain_mae.sh`, `lejepa.sh`, `option2.sh`, `option2_one.sh`, `option3_one.sh` | `project/jobs/train.sh` with a recipe from `project/recipes/` |
| `project/jobs/eval_encoder.sh`, `test_forecast.sh` | `project/jobs/test_card.sh` |
| `project/jobs/rope_demo.sh`, `rope_demo_int.sh` | `project/jobs/demo.sh` |
| `romae_lc/lewm.py`, `romae_lc/lejepa.py`, `examples/`, `lomb_scargle.py` | `legacy/`, same path |
| every other old file of `project/` | `legacy/project/`, same path (see `legacy/README.md`) |

## `legacy/`

Old code, frozen, with the old paths under `legacy/`: 28 modules in `legacy/project/`, 29 in
`legacy/project/eval/`, 62 job scripts, 24 test files, the two old library modules and the examples.
Nothing outside `legacy/` imports it, and it is not expected to run from this branch.
`legacy/README.md` says what is there and how to run it from the tag `pre-cleanup`.
