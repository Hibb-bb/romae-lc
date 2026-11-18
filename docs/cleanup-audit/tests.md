TEST AUDIT for the cleanup branch (/projects/bfrf/hibb/romae-lc-cleanup). Read only. I ran only the core files listed under "Measured run time". I did not edit or move anything.

## 0. Key facts that cut across the core/legacy line (read first)

1. `romae_lc/__init__.py` imports every submodule at load time: analysis, frames, lejepa and lewm (lines 10-60). So any `from romae_lc... import` loads lewm and lejepa. This includes `from romae_lc.model import ...`, which still runs `__init__` first. If you move lewm.py or lejepa.py, every test and every core script breaks until `__init__` is trimmed.
2. Part of `romae_lc/frames.py` is core.
   - `project/mae_data.py:32` imports `FrameConfig, Record`.
   - `project/cache_latents.py:69` imports `FrameConfig, Record, frame_grid`.
   - `project/common.py:33-58` imports `FrameConfig, FrameDataset, collate_frames, frame_grid`.
   So `FrameConfig` and `frame_grid` must stay in the library. `FrameDataset` and `collate_frames` are used by `common.frame_loader` and `diagnostics`.
3. Part of `romae_lc/analysis.py` is core. `project/common.py` `measure_wavelengths` (around line 563) calls `suggest_time_encoding`, and `measure_ladder` (around line 590) calls `timescales_for`. Through `get_ladder` this runs in every pretrain_mae run (`pretrain_mae.py:467`). Also `project/diagnostics.py:18` uses `time_shuffle_score`, and `pretrain_mae.py:92` imports `diagnostics.shuffle_score` at the top. Do not move analysis.py wholesale.
4. `project/common.py:33-58` imports `ARPredictor, LeWorldModel, lewm_mlp` at the top. They are used by `build_wm`/`load_wm`/`use_fused_encode` (lines 789, 926-945, 1099-1129). common.py must be split before lewm can go.
5. Core code imports legacy project modules.
   - `pretrain_mae.py:55-56` imports `project.bottleneck` and `project.gruode` at the top.
   - `pretrain_mae.py` imports these lazily: phase_decoder (623, 722), dyn (631, 634, 882), templates (696-698, 811, 842), decoder (724).
   - `project/jepa.py:371` `forward_views` does `from project.dyn import sigreg`. The le1 path (`--jepa-no-ema`) depends on project/dyn.py. Move `sigreg` (dyn.py:43) into jepa.py before dyn.py goes to legacy.
   - `project/period_probe.py:68` imports `project.fold` at the top. Line 282 lazily imports `train_predictor.crop_last, load_predictor`.
   - `project/phase_probe.py:35` imports `project.inject.band_templates` at the top. `eval/token_phase_probe.py:34` needs `phase_probe.reference_phase`, so inject is pulled into core eval.
   - `eval/token_phase_probe.py:207` lazily imports `project.dyn.DynHead`. It only runs when the checkpoint has `aux.dyn`.
   - `project/common.py:1075-1083` `load_mae` lazily imports bottleneck, jepa and gruode by checkpoint kind.
6. Shared test fixture across the line: `project/tests/test_ls_benchmark.py:11` `MAE_ARGS`. `eval/ls_benchmark` is not in the current pipeline. Seven files import it:
   - core: test_jepa.py:11, test_classify_rf.py:9
   - legacy: test_gruode.py:14, test_period_report.py:11, test_period_table.py:10, test_runtime.py:10, test_budget_sweep.py:10
   Fix: move `MAE_ARGS` into `project/tests/conftest.py` or a `project/tests/_args.py`, then import it from there. Other files (test_mae_data, test_period_probe, test_phase_probe, test_encoder_loading, ...) define their own local `MAE_ARGS`.
7. pytest config: `pyproject.toml [tool.pytest.ini_options] testpaths = ["tests"]`. A bare `pytest` runs only the library tests. project/tests must be named explicitly.

## 1. Library tests (tests/)

| file | tests | covers | verdict |
|---|---|---|---|
| test_model.py | 14 | RoMAE, RoMAEForPreTraining, RoMAEForClassification, gen_mask, MaskedDecoder, decoder_layout, per-layer layouts | CORE |
| test_rope.py | 18 | AxialRope/BlockRope/SimplexRope, layout, ladder_rows, collapse_layout | CORE |
| test_spectral.py | 3 | SpectralLayer, spectral_aux_loss, MAE+spectral round trip. Line 31 imports `project.period_probe.gls_power` (library test depends on project/) | CORE (note the project import) |
| test_tokenize.py | 6 | tokenize, Tokens, wavelengths_for | CORE |
| test_data.py | 14 | simulate, normalize, make_views/sample_view, LightCurveDataset, collate, load_pc, load_phoebe | CORE (views/phoebe parts are old, but cheap) |
| test_transformer.py | 9 | Transformer, TransformerConfig, attention_mask, BlockRope | CORE |
| test_analysis.py | 19 | suggest_time_encoding, periodogram, spectral_support, rotary_*, time_shuffle_score, plot_time_encoding | CORE because the ladder needs it (see 0.3). The plot/kernel tests are diagnostic. |
| test_frames.py | 8 | FrameConfig validation, sample_frames, frame_grid, FrameDataset, collate_frames | Keep test_frame_config_validation and test_frame_grid (core). The rest is world-model frames, so legacy, but cheap. |
| test_lejepa.py | 8 | romae_lc.lejepa LeJEPA, SlicedEppsPulley, mlp | LEGACY |
| test_lewm.py | 23 | romae_lc.lewm LeWorldModel, ARPredictor, ActionEncoder, straightness. Also uses lejepa.EppsPulley | LEGACY |

## 2. Project tests (project/tests/)

CORE or mostly core:
- **test_jepa.py (8)**
  - Core: test_jepa_trains_saves_and_loads (`--jepa --mask-mode blockplus`, cache_latents, load_encoder), test_jepa_hybrid_and_warm_start (`--jepa-recon-weight`, `--jepa-init`), test_jepa_projection (`--jepa-proj`, `--jepa-proj-hidden`), test_jepa_no_ema_views (le1: `--jepa-no-ema --sig-slices --spectral --spectral-aux --phase-rope --view-keep --local-span`, `forward_views` under bf16 autocast).
  - Legacy, they need project/templates.py: test_smooth_targets_mae_and_jepa (`--target mix/template`), test_jepa_clean_target (`--jepa-clean-target --target template`), test_query_tokens_mae_and_clean_jepa (`--query-extra`), test_record_templates_and_training (`templates.fit_record_templates`, `--template-fit record`). All ICML runs use `target=obs` and `query_extra=0`.
- **test_mae_data.py (8)**
  - Core: block_mask, make_mask, mix, MultiWindowDataset, error_weights, absolute time features, blockplus.
  - test_pretrain_with_stretches_and_window_range (`--mask-mode mix --window-range`, then cache_latents and period_probe) is core-adjacent.
  - Legacy: test_stretches_refused_for_the_bottleneck (bottleneck).
- **test_classify_rf.py (1)**: pretrain, cache_latents (train and test splits), eval/classify_rf. CORE.
- **test_period_probe.py (5)**: period_probe end to end, gls_power, lomb-scargle, recovery and score. CORE. Note: the end-to-end test runs fold (`fold_r2.png`, `res["fold"]`), and period_probe.py:68 imports fold at the top. eval_encoder.sh uses `--no-fold`. Either keep fold.py in core or make the import lazy and test with `--no-fold`.
- **test_phase_probe.py (1)**: phase_probe.phase_error, window_targets, fit_ridge, to_phase, main. It is core-adjacent because token_phase_probe uses `reference_phase`. It drags in inject (see 0.5).
- **test_cycles.py (6)**
  - Core: test_phase_rope_runs_and_the_oracle_frequency_changes_the_output (phase_rope), test_frequency_estimators_agree_on_a_clean_peak (`spectral_frequency` soft/peak/local; runs use `phase_freq=local`), test_warm_start_copies_matching_tensors_only (`common.warm_start`, used by `--jepa-init`).
  - Legacy: the phasor tests and test_queries_in_cycles (it imports `project.templates.sample_queries_cycles` at line 7, at the top).
- **test_encoder_loading.py (4)**
  - Core: test_load_encoder_mae.
  - Legacy: test_load_wm_accepts_mae, test_load_encoder_wm, test_train_decoder_on_mae (train_wm, train_decoder).
- **test_project.py (22)**: mostly LEGACY. Its top-level imports pull in lewm, baselines, flow, decoder, diagnostics, energy, inject and residual (lines 12-50). After the move, the file will fail to import.
  - Core common.py tests to rescue into a new file (e.g. project/tests/test_common.py): test_err_channel_and_collate, test_grid_item_and_starts (grid_item, grid_starts, which cache_latents uses), test_data_args_from, test_dense_ladder_dealing_and_geometry, test_dense_backbone_and_flat_layout, test_probe_metrics_and_baseline.
  - test_batch_stats_context uses the LeWorldModel `model` fixture (fixtures around lines 87-100), so it is legacy as written.
  - test_pretrain_mae_then_init_backbone and test_freeze_backbone pair pretrain_mae with train_wm, so they are legacy.
- **test_predictor.py (13)**: LEGACY (train_predictor, flow). But 3 of its tests are the only deep cache_latents tests: test_cache_layout_and_starts, test_cache_realisations, test_cache_max_objects_and_seed. They check `cache_latents.object_windows` and the dtypes and meta of the cache. Move them into a core test_cache_latents.py, because the file imports train_predictor at the top (line 19).

LEGACY (move to legacy/tests):
- test_anomaly (anomaly, train_predictor)
- test_bottleneck (bottleneck, decoder, train_wm, FrameDataset)
- test_budget_sweep (eval/budget_sweep)
- test_forecast (forecast, lookback, train_decoder, train_predictor)
- test_gate (gate)
- test_gruode (gruode)
- test_lookback (lookback)
- test_ls_benchmark (eval/ls_benchmark; but move MAE_ARGS out first)
- test_period_head (period_head)
- test_period_report (eval/period_report)
- test_period_sensitivity (eval/period_sensitivity)
- test_period_table (eval/period_table)
- test_phase_decoder (phase_decoder, train_predictor)
- test_pretrain_phase (pretrain_phase, phase_decoder)
- test_runtime (eval/runtime)
- test_token_head (token_head)

## 3. Gaps: current pipeline code with no test
- `--forecast-tail` and `pretrain_mae.forecast_tail_mask` (pretrain_mae.py:408, used at 800-806). Used by o2_fc, o2_fcbig, o2_fcbig_ph, o3_fcph, o3_fcph80 and o3_fcph_pband.
- `--mask-mode phaseband` and `mae_data.phase_band_mask` (`--phase-band`). Used by o3_fcph_pband. MASK_MODES is at mae_data.py:35.
- `--phase-freq local` in a pretrain run. Only the estimator itself is tested, in test_cycles.
- `--band-drop` and `--views` (le1).
- eval/spectral_peak_probe.py, eval/token_phase_probe.py, eval/attn_probe.py, eval/jepa_forecast_probe.py, eval/jepa_forecast_demo.py have no tests.

## 4. Core test set to keep running after the clean-up
Library: tests/test_model.py, test_rope.py, test_spectral.py, test_tokenize.py, test_data.py, test_transformer.py, test_analysis.py, test_frames.py (the frame_grid and FrameConfig tests).
Project:
- test_jepa.py, keeping the 4 core tests
- test_mae_data.py, minus the bottleneck test
- test_classify_rf.py
- test_period_probe.py
- test_phase_probe.py, if token_phase_probe stays
- test_cycles.py, keeping the 3 core tests
- test_encoder_loading.py::test_load_encoder_mae
- a new test_common.py made from the 6 core test_project tests
- a new test_cache_latents.py made from the 3 cache tests in test_predictor.py

## 5. Measured run time (CPU, gh-login03, run from the cleanup checkout)
- Library, 8 core files: 106 passed in 52.8 s. Slowest: test_load_pc_reads_arrow_columns 13.1 s, test_suggest_time_encoding_quantile_ladder 9.9 s, test_spectrum_matches_lomb_scargle_peak 8.6 s.
- test_jepa.py + test_mae_data.py: 16 passed in 258 s.
  - test_jepa_no_ema_views 116.6 s (by far the slowest)
  - test_pretrain_with_stretches_and_window_range 52.0 s
  - test_jepa_trains_saves_and_loads 21.0 s
  - test_jepa_hybrid_and_warm_start 13.5 s
  - test_jepa_projection 7.4 s
  - the 4 legacy template tests together about 42 s
- test_classify_rf + test_period_probe + test_phase_probe + test_cycles + test_load_encoder_mae: 14 passed in 120 s. classify_rf 51 s; period_probe end to end 45 s + 16 s setup; phase_probe 16 s.
- Not run: the rescued test_project and test_predictor tests. They need a small pretrain plus a cache, so expect about 30-60 s more.
- **Estimate for the core set: about 6.5-7.5 minutes on CPU**:
  - library about 1 min
  - test_jepa core about 2.6 min (le1 test alone about 2 min)
  - test_mae_data about 1 min
  - eval group about 2 min
  - rescued tests about 0.5-1 min
  Dropping the 4 template tests saves about 42 s. To make it faster, cut `--steps` or `--sig-slices` in test_jepa_no_ema_views, or share one pretrained checkpoint through a session-scoped fixture. Many tests rerun `pretrain_mae.main` with `--steps 2`.

## 6. romae_lc/__init__.py exports
- CORE:
  - model: MAEOutput, MaskedDecoder, RoMAE, RoMAEForClassification, RoMAEForPreTraining, decoder_layout, gen_mask
  - rope: AxialRope, BlockRope, SimplexRope, collapse_layout, ladder_rows, layout, min_axial_dim, min_simplex_scales, resample_ladder
  - tokenize: BAND_WAVELENGTHS_NM, Tokens, tokenize, wavelengths_for
  - transformer: SIZES, Transformer, TransformerConfig, attention_mask
  - data: CLASSES, DEFAULT_SURVEYS, PC_BANDS, PC_SUPERCLASSES, PHOEBE_BANDS, PHOEBE_MORPHOLOGIES, LightCurveDataset, Record, SimConfig, SurveyConfig, ViewConfig, collate, load_pc, load_phoebe, make_views, normalize, sample_view, simulate, time_series
- CORE, used by project code (keep, or move into project/):
  - frames: FrameConfig, frame_grid (also FrameDataset, collate_frames while common.frame_loader and diagnostics use them)
  - analysis: suggest_time_encoding, timescales_for, time_shuffle_score (via diagnostics.shuffle_score)
- Diagnostic only, not used by core code: analysis TimeEncodingReport, nd_rope_theta_bound, periodogram, plot_time_encoding, rotary_kernel, rotary_ladder, rotary_wavelengths, sampling_stats, spectral_support, wave_vectors; frames sample_frames.
- LEGACY:
  - lejepa: LeJEPA, LeJEPAOutput, SlicedEppsPulley, mlp. No core project file imports romae_lc.lejepa; core uses `project.dyn.sigreg` instead.
  - lewm: ActionEncoder, ARPredictor, LeWMOutput, LeWorldModel, lewm_mlp, straightness. common.py still imports ARPredictor, LeWorldModel and lewm_mlp at the top.
- Not exported: `romae_lc/spectral.py` (SpectralLayer, spectral_aux_loss) is core but missing from `__init__`. It is reached via `romae_lc.model` and `romae_lc.spectral`.
- The module docstring (lines 1-8) advertises LeJEPA and LeWorldModel. Update it when they move.

Safe move order:
1. Move `sigreg` from dyn.py into jepa.py.
2. Split common.py so the WM loaders leave it.
3. Make the bottleneck and gruode imports in pretrain_mae.py lazy, or remove them.
4. Make the fold and inject imports lazy in period_probe.py and phase_probe.py.
5. Trim `romae_lc/__init__.py` (drop lejepa and lewm).
6. Move `MAE_ARGS` to conftest.py.
7. Then move the legacy tests.

Files: /projects/bfrf/hibb/romae-lc-cleanup/romae_lc/__init__.py, /projects/bfrf/hibb/romae-lc-cleanup/project/common.py, /projects/bfrf/hibb/romae-lc-cleanup/project/jepa.py, /projects/bfrf/hibb/romae-lc-cleanup/project/pretrain_mae.py, /projects/bfrf/hibb/romae-lc-cleanup/project/tests/test_ls_benchmark.py, /projects/bfrf/hibb/romae-lc-cleanup/project/tests/test_jepa.py, /projects/bfrf/hibb/romae-lc-cleanup/project/tests/test_project.py, /projects/bfrf/hibb/romae-lc-cleanup/project/tests/test_predictor.py, /projects/bfrf/hibb/romae-lc-cleanup/pyproject.toml