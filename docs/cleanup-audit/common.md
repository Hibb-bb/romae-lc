AUDIT: project/common.py, project/cache_latents.py, project/mae_data.py (branch "cleanup", /projects/bfrf/hibb/romae-lc-cleanup). Read only. I changed nothing.

Method: an AST scan of every `from project.common/cache_latents/mae_data import ...` in the repo, plus an internal call graph of common.py. Helper scripts are in /tmp/claude-91745/-projects-bfrf-hibb-romae-lc/f846f30c-9ae3-4667-b5f6-cb47d35ed7de/scratchpad/uses.py and /tmp/claude-91745/-projects-bfrf-hibb-romae-lc/f846f30c-9ae3-4667-b5f6-cb47d35ed7de/scratchpad/graph.py.

"CUR" means a current pipeline file uses it, directly or through a kept function. "LEG" means only legacy code and tests use it. "DEAD" means nothing uses it.

== 1. What the ICML checkpoints need (load_mae / load_encoder / kind) ==

I opened each checkpoint with torch.load(mmap=True) and checked its keys:
- enc_j1, enc_o2_noadd, enc_o2_fc, enc_o2_fcbig, enc_o2_fcbig_ph, enc_o3_fcph, enc_o3_ph, enc_o3_fcph80 and enc_o3_fcph_pband all have kind == "jepa".
- In all of them, aux == {}. So there are no dyn, fold_dec or phase_head weights.
- Their keys are: mae = {decoder, mask_ratio, target_channels}; jepa = {ema, ema_end, loss, use_target, recon_weight, clean_target, proj_dim, proj_hidden}, plus {phase_rope, phase_freq} in the _ph and o3 runs.
- enc_le1 has no mae.pt yet. Its args.json has --jepa --jepa-no-ema, so jepa_state will save it with kind "jepa" too.
- In args.json, every ICML run has dyn=0, bottleneck=False and gruode=None.

The load path every eval uses:
- load_encoder (common.py:1189) calls _read_ckpt (1037), then checkpoint_kind (1045).
- checkpoint_kind returns "mae" when "mae" is in the ckpt or kind is in (bottleneck, jepa, gruode).
- That leads to load_mae (1066). Its kind=="jepa" branch (1078-1081) does `from project.jepa import TokenJEPA` and calls TokenJEPA.from_checkpoint (jepa.py:169).
- Then LatentEncoder(model, "mae") (1136), which uses PooledEncoder (742) and fused_features (722).
- Meta comes from _meta_from (1054): TokenSpec, FrameConfig, Ladder, classes, args, step.
- load_wm (1099) is never reached for ICML checkpoints.

Branches the ICML checkpoints need:
- kind "jepa": YES.
- Plain MAE (no kind; else branch 1086-1088, romae_lc only): NO. Keep it only if the plain-MAE baseline stays. It needs no legacy import.
- "bottleneck" (1074-1077, imports project.bottleneck): NO.
- "gruode" (1082-1085, imports project.gruode): NO.
- "wm" (load_wm, the LatentEncoder "wm" kind, checkpoint_kind returning "wm"): NO.

Suggested simplification:
- checkpoint_kind can go. load_encoder can call load_mae directly.
- LatentEncoder keeps only the PooledEncoder path. Drop KINDS, the "wm" branch at 1161-1163, and `history`.
- Keep `kind` as a fixed string. cache_latents.py:296 and :361 print and store enc.kind.

== 2. project/common.py, function by function ==

-- data --
- DATA_ROOT L62. Default for --data. CUR (add_data_args). Legacy: slides_figs, starembed_feats.
- PROJECT_DIR L63. DEAD (no users).
- LABEL_FIELD L64. The "class_str" label column. CUR (load_data, class_vocabulary).
- Data L71. Dataclass of splits, classes and band wavelengths. CUR (returned by load_data).
- add_data_args L83. The --data, --classes, --max-rows, --min-points, --norm-days, --n-sim and --seed flags. CUR: pretrain_mae.py:431.
- class_vocabulary L109. Sorted class_str over all splits. CUR (load_data).
- superclass L121. Reads meta superclass_str. CUR: cache_latents.py:354, period_probe, phase_probe.py:34, token_phase_probe.py:32, probe_metrics.
- fine_class L125. Reads meta class_str. CUR: period_probe.py:229, probe_metrics.
- ANOMALY_CLASSES L134. Classes kept out of the class probe. CUR: attn_probe.py:22, classify_rf.py:26, probe_metrics.
- load_data L137. Loads and normalises splits. CUR: pretrain_mae:453, cache_latents:289, period_probe, phase_probe, spectral_peak_probe:6, token_phase_probe:32, attn_probe:22, classify_rf:26, jepa_forecast_probe:33, jepa_forecast_demo:19.
  - The "sim" branch (139-153) only serves tests and --data sim.
  - The Chen branch (154-157) is LEG. See section 5.
- DATA_KEYS L179 and data_args_from L182. Rebuild the data args saved in a checkpoint. CUR: cache_latents:285 and every eval listed above.
- subset L193. Seeded subset of records. CUR: pretrain_mae:510, :933, :934; diagnostics.shuffle_score (pretrain_mae:974).

-- error channel and tokens --
- err_stats L204. Mean and std of log err. CUR: pretrain_mae:469.
- err_channel L210. Builds the 2-channel token values. CUR (through TokenSpec).
- collate_err_frames L222. collate_frames plus err_channel. CUR (through TokenSpec.collate).
- TokenSpec L232. Tokenizer settings and err stats, saved in every checkpoint. CUR: pretrain_mae:464/470, cache_latents:198, _meta_from.

-- frames --
- add_frame_args L268. The --window, --advance, --n-frames, --min-tokens and --max-tokens flags. CUR: pretrain_mae:432.
- frame_config L291. Builds the FrameConfig. CUR: pretrain_mae:459.
- frame_loader L302. DataLoader over FrameDataset. CUR: pretrain_mae:483, :511.
- subsample_frame L334. Seeded point cap. CUR: cache_latents:102 and :114. Legacy: gate.py.
- grid_starts L342. Window starts of frame_grid. CUR: cache_latents:93 and :149. Legacy: diagnostics, infer, inject.
- grid_item L353. One frame_grid item for collate. LEG: diagnostics.grid_surprise:64, recon_demo, infer, tests. pretrain_mae only calls diagnostics.shuffle_score, and that does not use it.

-- time ladder --
All of these are CUR. pretrain_mae calls get_ladder and rope_geometry at :467 and rope_layouts at :536-579, and _meta_from rebuilds Ladder.
- Ladder L376.
- RopeGeometry L422.
- rope_geometry L439.
- DEALS L450.
- deal_ladder L453.
- _nest L484.
- add_ladder_args L495. CUR: pretrain_mae:433.
- measure_wavelengths L547.
- measure_ladder L584.
- dense_ladder L599.
- check_ladder L621.
- get_ladder L634.
- _n_active L683.
- time_block_dim L542 is DEAD.

-- fused encode and encoder wrappers --
- fuse_frames L690. Stacks T token batches. CUR: pretrain_mae:771, :1067. Legacy: bottleneck.fuse_tokens.
- fused_encode L712. LeWorldModel.encode replacement. LEG (only use_fused_encode).
- fused_features L722. [B,T,D] features in one call. CUR (FrameEncoder, LatentEncoder).
- FrameEncoder L729. Adds encode(frames) to a backbone for probe(). CUR: pretrain_mae:605.
- PooledEncoder L742. CLS or attn-pool vector of an MAE or JEPA model. CUR: pretrain_mae:605, LatentEncoder.
- batch_stats L771. BatchNorm on batch stats. LEG: train_wm, tests.
- use_fused_encode L789. LEG: build_world_model, load_wm.

-- model args --
- PRESETS L804. Size presets. CUR.
- add_model_args L812. CUR: pretrain_mae:434.
  - The world-model flags at L842-852 are LEG: --pred-depth, --pred-heads, --pred-dim-head, --pred-mlp, --pred-dropout, --proj-hidden, --history, --lamb, --n-slices, --knots, --t-max.
  - Neither pretrain_mae.py nor jepa.py reads args.pred_*, args.proj_hidden, args.history, args.lamb, args.n_slices, args.knots or args.t_max. lejepa.sh uses --sig-slices, which pretrain_mae defines itself.
  - Loading never reads them either, because load_mae uses ckpt["backbone"/"mae"/"jepa"]. So dropping them is safe.
- MODEL_KEYS L857. LEG (train_wm).
- resolve_model_args L870. Fills width, heads and depth from --size. CUR: pretrain_mae:438.
- encoder_config L878. CUR: pretrain_mae:535-576.
- rope_layouts L888. CUR: pretrain_mae:536-579.
- build_backbone L905. LEG: bench, train_wm, tests.
- flat_layout L914. LEG: bottleneck, lookback, token_head, train_decoder.
- build_world_model L920. LEG: bench, train_wm.
- n_params L955. CUR: pretrain_mae:607.

-- checkpoints --
- WMMeta L963. Checkpoint meta. CUR, as the return type of load_encoder. Rename it to EncoderMeta.
- wm_state L976. LEG: train_wm, tests.
- warm_start L997. Copies tensors whose names and shapes match. CUR: pretrain_mae:591 (--init). No ICML run used --init.
- mae_state L1010. Plain-MAE checkpoint dict. CUR only for non-JEPA runs (pretrain_mae:610 state_fn). ICML runs use jepa_state. Legacy: gruode.py:129.
- _read_ckpt L1037. CUR. Legacy: dataset_summary.
- checkpoint_kind L1045. CUR today, but it can be removed. See section 1.
- _meta_from L1054. CUR.
- load_mae L1066. CUR. Keep the jepa branch and the plain branch.
- save_atomic L1092. CUR: pretrain_mae:942.
- load_wm L1099. LEG: infer, inject, residual, train_decoder, tests.
- LatentEncoder L1136. CUR. Keep only the "mae" kind.
- load_encoder L1189. CUR: cache_latents:277, period_probe:1036, phase_probe, spectral_peak_probe, token_phase_probe, jepa_forecast_probe, jepa_forecast_demo.

-- probes --
pretrain_mae.evaluate uses all of these: probe at :963, describe_probe at :1004 and :1035, baseline_probe at :1031.
- _standardize L1202. CUR.
- linear_probe L1207. CUR.
- macro_f1 L1237. CUR.
- balanced_accuracy L1253. CUR.
- ridge_r2 L1261. CUR.
- embed_records L1272. CUR.
- probe_metrics L1287. CUR.
- probe L1332. CUR.
- HEADLINE L1346. CUR. Also used by period_probe:157.
- describe_by_class L1349. CUR.
- describe_probe L1360. CUR.
- hand_features L1369. CUR. Also used by period_probe:239.
- baseline_probe L1412. CUR.

-- misc --
- cosine_schedule L1424. CUR: pretrain_mae:907.
- seed_all L1435. CUR: pretrain_mae:449, period_probe:1026.
- add_device_arg L1441 and get_device L1447. CUR: pretrain_mae, cache_latents, period_probe, phase_probe.
- to_device L1451. Returns (frames, actions). LEG: bench, residual, train_decoder, train_wm.
- JsonlLog L1456. CUR: pretrain_mae:932, period_probe:1030.
- _json_default L1471 and dump_json L1479. CUR: cache_latents, pretrain_mae, period_probe, phase_probe, classify_rf.

== 3. project/cache_latents.py ==

- object_windows L85. The grid windows of one record after the cap.
  - CUR (WindowDataset).
  - Also used by legacy anomaly, forecast, lookback, eval/forecast_demo, eval/period_sensitivity, eval/runtime and tests/test_predictor.
- realise L106. A noise realisation of a window. LEG in practice.
  - It only runs with --realisations > 1.
  - Only train_predictor.py and tests/test_predictor.py read z_alt.
  - eval_encoder.sh does not pass the flag.
- WindowDataset L117. A flat index over every window.
  - CUR: cache_latents main, spectral_peak_probe:7, token_phase_probe:33, jepa_forecast_probe:34, jepa_forecast_demo:20.
  - Legacy: period_chen, state_decode_probe.
  - token_phase_probe:203 passes n_real=1 and real_drop=0.5 by position. Keep that signature, or fix that call when the n_real/real_drop parameters are dropped.
- make_collate L198. Tokenises single windows. CUR, same users as WindowDataset.
- _pick L216. Same rng as subset. CUR (main).
- main L223. The script run by eval_encoder.sh:6-7.
  - --realisations, --real-drop, z_alt and the "real" column are LEG. Only the predictor uses them.
  - --window is not used by the ICML jobs. Keep it, because it is cheap.
- The module docstring (L1-56) describes stage 2 and train_predictor. It needs a rewrite.

== 4. project/mae_data.py ==

- MASK_MODES L35. CUR: pretrain_mae:148.
- block_mask L38. CUR through make_mask. ICML runs use --mask-mode blockplus, and that calls block_mask at L120.
- make_mask L95. CUR: pretrain_mae:796. Legacy: forecast_demo, recon_demo, sine_demo, sine_test.
- longest_gap L142. Tests only (test_mae_data). Not CUR.
- MultiWindowDataset L158. Used only with --window-range (pretrain_mae:495-496). No ICML run sets window_range. Optional to keep.
- phase_band_mask L248. CUR: pretrain_mae:785, through a lazy import, for enc_o3_fcph_pband (--mask-mode phaseband).
- Its imports are romae_lc only. No legacy imports.

== 5. Legacy-module imports in these three files ==

1. common.py:154 `from project.chen_data import is_cache, load_chen` (inside load_data).
   - It runs on every non-sim load.
   - Not needed for ICML checkpoints. Their data is /projects/bfrf/data/PC_matches/ZTFxPC, and the PLAN says Chen labels and periods are not used for training or evaluation.
   - It can be dropped. Remove the branch at 154-157.
   - Caveat: the PLAN roadmap step 5 (Chen 800k pretraining) would need it back. Decide whether step 5 is still in scope. If it is, keep chen_data in the core.
2. common.py:1075 `from project.bottleneck import BottleneckAE`. Lazy, in load_mae. Drop it. No ICML checkpoint has kind "bottleneck".
3. common.py:1079 `from project.jepa import TokenJEPA`. KEEP. Every ICML checkpoint needs it. It can become a top-level import, because jepa.py does not import project.common.
4. common.py:1083 `from project.gruode import GRUODEForPreTraining`. Lazy. Drop it.
5. romae_lc names imported only for legacy code at common.py:33-60 can be dropped: ARPredictor, LeWorldModel, lewm_mlp, collapse_layout, and RoMAE (used only by build_backbone, flat_layout, load_wm and build_world_model).
   - Still needed: FrameDataset, FrameConfig, Tokens, tokenize, collate_frames, frame_grid, load_pc, normalize, simulate, CLASSES, DEFAULT_SURVEYS, PC_BANDS, wavelengths_for, timescales_for, time_series, suggest_time_encoding, resample_ladder, RoMAEForPreTraining, layout, model.pool_tokens.
6. cache_latents.py and mae_data.py import no legacy modules.
7. None of the three files imports train_predictor, inject, phase_decoder, decoder, templates, dyn, lookback or flow.

The current pipeline files themselves do import legacy modules. These must be fixed before the move, or ICML loading and training break:
- jepa.py:371 `from project.dyn import sigreg` in forward_views. This is required by enc_le1 (--jepa-no-ema).
  - pretrain_mae.py:882 imports the same thing (the --cls-inv / --cls-sig path).
  - Move sigreg (dyn.py:43-58, pure torch) into jepa.py before dyn.py goes to legacy.
- pretrain_mae.py:55-56 import project.bottleneck (BottleneckAE, bottleneck_state, fuse_tokens) and project.gruode at top level. Both are used at :610 and :769-770. Remove those branches.
- pretrain_mae.py lazy imports of phase_decoder (:623, :722), dyn (:631, :634), templates (:696, :698) and decoder (:724). These run only with --fold-loss, --dyn or the template flags, and all are off in the ICML runs.
- phase_probe.py:35 `from project.inject import band_templates` is a top-level import. It is used at :43 in reference_phase, and token_phase_probe.py:34 imports reference_phase. So inject.py cannot move unless band_templates moves with phase_probe.
- period_probe.py:282 lazy `from project.train_predictor import crop_last, load_predictor`, inside seq_features. That path is only for a predictor. It is wrapped in try/except, but it is better to delete it.
- token_phase_probe.py:207 lazy `from project.dyn import DynHead`. It only runs when ckpt["aux"]["dyn"] exists, and it is empty for all ICML checkpoints. Drop it.
- diagnostics.py (pretrain_mae:92 shuffle_score) imports only project.common. It is fine, but it brings grid_item along. Move shuffle_score next to the probes, and send grid_surprise and surprise_summary to legacy.

== 6. Proposed split (what remains after dropping the LEG and DEAD items) ==

- project/data.py (records):
  - DATA_ROOT, LABEL_FIELD, Data, add_data_args, class_vocabulary, superclass, fine_class, ANOMALY_CLASSES, HEADLINE, load_data (sim and ZTFxPC; Chen only if kept), DATA_KEYS, data_args_from, subset.
- project/windows.py (records to windows to tokens):
  - err_stats, err_channel, collate_err_frames, TokenSpec, add_frame_args, frame_config, frame_loader, subsample_frame, grid_starts, fuse_frames.
  - From cache_latents: object_windows, WindowDataset (without realise and n_real, or keep them for compatibility), make_collate.
  - From mae_data, optional: MultiWindowDataset.
  - With this, the evals stop importing a script module for WindowDataset.
- project/masks.py (rename of mae_data.py): MASK_MODES, block_mask, make_mask, phase_band_mask. longest_gap goes with its test, or is dropped.
- project/backbone.py (encoder size and rotary ladder):
  - PRESETS, add_model_args (encoder flags only: --size, --width, --depth, --heads, --mlp-ratio, --p-rope, --time-frac, --attention, --no-err-channel), resolve_model_args, encoder_config, n_params.
  - Ladder, RopeGeometry, rope_geometry, DEALS, deal_ladder, _nest, add_ladder_args, measure_wavelengths, measure_ladder, dense_ladder, check_ladder, get_ladder, _n_active, rope_layouts.
- project/checkpoint.py (checkpoint I/O):
  - WMMeta renamed to EncoderMeta, mae_state, warm_start, save_atomic, _read_ckpt, _meta_from, load_mae (jepa and plain branches only), fused_features, PooledEncoder, FrameEncoder, LatentEncoder (mae kind only), load_encoder.
  - jepa_state could also move here from jepa.py:446 so that all save and load code sits together.
- project/probes.py (frozen-feature probes during training):
  - _standardize, linear_probe, macro_f1, balanced_accuracy, ridge_r2, embed_records, probe_metrics, probe, describe_by_class, describe_probe, hand_features, baseline_probe, and diagnostics.shuffle_score.
- project/utils.py:
  - cosine_schedule, seed_all, add_device_arg, get_device, JsonlLog, _json_default, dump_json.
- project/cache_latents.py stays a script with only main and _pick. Drop --realisations and --real-drop, or keep them hidden.

Goes to legacy:
- grid_item, fused_encode, batch_stats, use_fused_encode, MODEL_KEYS, build_backbone, flat_layout, build_world_model, wm_state, load_wm, checkpoint_kind, to_device, realise, the world-model flags of add_model_args, and the Chen branch of load_data.

Dead, delete:
- PROJECT_DIR, time_block_dim.

== 7. Transition notes ==

- Legacy files import heavily from project.common. For example, add_device_arg and load_data are imported by about 30 files, and load_wm, flat_layout and to_device are legacy-only. The simplest move is a verbatim copy of today's common.py into legacy/ (for example legacy/common.py), with the legacy imports rewritten to point at it. Then the new modules hold only kept code. Legacy is frozen and is not expected to run.
- Optionally, keep project/common.py as a thin re-export of the new modules for one step, so the eval and job scripts keep working while their imports are updated.
- Tests that depend on legacy pieces need trimming or moving:
  - project/tests/test_encoder_loading.py (load_wm, wm_state, checkpoint_kind, LatentEncoder "wm").
  - project/tests/test_project.py (build_backbone at :836, load_mae/load_wm at :610 and :667, use_fused_encode at :585, batch_stats at :925, flat_layout).
  - test_bottleneck.py, test_gruode.py, test_predictor.py, test_lookback.py.
  - test_jepa.py and test_mae_data.py stay, and should be updated for the renames.
- No ICML checkpoint needs a code change to load if load_mae keeps the jepa branch, _meta_from and the TokenSpec, Ladder and FrameConfig formats unchanged. The checkpoint keys are kind, backbone, mae, jepa, state_dict, spec, frames, ladder, classes, args and step.