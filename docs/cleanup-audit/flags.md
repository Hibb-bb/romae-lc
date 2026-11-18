Flag audit of project/pretrain_mae.py (cleanup checkout, commit 27191a2). Line numbers refer to /projects/bfrf/hibb/romae-lc-cleanup/project/pretrain_mae.py unless another file is named. The defaults come from the real parser (add_data_args, add_frame_args, add_ladder_args, add_model_args, add_train_args, add_wandb_args). I compared them with each run's args.json. All 10 args.json files exist. enc_le1 has no DONE file yet. All other runs are DONE. Script: /tmp/claude-91745/-projects-bfrf-hibb-romae-lc/f846f30c-9ae3-4667-b5f6-cb47d35ed7de/scratchpad/diff.py

## (a) Flags set to a non-default value by any ICML run

Columns: j1 | o2_noadd | o2_fc | o2_fcbig | o2_fcbig_ph | o3_fcph | o3_ph | o3_fcph80 | o3_fcph_pband | le1  ("." = default)

Model and objective flags:
- --size (light): wide in all 10. --width/--heads/--depth are saved as 384/6/6. They come from the preset through resolve_model_args (common.py:870). They are not set by hand.
- --spectral (off): on in all 10.
- --spectral-aux (0): 0.5 in all 10.
- --jepa (off): on in all 10.
- --jepa-proj (0): 32 in the 9 EMA runs | 16 in le1.
- --jepa-recon-weight (0): 0.5 in the 9 EMA runs | . in le1 (it must be 0 with --jepa-no-ema).
- --jepa-init (None): . | then project/runs/enc_j1/mae.pt for all o2 and o3 runs | . in le1. The path is relative, so the run must start in the repo root and enc_j1 must exist.
- --jepa-no-ema (off): only le1.
- --mask-mode (random): blockplus in j1, all o2 runs, o3_fcph, o3_ph and o3_fcph80 | phaseband in o3_fcph_pband | . in le1.
- --mask-ratio (0.5): 0.8 in o3_fcph80 | 0.85 in le1 | . in the rest.
- --forecast-tail (None): [10, 60] in o2_fc, o2_fcbig, o2_fcbig_ph, o3_fcph, o3_fcph80 and o3_fcph_pband | . in j1, o2_noadd, o3_ph and le1.
- --dec-width/--dec-heads/--dec-depth (192/3/2): 384/6/4 in o2_fcbig and o2_fcbig_ph only.
- --phase-rope (0): 4 in o2_fcbig_ph, all o3 runs and le1 | . in j1, o2_noadd, o2_fc and o2_fcbig.
- --phase-freq (soft): local in every run that has --phase-rope 4. The CLI default "soft" is never used by an ICML run. The TokenJEPA constructor default is already "local" (jepa.py:70).

Frame and data flags:
- --window (500): 250 in all 10. This comes from pretrain_mae.sh.
- --n-frames (8): . in the 9 EMA runs | 1 in le1.
- --batch-size (64): . | 256 in le1.

Run-control flags:
- --steps (50000): . in j1 | 20000 in the 8 o2/o3 runs | 33500 in le1.
- --time-budget (0): 6000 | 30000 for o2/o3 | 84000 for le1.
- --ckpt-every (1000): 500 in all. This comes from pretrain_mae.sh.
- --pin-memory and --wandb: on in all.
- --device cuda.
- --out is project/runs/<name>.
- --wandb-group: jepa_aux | option2 | option3 | lejepa.

pretrain_mae.sh (project/jobs/pretrain_mae.sh, last command) always passes: --window 250 --min-tokens 16 --max-tokens 256 --steps 50000 --batch-size 64 --workers 8 --pin-memory --eval-every 2500 --ckpt-every 500 --time-budget $BUDGET --wandb. The extra args are placed after these and override them.

Older args.json files lack keys for flags added later. The keys are still in the parser, so the parser default applies:
- enc_j1 lacks forecast_*, mask_ratio_range, phase_band, attn_pool, dyn_* (newer ones), jepa_no_ema, sig_*, views, view_keep, local_span and band_drop.
- The o2 and o3 runs lack jepa_no_ema, sig_lambda, sig_slices, views, view_keep, local_span and band_drop.

## (b) Flags left at default but still read on the ICML code path (needed to reproduce)

Data (common.py:83-106, load_data at common.py:137):
- --data (/projects/bfrf/data/PC_matches/ZTFxPC), --min-points 8, --seed 0.
- --classes, --max-rows and --n-sim are read in load_data. --norm-days is read by getattr (common.py:152,177).

Frames (common.py:268-299):
- --advance [1.0, 1.5], --min-tokens 16, --max-tokens 256, --n-frames.

Ladder (get_ladder, common.py:634-680; it reuses out/ladder.json when that file exists):
- --time-spacing quantile, --time-mix 0.75, --max-freq 50, --ladder-curves 128, --lam-min/--lam-max None, --ladder-mode dense, --ladder-deal bands, --ladder None.
- --rope-wavelengths and --time-scale are read but only matter when set (dead for ICML).

Encoder (encoder_config and rope_layouts, common.py:878-902; rope_geometry, common.py:439):
- --mlp-ratio 4, --p-rope 0.75, --time-frac 0.875, --attention softmax.
- --no-err-channel is read at line 469. The error channel is needed because aux=True (line 618).

Spectral layer (spectral_kw, lines 524-526):
- --spectral-p 0.02 500, --spectral-rel 5e-4, --spectral-channels 8, --spectral-reader 32, --spectral-depth 3, --spectral-after 2.

Optimisation:
- --lr 1e-4, --wd 0.05, --warmup 0.02, --clip 1.0. The schedule at line 907 uses cosine_schedule(--steps, --warmup).
- --steps also sets the EMA schedule (ema_update(step, args.steps) at line 1088, jepa.py:213).

Predictor size (decoder dict, lines 518-523):
- --dec-width/--dec-heads/--dec-depth, plus --attention.

Plain EMA JEPA (TokenJEPA, lines 543-556):
- --jepa-ema 0.996, --jepa-ema-end 1.0, --jepa-loss smoothl1, --jepa-use-context (False, so the EMA target encoder is used downstream), --jepa-proj-hidden 256.
- --abs-time and --attn-pool are passed to the constructor but stay off (lines 546, 548). --jepa-clean-target is passed as False (line 548).

--jepa-recon-weight: TokenJEPA builds recon_head (jepa.py:110). It adds recon_weight × MSE on the hidden brightness (jepa.py:283-290). It reads no other flag. Its last_recon output is used by jepa_forecast_demo.py:107. Its recon_head is used by jepa_forecast_probe.py:166.

--phase-rope / --phase-freq: these go to the context RoMAEForPreTraining. The decoder rope is extended there (romae_lc/model.py:708-711). They are also used in TokenJEPA.forward (jepa.py:269-275, spectral_frequency). The check at line 598 says they need --spectral.

--mask-mode blockplus: make_mask (lines 795-799, mae_data.py:117-132) reads --mask-ratio, --mask-block-share 0.5 and --mask-block-pos random. --mask-blocks is NOT read for blockplus, because it is fixed to (1, 1).

--mask-mode phaseband: lines 784-790 call phase_band_mask (mae_data.py:248). They read --phase-band 0.1 0.3 and --mask-ratio. Hidden coupling: periods_rows exists only when aux is True (line 1055). So phaseband works only because --spectral-aux > 0. Otherwise it raises at line 788.

--mask-mode random: training uses the default gen_mask inside TokenJEPA.forward. Its ratio is context.mask_ratio, which is --mask-ratio.

Every run, during evaluation: evaluate() (lines 944-1027) runs forward(frames, "random") and forward(frames, "block"). The block pass reads --mask-blocks 1 6, --mask-block-pos and --mask-ratio. So --mask-blocks stays live through val_loss_block.

--mask-ratio: read at line 780. It is stored in ckpt["mae"]["mask_ratio"]. For le1 it is used only by the eval val loss (TokenJEPA.forward with gen_mask), not by forward_views.

--forecast-tail: lines 800-807 call forecast_tail_mask (lines 408-425). They read --forecast-share 1.0 and --window. The mask starts from gen_mask(ratio) when the mode is random.

--spectral and --spectral-aux:
- aux_losses at lines 755-760 call romae_lc.spectral.spectral_aux_loss with periods_rows.
- The checks at lines 615-619 apply.
- train_periods is built at line 690.

--jepa-no-ema (training loop, lines 1064-1078, jepa.forward_views at jepa.py:367):
- Reads --views 2 4, --view-keep 0.15, --local-span 30 80, --band-drop 0.2, --window (passed to make_views but unused there), --sig-lambda 0.05, --spectral-aux and --sig-slices 1024 (TokenJEPA constructor).
- Reads --phase-freq. In forward_views it is used for period-hit monitoring even when --phase-rope is 0.
- Reads --jepa-proj (it must be > 0) and --spectral.
- The check at lines 551-552 reads --jepa-proj, --jepa-recon-weight, --dyn and --forecast-tail.
- forward_views hard-codes the MSE loss. --jepa-loss, --jepa-ema, --jepa-ema-end and --jepa-use-context have no effect in training.
- The mask flags are read only by evaluate().

Evaluation and run control:
- --probe-train 4000, --probe-val 2000, --val-objects 1000, --shuffle-objects 512, --batch-size, --seed.
- --eval-every, --log-every, --workers, --prefetch, --persistent-workers, --no-resume, --no-eval-at-start.
- --wandb-project/--wandb-entity/--wandb-name/--wandb-tags (tracking.py:22-29).
- The RUN_CONTROL tuple (lines 96-120) lists the flags taken from the CLI on resume. Everything else is restored from ckpt["args"] (lines 443-448).

## (c) Flags read only by dead paths, with the module each pulls in

- Bottleneck: --bottleneck, --bottleneck-loss, --bottleneck-var, --denoise. Module: project/bottleneck.py (BottleneckAE, bottleneck_state, fuse_tokens), imported at the top (line 55). It uses project/decoder.py QueryDecoder.
- GRU-ODE: --encoder, --rnn-band, --rnn-ode-hidden, --rnn-time-unit, --rnn-steps. Module: project/gruode.py, imported at the top (line 56).
- Templates and query tokens: --target, --template-harmonics, --template-min-r2, --template-min-points, --template-periods, --template-column, --template-fit, --template-workers, --query-extra, --query-delta, --query-ramp, --query-cycles, --query-cycles-min, --query-random-share, --input, --jepa-clean-target. Module: project/templates.py (lazy imports at lines 696, 698, 811, 842).
- Phase and fold losses: --phase-loss, --fold-loss, --fold-harm. Modules: project/phase_decoder.py (PhaseDecoder, band_ids, band_table at lines 623, 722, 875) and project/decoder.py gaussian_var (line 724).
- Dyn add-on: --dyn, --dyn-weight, --dyn-nce, --dyn-sig, --dyn-mean, --dyn-tau, --dyn-pairs, --dyn-dt, --dyn-ramp, --dyn-freq, --dyn-lambda, --dyn-phase-pos, --dyn-ham, --dyn-ham-steps, --dyn-ham-close, --dyn-cross, --dyn-target, --dyn-pred, --dyn-no-rot, --dyn-decode, --dyn-sig-floor, --dyn-dec-band, --dyn-dec-nll, --dyn-sig-mode, --dyn-freeze. Module: project/dyn.py (DynHead and HamHead at lines 631 and 634), plus phase_decoder.band_ids.
- CLS / LeNEPA: --cls-inv, --cls-sig, --attn-pool. Modules: project.dyn.sigreg (line 882) and romae_lc AttnPool.
- Absolute time: --abs-time. Uses abs_timescales() (lines 382-387) and romae_lc abs_proj.
- Phasor: --phasor, --phasor-init. Uses romae_lc PhasorFeatures and the optimiser group at lines 903-906.
- Variable windows: --window-range. Uses MultiWindowDataset (mae_data.py:158) and torch DataLoader.
- --mask-ratio-range (lines 781-783, 791-794).
- --mask-block-prob, plus --mask-mode block/mix in training. These are mix-only.
- --loss-weight, --loss-weight-cap. They use error_weights (lines 389-404). TokenJEPA.forward ignores `weight` anyway.
- --init: plain-MAE warm start through common.warm_start (line 591).
- World-model flags from add_model_args: --pred-depth, --pred-heads, --pred-dim-head, --pred-mlp, --pred-dropout, --proj-hidden, --history, --lamb, --n-slices, --knots, --t-max. pretrain_mae never reads them. Only common.build_world_model reads them (common.py:920-953). They need a pretrain-only model-args group.
- The plain masked autoencoder path (no --jepa) is not used by any ICML run. This covers lines 541-542 and 571-595, with phasor and --init. Decide whether it stays as the MAE baseline or goes to legacy/.

WARNING: project/dyn.py is to move to legacy/, but its sigreg (dyn.py:43) is LIVE. forward_views imports it (jepa.py:371: `from project.dyn import sigreg`). Move sigreg into jepa.py, or into a small module, before moving dyn.py.

## Dead code in pretrain_mae.py (line ranges)

Top-level code:
- Imports at lines 55-56 (bottleneck, gruode). The MultiWindowDataset import at line 58.
- abs_timescales() at lines 382-387. error_weights() at lines 389-404.

Inside main():
- 460-461: window_range sets lam_max. 473-474: bottleneck error-channel check. 495-508: MultiWindowDataset loader.
- 527-528: resume of a bottleneck checkpoint. 531-532: resume of a gruode checkpoint. 533-540: GRU-ODE build. 541-542: resume of a plain MAE checkpoint.
- 553-554: clean-target check. 557-570: bottleneck build. 571-595: plain MAE build and --init (see above).
- 596-597: attn-pool check. The phasor half of 598-599. 600-603: query-cycles and --input checks.
- 610: the bottleneck, gruode and mae branches of state_fn. 612-613: bottleneck mask check.
- 615: the phase-loss, fold-loss and bottleneck terms. 619: its error message names phase/fold, but it still fires for spectral-aux.
- 620-626: phase_head and fold_dec. 627-653: the dyn build. 654: aux_modules (always empty for ICML). 656-689: dyn_term().
- 691-716: smooth/templates. 717: the phase, fold and template_share keys of aux_stats.
- Inside aux_losses (719-761): the imports at 722-724, the phase head at 730-740 and the fold decoder at 741-754. Only 755-760 (the spectral aux) is live.
- Inside forward() (763-897): 769-770 bottleneck, 772-777 dyn-freeze, 781-783 and 791-794 mask-ratio-range, 808 loss weight, 809-859 templates, queries and --input, 864-879 dyn, 880-896 cls-inv/cls-sig.
- 903-906: phasor param group. This can become AdamW(model.parameters()).
- 911-913: resume of the aux modules. The aux= part of save() at line 939.
- 949 and 956: the `plain` / BottleneckAE branch in evaluate().
- 1056-1058: template, index and start rows. 1059-1062: dyn_starts. The aux_modules part of clip at line 1084.

Live core of main():
- Parsing and resume: 427-475.
- Loaders: 483-516.
- spectral_kw and the TokenJEPA build: 518-526 and 543-556 (the 529-530 resume included).
- Model to device, encoder and sizes: 604-609.
- 690.
- Spectral aux: 755-760.
- Mask, forecast-tail and model call in forward(): 778-807 and 860-863.
- Optimiser, schedule and tracker: 899-931.
- save() and evaluate(): 936-1027.
- Baseline probe: 1029-1044.
- Training loop with the forward_views branch: 1046-1164.

## What the evaluation reads from saved args and checkpoints (must survive)

- No eval script reads args.json itself. They read ckpt["args"], which is the same dict. load_encoder/load_mae (common.py:1066, 1189) puts it in meta.args. data_args_from (common.py:182) reads only DATA_KEYS = data, classes, max_rows, min_points, n_sim, seed, norm_days (common.py:179). These must stay as arg names in the parser and in the checkpoint. Users:
  - cache_latents.py:285
  - period_probe.py:1038
  - eval/spectral_peak_probe.py:21
  - eval/token_phase_probe.py:52
  - eval/attn_probe.py:108 (ck["args"])
  - eval/classify_rf.py:95
  - eval/jepa_forecast_probe.py:256
  - eval/jepa_forecast_demo.py:51
- Frames, spec and ladder come from the ckpt keys frames, spec and ladder (meta.cfg, meta.spec), not from args. So jepa_state (jepa.py:446) and its keys must stay.
- TokenJEPA.from_checkpoint (jepa.py:155) calls cls(**backbone, **mae, **jepa).
  - ckpt["jepa"] holds ema, ema_end, loss, use_target, recon_weight, clean_target, proj_dim, proj_hidden, and also phase_rope/phase_freq, no_ema and sig_slices when they are set.
  - The backbone hparams include abs_timescales (always, possibly None) and spectral.
  - So TokenJEPA and RoMAEForPreTraining must keep accepting clean_target and abs_timescales (and attn_pool) kwargs, even if they only ignore them. Otherwise the existing ICML checkpoints will not load.
- eval/jepa_forecast_probe.py:35 imports forecast_tail_mask from project.pretrain_mae. Keep that function there, or move it to mae_data and update the import.
- eval/token_phase_probe.py:205-213 lazily imports project.dyn.DynHead, but only when ckpt["aux"]["dyn"] exists. The ICML runs have dyn=0, so this branch can go to legacy with dyn.py.
- token_phase_probe imports project.phase_probe.reference_phase (line 34). This is live.
- period_probe imports project.fold, and project.train_predictor lazily at line 282. Check these before moving them.
- attn_probe and classify_rf import project.period_probe.object_table. This is live.
- The o2 and o3 runs depend on project/runs/enc_j1/mae.pt through the relative --jepa-init path. init_from_mae takes the JEPA branch (jepa.py:166-180).