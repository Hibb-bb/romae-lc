AUDIT: project/eval/ and project/jobs/ on branch "cleanup" (/projects/bfrf/hibb/romae-lc-cleanup). Read only. Nothing was changed.

## 1. Eval scripts on the CURRENT test card (keep)

Some of these are not in eval/ but the card uses them: project/cache_latents.py and project/period_probe.py.

| script | purpose | inputs | split scored | outputs |
|---|---|---|---|---|
| project/cache_latents.py | Window latents of a frozen encoder | `--ckpt`, `--splits` (default train validation, cache_latents.py:243) | train+val, or test | project/runs/R/latents.pt and latents_test.pt |
| project/period_probe.py | Period read-out (bins/joint MLP) from pooled latents, with LS next to it | `--latents` (train+val cache only, period_probe.py:1038) | **validation** | project/results/period_enc_R/ (results.json, predictions.npz, tables.md, worst.csv, png) |
| eval/spectral_peak_probe.py | Spectral-layer peak period hit rates (strict and alias), median over windows with >=32 tokens | `--ckpts`, `--data`, `--n`, `--save` | **validation only**, hard-coded (lines 24, 58) | only stdout. `--save` writes an npz. eval_encoder.sh does not pass `--save`, so the numbers live only in Slurm logs (e.g. project/logs/o2-l3-3563722.out, eval-j1-3469218.out) |
| eval/token_phase_probe.py | Per-token phase R2 (cos/sin) with ridge and MLP. Controls: raw, cls, shuffled, `--random-init` | `--ckpts`, `--n-stars`, `--out` | **validation only**, hard-coded (lines 55, 194). Split 70/30 by star inside it | project/results/token_phase/R.json |
| eval/attn_probe.py | Star-level attention probe (`StarProbe`). Tasks `class` (13 fine classes) and `logp` | `--latents`, `--test-latents` | fit on train, epoch picked on val, **TEST** | project/results/attn_probe_R/results.json |
| eval/classify_rf.py | Random forest on the pooled latent (collaborators' protocol). Optional, `RF=1` | `--latents`, `--test-latents`, `--features` | **TEST** | project/results/classify_rf_R/ (results.json, tables.md, report csv, confusion png) |
| eval/jepa_forecast_probe.py | Masked-predictor forecast. Modes: tail (`--tail`), random gaps (`--gap-share`), phase gap (`--phase-gap`, catalogue P at eval only). Scores z in sigma for these rows: recon head, predicted latent [mlp], target latent, window mean, Fourier at f of the window, Fourier at catalogue f | `--ckpts`, `--split`, `--fit-split`, `--n-fit`, `--n-stars`, `--stride` | default **validation** with 70/30. test_forecast.sh uses `--split test --fit-split validation --n-fit 1000 --n-stars 0 --stride 1.0` | `--out` json, `--save-npz` pairs_<label>.npz |
| eval/jepa_forecast_demo.py | Demo plots: one val star per class, last `--hide` days hidden, a dense query grid, the model curve against 4-harmonic Fourier | `--ckpt`, `--hide`, `--grid`, `--classes` | validation (line 54) | `--out`.png/.pdf (project/results/figures/jepa_forecast_demo_*) |
| eval/dataset_figs_style.py | Shared colours, `tidy`, `save` | – | – | – (jepa_forecast_demo imports it, line 22) |

## 2. Legacy eval scripts (move to legacy/eval/)

- **budget_sweep.py**: model-seeded fine search against trial budget. Inputs `--ckpt`, `--predictions`. Writes results.json and per_star.npz. Group: LS benchmark / period pages.
- **ls_benchmark.py**: model against Lomb-Scargle trials and wall-clock. Inputs `--ckpt`, `--predictions`, `--head`. Writes results.json, per_star.npz, tables.md and figures. Group: LS benchmark. It imports `project.eval.lomb_scargle`, which does not exist in either checkout (line 56). It falls back to a root `lomb_scargle`.
- **ls_allstars.py**: astropy LS on every validation star (CPU). Writes per_star.npz and results.json. Group: LS benchmark.
- **runtime.py**: wall-clock per step, quantisation. Writes per_star.npz, tables.md, runtime.png. Group: runtime.
- **period_report.py**: period report on all val stars (model, refined, candidates, LS). Writes per_star.npz, tables.md, figures. Group: period pages.
- **period_page.py**: builds the period HTML page. Group: period pages.
- **period_table.py**: a period for every star (`--template-periods` source). Writes periods.csv and per_star.npz. Group: period pages. None of the ICML runs set `template_periods` (checked args.json).
- **period_chen.py**: model periods on the 800k Chen set. Group: Chen (PLAN: Chen not used). project/chen_data.py:54 imports `load_shard` from it.
- **period_sensitivity.py**: the smallest period shift the latent can see. Group: old period stage.
- **period_figs.py**: period_hits.{png,pdf,json} from figures/peak_maew_spec_val.npz plus ls_allstars. Group: old-stage figure.
- **hit_curves.py, scatter_by_class.py, scatter_per_class.py, slides_figs.py, fold_examples.py**: slide figures from period_report_spec4. Group: old-stage figures.
- **method_page.py, diagrams.py**: talk page and diagrams. Group: old.
- **forecast_demo.py, recon_demo.py, sine_demo.py**: MAE / cyc-rope / sine-test demos. Group: old stage.
- **state_decode_probe.py**: forecast test of the DynHead add-on. It imports `project.dyn` at line 350. Group: add-on.
- **decode_figs.py**: decode_zscore figure from the state_decode npz. Group: add-on.
- **dyn_curves.py**: add-on training curves. Group: add-on.
- **forecast_analysis.py**: Phase 1 analysis figure (add-on curves). Group: add-on.
- **nepa_class_fig.py**: J3 against maew_spec F1 bars. Group: old figure.
- **starembed_feats.py**: StarEmbed hand features for the RF baseline. Group: baseline (borderline). Keep it only if the hand-feature baseline row stays on the card. It is used through classify_rf `--extra-features`.
- **dataset_summary.py, dataset_figs.py, fold_figs.py**: dataset figures (labels, periods, folds) and maew_spec classification bars. Group: paper figures, not the test card. The ztfxpc_labels/periods figures are still valid. classification.png is from an old stage.

## 3. Job scripts

**Keep (current pipeline):**
- **pretrain_mae.sh**: the training chain. Self-relink logic: lines 65-71 (`LINK`, `MAX_LINKS`, `BUDGET`, `queue()`). The `THEN`/`PIPELINE`/`WM_OUT` handling at lines 40-44 and 57-63 only feeds legacy stages (train_wm.sh, cache_latents.sh→train_predictor, gate.sh, train_decoder.sh). The header comment at lines 14-28 describes those stages. Keep the self-chain and drop the stage hand-off. lejepa.sh, option2*.sh, option3_one.sh, phase1*.sh, encoder_ablation.sh, submit_pipeline.sh and pretrain_mae_bn/wide.sh all call it.
- **lejepa.sh**: enc_le1. It calls pretrain_mae.sh, then eval_encoder.sh and test_forecast.sh.
- **eval_encoder.sh**: cache_latents, period_probe, spectral_peak_probe, token_phase_probe, attn_probe, and optionally (RF=1) classify_rf. Called by lejepa.sh:34, option2.sh:38, option2_one.sh:38, option3_one.sh:40, phase1.sh:34, phase1b.sh:35.
- **test_forecast.sh**: jepa_forecast_probe on TEST in 3 modes (tail10, gap0.3, pgap0.2). Called by lejepa.sh:35.
- **option2.sh** (4-GPU) and **option2_one.sh** (1-GPU lanes): recipes of enc_o2_noadd/fc/fcbig/fcbig_ph. They are near duplicates (same `B`, `FC`, `BIG`, `PH` strings at line 25). Keep one, or keep both as run recipes.
- **option3_one.sh**: recipes of enc_o3_fcph/ph/fcph80/fcph_pband.
- **rope_demo.sh, rope_demo_int.sh**: jepa_forecast_demo figures, plus tail60 forecast runs. Keep as demo jobs, or fold them into one `demo.sh`.
- **attn_probe.sh, token_phase_probe.sh, cache_latents.sh, period_probe.sh, classify_rf.sh**: standalone wrappers of card steps. They can stay, but their default CKs point at old runs (token_phase_probe.sh CK=maew_spec, enc_chenft...). cache_latents.sh `THEN` chains to train_predictor.sh (legacy).
- **after_done.sh, submit_queue.sh**: generic helpers (keep or legacy, your choice). submit_queue.sh comments name gate.sh.
- **No job script trains enc_j1.** It has no jepa_init and no dyn, and mask_mode is blockplus. Its recipe exists only in project/runs/enc_j1/args.json. Every o2/o3 run warm-starts from it (`--jepa-init project/runs/enc_j1/mae.pt`). A recipe file for J1 is needed, or the card cannot be reproduced from scratch.

**Legacy:**
- **Add-on / Phase 1:** phase1.sh (state_decode_probe, DynHead runs), phase1b.sh (state_decode_probe plus jepa_forecast_probe), state_decode.sh, density_probe.sh.
- **Superseded by test_forecast.sh:** gap_probe.sh and pgap_probe.sh (validation-split gap and phase-gap runs).
- **World model, predictor and decoders:** train_wm.sh, train_wm_w60.sh, train_predictor.sh, train_decoder.sh, gate.sh, inject.sh, infer.sh, pipeline.sh, residual.sh, anomaly.sh, forecast.sh, train_lookback.sh, smoke_wandb.sh, bench.sh.
- **Old period stage:** budget_sweep.sh, ls_benchmark.sh, ls_allstars.sh, runtime.sh, eval_bundle.sh, period_report.sh, period_report_inner.sh, period_table.sh, period_sensitivity.sh, period_head.sh, period_chen.sh, chen_cache.sh, spectral_peak_probe.sh (Chen variant).
- **Old ablations:** encoder_ablation.sh, rope_ablation.sh, submit_pipeline.sh (it calls encoder_ablation.sh), pretrain_mae_bn.sh, pretrain_mae_wide.sh, pretrain_phase.sh, phase_probe.sh, phase_decoder.sh, token_head.sh, sine_test.sh.
- **Other:** dataset_figs.sh (paper figures), starembed_feats.sh and delta_cpu_starembed.sh (baseline features), lcjepa_synth.sh (Dennis test bed at the repo root; those root files are untracked).

## 4. Results in /projects/bfrf/hibb/romae-lc/project/results made by current-card scripts

- attn_probe_<run>/results.json: enc_o2_noadd, enc_o2_fc, enc_o2_fcbig, enc_o2_fcbig_ph, enc_o3_fcph, enc_o3_ph, enc_o3_fcph80, enc_o3_fcph_pband. **There is none for enc_j1** (its card has classify_rf_enc_j1 instead). Older runs also have one (j3, j3b, j3c, j4, p1_*, maew_spec).
- period_enc_enc_<run>/ (period_probe): j1, all o2, all o3.
- token_phase/<run>.json: j1, all o2, all o3. Older runs too.
- classify_rf_enc_j1/, classify_rf_enc_j3/.
- jepa_forecast/<run>.json and pairs_<run>.npz (validation, 600 stars, 70/30): j1, o2*, o3*, and p1_fc, p1_fc75, p1_mask75, p1_nce0. Also enc_o2_fcbig_ph_tail60.json and enc_o2_noadd_tail60.json (from rope_demo.sh).
- jepa_gap/<run>_gap{0.1,0.3,0.5}.json and _pgap0.2.json plus pairs (validation): j1, o2*, o3*, p1_fc, p1_noadd.
- **test_forecast/ (the TEST split, the real card numbers) has only enc_o2_noadd and enc_o3_ph** (tail10, gap0.3, pgap0.2 each).
- figures/jepa_forecast_demo_{o2_fcbig_ph,o3_fcph_pband}_h{30,60}.{png,pdf} (jepa_forecast_demo).
- figures/peak_enc_j1_val.npz (spectral_peak_probe `--save`).
- Spectral-peak hit rates for the other runs exist only in Slurm logs.
- enc_le1 has no results. Its run folder has only args.json, log.jsonl (no train steps yet) and ladder.json. There is no mae.pt or DONE.
- Everything else in results/ (period_*, phase_*, sens_*, ls_*, runtime_*, budget_sweep_spec, period_report_spec4, period_page, hit_curves, slides, method_page, sine_*, forecast_demo*, recon_demo_*, state_decode, anomaly_maew, forecast_maew, starembed_feats, period_chen, and figures decode_zscore*, dyn_curves*, forecast_analysis, nepa_class, period_hits, fold_*, classification, ztfxpc_*) comes from legacy scripts.

## 5. Problems found

1. **The card does not all score on TEST.** spectral_peak_probe.py and token_phase_probe.py hard-code "validation". period_probe.py only loads train+validation (line 1038). attn_probe logp (TEST) is the only TEST period number. jepa_forecast_probe defaults to validation; only test_forecast.sh sets `--split test`.
2. **docs/PLAN.md:24-30 is out of date.** Card item 4 still names state_decode_probe.py and decode_figs.py (the add-on). The current forecast test is jepa_forecast_probe.py.
3. **token_phase_probe.py:204-213 imports `project.dyn.DynHead`** when the checkpoint has `aux.dyn`. Moving dyn.py to legacy/ breaks this branch. Drop the branch, or guard it. `embed(..., dyn=None)` at line 88 also carries the dyn path.
4. **Hard-coded `sys.path.insert(0, "/projects/bfrf/hibb/romae-lc")`** in jepa_forecast_probe.py:32, jepa_forecast_demo.py:15, spectral_peak_probe.py:5, token_phase_probe.py:31 and state_decode_probe.py:39. In the cleanup worktree this imports the MAIN checkout's code, not the branch's. Remove it and rely on `python -m`.
5. **lejepa.sh:32 may never run its evals (likely).** It runs pretrain_mae.sh with MAX_LINKS=2. Link 1 queues link 2 as plain pretrain_mae.sh (`SCRIPT`, pretrain_mae.sh:39,68-70). If 33,500 steps do not finish inside BUDGET=84000 s, lejepa.sh sees no DONE and skips the evals, and link 2 never runs them.
6. **Tests import legacy eval modules:** project/tests/test_budget_sweep.py, test_ls_benchmark.py, test_period_report.py, test_period_table.py, test_period_sensitivity.py and test_runtime.py. They must move with them. test_classify_rf.py stays.
7. **Comments in core code name eval modules:** project/pretrain_mae.py:279 (`--template-periods` from period_table) and project/templates.py:26 (legacy). project/jepa.py:304 names jepa_forecast_probe (fine).
8. **Script-level argparse.** jepa_forecast_probe.py, spectral_peak_probe.py, token_phase_probe.py and jepa_forecast_demo.py parse args at import time and use the globals `a`, `GAP`, `PGAP`, `DT_BINS`. They cannot be imported by a runner until they are wrapped in functions.

## 6. Duplicated code today

- **Fourier forecast, 3 copies:**
  - `fourier_forecast` in jepa_forecast_probe.py:93
  - `fourier_forecast` in state_decode_probe.py:107
  - `fourier` in jepa_forecast_demo.py:38
  - All three are the same weighted lstsq fit with K harmonics.
- **`zstats`, 2 copies:** jepa_forecast_probe.py:104 and state_decode_probe.py:120. The second also returns `within_3` and has no empty-input guard.
- **The encoder → spectral frequency loop** (WindowDataset + make_collate + `spectral_frequency(layer, None, "local")`) is in jepa_forecast_probe.py `collect` (line 140), spectral_peak_probe.py:65-71, jepa_forecast_demo.py and state_decode_probe.py `encode_all`.
- **The `WindowDataset([(split, recs, idx)], cfg, stride, cfg.max_tokens, 0, 1, 0.5)` boilerplate** is in spectral_peak_probe.py:58, token_phase_probe.py:194, jepa_forecast_probe.py:272 and jepa_forecast_demo.py:59.
- **Star picking** (the load_data split, the filter on period / `r.n >= 64`, rng shuffle, a 70/30 `star_train`) is written again in token_phase_probe.py:55-71 and jepa_forecast_probe.py:259-270.
- **Period hit rates (strict and alias-tolerant):** `hits` in spectral_peak_probe.py:44, `hit_within` / `recovery` in period_probe.py:532-544, and `rel_error` / `hit_table` in eval/period_report.py (legacy).
- **The latents → star table step** (`object_table` + the train / validation / test `caches` dict) is in attn_probe.py:105-111 and classify_rf.py:91-98.
- **option2.sh and option2_one.sh** repeat the same recipe strings. The `train()` / `lane()` helpers are copied in option2*, option3_one, phase1 and phase1b.

## 7. Proposed eval/ layout

```
project/eval/
  common_eval.py     shared helpers, no CLI
  card_class.py      item 1: macro F1, 13 classes
  card_period.py     item 2: log-period R2 + spectral-peak hit rate
  card_phase.py      item 3: per-token phase R2
  card_forecast.py   item 4: tail / gap / phase-gap
  card_demo.py       demo plots (not scored)
  test_card.py       runner: one JSON per run, TEST split
  summary.py         table over runs
  style.py           (= dataset_figs_style.py)
  classify_rf.py     kept as the reference RF baseline
legacy/eval/         everything in section 2
legacy/jobs/         every legacy job in section 3
```

**common_eval.py** reuses:
- `fourier_forecast` and `zstats` (from jepa_forecast_probe.py; delete the other copies).
- `load_encoder`, `load_data`, `data_args_from`, `superclass` (project/common.py).
- `WindowDataset`, `make_collate` (project/cache_latents.py).
- `spectral_frequency` (romae_lc.model).

New helpers in it:
- `pick_stars(split, n, need_period, min_n, seed)`
- `window_loader(recs, cfg, spec, stride)`
- `period_hits(p_hat, p, tols)`, taken from `hits` in spectral_peak_probe.py:44.

**card_class.py** reuses `StarProbe`, `batches`, `scores` and `run` from attn_probe.py (move the file and rename it, or import it), plus `object_table` from period_probe.py. It scores `class/attn` and `class/mean` on TEST. classify_rf.run stays optional.

**card_period.py**:
- log-period R2: from attn_probe `logp` (TEST), the same run as card_class.
- spectral-peak hit rate: the body of spectral_peak_probe.py:52-77, made into a function `peak_periods(enc, meta, recs)` with a `split` argument. It also needs `period_hits`.
- project.period_probe (bin read-out, validation) becomes optional, or gets a test-cache option, because today it has no TEST path.

**card_phase.py** reuses `pick_windows`, `embed` (without the dyn branch), `r2` and `probes` from token_phase_probe.py. Add a `split` argument. `reference_phase` stays in project/phase_probe.py, so phase_probe.py, or that one function, must stay out of legacy/. The `random-init` control stays.

**card_forecast.py** reuses `gap_mask`, `phase_gap_mask`, `collect` and `score` from jepa_forecast_probe.py, plus `forecast_tail_mask` (project/pretrain_mae.py:408). Change `a.*`, `GAP`, `PGAP`, `DT_BINS` and `LOCAL` into a mode argument. Keep the test_forecast.sh settings as defaults: split test, fit-split validation, n-fit 1000, n-stars 0, stride 1.0, tail 10, gap 0.3, pgap 0.2. One `collect` pass per mode.

**card_demo.py** is jepa_forecast_demo.py with its `fourier` replaced by `common_eval.fourier_forecast`, wrapped in `main()`.

**test_card.py**:
- Input: `--run NAME` (or a ckpt).
- Makes latents.pt and latents_test.pt with cache_latents if they are missing (as eval_encoder.sh:29-30 does).
- Runs class, period, phase and forecast on TEST.
- Writes project/results/test_card/NAME.json with keys class.macro_f1 (mean±std over 3 seeds), period.logp_r2, period.peak_hit_{10,1,0.1,0.01}% (strict and alias), phase.cos_r2 (h+cls MLP, raw and shuffled controls), and forecast.{tail10,gap0.3,pgap0.2}.within_2 for the predicted latent, the recon head, Fourier and the window mean.
- It also records the ckpt step, the git commit, and args.json.

**summary.py** reads project/results/test_card/*.json and writes one markdown/CSV table, one row per run.

**Jobs after the cleanup:** pretrain_mae.sh (self-chain only), recipes/ (enc_j1 recovered from args.json, option2 lanes, option3 lanes, lejepa), test_card.sh (replaces eval_encoder.sh plus test_forecast.sh), demo.sh (replaces rope_demo*.sh).

Files read:
- /projects/bfrf/hibb/romae-lc-cleanup/project/eval/*.py
- /projects/bfrf/hibb/romae-lc-cleanup/project/jobs/*.sh
- /projects/bfrf/hibb/romae-lc-cleanup/docs/PLAN.md
- /projects/bfrf/hibb/romae-lc/project/runs/*/args.json
- /projects/bfrf/hibb/romae-lc/project/results/