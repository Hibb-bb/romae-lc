## How train / validation / test are used today

Code was read in /projects/bfrf/hibb/romae-lc-cleanup (branch `cleanup`). Run args and results were read in /projects/bfrf/hibb/romae-lc/project. Nothing was changed.

### 0. Where the splits come from
- On disk: `/projects/bfrf/data/PC_matches/ZTFxPC`, a HF `DatasetDict` with fixed `train` / `validation` / `test` splits made upstream. Rows: 43,497 / 5,472 / 5,464. I checked: no `gaia_dr3_source_id` appears in two splits, and every row has period > 0.
- `project/common.py:137` `load_data(args, splits=("train","validation"))` calls `romae_lc/data.py:339` `load_pc` once per split. The default only loads train and validation, so test has to be asked for by name.
- Filters in `load_pc`:
  - Drops unclean points (`clean`, `quality_flag`) and non-finite points (data.py:420-428).
  - Drops rows with a label outside the vocabulary or no positive period (:439).
  - Drops records with fewer than `--min-points` points (default 8, common.py:97; all ICML runs use 8) (data.py:450).
  - `--max-rows` keeps the FIRST rows, not a random set (data.py:396). No ICML run uses it.
- One class vocabulary is taken from all three splits: `class_vocabulary` (common.py:109-118) reads label names from test too. This uses names only, not rows, so it is not a leak.
- Normalisation: `normalize(r, ref_days=norm_days)` (common.py:175). For every ICML run `norm_days=None`, so each star is scaled with the median and MAD of its WHOLE light curve, including any points hidden later at evaluation.
- After `--min-points 8` (log of enc_j1): 43,191 train / 5,414 val records.
- Span filter for training sequences: `FrameDataset` drops records shorter than window·(1+(n_frames−1)·advance_lo) (romae_lc/frames.py ~279; common.py:271). That is 2,000 d for 8×250 d windows, which leaves 40,309 train sequences. enc_le1 uses n_frames=1, so it needs 250 d.
- `--data sim` makes its own 60/20/20 split (common.py:139-152). The Chen cache splits by id hash (common.py:156). Neither is used in the ICML runs.

### Table: step -> split(s) -> for what -> file:line

| step | split(s) | for what | file:line |
|---|---|---|---|
| Pretrain: gradient steps | train (all 43,191; sequences that pass the span filter) | encoder loss | pretrain_mae.py:453-454, 491-508 |
| Pretrain: RoPE ladder | train (periodograms of `--ladder-curves 128` records) | time-encoding wavelengths, no labels | pretrain_mae.py:467; common.py:634 |
| Pretrain: error-channel stats | train | token standardisation | pretrain_mae.py:469 |
| Pretrain: catalogue period as training target | train only (`train_periods`) | `--spectral-aux 0.5` cross-entropy on the period bin (ALL 10 ICML runs); `--mask-mode phaseband` mask (enc_o3_fcph_pband); period-hit logging in `forward_views` | pretrain_mae.py:690, 719-760, 785-790, 862-863, 1055, 1068-1074; jepa.py:366-442; romae_lc/spectral.py:194 |
| Pretrain: val loss | validation, `subset(val, --val-objects 1000, seed)` → 931 that pass the span filter | recon / JEPA loss with "random" and "block" masks, no aux | pretrain_mae.py:510-512, 949-962 |
| Pretrain: in-training probe | fit `subset(train_kept, --probe-train 4000)`; score `subset(val_kept, --probe-val 2000)`, but `val_kept` has only 931, so ≤931 stars are scored | linear class probe (LBFGS, 13 fine classes, anomaly classes out) + ridge log10 P R2 (pooled, within superclass, per superclass) | pretrain_mae.py:933-934, 963-973; common.py:1287-1346 |
| Pretrain: hand-feature baseline | same probe_tr / probe_va | reference line | pretrain_mae.py:1031 |
| Pretrain: shuffle score | validation (`val_kept`, `--shuffle-objects 512`) | time-shuffle diagnostic | pretrain_mae.py:974-982; diagnostics.py:24 |
| Pretrain: checkpoint | none | `mae.pt` = last step, no best-on-val selection | pretrain_mae.py:1157 |
| cache_latents → latents.pt | train + validation (default `--splits`) | window latents, `--stride 0.25` window units, every window that fits (`fill=True`, ≥1 per star), all stars | cache_latents.py:229-243, 289-294; eval_encoder.sh:29 |
| cache_latents → latents_test.pt | test (`--splits test`) | same, stride 0.25 | eval_encoder.sh:30 |
| period_probe | fit train, score validation, never test | 13 read-outs (ridge/mlp/bins/joint/winjoint × mean/meanmax/hand). "best" = max rec10 on VALIDATION and reported on validation. LS on `--n-ls 50` val stars. LS P_min from train periods | period_probe.py:1038-1043, 1064-1101, 1103-1107; eval_encoder.sh:31 |
| spectral_peak_probe | validation only (`--n 100000` = all val stars with a period; 5,199 with an estimate) | no fit; spectral-layer peak vs catalogue P; windows ≥32 tokens, stride 0.25 | spectral_peak_probe.py:24-29, 58, 75; eval_encoder.sh:32 |
| token_phase_probe | validation only: 600 val stars, split 70/30 by star (`rng.random<0.7`) | ridge (RidgeCV alpha by internal CV) + sklearn MLP (internal early stopping) on token phase; target from catalogue P + template ref phase | token_phase_probe.py:55-72, 143-158, 203; eval_encoder.sh:33 |
| attn_probe | fit train; epoch chosen on validation (macro F1 / R2, `val_best`); score test; seeds 42/100/200; `--epochs 40` in the job (default 60) | 13-class F1 and log P (r2/rec1/rec10). Labels, class weights, latent mean/std, y mean/std all from train | attn_probe.py:33, 108-119, 134-138, 162-183; eval_encoder.sh:34 |
| classify_rf (only with RF=1) | grid search: fit train, score validation (PredefinedSplit); final: fit train+validation, score test, 3 seeds | 16 classes (anomaly kept unless `--exclude-anomaly`). The job uses `--skip-hpo` with fixed `{"max_depth": null, "min_samples_split": 5, "n_estimators": 100}`, and I could not find where these came from | classify_rf.py:121, 146-158; eval_encoder.sh:37-38 |
| jepa_forecast_probe, default | validation only: `--n-stars 600` val stars with n≥64, 70/30 by star; MLP decoder fit on the 70 %, scored on the 30 %; stride 0.25 | tail / gap / phase-gap z-scores (option2/3 jobs) | jepa_forecast_probe.py:42-47, 258-264, 183-189; option3_one.sh:32-34 |
| jepa_forecast_probe, test | `--split test --fit-split validation --n-fit 1000 --n-stars 0 --stride 1.0`: decoder on 1,000 val stars, scored on all test stars with n≥64 | final forecast numbers | jepa_forecast_probe.py:265-270; test_forecast.sh:18 |
| jepa_forecast_demo | validation (stars with n≥200 and a period) | figures only | jepa_forecast_demo.py:54, 59 |

### Where the catalogue period is used (point 9)
- **Training target, train split:**
  - `--spectral-aux 0.5` is set in every ICML run (enc_j1, o2_*, o3_*, le1). See pretrain_mae.py:755-760 and, for le1, :1068-1074 → jepa.py:393-396.
  - The `phaseband` mask in enc_o3_fcph_pband (pretrain_mae.py:785-790, mae_data.py:251-269).
  - Train-time period-hit stats in `forward_views` (jepa.py:403-442). These are logged only.
  - `--phase-loss`, `--fold-loss` and `--template-*` also use it, but they are off in all ICML runs.
- **Encoder input:** never. The phasor / phase-rope uses the spectral layer's own estimate (jepa.py:79).
- **Evaluation only:**
  - It is the target in the in-training logP probe, period_probe, spectral_peak_probe and attn_probe logp.
  - token_phase_probe uses it for the phase target and the reference phase (:57, 61, 104-107).
  - jepa_forecast_probe uses it for the `--phase-gap` mask (:127-128) and the "catalogue f" Fourier baseline (:201-205).
  - jepa_forecast_demo uses it only to pick stars and to fold the plots.

### Inconsistencies and leaks
1. **Test is used to choose between runs.** No script selects on test inside its own code. But the PLAN "Option 3 results" table (docs/PLAN.md:212-227) compares runs and draws recipe decisions ("rope costs about 0.015 F1", "80 % hurts", "phase band helps") from the "class F1" column. That column is attn_probe `class/mean` macro F1 on TEST (e.g. o2_noadd 0.595 = attn_probe_enc_o2_noadd test). The validation score is already saved as `val_best` (o2_noadd 0.652) and could be used instead. lejepa.sh:22 also says "the test card on the TEST split" for a run that is still in development.
2. **The PLAN table mixes splits with no label.** Forecast columns = validation (600 stars, 30 % scored, stride 0.25). Class F1 = test. "Period within 10 %" = period_probe best-of-13 read-outs, picked AND reported on validation (e.g. o2_noadd 0.717 = mean/joint). That is a max over 13 models, so it is optimistic. attn_probe's test rec10 for the same run is 0.758 (mean) / 0.746 (attn); spectral_peak gives 0.850.
3. **The period read-out never reaches test** in period_probe (period_probe.py:1038 loads only train + validation). The period numbers in PLAN are not held-out-test numbers.
4. **Different fit sets per probe.**
   - attn_probe fits on train.
   - classify_rf fits on train+validation.
   - The forecast decoder fits on validation stars (both modes) and never on train.
   - token_phase fits on 70 % of validation.
   - So "validation" is a fit set for some probes and a selection set for others.
5. **Different protocols for the same metric.**
   - Validation forecast: stride 0.25, 600 stars, decoder on the 70 %.
   - Test forecast: stride 1.0, all stars, decoder on 1,000 val stars.
   - attn_probe: 13 classes. RF: 16 classes (no `--exclude-anomaly` in eval_encoder.sh:37).
   - The in-training probe: linear, 13 classes.
6. **In-training probe size is smaller than the flags say.** `--probe-val 2000` is taken from `val_kept`, but `val_kept` comes from `--val-objects 1000` and has 931 stars after the span filter (pretrain_mae.py:510-512, 934). The probe also only sees val stars that span 2,000 d, while the eval scripts use every star.
7. **Labels in a "label-free" encoder.** Every ICML run trains with `--spectral-aux 0.5` on the catalogue period (train split). This is not a split leak. But the period and phase numbers are not label-free, and the PLAN rule (docs/PLAN.md:20) only covers inference. It should say this.
8. **Within-star leak in forecast and gap tests.** With `norm_days=None`, the per-band median and MAD come from the whole light curve, including the hidden tail or gap points (data.py:479-520; common.py:175). `--norm-days` exists, but no ICML run sets it.
9. **RF hyperparameters have no known source.** The fixed `--best-params` in eval_encoder.sh:38 have no trace in the repo. They may have been tuned on another encoder, or on test.
10. **The forecast masks are not seeded.** jepa_forecast_probe uses `torch.rand` (:69, :74, :84, :89) with no `torch.manual_seed`, so gap and phase-gap numbers change from run to run. `--seed` only seeds the numpy star shuffle.
11. **Clean-up hazards found along the way (not leaks):**
    - `sys.path.insert(0, "/projects/bfrf/hibb/romae-lc")` in spectral_peak_probe.py:5, token_phase_probe.py:31, jepa_forecast_probe.py:32 and jepa_forecast_demo.py:15. Run from the cleanup checkout, these import the MAIN checkout's code.
    - token_phase_probe.py:207 imports `project.dyn.DynHead`, which is moving to legacy/.
    - jepa_forecast_probe.py:35 imports `forecast_tail_mask` from pretrain_mae.
    - jepa_forecast_probe and jepa_forecast_demo hard-code `--data` to ZTFxPC and override the checkpoint's data (:41/:257, :26).
    - PLAN test card item 4 (docs/PLAN.md:27-28) still names `state_decode_probe.py` (legacy), not `jepa_forecast_probe.py`.

### Proposed single rule
- **Train:** fit everything. That covers the encoder, ladder, error stats, catalogue-period training targets, and every probe, decoder, regressor, scaler and class-weight set. Every probe fits on train stars only, the same stars for every probe.
- **Validation:** make every choice.
  - Probe epochs and early stopping, RF and ridge hyperparameters, which read-out to report, and every run-vs-run or recipe decision in PLAN.
  - In-training monitoring and all demos and figures.
  - Any PLAN table that compares runs uses validation numbers only.
- **Test:** one final pass per frozen recipe, with the fit and selection rules above. Do not read test numbers while comparing runs. Each test number has a validation twin made with the exact same protocol (same stride, star filter, class set and seeds).
- **Catalogue period:** allowed as a training target on train stars, and stated as such. At evaluation it is only a target, a mask for a test, or a reference baseline. It is never an encoder input.

Concrete changes this rule implies:
- attn_probe: report `val_best` for run comparisons.
- period_probe: add test scoring, with the read-out picked on validation.
- jepa_forecast_probe: fit the decoder on train stars in both modes, use one stride, seed the masks.
- classify_rf: fit on train only, or state the train+val exception.
- eval_encoder.sh: use one class set.
- Pretrain: draw the in-training probe's validation stars from more than the 1,000 val-loss stars.
- Forecast tests: use `--norm-days`, or scale from the visible points only.