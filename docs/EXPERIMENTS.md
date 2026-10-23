# Experiment ledger

One row per run and per result folder, with the recipe, the headline number,
the verdict and what the paper needs from it. Written 2026-10-19 as step 1
of the clean-up (record first, remove later). Nothing in the tree was moved
or deleted for this file. Git tag `pre-cleanup-2026-10-19` marks the state
of the code when it was written.

How to read the numbers. Unless stated, a run's score is the **period
probe**: a read-out trained on the frozen pooled latents of the 250-day
windows, scored on the 5,346 validation stars of ZTFxPC as the share of
stars whose predicted period is within 10 % / within 1 % of the catalogue
period (the best read-out of the folder, usually `mean/winjoint` or
`mean/bins`). Hand-feature baseline on the same stars: 0.38 / 0.05. The
raw numbers are in `project/results/<folder>/results.json` and `tables.md`.

Verdicts: **PAPER** = evidence the encoder paper needs; **DECODER** =
evidence the decoder work needs; **ABLATION** = a negative or side result
worth one line in the paper; **DEAD** = tried, did not help, nothing depends
on it; **SUPERSEDED** = replaced by a later run of the same idea.

## 1. Encoder runs on ZTFxPC (stage 1)

Window 250 d, up to 256 tokens, 8 windows per star per batch, dense quantile
rotary ladder unless stated. "light" = 192 wide x 6 blocks (3 heads), "wide"
= 384 x 6 x 6 heads (10.7 M), the paper's encoder.

### 1a. The path to the paper's encoder

| run | date | recipe | steps / batch | within 10 % / 1 % | verdict |
|---|---|---|---|---|---|
| `rope_standard` | 09-29 | light, standard log-spaced rotary ladder, random masks | 15k / 64 | 0.47 / 0.10 | ABLATION (ladder control) |
| `rope_dense` | 09-29 | light, dense quantile ladder (the one adopted) | 15k / 64 | 0.61 / 0.10 | ABLATION (ladder choice) |
| `rope_sharedlog`, `rope_sharedq` (results only) | 09-29 | light, ladder shared over heads and layers, log / quantile | 15k / 64 | 0.63 / 0.13, 0.63 / 0.11 | ABLATION |
| `mae_w250` | 09-26 | light, 50k, the first frozen encoder of the world model | 50k / 64 | 0.66 / 0.36 | SUPERSEDED (kept: stage-2/3 runs hang off it) |
| `mae_norm` | 09-30 | light, past-only normalisation | 15k / 64 | 0.59 / 0.09 | ABLATION |
| `mae_base` | 10-02 | 720 x 12 blocks, batch 32 (the original RoMAE base size) | 80k / 32 | 0.70 / 0.40 | ABLATION (depth does not help) |
| `maew_20k` | 10-01 | wide, random masks | 20k / 128 | 0.62 / 0.38 | ABLATION (width sweep) |
| `maex_w250` | 10-01 | 512 wide | 20k / 128 | 0.65 / 0.32 | ABLATION (width sweep) |
| `maexx_w250` | 10-02 | 768 wide, batch 64 | 20k / 64 | 0.63 / 0.21 | ABLATION (width sweep) |
| `maew_lin`, `maex_lin` | 10-01 | wide / 512 with linear attention | 20k / 128 | 0.40 / 0.12, 0.44 / 0.12 | DEAD (linear attention loses the period) |
| `maew_w250` | 10-01 to 10-05 | wide, 150k planned, stopped at 111.8k | 112k / 128 | 0.71 / 0.33 (0.58 after the fine search) | SUPERSEDED (kept: the world-model forecast runs hang off it) |
| `maew_abs` (results only) | 10-02 | wide + absolute-time sinusoids in the tokens | 20k / 128 | 0.66 / 0.39 | DEAD (no gain, phase still at chance) |
| `enc_spec` | 10-04 | light + spectral layer after block 2, plain reconstruction | 15k / 64 | 0.72 / 0.18 | PAPER (the spectral layer gives +0.11 at 15k) |
| `enc_specaux` | 10-04 | the same + auxiliary cross-entropy on the catalogue period's bin, weight 0.5 | 15k / 64 | 0.74 / 0.20 | PAPER |
| `mae_spec50` | 10-05 | light, spectral + aux 0.5 | 50k / 64 | 0.76 / 0.35 | PAPER (size row) |
| **`maew_spec`** | 10-05 to 10-08 | **wide, spectral + aux 0.5, random masks, 100k steps: the paper's encoder** | 100k / 128 | **0.79 / 0.72** (pooled read-out); spectral peak alone 0.82 / 0.81 / 0.80 / 0.71 within 10 / 1 / 0.1 / 0.01 % | PAPER |
| `enc_maespec50` | 10-16 | wide, spectral, NO aux, stretch + random masks | 50k / 64 | 0.72 / 0.19 | PAPER (the aux loss gives the precision) |
| `chen_spec` | 10-19 to 10-20 | wide, spectral + aux 0.5 on OUR periods, 800k-star Chen set, 100k steps (14.9 GPU h) | 100k / 128 | spectral peak, 1000 held-out stars each: Chen val vs Chen's sharpened period 0.77 / 0.75 / 0.74 / 0.68 within 10 / 1 / 0.1 / 0.01 % (alias-tolerant 0.82 / 0.79); ZTFxPC val 0.82 / 0.81 / 0.80 / 0.71 (alias-tolerant 0.88 / 0.86), the same as `maew_spec` there | open (no overlap between the Chen set and ZTFxPC, per Dennis; same-star comparison with `maew_spec` queued) |

Command of the paper's encoder (the chain script adds window, token and
optimiser defaults):

```bash
sbatch --job-name=mae-wspec --export=ALL,OUT=project/runs/maew_spec project/jobs/pretrain_mae.sh \
    --size wide --steps 100000 --batch-size 128 --eval-every 10000 --spectral --spectral-aux 0.5
```

### 1b. Objective and input ablations on the light encoder (15k steps, control `rope_dense` 0.61 / 0.10)

All run through `project/jobs/encoder_ablation.sh <name>`. The checkpoints of
the ones marked "results only" were removed earlier; the probe results stay.

| run | what changed | within 10 % / 1 % | verdict |
|---|---|---|---|
| `enc_block` (results only) | whole stretches of time hidden | 0.36 / 0.04 | DEAD |
| `enc_mix` (results only) | half stretch masks, half random | 0.61 / 0.09 | DEAD (no gain) |
| `enc_range` (results only) | training windows 60 to 1000 d, four window lengths joined at probe time | 0.61 / 0.12 | DEAD (no gain) |
| `enc_both` (results only) | mix + range | 0.57 / 0.10 | DEAD |
| `enc_errw` (results only) | hidden-point loss divided by sigma^2 | 0.58 / 0.10 | DEAD |
| `enc_abs` (results only) | absolute-time sinusoids in the tokens | 0.55 / 0.09 | DEAD |
| `enc_absbn` (results only) | the same + bottleneck objective | 0.35 / 0.04 | DEAD |
| `enc_phase` (results only) | + per-token phase objective | 0.62 / 0.10 | DEAD (phase probe still at chance) |
| `enc_fold` (results only) | + phase-folded reconstruction objective | 0.60 / 0.10 | DEAD |
| `enc_pfold` (results only) | both phase objectives | 0.61 / 0.10 | DEAD |
| `enc_long` (results only) | whole curve as one window (2000 d, 1024 tokens) | 0.57 / 0.08 | DEAD |
| `enc_hidden` | whole curve, one hidden 250-day stretch + random points | 0.51 / 0.07 | DEAD (the hidden-window objective) |
| `enc_hiddensp` | the same + spectral layer and aux | 0.47 / 0.09 | DEAD |
| `bnu_w250` (results only) | bottleneck autoencoder, unit variance (also bn, bnkv) | 0.38 / 0.05 | DEAD (the pooled-only decoder never learns period) |
| `phase_bn`, `phase_bn_init` (results only) | phase bottleneck: encoder trained through the phase decoder (`pretrain_phase.py`) | 0.36 / 0.05, 0.54 / 0.09 | DEAD |
| `wm_w250_dense`, `wm_w250_mae` (results only) | joint LeWorldModel training (the old stage 1) | 0.30 / 0.03 | DEAD (the prediction loss erases the period; this is why the autoencoder is frozen) |

### 1c. Wide 50k ablations (stretch + random masks, batch 64)

| run | what | within 10 % / 1 % | verdict |
|---|---|---|---|
| `enc_jepa` | token-level JEPA (EMA target, no reconstruction, no aux) | 0.63 / 0.13 | ABLATION |
| `enc_jepainit` | JEPA warm-started from `maew_spec` | 0.73 / 0.58 | ABLATION |
| `enc_jepahyb` | JEPA + reconstruction weight 0.5, blockplus masks, no aux | 0.72 / 0.48 | ABLATION (better 1 % than the no-aux baseline `enc_maespec50`, below `enc_jepainit`) |
| `enc_longspecwtpl` | whole curve as one window, wide + spectral, template targets on the catalogue period, batch 32 (an eighth of the tokens per step) | 0.60 / 0.11 | DEAD (unmatched; see `enc_wholecurve`) |
| `enc_wholecurve` | 10-21 | whole curve as one 2000 d window, 1024 random points, batch 256 (tokens per step = `maew_spec`), spectral + aux 0.5, observed targets, random masks, 50k steps, 0.30 s/step | spectral peak, same 1000 stars: ZTFxPC 0.81 / 0.78 / 0.77 / 0.70, Chen (sharpened) 0.69 / 0.63 / 0.61 / 0.56 | ABLATION (matched run, 1 to 3 points behind the 250 d window on ZTFxPC, 3 to 6 on Chen; the window stays) |
| `enc_chenft` | 10-21 to 10-22 | `chen_spec` post-trained on ZTFxPC with the `maew_spec` recipe (warm start, fresh optimiser, 30k steps, aux on the ZTFxPC catalogue period; Chen's own periods are never a target) | spectral peak, same 1000 stars: ZTFxPC 0.83 / 0.82 / 0.81 / 0.73, Chen (sharpened) 0.74 / 0.72 / 0.69 / 0.62 | open (ties `maew_spec` on ZTFxPC, keeps half of the 800k gain on Chen) |
| `enc_gruode` | GRU-ODE encoder, 768 wide, ODE 256 (2.2 M), stopped at 16.9k of 30k | 0.38 / 0.06 | ABLATION (baseline row; 744 ms per star single, 24 ms batched) |

Spectral-peak hit rates on the SAME held-out stars (`project/eval/spectral_peak_probe.py`,
job peak-all 10-23; 1000 stars per set, seed 0; strict within 10 / 1 / 0.1 / 0.01 %;
noise about 1.3 points). Chen truth = Chen's sharpened period.

| encoder | ZTFxPC val (953 stars) | Chen val (1000 stars) |
|---|---|---|
| `maew_spec` (ZTFxPC only, 100k) | 0.842 / 0.829 / 0.814 / 0.724 | 0.720 / 0.688 / 0.662 / 0.596 |
| `enc_chenft` (800k then ZTFxPC 30k) | 0.832 / 0.823 / 0.814 / 0.731 | 0.744 / 0.723 / 0.690 / 0.624 |
| `chen_spec` (800k only, 100k) | 0.821 / 0.811 / 0.801 / 0.710 | 0.770 / 0.754 / 0.740 / 0.678 |
| `enc_wholecurve` (one 2000 d window, 50k) | 0.807 / 0.778 / 0.769 / 0.700 | 0.694 / 0.628 / 0.610 / 0.563 |

Reading: 800k pretraining lifts Chen stars by 5 to 8 points and costs 2 points on
ZTFxPC; the post-train gets the ZTFxPC points back and keeps half of the Chen gain;
the whole curve is behind everywhere. The old `maew_spec` figure 0.82 / 0.81 / 0.80 / 0.71
came from a different star set.

## 2. The decoder line (warm start from `maew_spec`, 50k steps, batch 64)

Targets = one whole-record Fourier template per star on the table period
(`project/results/period_table_spec/periods.csv`), 128 extra queries per
window drawn in cycles of the period (log-uniform 5 to 300, a fifth at random
times), random masks, aux 0.5 kept. The probe numbers below are the pooled
read-out; for `cycrope2` the period moved into the spectral peak, so the
pooled number understates it.

| run | date | what is added | within 10 % / 1 % | forecast result | verdict |
|---|---|---|---|---|---|
| `enc_cyc` | 10-11 | cycle queries only | 0.81 / 0.73 | copies the last cycle; reach about 20 cycles | DECODER (control) |
| `enc_cycph` | 10-12 | + phasor features at the spectral-peak period | 0.80 / 0.73 | same as cyc (the phasor map decayed to zero under weight decay) | DEAD (phasor route) |
| `enc_cycrope` | 10-15 | + phase rope (4 heads, soft-peak frequency) | 0.77 / 0.66 | 0.40 vs 0.17 explained at 0.5 to 2 cycles | DECODER |
| **`enc_cycrope2`** | 10-18 | + local-centroid frequency (`--phase-freq local`) | 0.75 / 0.23 (pooled) | 0.83 to 0.94 explained at 20 to 100 cycles for the 56 % of windows with a clean peak | DECODER (current best) |
| `enc_cycropetpl` | 10-18 | the same with the template as input too | 0.73 / 0.16 | no better | DEAD |

Sine tests (`project/sine_test.py`, 10k steps, clean sine at random times,
explained variance at the hidden queries): plain decoder 0.47 (a copier),
phasor 0.00 (collapses on clean data), phase rope 0.99, phase rope + noise
0.91, oracle phase 1.00, real templates with the phase rope 0.97 when the
frequency is clean and 0.02 when it must be estimated. Folders
`project/results/sine_test*` (the 3k-step `sine_test` is superseded by the
10k ones; `sine_test10k_tpl` is the template rung). Verdict: DECODER.

Phase probes (`project/results/phase_*`, 27 folders): does the pooled latent
know where in the cycle the window starts? Every encoder, including the
phase objectives and bottlenecks, sits at chance (median error 0.25 cycles).
Verdict: PAPER (one negative sentence) and the reason the decoder gets the
phase through a rotation, not through the latent.

## 3. Evaluation result folders of the paper's encoder

| folder | what | headline | script |
|---|---|---|---|
| `period_maew_spec` (+ `_spec54`, `_spec_relu`, `_spec_silu2`) | period probe on `maew_spec` latents; 54 = at 54k steps; relu / silu2 = read-out activation | 0.790 / 0.724 | `project/period_probe.py` |
| `period_table_spec` | a period for every star of every split (53,307), model + fine search + candidates | 49 min on one GPU | `project/eval/period_table.py` |
| `period_report_spec4` | the all-star report: model alone 0.867/0.790/0.724/0.212/0.023 within 20/10/1/0.1/0.01 %; + fine search 0.752 at 0.01 %; + candidates 0.799; LS 500k trials 0.179 strict, 0.844 alias-tolerant | the paper's main table | `project/eval/period_report.py` |
| `hit_curves` | the same as curves, strict and alias-tolerant | figure | `project/eval/hit_curves.py` |
| `ls_benchmark_spec100` (+ `_spec54`) | search efficiency: model 7,550 trials at 0.77 within 0.01 % vs Lomb-Scargle 500k trials at 0.19 | figure + table | `project/eval/ls_benchmark.py` |
| `ls_allstars_spec` | the collaborator's astropy search on all validation stars at several budgets | table | `project/eval/ls_allstars.py` |
| `budget_sweep_spec` | model-seeded search over its own budget | figure | `project/eval/budget_sweep.py` |
| `runtime_spec`, `_batched`, `_fp8`, `_int8wo`, `_int8dq`, `runtime_gruode` | ms per star: model whole path 66 ms single, about 13 ms batched; astropy multiband 500k trials 2.2 s | table | `project/eval/runtime.py` |
| `classify_rf_spec` (+ `_spec5`) | random forest on frozen features, 16 classes, test macro F1 0.46 vs hand features 0.34 | table | `project/eval/classify_rf.py` |
| `sens_*` (10 folders) | smallest period change an encoder tells from noise, 591 stars; share resolved at 1 % / 0.1 % per encoder | table | `project/eval/period_sensitivity.py` |
| `period_chen` | model periods for the 800k-star Chen set (712,278 stars with a period); seen periodic classes median fold R2 0.786 vs Chen sharpened 0.788 | `periods.csv` | `project/eval/period_chen.py` |
| `recon_demo_spec`, `recon_demo_hiddensp` | masked-reconstruction pictures | figure | `project/eval/recon_demo.py` |
| `forecast_demo*` | forecast pictures of cyc vs cycrope2 (time view, folded hidden stretch, whole curve folded) | figures | `project/eval/forecast_demo.py` |
| `sine_demo` | the sine test as a picture | figure | `project/eval/sine_demo.py` |
| `slides` | talk figures: raw light curve, by-class bars, fold examples (validation and Chen), per-class scatter | figures | `project/eval/slides_figs.py`, `fold_examples.py`, `scatter_*.py` |
| `period_page` | the period artifact page (https://claude.ai/artifact/JoubE7CVCj2KSB1QtGzwub) | html | `project/eval/period_page.py` |
| `method_page` | the method artifact page (https://claude.ai/artifact/YVvBPfc7XQtBFumbsbTant) | html | `project/eval/method_page.py` |
| `starembed_feats` | empty: the StarEmbed features never ran (x86 wheel on Delta) | nothing | `project/eval/starembed_feats.py` |

Duplicate result folders (same numbers twice): `period_mae20` = `period_mae`;
`period_maew_fold`, `period_maew_refine2`, `period_maew_seq` = `period_maew_fine`
(the seq one adds the sequence read-outs); `period_rope_standard20` =
`period_rope_standard`; `period_maew_spec_silu2` = `period_maew_spec`.
Candidates for removal in step 2, after a look.

## 4. The world-model line (September 26 to October 1), now parked

The design doc's pipeline: frozen autoencoder, a flow or sequence predictor
on cached latents, a decoder on the same latents, energies for inference.
It produced the frozen-encoder recipe and the finding that the latent holds
shape, amplitude and level but not the placement in time, which is what the
decoder line now solves with the phase rope.

| run or folder | what | verdict |
|---|---|---|
| `pred_w250` | flow predictor on `mae_w250` latents, 400k steps, batch 1024 | DEAD (parked line) |
| `pred_maew_seq2`, `pred_mae_seq` | sequence predictors (`--arch seq`) on `maew_w250` / `mae_w250` latents, 400k / 300k | DEAD (parked line) |
| `mae_w250/dec_mse`, `dec_mse_clean`, `maew_w250/dec_mse` | stage-3 decoders on frozen latents | DEAD |
| `mae_w250/lookback` | the lookback decoder trained on the forecast task | DEAD |
| `mae_w250/phase_ceiling(t)`, `phase_ctx(t)` | the phase decoder ladder: with oracle period and phase the latent alone gives skill 0.73 (no latent 0.43) | ABLATION (the "latent holds shape" evidence) |
| `mae_w250/gate.json`, `maew_w250/gate.json` | the gate on frozen latents | DEAD |
| `forecast_maew` | end-to-end brightness forecast with predictor + decoder, 1,847 forecasts, nulls next to it | DEAD (parked line; keep the folder until the decoder paper's forecast section is written) |
| `anomaly_maew` | injected anomalies scored with the predictor's surprise | DEAD |
| `rope_standard/period_head` | the period head (conv net on a Lomb-Scargle spectrum): rejected because its learned part reads the spectrum, not the raw curve | DEAD |
| token head (no folder kept) | per-token read-out: period no finer than the pooled vector | DEAD |
| top-level `runs/` (`lejepa.pt`, `lewm_pc.pt`, `lewm_pc_wm.pt`), `results/pc_period`, `logs/` | the first library runs of 09-23/24 and the period analysis that chose the ladder (`examples/analyze_pc_periods.py`) | DEAD except `results/pc_period/recommendation.json` (the ladder choice; copy it into docs) |

## 5. What the paper needs, in one list

Checkpoints: `maew_spec/mae.pt` (+ `latents100k*.pt`), `enc_cycrope2/mae.pt`,
`enc_cyc/mae.pt`, and for the ablation table `enc_spec`, `enc_specaux`,
`mae_spec50`, `enc_maespec50`, `enc_jepa`, `enc_jepainit`, `enc_gruode`,
`rope_dense`, `rope_standard`, `maew_20k`, `maex_w250`, `maexx_w250`,
`mae_base`, later `chen_spec`. Everything else in `project/runs` can go to
cold storage once step 2 has listed it.

Result folders: `period_maew_spec`, `period_table_spec`, `period_report_spec4`,
`hit_curves`, `ls_benchmark_spec100`, `ls_allstars_spec`, `budget_sweep_spec`,
`runtime_*`, `classify_rf_spec`, `sens_*`, `period_chen`, the `period_enc_*`
and `period_mae*` rows of the tables above, the `phase_*` folders (one
sentence), `sine_test10k*`, `forecast_demo*`, `slides`, `period_page`,
`method_page`.
