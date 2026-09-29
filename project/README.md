# Latent world model for light curves with energy-based inference

An extension of `romae_lc` that implements `../lc-world-model-design.md` on
the ZTF light curves of PC_matches (`/projects/bfrf/data/PC_matches/ZTFxPC`).
Every script is a module run from the repository root:

```bash
source .venv/bin/activate
python -m project.train_predictor --help
python -m pytest project/tests -q     # CPU, a few minutes
```

Stage numbering (since 2026-09-26): **stage 1** is the autoencoder
(`pretrain_mae.py`, `mae.pt`), **stage 2** the predictor on its frozen
latents (`train_predictor.py`; `train_wm.py` is the LeWorldModel variant),
**stage 3** the decoder on the same frozen latents (`train_decoder.py`).
The design doc's "stage 1 = joint encoder + predictor" and "stage 2 =
residual" are superseded (see its revision note and the decisions below).

## Decisions (2026-09-24)

- Single survey, ZTF, all fine classes (`class_str`, 16 labels; the
  superclass is kept in `record.meta` for grouped summaries). Nothing is
  excluded: LPVs are in, so the first window is 500 d (they need at least one
  period per window).
- Added 2026-09-26: the fine classes are the classification target, and
  three of them are anomaly classes rather than classes to name: RRab-Blazhko,
  RRc-Blazhko and EW/EB-OC (`common.ANOMALY_CLASSES`). They stay in the data
  and in every period metric, and are left out of the class probe's fit and
  score (`n_class_excluded` in the probe output) so that no classifier is
  trained to recognise what the anomaly scores are meant to find. A later
  presentation: phase-fold a light curve on the period the model predicts,
  with no catalogue period involved.
- The per-point error is a token channel from the start: every token is
  `(m, (log sigma - mu) / sd)`; `--no-err-channel` is the ablation.
- The catalogue period never enters a model. It is a probe target, the
  reference of the period-recovery task, and the knowledge the injected
  anomalies are built from (evaluation side).
- Light models: encoder 192 wide, 6 deep, 3 heads (`--size light`; `--size
  wide` is 384 wide, 6 heads, 6 deep); predictor 3 layers of 4 heads x 48,
  MLP 768; decoder 3 layers, 192 wide. 50k steps at batch 64.
- Package change: `FrameConfig(with_err=True)` makes every frame a
  `(t, y, band, err)` quadruple and `collate_frames` passes `err` to the
  tokenizer as `extras` (backward compatible; `tests/test_frames.py`).

## Decisions (2026-09-25): the period fix

The first runs (windows 250 and 500 d, 50k steps) collapsed to static
per-object statistics: the class probe never beat the untrained encoder, the
within-class period R2 was about zero (negative for ROT), the shuffle score
stayed at 0.79 and straightness went to -0.35, the value of a fixed point
plus per-window noise. The diagnosis is structural: an eclipsing binary has
700 cycles in a 250 d window and one point per five cycles, so phase can
only come from folding across hundreds of cycles, and a rotary rung folds a
period only if it matches it to about 1 / (4 x cycles), while every head and
layer shared the same 12 rungs. These changes address it, in the order they
matter:

1. **Dense rotary ladder** (`--ladder-mode dense`, the default): one distinct
   measured wavelength per active time angle of every head of every encoder
   layer, dealt so that each head is a narrow band of timescales and each
   layer spans the whole range (`--ladder-deal bands`; `comb` gives every
   head the whole range). The light encoder resolves 378 timescales, the
   wide one 756. Package change: `AxialRope` takes a per-head ladder,
   `RoMAE` a per-layer list of layouts (`rope_layout`, `rope_layers`,
   `rotations`), and `collapse_layout` folds them back for decoders.
2. **Time gets the head**: `--time-frac 0.875` rotates 56 of 64 channels by
   time and 8 by wavelength (ZTF has two bands).
3. **Wider encoder** on demand: `--size wide` (the rung count is set by the
   width, not the head count; more heads only regroup the same rungs).
4. **Reconstruction demand on the encoder**, the objective that needs
   period, phase and shape where next-latent prediction is satisfied by
   static statistics: `pretrain_mae.py` (masked pretraining, the RoMAE
   recipe, on the same windows, ladder and token spec) feeds
   `train_wm.py --init-backbone mae.pt`, and `--recon-weight w` adds a masked
   reconstruction loss (`romae_lc.MaskedDecoder` on the same backbone) to
   every `train_wm` step (the joint ablation of today's stage 2). The LeWM
   ablation against joint reconstruction is about pixel detail the
   predictor cannot forecast; here the detail forced into the latent is
   period and shape, constant per object.
5. **More and denser frames**: 8 windows per sequence, the advance
   curriculum on by default (0.25 to 0.5 window lengths ramping to 1 to 1.5
   over 60% of the steps; `--no-advance-curriculum` for the old behaviour).
   This is for the dynamics collapse, not for period.
6. **Evaluation that cannot be fooled**: macro F1, balanced accuracy and the
   majority-class accuracy next to the probe accuracy (the largest class is
   69% of the validation set); log-period R2 *within* superclasses as the
   headline (`ROT` printed first, per-class values with their counts); a
   step-0 evaluation of the untrained encoder as the reference line; a
   hand-feature baseline probe (shape, scatter and noise statistics, no
   model) printed once; SIGReg evaluated with the projector's BatchNorm on
   batch statistics next to the eval-mode value (`val_sigreg_bn_train`), to
   expose a running-statistics gap.

What would confirm the diagnosis: ROT period R2 turning clearly positive
with (1) and (2) alone; the step-0 and final probes separating; straightness
above zero during the overlap phase; the shuffle score well under 0.79. ECL
and RR may stay low at 250 d windows (700 cycles need thousands of rungs),
which is where a fold inside the model comes back on the table.

## Decisions (2026-09-26): freeze the autoencoder

The runs of 2026-09-25 (`SESSION-2026-09-25.md`, section 4) settled the
question. Masked pretraining alone learns period: `mae_w250` reaches a
within-superclass log-period R2 of 0.53 (ROT 0.58, RR 0.69, ECL 0.81) and a
macro F1 of 0.50 against a hand-feature baseline of 0.27 / 0.33. The joint
world-model objective then erases it: started from that encoder, the
next-latent MSE plus SIGReg with gradients into the encoder falls from
within 0.50 to 0.24 and ROT from 0.42 to -0.21 in 50k steps while its
prediction loss keeps improving and straightness goes to -0.36. A static
per-object latent is the easiest solution of a prediction loss, and the
dense ladder, the window size and the curriculum change nothing under it.

The fix is the recipe of the molecular world-model kit
(`/projects/bfrf/hibb/md-world-model`, `references/model-recipe.md` and
pitfall 7 of `references/pitfalls.md`: "joint encoder + dynamics training
collapses the latent; train the autoencoder first and freeze it"):

- **Autoencoder first, then freeze it.** The reconstruction objective is the
  only one that demands period, phase and shape; once trained, the encoder
  receives no gradient from anything downstream. The latent is shaped for
  reconstruction, not prediction; the predictor gets the capacity and the
  budget to work in that geometry.
- **The transition is a conditional flow in latent space anchored at the
  last latent** (`x_0 = z_H + sigma0 eps`, `x_1 = z_{H+1}`), so it learns
  the change over one step and its exact log density is the stochastic
  transition. There is no separate residual model any more.
- **Training budget beats capacity**: hundreds of thousands of steps on
  latents that were encoded once and live on the GPU (`cache_latents.py`,
  400k steps at batch 1024 in `jobs/train_predictor.sh`), not a wider
  network.
- **A gate before GPU time**: `gate.py` measures, on the frozen latents,
  whether a probe beats the hand features and whether the latent moves more
  under an advance than under a resample of the same window; a failed gate
  stops the pipeline.
- **Every number next to a null**: persistence, the history mean, a
  Gaussian fitted to the training changes, the replicate floor, the GP
  imputation baselines.

The stages are now:

| stage | what | script | checkpoint |
|---|---|---|---|
| 1 | the autoencoder: masked pretraining, or the bottleneck variant whose decoder reads only the pooled latent (`--bottleneck`) | `pretrain_mae.py` | `mae.pt` |
| 1 -> 2 | gate on the frozen latents; the latents cached once over a window grid | `gate.py`, `cache_latents.py` | `gate.json`, `latents.pt` |
| 2 | the predictor on frozen latents: the flow predictor (main path) and its `--kind mse` ablation on the cache; `train_wm.py --freeze-backbone` as the LeWorldModel MSE ablation, `train_wm.py --recon-weight` as the joint ablation (the old stage 1) | `train_predictor.py`, `train_wm.py` | `pred.pt`, `wm.pt` |
| 3 | the decoder on the same frozen latents (`--ckpt` takes `mae.pt` or `wm.pt`) | `train_decoder.py` | `dec.pt` |

The residual stage of the design doc (`residual.py`, M4 / M8) folds into the
flow predictor's log density; `residual.py` stays as the residual of a
`train_wm.py` MSE predictor. The milestones keep their names: M1 is the
predictor (stage 2), M2 the injected anomalies on a stage-2 world model, M3
the error channel (done, stage 1), M4 and M8 the stochastic transition
(stage 2, the flow predictor), M5 the decoder (stage 3), M6 and M7 inference
with the stage 1-3 pieces.

What confirms success, in the order the pipeline produces it: the gate's
effect ratio at the training advance above 1.5 with the probe above the
baseline; `val/mse_ratio` of the predictor well below 1 (persistence is the
null) and `val/mse_ratio_ridge` below 1 (the ridge regression from the
history is the null); `val/nll` below `val/nll_ridge_full` (the ridge
predictor with a full residual covariance; the diagonal Gaussian
`val/nll_persist_gauss` is too weak a null); `val/rollout_norm_drift_h*` near 1 (the chain stays on
the encoder's manifold); `val/sample_std` comparable to `val/true_std` (the
flow has not collapsed to its mean); the stage-3 decoder's imputation NLL
below the RBF GP's.

## Layout

| file | stage | what |
|---|---|---|
| `common.py` | all | data loading (fine classes, one vocabulary over the splits), error channel, `TokenSpec`, frame helpers, time ladder, light model builders, checkpoints (`load_mae`, `load_wm` accepting `mae.pt`, `load_encoder` / `LatentEncoder`: one frozen encoder over both checkpoint kinds), probes |
| `diagnostics.py` | evals | time-shuffle score, surprise along window grids |
| `pretrain_mae.py` | 1 | the autoencoder: masked pretraining of the window encoder (RoMAE recipe) on the stage-2 windows, or the bottleneck autoencoder with `--bottleneck [--bottleneck-loss all\|hidden]`; writes `mae.pt`, the frozen encoder of every later stage (`train_wm.py --init-backbone`, `cache_latents.py`, `gate.py`, `train_decoder.py`) |
| `bottleneck.py` | 1 | `BottleneckAE`: the same RoMAE encoder plus a `QueryDecoder` that reconstructs a window's magnitudes from the pooled latent `z [B, D]` alone (the reconstruction demand acts on the latent the later stages read); `mae.pt` with `kind == "bottleneck"`, loaded like a masked-pretraining one |
| `gate.py` | 1 -> 2 | the gate on frozen latents: probe vs hand-feature baseline, advance effect vs replicate floor, optionally the stage-3 decoder vs the GP; `gate.json`, exit 1 on failure |
| `cache_latents.py` | 1 -> 2 | encodes a window grid (`--stride` window lengths between starts) of every record once with the frozen encoder into `latents.pt` (float16 `[N, D]` plus object / window / start / token-count indices and CSR pointers per split); `--realisations K` adds `K - 1` noise realisations of every window (magnitudes redrawn from the errors, `--real-drop` of the points dropped) as `z_alt [K - 1, N, D]`, which the predictor trains on (a random realisation per row) and reports as the replicate floor `val_replicate_std` next to its own sample spread |
| `train_predictor.py` | 2 (M1, M4, M8) | the flow predictor on cached latents: conditional flow matching anchored at the last latent (`--kind flow`; standardised latents, history of `--history` latents and log advances, residual MLP trunk, Euler sampler, Hutchinson log density), `--kind mse` as the ablation; `--arch seq` reads the whole light curve as a sequence of window latents through a causal transformer (`val_mse_by_history` says whether more history helps); every eval number next to persistence, the history mean and a Gaussian null; resumable, `pred.pt` |
| `train_wm.py` | 2 (M1) | step-based, resumable LeWorldModel training with the error channel, dense ladder, advance curriculum; `--init-backbone mae.pt --freeze-backbone` is the LeWorldModel MSE ablation on the frozen encoder (identity projector, SIGReg off), `--recon-weight` the joint ablation (the old stage 1); evals: val losses next to the persistence and history-mean nulls (`val_pred_ratio`), class probe (accuracy, macro F1, balanced accuracy, majority), log-period R2 within and per superclass, shuffle score, surprise; hand-feature baseline and step-0 probe |
| `inject.py` | M2 | injected anomalies (phase, amp, period, bump, color) by template resynthesis on a stage-2 `wm.pt`; hit rate, object / window AUROC, Delta profiles |
| `residual.py` | 2 (M4, LeWorldModel path) | Gaussian residual of a `train_wm.py` MSE predictor per Delta bin (+ optional flow residual), multimodality diagnostic, latent forecast NLL vs persistence; the flow predictor replaces it in the frozen pipeline |
| `flow.py` | 2, 3 | conditional flow matching (linear interpolant), Euler / midpoint sampler, Hutchinson log density, time embedding |
| `decoder.py`, `train_decoder.py` | 3 (M5) | `QueryDecoder` (flow or mse kind) on frozen latents of `mae.pt` or `wm.pt`; imputation NLL under known errors vs constant / linear / GP baselines and the periodic-GP oracle |
| `baselines.py` | M5 | the per-band baselines |
| `energy.py` | M6, M7 | `E_prior`, `E_dyn` (the `train_wm.py` predictor, or the `residual.py` Gaussian), `E_obs` (stage 3), MAP (Adam), Langevin, periodicity energy |
| `infer.py` | M6, M7 | `calibrate`, `smooth`, `forecast`, `anomaly` (with injection), `period` on a stage-2 `wm.pt` plus the stage-3 decoder |
| `tracking.py` | all | Weights & Biases wrapper (`--wandb`), GPU stats, data-wait / compute step timer |
| `bench.py` | 2 | loader throughput benchmark of `train_wm`: seconds per step, data-wait fraction, GPU utilisation per configuration |
| `jobs/` | | Slurm scripts on `ghx4-interactive`: `pretrain_mae.sh` (stage 1), `train_wm.sh`, `train_predictor.sh` and `train_decoder.sh` (stages 2 and 3) are self-resubmitting 2 h chains (`THEN=script` runs one stage after a chain, `PIPELINE=a.sh:b.sh` several, each chain script popping the first and handing its successor its checkpoint as the first argument: `pretrain_mae.sh` hands `$OUT/mae.pt`, which `train_wm.sh` turns into `--init-backbone` (OUT then defaults to `wm_w<W>_mae`) and `gate.sh`, `cache_latents.sh` and `train_decoder.sh` take positionally; `train_wm.sh` hands `$OUT/wm.pt`, `train_predictor.sh` `$OUT/pred.pt`); `cache_latents.sh` (1 h, `THEN=` gets the latents path) and `gate.sh` (30 min, `THEN=` runs only on a pass) are single jobs. That QOS runs one job per user and accepts at most two submitted jobs, so a chain retries a refused resubmission every 30 s for 20 min (`queue` in the chain scripts) before it gives up with a log line; keep at most one other job queued next to a chain. `train_wm_w60.sh` is the window-60 wrapper, `smoke_wandb.sh` the short tracked run plus the benchmark, `pipeline.sh` runs M4 to M7 from a `train_wm.py` checkpoint |
| `tests/` | | CPU tests on the toy simulator (`test_project.py`, `test_predictor.py`, `test_gate.py`, `test_bottleneck.py`, `test_encoder_loading.py`) |

Outputs go to `project/runs/<run>/` (checkpoints, `log.jsonl`, `ladder.json`,
`args.json`, `latents.pt`, `gate.json`) and `project/results/`; both are
gitignored, as is `project/logs/`.

## Pipeline

The frozen-autoencoder path (every flag below is the one the job script
takes; the scripts add the run-length, loader and `--wandb` arguments):

```bash
# Stage 1: the autoencoder (chain of 2 h jobs, 50k steps, window 250, dense ladder)
sbatch project/jobs/pretrain_mae.sh                                          # -> project/runs/mae_w250/mae.pt
# the bottleneck variant (the decoder reads only the pooled latent)
OUT=project/runs/bn_w250 sbatch --job-name=bn-w250 project/jobs/pretrain_mae.sh --bottleneck
# Gate on the frozen latents (30 min; exit 1 and no successor when it fails)
sbatch project/jobs/gate.sh project/runs/mae_w250/mae.pt                    # -> project/runs/mae_w250/gate.json
# Cache the latents once (1 h; --stride 0.5 for a finer grid)
sbatch project/jobs/cache_latents.sh project/runs/mae_w250/mae.pt           # -> project/runs/mae_w250/latents.pt
sbatch project/jobs/cache_latents.sh project/runs/bn_w250/mae.pt --realisations 4   # with 3 noise realisations per window
sbatch project/jobs/pretrain_mae_wide.sh  # the wide plain MAE (maew_w250, 756 rungs, 150k steps), a chain
# the bottleneck autoencoder (pretrain_mae_bn.sh, --bottleneck) is kept as an ablation only: with a learned,
# a known or a unit variance its pooled-only decoder never learned period (bn_w250, bnkv_w250, bnu_w250,
# within-class R2 0.24-0.27 against the plain MAE's 0.53), so the token-level MAE is the stage-1 encoder
# the whole frozen pipeline from one submission: realisation cache -> predictor -> wide MAE
sbatch --job-name=cache-r4 --export=ALL,LATENTS=project/runs/mae_w250/latents_r4.pt,THEN=project/jobs/train_predictor.sh,THEN_OUT=project/runs/pred_w250_r4,THEN_PIPELINE=project/jobs/pretrain_mae_wide.sh project/jobs/cache_latents.sh project/runs/mae_w250/mae.pt --realisations 4
# Stage 2: the flow predictor on the cache (chain, 400k steps at batch 1024)
LATENTS=project/runs/mae_w250/latents.pt OUT=project/runs/pred_w250 sbatch --job-name=pred project/jobs/train_predictor.sh
# the same cache and predictor with the MSE objective (ablation)
OUT=project/runs/pred_w250_mse sbatch --job-name=pred-mse project/jobs/train_predictor.sh project/runs/mae_w250/latents.pt --kind mse
# cache, then predictor, as one submission (the cache job submits the chain with the latents path)
THEN=project/jobs/train_predictor.sh sbatch --export=ALL,THEN=project/jobs/train_predictor.sh project/jobs/cache_latents.sh project/runs/mae_w250/mae.pt
# Stage 2, LeWorldModel MSE ablation on the frozen encoder (identity projector, SIGReg off)
OUT=project/runs/wm_w250_frozen sbatch --job-name=wm-frozen project/jobs/train_wm.sh --init-backbone project/runs/mae_w250/mae.pt --freeze-backbone
# Stage 2, joint ablation (the old stage 1: the encoder trains under the prediction loss plus reconstruction)
OUT=project/runs/wm_w250_joint sbatch --job-name=wm-joint project/jobs/train_wm.sh --init-backbone project/runs/mae_w250/mae.pt --recon-weight 0.5
# Stage 3: decoders on the frozen MAE latents (load_wm wraps mae.pt; output next to the checkpoint, dec_<KIND>)
KIND=mse  sbatch --job-name=dec-mse  project/jobs/train_decoder.sh project/runs/mae_w250/mae.pt --baselines
KIND=flow sbatch --job-name=dec-flow project/jobs/train_decoder.sh project/runs/mae_w250/mae.pt --baselines
# the gate again with its third part (decoder vs GP) once a decoder result exists
sbatch project/jobs/gate.sh project/runs/mae_w250/mae.pt --decoder-results project/runs/mae_w250/dec_mse/log.jsonl
```

`train_predictor.sh` expects a `latents.pt` as its first argument, so it
follows `cache_latents.sh`, never `pretrain_mae.sh` directly (which hands
its successor `mae.pt`; `THEN=project/jobs/cache_latents.sh` or
`THEN=project/jobs/gate.sh` on the stage-1 chain works). `gate.sh` with
`THEN=` submits that script with the checkpoint only on a pass; neither
`gate.sh` nor `cache_latents.sh` reads `PIPELINE`, so a longer pipeline
stops at them.

The joint LeWorldModel runs (the old stage 1) and the milestones that
consume a `train_wm.py` checkpoint, unchanged:

```bash
# The runs that tested the period fix (window 250, dense ladder, curriculum, 8 frames), 2026-09-25:
# (a) joint LeWorldModel training alone
OUT=project/runs/wm_w250_dense sbatch --job-name=wm-dense project/jobs/train_wm.sh
# (b) the autoencoder, then joint training from its encoder (the chain submits train_wm.sh itself)
THEN=project/jobs/train_wm.sh sbatch --export=ALL,THEN=project/jobs/train_wm.sh project/jobs/pretrain_mae.sh
# (a) then (b) as one submission: PIPELINE lists the stages a chain starts one after the other
# (the QOS holds one running and one queued job, which a chain already uses)
sbatch --job-name=wm-dense --export=ALL,OUT=project/runs/wm_w250_dense,PIPELINE=project/jobs/pretrain_mae.sh:project/jobs/train_wm.sh project/jobs/train_wm.sh
# (c) window 60 with a log ladder over 0.15 - 120 d (train_wm_w60.sh): the sharpest period test for
# rotation-type stars (14 cycles per window, which a 2% ladder resolves; the 250 d census ladder
# spends 264 of 378 rungs below one day). The submission of 2026-09-25 chained (a), (b) and (c):
sbatch --job-name=wm-dense --export=ALL,OUT=project/runs/wm_w250_dense,PIPELINE=project/jobs/pretrain_mae.sh:project/jobs/train_wm.sh:project/jobs/train_wm_w60.sh project/jobs/train_wm.sh
# variants: --size wide (756 rungs), --ladder-mode shared (ablation)
OUT=project/runs/wm_w250_wide sbatch --job-name=wm-wide project/jobs/train_wm.sh --size wide
# M1 on the joint path (chain of 2 h jobs; DONE marker when 50k steps are reached, then inject.sh runs)
THEN=project/jobs/inject.sh sbatch --export=ALL,THEN=project/jobs/inject.sh project/jobs/train_wm.sh
# the same with the PC_matches census ladder (shared over heads and layers) instead of a measured one
sbatch project/jobs/train_wm.sh --rope-wavelengths 0.01 0.053 0.107 0.218 0.428 0.835 1.64 3.43 6.98 14.3 34.9 6000 --time-scale 0.0015915
# M2: injected anomalies with the raw surprise of a stage-2 world model
sbatch project/jobs/inject.sh project/runs/wm_w250/wm.pt
# M4 on the LeWorldModel path: residual of its MSE predictor (Gaussian; add --flow for level 2)
sbatch project/jobs/residual.sh project/runs/wm_w250/wm.pt project/runs/wm_w250/res
# M5: decoders on a world model's latents (mse first, it is the workhorse of E_obs; flow is the design doc's choice)
KIND=mse  sbatch --job-name=dec-mse  project/jobs/train_decoder.sh project/runs/wm_w250/wm.pt --baselines
KIND=flow sbatch --job-name=dec-flow project/jobs/train_decoder.sh project/runs/wm_w250/wm.pt --baselines
# M6: energy weights, smoothing, forecasting, anomaly scores with the full energy
W=project/runs/wm_w250/wm.pt; D=project/runs/wm_w250/dec_mse/dec.pt; R=project/runs/wm_w250/res/residual.pt; O=project/results/infer_w250
sbatch project/jobs/infer.sh calibrate --ckpt $W --decoder $D --residual $R --n-objects 200 --out $O
sbatch project/jobs/infer.sh smooth   --ckpt $W --decoder $D --residual $R --weights $O/weights.json --n-objects 100 --plot 8 --out $O
sbatch project/jobs/infer.sh forecast --ckpt $W --decoder $D --residual $R --weights $O/weights.json --n-objects 100 --out $O
sbatch project/jobs/infer.sh anomaly  --ckpt $W --decoder $D --residual $R --weights $O/weights.json --inject phase --n-objects 200 --out $O
# M7: periodicity energy (LPVs at window 500)
sbatch project/jobs/infer.sh period --ckpt $W --n-objects 500 --out $O
```

`E_dyn` in `energy.py` and `infer.py` is still the LeWorldModel predictor of
a `wm.pt` (jointly trained or `--freeze-backbone`) or the `residual.py`
Gaussian; the flow predictor's `pred.pt` is not wired into the energies yet,
so M2, M6 and M7 run on a `train_wm.py` checkpoint for now.

`train_wm.sh` takes `WINDOW` (default 250) and `OUT` (default
`project/runs/wm_w$WINDOW`); a `train_wm` run with a shorter window (finer
anomaly localisation, more windows per object) is one line:
`WINDOW=60 sbatch --job-name=wm-w60 project/jobs/train_wm.sh --min-tokens 8`.

## Ladder, frames and curriculum

`ladder.json` in the run directory holds the ladder: `timescales` (position
units) and `wavelengths` (days) nested `[layers][heads][angles]` for a dense
ladder, flat for a shared one, plus `layers`, `heads` and `deal`. A run
prints the number of distinct wavelengths and their range at the start.
`--ladder path/ladder.json` reuses one (it must fit the model geometry).

The first run (window 500, advance 1 to 2) learned static per-object
statistics (see the decisions above). With windows 500 to 1000 days apart
the next window's phase is a random draw unless the period is known to one
part in ten thousand, so nothing pushes the latent toward phase.
`--advance-start LO HI` (default 0.25 0.5 in `train_wm`) starts the advance
range small (the next window begins inside the current one, so its content
is readable locally) and ramps it linearly to `--advance` (default 1 1.5)
over `--advance-ramp` of the steps; the checkpoint keeps the final range,
the evaluation grids use it throughout, and the records are filtered for
the final range (a record needs `window * (1 + 7 * advance_lo)` days for the
default 8 frames: 2000 d at window 250, which 99% of the ZTF records have).
`train_predictor` has the same curriculum (`--advance-start`, off by
default) on top of its latent grid: an advance is rounded to a whole number
of `--stride` steps (at least one), and the model sees the realised value.
`train/advance_lo` and `train/advance_hi` show the ramp in wandb,
`val/logP_r2_within` the headline period metric and
`val/logP_r2_by_superclass` the period R2 within each superclass.

## Tracking (Weights & Biases)

`wandb` is the `track` extra (`uv sync --all-extras`); the key lives in
`~/.netrc` (`wandb login`), never in the repository. Every training script
takes `--wandb` (project `lc-world-model`, one run per `--out`, resumed
across job chain links through the run id stored in the checkpoints):

- stage 1 (`pretrain_mae`): `train/loss` (masked or bottleneck
  reconstruction), `val/loss`, the probe metrics;
- stage 2, flow predictor (`train_predictor`): `train/loss`, `train/lr`,
  `train/advance_lo`, `train/advance_hi`, `perf/s_per_step`,
  `perf/seq_per_s`, and every eval key without its `val_` prefix
  (`val/mse`, `val/mse_persist`, `val/mse_histmean`, `val/mse_ratio`,
  `val/nll`, `val/nll_persist_gauss`, `val/rollout_mse_h*`,
  `val/rollout_persist_h*`, `val/rollout_norm_drift_h*`, `val/sample_std`,
  `val/true_std`; with `--arch seq` also `val/mse_by_history/<bin>/mse`,
  `.../persist` and `.../n`), the final values in the run summary;
- stage 2, LeWorldModel (`train_wm`): `train/loss`, `train/pred_loss` (the
  predictor's next-latent error), `train/sigreg_loss` (the encoder's
  Gaussianity regulariser), `train/straightness`, `train/recon_loss` (with
  `--recon-weight`), the `val/...` metrics of every evaluation including
  `val/pred_persist`, `val/pred_histmean` and `val/pred_ratio`, the
  hand-feature baseline as `baseline_...` in the run summary;
- gate: with `--wandb` the headline numbers go to a run summary (job type
  `gate`);
- decoder: `train/loss` (flow-matching or Gaussian NLL), `val/nll_*` and
  `val/rmse_*` of the decoder and the baselines;
- residual: `train/flow_loss`, the Gaussian and flow NLLs, the forecast NLLs;
- everywhere: `perf/s_per_step`, `perf/data_frac` (fraction of a step spent
  waiting for the loader, where there is a loader) and `sys/gpu_util`,
  `sys/gpu_mem_used_gb`, next to wandb's own system panel.

`python -m project.bench` times the loader configurations of
`bench.CONFIGS` (workers, persistent workers, prefetch, pinned memory,
`torch.compile`, batch 128) and prints sequences per second and GPU
utilisation for each; `jobs/smoke_wandb.sh` runs a 600-step tracked
`train_wm` run followed by the benchmark.

## GPU utilisation

The first benchmark (window 500, batch 64, light model) showed the loader is
not the bottleneck: 4% of a step waits for data with 8 workers and every
loader setting (16 workers, persistent workers, prefetch 4, pinned memory)
lands at 0.10 s per step with the GPU at 33 to 35%. The step is bound by
kernel-launch overhead: the package encodes the four frames of a sequence in
four backbone calls. `common.fused_encode` (installed on every model built
or loaded by the project) pads the frames to a common length and runs the
backbone once, which also speeds up every evaluation that encodes window
grids. `torch.compile` is not usable on this node (Triton cannot build its
CUDA helper on aarch64). Measured at window 250, batch 64: per-frame encoder
0.111 s per step at 30% GPU; fused encoder 0.043 s at 58%; fused plus pinned
memory 0.041 s at 62% (the default of the job scripts); batch 128 would reach
85% at 2,366 sequences per second but batch 64 is kept because it was asked for.
The stage-2 flow predictor has no loader at all: the cache lives on the GPU
and a step is one batch of index gathers plus the MLP.

## Reading the stage-2 predictor log

`project/runs/<run>/log.jsonl` of `train_predictor` has one `train` line
per `--log-every` steps (`loss`, `lr`, `advance_lo`, `advance_hi`,
`s_per_step`) and one `eval` line at step 0 and every `--eval-every` steps
on a fixed, seeded set of `--val-sequences` validation sequences drawn at
the final advance range. Every model number sits next to its null, computed
on the same sequences:

- `val_mse` (one-step error in standardised latent units) next to
  `val_mse_persist` (predict `z_H`), `val_mse_histmean` (predict the mean
  of the history) and `val_mse_ridge` (a ridge regression of the change on
  the history and the advances, fitted to 50k training sequences: the
  strongest cheap predictor); `val_mse_ratio = val_mse / val_mse_persist`
  and `val_mse_ratio_ridge = val_mse / val_mse_ridge`. Persistence is what
  a static per-object latent makes trivially true, so a ratio near 1 means
  the predictor learned nothing beyond it; the ridge ratio is the one that
  says whether the flow does more than a linear model.
- `val_nll` (flow only: the log density per dimension, standardised units,
  from the reverse Euler flow with `--nll-steps` steps and a Hutchinson
  divergence) next to three nulls fitted to the same 50k training
  sequences: `val_nll_persist_gauss`, a diagonal Gaussian `N(z_H + m, diag
  v)`; `val_nll_persist_full`, the same with the full covariance; and
  `val_nll_ridge_full`, the ridge predictor with the full covariance of its
  residual. The diagonal null is far too weak when the change lives in a
  low-dimensional subspace of the latent (2026-09-26 run: +0.84 diagonal,
  -0.85 full, -0.96 ridge, flow -1.81 nats per dimension); read the flow
  against the ridge null. The linearised log-determinant of the reverse
  Euler flow is an upper bound on the log density that tightens with the
  step count (20 steps overstated it by 0.13, 100 by 0.04, checked against
  the exact Jacobian by `project/check_nll.py`). The `mse` kind reports NaN.
- `val_rollout_mse_h{h}` next to `val_rollout_persist_h{h}` for `h` up to
  `--eval-horizons`, repeating the last advance, and
  `val_rollout_norm_drift_h{h}`: the norm of the rolled latent over the norm
  of the true one. Near 1 the chain stays on the encoder's manifold; a
  drift away from 1 is the sampler leaving it (`--reproject norm` is the
  sphere-style fix, off by default).
- `val_sample_std` next to `val_true_std`: the spread of the flow's samples
  against the spread of the true changes. A flow whose samples do not
  spread has collapsed to its mean.

`pred.pt` and `DONE` appear at `--steps`; `last.pt` carries the run across
links. `load_predictor(path)` gives the model in eval mode and its meta
(kind, arch, hparams, the latent meta with the encoder checkpoint, args,
step, metrics).

### The sequence predictor (`--arch seq`)

`--arch mlp` (the default, everything above) sees a fixed history of
`--history` latents. `--arch seq` reads the whole light curve of a star as
a sequence of window latents: every valid window of the object in window
order, any number of them, so a star with a longer baseline gives more
frames. A causal transformer (`romae_lc.ARPredictor`, `--seq-hidden`,
`--seq-depth`, `--seq-heads`, `--seq-dim-head`, `--seq-mlp`,
`--seq-dropout`; each window is conditioned on its gap from the previous
one, in window units) summarises the past of every position into a state,
and the same flow head as above (`--hidden`, `--depth`), given the state
and the next gap, predicts the next latent anchored at the current one
(`--kind mse` is the same ablation). A training step draws `--batch-size`
objects (default 64), keeps each valid window with probability
`--seq-keep` (default 0.5, so the gaps vary and sparse curves are seen; at
least 3 windows are kept) and crops to `--max-len` windows (default 64) at
random. The evaluation scores the full sequences of `--val-objects`
validation objects (cropped to the last `--max-len` windows) at every
position with at least 3 real windows before it, so it reports the same
keys as the `mlp` evaluation (the nulls use the previous 3 latents and
their gaps; the rollouts start from a random position and follow the true
gaps; `val_sequences` counts the scored positions), plus
`val_mse_by_history`: `mse`, `persist` and `n` in bins of the number of
windows before the scored one (`3-5`, `6-10`, `11-20`, `21+`). That is the
number that says whether more history helps: the `mse` (and its ratio to
`persist`) should fall from bin to bin. Old `pred.pt` files have no `arch`
and load as `mlp`. `model.states(z, gaps, mask)` gives the states `[B, T,
hidden]` of a batch of sequences (`LatentStore.draw_sequences`), the input
of anything that reads a whole light curve, such as a period regression.

## Gate

`python -m project.gate --ckpt mae.pt|wm.pt` (or `jobs/gate.sh`) writes
`gate.json` next to the checkpoint with three parts, each with `status`,
`passed` and its `criterion`, plus the overall `passed` (over the parts that
ran) and the `verdict` printed last (`GATE PASSED` or `GATE FAILED: part N
...`; exit 1):

1. **Probe vs baseline** (hard gate; `--skip-probe` skips it): `probe`
   (accuracy, macro F1, balanced accuracy, majority accuracy, pooled and
   within-superclass log-period R2, per superclass) on the frozen latents
   and `baseline` (the hand features) on exactly the same records,
   `shuffle_score` reported but not gated. Passes when `r2_within` and
   `macro_f1` both beat the baseline.
2. **Advance effect vs replicate floor** (hard gate): for `--n-objects`
   validation objects the latent of a reference window, of the same window
   shifted by every advance of `--advances` (window units) and of two
   independent random drops of `--drop` of its points (the floor). Per
   advance: `ratio_median` (median effect over the median floor),
   `ratio_mean`, `effect_median`, `cosine_median`, `ratio_by_superclass`,
   counts and skipped windows; `floor_median`, `floor_by_superclass`,
   `suggested_advance_range` (first advance with the ratio above
   `--min-effect` up to the last with median cosine at least 0.5). Passes
   when the ratio at `--train-advance` (default 1.0, the smallest advance of
   the stage-2 range) is at least `--min-effect` (default 1.5).
3. **Decoder vs GP** (only with `--decoder-results`, a `train_decoder`
   `log.jsonl`, `dec.pt` / `last.pt` or json): `nll` and `rmse` of the
   decoder, the linear, constant and RBF-GP baselines (and the periodic GP
   when present); passes when the decoder's NLL beats the RBF GP; a result
   without the GP baseline (a run without `--baselines`) fails with a note.

The reference numbers of `SESSION-2026-09-25.md` say what to expect: the
`mae_w250` latents (within 0.53 vs 0.27, macro F1 0.50 vs 0.33) pass part
1, every jointly trained `wm.pt` fails it.

## Reading the train_wm log

`project/runs/<run>/log.jsonl` of `train_wm` has one `baseline` line (the
hand-feature probe), one `train` line per 100 steps (loss, pred, sigreg,
straightness, recon, learning rate, seconds per step) and one `eval` line
at step 0 and per 2500 steps: validation `pred` next to `val_pred_persist`,
`val_pred_histmean` and `val_pred_ratio` (persistence and history-mean
nulls on the same batches; a ratio near 1 means nothing beyond persistence
was learned), `sigreg` in eval mode and `val_sigreg_bn_train` (the
projector's BatchNorm on batch statistics), the linear probe on the fine
class (`probe_acc`, `probe_macro_f1`, `probe_balanced_acc`,
`probe_majority_acc` to beat; frozen backbone features, mean over the
windows of one sequence per star), the ridge R2 on log period
(`logP_r2_within` the headline, `logP_r2` pooled, `logP_r2_by_superclass`
with `n_by_superclass`), the time-shuffle score (near 1 = blind to time)
and the surprise along the window grids of validation objects (mean,
fraction of valid windows, per superclass). The step-0 line is the
untrained encoder (or, with `--init-backbone`, the stage-1 encoder as it
was): an evaluation that does not separate from it has not learned anything
the probe can use, and with `--freeze-backbone` the probe numbers do not
move at all, only `pred` and its ratio do. On the joint path, what "done"
looked like was section 4 of the design doc (`pred` down smoothly, `sigreg`
down early then flat, probes above the baseline and the step-0 line,
shuffle score well below 1, straightness rising); the runs of 2026-09-25
showed the probes falling instead, which is why the encoder is frozen now.

## Caveats carried from the design review

- Every usable ZTF window holds many cycles of the typical 0.35 d object, so
  phase at the next window is predictable only through a precise period; the
  injected phase jump (M2) is the test of that hypothesis.
- With 500 d windows an object has about 5 windows, so anomaly localisation
  is coarse and the forecast evaluation is limited to one window ahead
  (`residual.py` says so when it drops horizons). The 60 d run is the
  complement.
- The periodicity energy lives on a per-window path: only periods comparable
  to the window spacing (LPVs) are resolvable with it.
- `E_dyn`, `E_obs` and `E_prior` are per-window and O(1) by construction, but
  their relative weights remain a real hyperparameter; `infer.py calibrate`
  gives the equal-magnitude starting point.
