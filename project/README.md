# Latent world model for light curves with energy-based inference

An extension of `romae_lc` that implements `../lc-world-model-design.md` on
the ZTF light curves of PC_matches (`/projects/bfrf/data/PC_matches/ZTFxPC`).
Every script is a module run from the repository root:

```bash
source .venv/bin/activate
python -m project.train_wm --help
python -m pytest project/tests -q     # CPU, about a minute
```

## Decisions (2026-09-24)

- Single survey, ZTF, all fine classes (`class_str`, 16 labels; the
  superclass is kept in `record.meta` for grouped summaries). Nothing is
  excluded: LPVs are in, so the first window is 500 d (they need at least one
  period per window).
- The per-point error is a token channel from the start: every token is
  `(m, (log sigma - mu) / sd)`; `--no-err-channel` is the ablation.
- The catalogue period never enters a model. It is a probe target, the
  reference of the period-recovery task, and the knowledge the injected
  anomalies are built from (evaluation side).
- Light models: encoder 192 wide, 6 deep, 3 heads; predictor 3 layers of 4
  heads x 48, MLP 768; decoder 3 layers, 192 wide. 50k steps at batch 64.
- Package change: `FrameConfig(with_err=True)` makes every frame a
  `(t, y, band, err)` quadruple and `collate_frames` passes `err` to the
  tokenizer as `extras` (backward compatible; `tests/test_frames.py`).

## Layout

| file | stage | what |
|---|---|---|
| `common.py` | all | data loading (fine classes, one vocabulary over the splits), error channel, `TokenSpec`, frame helpers, time ladder, light model builders, checkpoints, probes |
| `diagnostics.py` | 1 | time-shuffle score, surprise along window grids |
| `train_wm.py` | 1 (M1) | step-based, resumable LeWorldModel training with the error channel; evals: val losses, class probe, log-period R2, shuffle score, surprise |
| `inject.py` | M2 | injected anomalies (phase, amp, period, bump, color) by template resynthesis; hit rate, object / window AUROC, Delta profiles |
| `residual.py` | 2 (M4) | Gaussian residual per Delta bin (+ optional flow residual), multimodality diagnostic, latent forecast NLL vs persistence |
| `flow.py` | 2, 3 | conditional flow matching (linear interpolant), Euler / midpoint sampler, Hutchinson log density |
| `decoder.py`, `train_decoder.py` | 3 (M5) | `QueryDecoder` (flow or mse kind) on frozen latents; imputation NLL under known errors vs constant / linear / GP baselines and the periodic-GP oracle |
| `baselines.py` | M5 | the per-band baselines |
| `energy.py` | M6, M7 | `E_prior`, `E_dyn` (stage 1 or 2), `E_obs` (stage 3), MAP (Adam), Langevin, periodicity energy |
| `infer.py` | M6, M7 | `calibrate`, `smooth`, `forecast`, `anomaly` (with injection), `period` |
| `tracking.py` | all | Weights & Biases wrapper (`--wandb`), GPU stats, data-wait / compute step timer |
| `bench.py` | 1 | loader throughput benchmark: seconds per step, data-wait fraction, GPU utilisation per configuration |
| `jobs/` | | Slurm scripts; `train_wm.sh` and `train_decoder.sh` are self-resubmitting 2 h chains on `ghx4-interactive` (that QOS runs one job per user and accepts at most two submitted jobs, so the chain retries a refused resubmission for 20 min; keep at most one other job queued next to a chain); `smoke_wandb.sh` is the short tracked run plus the benchmark; `pipeline.sh` runs M4 to M7 from a stage-1 checkpoint |
| `tests/` | | CPU tests on the toy simulator |

Outputs go to `project/runs/<run>/` (checkpoints, `log.jsonl`, `ladder.json`,
`args.json`) and `project/results/`; both are gitignored, as is `project/logs/`.

## Pipeline

```bash
# M1: stage 1 (chain of 2 h jobs; DONE marker when 50k steps are reached, then inject.sh runs)
THEN=project/jobs/inject.sh sbatch --export=ALL,THEN=project/jobs/inject.sh project/jobs/train_wm.sh
# the same with the PC_matches census ladder instead of a measured one
sbatch project/jobs/train_wm.sh --rope-wavelengths 0.01 0.053 0.107 0.218 0.428 0.835 1.64 3.43 6.98 14.3 34.9 6000 --time-scale 0.0015915
# M2: injected anomalies with the raw surprise
sbatch project/jobs/inject.sh project/runs/wm_w500/wm.pt
# M4: residual model (Gaussian; add --flow for level 2)
sbatch project/jobs/residual.sh project/runs/wm_w500/wm.pt project/runs/wm_w500/res
# M5: decoders (mse first, it is the workhorse of E_obs; flow is the design doc's choice)
KIND=mse  sbatch --job-name=dec-mse  project/jobs/train_decoder.sh project/runs/wm_w500/wm.pt --baselines
KIND=flow sbatch --job-name=dec-flow project/jobs/train_decoder.sh project/runs/wm_w500/wm.pt --baselines
# M6: energy weights, smoothing, forecasting, anomaly scores with the full energy
W=project/runs/wm_w500/wm.pt; D=project/runs/wm_w500/dec_mse/dec.pt; R=project/runs/wm_w500/res/residual.pt; O=project/results/infer_w500
sbatch project/jobs/infer.sh calibrate --ckpt $W --decoder $D --residual $R --n-objects 200 --out $O
sbatch project/jobs/infer.sh smooth   --ckpt $W --decoder $D --residual $R --weights $O/weights.json --n-objects 100 --plot 8 --out $O
sbatch project/jobs/infer.sh forecast --ckpt $W --decoder $D --residual $R --weights $O/weights.json --n-objects 100 --out $O
sbatch project/jobs/infer.sh anomaly  --ckpt $W --decoder $D --residual $R --weights $O/weights.json --inject phase --n-objects 200 --out $O
# M7: periodicity energy (LPVs at window 500)
sbatch project/jobs/infer.sh period --ckpt $W --n-objects 500 --out $O
```

A second stage-1 run with a shorter window (finer anomaly localisation, more
windows per object) is one line:
`OUT=project/runs/wm_w60 sbatch --job-name=wm-w60 project/jobs/train_wm.sh --window 60 --min-tokens 8`.

## Advance curriculum

The first run (window 500, advance 1 to 2) learned static per-object
statistics: probe accuracy flat from the first evaluation, log-period R2
about 0.3, shuffle score 0.85, straightness negative. With windows 500 to
1000 days apart the next window's phase is a random draw unless the period
is known to one part in ten thousand, so nothing pushes the latent toward
phase. `--advance-start LO HI` starts the advance range small (the next
window begins inside the current one, so its phase is readable locally) and
ramps it linearly to `--advance` over `--advance-ramp` of the steps; the
checkpoint keeps the final range, the evaluation grids use it throughout.
`train/advance_lo` and `train/advance_hi` show the ramp in wandb, and
`val/logP_r2_by_superclass` the period R2 within each superclass.

## Tracking (Weights & Biases)

`wandb` is the `track` extra (`uv sync --all-extras`); the key lives in
`~/.netrc` (`wandb login`), never in the repository. Every training script
takes `--wandb` (project `lc-world-model`, one run per `--out`, resumed
across job chain links through the run id stored in the checkpoints):

- stage 1: `train/loss`, `train/pred_loss` (the predictor's next-latent
  error), `train/sigreg_loss` (the encoder's Gaussianity regulariser),
  `train/straightness`, the `val/...` metrics of every evaluation;
- decoder: `train/loss` (flow-matching or Gaussian NLL), `val/nll_*` and
  `val/rmse_*` of the decoder and the baselines;
- residual: `train/flow_loss`, the Gaussian and flow NLLs, the forecast NLLs;
- everywhere: `perf/s_per_step`, `perf/data_frac` (fraction of a step spent
  waiting for the loader) and `sys/gpu_util`, `sys/gpu_mem_used_gb`, next to
  wandb's own system panel.

`python -m project.bench` times the loader configurations of
`bench.CONFIGS` (workers, persistent workers, prefetch, pinned memory,
`torch.compile`, batch 128) and prints sequences per second and GPU
utilisation for each; `jobs/smoke_wandb.sh` runs a 600-step tracked stage-1
run followed by the benchmark.

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

## Reading the stage-1 log

`project/runs/<run>/log.jsonl` has one `train` line per 100 steps (loss,
pred, sigreg, straightness, learning rate, seconds per step) and one `eval`
line per 2500 steps: validation `pred` / `sigreg` in eval mode, the linear
probe accuracy on the fine class and the ridge R2 on log period (frozen
backbone features, mean over the windows of one sequence per star), the
time-shuffle score (near 1 = blind to time) and the surprise along the window
grids of validation objects (mean, fraction of valid windows, per superclass).
What "done" looks like is section 4 of the design doc: `pred` down smoothly,
`sigreg` down early then flat, probes useful, shuffle score well below 1,
straightness rising.

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
