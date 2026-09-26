# Latent World Model for Light Curves with Energy-Based Inference

> **Revision 2026-09-26.** Sections 4 and 5 below (stage 1 = encoder and
> predictor trained jointly on next-latent MSE + SIGReg, stage 2 = a residual
> model on top of it) are superseded by the frozen-autoencoder pipeline of
> `project/README.md` ("Decisions (2026-09-26)"). Reason: started from a
> masked-pretrained encoder that had learned period (within-superclass
> log-period R2 0.53), the joint objective erased it in 50k steps (0.50 to
> 0.24, ROT 0.42 to -0.21) while its prediction loss kept improving, because a
> static per-object latent is the easiest solution of a prediction loss with
> gradients into the encoder (`project/SESSION-2026-09-25.md`). The stages are
> now: 1 the autoencoder (`project/pretrain_mae.py`, frozen afterwards); 2 a
> conditional flow-matching predictor on cached frozen latents, anchored at the
> last latent (`project/train_predictor.py`; its log density replaces the
> residual model of section 5); 3 the decoder of section 6
> (`project/train_decoder.py`) on the same frozen latents. Sections 6 to 8 and
> the milestones keep their content; their stage numbers are the old ones.

Design doc, v0.1. Single survey (ZTF via PC_matches `ZTFxPC`), built on
[`romae-lc`](https://github.com/Hibb-bb/romae-lc).

---

## 1. Goal

Learn a latent state-space model of light curves in three parts:

1. An encoder that maps a time window of observations to a latent `z`.
2. A predictor that maps `(z_t, Δt)` to `z_{t+Δt}`.
3. A decoder that maps `z` back to clean magnitudes.

Then use the learned pieces as **energy terms** and do inference by
optimization or Langevin sampling over latent paths. Target uses:

- Fill gaps with uncertainty (smoothing).
- Forecast with a spread of futures.
- Score anomalies per window.
- Detect period and shape changes by adding a periodicity energy.
- Later: infer physical parameters, compose with physical templates.

Out of scope for v0.1: multi-survey composition, follow-up planning, transients
with strong intrinsic randomness. The design leaves room for all three.

---

## 2. Model

### 2.1 Notation

One object has observations `{(t_i, m_i, σ_i, b_i)}`: time (days), magnitude,
reported error, band. We assume

```
m_i = μ(t_i, b_i) + σ_i ε_i,   ε_i ~ N(0, 1)
```

with `μ` the unknown clean magnitude. A **window** `x_τ` is the set of points
in `[τ, τ + W]`, time re-zeroed at `τ`.

### 2.2 Generative model

```
z_1 ~ N(0, I)
z_{k+1} ~ p_θ(z | z_k, Δ_k)          # transition
x_k    ~ p_ψ(x | z_k)                # emission
```

Path energy (negative log of the joint):

```
E(z_1:K) = ½‖z_1‖²
         + Σ_k E_dyn(z_k, z_{k+1}, Δ_k)
         + Σ_k E_obs(z_k, x_k)
```

Every inference procedure in §6 is optimization or sampling on this energy
with terms added or removed.

### 2.3 Components and how they map to `romae_lc`

| Part | Role | `romae_lc` object | Status |
|---|---|---|---|
| Encoder `E_φ` | window → `z` | `RoMAE` backbone + `LeWorldModel.projector` | exists |
| Predictor `P_θ` | `(z, Δ)` → `ẑ'` | `LeWorldModel.predictor` (`ARPredictor`, AdaLN-zero on the action) | exists |
| Action | `Δ` in window units | `FrameConfig.advance`, `collate_frames` | exists |
| Anti-collapse | SIGReg on latents | `LeWorldModel(lamb=...)` | exists |
| Residual model `p(r \| z, Δ)` | spread around `P_θ` | new | §4 |
| Decoder `D_ψ` | `z` → `μ(t, b)` | new (flow matching) | §5 |
| Error input | `σ_i` as token feature | extension to `tokenize` | §3.3 |

The action in the package is the advance to the next window in window units.
That is exactly `Δt` conditioning. Nothing else is needed on the action side.

---

## 3. Data (single survey: ZTF)

### 3.0 Why one survey, and why ZTF

One survey first, because the composition story (§7.5) only makes sense once
each single-survey energy is trustworthy on its own. Mixing surveys at stage 1
also mixes cadences, depths and error models, which confounds the window and
time-ladder choices you are trying to settle.

ZTF over the others in PC_matches, for four reasons:

- Two bands (g, r) on a regular multi-day cadence, with seasonal gaps. That is
  the hard, realistic case for the `Δ`-conditioned predictor, and it has
  enough points per window to pin phase.
- Multi-year baseline, so slow period changes and Blazhko-scale modulation are
  in reach within one survey.
- Per-point errors are reported and well behaved, which the emission model in
  §6 depends on.
- The package already has the loader, the period census and a Slurm job for
  it (`jobs/train_lewm_pc.sh`, `results/pc_period/`).

What ZTF does not give you: dense sub-day sampling. Short-period objects
(δ Scuti, some contact binaries) will be at the edge of the rotary ladder.
Either exclude them for v0.1 or accept that the encoder will see them as
period/amplitude only. TESS is the natural second survey for that regime, and
Gaia for sparse-cadence stress tests.

### 3.1 Loading

```python
from romae_lc import PC_BANDS, load_pc, normalize, wavelengths_for

root = "/projects/bfrf/data/PC_matches/ZTFxPC"
train = [normalize(r) for r in load_pc(root, "train")]
val   = [normalize(r) for r in load_pc(root, "val")]
wavelengths = wavelengths_for(PC_BANDS)
```

Notes from the package:
- Values are `-mag` (brighter is up). `normalize` standardizes per band by
  median/MAD.
- Unclean and non-finite points are dropped.
- Times are MJD re-zeroed at each record's first point.
- Labels are the 8 `PC_SUPERCLASSES` by default.

### 3.2 Filters for v0.1

- Keep records with at least `--min-points 8` (package default) in total and
  at least `min_tokens` per window (see §3.4).
- Start with periodic superclasses only. Transients come later.
- Hold out a fixed set of objects for validation. Never split one object across
  train and val.

### 3.3 Feeding the errors to the encoder (extension)

`tokenize` currently takes `(times, values, bands)` and builds
`values [B, N, 1]`. The encoder never sees `σ_i`. It should, so it can
down-weight noisy points.

Smallest change: make the token value 2-channel, `(m_i, log σ_i)`, with
`σ` standardized per band the same way `normalize` handles `y`.

```python
# sketch; check the tokenize signature before editing
values = [torch.stack([torch.from_numpy(r.y), torch.log(torch.from_numpy(r.err))], -1)
          for r in records]           # each [n, 2]
tokens = tokenize(times, values, bands, band_wavelengths=wavelengths, time_scale=ts)
# tokens.values -> [B, N, 2]; the input projection of RoMAE must accept 2 channels
```

If that is more than a one-line change in `RoMAE`, a fallback is to leave the
encoder as is for stage 1 and use `σ` only in the decoder loss (§5). The
encoder can be upgraded later without changing anything downstream.

### 3.4 Window and frame settings

`FrameConfig(n_frames, window, advance, min_tokens, max_tokens)` controls the
sampler. Decisions:

- **`window`.** The package docstring says it: if the window is many periods
  long, the next window's phase is unpredictable and the model learns
  period/shape/amplitude, not phase. For v0.1 we want phase, so set `window`
  to a few periods of the typical object. Start with `window = 30` days (the
  example default) and sweep `{10, 30, 90}`. Expect a per-class sweet spot;
  a mixed dataset may need the long window.
- **`advance`.** `(1.0, 2.0)` gives non-overlapping windows at most one window
  apart. Keep it. This is what makes `Δ` a real variable. Also add draws with
  larger advance (`(1.0, 6.0)`) in a second run so the predictor has seen
  seasonal gaps.
- **`n_frames = 4`, `history = 3`.** Package defaults; matches the paper.
- **`min_tokens`.** Set from the ZTF cadence census in `results/pc_period/`.
  Windows below it are re-drawn during training and masked at eval.
- **`use_cls=True`.** Required. It is what anchors a window's phase.

### 3.5 Time unit

Run `suggest_time_encoding` on the training curves (or `--auto-time` in the
script). With `--auto-time` the ladder's `lam_max` is `2 * window`. Prefer the
quantile ladder built from the ZTF period census (`--time-spacing quantile`),
so rotary channels crowd where the periods actually are. Check
`time_shuffle_score` after training; near 1 means the encoder is blind to time.

---

## 4. Stage 1 — latent world model (deterministic predictor)

This is `romae_lc.LeWorldModel` as shipped. Loss:

```
L = ‖P_θ(z_t, Δ_t) − z_{t+1}‖²  +  λ · SIGReg({z})
```

No stop-gradient, no EMA, no decoder loss. Both the encoder and the predictor
get gradients from the prediction term. SIGReg is applied per window.

```python
from functools import partial
from torch.utils.data import DataLoader
from romae_lc import FrameConfig, FrameDataset, LeWorldModel, RoMAE, collate_frames

cfg = FrameConfig(n_frames=4, window=30.0, advance=(1.0, 2.0))
loader = DataLoader(
    FrameDataset(train, cfg), batch_size=128,
    collate_fn=partial(collate_frames, band_wavelengths=wavelengths, time_scale=ts),
)
backbone = RoMAE(encoder=dict(d_model=192, nhead=3, depth=12), rope_base=base)
model = LeWorldModel(backbone, history=3, lamb=0.1)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
for batch in loader:
    out = model(batch["frames"], batch["actions"])
    opt.zero_grad(); out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
```

Or just `examples/train_lewm.py --data <root> --label superclass --auto-time
--window 30 --advance 1 2`, which also prints the linear-probe accuracy and
ridge R² on log period every few epochs and saves `runs/lewm_wm.pt`.

**What "done" looks like for stage 1**
- `pred_loss` decreases smoothly; `sigreg_loss` drops early and plateaus.
- Linear probe on `z` recovers class and log period at a useful level.
- `time_shuffle_score` well below 1.
- `LeWMOutput.straightness` rises over training (a free sanity signal).
- `model.surprise(...)` on `frame_grid(..., fill=True)` gives low, flat values
  on clean periodic curves and spikes where you inject a perturbation
  (see §7 for the injected-anomaly test).

Stage 1 alone already enables smoothing, forecasting and anomaly scoring
(§6). Everything below is added on top with the encoder frozen.

---

## 5. Stage 2 — residual model (stochastic predictor)

Do **not** replace the MSE predictor with a likelihood model trained
end-to-end. A likelihood loss has no pressure toward predictable features: the
encoder can make `z_{t+1}` independent of `z_t`, the conditional collapses to
the SIGReg Gaussian, and a trivial velocity field fits it. The MSE term is what
keeps the latent predictable.

Instead, model the residual with the encoder and predictor frozen:

```
r_t = z_{t+1} − P_θ(z_t, Δ_t)
r_t ~ p(r | z_t, Δ_t)
```

Two levels:

1. **Gaussian residual.** Fit `Σ(Δ)` (diagonal, or full in 192-d) by class or
   pooled. Check that the width grows with `Δ`. If residuals look unimodal,
   stop here. This is enough for periodic stars.
2. **Flow-matching residual.** A small conditional flow `v_θ(r_s, s | z_t, Δ_t)`
   from `N(0, I)` to the residual. Residuals are small and centered, so a few
   ODE steps suffice.

Rollout with the residual: mean step, sample `r`, add. Run many chains in
parallel to get a spread of futures.

Energy from the residual: for Gaussian, `½ rᵀ Σ(Δ)⁻¹ r` exactly. For the
flow, either the divergence-corrected log density (Hutchinson, cheap in 192-d)
or the residual norm as a surrogate. If exact energies at every step matter
from the start, parameterize the residual as a diffusion (score) model
instead, which gives the energy gradient for free.

Diagnostic that decides between level 1 and 2: histogram `r_t` projected on
its top principal directions, split by class and by `Δ` bin. Multi-modal →
flow. Unimodal → Gaussian.

---

## 6. Stage 3 — decoder `z → μ(t, b)`

A conditional flow-matching model over clean magnitudes at query times inside
the window, trained on frozen latents.

```
D_ψ: (z, {t_j − τ, b_j}) → μ(t_1:q, b_1:q)
```

Conditioning: `z` via AdaLN or cross-attention; each query token carries
`(t_j − τ, b_j)` with the same rotary time encoding as the encoder.

Loss (flow matching, targets are observed magnitudes):

```
x^(s) = (1 − s) ε + s · m_1:q
L_dec = E_s,ε ‖ v_ψ(x^(s), s | z, t, b) − (m_1:q − ε) ‖²
```

**Handling the known errors.** Three options, in order of effort:
1. Condition `v_ψ` on `σ_1:q` too. At test time query with `σ = 0` to get `μ`.
   Start here.
2. Weight the loss per point by `1/σ_j²`.
3. Treat `m_j` as `μ_j` already noised to the flow level matching `σ_j`, and
   train the denoiser only above that level. Exact under the Gaussian
   measurement model. Move here if errors vary a lot across the sample.

**Gradient flow.** Frozen encoder. The LeWM paper's own ablation (their
Table 7) shows a joint reconstruction loss hurts planning (96 → 86 on Push-T),
because the latent starts storing detail the predictor cannot forecast. The
decoder is a read-out, not a training signal, in v0.1. If decoding is too
lossy, revisit with a small joint weight.

**Mean estimate for energies.** `μ̂_ψ(t, b | z)` = one deterministic ODE solve
from a fixed `ε = 0`, or the average of a few samples.

---

## 7. Energies and inference procedures

With stages 1–3 done we have, over any latent path `z_1:K` on any time grid:

```
E_prior(z_k)            = ½ ‖z_k‖²
E_dyn(z_k, z_{k+1})     = ‖P_θ(z_k, Δ_k) − z_{k+1}‖²            (stage 1)
                        or ½ rᵀ Σ(Δ_k)⁻¹ r                      (stage 2)
E_obs(z_k, x_k)         = Σ_j (m_j − μ̂_ψ(t_j, b_j | z_k))² / (2 σ_j²)   (stage 3)
```

`E_obs` uses the survey's own `σ`: good points anchor the path, bad ones do
not, with no tuning.

### 7.1 Smoothing / gap filling

```
inputs : observed windows x_k for k ∈ O, time grid for all k ∈ [1..K]
init   : z_k = E_φ(x_k) for k ∈ O; rollout from nearest observed k otherwise
energy : Σ_k E_dyn + Σ_{k∈O} E_obs + Σ_k E_prior
solve  : gradient descent (MAP) or Langevin (posterior samples)
output : decode μ̂(t) on the gap grid; spread across chains = uncertainty
```

Weights: `E_dyn` and `E_obs` are on different scales. Start with the weight
that makes their per-window magnitudes equal on the training set, then sweep.

### 7.2 Forecasting

Same as 7.1 with no `E_obs` on future nodes. Report the spread of the decoded
futures across chains at several horizons.

### 7.3 Anomaly score

After smoothing, read `E_dyn` per step and `E_obs` per window. Report:
- Object-level: sum over the curve, normalized by number of valid windows.
- Time-localized: the per-window series, so you can see *where* it goes wrong.

Calibrate thresholds on validation objects. With the deterministic predictor
the score ranks well but is not a log density.

**Injected-anomaly test (build this first).** Take clean periodic validation
curves and inject: (a) a phase jump at a random time, (b) an amplitude change,
(c) a period change of 1–5%, (d) an added transient bump, (e) a color change
(band offset only). Measure whether the per-window score spikes at the
injection time. (e) should produce a weaker response than (a)–(d), mirroring
the visual-vs-physical result in the LeWM paper.

### 7.4 Period and period-change detection

Add a periodicity energy over the smoothed path with `P` free:

```
E_per(z_1:K, P) = Σ_k ‖ z(t_k) − z(t_k + P) ‖²
```

where `z(t_k + P)` is obtained by interpolating on the path or by re-running
the smoother on a grid shifted by `P`. Minimize over `(z, P)` jointly.
Compare `P` to Lomb–Scargle. Then scan a sliding window: where the composed
energy rises, the star stopped being periodic at that `P`. Validate on known
Blazhko RR Lyrae and known period-changing Cepheids in the sample.

### 7.5 Later additions (not v0.1)

- Physical template energy on the decoded `μ` (Bazin, SN Ia, blackbody).
- Parameter inference: condition `P_θ` or `D_ψ` on physical parameters `ω`
  from simulations, fix the data, minimize over `(z, ω)`.
- Multi-survey composition: one model per survey, shared time axis, summed
  energies.
- Follow-up planning: pick the next observation time and band that most
  reduces the spread of sampled futures.

---

## 8. Suggested hyperparameters

Numbers below come from the papers named in each table. "Start" is the value
to use first. "Sweep" is what to vary if the start value fails. Sources:
LeWM (Maes et al. 2026, Sec. 3.1, App. D, App. G), the `romae-lc` README
(which ports the released LeWM config), RoMAE (Zivanovic et al. 2025, App. D,
ELAsTiCC light-curve setup), Diffusion Autoencoders (Preechakul et al. 2022,
Sec. 3–5), LeJEPA (Balestriero & LeCun 2025) for SIGReg, and standard
conditional flow matching (Lipman et al. 2023).

### 8.1 Stage 1 — encoder + predictor + SIGReg

| Hyperparameter | Start | Sweep | Source / note |
|---|---|---|---|
| Latent width `d_model` | 192 | 96, 192, 384 | LeWM: performance saturates above ~184 on Push-T; falls off sharply below. Watch `sigreg_loss` on a low-diversity class; the paper notes the Gaussian target is hard to hit when intrinsic dimension is low. |
| Encoder depth / heads | 12 / 3 | 2/3 (`tiny-shallow`) for smoke tests | LeWM uses ViT-Tiny (~5M). RoMAE `tiny` is 180/3/12 (head dim must divide by 6 for RoPE). |
| Predictor | 6 layers, 16 heads × 64, MLP 2048, dropout 0.1 | dropout {0.0, 0.1, 0.2}; size {tiny, small} | LeWM App. G: ViT-S predictor best; ViT-T and ViT-B both worse. Dropout 0.1 gave 96% vs 78% at 0.0 and 67% at 0.5. |
| `history` | 3 | 1, 3 | LeWM: 3 for PushT/Cube, 1 for TwoRoom. |
| SIGReg weight `lamb` | 0.1 | bisection on [0.01, 0.2] | LeWM Fig. 16: >80% success across [0.01, 0.2], peak near 0.09; collapses at 0.5. Released config uses 0.09. This is the only hyperparameter they say needs tuning. |
| SIGReg projections `M` | 1024 | 64–1024 | LeWM: no measurable effect on downstream performance. |
| SIGReg knots | 17 on [0, 3] | 4–32 | LeWM: insensitive. |
| Projector | Linear–BatchNorm–GELU–Linear, hidden 2048 | — | LeWM: BN in the projector is required because the ViT's final LayerNorm blocks SIGReg. Keep it. |
| Optimizer | AdamW, lr 5e-5, wd 1e-3 | lr {2e-5, 5e-5, 1e-4} | LeWM defaults. RoMAE's own light-curve runs use a much higher base lr (6.4e-3) but at batch 16384 with warmup and cosine; do not mix the two recipes. |
| Batch size | 128 | 64–256 | LeWM. Note `val sigreg` only matches the training scale when validation batches have the same size (README). |
| Grad clip | 1.0 | — | LeWM. |
| Epochs | 10 on ~10k–20k trajectories | — | LeWM reports 10 epochs is enough. With ZTF you have far more objects but shorter per-object sequences; count optimizer steps, not epochs. Target ~100k–200k steps as in their training curves. |
| Precision | BF16 or FP32 | — | RoMAE App. D: FP16 gave NaNs on light-curve inputs because of the input range; BF16 was fine. |
| `n_frames` | 4 | 4–8 | LeWM uses sub-trajectories of 4 frames. |
| `window` (days) | 30 | 10, 30, 90 | README docstring: window of many periods → model learns period/shape/amplitude, not phase. Choose per the period census. |
| `advance` (window units) | (1.0, 2.0) | add a (1.0, 6.0) run | Package default. The second run is for seasonal gaps. |
| `min_tokens` | from cadence census | — | Must be high enough that a window pins phase. |
| Rotary time ladder | `--auto-time`, quantile spacing | log spacing | README: with the LM default `base=1e4` and time in days, sub-day periods are invisible. Always run `suggest_time_encoding`. |
| `p_rope` | 0.75 | 0.5–1.0 | Package default (fraction of active rotary channels). |
| `use_cls` | True | — | Required to anchor phase. |

### 8.2 Stage 2 — residual model

| Hyperparameter | Start | Sweep | Note |
|---|---|---|---|
| Residual covariance | diagonal, per `Δ` bin (4–6 bins in log Δ) | full covariance; per class | Gaussian level. Fit by MLE on frozen latents. |
| Flow network | MLP, 4–6 layers, width 512–1024, skip connections, AdaLN or concat conditioning on `(z_t, Δ, s)` | depth 10–20 | DAE Sec. 4: for non-spatial vectors, deep MLPs (10–20 layers) with skips worked well for their latent diffusion; they also found L1 beat L2 for the latent model. Start smaller since residuals are low-variance. |
| Flow path | linear (OT) interpolant, `σ_min = 1e-4` | — | Lipman et al. 2023 default. |
| Time sampling `s` | uniform on [0, 1] | logit-normal | Uniform is fine for a small latent. |
| Optimizer | AdamW, lr 1e-4, wd 0.01, β=(0.9, 0.999) | lr {3e-5, 1e-4, 3e-4} | DiTo (diffusion tokenizer) defaults; standard for small flow models. |
| Batch size | 256–1024 | — | Residual vectors are cheap; use a big batch. |
| Sampling steps | 8 Euler / 4 midpoint | 2–32 | Residuals are near-Gaussian; few steps suffice. |
| Normalization | z-score residuals per dimension before fitting; unnormalize after | — | DAE normalizes the latent to zero mean / unit variance before fitting the latent DPM. |

### 8.3 Stage 3 — flow decoder `z → μ(t, b)`

| Hyperparameter | Start | Sweep | Note |
|---|---|---|---|
| Decoder architecture | transformer over query tokens `(t_j − τ, b_j)`, 4 layers, `d_model` 192–256, 4 heads | 2–8 layers | Same rotary time encoding as the encoder. Small: the latent already carries the structure (DAE Sec. 5.5: conditioning on `z_sem` makes denoising much easier). |
| Conditioning on `z` | AdaLN (scale/shift per layer), zero-init | cross-attention | DAE conditions the UNet with AdaGN on `z_sem` and `t`; LeWM uses AdaLN-zero for the action. Same idea, zero-init for stability. |
| Conditioning on `σ` | concat `log σ_j` to each query token | none (option 2/3 in §6) | Option 1 in §6. |
| Flow path | linear interpolant, `σ_min = 1e-4`, velocity prediction | ε- or x-prediction | DAE used ε-prediction with `T = 1000`; the 2025 "Revisiting DAE training" paper argues the linear-β schedule wastes steps at high noise for reconstruction and that x/v-prediction helps. Flow matching sidesteps the schedule question. |
| Sampling steps | 20 | 10–100 | DAE Table 2: with a good semantic code, `T = 20` already beats unconditional DDIM at `T = 100`. |
| Mean estimate for `E_obs` | deterministic solve from `ε = 0`, 20 steps | average of 4 samples | See §6. |
| Optimizer | AdamW, lr 1e-4, wd 0.01 | lr {3e-5, 1e-4} | DiTo defaults. |
| Batch size | 128 windows | — | — |
| Steps | 100k–300k | — | DiTo trains tokenizers for 300k steps at batch 64. Expect fewer to be enough here. |
| Latent dimension effect | fixed by stage 1 | — | DAE Table 2: reconstruction fidelity rises monotonically from 64-D to 512-D latents. If 192-D decodes poorly, that is a reason to try 384 in stage 1, not to change the decoder. |
| Gradient into encoder | none (frozen) | small weight β ≤ 0.1 | LeWM Table 7: joint reconstruction loss hurt planning (96 → 86). DAE trains jointly, but its goal is reconstruction, not prediction. |

### 8.4 Inference (§7)

| Hyperparameter | Start | Sweep | Note |
|---|---|---|---|
| Energy weights `w_dyn : w_obs : w_prior` | equalize per-window magnitudes on training data; prior weight 0.1 | 3× up and down each | Du thesis Ch. 3: weights set the balance between composed landscapes; there is no free lunch. |
| Optimizer for MAP | Adam, lr 1e-2 on `z`, 200–500 steps | L-BFGS | LeWM App. G (Table 10): for their action-space planning, Adam (84%) beat SGD (26%) and RMSProp (67%); CEM (96%) was best. For latent-path smoothing use gradients, since the variable is high-dimensional. |
| Langevin step `η` | 1e-3 in normalized latent units | 1e-4–1e-2 | Du overview page example uses `η = 0.003` at `T = 1`. Tune so the acceptance/energy trace is stable over ~1000 steps. |
| Langevin chains | 64–128 | — | Du overview page uses 128 chains for a 1-D toy. |
| Langevin steps | 1000, after MAP init | 300–3000 | Init from the MAP solution so chains start in the right basin. |
| Temperature | 1.0 | 0.5–2.0 | Below 1 sharpens toward the MAP; above 1 widens. Report which you used. |
| Periodicity energy weight | equalize to `E_dyn` | — | §7.4. |

### 8.5 Things the sources agree on

- One regularization weight (`lamb`) is the real hyperparameter of stage 1.
  Everything else in LeWM is insensitive or has a stated default.
- Small predictor dropout (0.1) matters more than it looks.
- A conditioned decoder needs far fewer sampling steps than an unconditioned one.
- Keep the decoder out of the encoder's gradient unless you have a reason.

---

## 9. Evaluation

| Metric | What it tests | Baseline |
|---|---|---|
| Gap-imputation NLL under known `σ` | stages 1+3 | Gaussian process per band; linear interpolation |
| Forecast NLL at 1, 3, 10 windows | stages 1+2 | persistence; GP |
| Linear probe: class, log period, amplitude | stage 1 latent | RoMAE MAE embedding; LeJEPA embedding |
| Period recovery on known periodic stars | §7.4 | Lomb–Scargle |
| Injected-anomaly detection AUROC and localization | §7.3 | per-band residual from a GP fit |
| Straightness, `time_shuffle_score` | stage 1 sanity | — |

The imputation NLL is the honest headline number. The noise model is given, so
there is no way to cheat it.

---

## 10. Milestones

1. **M1.** Stage 1 on `ZTFxPC`, periodic superclasses, `window = 30`.
   Probes, straightness, surprise on `frame_grid`. Window sweep.
2. **M2.** Injected-anomaly test with the raw `surprise` signal. This is the
   first result worth showing.
3. **M3.** Error channel in the encoder (§3.3). Re-run M1 and compare probes.
4. **M4.** Gaussian residual (§5, level 1). Rollouts with spread. Forecast NLL.
5. **M5.** Flow decoder (§6). Imputation NLL vs GP baseline.
6. **M6.** Smoothing and anomaly scores with the full energy (§7.1, 7.3).
7. **M7.** Periodicity energy (§7.4). Period-change detection on known cases.
8. **M8.** Flow residual (§5, level 2) only if M4 shows multi-modal residuals.

Each milestone reuses the previous checkpoint. Nothing is retrained from
scratch after M1 except in the window sweep.

---

## 11. Open questions and risks

- **Window vs phase.** A window long enough for slow classes may lose phase
  for fast ones. A per-class window, or a two-scale model, may be needed.
  Decide after the M1 sweep.
- **SIGReg on a low-dimensional dataset.** The LeWM paper notes that low
  intrinsic dimensionality makes the Gaussian target hard to hit and hurts
  the latent. A single periodic class is low-dimensional. Watch `sigreg_loss`
  and consider a smaller `d_model` if it never drops.
- **Δ coverage.** The predictor only knows gaps it has seen. Seasonal gaps
  need the larger-advance run, or a latent ODE later.
- **Energy scale mismatch.** `E_dyn` is in latent units, `E_obs` in magnitude
  units. The relative weight is a real hyperparameter, not a nuisance.
- **Latent identifiability.** Do not read latent axes directly. Probes and
  the decoder are the interface to physics.
- **Deterministic predictor on branching behavior.** Not an issue for periodic
  stars. It is for transients; that is what stage 2 is for.
- **Tokenizer change (§3.3).** Adding a channel may touch `RoMAE`'s input
  projection, the MAE decoder, and the checkpoints' `hparams`. Budget time.

---

## 12. References

- Maes, Le Lidec, Scieur, LeCun, Balestriero (2026). LeWorldModel. arXiv:2603.19312.
- Balestriero, LeCun (2025). LeJEPA / SIGReg. arXiv:2511.08544.
- Du (2024). Learning Generalizable Systems by Learning Composable Energy
  Landscapes. MIT PhD thesis. Ch. 2 (energy landscapes), Ch. 3 (composition
  algebra), Ch. 6.1 (planning through model composition).
- Du, Lin, Mordatch (2019). Model Based Planning with Energy Based Models. CoRL.
- Du et al. (2023). Reduce, Reuse, Recycle. ICML. (samplers for composed
  diffusion energies)
- Comas, Du et al. (2023). Inferring Relational Potentials in Interacting
  Systems. ICML. (hand-crafted potentials added at test time)
- Wang, Du (2025). Equilibrium Matching. arXiv:2510.02300. (flow ↔ energy)
- Zivanovic et al. (2025). RoMAE. arXiv:2505.20535.
