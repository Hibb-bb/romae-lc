# romae-lc

RoMAE for light curves: a continuous rotary transformer in which every observation of an asynchronous multi-band light curve is one token with a real-valued (time, wavelength) position, so nothing is binned, aligned or imputed. It ships axial p-RoPE and nD-RoPE (simplex) position encodings, softmax or linear attention, three pretraining frameworks (RoMAE masked autoencoding, LeJEPA and LeWorldModel) and an analysis module that reads the right rotary time unit off a dataset. The library depends on `torch` and `numpy` only; `datasets` (PHOEBE loader) and `matplotlib` (plots) are optional extras.

## Install

```bash
uv sync               # library + dev tools (pytest, black)
uv sync --all-extras  # also datasets and matplotlib
uv run pytest         # CPU test suite, under a minute
uv run black .        # formatting (line length 88)
```

## Quickstart

Tokenize a list of light curves (any epoch order, any number of points per band) and embed them:

```python
import torch
from romae_lc import RoMAE, tokenize, wavelengths_for

# three objects, epochs in days, two bands; one token per observation
times = [torch.rand(n) * 300 for n in (120, 80, 150)]
values = [torch.randn(n) for n in (120, 80, 150)]
bands = [torch.randint(0, 2, (n,)) for n in (120, 80, 150)]

tokens = tokenize(
    times,
    values,
    bands,
    band_wavelengths=wavelengths_for(["ZTF_g", "ZTF_r"]),  # {0: 472.0, 1: 634.0} nm
    time_scale=0.05,  # days per position unit, see "Choosing the time unit"
)
backbone = RoMAE(encoder=dict(d_model=96, nhead=3, depth=2), rope_base=1e5)
z = backbone(*tokens)  # [3, 96], CLS-pooled
```

`tokens` is a `Tokens(values [B, N, 1], positions [B, 2, N], pad_mask [B, N])` batch, sorted by time and padded. `encoder` takes a size name (`"tiny-shallow"`, `"tiny"`, `"small"`, `"base"` follow Table 8 of the RoMAE paper; `"large"` = 960/16/24 is an extrapolation added here) or a dict of `TransformerConfig` fields.

**LeJEPA.** Views are random time windows of the same curves (`ViewConfig`); the loss is invariance across views plus SIGReg, the sliced Epps-Pulley test that keeps the projected embeddings Gaussian:

```python
from functools import partial

from torch.utils.data import DataLoader

from romae_lc import DEFAULT_SURVEYS, LeJEPA, LightCurveDataset, ViewConfig
from romae_lc import collate, mlp, normalize, simulate

records = [normalize(r) for r in simulate(64, seed=0)]
wavelengths = {i: s.wavelength_nm for i, s in enumerate(DEFAULT_SURVEYS)}
loader = DataLoader(
    LightCurveDataset(records, ViewConfig(n_global=2, n_local=4)),
    batch_size=16,
    collate_fn=partial(collate, band_wavelengths=wavelengths, time_scale=0.05),
)

backbone = RoMAE(encoder=dict(d_model=96, nhead=3, depth=2), rope_base=1e5)
model = LeJEPA(backbone, proj=mlp([96, 256, 256, 64]), lamb=0.02)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
for batch in loader:
    out = model(global_views=batch["global"], local_views=batch["local"])
    opt.zero_grad()
    out.loss.backward()  # out.inv_loss, out.sigreg_loss, out.embedding, ...
    opt.step()

model.eval()
z = model.embed(*batch["full"])  # [16, 96]; "full" is capped at view_cfg.max_tokens
# (512) here; a LightCurveDataset(records) without a view_cfg yields whole curves
```

`LeJEPA` exposes `.backbone`, `.proj` and an optional `.predictor` (`predictor=mlp([D, 1024, D]), predictor_inst=k` maps the views of instrument `k` into the shared space before the loss). With a predictor the training call must also receive `view_inst`, an `[N, n_global + n_local]` integer tensor giving each view's instrument id (global views first, then local views); omitting it raises `ValueError`. Nothing on the data side knows an instrument (`Record` has no such field and `collate` does not emit one), so build it yourself, e.g. `view_inst = torch.full((N, n_views), inst_id)` or from a field you keep in `Record.meta`, and call `model(global_views=..., local_views=..., view_inst=view_inst)`.

**LeWorldModel.** A frame is a time window of a curve (all bands, `FrameConfig`), the action the advance to the next window in window units; the loss is next-window latent prediction with a causal AdaLN-zero predictor plus SIGReg on the latents of every window, no stop-gradient or EMA (Maes, Le Lidec et al. 2026):

```python
from romae_lc import FrameConfig, FrameDataset, LeWorldModel, collate_frames, frame_grid

cfg = FrameConfig(n_frames=4, window=30.0)
loader = DataLoader(
    FrameDataset(records, cfg),
    batch_size=16,
    collate_fn=partial(collate_frames, band_wavelengths=wavelengths, time_scale=0.05),
)
backbone = RoMAE(encoder=dict(d_model=192, nhead=3, depth=12), rope_base=1e5)
model = LeWorldModel(backbone, history=3, lamb=0.1)  # predictor: 6 layers, 16 heads, dropout 0.1
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
for batch in loader:
    out = model(batch["frames"], batch["actions"])  # LeWMOutput(loss, pred_loss, sigreg_loss, embedding [B, 4, D], ...)
    opt.zero_grad()
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

model.eval()
z = model.encode(batch["frames"])                                   # [16, 4, D] latents z_t
z_hat = model.rollout(batch["frames"][:3], batch["actions"])        # [16, 5, D] open-loop (4 actions -> 5 latents)
frames, actions = frame_grid(records[0], cfg, fill=True)            # every contiguous 30-d window of one curve
grid = collate_frames(
    [dict(frames=frames, actions=actions, label=0, index=0)],
    band_wavelengths=wavelengths,
    time_scale=0.05,
)
s = model.surprise(grid["frames"], grid["actions"])                 # [1, n - 1] per-window prediction error
s[grid["n_tokens"][:, 1:] < cfg.min_tokens] = float("nan")          # empty windows in season gaps carry no information
```

`LeWorldModel` exposes `.backbone`, `.projector`, `.predictor`, `.action_encoder` and `.pred_proj`; the parts are `ARPredictor` (the causal AdaLN-zero transformer), `ActionEncoder` (`Conv1d(A, 10, 1)-Linear-SiLU-Linear`), `lewm_mlp` (the `Linear-BatchNorm-GELU-Linear` projector used for both `projector` and `pred_proj`) and `straightness` (the latent-path diagnostic of the paper, also in `LeWMOutput.straightness`). Passing `actions=None` runs the predictor unconditioned (the AdaLN layers learn constant per-layer shifts, scales and gates); this is an extension, the official model is always action-conditioned. Custom actions are any `[B, T, A]` tensor with `action_dim=A`, one per frame, the last being the action that would follow the last frame; z-score them, the default advance is O(1) by construction. Two tensors can be probed: `LeWMOutput.embedding` / `encode(frames)` is the paper's latent `z_t` (post-projector, the quantity predicted, regularised and probed there), `LeWMOutput.features` / `encode(frames, project=False)` the backbone output, which is what `examples/train_lewm.py` probes and what its `runs/lewm.pt` checkpoint restores (`runs/lewm_wm.pt` holds the full world model, see Examples).

`FrameConfig` sets the sampler: `n_frames` windows of `window` days, the start-to-start advance drawn from `U(*advance)` in window units (`(1.0, 2.0)`: non-overlapping, at most one window apart; `(1.0, 1.0)` is a fixed frame skip), `min_tokens` points per window else the draw is repeated up to `max_tries` times before the best draw is kept, `max_tokens` random subsampling and the optional `resample` augmentation (off, as in the official pipeline). `frame_grid` lays deterministic windows for evaluation (`fill=True`: every window that fits, sparse or empty ones kept, mask them with `n_tokens` as above). Records shorter than `window * (1 + (n_frames - 1) * advance[0])` cannot host a sequence and are dropped by `FrameDataset` with a warning. Windows are tokenized with time re-zeroed at the window start: `use_cls=True` (the `RoMAE` default) is what anchors a window's phase, since without the CLS token at position 0 the encoder is purely relative in time and a window's embedding is invariant to a shift of its contents; with `use_cls=False, pool="mean"` the model can only learn shift-invariant window statistics and every window must be non-empty.

All model and optimiser defaults are those of the paper and the released code (Sec. 3.1, App. D, `config/train/lewm.yaml`): latent width 192, predictor of 6 layers, 16 heads of 64, MLP 2048, dropout 0.1, learned positional embedding over `history=3` frames, projector hidden width 2048, `lamb=0.1` (the released config uses 0.09), 1024 SIGReg projections on 17 knots over `[0, 3]`, AdamW at `lr=5e-5` with weight decay `1e-3`, gradient clipping at 1, batch size 128. The window, advance and action are the light-curve translation (table in the docstring of `FrameConfig`). With `window` many periods long the next window's phase is unpredictable and the world model learns period/shape/amplitude dynamics, not phase: make `window` a few periods when you want phase. With `--auto-time` the rotary ladder's `lam_max` is `2 * window`, since the encoder only ever sees windows.

**Masked autoencoding** (the RoMAE recipe): the encoder sees the visible tokens and a light decoder reconstructs the masked fluxes from MASK tokens at their positions.

```python
from romae_lc import RoMAEForPreTraining

model = RoMAEForPreTraining(
    decoder=dict(d_model=48, nhead=3, depth=1),  # the paper uses "tiny-shallow"
    mask_ratio=0.5,
    encoder=dict(d_model=96, nhead=3, depth=2),
    rope_base=1e5,
)
out = model(*tokens)  # MAEOutput(loss, pred [B, K, 1], target, mask [B, N])
# mask_ratio must lie in (0, 1]; a mask that selects no token raises
out.loss.backward()
backbone = model.backbone()  # a RoMAE carrying the pretrained encoder
```

**Classification**, from scratch or on pretrained weights:

```python
from romae_lc import RoMAEForClassification

clf = RoMAEForClassification(n_classes=5, encoder=dict(d_model=96, nhead=3, depth=2))
logits = clf(*tokens)  # [3, 5]
```

## Positional encodings

Positions are `[B, n_axes, N]`: axis 0 is time `t / time_scale`, axis 1 the wavelength coordinate `log(lambda / lambda_ref) / wavelength_scale` (the tokenizer offsets both by +1 so that the CLS token alone sits at 0). The head dimension is cut into blocks, each rotated by its own axes with its own scheme (`rope=` of every model):

| `rope=` | time axis | other axes |
|---|---|---|
| `"axial"` (default) | axial p-RoPE, base `rope_base`, fraction `p_rope` | one axial p-RoPE block per axis, base 1e4; the head is split equally |
| `"simplex"` | axial p-RoPE on half the head | one nD-RoPE block over all remaining axes: wave vectors along the directions of a regular simplex, scales `theta ** (-s / S)`, a random rotation per head (a single wavelength axis reduces it to an axial ladder with base `theta`) |
| `list[dict]` | any block layout `BlockRope` accepts, see below | |

`layout()` builds the two string layouts; the same dicts can be edited or written by hand:

```python
from romae_lc import layout

layout(32, n_axes=2, kind="axial", time_base=1e5)
# [{'kind': 'axial', 'axes': [0], 'dim': 16, 'base': 100000.0, 'p': 0.75},
#  {'kind': 'axial', 'axes': [1], 'dim': 16, 'base': 10000.0, 'p': 0.75}]
layout(32, n_axes=4, kind="simplex", time_frac=0.5, theta=100.0)
# [{'kind': 'axial', 'axes': [0], 'dim': 16, 'base': 10000.0, 'p': 0.75},
#  {'kind': 'simplex', 'axes': [1, 2, 3], 'dim': 16, 'theta': 100.0, 'p': 0.75,
#   'seed': 0}]

blocks = [
    {"kind": "axial", "axes": [0], "dim": 16, "base": 1e5, "p": 0.75},
    {"kind": "simplex", "axes": [1, 2, 3], "dim": 16, "theta": 100.0, "p": 0.75},
]
model = RoMAE(encoder=dict(d_model=96, nhead=3, depth=2), n_axes=4, rope=blocks)
```

Block dims must sum to `head_dim = d_model // nhead`, and a simplex block over `k > 1` axes needs a multiple of `2 (k + 1)` channels. `p` is the p-RoPE fraction: the remaining angles are NoPE channels that carry no position.

The ladders need not be geometric. An axial block takes an explicit `timescales` list (position units, any spacing, at most `dim / 2` values; the rest of the angles are NoPE and `p` becomes the active fraction) and a simplex block an explicit `scales` list of wave-vector magnitudes, both stored as plain floats in the layout (so `hparams` and checkpoints carry them). `RoMAE(..., rope_timescales=[...])` passes a time ladder into the string layouts, and `layout(..., time_timescales=[...])` does the same by hand:

```python
import math

from romae_lc import RoMAE, rotary_ladder, timescales_for

wavelengths = rotary_ladder(8, 0.05, 4000.0, "quantile", samples=periods, mix=0.75)  # days
backbone = RoMAE(
    encoder=dict(d_model=96, nhead=3, depth=2),  # head_dim 32: 16 time channels, 8 angles
    rope_timescales=timescales_for(wavelengths, time_scale=0.05 / (2 * math.pi)),
)
```

`rotary_ladder(n, lam_min, lam_max, spacing)` gives `n` wavelengths in days with `"log"` (the geometric ladder `time_scale` and `base` encode), `"linear"` or `"quantile"` spacing; the quantile ladder is the inverse CDF of `log(samples)` (catalogue periods, measured timescales, ...) clipped to the band, blended in log space with the log ladder by `mix` (1 = pure quantile, 0 = pure log; a little log floor keeps sparse parts of the band covered), so channels crowd where the dataset's timescales are dense. `timescales_for(wavelengths, time_scale)` converts days into the position units the block works in. The MAE decoder resamples an explicit ladder to its own number of angles in log space, ends kept (`resample_ladder`). Four axes arise when `tokenize(..., band_positions={band: (c1, c2, c3)})` describes each band by several coordinates instead of one wavelength. `nd_rope_theta_bound(dim, n_axes)` is the largest `theta` at which every wave vector of a simplex block stays resolvable (nD-RoPE paper, App. E); its arguments are the block's own channel count and number of axes, not `d_model // nhead` and the model's `n_axes`, since the time block owns the rest of the head: for the layout above, `nd_rope_theta_bound(blocks[1]["dim"], len(blocks[1]["axes"]))` = `nd_rope_theta_bound(16, 3)` = 1.95, with positions in units of the smallest step to resolve. `layout()`'s default `theta=100` is well above this bound for small blocks. Every block must keep at least one active angle after `p` is applied: an axial block needs `min_axial_dim(p)` channels (`2 * ceil(1 / p)`, so 4 at the default `p=0.75`) and a simplex block `2 (k + 1) * min_simplex_scales(p)`; `layout()` and the rope constructors raise otherwise instead of returning a dead (NoPE) block.

## PC_matches data

`load_pc(source, split)` reads one sub-dataset of the PC_matches collection (`/projects/bfrf/data/PC_matches/<name>`, a `save_to_disk` directory with a nested `lightcurve.<band>.{mjd, mag, mag_unc, clean}` column) into `Record`s. Band ids index `PC_BANDS`, a fixed list of every band of every survey, so single-survey and multi-survey (`*-isect`) sources share ids and the tokenizer table is always `wavelengths_for(PC_BANDS)`. Values are `-mag` (brighter is up, `normalize` standardises per band), unclean and non-finite points are dropped, and times are MJD re-zeroed at each record's first point (`meta["t0"]`; float32 positions cannot hold an absolute MJD at intra-night resolution). Labels are the 8 `PC_SUPERCLASSES` by default, or `label_field="class_str"` with a shared `classes` vocabulary.

```python
from romae_lc import PC_BANDS, load_pc, normalize, wavelengths_for

train = [normalize(r) for r in load_pc("/projects/bfrf/data/PC_matches/ZTFxPC", "train")]
wavelengths = wavelengths_for(PC_BANDS)
```

The nested columns are read as Arrow arrays (about 1 ms per row). The training scripts detect the layout from the path: `--data /projects/bfrf/data/PC_matches/ZTFxPC --label superclass --min-points 8 --max-rows N`. `jobs/train_lewm_pc.sh` is a Slurm job that trains a LeWorldModel on it; `examples/analyze_pc_periods.py` (`jobs/analyze_pc_periods.sh`) is the period and cadence census whose results in `results/pc_period/` motivate the quantile time ladder above.

## Linear attention

`attention="linear"` in an encoder (or decoder) config replaces softmax attention by kernel attention with `phi(x) = elu(x) + 1` features and the rotary phase applied inside the numerator, so the cost is linear in the number of tokens and the position encoding is unchanged:

```python
backbone = RoMAE(encoder=dict(d_model=96, nhead=3, depth=2, attention="linear"))
```

Numerator and denominator are accumulated in float32. Only key (padding) masks are supported: the mask has to be the same for every query, which is what `romae_lc.transformer.attention_mask(pad_mask)` produces; a query-dependent mask raises `NotImplementedError`.

## Choosing the time unit

A rotary channel of wavelength `lambda` turns once per lag `lambda`: it separates tokens closer than that and wraps around beyond. The active channels of the time block form the ladder `2 pi time_scale base ** (2 i / dim)` days, so the ladder has to reach below the shortest timescale that carries signal (the high-frequency end of the periodogram, harmonics included) and up to the baseline of the curves. Time in days with the language-model default `base = 1e4` puts the shortest wavelength at `2 pi` days: sub-day periods are invisible and the encoder degrades into a bag of magnitudes. `suggest_time_encoding` measures both ends with a generalised Lomb-Scargle periodogram over the dataset and inverts the ladder to `time_scale` and `base`; `time_shuffle_score` is the check on an encoder.

```python
from romae_lc import suggest_time_encoding, time_series, time_shuffle_score

records = [normalize(r) for r in simulate(128, seed=0)]
times, values, bands = time_series(records)
encoder = dict(d_model=96, nhead=3, depth=2)
dim = RoMAE(encoder=encoder).rope.dims[0]  # channels of the time block
report = suggest_time_encoding(times, values, dim, p=0.75, bands=bands)
print(report)  # sampling limits, spectral support, time_scale, base, ladder

backbone = RoMAE(encoder=encoder, rope_base=report.base)
curves = records[:32]
tokens = tokenize(
    [torch.from_numpy(r.t) for r in curves],
    [torch.from_numpy(r.y) for r in curves],
    [torch.from_numpy(r.band) for r in curves],
    band_wavelengths=wavelengths,
    time_scale=report.time_scale,
)
score = time_shuffle_score(backbone, tokens).mean()  # ~1: blind to time
```

`suggest_time_encoding(..., spacing="quantile", samples=None, mix=1.0)` returns the same report with a data-driven ladder: `report.wavelengths` follows the quantiles of the pooled shortest, peak and longest timescales of the detected curves (or of `samples`, e.g. a catalogue's periods and their harmonics), and `report.timescales` is that ladder in position units for `RoMAE(..., rope_timescales=report.timescales)` (`report.base` is then only the log-spaced equivalent). `spacing="linear"` is the arithmetic ladder. The training scripts expose this as `--time-spacing {log,linear,quantile} --time-mix`, next to `--auto-time`, and `--rope-wavelengths` (days) sets any ladder by hand.

`suggest_time_encoding` works curve by curve: each light curve gets a generalised Lomb-Scargle periodogram on a linear grid fine enough to resolve peaks over its own baseline (`max_evals` bounds the cost by subsampling observations, never the grid; at least 256 observations are kept), a curve counts as detected when its peak clears the noise floor `1 - (1 - median) ** (2 log2 n_freq)` (the Beta tail of a white-noise periodogram pushed through its median), and its support runs out to the highest harmonic holding a tenth of the peak power (`rel`). The pooled support takes quantiles over the detected curves (`q`), so a broad-band class cannot hide the short periods of another; `lam_min` and `lam_max` override the two ends when you know them (the shortest period in a catalogue, the survey baseline). Curves with fewer than three distinct epochs are skipped (`report.n_skipped`). `plot_time_encoding(report)` draws the detected periods against the ladder; `rotary_wavelengths` / `time_encoding` are the ladder and its inverse, `rotary_kernel` the lag response of a ladder and `wave_vectors` the nD-RoPE wave vectors of a `BlockRope`, for figures like those of the nD-RoPE paper. `time_shuffle_score` is only informative for a trained encoder: at initialisation attention is nearly uniform, so every untrained model scores about 1. It is a batch statistic (embeddings are standardised over the batch, so a single curve raises): to compare or pool the scores of several batches pass the same `stats=(mean, std)` of a reference embedding set to every call.

## Data

A `Record` holds one object: flat float32 arrays `t` (days), `y`, `err` over all bands, int64 `band` ids, a class `label`, a `period` and a `meta` dict; build them directly for your own data. `simulate` is a port of the lc-sim-model toy generator (sinusoid, RR Lyrae, eclipsing, double-mode, damped random walk, with band-dependent amplitude and phase lag) evaluated at irregular epochs inside yearly seasons; `SurveyConfig` and `SimConfig` set the bands and the population. `normalize` standardises fluxes per band by median/MAD (falling back to the std, then to a scale of 1, when the MAD is zero) and `make_views` cuts the LeJEPA windows:

```python
import numpy as np

from romae_lc import CLASSES, make_views

records = simulate(16, seed=0)  # bands g, r, i over a 1000 d baseline
r = normalize(records[0])
print(r.n, r.bands, CLASSES[r.label], r.period, sorted(r.meta))
rng = np.random.default_rng(0)
global_views, local_views = make_views(r, ViewConfig(), rng)  # (t, y, band) triples
```

`LightCurveDataset(records, ViewConfig())` with `collate` (a `functools.partial` carrying the `tokenize` keywords, as in the quickstart) yields `dict(global=[Tokens, ...], local=[...], full=Tokens, label, index)` batches with fresh views every epoch; `full` is the whole curve unless capped by `max_tokens` (the dataset's own argument, else the `ViewConfig`'s), in which case it is a random subsample re-drawn per epoch; `time_series(records)` gives the `(times, values, bands)` lists for `suggest_time_encoding`.

**PHOEBE eclipsing binaries.** `load_phoebe(source, split)` reads the PHOEBE 2 dataset of lc-sim-model from a Hugging Face repo id or a directory written by `DatasetDict.save_to_disk`. Rows carry `<band>_time`, `<band>_flux` and `<band>_flux_err` for each band in `PHOEBE_BANDS` (`Gaia_G`, `LSST_g`, `LSST_r`, `LSST_i`, `TESS_T`) plus `id`, `period`, `t0` and `morphology` (`contact`, `detached`, `semidetached` become labels 0, 1, 2):

```python
from romae_lc import PHOEBE_BANDS, load_phoebe

records = [normalize(r) for r in load_phoebe("path/to/phoebe", split="train")]
band_wavelengths = wavelengths_for(PHOEBE_BANDS)  # {0: 641.5, 1: 480.7, ...}
```

## Examples

```bash
uv run python examples/train_lejepa.py --n 2048 --epochs 20 --auto-time
uv run python examples/train_mae.py --n 2048 --epochs 20 --rope simplex --attention linear --auto-time
uv run python examples/train_lewm.py --n 2048 --epochs 20 --auto-time --window 30 --advance 1 2
uv run python examples/analyze_time_encoding.py --n 512 --plot runs/time_encoding.png --ckpt runs/lejepa.pt
```

The training scripts take `--data sim` (default), a PHOEBE source, or a PC_matches sub-dataset directory (see below), the model arguments `--width --depth --heads --rope --attention --p-rope`, the MAE decoder's own `--decoder-width --decoder-depth --decoder-heads` (default tiny-shallow, whatever the encoder size), the time unit as `--time-scale --rope-base` or `--auto-time` (the `suggest_time_encoding` recommendation on the training curves), and print a linear-probe accuracy (class) and a ridge R2 (log period) of the frozen embeddings every `--eval-every` epochs. Checkpoints in `runs/` hold the state dict, the backbone's `hparams` and the tokenizer keywords; `examples/common.py` restores them with `load_backbone`. `train_lewm.py` additionally writes `<out>_wm.pt` (the whole `LeWorldModel` state dict, its `hparams`, the backbone's `hparams`, the tokenizer keywords, the `FrameConfig` and a `no_actions` flag), restored with `load_world_model(path) -> (model, tokenize_kwargs, cfg)`. Its validation losses are computed in eval mode (BatchNorm running statistics, no predictor dropout, as the official validation step), so in very short runs `val pred` lags the train-mode `pred` until the running statistics catch up, and SIGReg is the Epps-Pulley statistic on a batch of latents, which carries a factor of the batch size: `val sigreg` (printed with its batch size) is on the training scale only when the validation batches have `--batch-size` rows. A CPU smoke run is `--n 256 --epochs 2 --width 96 --depth 2 --heads 3`; on a shared many-core node set `OMP_NUM_THREADS` to something like 16, since torch's default thread count oversubscribes it. The analysis script prints the report, plots it and, given a checkpoint, the mean `time_shuffle_score` on the validation curves.


```bash
python examples/train_lejepa.py ... --time-scale 0.0015915 \
  --rope-wavelengths 0.01 0.053 0.107 0.218 0.428 0.835 1.64 3.43 6.98 14.3 34.9 6000
```

Do it for Le world model

## Acknowledgements

- RoMAE, Zivanovic et al. 2025, [arXiv:2505.20535](https://arxiv.org/abs/2505.20535): the model, the masked pretraining recipe and the tiny-shallow/tiny/small/base encoder sizes.
- nD-RoPE, Li et al. 2026, [arXiv:2606.12146](https://arxiv.org/abs/2606.12146): the simplex wave vectors, the `theta` bound and the frequency-domain view of the encoding.
- LeJEPA, Balestriero and LeCun 2025, [arXiv:2511.08544](https://arxiv.org/abs/2511.08544): SIGReg and the multi-view objective. Next-latent prediction on top of SIGReg is the LeWorldModel of the Quickstart.
- LeWorldModel, Maes, Le Lidec, Scieur, LeCun, Balestriero 2026, [arXiv:2603.19312](https://arxiv.org/abs/2603.19312): the predictor, AdaLN-zero conditioning, projector and loss, ported from [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) (MIT).
- [Hibb-bb/lc-sim-model](https://github.com/Hibb-bb/lc-sim-model): the toy simulator ported in `romae_lc.data` and the PHOEBE dataset.
- [stable-pretraining](https://github.com/Hibb-bb/LeJEPA-Code): the light-curve RoMAE, rotary and LeJEPA code was ported and trimmed from its light-curve experiments.

MIT license.
