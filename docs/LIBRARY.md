# The `romae_lc` library

This page is the reference of the library in `romae_lc/`. The research code that trains and tests
encoders with it is in `project/`; the root `README.md` explains that part.

The old LeJEPA and LeWorldModel classes (`romae_lc/lejepa.py`, `romae_lc/lewm.py`) and the example
scripts moved to `legacy/` in the clean-up. They are not described here.

RoMAE for light curves: a continuous rotary transformer in which every observation of an asynchronous multi-band light curve is one token with a real-valued (time, wavelength) position, so nothing is binned, aligned or imputed. It ships axial p-RoPE and nD-RoPE (simplex) position encodings, softmax or linear attention, masked-autoencoding pretraining (RoMAE) and an analysis module that reads the right rotary time unit off a dataset. The library depends on `torch` and `numpy` only; `datasets` (PHOEBE loader) and `matplotlib` (plots) are optional extras.

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

The nested columns are read as Arrow arrays (about 1 ms per row). `project/data.py` (`load_data`) is the loader the training and evaluation scripts use.

## Linear attention

`attention="linear"` in an encoder (or decoder) config replaces softmax attention by kernel attention with `phi(x) = elu(x) + 1` features and the rotary phase applied inside the numerator, so the cost is linear in the number of tokens and the position encoding is unchanged:

```python
backbone = RoMAE(encoder=dict(d_model=96, nhead=3, depth=2, attention="linear"))
```

Numerator and denominator are accumulated in float32. Only key (padding) masks are supported: the mask has to be the same for every query, which is what `romae_lc.transformer.attention_mask(pad_mask)` produces; a query-dependent mask raises `NotImplementedError`.

## Choosing the time unit

A rotary channel of wavelength `lambda` turns once per lag `lambda`: it separates tokens closer than that and wraps around beyond. The active channels of the time block form the ladder `2 pi time_scale base ** (2 i / dim)` days, so the ladder has to reach below the shortest timescale that carries signal (the high-frequency end of the periodogram, harmonics included) and up to the baseline of the curves. Time in days with the language-model default `base = 1e4` puts the shortest wavelength at `2 pi` days: sub-day periods are invisible and the encoder degrades into a bag of magnitudes. `suggest_time_encoding` measures both ends with a generalised Lomb-Scargle periodogram over the dataset and inverts the ladder to `time_scale` and `base`; `time_shuffle_score` is the check on an encoder.

```python
from romae_lc import DEFAULT_SURVEYS, normalize, simulate
from romae_lc import suggest_time_encoding, time_series, time_shuffle_score

records = [normalize(r) for r in simulate(128, seed=0)]
wavelengths = {i: s.wavelength_nm for i, s in enumerate(DEFAULT_SURVEYS)}  # the simulator's bands
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

`suggest_time_encoding(..., spacing="quantile", samples=None, mix=1.0)` returns the same report with a data-driven ladder: `report.wavelengths` follows the quantiles of the pooled shortest, peak and longest timescales of the detected curves (or of `samples`, e.g. a catalogue's periods and their harmonics), and `report.timescales` is that ladder in position units for `RoMAE(..., rope_timescales=report.timescales)` (`report.base` is then only the log-spaced equivalent). `spacing="linear"` is the arithmetic ladder. `project/backbone.py` (`get_ladder`) measures the ladder of a training run this way.

`suggest_time_encoding` works curve by curve: each light curve gets a generalised Lomb-Scargle periodogram on a linear grid fine enough to resolve peaks over its own baseline (`max_evals` bounds the cost by subsampling observations, never the grid; at least 256 observations are kept), a curve counts as detected when its peak clears the noise floor `1 - (1 - median) ** (2 log2 n_freq)` (the Beta tail of a white-noise periodogram pushed through its median), and its support runs out to the highest harmonic holding a tenth of the peak power (`rel`). The pooled support takes quantiles over the detected curves (`q`), so a broad-band class cannot hide the short periods of another; `lam_min` and `lam_max` override the two ends when you know them (the shortest period in a catalogue, the survey baseline). Curves with fewer than three distinct epochs are skipped (`report.n_skipped`). `plot_time_encoding(report)` draws the detected periods against the ladder; `rotary_wavelengths` / `time_encoding` are the ladder and its inverse, `rotary_kernel` the lag response of a ladder and `wave_vectors` the nD-RoPE wave vectors of a `BlockRope`, for figures like those of the nD-RoPE paper. `time_shuffle_score` is only informative for a trained encoder: at initialisation attention is nearly uniform, so every untrained model scores about 1. It is a batch statistic (embeddings are standardised over the batch, so a single curve raises): to compare or pool the scores of several batches pass the same `stats=(mean, std)` of a reference embedding set to every call.

## Data

A `Record` holds one object: flat float32 arrays `t` (days), `y`, `err` over all bands, int64 `band` ids, a class `label`, a `period` and a `meta` dict; build them directly for your own data. `simulate` is a port of the lc-sim-model toy generator (sinusoid, RR Lyrae, eclipsing, double-mode, damped random walk, with band-dependent amplitude and phase lag) evaluated at irregular epochs inside yearly seasons; `SurveyConfig` and `SimConfig` set the bands and the population. `normalize` standardises fluxes per band by median/MAD (falling back to the std, then to a scale of 1, when the MAD is zero) and `make_views` cuts global and local views of a record:

```python
import numpy as np

from romae_lc import CLASSES, ViewConfig, make_views

records = simulate(16, seed=0)  # bands g, r, i over a 1000 d baseline
r = normalize(records[0])
print(r.n, r.bands, CLASSES[r.label], r.period, sorted(r.meta))
rng = np.random.default_rng(0)
global_views, local_views = make_views(r, ViewConfig(), rng)  # (t, y, band) triples
```

`LightCurveDataset(records, ViewConfig())` with `collate` (a `functools.partial` carrying the `tokenize` keywords) yields `dict(global=[Tokens, ...], local=[...], full=Tokens, label, index)` batches with fresh views every epoch; `full` is the whole curve unless capped by `max_tokens` (the dataset's own argument, else the `ViewConfig`'s), in which case it is a random subsample re-drawn per epoch; `time_series(records)` gives the `(times, values, bands)` lists for `suggest_time_encoding`.

**PHOEBE eclipsing binaries.** `load_phoebe(source, split)` reads the PHOEBE 2 dataset of lc-sim-model from a Hugging Face repo id or a directory written by `DatasetDict.save_to_disk`. Rows carry `<band>_time`, `<band>_flux` and `<band>_flux_err` for each band in `PHOEBE_BANDS` (`Gaia_G`, `LSST_g`, `LSST_r`, `LSST_i`, `TESS_T`) plus `id`, `period`, `t0` and `morphology` (`contact`, `detached`, `semidetached` become labels 0, 1, 2):

```python
from romae_lc import PHOEBE_BANDS, load_phoebe

records = [normalize(r) for r in load_phoebe("path/to/phoebe", split="train")]
band_wavelengths = wavelengths_for(PHOEBE_BANDS)  # {0: 641.5, 1: 480.7, ...}
```

## Acknowledgements

- RoMAE, Zivanovic et al. 2025, [arXiv:2505.20535](https://arxiv.org/abs/2505.20535): the model, the masked pretraining recipe and the tiny-shallow/tiny/small/base encoder sizes.
- nD-RoPE, Li et al. 2026, [arXiv:2606.12146](https://arxiv.org/abs/2606.12146): the simplex wave vectors, the `theta` bound and the frequency-domain view of the encoding.
- LeJEPA, Balestriero and LeCun 2025, [arXiv:2511.08544](https://arxiv.org/abs/2511.08544): SIGReg and the multi-view objective (`project/jepa.py`, the `enc_le1` recipe).
- LeWorldModel, Maes, Le Lidec, Scieur, LeCun, Balestriero 2026, [arXiv:2603.19312](https://arxiv.org/abs/2603.19312): the frame windows in `romae_lc/frames.py`; the model itself is in `legacy/romae_lc/lewm.py`, ported from [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) (MIT).
- [Hibb-bb/lc-sim-model](https://github.com/Hibb-bb/lc-sim-model): the toy simulator ported in `romae_lc.data` and the PHOEBE dataset.
- [stable-pretraining](https://github.com/Hibb-bb/LeJEPA-Code): the light-curve RoMAE, rotary and LeJEPA code was ported and trimmed from its light-curve experiments.

MIT license.
