"""Light-curve records, a toy simulator, PHOEBE loading, views and collation.

Everything downstream works on :class:`Record`: one object's observations in
all bands as flat arrays (``t`` in days, ``y``, ``err``, integer ``band``)
with a class label and a period. Records come from :func:`simulate`, a
compact port of the toy variable-star generator in Hibb-bb/lc-sim-model
(``lcsim/simulate.py``) that keeps its five classes, per-star latents and
band-dependent amplitude and phase lag but evaluates the shapes at irregular
epochs inside yearly observing seasons, or from :func:`load_phoebe`, which
reads the PHOEBE 2 eclipsing-binary dataset of the same repository.

:func:`normalize` standardises fluxes per band, :func:`sample_view` and
:func:`make_views` cut the random time windows LeJEPA compares, and
:class:`LightCurveDataset` with :func:`collate` turn records into padded
:class:`~romae_lc.tokenize.Tokens` batches via :func:`~romae_lc.tokenize.tokenize`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenize import tokenize

LAMBDA_REF = 550.0  # nm; the simulated amplitude is defined at this wavelength
CLASSES = ("sinusoid", "rrlyrae", "eclipsing", "doublemode", "drw")
PHOEBE_BANDS = ("Gaia_G", "LSST_g", "LSST_r", "LSST_i", "TESS_T")
PHOEBE_MORPHOLOGIES = ("contact", "detached", "semidetached")


@dataclass
class Record:
    """One object's observations in every band as flat per-point arrays:
    epochs ``t`` in days, flux ``y`` (any unit, see :func:`normalize`), its
    one-sigma ``err`` (all float32) and int64 ``band`` ids indexing the band
    list the record came from (``DEFAULT_SURVEYS``, ``PHOEBE_BANDS``), plus
    the class ``label``, the ``period`` in days (the timescale of aperiodic
    classes) and extra latents or catalogue fields in ``meta``. Points are
    grouped by band and sorted in time within a band, so ``t`` is not
    globally sorted (the tokenizer sorts)."""

    t: np.ndarray
    y: np.ndarray
    err: np.ndarray
    band: np.ndarray
    label: int
    period: float
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def bands(self) -> np.ndarray:
        """Sorted unique band ids."""
        return np.unique(self.band)


@dataclass
class SurveyConfig:
    """One simulated band: ``n_points`` epochs per star drawn inside observing
    seasons of ``season_len`` days per ``year``, with per-point noise
    ``sigma`` in flux units (scattered by U(0.7, 1.3))."""

    name: str
    wavelength_nm: float
    n_points: int
    sigma: float
    season_len: float = 240.0
    year: float = 365.25


@dataclass
class SimConfig:
    """Population and shape parameters of :func:`simulate`: epochs in
    ``[0, baseline_days]``; uniform ranges of log10 period, amplitude at
    550 nm and the narrow-bump amplitude ``fine`` (bump width and centre in
    phase units); the colour model ``amp * (550 / lambda) ** (color_gamma *
    color)`` with phase lag ``color_lag * color * (lambda - 550) / 550``;
    class weights (uniform when ``None``)."""

    baseline_days: float = 1000.0
    logP_range: tuple[float, float] = (-0.7, 1.5)
    amp_range: tuple[float, float] = (0.2, 1.0)
    fine_range: tuple[float, float] = (0.0, 0.25)
    bump_width: float = 0.06
    bump_phase: float = 0.72
    color_gamma: float = 1.5
    color_lag: float = 0.04
    class_probs: tuple[float, ...] | None = None


DEFAULT_SURVEYS = (
    SurveyConfig("g", 480.7, 300, 0.02),
    SurveyConfig("r", 622.1, 200, 0.03),
    SurveyConfig("i", 755.9, 100, 0.05),
)


def _sinusoid(ph, shape):
    return 0.5 * np.sin(2 * np.pi * ph)


def _rrlyrae(ph, shape):
    """Fast rise over ``rise`` of the phase, slow decline, corners softened."""
    rise = 0.10 + 0.15 * shape
    x = np.mod(ph, 1.0)
    y = np.where(x < rise, -0.5 + x / rise, 0.5 - (x - rise) / (1 - rise))
    return y - 0.08 * np.sin(4 * np.pi * x)


def _eclipsing(ph, shape):
    """Gaussian primary eclipse at phase 0 and a 45% secondary at 0.5."""
    w = 0.04 + 0.06 * shape
    x, x2 = np.mod(ph + 0.5, 1.0) - 0.5, np.mod(ph, 1.0) - 0.5
    y = -(np.exp(-0.5 * (x / w) ** 2) + 0.45 * np.exp(-0.5 * (x2 / w) ** 2))
    return y - y.mean()


def _doublemode(ph, shape):
    ratio = 0.72 + 0.06 * shape
    return 0.35 * np.sin(2 * np.pi * ph) + 0.2 * np.sin(2 * np.pi * ph / ratio + 1.0)


_SHAPES = (_sinusoid, _rrlyrae, _eclipsing, _doublemode)


def _drw(t: np.ndarray, tau: float, rng: np.random.Generator) -> np.ndarray:
    """Damped random walk (OU process) at sorted epochs ``t``: zero mean,
    standard deviation 0.35 like the grid version."""
    rho = np.exp(-np.diff(t) / tau).tolist()
    jump = np.sqrt(1.0 - np.square(rho)).tolist()
    eps = rng.standard_normal(len(t)).tolist()
    y = [eps[0]]
    for r, j, e in zip(rho, jump, eps[1:]):
        y.append(r * y[-1] + j * e)
    y = np.asarray(y)
    y = y / (y.std() + 1e-8) * 0.35
    return y - y.mean()


def _bump(ph, width, center):
    x = np.mod(ph - center + 0.5, 1.0) - 0.5
    return np.exp(-0.5 * (x / width) ** 2)


def sample_epochs(rng, n_points: int, baseline: float, season_len: float, year: float):
    """``n_points`` sorted uniform epochs in ``[0, baseline]`` kept inside
    yearly seasons of ``season_len`` days (random season offset), as in the
    PHOEBE generator."""
    offset = rng.uniform(0, year)
    t = rng.uniform(0, baseline, size=3 * n_points)
    t = t[((t + offset) % year) < season_len]
    if len(t) < n_points:
        t = np.concatenate([t, rng.uniform(0, baseline, size=n_points - len(t))])
    return np.sort(rng.choice(t, size=n_points, replace=False))


def _flux(lat: dict, epochs: list[np.ndarray], surveys, cfg: SimConfig, rng):
    """Noise-free flux of one star in every band at its own epochs."""
    period = 10.0 ** lat["logP"]
    is_drw = lat["cls"] == len(_SHAPES)
    if is_drw:  # one OU process shared by all bands
        t_all = np.concatenate(epochs)
        order = np.argsort(t_all)
        base = np.empty_like(t_all)
        base[order] = _drw(t_all[order], period, rng)
        bases = np.split(base, np.cumsum([len(t) for t in epochs])[:-1])
    out = []
    for i, (t, s) in enumerate(zip(epochs, surveys)):
        wl = s.wavelength_nm
        amp = lat["amp"] * (LAMBDA_REF / wl) ** (cfg.color_gamma * lat["color"])
        lag = cfg.color_lag * lat["color"] * (wl - LAMBDA_REF) / LAMBDA_REF
        ph = t / period + lat["phase"] + lag
        y = bases[i] if is_drw else _SHAPES[lat["cls"]](ph, lat["shape"])
        out.append(amp * (y + lat["fine"] * _bump(ph, cfg.bump_width, cfg.bump_phase)))
    return out


def simulate(
    n: int,
    surveys: Sequence[SurveyConfig] = DEFAULT_SURVEYS,
    cfg: SimConfig | None = None,
    seed: int = 0,
) -> list[Record]:
    """Simulate ``n`` variable stars observed in every survey band.

    Per-star latents ``cls, logP, amp, phase, color, fine, shape`` are drawn
    first; each band then gets its own irregular epochs, the class shape at
    those epochs plus the narrow bump, band-dependent amplitude and phase
    lag, and Gaussian noise ``sigma * U(0.7, 1.3)`` per point. The DRW class
    is one OU process on the union of all epochs. Deterministic given ``seed``.

    Args:
        n: Number of stars.
        surveys: Bands; ``band`` ids index this sequence.
        cfg: Population parameters, default :class:`SimConfig`.
        seed: RNG seed.

    Returns:
        Records with ``label = cls``, ``period = 10 ** logP`` and the other
        latents in ``meta``.
    """
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    if cfg.class_probs is None:
        cls = rng.integers(0, len(CLASSES), size=n)
    else:
        pr = np.asarray(cfg.class_probs, dtype=float)
        cls = rng.choice(len(CLASSES), size=n, p=pr / pr.sum())
    latents = dict(
        cls=cls,
        logP=rng.uniform(*cfg.logP_range, size=n),
        amp=rng.uniform(*cfg.amp_range, size=n),
        phase=rng.uniform(0, 1, size=n),
        color=rng.uniform(0, 1, size=n),
        fine=rng.uniform(*cfg.fine_range, size=n),
        shape=rng.uniform(0, 1, size=n),
    )
    band = np.repeat(np.arange(len(surveys)), [s.n_points for s in surveys])
    records = []
    for i in range(n):
        lat = {k: v[i].item() for k, v in latents.items()}
        epochs = [
            sample_epochs(rng, s.n_points, cfg.baseline_days, s.season_len, s.year)
            for s in surveys
        ]
        err = [s.sigma * rng.uniform(0.7, 1.3, size=s.n_points) for s in surveys]
        flux = _flux(lat, epochs, surveys, cfg, rng)
        y = [f + rng.standard_normal(len(f)) * e for f, e in zip(flux, err)]
        t, y, err = (np.concatenate(a).astype(np.float32) for a in (epochs, y, err))
        meta = {k: lat[k] for k in ("logP", "amp", "color", "fine", "shape")}
        label, period = lat["cls"], 10.0 ** lat["logP"]
        records.append(Record(t, y, err, band.copy(), label, period, meta))
    return records


def load_phoebe(
    source: str,
    split: str = "train",
    bands: Sequence[str] = PHOEBE_BANDS,
    max_rows: int | None = None,
) -> list[Record]:
    """Load the PHOEBE 2 eclipsing-binary dataset (Hibb-bb/lc-sim-model).

    Rows carry ``<band>_time``, ``<band>_flux`` and ``<band>_flux_err`` per
    band plus ``period``, ``t0`` and ``morphology``; the bands are
    concatenated in the order of ``bands`` (their ids index that sequence).

    Args:
        source: Hugging Face repo id, or a directory written by
            ``DatasetDict.save_to_disk``.
        split: ``"train"``, ``"validation"`` or ``"test"``.
        bands: Band names as stored in the columns (``"LSST_g"``).
        max_rows: Keep only the first rows.

    Returns:
        Records with ``label = PHOEBE_MORPHOLOGIES.index(morphology)`` and
        ``meta = dict(id, t0, morphology)``.
    """
    try:
        import datasets
    except ImportError as e:
        raise ImportError("load_phoebe needs 'datasets' (uv sync --extra hf)") from e
    if Path(source).is_dir():
        ds = datasets.load_from_disk(source)[split]
    else:
        ds = datasets.load_dataset(source, split=split)
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    keys = ("time", "flux", "flux_err")
    cols = ["id", "period", "t0", "morphology"]
    cols += [f"{b}_{k}" for b in bands for k in keys]
    records = []
    for row in ds.select_columns(cols).with_format("numpy"):
        t, y, err = (np.concatenate([row[f"{b}_{k}"] for b in bands]) for k in keys)
        t, y, err = (a.astype(np.float32) for a in (t, y, err))
        band = np.repeat(np.arange(len(bands)), [len(row[f"{b}_time"]) for b in bands])
        morph = str(row["morphology"])
        meta = dict(id=int(row["id"]), t0=float(row["t0"]), morphology=morph)
        label, period = PHOEBE_MORPHOLOGIES.index(morph), float(row["period"])
        records.append(Record(t, y, err, band, label, period, meta))
    return records


#: Band names of the PC_matches datasets (``lightcurve.<band>`` columns), in
#: a fixed order shared by every sub-dataset so that band ids agree across
#: single-survey and multi-survey (``*-isect``) sources.
PC_BANDS = (
    "g_ZTF",
    "r_ZTF",
    "i_ZTF",
    "c_ATLAS",
    "o_ATLAS",
    "g_ASASSN",
    "V_ASASSN",
    "V_CSS",
    "r_LINEAR",
    "g_PTF",
    "R_PTF",
    "J_PGIR",
    "TESS",
    "g_PS1",
    "r_PS1",
    "i_PS1",
    "z_PS1",
    "y_PS1",
    "G_Gaia",
    "BP_Gaia",
    "RP_Gaia",
)
#: Variability superclasses of PC_matches (``superclass_str``), label order.
PC_SUPERCLASSES = ("CEP", "DSCT", "ECL", "ELL", "LPV", "PCEB", "ROT", "RR")


def _pc_list_column(struct, name: str, dtype=np.float64, fill=np.nan):
    """``(values, offsets)`` of one ``lightcurve.<band>.<name>`` list field of
    a pyarrow struct array (``offsets[i]:offsets[i + 1]`` is row ``i``);
    nulls become ``fill`` and the values are cast to ``dtype``."""
    import pyarrow.compute as pc

    arr = struct.field(name)
    flat, lengths = pc.list_flatten(arr), pc.list_value_length(arr)
    values = np.asarray(
        flat.to_pylist() if flat.null_count else flat.to_numpy(zero_copy_only=False)
    )
    if values.dtype == object:
        values = np.array([fill if v is None else v for v in values], dtype=dtype)
    lengths = lengths.fill_null(0).to_numpy(zero_copy_only=False)
    return values.astype(dtype), np.concatenate([[0], np.cumsum(lengths)])


def load_pc(
    source: str,
    split: str = "train",
    bands: Sequence[str] = PC_BANDS,
    label_field: str = "superclass_str",
    classes: Sequence[str] | None = None,
    max_rows: int | None = None,
    min_points: int = 1,
    clean_only: bool = True,
) -> list[Record]:
    """Load one PC_matches sub-dataset (``/projects/bfrf/data/PC_matches/<name>``).

    Rows carry ``lightcurve.<band>.{mjd, mag, mag_unc, clean?, quality_flag?}``
    for the bands of that survey, plus ``period`` (days), ``class_str`` and
    ``superclass_str``. The bands present in the dataset are read in the
    order of ``bands`` (ids index that sequence, so ``wavelengths_for(bands)``
    is the tokenizer's band table; bands absent from the dataset simply never
    occur). The nested columns are read as Arrow arrays, not row by row.

    Values are ``-mag`` (brighter is up; :func:`normalize` removes the offset
    and scale per band anyway), errors ``mag_unc``. Points flagged unclean
    (``clean == False`` or ``quality_flag != 0``) are dropped when
    ``clean_only``, as are non-finite points. Times are MJD re-zeroed at each
    record's first point (``meta["t0"]`` keeps the offset): positions are
    float32 in the tokenizer and an absolute MJD would lose the intra-night
    resolution. Records with fewer than ``min_points`` points or without a
    positive period are dropped.

    Args:
        source: A directory written by ``DatasetDict.save_to_disk`` (one
            sub-dataset) or a Hugging Face repo id.
        split: ``"train"``, ``"validation"`` or ``"test"``.
        bands: Band names to read, in id order.
        label_field: ``"superclass_str"`` (8 classes, :data:`PC_SUPERCLASSES`)
            or ``"class_str"`` (finer, then pass ``classes`` so that every
            split uses the same vocabulary).
        classes: Label vocabulary; defaults to :data:`PC_SUPERCLASSES` for
            the superclass field and to the sorted values of the split
            otherwise. Rows with a label outside it are dropped.
        max_rows: Keep only the first rows.
        min_points: Fewest clean points a record must keep.
        clean_only: Apply the per-point quality flags.

    Returns:
        Records with ``label = classes.index(row[label_field])``, ``period``
        and ``meta = dict(id, class_str, superclass_str, t0, source)``.
    """
    try:
        import datasets
        import pyarrow as pa
    except ImportError as e:
        raise ImportError("load_pc needs 'datasets' (uv sync --extra hf)") from e
    if Path(source).is_dir():
        ds = datasets.load_from_disk(source)[split]
    else:
        ds = datasets.load_dataset(source, split=split)
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    n = len(ds)
    if classes is None:
        classes = (
            PC_SUPERCLASSES
            if label_field == "superclass_str"
            else tuple(sorted(set(ds[label_field])))
        )
    lookup = {c: i for i, c in enumerate(classes)}
    labels = np.array([lookup.get(c, -1) for c in ds[label_field]])
    periods = np.array(
        [np.nan if p is None else float(p) for p in ds["period"]], dtype=np.float64
    )
    class_str, super_str = ds["class_str"], ds["superclass_str"]
    ids = ds["gaia_dr3_source_id"] if "gaia_dr3_source_id" in ds.column_names else None

    lc = ds.data.column("lightcurve").combine_chunks()  # StructArray
    present = [b for b in bands if b in lc.type.names]
    if not present:
        raise ValueError(f"none of {tuple(bands)} in {lc.type.names}")
    per_band = []
    for b in present:
        band = lc.field(b)
        t, off = _pc_list_column(band, "mjd")
        y, _ = _pc_list_column(band, "mag")
        e, _ = _pc_list_column(band, "mag_unc")
        keep = np.isfinite(t) & np.isfinite(y)
        if clean_only:
            if "clean" in band.type.names:
                c, _ = _pc_list_column(band, "clean", bool, True)
                keep &= c
            if "quality_flag" in band.type.names:
                q, _ = _pc_list_column(band, "quality_flag", np.int64, 0)
                keep &= q == 0
        e = np.where(np.isfinite(e) & (e > 0), e, np.nan)
        per_band.append((bands.index(b), t, y, e, keep, off))

    records = []
    for i in range(n):
        if labels[i] < 0 or not periods[i] > 0:
            continue
        ts, ys, es, bs = [], [], [], []
        for bid, t, y, e, keep, off in per_band:
            lo, hi = off[i], off[i + 1]
            m = keep[lo:hi]
            if not m.any():
                continue
            ts.append(t[lo:hi][m])
            ys.append(-y[lo:hi][m])
            es.append(e[lo:hi][m])
            bs.append(np.full(int(m.sum()), bid, dtype=np.int64))
        if not ts:
            continue
        t = np.concatenate(ts)
        if t.size < min_points:
            continue
        err = np.concatenate(es)
        if np.isnan(err).any():
            fill = np.nanmedian(err) if np.isfinite(err).any() else 1.0
            err = np.where(np.isfinite(err), err, fill)
        t0 = float(t.min())
        order = np.lexsort((t, np.concatenate(bs)))  # by band, then time
        meta = dict(
            id=None if ids is None else ids[i],
            class_str=class_str[i],
            superclass_str=super_str[i],
            t0=t0,
            source=str(source),
        )
        records.append(
            Record(
                (t - t0).astype(np.float32)[order],
                np.concatenate(ys).astype(np.float32)[order],
                err.astype(np.float32)[order],
                np.concatenate(bs)[order],
                int(labels[i]),
                float(periods[i]),
                meta,
            )
        )
    return records


def normalize(record: Record, mode: str = "band", eps: float = 1e-6) -> Record:
    """Robust standardisation ``(y - median) / (1.4826 * MAD)``.

    A group whose MAD is at most ``eps`` (more than half of its fluxes
    identical, e.g. quantised or saturated data) falls back to the standard
    deviation, and one without any spread (constant or a single point) to a
    unit scale, so it is centred but ``err`` keeps its flux units instead of
    being inflated.

    Args:
        record: Input record (not modified).
        mode: ``"band"`` standardises each band separately, ``"object"``
            all points together.
        eps: Smallest scale accepted before falling back.

    Returns:
        A new record; ``err`` is divided by the same scale.
    """
    if mode not in ("band", "object"):
        raise ValueError(f"mode must be 'band' or 'object', got {mode!r}")
    y, err = record.y.astype(np.float32), record.err.astype(np.float32)
    for b in record.bands if mode == "band" else [None]:
        g = slice(None) if b is None else record.band == b
        med = np.median(y[g])
        scale = 1.4826 * float(np.median(np.abs(y[g] - med)))
        if scale <= eps:
            scale = float(y[g].std())
        if scale <= eps:
            scale = 1.0
        y[g] = (y[g] - med) / scale
        err[g] = err[g] / scale
    return replace(record, y=y, err=err)


@dataclass
class ViewConfig:
    """How :func:`make_views` cuts LeJEPA views: ``n_global`` / ``n_local``
    views whose window length is a random fraction (``global_frac`` /
    ``local_frac``) of the time span, at most ``max_tokens`` points each
    (random subsample), with a random fraction in ``drop_frac`` of the points
    dropped and ``y`` redrawn from ``N(y, err)`` when ``resample``; views
    with fewer than ``min_tokens`` points fall back to the whole curve."""

    n_global: int = 2
    n_local: int = 4
    global_frac: tuple[float, float] = (0.5, 1.0)
    local_frac: tuple[float, float] = (0.1, 0.4)
    max_tokens: int = 512
    resample: bool = True
    drop_frac: tuple[float, float] = (0.0, 0.3)
    min_tokens: int = 8


def sample_view(
    record: Record,
    rng: np.random.Generator,
    frac_range: tuple[float, float],
    cfg: ViewConfig,
    augment: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One random contiguous time window of a record, all bands.

    The window length is ``U(*frac_range) * span``; ``augment`` applies the
    point dropping and resampling of ``cfg``. Times are relative to the
    window start (the first epoch after a whole-curve fallback). The window
    arithmetic runs in float64 so a whole-span window keeps the last epoch.

    Returns:
        ``(t, y, band)`` arrays of the selected points.
    """
    t = record.t.astype(np.float64)
    t0, span = t.min(), t.max() - t.min()
    length = rng.uniform(*frac_range) * span
    start = t0 + rng.uniform(0, span - length)
    if length >= span:  # the whole curve, independent of the mask
        idx = np.arange(record.n)
    else:
        idx = np.flatnonzero((t >= start) & (t <= start + length))
    if augment:
        idx = idx[rng.uniform(size=len(idx)) >= rng.uniform(*cfg.drop_frac)]
    if len(idx) < cfg.min_tokens:
        idx, start = np.arange(record.n), t0
    if len(idx) > cfg.max_tokens:
        idx = np.sort(rng.choice(idx, size=cfg.max_tokens, replace=False))
    y = record.y[idx]
    if augment and cfg.resample:
        y = y + rng.standard_normal(len(idx)) * record.err[idx]
    return (t[idx] - start).astype(np.float32), y.astype(np.float32), record.band[idx]


def make_views(record: Record, cfg: ViewConfig, rng: np.random.Generator):
    """``(global_views, local_views)``: lists of :func:`sample_view` triples."""
    glob = [sample_view(record, rng, cfg.global_frac, cfg) for _ in range(cfg.n_global)]
    loc = [sample_view(record, rng, cfg.local_frac, cfg) for _ in range(cfg.n_local)]
    return glob, loc


class LightCurveDataset(Dataset):
    """Records as a map-style dataset yielding views and the full curve.

    Each item is ``dict(views, full, label, index)``: ``views`` is
    ``make_views(record, view_cfg, rng)`` (``None`` without a ``view_cfg``)
    and ``full`` the whole curve as a ``(t, y, band)`` triple with the time
    origin at its first epoch. ``full`` is a random subsample of at most
    ``max_tokens`` points when that is given, else of ``view_cfg.max_tokens``
    when a ``view_cfg`` is given, and the whole curve otherwise. The rng is
    seeded from ``(seed, index)`` plus, with ``epoch_seed``, the number of
    times the item was requested and ``torch.initial_seed()`` (refreshed by
    the DataLoader in every worker each epoch), so views (and a capped
    ``full``) differ from epoch to epoch; ``epoch_seed=False`` makes every
    item deterministic.

    Args:
        records: The light curves.
        view_cfg: View sampling parameters; ``None`` yields no views.
        seed: Base seed.
        epoch_seed: Re-randomise items on every request.
        max_tokens: Cap on the points of ``full``; ``None`` defers to
            ``view_cfg.max_tokens`` or, without a ``view_cfg``, no cap.
    """

    def __init__(
        self,
        records: Sequence[Record],
        view_cfg: ViewConfig | None = None,
        seed: int = 0,
        epoch_seed: bool = True,
        max_tokens: int | None = None,
    ):
        self.records = list(records)
        self.view_cfg, self.seed, self.epoch_seed = view_cfg, seed, epoch_seed
        self.max_tokens = max_tokens
        self.calls = [0] * len(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        entropy = [self.seed, index]
        if self.epoch_seed:
            self.calls[index] += 1
            entropy += [self.calls[index], torch.initial_seed()]
        rng = np.random.default_rng(entropy)
        record, cfg = self.records[index], self.view_cfg
        views = make_views(record, cfg, rng) if cfg is not None else None
        cap = self.max_tokens
        if cap is None:
            cap = cfg.max_tokens if cfg is not None else record.n
        full_cfg = replace(cfg or ViewConfig(), max_tokens=cap)
        full = sample_view(record, rng, (1.0, 1.0), full_cfg, augment=False)
        return dict(views=views, full=full, label=record.label, index=index)


def collate(batch: list[dict], **tokenize_kwargs) -> dict:
    """DataLoader ``collate_fn`` for :class:`LightCurveDataset` items; use
    ``functools.partial(collate, band_wavelengths=..., time_scale=...)`` to
    pass the :func:`~romae_lc.tokenize.tokenize` keywords.

    Returns:
        ``dict(global=[Tokens per global view], local=[...], full=Tokens,
        label=LongTensor, index=LongTensor)``; the view lists are empty when
        the items carry no views.
    """

    def tok(triples):
        cols = [[torch.from_numpy(a) for a in c] for c in zip(*triples)]
        return tokenize(*cols, **tokenize_kwargs)

    out = dict(
        full=tok([item["full"] for item in batch]),
        label=torch.tensor([item["label"] for item in batch], dtype=torch.long),
        index=torch.tensor([item["index"] for item in batch], dtype=torch.long),
    )
    views = [item["views"] for item in batch]
    glob, loc = zip(*views) if views[0] is not None else ([], [])
    out["global"] = [tok(v) for v in zip(*glob)]
    out["local"] = [tok(v) for v in zip(*loc)]
    return out


def time_series(records: Sequence[Record]):
    """``(times, values, bands)`` lists of 1-D arrays for
    :func:`romae_lc.analysis.suggest_time_encoding`."""
    return [r.t for r in records], [r.y for r in records], [r.band for r in records]
