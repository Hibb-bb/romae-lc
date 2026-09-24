"""Choose and inspect the time encoding of a light-curve RoMAE.

A rotary channel of wavelength ``lambda`` completes one turn per position lag
``lambda``: tokens that far apart look coincident to it, tokens much closer
are barely told apart. With time positions in units of ``time_scale`` days,
an axial block of ``dim`` channels, base ``base`` and p-RoPE fraction ``p``
has ``n_ang = int(p * dim // 2)`` active angles whose wavelengths form the
ladder ``2 pi time_scale base ** (2 i / dim)`` days. To see phase, the ladder
must reach below the shortest timescale that carries signal (the highest
harmonic of the shortest periods) and up to the baseline of the curves. The
data tells you both, so the hyperparameters follow: ``time_scale = lam_min /
2 pi``, ``base = (lam_max / lam_min) ** (dim / (2 (n_ang - 1)))``.

The geometric ladder is only the default spacing. :func:`rotary_ladder`
builds a ladder of ``n`` wavelengths between ``lam_min`` and ``lam_max``
with log, linear or data-driven (quantile) spacing, and :func:`timescales_for`
turns it into the ``timescales`` an :class:`~romae_lc.rope.AxialRope` takes
(``RoMAE(..., rope_timescales=...)``), so the channel density can follow the
period distribution of a dataset instead of being uniform in log period.

This fixes a silent failure: time in days with the language-model default
``base = 1e4`` has a shortest wavelength of ``2 pi`` days, so sub-day periods
are invisible and the encoder degrades into a bag of magnitudes.
:func:`suggest_time_encoding` reads the band off a dataset, curve by curve,
with a generalised Lomb-Scargle periodogram, in the spirit of the
frequency-domain look at what a rotary encoding covers in nD-RoPE (Li et al.
2026, arXiv:2606.12146); :func:`time_shuffle_score` is the empirical check on
a trained encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from .rope import AxialRope, BlockRope, SimplexRope
from .tokenize import Tokens

#: Largest ``[n_freq, n_obs]`` broadcast evaluated at once (two float64 arrays).
_CHUNK_ELEMENTS = 20_000_000
#: Fewest observations :func:`periodogram` keeps when it subsamples.
_MIN_OBS = 256


def _gaps(times: np.ndarray) -> np.ndarray:
    """Positive gaps between sorted epochs."""
    dt = np.diff(np.sort(np.asarray(times, dtype=np.float64)))
    return dt[dt > 0]


def sampling_stats(times: np.ndarray) -> dict:
    """``n, span, dt_min, dt_median`` of one epoch array (gaps over sorted
    times; zero gaps from simultaneous bands do not count, so a curve with
    fewer than two distinct epochs has ``nan`` gaps)."""
    t = np.asarray(times, dtype=np.float64)
    dt = _gaps(t)
    return dict(
        n=int(t.size),
        span=float(np.ptp(t)) if t.size else 0.0,
        dt_min=float(dt.min()) if dt.size else float("nan"),
        dt_median=float(np.median(dt)) if dt.size else float("nan"),
    )


def lomb_scargle(
    t: np.ndarray, y: np.ndarray, freqs: np.ndarray, dy: np.ndarray | None = None
) -> np.ndarray:
    """Generalised (floating-mean) Lomb-Scargle power on a frequency grid.

    Zechmeister & Kürster (2009), normalised as ``(chi2_0 - chi2(f)) / chi2_0``
    so the power lies in [0, 1] and is invariant to the scale and offset of
    ``y``. Vectorised over the grid; the ``[n_freq, n_obs]`` broadcast is
    evaluated in chunks to bound memory.

    Args:
        t: Epochs ``[n_obs]`` (any order).
        y: Values ``[n_obs]``.
        freqs: Frequencies ``[n_freq]`` in cycles per unit of ``t``.
        dy: Optional uncertainties ``[n_obs]`` (inverse-variance weights).
    """
    t, y, freqs = (np.asarray(a, dtype=np.float64) for a in (t, y, freqs))
    w = np.ones_like(t) if dy is None else np.asarray(dy, dtype=np.float64) ** -2.0
    w = w / w.sum()
    y = y - w @ y
    wy, yy = w * y, w @ (y * y)
    power = np.zeros(freqs.shape)
    step = max(1, _CHUNK_ELEMENTS // max(t.size, 1))
    for lo in range(0, freqs.size, step):
        c = np.outer(2 * np.pi * freqs[lo : lo + step], t)
        s = np.sin(c)
        c = np.cos(c, out=c)
        cm, sm = c @ w, s @ w
        yc, ys = c @ wy, s @ wy
        cc = np.einsum("fi,fi,i->f", c, c, w) - cm * cm
        cs = np.einsum("fi,fi,i->f", c, s, w) - cm * sm
        ss = 1.0 - cc - cm * cm - sm * sm  # sum w = 1, so SS_hat = 1 - CC_hat
        num = ss * yc * yc + cc * ys * ys - 2 * cs * yc * ys
        den = yy * (cc * ss - cs * cs)
        np.divide(num, den, out=power[lo : lo + step], where=den > 0)
    return np.clip(power, 0.0, 1.0)


def periodogram(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray | None = None,
    min_freq: float | None = None,
    max_freq: float | None = None,
    oversample: float = 4.0,
    max_evals: float = 1e7,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(freqs, power)`` of one light curve on a linear frequency grid.

    Peaks are ``~1 / span`` wide, so the grid step is ``1 / (oversample *
    span)`` whatever the baseline; by default it runs from half a cycle over
    the baseline to the pseudo-Nyquist frequency of the 5th-percentile gap,
    which needs at least three distinct epochs. When ``n_freq * n_obs``
    exceeds ``max_evals`` a random subset of the observations is used (peak
    positions survive, sensitivity drops); the grid itself is never
    coarsened, and at least 256 observations (or all of them) are always
    kept, so ``max_evals`` is a soft budget. Values need no normalisation.
    """
    t, y = np.asarray(t, dtype=np.float64), np.asarray(y, dtype=np.float64)
    span = np.ptp(t) if t.size else 0.0
    if span <= 0:
        raise ValueError("periodogram needs at least 2 distinct epochs")
    if min_freq is None:
        min_freq = 0.5 / span
    if max_freq is None:
        max_freq = 0.5 / np.quantile(_gaps(t), 0.05)
    if not 0 < min_freq < max_freq:
        raise ValueError(f"need 0 < min_freq < max_freq, got {min_freq}, {max_freq}")
    n_freq = int(np.ceil((max_freq - min_freq) * oversample * span)) + 1
    freqs = np.linspace(min_freq, max_freq, n_freq)
    n_use = min(max(int(max_evals // n_freq), _MIN_OBS), t.size)
    if n_use < t.size:
        keep = (rng or np.random.default_rng(0)).choice(t.size, n_use, False)
        t, y = t[keep], y[keep]
        dy = None if dy is None else np.asarray(dy)[keep]
    return freqs, lomb_scargle(t, y, freqs, dy)


class Support(NamedTuple):
    """Frequency band of one periodogram: ``[f_lo, f_hi]`` holds every peak
    with at least ``rel`` of the maximum power, ``f_peak`` is the maximum and
    ``detected`` says whether it rose above the noise floor."""

    f_lo: float
    f_hi: float
    f_peak: float
    detected: bool


def spectral_support(
    freqs: np.ndarray,
    power: np.ndarray,
    rel: float = 0.1,
    noise_factor: float | None = None,
) -> Support:
    """Where one curve's signal lives in frequency, harmonics included.

    The noise floor is ``1 - (1 - median) ** noise_factor``: for white noise
    the power at one frequency has the tail ``(1 - z) ** b`` (Baluev 2008),
    so the median ``m`` fixes ``(1 - m) ** b = 1 / 2`` and the largest of
    ``n`` values sits near ``1 - (1 - m) ** log2(n)``, below the default
    ``noise_factor = 2 * log2(n_freq)``. This reads ``noise_factor * median``
    for the small medians of long curves and stays below the maximum power 1
    for short ones, where the median is of order ``1 / n_obs``. A curve is
    detected when its peak clears the floor; its support is the outermost
    pair of frequencies whose power reaches ``max(rel * peak, floor)``, so
    sharp features (eclipses) push ``f_hi`` up through their harmonics.
    Undetected curves get the grid ends.
    """
    freqs, power = np.asarray(freqs), np.asarray(power)
    if noise_factor is None:
        noise_factor = 2 * np.log2(max(freqs.size, 2))
    floor = 1.0 - (1.0 - np.median(power)) ** noise_factor
    i_peak = int(power.argmax())
    detected = bool(power[i_peak] > floor)
    if not detected:
        return Support(float(freqs[0]), float(freqs[-1]), float(freqs[i_peak]), False)
    above = np.flatnonzero(power >= max(rel * power[i_peak], floor))
    return Support(
        float(freqs[above[0]]), float(freqs[above[-1]]), float(freqs[i_peak]), True
    )


def _n_angles(dim: int, p: float) -> int:
    n_ang = int(p * dim // 2)
    if n_ang < 1:
        raise ValueError(f"no active rotary angle for dim={dim}, p={p}")
    return n_ang


def rotary_wavelengths(
    dim: int, p: float = 0.75, base: float = 10000.0, time_scale: float = 1.0
) -> np.ndarray:
    """Wavelengths in days of the active angles of an axial time block,
    ``2 pi time_scale base ** (2 i / dim)`` for positions ``t / time_scale``."""
    i = np.arange(_n_angles(dim, p))
    return 2 * np.pi * time_scale * base ** (2 * i / dim)


SPACINGS = ("log", "linear", "quantile")


def rotary_ladder(
    n: int,
    lam_min: float,
    lam_max: float,
    spacing: str = "log",
    samples: np.ndarray | None = None,
    mix: float = 1.0,
) -> np.ndarray:
    """``n`` increasing wavelengths (days) from ``lam_min`` to ``lam_max``.

    ``spacing="log"`` is the geometric ladder of an :class:`AxialRope` (what
    ``time_scale`` and ``base`` encode), ``"linear"`` an arithmetic one, and
    ``"quantile"`` follows the data: the ladder is the inverse CDF of
    ``log(samples)`` (periods, timescales, ... in days, clipped to the band)
    at ``n`` evenly spaced probabilities, so channels crowd where the
    timescales of the dataset are dense. ``mix`` in [0, 1] blends it in log
    space with the log ladder (``mix=1`` pure quantile, ``mix=0`` pure log):
    a little log-uniform floor keeps sparse parts of the band covered and
    keeps the ladder strictly increasing where the samples pile up. Both ends
    are always ``lam_min`` and ``lam_max``.
    """
    if n < 1:
        raise ValueError(f"need n >= 1, got {n}")
    if not 0 < lam_min <= lam_max:
        raise ValueError(f"need 0 < lam_min <= lam_max, got {lam_min}, {lam_max}")
    if spacing not in SPACINGS:
        raise ValueError(f"spacing must be one of {SPACINGS}, got {spacing!r}")
    if n == 1:
        return np.array([lam_min], dtype=np.float64)
    if spacing == "linear":
        return np.linspace(lam_min, lam_max, n)
    log_ladder = np.linspace(np.log(lam_min), np.log(lam_max), n)
    if spacing == "log":
        return np.exp(log_ladder)
    if samples is None:
        raise ValueError("spacing='quantile' needs samples")
    if not 0 <= mix <= 1:
        raise ValueError(f"mix must be in [0, 1], got {mix}")
    x = np.log(np.asarray(samples, dtype=np.float64).ravel())
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("no finite positive sample")
    x = np.clip(x, np.log(lam_min), np.log(lam_max))
    quant = np.quantile(x, np.linspace(0.0, 1.0, n))
    quant[0], quant[-1] = np.log(lam_min), np.log(lam_max)
    out = np.exp((1 - mix) * log_ladder + mix * quant)
    out[0], out[-1] = lam_min, lam_max
    return np.maximum.accumulate(out)


def timescales_for(wavelengths: np.ndarray, time_scale: float) -> list[float]:
    """Wavelengths in days -> ``AxialRope`` timescales in position units
    (``lambda / (2 pi time_scale)``), the ``rope_timescales`` of a model
    whose positions are ``t / time_scale``."""
    w = np.asarray(wavelengths, dtype=np.float64).ravel()
    if time_scale <= 0:
        raise ValueError(f"time_scale must be positive, got {time_scale}")
    return [float(v) for v in w / (2 * np.pi * time_scale)]


def time_encoding(
    lam_min: float, lam_max: float, dim: int, p: float = 0.75
) -> tuple[float, float]:
    """``(time_scale, base)`` whose ladder spans ``[lam_min, lam_max]`` days;
    the exact inverse of :func:`rotary_wavelengths`."""
    if not 0 < lam_min <= lam_max:
        raise ValueError(f"need 0 < lam_min <= lam_max, got {lam_min}, {lam_max}")
    n_ang = _n_angles(dim, p)
    base = 1.0 if n_ang == 1 else (lam_max / lam_min) ** (dim / (2 * (n_ang - 1)))
    return lam_min / (2 * np.pi), float(base)


def nd_rope_theta_bound(dim: int | SimplexRope | BlockRope, n_axes: int = 0) -> float:
    """Largest simplex base keeping every nD-RoPE wave vector resolvable.

    nD-RoPE (Li et al. 2026, App. E) bounds the base of a ladder of ``S =
    dim / (2 M)`` scales over ``n_axes`` axes by ``exp(dim / (2 M n_axes))``,
    the base at which adjacent scales differ by ``exp(1 / n_axes)``. Here
    ``dim`` is the channel count of the *simplex block* (its layout dict's
    ``dim``, ``rope.dims[i]``), not of the head, since the block takes only
    part of the head in every ``layout(..., "simplex")``; ``n_axes`` is the
    number of axes the block spans (``len(block["axes"])``, which excludes
    the time axis), and ``M = n_axes + 1`` as in :class:`SimplexRope`, which
    reduces to ``M = 1`` for a single axis. Pass a :class:`SimplexRope`, or a
    :class:`BlockRope` holding one, to read both off the block.
    """
    if isinstance(dim, BlockRope):
        simplex = [b for b in dim.blocks if isinstance(b, SimplexRope)]
        if len(simplex) != 1:
            raise ValueError(f"BlockRope has {len(simplex)} simplex blocks, need 1")
        dim = simplex[0]
    if isinstance(dim, SimplexRope):
        dim, n_axes = dim.dim, dim.n_axes
    if n_axes < 1:
        raise ValueError(f"need n_axes >= 1, got {n_axes}")
    m = n_axes + 1 if n_axes > 1 else 1
    return float(np.exp(dim / (2 * m * n_axes)))


@dataclass
class TimeEncodingReport:
    """Result of :func:`suggest_time_encoding`, all times in days.

    ``n_detected`` of the ``n_curves`` analysed curves rose above their
    noise floor (``n_skipped`` further curves had fewer than three distinct
    epochs and no periodogram); ``peak_periods``, ``short_periods`` (highest
    harmonic) and ``long_periods`` (lowest) are their per-curve values,
    ``spans`` the baselines of all curves. ``period_lo`` / ``period_hi`` are the pooled
    spectral support, ``lam_min`` / ``lam_max`` the band the recommended
    ``time_scale`` and ``base`` realise for a time block of ``dim`` channels
    and fraction ``p``, with its ladder ``wavelengths [n_ang]`` (days) and
    the same ladder as ``timescales`` in position units, ready for
    ``RoMAE(..., rope_timescales=report.timescales)``. With ``spacing`` other
    than ``"log"`` the ladder is not geometric: ``base`` is then only the
    log-spaced equivalent and the model must take ``timescales``.
    """

    n_curves: int
    n_detected: int
    dt_min: float
    dt_median: float
    span: float
    period_lo: float
    period_hi: float
    lam_min: float
    lam_max: float
    time_scale: float
    base: float
    dim: int
    p: float
    wavelengths: np.ndarray
    peak_periods: np.ndarray
    short_periods: np.ndarray
    long_periods: np.ndarray
    spans: np.ndarray
    n_skipped: int = 0
    spacing: str = "log"
    timescales: list[float] | None = None

    def __str__(self) -> str:
        w = self.wavelengths
        inside = int(((w >= self.period_lo) & (w <= self.period_hi)).sum())
        skipped = f", {self.n_skipped} skipped" if self.n_skipped else ""
        if self.spacing == "log":
            rec = f"time_scale = {self.time_scale:.4g}, base = {self.base:.4g}"
        else:
            rec = (
                f"time_scale = {self.time_scale:.4g}, {self.spacing} ladder "
                f"(rope_timescales = report.timescales; base {self.base:.4g} "
                f"is the log-spaced equivalent)"
            )
        return (
            f"time encoding from {self.n_curves} curves, {self.n_detected} with a "
            f"detected signal{skipped} (time block dim={self.dim}, p={self.p}, "
            f"{w.size} rotary channels)\n"
            f"  sampling: dt_min {self.dt_min:.4g} d, dt_median {self.dt_median:.4g} d,"
            f" span {self.span:.4g} d\n"
            f"  spectral support: periods {self.period_lo:.4g} - {self.period_hi:.4g} d\n"
            f"  recommended: {rec}\n"
            f"  rotary band: {self.lam_min:.4g} - {self.lam_max:.4g} d, "
            f"{inside} / {w.size} channels inside the support"
        )


def _standardize(y: np.ndarray, err: np.ndarray | None, band: np.ndarray):
    """Per-band ``(y - median) / (1.4826 MAD)``, errors scaled alike; a zero
    MAD falls back to the std and then to a scale of 1 (like
    :func:`~romae_lc.data.normalize`)."""
    y = y.copy()
    err = None if err is None else err.copy()
    eps = 1e-12
    for b in np.unique(band):
        m = band == b
        med = np.median(y[m])
        scale = 1.4826 * np.median(np.abs(y[m] - med))
        if scale <= eps:
            scale = float(y[m].std())
        if scale <= eps:
            scale = 1.0
        y[m] = (y[m] - med) / scale
        if err is not None:
            err[m] /= scale
    return y, err


def suggest_time_encoding(
    times: list[np.ndarray],
    values: list[np.ndarray],
    dim: int,
    p: float = 0.75,
    bands: list[np.ndarray] | None = None,
    errors: list[np.ndarray] | None = None,
    q: tuple[float, float] = (0.02, 0.98),
    rel: float = 0.1,
    max_curves: int = 128,
    max_evals: float = 1e7,
    seed: int = 0,
    lam_min: float | None = None,
    lam_max: float | None = None,
    spacing: str = "log",
    samples: np.ndarray | None = None,
    mix: float = 1.0,
    max_freq: float | None = None,
) -> TimeEncodingReport:
    """Recommend ``time_scale`` and ``base`` for a dataset of light curves.

    Every curve gets its own :func:`periodogram` and :func:`spectral_support`;
    the pooled support runs from the ``q[0]`` quantile of the detected curves'
    shortest timescales (``period_lo``) to the ``q[1]`` quantile of their
    longest (``period_hi``), both taken as order statistics of actual curves,
    so one broad-band class cannot hide the short periods of another. The ladder then reaches ``period_lo`` (``lam_min``)
    and twice the ``q[1]`` quantile of the spans (``lam_max``: the slowest
    channel makes half a turn over the longest curve), see
    :func:`time_encoding`. Without any detected curve the support falls back
    to the sampling limits (twice the 5th-percentile gap, the longest span).
    Curves with fewer than three distinct epochs have no periodogram and are
    skipped (counted in ``n_skipped``); a dataset without any other curve
    raises ``ValueError``.

    Args:
        times: One 1-D epoch array per curve, in days, all bands concatenated.
        values: Matching value arrays.
        dim: Channels of the time block (``RoMAE(...).rope.dims[0]``).
        p: p-RoPE fraction of the time block.
        bands: Optional band id arrays; values are then standardised per band
            (median / MAD) so band offsets do not enter the periodogram.
        errors: Optional uncertainty arrays (inverse-variance weights).
        q: Quantiles over curves bounding the pooled support; ``q[1]`` is
            also the span quantile behind ``lam_max``.
        rel: Fraction of a curve's peak power a harmonic must reach to count.
        max_curves: Random subsample size (seeded) that bounds the cost.
        max_evals: Periodogram evaluations per curve (see :func:`periodogram`).
        seed: Subsampling seed.
        lam_min: Override the shortest wavelength (days).
        lam_max: Override the longest wavelength (days).
        spacing: Ladder spacing, see :func:`rotary_ladder`: ``"log"`` (the
            geometric ladder ``time_scale`` and ``base`` encode), ``"linear"``
            or ``"quantile"``.
        samples: Timescales in days whose distribution a ``"quantile"``
            ladder follows; by default the detected curves' shortest, peak
            and longest timescales pooled (pass catalogue periods and their
            harmonics to follow a catalogue instead).
        mix: Quantile-versus-log blend of a ``"quantile"`` ladder.
        max_freq: Cap on the periodogram grid in cycles per day. The default
            grid runs to the pseudo-Nyquist frequency of each curve's 5th
            percentile gap, which for surveys with intra-night cadence and
            multi-year baselines (ZTF: ~500 cycles per day over 2700 d) means
            millions of frequencies per curve and an hour-long report; a cap
            of 50 to 100 keeps it to seconds per curve and still resolves
            periods down to 0.01 to 0.02 d.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(times))
    if len(times) > max_curves:
        idx = np.sort(rng.choice(idx, max_curves, False))
    ts = [np.asarray(times[i], dtype=np.float64) for i in idx]
    ok = [np.unique(t).size >= 3 for t in ts]
    n_skipped = len(ts) - sum(ok)
    if n_skipped == len(ts):
        raise ValueError("no curve with at least 3 distinct epochs")
    idx = idx[np.array(ok)]
    ts = [t for t, o in zip(ts, ok) if o]
    gaps = np.concatenate([_gaps(t) for t in ts])
    spans = np.array([np.ptp(t) for t in ts])

    supports = []
    for i, t in zip(idx, ts):
        y = np.asarray(values[i], dtype=np.float64)
        err = None if errors is None else np.asarray(errors[i], dtype=np.float64)
        if bands is not None:
            y, err = _standardize(y, err, np.asarray(bands[i]))
        f_hi = None
        if max_freq is not None:
            f_hi = min(max_freq, 0.5 / np.quantile(_gaps(t), 0.05))
            if not 0.5 / np.ptp(t) < f_hi:  # curve shorter than one cycle at the cap
                f_hi = None
        supports.append(
            spectral_support(
                *periodogram(t, y, err, max_freq=f_hi, max_evals=max_evals, rng=rng),
                rel,
            )
        )
    detected = [s for s in supports if s.detected]
    peak = np.array([1 / s.f_peak for s in detected])
    short = np.array([1 / s.f_hi for s in detected])
    long = np.array([1 / s.f_lo for s in detected])
    if detected:
        period_lo = float(np.quantile(short, q[0], method="lower"))
        period_hi = float(np.quantile(long, q[1], method="higher"))
    else:
        period_lo, period_hi = 2 * float(np.quantile(gaps, 0.05)), float(spans.max())

    lam_min = period_lo if lam_min is None else lam_min
    if lam_max is None:
        lam_max = 2.0 * float(np.quantile(spans, q[1], method="higher"))
    time_scale, base = time_encoding(lam_min, lam_max, dim, p)
    if spacing == "log":
        wavelengths = rotary_wavelengths(dim, p, base, time_scale)
    else:
        if samples is None:
            samples = np.concatenate([short, peak, long])
        wavelengths = rotary_ladder(
            _n_angles(dim, p), lam_min, lam_max, spacing, samples, mix
        )
    return TimeEncodingReport(
        n_curves=len(ts),
        n_detected=len(detected),
        dt_min=float(gaps.min()),
        dt_median=float(np.median(gaps)),
        span=float(spans.max()),
        period_lo=period_lo,
        period_hi=period_hi,
        lam_min=lam_min,
        lam_max=lam_max,
        time_scale=time_scale,
        base=base,
        dim=dim,
        p=p,
        wavelengths=wavelengths,
        peak_periods=peak,
        short_periods=short,
        long_periods=long,
        spans=spans,
        n_skipped=n_skipped,
        spacing=spacing,
        timescales=timescales_for(wavelengths, time_scale),
    )


def rotary_kernel(
    wavelengths: np.ndarray | torch.Tensor, lags: np.ndarray
) -> np.ndarray:
    """Relative-position response of a ladder, ``mean_i cos(2 pi lag /
    lambda_i)``: the logit between identical query and key vectors as a
    function of their lag (the nD-RoPE paper's frequency-domain view, here in
    the lag domain). Pass ``AxialRope.wavelengths * time_scale`` or
    ``TimeEncodingReport.wavelengths`` and lags in days."""
    if isinstance(wavelengths, torch.Tensor):
        wavelengths = wavelengths.detach().cpu().numpy()
    lags = np.asarray(lags, dtype=np.float64)
    return np.cos(2 * np.pi * lags[:, None] / wavelengths[None, :]).mean(1)


def wave_vectors(rope: BlockRope) -> list[np.ndarray]:
    """Per block, the active wave vectors in position units: magnitudes
    ``[n_ang]`` for an axial block, vectors ``[nhead, M * S, n_axes]`` for a
    simplex block (for scatter plots like the nD-RoPE paper's Figure 2)."""
    out = []
    for block in rope.blocks:
        if isinstance(block, AxialRope):
            ts = block.timescale
            out.append((1.0 / ts[torch.isfinite(ts)]).cpu().numpy())
        else:
            out.append(block.wave_vectors.cpu().numpy())
    return out


@torch.no_grad()
def time_shuffle_score(
    backbone,
    tokens: Tokens,
    generator: torch.Generator | None = None,
    stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Does the encoder use *when* things were observed, or only *what*?

    Channel 0 of the values is permuted among the real tokens sharing a
    wavelength coordinate (axis 1 of the positions; all real tokens with a
    time-only axis), which destroys every temporal structure while keeping
    the magnitude distribution, the bands and the cadence. The score is the
    cosine similarity between the embeddings of the original and the shuffled
    batch, both standardised by the mean and std of the original batch. A
    score of ~1 means the encoder is a bag of magnitudes, blind to time and
    phase; lower is better. This is the empirical check that the chosen time
    band lets a *trained* model see phase: at initialisation attention is
    nearly uniform, so untrained encoders score ~1 whatever the band.

    The standardisation makes this a batch statistic: it needs several curves
    (a single curve standardises to zero and raises), and scores of different
    batches are on a common scale only when they share ``stats``, the
    ``(mean [D], std [D])`` of the embeddings of a reference set (the whole
    validation set, say), computed once and passed to every call.

    Args:
        backbone: ``(values, positions, pad_mask) -> [B, D]``, used in its
            current mode.
        tokens: The token batch.
        generator: Optional CPU generator for reproducible permutations.
        stats: Optional ``(mu, sd)`` replacing the batch's own mean and std.

    Returns:
        Scores ``[B]``.
    """
    values, positions, pad_mask = tokens
    if stats is None and values.shape[0] < 2:
        raise ValueError("time_shuffle_score needs a batch of at least 2 curves")
    shuffled = values.clone()
    for b in range(values.shape[0]):
        real = (~pad_mask[b]).nonzero().squeeze(1)
        coord = positions[b, 1, real] if positions.shape[1] > 1 else real.new_zeros(1)
        for u in torch.unique(coord):
            rows = real if coord.numel() == 1 else real[coord == u]
            perm = torch.randperm(rows.numel(), generator=generator).to(rows.device)
            shuffled[b, rows, 0] = values[b, rows[perm], 0]
    z0 = backbone(values, positions, pad_mask).float()
    z1 = backbone(shuffled, positions, pad_mask).float()
    if stats is None:
        mu, sd = z0.mean(0), z0.std(0, unbiased=False) + 1e-6
    else:
        mu, sd = (s.to(z0).float() for s in stats)
    return F.cosine_similarity((z0 - mu) / sd, (z1 - mu) / sd)


def plot_time_encoding(report: TimeEncodingReport, ax=None):
    """Histograms of the detected curves' peak periods and shortest
    timescales against the recommended rotary ladder and the shaded spectral
    support (matplotlib, imported lazily). Returns the axis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("plot_time_encoding needs matplotlib") from e
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    lo = min(report.lam_min, report.short_periods.min(initial=report.lam_min))
    hi = max(report.lam_max, report.long_periods.max(initial=report.lam_max))
    bins = np.geomspace(lo / 1.5, hi * 1.5, 60)
    ax.axvspan(report.period_lo, report.period_hi, color="0.85", label="support")
    ax.hist(report.peak_periods, bins, color="0.25", label="peak period")
    ax.hist(
        report.short_periods, bins, color="C1", alpha=0.6, label="shortest timescale"
    )
    for i, w in enumerate(report.wavelengths):
        ax.axvline(w, color="C0", lw=0.8, alpha=0.7, label="rotary" if i == 0 else None)
    ax.set_xscale("log")
    ax.set_xlabel("period [d]")
    ax.set_ylabel("curves")
    if report.spacing == "log":
        ax.set_title(f"time_scale = {report.time_scale:.4g}, base = {report.base:.4g}")
    else:
        ax.set_title(f"time_scale = {report.time_scale:.4g}, {report.spacing} ladder")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    return ax
