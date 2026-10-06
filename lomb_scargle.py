"""Standalone, single-run Lomb-Scargle search for one light curve.

Input is either one band, ``{"mjd": ..., "mag": ..., "mag_unc": ...}``, or a
mapping of band names to those dictionaries. Times and periods use days;
``mag_unc`` contains positive 1-sigma magnitude errors. Non-finite points and
non-positive uncertainties are discarded within each band. Bands with fewer
than three remaining points are ignored.
"""

import numpy as np
from astropy.timeseries import LombScargle, LombScargleMultiband

__all__ = ["run_lomb_scargle"]


_MIN_POINTS = 3
_DEFAULT_MIN_PERIOD = 0.05
_DEFAULT_MAX_PERIOD = 400.0
_DEFAULT_SAMPLES_PER_PEAK = 10
_DEFAULT_MULTIBAND_METHOD = "fast"


def _gather(light_curve, selected):
    """Return concatenated time, magnitude, error, label arrays and used bands."""
    if {"mjd", "mag", "mag_unc"} <= light_curve.keys():
        bands = {"single": light_curve}
    else:
        bands = light_curve
    names = list(bands) if selected is None else list(selected)

    t, y, dy, labels, used = [], [], [], [], []
    for name in names:
        if name not in bands:
            continue
        data = bands[name]
        mjd = np.asarray(data["mjd"], dtype=float)
        mag = np.asarray(data["mag"], dtype=float)
        unc = np.asarray(data["mag_unc"], dtype=float)
        if mjd.ndim != 1 or mag.ndim != 1 or unc.ndim != 1:
            raise ValueError(f"Band {name!r} arrays must be one-dimensional")
        if not (mjd.size == mag.size == unc.size):
            raise ValueError(f"Band {name!r} arrays must have equal lengths")
        good = np.isfinite(mjd) & np.isfinite(mag) & np.isfinite(unc) & (unc > 0)
        if np.count_nonzero(good) < _MIN_POINTS:
            continue
        t.append(mjd[good])
        y.append(mag[good])
        dy.append(unc[good])
        labels.append(np.full(np.count_nonzero(good), name, dtype=object))
        used.append(name)
    if not used:
        return None
    return (np.concatenate(t), np.concatenate(y), np.concatenate(dy),
            np.concatenate(labels), used)


def _freq_bounds(t, min_period, max_period):
    """Use the requested bounds, with the long-period limit capped at 1/T."""
    baseline = float(np.ptp(t))
    f_max = 1.0 / max(min_period, 1e-9)
    f_min = 1.0 / max(max_period, 1e-9)
    if baseline > 0:
        f_min = max(f_min, 1.0 / baseline)
    return f_min, f_max


def _build(t, y, dy, labels, multiband):
    if multiband:
        return LombScargleMultiband(t, y, labels, dy), "multiband"
    return LombScargle(t, y, dy, fit_mean=True), "single"


def _evaluate(ls, multiband, t, *, autopower, min_period, max_period,
              samples_per_peak, n_grid, multiband_method):
    """Return period-ascending grid, powers, peak frequency and grid bounds."""
    f_min, f_max = _freq_bounds(t, min_period, max_period)
    if f_min >= f_max:
        return None
    method = multiband_method if multiband else "fast"
    if autopower:
        frequency, power = ls.autopower(
            minimum_frequency=f_min, maximum_frequency=f_max,
            samples_per_peak=samples_per_peak, method=method,
        )
    else:
        frequency = np.linspace(f_min, f_max, int(n_grid))
        power = ls.power(frequency, method=method)
    frequency = np.asarray(frequency, dtype=float)
    power = np.asarray(power, dtype=float)
    good = (frequency > 0) & np.isfinite(power)
    frequency, power = frequency[good], power[good]
    if frequency.size == 0:
        return None
    best_freq = float(frequency[np.argmax(power)])
    order = np.argsort(1.0 / frequency)
    return (1.0 / frequency)[order], power[order], best_freq, f_min, f_max


def _top_peaks(periods, power, n_peaks):
    """Strongest local maxima, backfilled from the grid if needed."""
    if not n_peaks or power.size == 0:
        return [], []
    if power.size >= 3:
        interior = np.flatnonzero(
            (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
        ) + 1
    else:
        interior = np.array([], dtype=int)
    idx = list(interior[np.argsort(power[interior])[::-1]])
    if len(idx) < n_peaks:
        for i in np.argsort(power)[::-1]:
            if i not in idx:
                idx.append(int(i))
            if len(idx) >= n_peaks:
                break
    idx = idx[:n_peaks]
    return [float(periods[i]) for i in idx], [float(power[i]) for i in idx]


def _baluev_fap(ls, multiband, peak_power):
    """Astropy has no Baluev FAP for its multiband Lomb-Scargle class."""
    if multiband:
        return None
    try:
        return float(ls.false_alarm_probability(peak_power, method="baluev"))
    except Exception:  # Astropy can fail on degenerate light curves.
        return None


def run_lomb_scargle(light_curve, selected=None, *, autopower=True,
                     min_period=_DEFAULT_MIN_PERIOD,
                     max_period=_DEFAULT_MAX_PERIOD,
                     samples_per_peak=_DEFAULT_SAMPLES_PER_PEAK,
                     n_grid=20000, n_peaks=6,
                     multiband_method=_DEFAULT_MULTIBAND_METHOD):
    """Run one weighted Lomb-Scargle search on a single or multiband light curve.

    ``selected`` optionally names the bands to use from a multiband mapping.
    One usable band uses ``LombScargle``; two or more use
    ``LombScargleMultiband``. Autopower and the manual grid are both uniform in
    frequency. The minimum frequency is at least one cycle over the observed
    baseline. The manual grid uses ``n_grid`` points; autopower uses
    ``samples_per_peak``. The search fits one sinusoid (one Fourier term).

    Returns a dict with period-ascending ``periods`` and ``power`` arrays,
    ``best_period`` and ``best_power`` at the maximum sampled power,
    ``peak_periods`` and ``peak_powers`` for the top local maxima, ``fap``
    (Baluev false-alarm probability for single-band only), ``kind``,
    ``bands_used``, ``f_min`` and ``f_max`` (cycles/day). Returns ``None``
    when fewer than three valid points remain in every band, the combined
    time baseline is zero, or no searched period fits within that baseline.
    """
    if (not np.isfinite(min_period) or not np.isfinite(max_period)
            or min_period <= 0 or max_period <= min_period):
        raise ValueError("period bounds must be finite and 0 < min_period < max_period")
    if int(samples_per_peak) < 1 or int(n_grid) < 2 or int(n_peaks) < 0:
        raise ValueError("samples_per_peak >= 1, n_grid >= 2 and n_peaks >= 0 are required")
    gathered = _gather(light_curve, selected)
    if gathered is None:
        return None
    t, y, dy, labels, bands_used = gathered
    if np.ptp(t) <= 0:
        return None
    multiband = len(bands_used) >= 2
    ls, kind = _build(t, y, dy, labels, multiband)
    evaluated = _evaluate(
        ls, multiband, t, autopower=autopower, min_period=min_period,
        max_period=max_period, samples_per_peak=samples_per_peak,
        n_grid=n_grid, multiband_method=multiband_method,
    )
    if evaluated is None:
        return None
    periods, power, best_freq, f_min, f_max = evaluated
    peak_periods, peak_powers = _top_peaks(periods, power, int(n_peaks))
    best_power = float(np.max(power))
    return {
        "periods": periods, "power": power, "kind": kind,
        "bands_used": bands_used, "best_period": 1.0 / best_freq,
        "best_power": best_power, "fap": _baluev_fap(ls, multiband, best_power),
        "peak_periods": peak_periods, "peak_powers": peak_powers,
        "f_min": f_min, "f_max": f_max,
    }
