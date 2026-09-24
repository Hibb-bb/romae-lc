"""Time-encoding analysis: periodograms, rotary band inversion, shuffle score."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from romae_lc.analysis import (
    lomb_scargle,
    nd_rope_theta_bound,
    periodogram,
    plot_time_encoding,
    rotary_kernel,
    rotary_ladder,
    rotary_wavelengths,
    sampling_stats,
    spectral_support,
    suggest_time_encoding,
    time_encoding,
    time_shuffle_score,
    timescales_for,
    wave_vectors,
)
from romae_lc.model import RoMAE
from romae_lc.rope import AxialRope, SimplexRope
from romae_lc.tokenize import Tokens, tokenize

ENCODER = dict(d_model=32, nhead=2, depth=2)
WL = {0: 480.0, 1: 620.0}
PERIODS = np.geomspace(0.3, 3.0, 8)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def sinusoids(periods, noise, baseline=200.0, n_points=400, seed=0):
    """Irregularly sampled sinusoids with the given periods."""
    rng = np.random.default_rng(seed)
    times, values = [], []
    for period in periods:
        t = np.sort(rng.uniform(0, baseline, n_points))
        times.append(t)
        values.append(np.sin(2 * np.pi * t / period) + noise * rng.normal(size=t.size))
    return times, values


@pytest.fixture(scope="module")
def report():
    """Recommendation for periods in [0.3, 3] d over a 200 d baseline."""
    return suggest_time_encoding(*sinusoids(PERIODS, 0.1), dim=16)


def test_sampling_stats_use_positive_gaps():
    stats = sampling_stats(np.array([3.0, 0.0, 1.0, 1.0]))
    assert stats == dict(n=4, span=3.0, dt_min=1.0, dt_median=1.5)


def test_lomb_scargle_peaks_at_the_true_frequency():
    rng = np.random.default_rng(1)
    t, f0 = np.sort(rng.uniform(0, 50, 400)), 1 / 1.7
    y = 3.0 + 2.0 * np.sin(2 * np.pi * f0 * t + 0.4)
    freqs = np.geomspace(0.01, 10, 4000)
    power = lomb_scargle(t, y, freqs)
    assert power.shape == freqs.shape and 0 <= power.min() <= power.max() <= 1
    assert abs(freqs[power.argmax()] / f0 - 1) < 2e-3 and power.max() > 0.99
    assert np.allclose(lomb_scargle(t, 10 * y - 7, freqs), power)
    assert np.allclose(lomb_scargle(t, y, freqs, dy=np.full_like(t, 0.5)), power)


def test_periodogram_grid_and_budget():
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 50, 300))
    y = np.sin(2 * np.pi * t / 1.7)
    freqs, power = periodogram(t, y)
    gaps = np.diff(t)
    assert freqs.shape == power.shape and np.isfinite(power).all()
    assert np.isclose(freqs[0], 0.5 / np.ptp(t))
    assert np.isclose(freqs[-1], 0.5 / np.quantile(gaps[gaps > 0], 0.05))
    assert np.allclose(np.diff(freqs), freqs[1] - freqs[0])
    assert freqs[1] - freqs[0] <= 1 / (4 * np.ptp(t))
    assert abs(1 / freqs[power.argmax()] - 1.7) < 0.01
    # A tight evaluation budget subsamples the observations, not the grid.
    freqs_b, power_b = periodogram(t, y, max_evals=freqs.size * 50)
    assert (
        freqs_b.shape == freqs.shape and abs(1 / freqs_b[power_b.argmax()] - 1.7) < 0.01
    )
    with pytest.raises(ValueError):
        periodogram(t, y, min_freq=5.0, max_freq=1.0)


def test_spectral_support_brackets_the_signal_and_its_harmonics():
    freqs = np.linspace(0.01, 10, 5000)
    flat = spectral_support(freqs, np.ones_like(freqs))
    assert not flat.detected and (flat.f_lo, flat.f_hi) == (0.01, 10.0)
    delta = np.zeros_like(freqs)
    delta[700] = 1.0
    assert spectral_support(freqs, delta) == (freqs[700], freqs[700], freqs[700], True)
    rng = np.random.default_rng(3)
    t, f0 = np.sort(rng.uniform(0, 50, 400)), 1 / 1.7
    y = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(4 * np.pi * f0 * t)
    s = spectral_support(freqs, lomb_scargle(t, y, freqs))
    assert s.detected and abs(s.f_peak / f0 - 1) < 0.01
    assert s.f_lo <= f0 and abs(s.f_hi / (2 * f0) - 1) < 0.01  # harmonic counted
    noise = spectral_support(freqs, lomb_scargle(t, rng.normal(size=t.size), freqs))
    assert not noise.detected


@pytest.mark.parametrize("dim, p", [(16, 0.75), (32, 1.0), (12, 0.5)])
def test_time_encoding_inverts_rotary_wavelengths(dim, p):
    lam_min, lam_max = 0.3, 400.0
    time_scale, base = time_encoding(lam_min, lam_max, dim, p)
    w = rotary_wavelengths(dim, p, base, time_scale)
    assert w.shape == (int(p * dim // 2),)
    assert np.isclose(w[0], lam_min) and np.isclose(w[-1], lam_max)
    assert np.allclose(np.diff(np.log(w)), np.log(w[1] / w[0]))
    rope_w = AxialRope(dim, base, p).wavelengths.double().numpy() * time_scale
    assert np.allclose(rope_w, w, rtol=1e-5)


def test_time_encoding_edge_cases():
    time_scale, base = time_encoding(2.0, 50.0, dim=4, p=0.5)  # one active angle
    assert base == 1.0 and np.isclose(time_scale, 2.0 / (2 * math.pi))
    assert np.allclose(rotary_wavelengths(4, 0.5, base, time_scale), [2.0])
    with pytest.raises(ValueError):
        time_encoding(5.0, 1.0, dim=16)
    with pytest.raises(ValueError):
        rotary_wavelengths(2, p=0.5)
    assert np.isclose(nd_rope_theta_bound(64, 2), math.exp(64 / 12))


def test_nd_rope_theta_bound_matches_the_simplex_ladder():
    # At the bound, adjacent scales of the block's ladder differ by exp(1 / n).
    for dim, n in [(16, 3), (12, 2), (16, 1)]:
        theta = nd_rope_theta_bound(dim, n)
        rope = SimplexRope(dim, n, nhead=1, theta=theta, rotate=False)
        assert np.isclose(rope.mag[0] / rope.mag[1], math.exp(1 / n))
    assert np.isclose(nd_rope_theta_bound(16, 1), math.exp(16 / 2))  # M = 1
    model = RoMAE(encoder=dict(d_model=96, nhead=3, depth=2), n_axes=4, rope="simplex")
    bound = nd_rope_theta_bound(model.rope)  # 16-channel block over 3 axes
    assert (
        bound == nd_rope_theta_bound(model.rope.blocks[1]) == nd_rope_theta_bound(16, 3)
    )
    assert bound < nd_rope_theta_bound(32, 3)  # the head, not the block
    with pytest.raises(ValueError):
        nd_rope_theta_bound(RoMAE(encoder=ENCODER, rope="axial").rope)
    with pytest.raises(ValueError):
        nd_rope_theta_bound(16, 0)


def test_spectral_support_detects_sparse_and_subsampled_curves():
    # Few observations: the white-noise median is ~1 / n_obs, so a floor that
    # is a multiple of it would exceed the maximum power 1.
    for seed, n_obs in [(2, 30), (3, 30), (4, 20), (5, 25)]:
        rng = np.random.default_rng(seed)
        t = np.sort(rng.uniform(0, 100, n_obs))
        y = np.sin(2 * np.pi * t / 1.3) + 0.05 * rng.normal(size=n_obs)
        s = spectral_support(*periodogram(t, y))
        assert s.detected and abs(1 / s.f_peak - 1.3) < 0.01
        noise = spectral_support(*periodogram(t, rng.normal(size=n_obs)))
        assert not noise.detected
    freqs = np.linspace(0.01, 10, 5000)
    half = spectral_support(freqs, np.full_like(freqs, 0.5))
    assert not half.detected and 0.5 < 1 - 0.5 ** (2 * np.log2(5000)) < 1
    # Two dense sectors far apart: the grid is huge, the observations are
    # subsampled (to no fewer than 256), and the signal is still detected.
    rng = np.random.default_rng(6)
    sector = np.arange(0, 5, 10 / 1440)
    t = np.concatenate([sector, 200 + sector])
    y = np.sin(2 * np.pi * t / 0.5) + 0.3 * rng.normal(size=t.size)
    freqs, power = periodogram(t, y)
    assert 1e7 // freqs.size < 256 < t.size  # the budget did bite
    s = spectral_support(freqs, power)
    assert s.detected and abs(1 / s.f_peak - 0.5) < 5e-3  # within the alias comb


def test_degenerate_curves_are_skipped_or_rejected():
    assert sampling_stats(np.array([1.0]))["n"] == 1
    assert np.isnan(sampling_stats(np.array([1.0, 1.0]))["dt_min"])
    for t in [np.array([1.0]), np.array([1.0, 1.0]), np.array([0.0, 1.0])]:
        with pytest.raises(ValueError):
            periodogram(t, np.ones_like(t))
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0, 100, 10))
    freqs, power = periodogram(t, np.sin(t), max_evals=1e3)  # never over-subsampled
    assert freqs.shape == power.shape and np.isfinite(power).all()
    times, values = sinusoids(PERIODS[:3], 0.1, baseline=30.0, n_points=80)
    times += [np.array([5.0]), np.array([0.0, 1.0])]
    values += [np.ones(1), np.ones(2)]
    report = suggest_time_encoding(times, values, dim=16)
    assert (report.n_curves, report.n_detected, report.n_skipped) == (3, 3, 2)
    assert report.spans.shape == (3,) and "2 skipped" in str(report)
    with pytest.raises(ValueError):
        suggest_time_encoding(times[3:], values[3:], dim=16)


def test_suggest_time_encoding_covers_the_signal_band(report):
    assert report.n_curves == report.n_detected == 8
    assert report.dim == 16 and report.p == 0.75
    assert np.allclose(np.sort(report.peak_periods), PERIODS, rtol=2e-2)
    assert np.allclose(report.short_periods, report.peak_periods, rtol=2e-2)
    assert 0.28 <= report.period_lo <= 0.31 and report.period_hi >= 2.9
    assert report.lam_min == report.period_lo and report.lam_max >= 200
    assert np.isclose(report.span, 200, rtol=1e-2)
    assert np.isclose(report.wavelengths[0], report.lam_min)
    assert np.isclose(report.wavelengths[-1], report.lam_max)
    assert np.allclose(
        rotary_wavelengths(16, 0.75, report.base, report.time_scale),
        report.wavelengths,
    )
    text = str(report)
    assert "8 with a detected signal" in text and "/ 6 channels" in text


def test_suggest_time_encoding_overrides_bands_and_subsampling():
    times, values = sinusoids(PERIODS[:6], 0.1)
    bands = [np.arange(len(t)) % 2 for t in times]
    shifted = [y + 5.0 * b for y, b in zip(values, bands)]
    kw = dict(dim=16, max_curves=4, lam_min=0.5, lam_max=100.0)
    a = suggest_time_encoding(times, values, bands=bands, **kw)
    b = suggest_time_encoding(times, shifted, bands=bands, **kw)
    assert a.n_curves == 4 and np.allclose(a.peak_periods, b.peak_periods)
    assert (a.lam_min, a.lam_max) == (0.5, 100.0)
    assert (a.time_scale, a.base) == time_encoding(0.5, 100.0, 16)
    errors = [np.full(len(t), 0.1) for t in times]
    c = suggest_time_encoding(times, values, errors=errors, **kw)
    d = suggest_time_encoding(times, values, **kw)
    assert np.allclose(c.peak_periods, d.peak_periods, rtol=1e-2)


def test_white_noise_falls_back_to_the_sampling_limits():
    rng = np.random.default_rng(4)
    times = [np.sort(rng.uniform(0, 200, 400)) for _ in range(6)]
    values = [rng.normal(size=400) for _ in times]
    report = suggest_time_encoding(times, values, dim=16)
    gaps = np.concatenate([np.diff(t) for t in times])
    spans = np.array([np.ptp(t) for t in times])
    assert report.n_detected == 0 and report.peak_periods.size == 0
    assert np.isclose(report.period_lo, 2 * np.quantile(gaps, 0.05))
    assert np.isclose(report.period_hi, spans.max())
    assert np.isclose(report.lam_max, 2 * np.quantile(spans, 0.98, method="higher"))
    assert report.lam_min == report.period_lo


def test_rotary_kernel_and_wave_vectors():
    kernel = rotary_kernel(np.array([2.0]), np.array([0.0, 1.0, 2.0]))
    assert np.allclose(kernel, [1.0, -1.0, 1.0])
    ladder = torch.tensor([1.0, 2.0, 4.0])
    assert rotary_kernel(ladder, np.linspace(0, 8, 50)).shape == (50,)
    assert np.isclose(rotary_kernel(ladder, np.array([0.0]))[0], 1.0)
    axial = wave_vectors(RoMAE(encoder=ENCODER, n_axes=2, rope="axial").rope)
    assert [v.shape for v in axial] == [(3,), (3,)]
    assert np.allclose(axial[0], 1 / AxialRope(8, p=0.75).timescale[:3].numpy())
    simplex = wave_vectors(RoMAE(encoder=ENCODER, n_axes=3, rope="simplex").rope)
    assert [v.shape for v in simplex] == [(3,), (2, 3, 2)]  # time 10 + simplex 6


def periodic_tokens(band_positions=None):
    """Six curves of period 1.3 d with different lengths (padding)."""
    rng = np.random.default_rng(5)
    times = [torch.tensor(np.sort(rng.uniform(0, 20, 64 - 8 * i))) for i in range(6)]
    values = [torch.sin(2 * math.pi * t / 1.3).float() for t in times]
    bands = [torch.randint(0, 2, t.shape) for t in times]
    if band_positions is not None:
        return tokenize(times, values, bands, band_positions=band_positions)
    return tokenize(times, values, bands, band_wavelengths=WL, time_scale=0.1)


def widen(model, std=0.3):
    """Widen the encoder weights so attention is position-selective: at the
    default init it is nearly uniform, hence nearly time-blind anyway."""
    for m in model.transformer.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
    return model.eval()


def test_time_shuffle_score_separates_blind_and_seeing_encoders():
    tokens = periodic_tokens()
    original = tokens.values.clone()
    blind = widen(RoMAE(encoder=ENCODER, p_rope=0.0))
    score = time_shuffle_score(blind, tokens, torch.Generator().manual_seed(0))
    assert score.shape == (6,) and torch.allclose(score, torch.ones(6), atol=1e-5)
    seeing = widen(RoMAE(encoder=ENCODER, p_rope=0.75))
    score = time_shuffle_score(seeing, tokens, torch.Generator().manual_seed(0))
    assert (score < 0.99).all() and score.mean() < 0.9
    again = time_shuffle_score(seeing, tokens, torch.Generator().manual_seed(0))
    assert torch.equal(score, again) and torch.equal(tokens.values, original)
    time_only = periodic_tokens(band_positions={0: (), 1: ()})
    blind = widen(RoMAE(encoder=ENCODER, n_axes=1, p_rope=0.0))
    score = time_shuffle_score(blind, time_only, torch.Generator().manual_seed(0))
    assert torch.allclose(score, torch.ones(6), atol=1e-5)


def test_time_shuffle_score_is_a_batch_statistic():
    tokens = periodic_tokens()
    seeing = widen(RoMAE(encoder=ENCODER, p_rope=0.75))
    one = Tokens(*(a[:1] for a in tokens))
    with pytest.raises(ValueError):
        time_shuffle_score(seeing, one)
    # Reference statistics put a single curve on the batch's scale.
    z = seeing(*tokens)
    stats = (z.mean(0), z.std(0, unbiased=False) + 1e-6)
    batch = time_shuffle_score(seeing, tokens, torch.Generator().manual_seed(0), stats)
    single = time_shuffle_score(seeing, one, torch.Generator().manual_seed(0), stats)
    assert single.shape == (1,) and single.item() < 0.99
    assert torch.allclose(single, batch[:1], atol=1e-4)


def test_plot_time_encoding_draws_the_ladder(report):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    ax = plot_time_encoding(report)
    assert ax.get_xscale() == "log" and len(ax.lines) == report.wavelengths.size
    assert "time_scale" in ax.get_title()
    assert plot_time_encoding(report, ax=ax) is ax
    mpl.pyplot.close("all")


def test_rotary_ladder_spacings():
    log = rotary_ladder(5, 0.1, 1000.0)
    assert np.allclose(log, np.geomspace(0.1, 1000.0, 5))
    lin = rotary_ladder(5, 0.1, 1000.0, "linear")
    assert np.allclose(lin, np.linspace(0.1, 1000.0, 5))
    rng = np.random.default_rng(0)
    samples = rng.lognormal(np.log(0.4), 0.1, 10_000)  # a tight period peak
    quant = rotary_ladder(7, 0.05, 2000.0, "quantile", samples)
    assert quant[0] == 0.05 and quant[-1] == 2000.0
    assert np.all(np.diff(quant) >= 0)
    assert np.all((0.3 < quant[1:-1]) & (quant[1:-1] < 0.55))  # channels crowd there
    mixed = rotary_ladder(7, 0.05, 2000.0, "quantile", samples, mix=0.5)
    assert np.all(np.diff(mixed) > 0)
    assert np.allclose(
        np.log(mixed),
        0.5 * np.log(quant) + 0.5 * np.log(log_ := rotary_ladder(7, 0.05, 2000.0)),
    )
    assert np.allclose(
        rotary_ladder(7, 0.05, 2000.0, "quantile", samples, mix=0.0), log_
    )
    assert rotary_ladder(1, 0.5, 9.0, "quantile", samples) == [0.5]
    for kw in (
        dict(spacing="cubic"),
        dict(spacing="quantile"),
        dict(spacing="quantile", samples=samples, mix=2),
    ):
        with pytest.raises(ValueError):
            rotary_ladder(4, 0.1, 10.0, **kw)
    with pytest.raises(ValueError):
        rotary_ladder(0, 0.1, 10.0)
    with pytest.raises(ValueError):
        rotary_ladder(4, 10.0, 0.1)
    # timescales_for is the inverse of AxialRope.wavelengths * time_scale
    ts = timescales_for(quant, time_scale=0.02)
    rope = AxialRope(16, timescales=ts)
    assert np.allclose(rope.wavelengths.double().numpy() * 0.02, quant, rtol=1e-5)
    with pytest.raises(ValueError):
        timescales_for(quant, 0.0)


def test_suggest_time_encoding_quantile_ladder():
    times, values = sinusoids(PERIODS, 0.1)
    log = suggest_time_encoding(times, values, dim=16)
    quant = suggest_time_encoding(times, values, dim=16, spacing="quantile", mix=1.0)
    assert log.spacing == "log" and quant.spacing == "quantile"
    assert (quant.time_scale, quant.base) == (log.time_scale, log.base)
    assert quant.wavelengths.shape == log.wavelengths.shape == (6,)
    assert quant.wavelengths[0] == log.wavelengths[0] == quant.lam_min
    assert np.isclose(quant.wavelengths[-1], log.wavelengths[-1])
    # the periods sit in [0.3, 3] d, so the inner quantile channels do too,
    # while the log ladder spends channels on the empty decades in between
    assert np.all(quant.wavelengths[1:-1] < 4.0)
    assert np.any(log.wavelengths[1:-1] > 4.0)
    assert np.allclose(
        quant.timescales, quant.wavelengths / (2 * np.pi * quant.time_scale)
    )
    assert np.allclose(log.timescales, log.wavelengths / (2 * np.pi * log.time_scale))
    assert "quantile ladder" in str(quant) and "base" in str(log)
    model = RoMAE(encoder=ENCODER, n_axes=1, rope_timescales=quant.timescales)
    rope_w = model.rope.blocks[0].wavelengths.double().numpy() * quant.time_scale
    assert np.allclose(rope_w, quant.wavelengths, rtol=1e-5)
    catalogue = suggest_time_encoding(
        times, values, dim=16, spacing="quantile", samples=np.full(100, 1.0)
    )
    assert np.allclose(catalogue.wavelengths[1:-1], 1.0)
    linear = suggest_time_encoding(times, values, dim=16, spacing="linear")
    assert np.allclose(np.diff(linear.wavelengths), np.diff(linear.wavelengths)[0])
    with pytest.raises(ValueError):
        suggest_time_encoding(times, values, dim=16, spacing="cubic")


def test_suggest_time_encoding_max_freq_caps_the_grid():
    times, values = sinusoids(PERIODS[:3], 0.1)
    capped = suggest_time_encoding(times, values, dim=16, max_freq=5.0)
    assert capped.n_detected == 3
    assert np.all(capped.short_periods >= 1 / 5.0)
    assert np.allclose(np.sort(capped.peak_periods), PERIODS[:3], rtol=2e-2)
    # a cap below the lowest frequency of a curve falls back to the default
    loose = suggest_time_encoding(times, values, dim=16, max_freq=1e-4)
    default = suggest_time_encoding(times, values, dim=16)
    assert np.allclose(loose.peak_periods, default.peak_periods)
