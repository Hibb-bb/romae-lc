"""Tokenizer: sorting, padding, wavelength coordinates, extras and lookups."""

from __future__ import annotations

import math

import pytest
import torch

from romae_lc.tokenize import Tokens, tokenize, wavelengths_for

WL = {0: 480.0, 1: 620.0}
TIMES = [torch.tensor([3.0, 1.0, 2.0]), torch.tensor([5.0])]
VALUES = [torch.tensor([30.0, 10.0, 20.0]), torch.tensor([50.0])]
BANDS = [torch.tensor([0, 1, 0]), torch.tensor([1])]


def test_tokens_are_sorted_by_time_and_padded():
    tok = tokenize(TIMES, VALUES, BANDS, band_wavelengths=WL)
    assert tok.values.shape == (2, 3, 1) and tok.positions.shape == (2, 2, 3)
    assert tok.pad_mask.tolist() == [[False, False, False], [False, True, True]]
    assert tok.values[0, :, 0].tolist() == [10.0, 20.0, 30.0]
    assert tok.positions[0, 0].tolist() == [2.0, 3.0, 4.0]
    assert tok.positions[1, 0, 0] == 6.0
    assert (tok.values[1, 1:] == 0).all() and (tok.positions[1, :, 1:] == 0).all()
    assert tok.n_real.tolist() == [3, 1]
    values, positions, pad_mask = tok
    assert values is tok.values and pad_mask is tok.pad_mask
    assert isinstance(tok.to("cpu"), Tokens) and tok.extras is None


def test_wavelength_axis_and_scales():
    tok = tokenize(
        TIMES, VALUES, BANDS, band_wavelengths=WL, time_scale=0.5, wavelength_scale=0.25
    )
    lam = torch.tensor([620.0, 480.0, 480.0])  # bands in time order
    expected = torch.log(lam / math.sqrt(480.0 * 620.0)) / 0.25 + 1
    assert torch.allclose(tok.positions[0, 1], expected)
    assert tok.positions[0, 0].tolist() == [3.0, 5.0, 7.0]
    tok = tokenize(TIMES, VALUES, BANDS, band_wavelengths=WL, ref_wavelength=480.0)
    assert tok.positions[0, 1, 1:].tolist() == [1.0, 1.0]
    raw = tokenize(TIMES, VALUES, BANDS)
    assert raw.positions[0, 1].tolist() == [2.0, 1.0, 1.0]


def test_extras_follow_the_sort():
    extras = [torch.tensor([0.3, 0.1, 0.2]), torch.tensor([0.5])]
    tok = tokenize(TIMES, VALUES, BANDS, band_wavelengths=WL, extras=extras)
    assert tok.extras.shape == (2, 3)
    assert torch.allclose(tok.extras[0], torch.tensor([0.1, 0.2, 0.3]))
    assert tok.extras[1].tolist() == [0.5, 0.0, 0.0]
    assert tok.to("cpu").extras is not None


def test_band_positions_and_features():
    tok = tokenize(TIMES, VALUES, BANDS, band_positions={0: (0.1, 0.2), 1: (0.3, 0.4)})
    assert tok.positions.shape == (2, 3, 3)
    assert torch.allclose(tok.positions[0, 1:, 0], torch.tensor([1.3, 1.4]))
    time_only = tokenize(TIMES, VALUES, BANDS, band_positions={0: (), 1: ()})
    assert time_only.positions.shape == (2, 1, 3)
    feats = {0: (1.0, 0.0), 1: (0.0, 1.0)}
    tok = tokenize(TIMES, VALUES, BANDS, band_wavelengths=WL, band_features=feats)
    assert tok.values.shape == (2, 3, 3)
    expected = [[10.0, 0.0, 1.0], [20.0, 1.0, 0.0], [30.0, 1.0, 0.0]]
    assert tok.values[0].tolist() == expected
    assert (tok.values[1, 1:] == 0).all()


def test_lookup_and_argument_errors():
    with pytest.raises(KeyError):
        tokenize(TIMES, VALUES, BANDS, band_wavelengths={0: 480.0})
    with pytest.raises(KeyError):
        tokenize(TIMES, VALUES, BANDS, band_positions={0: (0.1,)})
    with pytest.raises(ValueError):
        tokenize(TIMES, VALUES, BANDS, band_wavelengths=WL, band_positions={0: (0.1,)})
    with pytest.raises(ValueError):
        tokenize(TIMES[:1], VALUES, BANDS)
    with pytest.raises(ValueError):
        tokenize(TIMES, [VALUES[0][:2], VALUES[1]], BANDS)
    negative = [torch.tensor([-1, 0, 1]), torch.tensor([-3])]
    for table in ("band_wavelengths", "band_positions", "band_features"):
        value = WL if table == "band_wavelengths" else {0: (0.1,), 1: (0.2,)}
        with pytest.raises(KeyError, match=r"\[-1\]"):
            tokenize(TIMES, VALUES, negative[:1] + BANDS[1:], **{table: value})
        with pytest.raises(KeyError, match=r"\[-3\]"):
            tokenize(TIMES, VALUES, BANDS[:1] + negative[1:], **{table: value})
    with pytest.raises(ValueError, match="non-negative"):
        tokenize(TIMES, VALUES, BANDS, band_wavelengths={-1: 480.0, 0: 620.0})


def test_wavelengths_for():
    assert wavelengths_for(["ZTF_g", "Gaia_G"]) == {0: 472.0, 1: 641.5}
    with pytest.raises(KeyError):
        wavelengths_for(["Kepler_K"])
