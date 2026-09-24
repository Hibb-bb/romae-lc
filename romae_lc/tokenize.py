"""Tokenize asynchronous multi-band light curves for RoMAE.

Every observation becomes one token; bands may be sampled at different
epochs and in different numbers, with no alignment or imputation. The
n-dimensional rotary encoding then attends jointly over the time axis and the
wavelength axis.

The band axis is a **physical wavelength coordinate** ``log(lambda /
lambda_ref) / wavelength_scale``: cross-band attention distance reflects
spectral distance, and a new filter set is just new points on the same axis.
The relative scale between time and wavelength positions is the one honest
hyperparameter (``time_scale`` and ``wavelength_scale``), see
:mod:`romae_lc.analysis` for choosing the time unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

#: Effective wavelengths in nm of common passbands, keyed ``<survey>_<band>``.
BAND_WAVELENGTHS_NM: dict[str, float] = {
    "LSST_u": 368.0,
    "LSST_g": 480.7,
    "LSST_r": 622.1,
    "LSST_i": 755.9,
    "LSST_z": 869.0,
    "LSST_y": 971.0,
    "ZTF_g": 472.0,
    "ZTF_r": 634.0,
    "ZTF_i": 789.0,
    "Gaia_G": 641.5,
    "TESS_T": 797.2,
    "ASASSN_V": 551.0,
    "ASASSN_g": 480.7,
    # PC_matches band names (romae_lc.data.PC_BANDS); effective wavelengths
    "g_ZTF": 472.0,
    "r_ZTF": 634.0,
    "i_ZTF": 789.0,
    "c_ATLAS": 533.0,
    "o_ATLAS": 679.0,
    "g_ASASSN": 480.7,
    "V_ASASSN": 551.0,
    "V_CSS": 551.0,  # unfiltered, calibrated to V
    "r_LINEAR": 622.0,  # unfiltered, calibrated to r
    "g_PTF": 480.0,
    "R_PTF": 658.0,
    "J_PGIR": 1250.0,
    "TESS": 797.2,
    "g_PS1": 481.0,
    "r_PS1": 617.0,
    "i_PS1": 752.0,
    "z_PS1": 866.0,
    "y_PS1": 962.0,
    "G_Gaia": 641.5,
    "BP_Gaia": 513.0,
    "RP_Gaia": 783.0,
}


def wavelengths_for(bands: Sequence[str]) -> dict[int, float]:
    """``{band index: wavelength nm}`` for a band name list, by position."""
    return {i: BAND_WAVELENGTHS_NM[b] for i, b in enumerate(bands)}


@dataclass
class Tokens:
    """A padded token batch: ``values [B, N, C]``, ``positions [B, n_axes, N]``
    and ``pad_mask [B, N]`` (True = padding). Unpacks to those three tensors,
    so a backbone call is ``model(*tokens)``. ``extras [B, N]`` optionally
    carries a per-token side quantity (e.g. uncertainties) sorted and padded
    like the tokens."""

    values: torch.Tensor
    positions: torch.Tensor
    pad_mask: torch.Tensor
    extras: torch.Tensor | None = None

    def __iter__(self):
        return iter((self.values, self.positions, self.pad_mask))

    def to(self, device, non_blocking: bool = False) -> "Tokens":
        f = lambda t: None if t is None else t.to(device, non_blocking=non_blocking)
        return Tokens(
            f(self.values), f(self.positions), f(self.pad_mask), f(self.extras)
        )

    @property
    def n_real(self) -> torch.Tensor:
        return (~self.pad_mask).sum(1)


def _table(table: dict, what: str) -> torch.Tensor:
    """Dense ``[max_id + 1, width]`` lookup with NaN for missing ids."""
    if any(int(k) < 0 for k in table):
        raise ValueError(f"{what}: band ids must be non-negative")
    width = len(next(iter(table.values())))
    lut = torch.full((int(max(table)) + 1, width), float("nan"))
    for k, row in table.items():
        if len(row) != width:
            raise ValueError(
                f"{what}: band {k} has {len(row)} entries, expected {width}"
            )
        lut[int(k)] = torch.tensor([float(x) for x in row])
    return lut


def _lookup(lut: torch.Tensor, band: torch.Tensor, i: int, what: str) -> torch.Tensor:
    rows = lut[band.clamp(0, lut.shape[0] - 1)]
    bad = (band < 0) | (band >= lut.shape[0]) | torch.isnan(rows).any(-1)
    if bad.any():
        raise KeyError(
            f"object {i}: band ids {sorted(set(band[bad].tolist()))} missing from {what}"
        )
    return rows


def tokenize(
    times: Sequence[torch.Tensor],
    values: Sequence[torch.Tensor],
    bands: Sequence[torch.Tensor],
    band_wavelengths: dict[int, float] | None = None,
    band_positions: dict[int, Sequence[float]] | None = None,
    band_features: dict[int, Sequence[float]] | None = None,
    time_scale: float = 1.0,
    wavelength_scale: float = 1.0,
    ref_wavelength: float | None = None,
    extras: Sequence[torch.Tensor] | None = None,
) -> Tokens:
    """Tokenize a list of asynchronous multi-band light curves.

    Args:
        times: Per-object 1-D tensors of observation epochs (any order).
        values: Per-object 1-D tensors of fluxes or magnitudes.
        bands: Per-object 1-D integer tensors of band ids.
        band_wavelengths: ``{band id: effective wavelength}`` (any consistent
            unit); the wavelength position is ``log(lambda / lambda_ref) /
            wavelength_scale``. ``None`` uses the raw band id (ablation).
        band_positions: ``{band id: k extra coordinates}`` (already in final
            units, e.g. log transmission percentiles); replaces the
            ``band_wavelengths`` axis so positions become ``[time, *coords]``.
            ``k = 0`` gives a time-only encoding.
        band_features: ``{band id: F content features}`` appended to the value
            channel, so values become ``[B, N, 1 + F]``.
        time_scale: Time positions are ``t / time_scale``.
        wavelength_scale: Divides the log-wavelength coordinate.
        ref_wavelength: Center of the log-wavelength axis; default the
            geometric mean of ``band_wavelengths``.
        extras: Optional per-object 1-D tensors (e.g. uncertainties) returned
            sorted and padded like the tokens in ``Tokens.extras``.

    Returns:
        :class:`Tokens` with tokens sorted by time within each object. All
        positions carry a +1 offset so the CLS token alone owns position 0;
        padding positions are 0.
    """
    if not (len(times) == len(values) == len(bands)):
        raise ValueError("times, values and bands must have the same length")
    if band_positions is not None and band_wavelengths is not None:
        raise ValueError("pass band_positions or band_wavelengths, not both")
    n_obj = len(times)
    n_max = max((t.numel() for t in times), default=0)

    pos_lut = feat_lut = wave_lut = None
    if band_positions is not None:
        pos_lut = _table(band_positions, "band_positions")
    elif band_wavelengths is not None:
        wave_lut = _table(
            {k: (v,) for k, v in band_wavelengths.items()}, "band_wavelengths"
        )
        if ref_wavelength is None:
            ref_wavelength = wave_lut[~torch.isnan(wave_lut)].log().mean().exp().item()
    if band_features is not None:
        feat_lut = _table(band_features, "band_features")
    n_wave = pos_lut.shape[1] if pos_lut is not None else 1
    n_feat = feat_lut.shape[1] if feat_lut is not None else 0

    out_values = torch.zeros(n_obj, n_max, 1 + n_feat)
    out_pos = torch.zeros(n_obj, 1 + n_wave, n_max)
    pad_mask = torch.ones(n_obj, n_max, dtype=torch.bool)
    out_extra = torch.zeros(n_obj, n_max) if extras is not None else None

    for i, (t, v, b) in enumerate(zip(times, values, bands)):
        if not (t.numel() == v.numel() == b.numel()):
            raise ValueError(f"object {i}: times/values/bands lengths differ")
        n = t.numel()
        order = torch.argsort(t.float())
        t, v, b = t.float()[order], v.float()[order], b.long()[order]
        if pos_lut is not None:
            wave = _lookup(pos_lut, b, i, "band_positions").T  # (k, n)
        elif wave_lut is not None:
            lam = _lookup(wave_lut, b, i, "band_wavelengths")[:, 0]
            wave = (torch.log(lam / ref_wavelength) / wavelength_scale)[None]
        else:
            wave = b.float()[None]
        out_values[i, :n, 0] = v
        if feat_lut is not None:
            out_values[i, :n, 1:] = _lookup(feat_lut, b, i, "band_features")
        if out_extra is not None:
            out_extra[i, :n] = extras[i].float()[order]
        out_pos[i, 0, :n] = t / time_scale + 1.0
        out_pos[i, 1:, :n] = wave + 1.0
        pad_mask[i, :n] = False
    return Tokens(out_values, out_pos, pad_mask, out_extra)
