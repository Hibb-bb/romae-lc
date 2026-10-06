"""A spectral layer: a learned periodogram inside the encoder.

For every token the layer takes a learned vector ``a_j = W h_j`` (``M``
channels) and, for every trial frequency ``f_k`` on a fixed fine log grid,
forms the non-uniform Fourier sum over the window's tokens at their actual
times::

    Z[k, m] = sum_j a[j, m] exp(2 pi i f_k t_j)

The squared magnitudes ``|Z|^2`` (plus their logs) are a ``[2M, K]`` map
per window, the spectrum of the learned features. With ``a_j`` the
weighted centred brightness and one channel this is the Lomb-Scargle
periodogram (up to its sampling correction); with learned ``a_j`` it is a
periodogram of whatever the encoder's features find useful, trained end to
end. A small dilated 1-D convolutional reader along the frequency axis
turns the map into a logit per frequency bin and a summary vector that is
added to the CLS token, so the pooled latent can carry a period at the
grid's resolution. The reader's logits are kept (``last_logits``) for an
optional auxiliary loss on a catalogue period.

Why this and not a wider rotary ladder: rotary rungs are summed inside one
dot product before the softmax, so the power at a rung is never formed and
there are a few hundred rungs; here every frequency keeps its own output
and the cost is linear in the number of frequencies (``N x K x M`` per
window), so ``K`` can be ten thousand or more. See
``docs/SPECTRAL-LAYER.md``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SpectralLayer(nn.Module):
    """``x [B, 1 + N, d_model]`` (CLS first) with ``positions [B, n_axes,
    1 + N]`` and ``pad_mask [B, 1 + N]`` -> the same ``x`` with the
    spectrum's summary added to the CLS token.

    Args:
        d_model: Token width.
        time_scale: Days per position unit (positions carry ``t / time_scale
            + 1``; the window's start is position 1).
        p_min, p_max: Period range of the grid, days.
        rel: Relative frequency step of the grid (5e-4 = 20k bins over
            0.02-500 d).
        channels: ``M``, the learned channels summed over tokens.
        reader: Width of the convolutional reader along frequency.
        depth: Reader blocks (dilations 1, 2, 4, ...).
        kernel: Reader kernel size.
        chunk: Frequencies per chunk of the sum (memory).
        after_layer: Which encoder layer's output the layer reads (the
            encoder runs ``after_layer`` blocks, this layer, the rest).
    """

    def __init__(
        self,
        d_model: int,
        time_scale: float,
        p_min: float = 0.02,
        p_max: float = 500.0,
        rel: float = 5e-4,
        channels: int = 8,
        reader: int = 32,
        depth: int = 3,
        kernel: int = 9,
        chunk: int = 512,
        after_layer: int = 2,
    ):
        super().__init__()
        self.d_model, self.time_scale = int(d_model), float(time_scale)
        self.p_min, self.p_max, self.rel = float(p_min), float(p_max), float(rel)
        self.channels, self.reader_width, self.depth, self.kernel = int(channels), int(reader), int(depth), int(kernel)
        self.chunk, self.after_layer = int(chunk), int(after_layer)
        f_lo, f_hi = 1.0 / self.p_max, 1.0 / self.p_min
        self.step = math.log1p(self.rel)
        self.n_bins = int(math.ceil(math.log(f_hi / f_lo) / self.step)) + 1
        self.log_f0 = math.log(f_lo)
        freqs = torch.exp(self.log_f0 + self.step * torch.arange(self.n_bins, dtype=torch.float64))
        self.register_buffer("freqs", freqs.float())
        # the summed vector reads the token features AND the raw value channels
        # (brightness, standardised log sigma); channel 0 starts as the raw
        # brightness, so the layer begins as a Lomb-Scargle-like periodogram
        # of the signal instead of the spectrum of random features, which is
        # the sampling pattern's window function and carries no period
        # (the first run, 2026-10-02: the reader could not fit even the
        # catalogue bin, aux CE 9.9 -> 8.5 against log(20260) = 9.9)
        self.n_values = 2
        self.proj = nn.Linear(d_model + self.n_values, channels)
        self.shift = int(round(math.log(2.0) / self.step))  # bins per factor of two
        self.inp = nn.Conv1d(6 * channels, reader, kernel, padding=kernel // 2)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.GroupNorm(8, reader), nn.Conv1d(reader, reader, kernel, padding=2**i * (kernel // 2), dilation=2**i), nn.GELU(), nn.Conv1d(reader, reader, 1))
            for i in range(depth)
        )
        self.logit = nn.Conv1d(reader, 1, 1)
        self.summary_norm = nn.LayerNorm(reader + 2)
        self.summary = nn.Linear(reader + 2, d_model)
        nn.init.zeros_(self.summary.weight)
        nn.init.zeros_(self.summary.bias)
        with torch.no_grad():  # channel 0 = the brightness (value channel 0), the rest small
            self.proj.weight.mul_(0.1)
            self.proj.weight[0].zero_()
            self.proj.weight[0, d_model] = 1.0
            self.proj.bias.zero_()
        self.last_logits: torch.Tensor | None = None

    @property
    def hparams(self) -> dict:
        return dict(time_scale=self.time_scale, p_min=self.p_min, p_max=self.p_max, rel=self.rel, channels=self.channels,
                    reader=self.reader_width, depth=self.depth, kernel=self.kernel, chunk=self.chunk, after_layer=self.after_layer)  # fmt: skip

    def bin_of(self, period: torch.Tensor) -> torch.Tensor:
        """Continuous bin coordinate of periods (days)."""
        return (torch.log(1.0 / period.double()) - self.log_f0) / self.step

    def period_of(self, coord: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.exp(self.log_f0 + self.step * coord.double())

    def _chunk_power(self, a: torch.Tensor, t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """``|Z|^2 [B, M, k]`` for ``k`` frequencies: ``a [B, N, M]``
        (zero on padding), ``t [B, N]`` days."""
        ph = (2 * math.pi) * t[:, :, None] * freqs[None, None, :]  # [B, N, k], float32
        c, s = torch.cos(ph), torch.sin(ph)
        at = a.transpose(1, 2)  # [B, M, N]
        re, im = at @ c, at @ s  # [B, M, k]
        return re * re + im * im

    def spectrum(self, x_tokens: torch.Tensor, t_days: torch.Tensor, real: torch.Tensor, values: torch.Tensor | None = None) -> torch.Tensor:
        """The ``[B, 6M, K]`` map: ``|Z|^2`` per channel, read at ``f``,
        ``2f`` and ``f/2`` (shifted copies, so a harmonic sits next to its
        fundamental), each as is and log-scaled. ``values [B, N, n_values]``
        are the raw value channels of the tokens (zeros when absent)."""
        if values is None:
            values = x_tokens.new_zeros(*x_tokens.shape[:2], self.n_values)
        feats = torch.cat([x_tokens.float(), values[..., : self.n_values].float()], -1)
        r = real[..., None].float()
        a = self.proj(feats) * r  # [B, N, M]
        a = (a - (a * r).sum(1, keepdim=True) / r.sum(1, keepdim=True).clamp_min(1.0)) * r  # centred per window: no DC leak
        n = real.float().sum(1).clamp_min(1.0)[:, None, None]
        powers = []
        for lo in range(0, self.n_bins, self.chunk):
            f = self.freqs[lo : lo + self.chunk]
            if torch.is_grad_enabled() and a.requires_grad:
                p = checkpoint(self._chunk_power, a, t_days, f, use_reentrant=False)
            else:
                p = self._chunk_power(a, t_days, f)
            powers.append(p)
        power = torch.cat(powers, -1) / (n * n)  # scale-free in the token count
        k = power.shape[-1]
        twice, half = torch.zeros_like(power), torch.zeros_like(power)
        if self.shift < k:
            twice[..., : k - self.shift] = power[..., self.shift :]
            half[..., self.shift :] = power[..., : k - self.shift]
        lin = torch.cat([power, twice, half], 1)
        return torch.cat([lin, torch.log(lin + 1e-6)], 1)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, pad_mask: torch.Tensor | None, has_cls: bool = True,
                values: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
        b, n_all, _ = x.shape
        off = 1 if has_cls else 0
        tokens = x[:, off:]
        t_days = (positions[:, 0, off:].float() - 1.0) * self.time_scale
        real = torch.ones(b, n_all - off, dtype=torch.bool, device=x.device) if pad_mask is None else ~pad_mask[:, off:]
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = self.spectrum(tokens, t_days, real, values)  # [B, 6M, K]
            spec = (spec - spec.mean(-1, keepdim=True)) / (spec.std(-1, keepdim=True) + 1e-6)
            h = self.inp(spec)
            for blk in self.blocks:
                h = h + blk(h)
            logits = self.logit(h)[:, 0]  # [B, K]
            self.last_logits = logits
            p = torch.softmax(logits, -1)
            pooled = (h * p[:, None, :]).sum(-1)  # [B, reader]
            coord = (p * torch.arange(self.n_bins, device=x.device, dtype=torch.float32)).sum(-1) / self.n_bins
            entropy = -(p * torch.log(p + 1e-9)).sum(-1) / math.log(self.n_bins)
            summary = self.summary(self.summary_norm(torch.cat([pooled, coord[:, None], entropy[:, None]], -1)))
        if has_cls:
            x = torch.cat([(x[:, 0] + summary.to(x.dtype))[:, None], x[:, 1:]], 1)
        else:
            x = x + summary.to(x.dtype)[:, None, :]
        return x


def soft_bin_target(coord: torch.Tensor, n_bins: int, sigma: float = 1.0) -> torch.Tensor:
    idx = torch.arange(n_bins, device=coord.device, dtype=torch.float32)
    return torch.softmax(-0.5 * ((idx[None] - coord.float()[:, None]) / sigma) ** 2, 1)


def spectral_aux_loss(layer: SpectralLayer, periods: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Cross-entropy of the reader's logits against the bin of the catalogue
    period (softened), over rows with a finite positive period."""
    logits = layer.last_logits
    if logits is None:
        raise RuntimeError("the spectral layer has not run yet")
    ok = torch.isfinite(periods) & (periods > 0)
    if ok.sum() == 0:
        return logits.sum() * 0.0
    coord = layer.bin_of(periods[ok].to(logits.device)).float()
    target = soft_bin_target(coord, layer.n_bins, sigma)
    return -(target * F.log_softmax(logits[ok].float(), 1)).sum(1).mean()
