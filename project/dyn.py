"""Dynamic-state add-on for the masked autoencoder (label-free).

Dennis's synthetic test bed (top-level ``lcjepa.py``, 2026-10-23) showed that a
JEPA keeps the phase of each observation only when three things hold: the
predictor is told how much time passed and how fast the star's clock turns;
the predicted state must pick the true target moment out of the SAME star's
other moments, so nothing constant per star can help; and the state is
Gaussianised over time within a star, not across the batch. This module adds
that pressure to the real encoder's token outputs, on top of the
reconstruction loss.

Pieces, all on the visible-token outputs of one masked forward:

* ``s_i``: a small state per token (``2 K`` numbers, ``K`` rotation blocks)
  from a linear map of the token output.
* Predictor: ``s_j_hat = exp(dt A) s_i`` with ``A`` block-diagonal 2 x 2
  rotations at integer harmonics ``k f``, ``k = 1..K``, of the window's
  frequency ``f`` (cycles per day) and a damping per block from the CLS.
  ``f`` is the spectral layer's own local-centroid estimate
  (:func:`romae_lc.model.spectral_frequency`), so no catalogue period enters
  (the test bed used Lomb-Scargle, which failed on hard data; half-integer
  harmonics were unpredictable and are not used).
* Pairs ``(i, j)`` inside one window: ``i`` a random real token, the gap
  log-uniform in ``[dt_min, dt_max]``, ``j`` the real token nearest to
  ``t_i + gap``; ``dt_max`` ramps up over training (the caller passes it).
* Loss: squared error of the prediction; a contrastive term (the prediction
  must pick ``j`` among all real tokens of the window, squared-distance
  logits over ``tau``); SIGReg (LeJEPA's sliced Epps-Pulley test against
  N(0, I)) on the states centred per window; and the squared per-window mean
  of the states, so the state holds only what changes.

Nothing here is used at inference: the encoder is probed as before.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigreg(z: torch.Tensor, num_slices: int = 64, t_max: float = 3.0, n_t: int = 17, max_n: int = 2048) -> torch.Tensor:
    """LeJEPA's SIGReg: the sliced Epps-Pulley statistic of ``z [n, d]``
    against N(0, I) (from the test bed, unchanged)."""
    if z.shape[0] > max_n:
        z = z[torch.randperm(z.shape[0], device=z.device)[:max_n]]
    n, d = z.shape
    A = torch.randn(d, num_slices, device=z.device, dtype=z.dtype)
    A = A / A.norm(dim=0, keepdim=True)
    p = z @ A
    tt = torch.linspace(0, t_max, n_t, device=z.device, dtype=z.dtype)
    x = p[..., None] * tt
    re, im = x.cos().mean(0), x.sin().mean(0)
    target = torch.exp(-0.5 * tt ** 2)
    err = (re - target) ** 2 + im ** 2
    w = torch.exp(-0.5 * tt ** 2)
    return torch.trapezoid(err * w, tt, dim=-1).mean() * 10


class DynHead(nn.Module):
    """State head + rotation predictor + the losses (see the module docstring).

    Args:
        d_model: the encoder width (token and CLS outputs).
        K: rotation blocks; the state has ``2 K`` numbers.
    """

    def __init__(self, d_model: int, K: int = 8):
        super().__init__()
        self.K = int(K)
        self.norm = nn.LayerNorm(d_model)
        self.s_head = nn.Linear(d_model, 2 * self.K)
        self.damp = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, self.K))
        with torch.no_grad():  # start with little damping, as in the test bed (the weights stay random so the CLS gets a gradient from step 1)
            self.damp[-1].bias.fill_(-4.0)

    def states(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.s_head(self.norm(tokens.float()))

    def predict(self, s_i: torch.Tensor, c_i: torch.Tensor, f_i: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """``s_i [M, 2K]`` advanced by ``dt [M]`` days at frequency ``f_i [M]``
        (cycles per day) with damping from ``c_i [M, D]``."""
        g = F.softplus(self.damp(c_i.float()))  # [M, K]
        k = torch.arange(1, self.K + 1, device=s_i.device, dtype=s_i.dtype)
        ang = (2 * math.pi) * f_i[:, None] * k * dt[:, None]
        damp = torch.exp(-g * dt[:, None])
        x1, x2 = s_i[:, 0::2], s_i[:, 1::2]
        y1 = damp * (x1 * ang.cos() - x2 * ang.sin())
        y2 = damp * (x1 * ang.sin() + x2 * ang.cos())
        return torch.stack([y1, y2], -1).flatten(-2)

    def forward(self, tokens, cls, pad, t_days, freq, dt_max: float, n_pairs: int = 128, dt_min: float = 0.5,
                tau: float = 1.0, w_nce: float = 0.2, w_sig: float = 1.0, w_mean: float = 1.0):
        """``tokens [R, N, D]`` (no CLS), ``cls [R, D]``, ``pad [R, N]`` (True =
        padding; real tokens come first, sorted by time, as the tokenizer
        makes them), ``t_days [R, N]`` time since the window start, ``freq [R]``
        cycles per day. Returns ``(loss, stats)``."""
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            s = self.states(tokens)  # [R, N, 2K]
            R, N, ds = s.shape
            real = ~pad
            n_real = real.sum(1)
            t = torch.where(real, t_days.float(), torch.full_like(t_days.float(), float("inf")))
            P = int(n_pairs)
            u = torch.rand(R, P, device=s.device)
            i = (u * n_real[:, None]).long().clamp(min=0, max=N - 1)
            ti = t.gather(1, i)
            gap = dt_min * (dt_max / dt_min) ** torch.rand(R, P, device=s.device)
            want = (ti + gap).contiguous()
            j = torch.searchsorted(t.contiguous(), want).clamp(max=N - 1)
            jm = (j - 1).clamp(min=0)
            pick_prev = (t.gather(1, jm) - want).abs() < (t.gather(1, j) - want).abs()
            j = torch.where(pick_prev, jm, j)
            j = torch.minimum(j, (n_real - 1).clamp(min=0)[:, None])
            dt = t.gather(1, j) - ti
            ok = (n_real > 1)[:, None] & (dt > 0) & (dt <= 1.5 * dt_max) & (j != i) & real.gather(1, i) & real.gather(1, j)
            dt = torch.where(ok, dt, torch.zeros_like(dt))
            si = s.gather(1, i[..., None].expand(-1, -1, ds))
            sj = s.gather(1, j[..., None].expand(-1, -1, ds))
            D = cls.shape[-1]
            pred = self.predict(si.reshape(-1, ds), cls.float()[:, None, :].expand(R, P, D).reshape(-1, D),
                                freq.float()[:, None].expand(R, P).reshape(-1), dt.reshape(-1)).view(R, P, ds)
            m = ok.reshape(-1).float()
            n_ok = m.sum().clamp(min=1)
            l_pred = (((pred - sj) ** 2).mean(-1).reshape(-1) * m).sum() / n_ok
            # the prediction must pick j among the window's real tokens
            d2 = torch.cdist(pred, s).pow(2)  # [R, P, N]
            logits = (-d2 / tau).masked_fill(pad[:, None, :], float("-inf"))
            ce = F.cross_entropy(logits.reshape(-1, N), j.reshape(-1), reduction="none")
            l_nce = (ce * m).sum() / n_ok
            # the state holds only what changes within the window
            rf = real.float()[..., None]
            s_mean = (s * rf).sum(1) / n_real.clamp(min=1)[:, None]
            centred = (s - s_mean[:, None, :])[real]
            l_sig = sigreg(centred) if centred.shape[0] >= 16 else s.new_zeros(())
            l_mean = s_mean.pow(2).mean()
            loss = l_pred + w_nce * l_nce + w_sig * l_sig + w_mean * l_mean
            with torch.no_grad():
                hit = ((logits.argmax(-1) == j).float().reshape(-1) * m).sum() / n_ok  # share of pairs where the prediction's nearest token is j
            stats = dict(pred=float(l_pred.detach()), nce=float(l_nce.detach()), sig=float(l_sig.detach()), mean=float(l_mean.detach()),
                         hit=float(hit), npairs=float(n_ok), dt_max=float(dt_max))  # fmt: skip
        return loss, stats
