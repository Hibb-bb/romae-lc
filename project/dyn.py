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
                tau: float = 1.0, w_nce: float = 0.2, w_sig: float = 1.0, w_mean: float = 1.0, w_pred: float = 1.0):
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
            # the prediction must pick j among the window's real tokens. Squared distances are formed directly:
            # torch.cdist takes a square root whose gradient is NaN at an exact zero distance, and a pair that
            # was masked out (dt = 0, no damping) predicts exactly its own source token (the smoke of 10-26)
            d2 = ((pred[:, :, None, :] - s[:, None, :, :]) ** 2).sum(-1)  # [R, P, N]
            # padded columns get a large finite penalty, not -inf: a training window can be empty (every token padding),
            # its pairs are masked out, and an infinite cross-entropy times a zero mask is NaN (the smoke of 10-26)
            logits = (-d2 / tau).masked_fill(pad[:, None, :], -1e4)
            ce = F.cross_entropy(logits.reshape(-1, N), j.reshape(-1), reduction="none")
            ce = torch.where(ok.reshape(-1), ce, torch.zeros_like(ce))
            l_nce = (ce * m).sum() / n_ok
            # the state holds only what changes within the window
            rf = real.float()[..., None]
            s_mean = (s * rf).sum(1) / n_real.clamp(min=1)[:, None]
            centred = (s - s_mean[:, None, :])[real]
            l_sig = sigreg(centred) if centred.shape[0] >= 16 else s.new_zeros(())
            l_mean = s_mean.pow(2).mean()
            loss = w_pred * l_pred + w_nce * l_nce + w_sig * l_sig + w_mean * l_mean
            with torch.no_grad():
                hit = ((logits.argmax(-1) == j).float().reshape(-1) * m).sum() / n_ok  # share of pairs where the prediction's nearest token is j
                # scale-free: the prediction error over the error of predicting zero (the targets' mean square);
                # SIGReg sets the state scale, so the raw squared error alone cannot show learning
                tgt_ms = ((sj.float() ** 2).mean(-1).reshape(-1) * m).sum() / n_ok
                stay = (((si.float() - sj.float()) ** 2).mean(-1).reshape(-1) * m).sum() / n_ok  # error of "no change"
            stats = dict(pred=float(l_pred.detach()), nce=float(l_nce.detach()), sig=float(l_sig.detach()), mean=float(l_mean.detach()),
                         hit=float(hit), npairs=float(n_ok), dt_max=float(dt_max), state_ms=float(tgt_ms),
                         pred_rel=float(l_pred.detach() / tgt_ms.clamp(min=1e-8)), stay_rel=float(stay / tgt_ms.clamp(min=1e-8)))  # fmt: skip
        return loss, stats

    def forward_cross(self, tokens, cls, pad, t_abs, freq, n_frames: int, n_b: int, dt_max: float, n_pairs: int = 256,
                      dt_min: float = 0.5, tau: float = 1.0, w_nce: float = 0.2, w_sig: float = 1.0, w_mean: float = 1.0, w_pred: float = 1.0):
        """The same losses with the pairs ACROSS a star's windows (the test bed's design).

        Why: inside one window a token knows its time since the window start from its rotary
        position, so a state that is a clock at the spectral frequency, ``s_i = R(2 pi f t_i)``,
        satisfies the within-window predictor and contrast with no light-curve content (the
        `enc_dyn` run of 10-26: predictor hit rate 0.32, phase probe unchanged). A clock does
        not carry across windows, because every window is encoded on its own with time counted
        from its own start. So the source token sits in one window and the target in a later
        one, the negatives are every real token of the star, and only the true phase can pick
        the target.

        Fused rows are frame ``t`` of every star in rows ``t B .. (t + 1) B``. ``t_abs [R, N]``
        is the token time in record days (window start + time in the window), ``freq [R]`` the
        window's estimate in cycles per day; the star uses the median over its windows. The
        damping comes from the source window's CLS."""
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            T, B = int(n_frames), int(n_b)
            R, N, D = tokens.shape
            s = self.states(tokens)  # [R, N, 2K]
            ds = s.shape[-1]

            def star(x):  # [R, N, ...] -> [B, T N, ...]
                return x.reshape(T, B, N, *x.shape[2:]).transpose(0, 1).reshape(B, T * N, *x.shape[2:])

            S, real, TA = star(s), star(~pad), star(t_abs.float())
            WID = torch.arange(T, device=s.device).repeat_interleave(N)[None].expand(B, T * N)
            cls_star = cls.float().reshape(T, B, D).transpose(0, 1)  # [B, T, D]
            f_star = freq.float().reshape(T, B).transpose(0, 1).median(1).values  # [B]
            n_real = real.sum(1)
            t = torch.where(real, TA, torch.full_like(TA, float("inf")))
            order = t.argsort(1)
            t, real, WID = t.gather(1, order), real.gather(1, order), WID.gather(1, order)
            S = S.gather(1, order[..., None].expand(-1, -1, ds))
            TN = T * N
            P = int(n_pairs)
            u = torch.rand(B, P, device=s.device)
            i = (u * n_real[:, None]).long().clamp(min=0, max=TN - 1)  # source: a real token of the star
            ti, wi = t.gather(1, i), WID.gather(1, i)
            # target: a random real token of the star in ANOTHER window, at or after the source, at most dt_max later.
            # dt = 0 happens when windows overlap and the same observation sits in both: its two states must agree.
            # Short gaps only: the phase predicted over a gap drifts by f dt (relative frequency error), so long gaps cannot be scored.
            tt = torch.where(real, t, torch.zeros_like(t))
            gap_all = tt[:, None, :] - ti[..., None]  # [B, P, TN]
            cand = real[:, None, :] & (WID[:, None, :] != wi[..., None]) & (gap_all >= 0) & (gap_all <= dt_max)
            score = torch.rand(B, P, TN, device=s.device).masked_fill(~cand, -1.0)
            j = score.argmax(-1)
            dt = t.gather(1, j) - ti
            wj = WID.gather(1, j)
            ok = (n_real > 1)[:, None] & cand.gather(2, j[..., None])[..., 0] & real.gather(1, i)
            del gap_all, cand, score
            dt = torch.where(ok, dt, torch.zeros_like(dt))
            si = S.gather(1, i[..., None].expand(-1, -1, ds))
            sj = S.gather(1, j[..., None].expand(-1, -1, ds))
            ci = cls_star.gather(1, wi[..., None].expand(-1, -1, D))  # the source window's CLS
            pred = self.predict(si.reshape(-1, ds), ci.reshape(-1, D), f_star[:, None].expand(B, P).reshape(-1), dt.reshape(-1)).view(B, P, ds)
            m = ok.reshape(-1).float()
            n_ok = m.sum().clamp(min=1)
            l_pred = (((pred - sj) ** 2).mean(-1).reshape(-1) * m).sum() / n_ok
            # squared distances to every token of the star without the [B, P, TN, 2K] intermediate
            d2 = (pred.pow(2).sum(-1)[:, :, None] + S.pow(2).sum(-1)[:, None, :] - 2 * torch.bmm(pred, S.transpose(1, 2))).clamp(min=0)
            # negatives: the star's real tokens outside the source window (with overlapping windows the source's own
            # copy of an observation would otherwise compete, and telling the two copies apart needs the window clock)
            logits = (-d2 / tau).masked_fill(~real[:, None, :] | (WID[:, None, :] == wi[..., None]), -1e4)
            ce = F.cross_entropy(logits.reshape(-1, TN), j.reshape(-1), reduction="none")
            ce = torch.where(ok.reshape(-1), ce, torch.zeros_like(ce))
            l_nce = (ce * m).sum() / n_ok
            rf = real.float()[..., None]
            s_mean = (S * rf).sum(1) / n_real.clamp(min=1)[:, None]  # per star
            centred = (S - s_mean[:, None, :])[real]
            l_sig = sigreg(centred) if centred.shape[0] >= 16 else s.new_zeros(())
            l_mean = s_mean.pow(2).mean()
            loss = w_pred * l_pred + w_nce * l_nce + w_sig * l_sig + w_mean * l_mean
            with torch.no_grad():
                hit = ((logits.argmax(-1) == j).float().reshape(-1) * m).sum() / n_ok
                mdt = (dt.reshape(-1) * m).sum() / n_ok
                tgt_ms = ((sj.float() ** 2).mean(-1).reshape(-1) * m).sum() / n_ok
                stay = (((si.float() - sj.float()) ** 2).mean(-1).reshape(-1) * m).sum() / n_ok
            stats = dict(pred=float(l_pred.detach()), nce=float(l_nce.detach()), sig=float(l_sig.detach()), mean=float(l_mean.detach()),
                         hit=float(hit), npairs=float(n_ok), dt_max=float(dt_max), dt_mean=float(mdt), state_ms=float(tgt_ms),
                         pred_rel=float(l_pred.detach() / tgt_ms.clamp(min=1e-8)), stay_rel=float(stay / tgt_ms.clamp(min=1e-8)))  # fmt: skip
        return loss, stats
