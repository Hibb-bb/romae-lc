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

    def add_decoder(self, d_model: int, c_dim: int = 32, hidden: int = 256, n_bands: int = 0, learn_sig: bool = False):
        """A magnitude decoder on [predicted state, CLS code, band] (``--dyn-decode``): the predicted state of the
        target token, with the source window's CLS for the star's amplitude and shape, must give the target's
        observed magnitude within its error. It makes the predicted latent decodable, not just matchable.
        ``n_bands > 0``: the target's band enters as a learned 8-number code (magnitudes are scaled per band).
        ``learn_sig``: a second output, the log of the model's own spread, so the loss is a Gaussian likelihood
        with variance sigma_obs^2 + sigma_model^2 instead of a fixed error."""
        self.dec_c = nn.Linear(d_model, c_dim)
        self.dec_nb, self.dec_learn_sig = int(n_bands), bool(learn_sig)
        b_dim = 8 if self.dec_nb > 0 else 0
        if self.dec_nb > 0:
            self.dec_band = nn.Embedding(self.dec_nb, b_dim)
        self.dec = nn.Sequential(nn.Linear(2 * self.K + c_dim + b_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 if self.dec_learn_sig else 1))
        if self.dec_learn_sig:
            with torch.no_grad():  # start with a small model spread (exp(-3) = 0.05 scaled magnitudes)
                self.dec[-1].bias[1] = -3.0

    def decode_full(self, s: torch.Tensor, c: torch.Tensor, band: torch.Tensor | None = None):
        """``(mean, log sigma_model or None)`` of the magnitude."""
        x = [s.float(), self.dec_c(c.float())]
        if getattr(self, "dec_nb", 0) > 0:
            b = torch.zeros(s.shape[:-1], dtype=torch.long, device=s.device) if band is None else band.long().clamp(0, self.dec_nb - 1)
            x.append(self.dec_band(b))
        o = self.dec(torch.cat(x, -1))
        return o[..., 0], (o[..., 1].clamp(-8.0, 3.0) if getattr(self, "dec_learn_sig", False) else None)

    def decode(self, s: torch.Tensor, c: torch.Tensor, band: torch.Tensor | None = None) -> torch.Tensor:
        return self.decode_full(s, c, band)[0]

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
                      dt_min: float = 0.5, tau: float = 1.0, w_nce: float = 0.2, w_sig: float = 1.0, w_mean: float = 1.0, w_pred: float = 1.0,
                      period=None, phase_tol: float | None = None, w_close: float = 0.0,
                      tgt_tokens=None, tgt_pad=None, tgt_t_abs=None, tgt_mag=None, tgt_sig=None, w_dec: float = 0.0, sig_floor: float = 0.05,
                      tgt_band=None, sig_mode: str = "star"):
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
        damping comes from the source window's CLS.

        Targets (``tgt_tokens [R, N2, D]``, ``tgt_pad``, ``tgt_t_abs``; ``--dyn-target ema``): the target
        states and the negatives come from these tokens (the JEPA target encoder on the full window) through
        the state head with no gradient, as a JEPA target; SIGReg and the mean act on the source states.
        Unset: targets are the source set itself (both sides get gradient; the J3 runs).

        Decoding (``w_dec > 0``, needs ``add_decoder`` and ``tgt_mag``/``tgt_sig [R, N2]``): the decoded
        prediction's z-score against the target's magnitude and error (floor ``sig_floor``), Huber loss."""
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
            MAG2 = SIG2 = BAND2 = None
            if tgt_tokens is None:
                S2, real2, t2, WID2, TN2 = S, real, t, WID, TN
                if tgt_mag is not None:
                    MAG2, SIG2 = star(tgt_mag.float()).gather(1, order), star(tgt_sig.float()).gather(1, order)
                if tgt_band is not None:
                    BAND2 = star(tgt_band.long()).gather(1, order)
            else:
                N2 = tgt_tokens.shape[1]

                def star2(x):
                    return x.reshape(T, B, N2, *x.shape[2:]).transpose(0, 1).reshape(B, T * N2, *x.shape[2:])

                with torch.no_grad():  # a JEPA target: no gradient into the target states
                    s2 = self.states(tgt_tokens.detach())
                S2, real2, TA2 = star2(s2), star2(~tgt_pad), star2(tgt_t_abs.float())
                WID2 = torch.arange(T, device=s.device).repeat_interleave(N2)[None].expand(B, T * N2)
                t2 = torch.where(real2, TA2, torch.full_like(TA2, float("inf")))
                order2 = t2.argsort(1)
                t2, real2, WID2 = t2.gather(1, order2), real2.gather(1, order2), WID2.gather(1, order2)
                S2 = S2.gather(1, order2[..., None].expand(-1, -1, ds))
                TN2 = T * N2
                if tgt_mag is not None:
                    MAG2, SIG2 = star2(tgt_mag.float()).gather(1, order2), star2(tgt_sig.float()).gather(1, order2)
                if tgt_band is not None:
                    BAND2 = star2(tgt_band.long()).gather(1, order2)
            P = int(n_pairs)
            u = torch.rand(B, P, device=s.device)
            i = (u * n_real[:, None]).long().clamp(min=0, max=TN - 1)  # source: a real token of the star
            ti, wi = t.gather(1, i), WID.gather(1, i)
            # target: a random real token of the star in ANOTHER window, at or after the source, at most dt_max later.
            # dt = 0 happens when windows overlap and the same observation sits in both: its two states must agree.
            # Short gaps only: the phase predicted over a gap drifts by f dt (relative frequency error), so long gaps cannot be scored.
            tt = torch.where(real2, t2, torch.zeros_like(t2))
            gap_all = tt[:, None, :] - ti[..., None]  # [B, P, TN2]
            cand = real2[:, None, :] & (WID2[:, None, :] != wi[..., None]) & (gap_all >= 0) & (gap_all <= dt_max)
            score = torch.rand(B, P, TN2, device=s.device).masked_fill(~cand, -1.0)
            j = score.argmax(-1)
            dt = t2.gather(1, j) - ti
            ok = (n_real > 1)[:, None] & cand.gather(2, j[..., None])[..., 0] & real.gather(1, i)
            del gap_all, cand, score
            dt = torch.where(ok, dt, torch.zeros_like(dt))
            si = S.gather(1, i[..., None].expand(-1, -1, ds))
            sj = S2.gather(1, j[..., None].expand(-1, -1, ds))
            ci = cls_star.gather(1, wi[..., None].expand(-1, -1, D))  # the source window's CLS
            pred = self.predict(si.reshape(-1, ds), ci.reshape(-1, D), f_star[:, None].expand(B, P).reshape(-1), dt.reshape(-1)).view(B, P, ds)
            m = ok.reshape(-1).float()
            n_ok = m.sum().clamp(min=1)
            l_pred = (((pred - sj) ** 2).mean(-1).reshape(-1) * m).sum() / n_ok
            # squared distances to every token of the star without the [B, P, TN, 2K] intermediate
            d2 = (pred.pow(2).sum(-1)[:, :, None] + S2.pow(2).sum(-1)[:, None, :] - 2 * torch.bmm(pred, S2.transpose(1, 2))).clamp(min=0)
            # negatives: the star's real tokens outside the source window (with overlapping windows the source's own
            # copy of an observation would otherwise compete, and telling the two copies apart needs the window clock)
            logits = (-d2 / tau).masked_fill(~real2[:, None, :] | (WID2[:, None, :] == wi[..., None]), -1e4)
            onehot = torch.zeros_like(logits, dtype=torch.bool).scatter_(2, j[..., None], True)
            pos = onehot
            if phase_tol is not None and period is not None:
                # same-phase positives: every token of the star outside the source window whose phase is within
                # phase_tol cycles of the target's also counts as a match (training-time catalogue period).
                # What must match is the phase, not the time since the window start or the exact token.
                p_star = period.to(s.device).double().reshape(T, B)[0]  # [B]
                okp = torch.isfinite(p_star) & (p_star > 0)
                ph = torch.remainder(torch.where(real2, t2, torch.zeros_like(t2)).double() / torch.where(okp, p_star, torch.ones_like(p_star))[:, None], 1.0)
                ph_j = ph.gather(1, j)
                dph = (ph[:, None, :] - ph_j[..., None]).abs()
                dph = torch.minimum(dph, 1.0 - dph)
                same = real2[:, None, :] & (WID2[:, None, :] != wi[..., None]) & (dph < phase_tol) & okp[:, None, None]
                pos = onehot | same
                del dph, same
            ce = torch.logsumexp(logits, -1) - torch.logsumexp(logits.masked_fill(~pos, -1e4), -1)  # multi-positive cross-entropy
            ce = torch.where(ok, ce, torch.zeros_like(ce)).reshape(-1)
            l_nce = (ce * m).sum() / n_ok
            rf = real.float()[..., None]
            s_mean = (S * rf).sum(1) / n_real.clamp(min=1)[:, None]  # per star
            # sig_mode "star": SIGReg on the states centred per star (within-star Gaussian, the test bed's choice);
            # "batch": SIGReg on the raw states pooled over all stars (across-star Gaussian, LeJEPA's usual choice),
            # which leaves each star free to trace a loop instead of filling a Gaussian cloud
            centred = (S - s_mean[:, None, :])[real] if sig_mode == "star" else S[real]
            l_sig = sigreg(centred) if centred.shape[0] >= 16 else s.new_zeros(())
            l_mean = s_mean.pow(2).mean()
            loss = w_pred * l_pred + w_nce * l_nce + w_sig * l_sig + w_mean * l_mean
            l_close = None
            if hasattr(self, "closure") and w_close > 0:
                # one full cycle of the Hamiltonian flow must return to its start (HamHead)
                c_tok = cls_star.gather(1, WID[..., None].expand(-1, -1, D))  # the CLS of every token's window (WID is sorted)
                l_close = self.closure(S[real], c_tok[real])
                loss = loss + w_close * l_close
            l_dec = z_in2 = sig_model = None
            if hasattr(self, "dec") and w_dec > 0 and MAG2 is not None:
                bj = BAND2.gather(1, j).reshape(-1) if BAND2 is not None else None
                mhat, ls = self.decode_full(pred.reshape(-1, ds), ci.reshape(-1, D), bj)
                mhat = mhat.view(B, P)
                sobs = SIG2.gather(1, j).clamp(min=sig_floor)
                z_obs = (mhat - MAG2.gather(1, j)) / sobs  # against the observation's error only (the test's yardstick)
                okd = ok & torch.isfinite(z_obs)
                nd = okd.float().sum().clamp(min=1)
                if ls is None:  # fixed error: Huber on the z-score
                    z = torch.where(okd, z_obs, torch.zeros_like(z_obs))
                    l_el = F.huber_loss(z, torch.zeros_like(z), reduction="none", delta=1.0)
                else:  # Gaussian likelihood with variance sigma_obs^2 + sigma_model^2
                    var = sobs ** 2 + torch.exp(2 * ls.view(B, P))
                    r = torch.where(okd, mhat - MAG2.gather(1, j), torch.zeros_like(mhat))
                    l_el = 0.5 * r ** 2 / var + 0.5 * torch.log(var)
                l_dec = (torch.where(okd, l_el, torch.zeros_like(l_el))).sum() / nd
                loss = loss + w_dec * l_dec
                with torch.no_grad():
                    z_in2 = ((z_obs.abs() < 2).float() * okd.float()).sum() / nd
                    sig_model = (torch.exp(ls.view(B, P)) * okd.float()).sum() / nd if ls is not None else None
            with torch.no_grad():
                hit = ((logits.argmax(-1) == j).float().reshape(-1) * m).sum() / n_ok
                hit_phase = (pos.gather(2, logits.argmax(-1)[..., None])[..., 0].float().reshape(-1) * m).sum() / n_ok  # nearest token is a positive
                n_pos = (pos.sum(-1).float().reshape(-1) * m).sum() / n_ok
                mdt = (dt.reshape(-1) * m).sum() / n_ok
                tgt_ms = ((sj.float() ** 2).mean(-1).reshape(-1) * m).sum() / n_ok
                stay = (((si.float() - sj.float()) ** 2).mean(-1).reshape(-1) * m).sum() / n_ok
            stats = dict(pred=float(l_pred.detach()), nce=float(l_nce.detach()), sig=float(l_sig.detach()), mean=float(l_mean.detach()),
                         hit=float(hit), hit_phase=float(hit_phase), n_pos=float(n_pos), close=float(l_close.detach()) if l_close is not None else 0.0,
                         dec=float(l_dec.detach()) if l_dec is not None else 0.0, dec_in2=float(z_in2) if z_in2 is not None else 0.0, dec_sig_model=float(sig_model) if sig_model is not None else 0.0, npairs=float(n_ok), dt_max=float(dt_max), dt_mean=float(mdt), state_ms=float(tgt_ms),
                         pred_rel=float(l_pred.detach() / tgt_ms.clamp(min=1e-8)), stay_rel=float(stay / tgt_ms.clamp(min=1e-8)))  # fmt: skip
        return loss, stats


class HamHead(DynHead):
    """The add-on with a symplectic Hamiltonian predictor (LeHami, Yalovetzky, Balestriero & LeCun 2026),
    shaped for sparse, irregular light curves.

    State pairs ``(q_k, p_k) = (s[2k], s[2k+1])``. Time is counted in cycles of the spectral frequency,
    ``tau = f dt``. The Hamiltonian is ``H = H0 + K1(p) + V1(q, c)``: ``H0`` is the harmonic oscillator whose
    exact flow rotates pair k by ``2 pi k tau`` (our earlier predictor without damping), ``K1`` and ``V1`` are
    small learned MLPs (zero at the start), the potential conditioned on the window's CLS ``c`` as LeHami
    conditions it on the action. One substep of size h is the Strang splitting rotate(h/2), leapfrog of
    ``K1 + V1`` over h, rotate(h/2): every part is exactly symplectic, so the step keeps phase-space volume
    for any parameters and nothing contracts the states.

    LeHami steps the flow frame by frame; our gaps span 1 to 100 cycles, which leapfrog cannot cross
    cheaply or accurately. So the flow advances only by the FRACTION of the elapsed cycles, ``tau mod 1``,
    in ``n_sub`` substeps, and a closure loss (``closure``) makes one full cycle of the flow return to its
    start, the condition under which skipping whole cycles is exact."""

    def __init__(self, d_model: int, K: int = 8, n_sub: int = 16, hidden: int = 64, c_dim: int = 16):
        super().__init__(d_model, K)
        self.n_sub = int(n_sub)
        self.c_proj = nn.Linear(d_model, c_dim)
        self.k_mlp = nn.Sequential(nn.Linear(self.K, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.v_mlp = nn.Sequential(nn.Linear(self.K + c_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        with torch.no_grad():  # the learned part starts at zero: the flow starts as the exact harmonic rotation
            for m in (self.k_mlp[-1], self.v_mlp[-1]):
                m.weight.zero_()
                m.bias.zero_()

    def _rotate(self, q, p, h):
        k = torch.arange(1, self.K + 1, device=q.device, dtype=q.dtype)
        ang = (2 * math.pi) * h[:, None] * k
        c, s = ang.cos(), ang.sin()
        return q * c - p * s, q * s + p * c

    def _flow(self, q, p, cz, tau):
        """Advance ``(q, p) [M, K]`` by ``tau [M]`` cycles (tau in [0, 1]) with the CLS code ``cz [M, c_dim]``."""
        h = tau / self.n_sub
        with torch.enable_grad():
            if not q.requires_grad:
                q = q.detach().requires_grad_(True)
            if not p.requires_grad:
                p = p.detach().requires_grad_(True)

            def dV(q_):
                return torch.autograd.grad(self.v_mlp(torch.cat([q_, cz], -1)).sum(), q_, create_graph=True)[0]

            def dK(p_):
                return torch.autograd.grad(self.k_mlp(p_).sum(), p_, create_graph=True)[0]

            for _ in range(self.n_sub):
                q, p = self._rotate(q, p, h / 2)
                p = p - (h / 2)[:, None] * dV(q)
                q = q + h[:, None] * dK(p)
                p = p - (h / 2)[:, None] * dV(q)
                q, p = self._rotate(q, p, h / 2)
        return q, p

    def predict(self, s_i, c_i, f_i, dt):
        tau = torch.remainder(f_i.float() * dt.float(), 1.0)
        q, p = self._flow(s_i[:, 0::2].float(), s_i[:, 1::2].float(), self.c_proj(c_i.float()), tau)
        return torch.stack([q, p], -1).flatten(-2)

    def closure(self, s, c, max_n: int = 1024):
        """Squared distance after one full cycle of the flow, over the states' mean square: 0 when the flow
        has period 1. The states are detached, so this term trains the flow, not the encoder."""
        if s.shape[0] > max_n:
            idx = torch.randperm(s.shape[0], device=s.device)[:max_n]
            s, c = s[idx], c[idx]
        s = s.detach().float()
        q, p = self._flow(s[:, 0::2], s[:, 1::2], self.c_proj(c.float()), torch.ones(s.shape[0], device=s.device))
        out = torch.stack([q, p], -1).flatten(-2)
        return ((out - s) ** 2).mean() / (s ** 2).mean().clamp(min=1e-8)
