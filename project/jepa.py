"""Token-level JEPA: predict the latents of hidden tokens, not their values.

The context encoder (the same encoder as the masked autoencoder, spectral
layer included) sees the visible tokens of a window. An exponential moving
average (EMA) copy of it, the target encoder, sees the whole window; its
token outputs at the hidden positions, layer-normalised, are the targets.
The predictor is the autoencoder's light decoder with a mask token per
hidden position and a head that outputs the encoder width instead of a
brightness. Loss: smooth L1 (or MSE) between predicted and target latents
over the real hidden tokens. No reconstruction, no catalogue, no
Gaussianisation term: the EMA target, the stop-gradient and the asymmetric
predictor are what keeps the latents from collapsing (I-JEPA recipe).

Why token level: a next-window latent is predictable from anything constant
per star, so the encoder keeps the cheapest constant and drops the period;
a hidden point's latent depends on its brightness at that time, and with
hidden stretches of many days the predictor can only get it right by
knowing where the star is in its cycle.

No-EMA form (``no_ema``, LeJEPA): one encoder for both sides, gradients through the targets, and SIGReg on
the projected token latents keeps them from collapsing. Training then uses :func:`forward_views`:
2 global and 4 local views of a window, each with point and band dropout; the target of an observation is the
mean of the global views' projected latents of it; the predictor predicts it from every view.

The checkpoint has ``kind == "jepa"``; :func:`project.checkpoint.load_mae`
rebuilds it and every loader and probe treats it like ``mae.pt``. By default
the TARGET encoder is the one used downstream (``use_target``), as in
I-JEPA; ``encode`` / ``backbone`` follow that choice.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from romae_lc.model import MAEOutput, RoMAEForPreTraining, attention_mask, gen_mask, spectral_frequency
from romae_lc.spectral import spectral_aux_loss

from project.views import make_views, pack, take


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


class TokenJEPA(nn.Module):
    def __init__(
        self,
        decoder,
        mask_ratio: float,
        encoder: dict,
        n_channels: int,
        n_axes: int = 2,
        rope="axial",
        use_cls: bool = True,
        abs_timescales=None,
        spectral: dict | None = None,
        target_channels: int | None = None,
        ema: float = 0.996,
        ema_end: float = 1.0,
        loss: str = "smoothl1",
        use_target: bool = True,
        recon_weight: float = 0.0,
        clean_target: bool = False,
        proj_dim: int = 0,
        proj_hidden: int = 256,
        attn_pool: int = 0,
        phase_rope: int = 0,
        phase_freq: str = "local",
        no_ema: bool = False,
        sig_slices: int = 1024,
    ):
        super().__init__()
        d = encoder["d_model"] if isinstance(encoder, dict) else encoder.d_model
        if target_channels is not None and target_channels != d:
            raise ValueError(f"the JEPA head outputs the encoder width {d}, not {target_channels}")
        self.context = RoMAEForPreTraining(decoder=decoder, mask_ratio=mask_ratio, target_channels=d, encoder=encoder, n_channels=n_channels,
                                           n_axes=n_axes, rope=rope, use_cls=use_cls, abs_timescales=abs_timescales, spectral=spectral,
                                           attn_pool=attn_pool, phase_rope=phase_rope, phase_freq=phase_freq)  # fmt: skip
        # the predictor's rotary encoding gets the phase at the context encoder's own frequency as one more axis
        # (0 = off); the spectral layer's estimate, never the catalogue period
        self.phase_rope, self.phase_freq = int(phase_rope), str(phase_freq)
        self.no_ema, self.sig_slices = bool(no_ema), int(sig_slices)
        self.target = copy.deepcopy(self.context)  # unused (never updated) when no_ema
        self.target.requires_grad_(False)
        self.ema, self.ema_end, self.loss_kind, self.use_target = float(ema), float(ema_end), str(loss), bool(use_target) and not self.no_ema
        self.recon_weight = float(recon_weight)
        # clean_target (a denoising form) and attn_pool are options of older code. The arguments stay because
        # checkpoints store them; the JEPA forward no longer reads clean_target (its code is in legacy/).
        self.clean_target = bool(clean_target)
        # the alignment loss in a low-dimensional projection (BYOL / LeJEPA form): the encoder keeps what the
        # loss does not need, the projection discards it; the probes read the pre-projection latent
        self.proj_dim, self.proj_hidden = int(proj_dim), int(proj_hidden)
        if self.proj_dim > 0:
            self.proj = nn.Sequential(nn.Linear(d, proj_hidden), nn.GELU(), nn.Linear(proj_hidden, self.proj_dim))
            self.proj_target = copy.deepcopy(self.proj)
            self.proj_target.requires_grad_(False)
        else:
            self.proj = self.proj_target = None
        # the hybrid loss: a second head on the predictor's outputs reconstructs the hidden brightness, weighted by recon_weight
        self.recon_head = nn.Linear(self.context.dec_cfg.d_model, 1) if self.recon_weight > 0 else None
        self.last_stats: dict = {}

    # ---- what the loaders, probes and the pretraining script read
    @property
    def jepa_hparams(self) -> dict:
        return dict(ema=self.ema, ema_end=self.ema_end, loss=self.loss_kind, use_target=self.use_target, recon_weight=self.recon_weight, clean_target=self.clean_target,
                    proj_dim=self.proj_dim, proj_hidden=self.proj_hidden,
                    **({"phase_rope": self.phase_rope, "phase_freq": self.phase_freq} if self.phase_rope else {}),
                    **({"no_ema": True, "sig_slices": self.sig_slices} if self.no_ema else {}))

    @property
    def hparams(self) -> dict:
        return self.context.hparams

    @property
    def cfg(self):
        return self.context.cfg

    @property
    def dec_cfg(self):
        return self.context.dec_cfg

    @property
    def mask_ratio(self):
        return self.context.mask_ratio

    @property
    def target_channels(self):
        return self.context.target_channels

    @property
    def use_cls(self):
        return self.context.use_cls

    @property
    def rope_layout(self):
        return self.context.rope_layout

    @property
    def embed_dim(self):
        return self.context.embed_dim

    @property
    def transformer(self):
        return self.context.transformer

    @property
    def projection(self):
        return self.context.projection

    @property
    def spectral(self):
        return self.context.spectral

    @property
    def attn_pool(self):
        """The downstream encoder's attention-pooling head (None without one)."""
        return self.chosen.attn_pool

    @property
    def chosen(self) -> RoMAEForPreTraining:
        return self.target if self.use_target else self.context

    def encode(self, values, positions, pad_mask=None):
        return self.chosen.encode(values, positions, pad_mask)

    def backbone(self, pool: str = "cls"):
        return self.chosen.backbone(pool)

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "TokenJEPA":
        model = cls(**ckpt["backbone"], **ckpt["mae"], **ckpt["jepa"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    # ---- training
    def init_from_mae(self, ckpt: dict) -> None:
        """Warm start: the context encoder's backbone (projection, transformer,
        spectral layer, CLS, absolute-time features) and the predictor's
        decoder from a masked-autoencoder checkpoint of the same shape; the
        target encoder starts as a copy. The MAE head (width 1) is not
        copied: the JEPA head outputs the encoder width."""
        sd = ckpt["state_dict"]
        if ckpt.get("kind") == "jepa":
            # a JEPA checkpoint (e.g. J1): every tensor of the same name and shape, context, target, projections and the
            # reconstruction head; a resized predictor starts fresh. (Before 11-06 this path copied nothing: the keys
            # carry a "context." prefix, so J3, J3b, J4 and the Phase 1 runs all trained from random weights.)
            own = self.state_dict()
            picked = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
            missing = [k for k in own if k not in picked]
            self.load_state_dict(picked, strict=False)
            print(f"warm start from the JEPA: {len(picked)} tensors copied, {len(missing)} left as initialised "
                  f"({', '.join(missing[:6])}{'...' if len(missing) > 6 else ''})", flush=True)
            if len(picked) == 0:
                raise ValueError("--jepa-init copied nothing")
            return
        own = self.context.state_dict()
        picked = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape and not k.startswith("head.")}
        missing = [k for k in own if k not in picked]
        self.context.load_state_dict(picked, strict=False)
        self.target.load_state_dict(self.context.state_dict())
        print(f"warm start from the autoencoder: {len(picked)} tensors copied, {len(missing)} left as initialised ({', '.join(missing[:6])}{'...' if len(missing) > 6 else ''})", flush=True)
        if len(picked) == 0:
            raise ValueError("--jepa-init copied nothing")

    def _decode(self, x_enc, v_pos, v_pad, q_pos, q_pad, f=None):
        """The predictor: encoder outputs (CLS included) plus a mask token at each query position, with the
        phase at frequency ``f`` as one more rotary axis when the phase rope is on. Returns the query outputs."""
        enc = self.context
        b, k = x_enc.shape[0], q_pos.shape[2]
        x = enc.encoder_to_decoder(x_enc)
        x = torch.cat([x, enc.mask_token.expand(b, k, -1).to(x.dtype)], dim=1)
        pos = torch.cat([v_pos, q_pos], dim=2)
        pad = torch.cat([v_pad, q_pad], dim=1)
        if self.phase_rope:
            t_days = (pos[:, 0].float() - 1.0) * enc.spectral.time_scale
            pos = torch.cat([pos, (f[:, None] * t_days)[:, None].to(pos.dtype)], 1)
        x = enc.decoder(x, enc.decoder_rope.prepare(pos), attention_mask(pad))
        return x[:, -k:]

    def momentum(self, step: int, total: int) -> float:
        """Linear schedule from ``ema`` to ``ema_end`` over the run."""
        f = min(max(step / max(total, 1), 0.0), 1.0)
        return self.ema + (self.ema_end - self.ema) * f

    @torch.no_grad()
    def ema_update(self, step: int, total: int) -> float:
        if self.no_ema:
            return 0.0
        m = self.momentum(step, total)
        for pt, pc in zip(self.target.parameters(), self.context.parameters()):
            pt.mul_(m).add_(pc.detach(), alpha=1.0 - m)
        for bt, bc in zip(self.target.buffers(), self.context.buffers()):
            bt.copy_(bc)
        if self.proj is not None:
            for pt, pc in zip(self.proj_target.parameters(), self.proj.parameters()):
                pt.mul_(m).add_(pc.detach(), alpha=1.0 - m)
        return m

    def forward(self, values, positions, pad_mask=None, mask=None) -> MAEOutput:
        enc = self.context
        b, n, _ = values.shape
        if pad_mask is None:
            pad_mask = torch.zeros(b, n, dtype=torch.bool, device=values.device)
        if mask is None:
            mask = gen_mask(enc.mask_ratio, pad_mask)
        pos_t = positions.transpose(1, 2)

        def split(t, m):
            return t[m].reshape(b, -1, *t.shape[2:])

        # the targets: the EMA encoder's token latents of the whole window at the hidden positions
        with torch.no_grad():
            tg = self.context if self.no_ema else self.target
            xf = tg.embed(values, positions)
            xf, pf, pdf = tg.add_cls(xf, positions, pad_mask)
            xf = tg.run_transformer(xf, pf, pdf, values)
            toks = xf[:, 1:] if tg.use_cls else xf
            target = split(toks.float(), mask)
            ptg = self.proj if self.no_ema else self.proj_target
            if ptg is not None:
                target = ptg(target)
            if not self.no_ema:
                target = F.layer_norm(target, (target.shape[-1],))
        if target.shape[1] == 0:
            raise ValueError("mask selects no tokens; nothing to predict")
        m_pos, m_pad = split(pos_t, mask).transpose(1, 2), split(pad_mask, mask)
        v_pos, v_pad = split(pos_t, ~mask).transpose(1, 2), split(pad_mask, ~mask)
        v_values = split(values, ~mask)
        x = enc.embed(v_values, v_pos)
        x, v_pos, v_pad = enc.add_cls(x, v_pos, v_pad)
        x = enc.run_transformer(x, v_pos, v_pad, v_values)
        enc_tokens = x
        x = enc.encoder_to_decoder(x)
        k = target.shape[1]
        x = torch.cat([x, enc.mask_token.expand(b, k, -1).to(x.dtype)], dim=1)
        pos = torch.cat([v_pos, m_pos], dim=2)
        pad = torch.cat([v_pad, m_pad], dim=1)
        if self.phase_rope:
            # as RoMAE's decoder does with phase_rope: the phase in cycles at the context encoder's own frequency
            f = spectral_frequency(enc.spectral, None, self.phase_freq)
            t_days = (pos[:, 0].float() - 1.0) * enc.spectral.time_scale
            pos = torch.cat([pos, (f[:, None] * t_days)[:, None].to(pos.dtype)], 1)
        x = enc.decoder(x, enc.decoder_rope.prepare(pos), attention_mask(pad))
        hid = x[:, -k:]
        pred = enc.head(hid).float()
        if self.proj is not None:
            pred = self.proj(pred) if self.no_ema else F.layer_norm(self.proj(pred), (self.proj_dim,))
        real = (~m_pad).float()[..., None]
        err = F.smooth_l1_loss(pred, target, reduction="none") if self.loss_kind == "smoothl1" else F.mse_loss(pred, target, reduction="none")
        loss = (err * real).sum() / (real.sum() * target.shape[-1]).clamp_min(1e-8)
        if self.recon_head is not None:
            values_hidden = split(values[..., :1], mask).float()
            recon = self.recon_head(hid).float()
            loss_recon = (F.mse_loss(recon, values_hidden, reduction="none") * real).sum() / real.sum().clamp_min(1e-8)
            loss = loss + self.recon_weight * loss_recon
            self.last_stats_recon = float(loss_recon.detach())
            self.last_recon = recon.detach()  # the magnitude at the hidden spots, for project.eval.jepa_forecast_probe
        with torch.no_grad():  # collapse watch: the spread of the targets and the predictions across tokens
            self.last_stats = dict(target_std=float(target[real[..., 0] > 0].std(0).mean()), pred_std=float(pred[real[..., 0] > 0].std(0).mean()))
            if self.recon_head is not None:
                self.last_stats["recon"] = self.last_stats_recon
        return MAEOutput(loss=loss, pred=pred, target=target, mask=mask, enc_tokens=enc_tokens, enc_positions=v_pos, enc_pad=v_pad)


def forward_views(model: TokenJEPA, values, positions, pad, periods=None, n_glob=2, n_loc=4, keep=0.15,
                  local_span=(30.0, 80.0), band_drop=0.2, window=250.0, sig_max_n=4096):
    """The no-EMA multi-view loss (LeJEPA at the token level). Returns ``(pred, sigreg, spectral_aux, stats)``;
    the caller weights them. ``periods [b]`` (training-time catalogue period) feeds the spectral aux loss on the
    global views; the phase rope uses the encoder's own frequency from the first global view."""
    enc = model.context
    b, n = pad.shape
    ts = enc.spectral.time_scale if enc.spectral is not None else 1.0
    t_days = (positions[:, 0].float() - 1.0) * ts
    keeps = make_views(t_days, positions, pad, n_glob, n_loc, keep, local_span, band_drop, window)
    V = n_glob + n_loc

    def encode(kv):  # kv [m, b, n] -> per view-row packed outputs
        m = kv.shape[0]
        kk = kv.reshape(m * b, n)
        order, vpad = pack(kk)
        vals, pos = take(values.repeat(m, 1, 1), positions.repeat(m, 1, 1), order)
        x = enc.embed(vals, pos)
        x, pos_c, pad_c = enc.add_cls(x, pos, vpad)
        x = enc.run_transformer(x, pos_c, pad_c, vals)
        return x, pos_c, pad_c, order, vpad

    xg, pg, padg, og, vpg = encode(keeps[:n_glob])
    aux = None
    if periods is not None and enc.spectral is not None:
        aux = spectral_aux_loss(enc.spectral, periods.repeat(n_glob))
    f_g = spectral_frequency(enc.spectral, None, model.phase_freq).view(n_glob, b) if enc.spectral is not None else None
    f = f_g[0] if model.phase_rope else None
    xl, pl, padl, ol, vpl = encode(keeps[n_glob:])
    f_l = spectral_frequency(enc.spectral, None, model.phase_freq).view(n_loc, b) if enc.spectral is not None and n_loc else None

    def hits(fv):  # share of views whose frequency is within 2 % / 10 % of the catalogue's (monitoring only)
        if periods is None or fv is None:
            return {}
        ok = torch.isfinite(periods) & (periods > 0)
        if not ok.any():
            return {}
        rel = (fv[:, ok].float() * periods[ok].float()[None] - 1.0).abs()
        return dict(hit2=float((rel < 0.02).float().mean()), hit10=float((rel < 0.1).float().mean()))
    tok_g = xg[:, 1:] if enc.use_cls else xg
    tok_l = xl[:, 1:] if enc.use_cls else xl
    zg = model.proj(tok_g.float()).float()  # [G b, kg, D]; float32 under autocast too
    zl = model.proj(tok_l.float()).float()
    D = zg.shape[-1]
    # targets in the window layout: the mean of the global views' projected latents
    zsum = zg.new_zeros(b, n, D)
    cnt = zg.new_zeros(b, n)
    for g in range(n_glob):
        rows = slice(g * b, (g + 1) * b)
        w = (~vpg[rows]).float()
        zsum.scatter_add_(1, og[rows][..., None].expand(-1, -1, D), zg[rows] * w[..., None])
        cnt.scatter_add_(1, og[rows], w)
    has_t = cnt > 0
    T = zsum / cnt.clamp(min=1)[..., None]
    qo, qpad = pack(has_t)
    q_pos = positions.gather(2, qo[:, None].expand(-1, positions.shape[1], -1))
    Tq = T.gather(1, qo[..., None].expand(-1, -1, D))
    # predictor from every view to every target position
    preds = []
    for x, pc, padc, m in ((xg, pg, padg, n_glob), (xl, pl, padl, n_loc)):
        hid = model._decode(x, pc, padc, q_pos.repeat(m, 1, 1), qpad.repeat(m, 1), None if f is None else f.repeat(m))
        preds.append(model.proj(enc.head(hid).float()).float())
    pred = torch.cat(preds).view(V, b, -1, D)
    w = (~qpad).float()
    l_pred = (((pred - Tq[None]) ** 2).mean(-1) * w[None]).sum() / (w.sum() * V).clamp(min=1)
    z_all = torch.cat([zg[~vpg], zl[~vpl]])
    l_sig = sigreg(z_all.float(), num_slices=model.sig_slices, max_n=sig_max_n)
    with torch.no_grad():
        stats = dict(view_tokens_global=float((~vpg).sum(1).float().mean()), view_tokens_local=float((~vpl).sum(1).float().mean()), target_share=float(has_t.sum() / (~pad).sum()),
                     z_std=float(z_all.std(0).mean()), pred_std=float(pred[:, w > 0].std(1).mean()))
        stats.update({f"period_{k}_global": v for k, v in hits(f_g).items()})
        stats.update({f"period_{k}_local": v for k, v in hits(f_l).items()})
    return l_pred, l_sig, aux, stats
