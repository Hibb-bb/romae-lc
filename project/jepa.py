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

The checkpoint has ``kind == "jepa"``; :func:`project.common.load_mae`
rebuilds it and every loader and probe treats it like ``mae.pt``. By default
the TARGET encoder is the one used downstream (``use_target``), as in
I-JEPA; ``encode`` / ``backbone`` follow that choice.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from romae_lc.model import MAEOutput, RoMAEForPreTraining, attention_mask, gen_mask


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
    ):
        super().__init__()
        d = encoder["d_model"] if isinstance(encoder, dict) else encoder.d_model
        if target_channels is not None and target_channels != d:
            raise ValueError(f"the JEPA head outputs the encoder width {d}, not {target_channels}")
        self.context = RoMAEForPreTraining(decoder=decoder, mask_ratio=mask_ratio, target_channels=d, encoder=encoder, n_channels=n_channels,
                                           n_axes=n_axes, rope=rope, use_cls=use_cls, abs_timescales=abs_timescales, spectral=spectral)  # fmt: skip
        self.target = copy.deepcopy(self.context)
        self.target.requires_grad_(False)
        self.ema, self.ema_end, self.loss_kind, self.use_target = float(ema), float(ema_end), str(loss), bool(use_target)
        self.recon_weight = float(recon_weight)
        # the hybrid loss: a second head on the predictor's outputs reconstructs the hidden brightness, weighted by recon_weight
        self.recon_head = nn.Linear(self.context.dec_cfg.d_model, 1) if self.recon_weight > 0 else None
        self.last_stats: dict = {}

    # ---- what the loaders, probes and the pretraining script read
    @property
    def jepa_hparams(self) -> dict:
        return dict(ema=self.ema, ema_end=self.ema_end, loss=self.loss_kind, use_target=self.use_target, recon_weight=self.recon_weight)

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
        own = self.context.state_dict()
        picked = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape and not k.startswith("head.")}
        missing = [k for k in own if k not in picked]
        self.context.load_state_dict(picked, strict=False)
        self.target.load_state_dict(self.context.state_dict())
        print(f"warm start from the autoencoder: {len(picked)} tensors copied, {len(missing)} left as initialised ({', '.join(missing[:6])}{'...' if len(missing) > 6 else ''})", flush=True)

    def momentum(self, step: int, total: int) -> float:
        """Linear schedule from ``ema`` to ``ema_end`` over the run."""
        f = min(max(step / max(total, 1), 0.0), 1.0)
        return self.ema + (self.ema_end - self.ema) * f

    @torch.no_grad()
    def ema_update(self, step: int, total: int) -> float:
        m = self.momentum(step, total)
        for pt, pc in zip(self.target.parameters(), self.context.parameters()):
            pt.mul_(m).add_(pc.detach(), alpha=1.0 - m)
        for bt, bc in zip(self.target.buffers(), self.context.buffers()):
            bt.copy_(bc)
        return m

    def forward(self, values, positions, pad_mask=None, mask=None, weight=None) -> MAEOutput:
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
            tg = self.target
            xf = tg.embed(values, positions)
            xf, pf, pdf = tg.add_cls(xf, positions, pad_mask)
            xf = tg.run_transformer(xf, pf, pdf, values)
            toks = xf[:, 1:] if tg.use_cls else xf
            target = F.layer_norm(split(toks.float(), mask), (toks.shape[-1],))
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
        x = enc.decoder(x, enc.decoder_rope.prepare(pos), attention_mask(pad))
        hid = x[:, -k:]
        pred = enc.head(hid).float()
        real = (~m_pad).float()[..., None]
        err = F.smooth_l1_loss(pred, target, reduction="none") if self.loss_kind == "smoothl1" else F.mse_loss(pred, target, reduction="none")
        loss = (err * real).sum() / (real.sum() * target.shape[-1]).clamp_min(1e-8)
        if self.recon_head is not None:
            values_hidden = split(values[..., :1], mask).float()
            recon = self.recon_head(hid).float()
            loss_recon = (F.mse_loss(recon, values_hidden, reduction="none") * real).sum() / real.sum().clamp_min(1e-8)
            loss = loss + self.recon_weight * loss_recon
            self.last_stats_recon = float(loss_recon)
        with torch.no_grad():  # collapse watch: the spread of the targets and the predictions across tokens
            self.last_stats = dict(target_std=float(target[real[..., 0] > 0].std(0).mean()), pred_std=float(pred[real[..., 0] > 0].std(0).mean()))
            if self.recon_head is not None:
                self.last_stats["recon"] = self.last_stats_recon
        return MAEOutput(loss=loss, pred=pred, target=target, mask=mask, enc_tokens=enc_tokens, enc_positions=v_pos, enc_pad=v_pad)


def jepa_state(model: TokenJEPA, spec, cfg, ladder, classes, args, step, metrics=None) -> dict:
    """The stage-1 checkpoint of a token JEPA: ``mae_state``'s keys with
    ``kind="jepa"`` and the key ``jepa`` (EMA schedule, loss, which encoder
    is used downstream); :func:`project.common.load_mae` dispatches on ``kind``."""
    return dict(
        kind="jepa",
        step=step,
        args=dict(vars(args)) if isinstance(args, argparse.Namespace) else dict(args),
        state_dict=model.state_dict(),
        backbone=dict(model.hparams, encoder=asdict(model.cfg)),
        mae=dict(decoder=asdict(model.dec_cfg), mask_ratio=model.mask_ratio, target_channels=model.target_channels),
        jepa=model.jepa_hparams,
        spec=spec.to_dict(),
        frames=asdict(cfg),
        ladder=ladder.to_dict(),
        classes=list(classes),
        metrics=metrics,
    )
