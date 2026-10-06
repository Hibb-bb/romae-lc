"""Stage 1, bottleneck variant: an autoencoder whose decoder reads only the
pooled latent.

The masked-pretraining recipe (:class:`~romae_lc.RoMAEForPreTraining`,
``pretrain_mae.py`` without ``--bottleneck``) lets its decoder attend over the
output of every visible token, so nothing forces the pooled CLS feature alone
to hold the window's period, phase and shape; the later stages read only
that feature. :class:`BottleneckAE` closes the gap: the same RoMAE encoder
(CLS pooling, built from ``encoder_config`` / ``rope_layouts`` exactly like
the masked-pretraining model) embeds a window with ``mask_ratio`` of its
points hidden, and a :class:`~project.decoder.QueryDecoder` (``mse`` kind, on
the encoder's collapsed rotary layout) predicts the magnitudes of the query
points from the latent ``z [B, D]`` and the query positions ``(t, band, log
sigma)`` only, the Gaussian likelihood of :func:`project.decoder.decoder_loss`
restricted to the chosen points. ``loss_on="all"`` scores every real point of
the original window (the visible ones are then reconstructed through the
bottleneck too), ``"hidden"`` only the hidden ones (the imputation setting of
stage 3). The query decoder needs the per-point errors, so the token spec's
error channel is required.

The checkpoint written by :func:`bottleneck_state` is a stage-1 ``mae.pt``
with ``kind == "bottleneck"``: :func:`project.common.load_mae` rebuilds it
through :meth:`BottleneckAE.from_checkpoint`, ``load_wm`` / ``load_encoder``
and ``train_wm --init-backbone`` take it like a masked-pretraining one, since
the model exposes ``backbone(pool)``, ``encode``, ``use_cls``, ``rope_layout``
and ``embed_dim`` the same way.

    python -m project.pretrain_mae --bottleneck --out project/runs/bn_w250 --window 250
    model = BottleneckAE(encoder=encoder_config(args), n_channels=spec.n_channels,
                         rope=rope_layouts(args, ladder), err_stats=spec.err_stats,
                         decoder=dict(d_model=192, nhead=3, depth=2), mask_ratio=0.5)
    loss = model(fuse_tokens(batch["frames"])).loss
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn as nn

from romae_lc import RoMAE, Tokens
from romae_lc.model import pool_tokens

from project.common import flat_layout, fuse_frames
from project.decoder import QueryDecoder, drop_tokens, gaussian_var

LOSS_ON = ("all", "hidden")


@dataclass
class BottleneckOutput:
    """Forward result: ``loss`` (Gaussian NLL over the scored points),
    ``z [B, D]`` the pooled latent, ``mu`` / ``logvar [B, N]`` the decoder's
    mean and extra log-variance at every query token, ``scored [B, N]`` the
    points the loss counts and ``hidden [B, N]`` the points the encoder did
    not see (both True = selected)."""

    loss: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    scored: torch.Tensor
    hidden: torch.Tensor


def fuse_tokens(frames: Sequence[Tokens]) -> Tokens:
    """The ``T`` token batches of a frame sequence as one :class:`Tokens`
    (:func:`project.common.fuse_frames`, which drops ``extras``, plus the
    per-point errors padded the same way: 0 on padding)."""
    values, positions, pad = fuse_frames(frames)
    if any(f.extras is None for f in frames):
        raise ValueError("frames carry no extras (per-point sigma); need the error channel")
    b, n = frames[0].values.shape[0], values.shape[1]
    extras = frames[0].extras.new_zeros(b * len(frames), n)
    for i, f in enumerate(frames):
        extras[i * b : (i + 1) * b, : f.extras.shape[1]] = f.extras
    return Tokens(values, positions, pad, extras)


def masked_decoder_loss(
    dec: QueryDecoder,
    z: torch.Tensor,
    tokens: Tokens,
    scored: torch.Tensor,
    learned_var: bool | str = True,
):
    """:func:`project.decoder.decoder_loss` of the ``mse`` kind averaged over
    ``scored [B, N]`` instead of every real token; returns ``(loss, mu,
    logvar)``. The decoder sees only positions and errors of the query
    tokens, never their magnitudes, so scoring visible points leaks nothing.

    ``learned_var=False`` scores the points under the known error alone,
    ``var = sigma^2``: with a learned extra variance the decoder can explain
    a star's oscillation as scatter (a wide variance around the mean level)
    instead of predicting it, which is what the first bottleneck runs did
    (recon loss far below zero, period probe at the hand-feature baseline).
    Under the known error only, the oscillation has to be predicted, which
    needs the period and phase in the latent. ``learned_var="unit"`` is the
    plain squared error (variance 1 everywhere, the loss of the masked
    pretraining that is known to learn period): the errors then weight
    nothing, they only reach the model as an input channel."""
    if dec.kind != "mse":
        raise ValueError("the bottleneck decoder is the mse kind")
    m = tokens.values[..., 0].float()
    mu, logvar = dec(z, tokens.positions, tokens.pad_mask, dec.log_sigma(tokens))
    if learned_var == "unit":
        var = torch.ones_like(m)
    elif learned_var:
        var = gaussian_var(tokens.extras, logvar)
    else:
        var = tokens.extras.float().square().clamp_min(1e-8)
    nll = 0.5 * ((m - mu.float()).square() / var + var.log())
    scored = scored & ~tokens.pad_mask
    return (nll * scored).sum() / scored.sum().clamp(min=1), mu, logvar


class BottleneckAE(nn.Module):
    """A RoMAE encoder (CLS pooling) plus a :class:`~project.decoder.QueryDecoder`
    that reconstructs a window's magnitudes from the pooled latent alone.

    Args:
        encoder: The RoMAE encoder config (dict of
            :class:`~romae_lc.transformer.TransformerConfig` fields, what
            ``project.common.encoder_config`` gives).
        n_channels: Value channels per token (2 with the error channel).
        err_stats: ``(mu, sd)`` of ``log sigma`` (``TokenSpec.err_stats``),
            the query decoder's standardisation of the per-point errors.
        n_axes: Position axes (time, wavelength).
        rope: The rotary layout or per-layer layouts (``rope_layouts``).
        decoder: The query decoder's size, ``dict(d_model, nhead, depth)``
            (an ``attention`` entry is recorded but the query decoder is
            softmax only); ``d_model / nhead`` must equal the encoder's head
            dimension because the decoder rotates by the encoder's ladder.
        mask_ratio: Fraction of the real points hidden from the encoder in
            every window, in ``[0, 1]`` (at least 2 points stay visible).
        loss_on: ``"all"`` scores every real point of the window,
            ``"hidden"`` only the hidden ones.
        use_cls: Must stay True (the latent is the CLS token).
    """

    def __init__(
        self,
        encoder: dict,
        n_channels: int,
        err_stats: Sequence[float],
        n_axes: int = 2,
        rope="axial",
        decoder: dict | None = None,
        mask_ratio: float = 0.5,
        loss_on: str = "all",
        use_cls: bool = True,
        denoise: bool = False,
        learned_var: bool | str = True,
        abs_timescales=None,
    ):
        super().__init__()
        if not use_cls:
            raise ValueError("BottleneckAE pools the CLS token: use_cls must be True")
        if not 0 <= mask_ratio <= 1:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
        if loss_on not in LOSS_ON:
            raise ValueError(f"loss_on must be one of {LOSS_ON}, got {loss_on!r}")
        self.mask_ratio, self.loss_on = float(mask_ratio), loss_on
        if learned_var not in (True, False, "unit"):
            raise ValueError(f"learned_var must be True, False or 'unit', got {learned_var!r}")
        self.denoise, self.learned_var = bool(denoise), learned_var
        self.encoder = RoMAE(
            pool="cls", encoder=encoder, n_channels=n_channels, n_axes=n_axes, rope=rope,
            abs_timescales=abs_timescales,
        )  # fmt: skip
        self.dec_cfg = dict(d_model=192, nhead=3, depth=2, attention="softmax")
        self.dec_cfg.update(decoder or {})
        d_model, nhead = self.dec_cfg["d_model"], self.dec_cfg["nhead"]
        if d_model % nhead or d_model // nhead != self.encoder.cfg.head_dim:
            raise ValueError(
                f"decoder d_model / nhead ({d_model} / {nhead}) must equal the "
                f"encoder head_dim {self.encoder.cfg.head_dim} (same width / heads "
                "ratio), since the query decoder rotates by the encoder's ladder"
            )
        self.decoder = QueryDecoder(
            self.encoder.embed_dim,
            flat_layout(self.encoder),
            tuple(float(v) for v in err_stats),
            kind="mse",
            d_model=d_model,
            nhead=nhead,
            depth=self.dec_cfg["depth"],
        )

    # ---- the encoder's face, so PooledEncoder / pretrain_mae / load_wm work

    @property
    def cfg(self):
        return self.encoder.cfg

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim

    @property
    def use_cls(self) -> bool:
        return self.encoder.use_cls

    @property
    def cls(self):
        return self.encoder.cls

    @property
    def rope(self):
        return self.encoder.rope

    @property
    def rope_layers(self):
        return self.encoder.rope_layers

    @property
    def rope_layout(self):
        return self.encoder.rope_layout

    @property
    def per_layer_rope(self) -> bool:
        return self.encoder.per_layer_rope

    @property
    def transformer(self):
        return self.encoder.transformer

    @property
    def projection(self):
        return self.encoder.projection

    @property
    def hparams(self) -> dict:
        """The encoder's constructor arguments (what ``mae_state`` stores
        under ``backbone``); :attr:`bottleneck_hparams` holds the rest."""
        return self.encoder.hparams

    @property
    def bottleneck_hparams(self) -> dict:
        return dict(
            decoder=dict(self.dec_cfg),
            mask_ratio=self.mask_ratio,
            loss_on=self.loss_on,
            denoise=self.denoise,
            learned_var=self.learned_var,
            err_stats=self.decoder.err_stats.tolist(),
        )

    def encode(self, values, positions, pad_mask=None):
        """``(tokens [B, 1 + N, D], pad_mask)`` of the encoder, CLS first."""
        return self.encoder.encode(values, positions, pad_mask)

    def backbone(self, pool: str = "cls") -> RoMAE:
        """A :class:`~romae_lc.RoMAE` encoder initialised from the trained
        weights (what ``train_wm --init-backbone`` and ``load_wm`` take)."""
        model = RoMAE(pool=pool, **self.encoder.hparams)
        model.load_state_dict(self.encoder.state_dict())
        return model

    def latent(self, tokens: Tokens) -> torch.Tensor:
        """The pooled latent ``z [B, D]`` of a token batch."""
        x, pad = self.encode(*tokens)
        return pool_tokens(x, pad, "cls", True)

    def forward(self, tokens: Tokens, generator=None) -> BottleneckOutput:
        """Hide ``mask_ratio`` of every window's real points, encode the
        rest to ``z``, decode the magnitudes of the query points (every real
        point, or the hidden ones with ``loss_on="hidden"``)."""
        if tokens.extras is None:
            raise ValueError("tokens carry no extras (per-point sigma)")
        visible = drop_tokens(tokens, self.mask_ratio, generator, keep_min=2)
        hidden = visible.pad_mask & ~tokens.pad_mask
        if self.denoise and self.training:
            # denoising: the encoder sees magnitudes redrawn from N(m, sigma),
            # the decoder is still scored on the observed ones under sigma
            values = visible.values.clone()
            noise = torch.randn(
                values.shape[:-1], device=values.device, generator=generator
            )
            values[..., 0] = values[..., 0] + visible.extras.to(values.dtype) * noise.to(
                values.dtype
            ) * (~visible.pad_mask).to(values.dtype)
            visible = Tokens(values, visible.positions, visible.pad_mask, visible.extras)
        z = self.latent(visible)
        scored = hidden if self.loss_on == "hidden" else ~tokens.pad_mask
        loss, mu, logvar = masked_decoder_loss(
            self.decoder, z, tokens, scored, learned_var=self.learned_var
        )
        return BottleneckOutput(loss, z, mu, logvar, scored, hidden)

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "BottleneckAE":
        """Rebuild from ``ckpt["backbone"]`` (the encoder hparams incl. the
        encoder config) and ``ckpt["bottleneck"]``, then load the weights."""
        model = cls(**ckpt["backbone"], **ckpt["bottleneck"])
        model.load_state_dict(ckpt["state_dict"])
        return model


def bottleneck_state(
    model: BottleneckAE, spec, cfg, ladder, classes, args, step, metrics=None
) -> dict:
    """The stage-1 checkpoint of a bottleneck autoencoder: the dict of
    :func:`project.common.mae_state` with ``kind="bottleneck"`` and the key
    ``bottleneck`` (decoder size, mask ratio, loss_on, err_stats) in place of
    ``mae``; :func:`project.common.load_mae` dispatches on ``kind``."""
    return dict(
        kind="bottleneck",
        step=step,
        args=dict(vars(args)) if isinstance(args, argparse.Namespace) else dict(args),
        state_dict=model.state_dict(),
        backbone=dict(model.hparams, encoder=asdict(model.cfg)),
        bottleneck=model.bottleneck_hparams,
        spec=spec.to_dict(),
        frames=asdict(cfg),
        ladder=ladder.to_dict(),
        classes=list(classes),
        metrics=metrics,
    )
