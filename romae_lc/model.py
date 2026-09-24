"""RoMAE models for light curves (Zivanovic et al. 2025, arXiv:2505.20535).

A light curve is a set of observations; each observation is one token with a
value vector ``[C]`` (flux, optionally per-band descriptors) and a continuous
position vector ``[n_axes]`` (time, then wavelength coordinates), see
:mod:`romae_lc.tokenize`. Batches are padded to ``N`` tokens with a boolean
``pad_mask`` (True = padding).

* :class:`RoMAE` - encoder-only backbone returning a pooled embedding; the
  model to pretrain with :class:`~romae_lc.lejepa.LeJEPA` and to probe.
* :class:`RoMAEForPreTraining` - the paper's masked-autoencoding recipe:
  the encoder sees the visible tokens, a light decoder reconstructs the
  values of the masked tokens from MASK tokens at their positions.
* :class:`RoMAEForClassification` - encoder plus a linear head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import rope as rope_lib
from .rope import BlockRope
from .transformer import Transformer, TransformerConfig, attention_mask, config


def gen_mask(
    mask_ratio: float,
    pad_mask: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random token mask for masked pretraining; True = masked.

    Each row masks ``ceil(n_real * mask_ratio)`` uniformly chosen real tokens.
    Rows are then equalised to the batch maximum by additionally masking
    trailing (padding) tokens, so boolean-index reshapes stay rectangular; the
    reconstruction loss ignores those padding targets.

    Args:
        mask_ratio: Fraction of real tokens to mask, in [0, 1].
        pad_mask: Bool ``[B, N]``, True where the token is padding (real
            tokens come first in every row).
        generator: Optional CPU generator for reproducible masks.
    """
    if not 0 <= mask_ratio <= 1:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
    b, n = pad_mask.shape
    n_real = (~pad_mask).sum(1)
    k = (n_real * mask_ratio).ceil().long()
    scores = torch.rand(b, n, generator=generator).to(pad_mask.device)
    scores[pad_mask] = 2.0  # padding sorts last
    rank = scores.argsort(1).argsort(1)
    mask = rank < k[:, None]
    extra = k.max() - k  # equalise with trailing tokens
    idx = torch.arange(n, device=pad_mask.device)[None, :]
    return mask | (idx >= (n - extra)[:, None])


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.RMSNorm):
        nn.init.ones_(m.weight)


class RoMAEBase(nn.Module):
    """Layers shared by all RoMAE models: value projection, CLS token,
    rotary layout and the transformer encoder.

    Args:
        encoder: Size name (see :data:`~romae_lc.transformer.SIZES`), dict of
            :class:`~romae_lc.transformer.TransformerConfig` fields, or a
            config.
        n_channels: Value channels per token.
        n_axes: Position axes per token (rows of ``positions``).
        rope: ``"axial"`` (equal p-RoPE split over the axes), ``"simplex"``
            (time axial + nD-RoPE simplex over the other axes), or an
            explicit layout list (see :class:`~romae_lc.rope.BlockRope`).
        rope_base: Rotary base of the time axis (axis 0) for the string
            layouts. Together with the time unit of the positions it sets the
            band of resolvable timescales, see :mod:`romae_lc.analysis`.
        p_rope: p-RoPE fraction for the string layouts.
        rope_timescales: Explicit time ladder for the string layouts, in
            position units and any spacing (overrides ``rope_base`` and the
            time block's ``p_rope``), e.g. ``TimeEncodingReport.timescales``.
        use_cls: Prepend a learned CLS token at position 0.
    """

    def __init__(
        self,
        encoder: str | dict | TransformerConfig = "small",
        n_channels: int = 1,
        n_axes: int = 2,
        rope: str | list[dict] = "axial",
        rope_base: float = 10000.0,
        p_rope: float = 0.75,
        use_cls: bool = True,
        rope_timescales=None,
    ):
        super().__init__()
        self.cfg = config(encoder)
        self.n_channels, self.n_axes, self.use_cls = n_channels, n_axes, use_cls
        if isinstance(rope, str):
            frac = None if rope == "axial" else 0.5
            rope = rope_lib.layout(
                self.cfg.head_dim,
                n_axes,
                rope,
                time_frac=frac,
                time_base=rope_base,
                p=p_rope,
                time_timescales=rope_timescales,
            )
        elif rope_timescales is not None:
            raise ValueError("rope_timescales only applies to the string layouts")
        self.rope = BlockRope(self.cfg.head_dim, self.cfg.nhead, rope)
        if self.rope.n_axes > n_axes:
            raise ValueError(
                f"rope layout uses {self.rope.n_axes} axes, n_axes={n_axes}"
            )
        self.projection = nn.Linear(n_channels, self.cfg.d_model)
        self.transformer = Transformer(self.cfg)
        self.cls = nn.Parameter(torch.zeros(self.cfg.d_model)) if use_cls else None
        self.apply(_init_weights)
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)

    @property
    def embed_dim(self) -> int:
        return self.cfg.d_model

    @property
    def hparams(self) -> dict:
        """Constructor arguments that rebuild this backbone."""
        return dict(
            encoder=self.cfg,
            n_channels=self.n_channels,
            n_axes=self.n_axes,
            rope=self.rope.layout,
            use_cls=self.use_cls,
        )

    def add_cls(self, x, positions, pad_mask):
        """Prepend the CLS token (position 0, never padding) if enabled."""
        if self.cls is None:
            return x, positions, pad_mask
        b = x.shape[0]
        x = torch.cat([self.cls.expand(b, 1, -1).to(x.dtype), x], dim=1)
        positions = torch.cat(
            [positions.new_zeros(b, positions.shape[1], 1), positions], 2
        )
        if pad_mask is not None:
            pad_mask = torch.cat([pad_mask.new_zeros(b, 1), pad_mask], dim=1)
        return x, positions, pad_mask

    def encode(self, values, positions, pad_mask=None):
        """Project, add CLS, run the encoder. Returns ``(tokens, pad_mask)``
        with ``tokens [B, 1 + N, d_model]`` (CLS first if enabled). Without a
        CLS token a batch with no tokens at all (``N == 0``) raises
        ``ValueError``."""
        x, positions, pad_mask = self.add_cls(
            self.projection(values), positions, pad_mask
        )
        if x.shape[1] == 0:
            raise ValueError(
                "empty token batch: with use_cls=False the encoder needs at least "
                "one token in some row (an all-empty frame window; see FrameConfig)"
            )
        x = self.transformer(x, self.rope.prepare(positions), attention_mask(pad_mask))
        return x, pad_mask


def _check_pool(pool: str, use_cls: bool) -> None:
    if pool not in ("cls", "mean"):
        raise ValueError(f"pool must be 'cls' or 'mean', got {pool!r}")
    if pool == "cls" and not use_cls:
        raise ValueError("pool='cls' requires use_cls=True")


def pool_tokens(
    x: torch.Tensor, pad_mask: torch.Tensor | None, pool: str, has_cls: bool
):
    """``"cls"`` returns token 0; ``"mean"`` averages the real non-CLS tokens."""
    if pool == "cls":
        return x[:, 0]
    tokens = x[:, 1:] if has_cls else x
    if pad_mask is None:
        return tokens.mean(1)
    real = (~pad_mask[:, 1:] if has_cls else ~pad_mask).to(x.dtype)
    return (tokens * real[..., None]).sum(1) / real.sum(1, keepdim=True).clamp(min=1)


class RoMAE(RoMAEBase):
    """RoMAE encoder as a pooled feature extractor ``[B, d_model]``.

    Args:
        pool: ``"cls"`` (needs ``use_cls``) or masked ``"mean"`` over tokens.
        **kwargs: See :class:`RoMAEBase`.
    """

    def __init__(self, pool: str = "cls", **kwargs):
        super().__init__(**kwargs)
        _check_pool(pool, self.use_cls)
        self.pool = pool

    def forward(self, values, positions, pad_mask=None, return_tokens=False):
        """Encode a token batch.

        Args:
            values: ``[B, N, C]`` token values.
            positions: ``[B, n_axes, N]`` continuous positions.
            pad_mask: Optional bool ``[B, N]``, True = padding.
            return_tokens: Also return the per-token outputs ``[B, N, D]``
                (CLS dropped; padding rows are undefined).

        Returns:
            ``[B, D]`` pooled embedding, or ``(pooled, tokens)``.
        """
        x, pad = self.encode(values, positions, pad_mask)
        out = pool_tokens(x, pad, self.pool, self.use_cls)
        if return_tokens:
            return out, x[:, 1:] if self.use_cls else x
        return out


class RoMAEForClassification(RoMAEBase):
    """RoMAE encoder with a dropout + RMSNorm + linear head on the pooled token.

    Args:
        n_classes: Number of output logits.
        pool: ``"cls"`` (needs ``use_cls``) or masked ``"mean"`` over tokens.
        head_dropout: Dropout before the head norm.
        **kwargs: See :class:`RoMAEBase`.
    """

    def __init__(
        self, n_classes: int, pool: str = "cls", head_dropout: float = 0.0, **kwargs
    ):
        super().__init__(**kwargs)
        _check_pool(pool, self.use_cls)
        self.pool = pool
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.RMSNorm(self.cfg.d_model, eps=self.cfg.norm_eps),
            nn.Linear(self.cfg.d_model, n_classes),
        )
        self.head.apply(_init_weights)

    def forward(self, values, positions, pad_mask=None) -> torch.Tensor:
        """Logits ``[B, n_classes]``."""
        x, pad = self.encode(values, positions, pad_mask)
        return self.head(pool_tokens(x, pad, self.pool, self.use_cls))


@dataclass
class MAEOutput:
    """Masked-autoencoding forward result.

    Attributes:
        loss: Mean squared error over the real masked tokens.
        pred: Reconstructions ``[B, K, target_channels]`` of the masked tokens.
        target: Their true values, same shape.
        mask: The token mask ``[B, N]`` that was used (True = masked).
    """

    loss: torch.Tensor
    pred: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor


class RoMAEForPreTraining(RoMAEBase):
    """Masked autoencoding with a light decoder (the RoMAE pretraining recipe).

    The encoder sees only the visible tokens (plus CLS). Their outputs are
    projected to the decoder width and concatenated with a learned MASK token
    for every masked position; the decoder attends over all of them with the
    same continuous rotary encoding, and a linear head predicts the masked
    values. The paper masks 50% of the tokens for time series and uses the
    ``tiny-shallow`` decoder.

    Args:
        decoder: Decoder size name, dict or config.
        mask_ratio: Fraction of real tokens to mask when no mask is given,
            in (0, 1]; there must be something to reconstruct.
        target_channels: Leading value channels to reconstruct (1 = flux
            only, so appended band descriptors are never a target).
        **kwargs: See :class:`RoMAEBase`.
    """

    def __init__(
        self,
        decoder: str | dict | TransformerConfig = "tiny-shallow",
        mask_ratio: float = 0.5,
        target_channels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0 < mask_ratio <= 1:
            raise ValueError(f"mask_ratio must be in (0, 1], got {mask_ratio}")
        self.mask_ratio, self.target_channels = mask_ratio, target_channels
        self.dec_cfg = config(decoder)
        self.decoder = Transformer(self.dec_cfg)
        self.decoder_rope = BlockRope(
            self.dec_cfg.head_dim, self.dec_cfg.nhead, self._decoder_layout()
        )
        self.encoder_to_decoder = nn.Linear(self.cfg.d_model, self.dec_cfg.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dec_cfg.d_model))
        self.head = nn.Sequential(
            nn.RMSNorm(self.dec_cfg.d_model, eps=self.dec_cfg.norm_eps),
            nn.Linear(self.dec_cfg.d_model, target_channels),
        )
        for m in (self.decoder, self.encoder_to_decoder, self.head):
            m.apply(_init_weights)
        # Like the CLS token (and the MAE recipe): an exactly-zero MASK token
        # sits on the RMSNorm gradient singularity and is position-blind in
        # the first decoder layer.
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _decoder_layout(self) -> list[dict]:
        """The encoder's layout rescaled to the decoder head dimension.

        Every block after the first is rounded to its own channel unit (2 for
        axial, ``2 * (n_axes + 1)`` for a multi-axis simplex) and floored at
        the smallest size that keeps one active angle at its ``p``
        (:func:`~romae_lc.rope.min_axial_dim`,
        :func:`~romae_lc.rope.min_simplex_scales`); the first (time) block
        takes whatever is left. Explicit ladders (``timescales`` of an axial
        block, ``scales`` of a simplex block) are resampled in log space to
        the decoder block's number of active angles, ends kept
        (:func:`~romae_lc.rope.resample_ladder`).
        """
        enc, dec = self.cfg.head_dim, self.dec_cfg.head_dim
        blocks = [dict(spec) for spec in self.rope.layout]
        for spec in blocks[1:]:
            n, p = len(spec["axes"]), spec.get("p", 1.0)
            if spec["kind"] == "simplex" and n > 1:
                unit = 2 * (n + 1)
                floor = unit * rope_lib.min_simplex_scales(p)
            else:
                unit, floor = 2, rope_lib.min_axial_dim(p)
            spec["dim"] = max(floor, int(round(spec["dim"] * dec / enc / unit)) * unit)
        blocks[0]["dim"] = dec - sum(b["dim"] for b in blocks[1:])
        d_min = rope_lib.min_axial_dim(blocks[0].get("p", 1.0))
        if blocks[0]["dim"] < d_min or blocks[0]["dim"] % 2:
            raise ValueError(
                f"decoder head_dim {dec} too small for the rope layout at "
                f"p={blocks[0].get('p', 1.0)}"
            )
        for old, spec in zip(self.rope.layout, blocks):
            for key, unit in (("timescales", 2), ("scales", None)):
                if spec.get(key) is None:
                    continue
                if unit is None:  # simplex: 2 (n + 1) channels per scale
                    n = len(spec["axes"])
                    unit = 2 * (n + 1) if n > 1 else 2
                slots_enc, slots_dec = old["dim"] // unit, spec["dim"] // unit
                n_new = max(1, int(round(len(spec[key]) * slots_dec / slots_enc)))
                spec[key] = rope_lib.resample_ladder(spec[key], min(n_new, slots_dec))
        return blocks

    def forward(self, values, positions, pad_mask=None, mask=None) -> MAEOutput:
        """Mask, encode the visible tokens, decode the masked ones.

        Args:
            values: ``[B, N, C]`` token values.
            positions: ``[B, n_axes, N]`` continuous positions.
            pad_mask: Optional bool ``[B, N]``, True = padding.
            mask: Optional bool ``[B, N]`` token mask with the same number of
                True entries per row, at least one (see :func:`gen_mask`);
                sampled from ``mask_ratio`` when omitted.
        """
        b, n, _ = values.shape
        if pad_mask is None:
            pad_mask = torch.zeros(b, n, dtype=torch.bool, device=values.device)
        if mask is None:
            mask = gen_mask(self.mask_ratio, pad_mask)
        pos_t = positions.transpose(1, 2)  # [B, N, n_axes]

        def split(t, m):
            return t[m].reshape(b, -1, *t.shape[2:])

        target = split(values[..., : self.target_channels], mask)
        if target.shape[1] == 0:
            raise ValueError("mask selects no tokens; nothing to reconstruct")
        m_pos, m_pad = split(pos_t, mask).transpose(1, 2), split(pad_mask, mask)
        x = self.projection(split(values, ~mask))
        v_pos, v_pad = split(pos_t, ~mask).transpose(1, 2), split(pad_mask, ~mask)

        x, v_pos, v_pad = self.add_cls(x, v_pos, v_pad)
        x = self.transformer(x, self.rope.prepare(v_pos), attention_mask(v_pad))
        x = self.encoder_to_decoder(x)

        k = target.shape[1]
        x = torch.cat([x, self.mask_token.expand(b, k, -1).to(x.dtype)], dim=1)
        pos = torch.cat([v_pos, m_pos], dim=2)
        pad = torch.cat([v_pad, m_pad], dim=1)
        x = self.decoder(x, self.decoder_rope.prepare(pos), attention_mask(pad))
        pred = self.head(x[:, -k:])

        real = (~m_pad).to(pred.dtype)[..., None]
        loss = (F.mse_loss(pred.float(), target.float(), reduction="none") * real).sum()
        loss = loss / (real.sum() * self.target_channels).clamp(min=1)
        return MAEOutput(loss=loss, pred=pred, target=target, mask=mask)

    def backbone(self, pool: str = "cls") -> RoMAE:
        """A :class:`RoMAE` encoder initialised from the pretrained weights."""
        model = RoMAE(pool=pool, **self.hparams)
        model.projection.load_state_dict(self.projection.state_dict())
        model.transformer.load_state_dict(self.transformer.state_dict())
        if self.cls is not None:
            model.cls.data.copy_(self.cls.data)
        return model
