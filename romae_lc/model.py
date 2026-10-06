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
* :class:`MaskedDecoder` - the decoder side of that recipe alone, as a head
  over an encoder owned by another model (a reconstruction term next to a
  world-model or JEPA loss on the same backbone).
* :class:`RoMAEForClassification` - encoder plus a linear head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import rope as rope_lib
from .rope import BlockRope, Rotation
from .spectral import SpectralLayer
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


def _layouts(rope) -> list[list[dict]]:
    """A layout (list of block dicts) or a per-layer list of layouts as a
    list of layouts."""
    rope = list(rope)
    if not rope:
        raise ValueError("an explicit rope layout needs at least one block")
    if isinstance(rope[0], dict):
        return [rope]
    return [list(lay) for lay in rope]


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
            (time axial + nD-RoPE simplex over the other axes), an explicit
            layout list (see :class:`~romae_lc.rope.BlockRope`), or a list
            of ``depth`` such layouts, one per encoder layer, so that the
            layers (and, through per-head ladders, the heads) resolve
            different timescales. :attr:`rope` is the first layer's block
            and :attr:`rope_layout` the description that rebuilds them all.
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
        rope: str | list[dict] | list[list[dict]] = "axial",
        rope_base: float = 10000.0,
        p_rope: float = 0.75,
        use_cls: bool = True,
        rope_timescales=None,
        abs_timescales=None,
        spectral: dict | None = None,
    ):
        super().__init__()
        self.cfg = config(encoder)
        self.n_channels, self.n_axes, self.use_cls = n_channels, n_axes, use_cls
        if isinstance(rope, str):
            frac = None if rope == "axial" else 0.5
            layouts = [
                rope_lib.layout(
                    self.cfg.head_dim,
                    n_axes,
                    rope,
                    time_frac=frac,
                    time_base=rope_base,
                    p=p_rope,
                    time_timescales=rope_timescales,
                )
            ]
        else:
            if rope_timescales is not None:
                raise ValueError("rope_timescales only applies to the string layouts")
            layouts = _layouts(rope)
            if len(layouts) not in (1, self.cfg.depth):
                raise ValueError(
                    f"{len(layouts)} per-layer rope layouts for depth {self.cfg.depth}"
                )
        self.rope_layers = nn.ModuleList(
            BlockRope(self.cfg.head_dim, self.cfg.nhead, lay) for lay in layouts
        )
        for block in self.rope_layers:
            if block.n_axes > n_axes:
                raise ValueError(
                    f"rope layout uses {block.n_axes} axes, n_axes={n_axes}"
                )
        self.projection = nn.Linear(n_channels, self.cfg.d_model)
        # Absolute time features (NeRF-style): sines and cosines of the time
        # since the window's start (position units, the CLS sits at 0) at
        # every timescale in ``abs_timescales``, projected and added to the
        # token embedding. The rotary blocks only ever see time differences,
        # so without this the token content is translation invariant and the
        # pooled latent has no way to hold the phase of the window.
        self.abs_proj = None
        if abs_timescales is not None:
            ts = torch.as_tensor([float(v) for v in abs_timescales], dtype=torch.float32)
            if ts.numel() == 0 or not torch.isfinite(ts).all() or (ts <= 0).any():
                raise ValueError("abs_timescales must be positive finite values")
            self.register_buffer("abs_timescales", ts)
            self.abs_proj = nn.Linear(2 * ts.numel(), self.cfg.d_model, bias=False)
        self.transformer = Transformer(self.cfg)
        # the spectral layer (a learned periodogram over the tokens, see
        # romae_lc.spectral) between encoder blocks, its summary on the CLS
        self.spectral = SpectralLayer(self.cfg.d_model, **spectral) if spectral else None
        if self.spectral is not None and not (0 <= self.spectral.after_layer <= self.cfg.depth):
            raise ValueError(f"spectral after_layer {self.spectral.after_layer} must be within 0..{self.cfg.depth}")
        self.cls = nn.Parameter(torch.zeros(self.cfg.d_model)) if use_cls else None
        self.apply(_init_weights)
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)

    @property
    def embed_dim(self) -> int:
        return self.cfg.d_model

    @property
    def rope(self) -> BlockRope:
        """The rotary block of the first layer (of every layer when shared)."""
        return self.rope_layers[0]

    @property
    def per_layer_rope(self) -> bool:
        """Whether every encoder layer has its own rotary block."""
        return len(self.rope_layers) > 1

    @property
    def rope_layout(self) -> list:
        """The layout (shared) or the per-layer list of layouts that rebuilds
        :attr:`rope_layers`; the ``rope`` entry of :attr:`hparams`."""
        if self.per_layer_rope:
            return [block.layout for block in self.rope_layers]
        return self.rope.layout

    @property
    def hparams(self) -> dict:
        """Constructor arguments that rebuild this backbone."""
        return dict(
            encoder=self.cfg,
            n_channels=self.n_channels,
            n_axes=self.n_axes,
            rope=self.rope_layout,
            use_cls=self.use_cls,
            abs_timescales=None if self.abs_proj is None else self.abs_timescales.tolist(),
            spectral=None if self.spectral is None else self.spectral.hparams,
        )

    def run_transformer(self, x, positions, pad_mask, values=None):
        """The encoder blocks on ``x [B, L, d_model]``, with the spectral
        layer (when present) after block ``after_layer``; ``values [B, N,
        C]`` are the raw token values (no CLS row) it also reads."""
        rot, mask = self.rotations(positions), attention_mask(pad_mask)
        if self.spectral is None:
            return self.transformer(x, rot, mask)
        rots = list(rot) if isinstance(rot, (list, tuple)) else [rot] * len(self.transformer.layers)
        k = self.spectral.after_layer
        for layer, r in zip(self.transformer.layers[:k], rots[:k]):
            x = layer(x, r, mask)
        x = self.spectral(x, positions, pad_mask, has_cls=self.use_cls, values=values)
        for layer, r in zip(self.transformer.layers[k:], rots[k:]):
            x = layer(x, r, mask)
        return x

    def embed(self, values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Token embeddings ``[B, N, d_model]`` of ``values [B, N, C]``: the
        value projection plus, with ``abs_timescales``, the absolute time
        features of ``positions [B, n_axes, N]`` (time axis 0, whose origin
        1.0 is the window's start)."""
        x = self.projection(values)
        if self.abs_proj is not None:
            ang = (positions[:, 0].float() - 1.0)[..., None] / self.abs_timescales
            feats = torch.cat([torch.sin(ang), torch.cos(ang)], -1)
            x = x + self.abs_proj(feats.to(x.dtype))
        return x

    def rotations(self, positions: torch.Tensor) -> Rotation | list[Rotation]:
        """The rotary tables for ``positions [B, n_axes, N]``: one
        :class:`~romae_lc.rope.Rotation` shared by the layers, or one per
        layer, as :class:`~romae_lc.transformer.Transformer` takes them."""
        if self.per_layer_rope:
            return [block.prepare(positions) for block in self.rope_layers]
        return self.rope.prepare(positions)

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
            self.embed(values, positions), positions, pad_mask
        )
        if x.shape[1] == 0:
            raise ValueError(
                "empty token batch: with use_cls=False the encoder needs at least "
                "one token in some row (an all-empty frame window; see FrameConfig)"
            )
        x = self.run_transformer(x, positions, pad_mask, values)
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
    #: the encoder's outputs ``[B, 1 + V, d_model]`` (CLS first) of the visible
    #: tokens, their positions ``[B, n_axes, 1 + V]`` and padding ``[B, 1 + V]``,
    #: for auxiliary heads on the encoder (per-token phase, the fold decoder)
    enc_tokens: "torch.Tensor | None" = None
    enc_positions: "torch.Tensor | None" = None
    enc_pad: "torch.Tensor | None" = None


def decoder_layout(enc_layout, enc_head_dim: int, dec_head_dim: int) -> list[dict]:
    """An encoder's rotary layout rescaled to a decoder head dimension.

    A per-layer list of layouts, or per-head ladders, are first folded into
    one shared layout by :func:`~romae_lc.rope.collapse_layout`. Every block
    after the first is rounded to its own channel unit (2 for axial, ``2 *
    (n_axes + 1)`` for a multi-axis simplex) and floored at the smallest
    size that keeps one active angle at its ``p``
    (:func:`~romae_lc.rope.min_axial_dim`,
    :func:`~romae_lc.rope.min_simplex_scales`); the first (time) block takes
    whatever is left. Explicit ladders (``timescales`` of an axial block,
    ``scales`` of a simplex block) are resampled in log space to the
    decoder block's number of active angles, ends kept
    (:func:`~romae_lc.rope.resample_ladder`).
    """
    enc, dec = enc_head_dim, dec_head_dim
    base = rope_lib.collapse_layout(enc_layout)
    blocks = [dict(spec) for spec in base]
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
    for old, spec in zip(base, blocks):
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


def _mae_head(enc_cfg: TransformerConfig, enc_layout, dec_cfg, target_channels: int):
    """The decoder-side modules of the masked-autoencoding recipe:
    ``(decoder, decoder_rope, encoder_to_decoder, mask_token, head)``."""
    decoder = Transformer(dec_cfg)
    decoder_rope = BlockRope(
        dec_cfg.head_dim,
        dec_cfg.nhead,
        decoder_layout(enc_layout, enc_cfg.head_dim, dec_cfg.head_dim),
    )
    encoder_to_decoder = nn.Linear(enc_cfg.d_model, dec_cfg.d_model)
    mask_token = nn.Parameter(torch.zeros(1, 1, dec_cfg.d_model))
    head = nn.Sequential(
        nn.RMSNorm(dec_cfg.d_model, eps=dec_cfg.norm_eps),
        nn.Linear(dec_cfg.d_model, target_channels),
    )
    for m in (decoder, encoder_to_decoder, head):
        m.apply(_init_weights)
    # Like the CLS token (and the MAE recipe): an exactly-zero MASK token
    # sits on the RMSNorm gradient singularity and is position-blind in
    # the first decoder layer.
    nn.init.trunc_normal_(mask_token, std=0.02)
    return decoder, decoder_rope, encoder_to_decoder, mask_token, head


def _mae_forward(
    encoder: RoMAEBase,
    parts,
    mask_ratio: float,
    target_channels: int,
    values,
    positions,
    pad_mask=None,
    mask=None,
    weight=None,
) -> MAEOutput:
    """Mask, encode the visible tokens with ``encoder``, decode the masked
    ones with the modules of ``parts`` (see :func:`_mae_head`). ``weight
    [B, N]`` (optional) weights every token's squared error in the loss
    (e.g. ``1 / sigma^2``); the loss is then the weighted mean over the
    real masked tokens."""
    b, n, _ = values.shape
    if pad_mask is None:
        pad_mask = torch.zeros(b, n, dtype=torch.bool, device=values.device)
    if mask is None:
        mask = gen_mask(mask_ratio, pad_mask)
    pos_t = positions.transpose(1, 2)  # [B, N, n_axes]

    def split(t, m):
        return t[m].reshape(b, -1, *t.shape[2:])

    target = split(values[..., :target_channels], mask)
    if target.shape[1] == 0:
        raise ValueError("mask selects no tokens; nothing to reconstruct")
    m_pos, m_pad = split(pos_t, mask).transpose(1, 2), split(pad_mask, mask)
    v_pos, v_pad = split(pos_t, ~mask).transpose(1, 2), split(pad_mask, ~mask)
    x = encoder.embed(split(values, ~mask), v_pos)

    v_values = split(values, ~mask)
    x, v_pos, v_pad = encoder.add_cls(x, v_pos, v_pad)
    x = encoder.run_transformer(x, v_pos, v_pad, v_values)
    enc_tokens = x
    x = parts.encoder_to_decoder(x)

    k = target.shape[1]
    x = torch.cat([x, parts.mask_token.expand(b, k, -1).to(x.dtype)], dim=1)
    pos = torch.cat([v_pos, m_pos], dim=2)
    pad = torch.cat([v_pad, m_pad], dim=1)
    x = parts.decoder(x, parts.decoder_rope.prepare(pos), attention_mask(pad))
    pred = parts.head(x[:, -k:])

    real = (~m_pad).to(pred.dtype)[..., None]
    if weight is not None:
        real = real * split(weight.to(pred.dtype)[..., None], mask)
    loss = (F.mse_loss(pred.float(), target.float(), reduction="none") * real.float()).sum()
    loss = loss / (real.float().sum() * target_channels).clamp(min=1e-8)
    return MAEOutput(loss=loss, pred=pred, target=target, mask=mask, enc_tokens=enc_tokens, enc_positions=v_pos, enc_pad=v_pad)


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
        (
            self.decoder,
            self.decoder_rope,
            self.encoder_to_decoder,
            self.mask_token,
            self.head,
        ) = _mae_head(self.cfg, self.rope_layout, self.dec_cfg, target_channels)

    def _decoder_layout(self) -> list[dict]:
        """The encoder's layout rescaled to the decoder head dimension, see
        :func:`decoder_layout`."""
        return decoder_layout(
            self.rope_layout, self.cfg.head_dim, self.dec_cfg.head_dim
        )

    def forward(self, values, positions, pad_mask=None, mask=None, weight=None) -> MAEOutput:
        """Mask, encode the visible tokens, decode the masked ones.

        Args:
            values: ``[B, N, C]`` token values.
            positions: ``[B, n_axes, N]`` continuous positions.
            pad_mask: Optional bool ``[B, N]``, True = padding.
            mask: Optional bool ``[B, N]`` token mask with the same number of
                True entries per row, at least one (see :func:`gen_mask`);
                sampled from ``mask_ratio`` when omitted.
            weight: Optional ``[B, N]`` per-token loss weights (see
                :func:`_mae_forward`).
        """
        return _mae_forward(
            self,
            self,
            self.mask_ratio,
            self.target_channels,
            values,
            positions,
            pad_mask,
            mask,
            weight,
        )

    def backbone(self, pool: str = "cls") -> RoMAE:
        """A :class:`RoMAE` encoder initialised from the pretrained weights."""
        model = RoMAE(pool=pool, **self.hparams)
        model.projection.load_state_dict(self.projection.state_dict())
        if self.abs_proj is not None:
            model.abs_proj.load_state_dict(self.abs_proj.state_dict())
        model.transformer.load_state_dict(self.transformer.state_dict())
        if self.spectral is not None:
            model.spectral.load_state_dict(self.spectral.state_dict())
        if self.cls is not None:
            model.cls.data.copy_(self.cls.data)
        return model


class MaskedDecoder(nn.Module):
    """The decoder side of :class:`RoMAEForPreTraining` as a head over an
    encoder that belongs to another model.

    The encoder is not registered here (its parameters stay with their
    owner, e.g. the ``backbone`` of a :class:`~romae_lc.lewm.LeWorldModel`),
    it is passed to :meth:`forward`, so a masked reconstruction loss can be
    added to any objective trained on the same backbone. The decoder's
    rotary layout is the encoder's, rescaled by :func:`decoder_layout`
    (per-head and per-layer ladders collapsed to one), so build it after the
    encoder and rebuild it from the same encoder with :attr:`hparams`.

    Args:
        encoder: The :class:`RoMAEBase` whose outputs are decoded.
        decoder: Decoder size name, dict or config.
        mask_ratio: Fraction of real tokens to mask when no mask is given.
        target_channels: Leading value channels to reconstruct.
    """

    def __init__(
        self,
        encoder: RoMAEBase,
        decoder: str | dict | TransformerConfig = "tiny-shallow",
        mask_ratio: float = 0.5,
        target_channels: int = 1,
    ):
        super().__init__()
        if not 0 < mask_ratio <= 1:
            raise ValueError(f"mask_ratio must be in (0, 1], got {mask_ratio}")
        self.mask_ratio, self.target_channels = mask_ratio, target_channels
        self.dec_cfg = config(decoder)
        (
            self.decoder,
            self.decoder_rope,
            self.encoder_to_decoder,
            self.mask_token,
            self.head,
        ) = _mae_head(encoder.cfg, encoder.rope_layout, self.dec_cfg, target_channels)

    @property
    def hparams(self) -> dict:
        """Constructor arguments after ``encoder``."""
        return dict(
            decoder=self.dec_cfg,
            mask_ratio=self.mask_ratio,
            target_channels=self.target_channels,
        )

    def forward(
        self, encoder: RoMAEBase, values, positions, pad_mask=None, mask=None
    ) -> MAEOutput:
        """Mask, encode the visible tokens with ``encoder``, decode the masked
        ones; arguments as :meth:`RoMAEForPreTraining.forward`."""
        return _mae_forward(
            encoder,
            self,
            self.mask_ratio,
            self.target_channels,
            values,
            positions,
            pad_mask,
            mask,
        )
