"""Pre-norm transformer with rotary attention (softmax or linear kernel).

Every attention layer receives a prepared :class:`~romae_lc.rope.Rotation`
(one shared by all layers, or one per layer) and applies it to queries and
keys, so positional information enters only through the relative rotary
phase. Padding is handled with a boolean mask
``[B, 1, L, L]`` (True = may attend), built by :func:`attention_mask`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import Rotation


@dataclass
class TransformerConfig:
    """Hyperparameters of a :class:`Transformer` stack.

    Attributes:
        d_model: Embedding dimension.
        nhead: Number of attention heads (``d_model % nhead == 0`` and
            ``d_model // nhead`` even, since every head is rotated in pairs).
        depth: Number of blocks.
        mlp_ratio: Feed-forward hidden size as a multiple of ``d_model``.
        attention: ``"softmax"`` (scaled dot product) or ``"linear"`` (kernel
            attention with ``elu + 1`` features and a rotary numerator, O(L)).
        drop_path_rate: Stochastic depth, linearly ramped over the blocks.
        dropout: Dropout inside the feed-forward blocks.
        attn_dropout: Attention-probability dropout (softmax only).
        proj_dropout: Dropout after the attention output projection.
        pos_dropout: Dropout on the rotated queries and keys (softmax only).
        norm_eps: RMSNorm epsilon.
    """

    d_model: int = 432
    nhead: int = 6
    depth: int = 12
    mlp_ratio: float = 4.0
    attention: str = "softmax"
    drop_path_rate: float = 0.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    pos_dropout: float = 0.0
    norm_eps: float = 1e-12

    def __post_init__(self):
        if self.d_model % self.nhead:
            raise ValueError(
                f"d_model={self.d_model} not divisible by nhead={self.nhead}"
            )
        if self.head_dim % 2:
            raise ValueError(
                f"head_dim={self.head_dim} must be even for rotary attention"
            )
        if self.attention not in ("softmax", "linear"):
            raise ValueError(
                f"attention must be softmax|linear, got {self.attention!r}"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.nhead


#: Encoder sizes of the RoMAE paper (``tiny-shallow`` is its decoder) plus a
#: ``large`` extrapolation of this library that is not in the paper.
SIZES: dict[str, dict] = {
    "tiny-shallow": dict(d_model=180, nhead=3, depth=2),
    "tiny": dict(d_model=180, nhead=3, depth=12),
    "small": dict(d_model=432, nhead=6, depth=12),
    "base": dict(d_model=720, nhead=12, depth=12),
    "large": dict(d_model=960, nhead=16, depth=24),  # not in the paper
}


def config(
    spec: str | dict | TransformerConfig | None, **overrides
) -> TransformerConfig:
    """Resolve a size name, a dict of fields or a config into a config."""
    if isinstance(spec, TransformerConfig):
        return TransformerConfig(**{**asdict(spec), **overrides})
    if isinstance(spec, str):
        if spec not in SIZES:
            raise ValueError(f"unknown size {spec!r}; choose from {list(SIZES)}")
        spec = SIZES[spec]
    return TransformerConfig(**{**(spec or {}), **overrides})


def attention_mask(pad_mask: torch.Tensor | None, length: int | None = None):
    """Boolean key mask ``[B, 1, L, L]`` from ``pad_mask [B, L]`` (True = pad).

    Every query may attend to every non-padding key; padding queries still
    see the real keys, so no row is empty. Returns ``None`` without padding.
    """
    if pad_mask is None:
        return None
    b, n = pad_mask.shape
    return (~pad_mask)[:, None, None, :].expand(b, 1, length or n, n)


class DropPath(nn.Module):
    """Stochastic depth per sample (Huang et al. 2016)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep).div_(keep)


class Attention(nn.Module):
    """Multi-head attention with rotary q/k, bias-free (Llama style)."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.nhead, self.head_dim, self.kind = cfg.nhead, cfg.head_dim, cfg.attention
        # Explicit logit scale (a float attribute so mu-P code can retune it).
        self.scale = self.head_dim**-0.5
        self.attn_dropout = cfg.attn_dropout
        self.pos_dropout = nn.Dropout(cfg.pos_dropout)
        self.proj_dropout = nn.Dropout(cfg.proj_dropout)
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rot: Rotation | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.wq(x).view(b, n, self.nhead, self.head_dim)
        k = self.wk(x).view(b, n, self.nhead, self.head_dim)
        v = self.wv(x).view(b, n, self.nhead, self.head_dim)
        if self.kind == "linear":
            out = self._linear(q, k, v, rot, mask)
        else:
            if rot is not None:
                q, k = self.pos_dropout(rot(q)), self.pos_dropout(rot(k))
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                scale=self.scale,
            ).transpose(1, 2)
        return self.proj_dropout(self.wo(out.reshape(b, n, -1).to(x.dtype)))

    def _linear(self, q, k, v, rot, mask):
        """Kernel attention, O(L): ``phi(x) = elu(x) + 1`` (Katharopoulos et
        al. 2020) with the RoFormer rotary numerator (Su et al. 2021)::

            out_i = sum_j (R_i phi(q_i)) . (R_j phi(k_j)) v_j / sum_j phi(q_i) . phi(k_j)

        Runs in float32; supports key (padding) masks only.
        """
        b, n = q.shape[:2]
        key_ok = _key_mask(mask)
        with torch.autocast(device_type=q.device.type, enabled=False):
            q, k, v = F.elu(q.float()) + 1.0, F.elu(k.float()) + 1.0, v.float()
            if key_ok is not None:
                k = k * key_ok[:, :, None, None]
                v = v * key_ok[:, :, None, None]
            qr, kr = (rot(q), rot(k)) if rot is not None else (q, k)
            # plain matmuls with the token axis as the inner dimension: the
            # einsum forms produced an outer-product bmm in the backward
            # pass, which torch routes to a triton kernel that cannot build
            # on a machine without Python development headers (2026-10-01)
            kv = kr.permute(0, 2, 3, 1) @ v.permute(0, 2, 1, 3)  # [b, h, d, e]
            num = (qr.permute(0, 2, 1, 3) @ kv).permute(0, 2, 1, 3)  # [b, l, h, e]
            den = (q * k.sum(1, keepdim=True)).sum(-1).clamp(min=1e-6)  # [b, l, h]
            return num / den[..., None]


def _key_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    """``[B, L]`` key mask from a ``[B, 1, L, L]`` mask that is the same for
    every query row (the only kind linear attention can factorise)."""
    if mask is None:
        return None
    m = mask[:, 0]
    if not bool((m == m[:, :1, :]).all()):
        raise NotImplementedError("linear attention supports key (padding) masks only")
    return m[:, 0, :]


class FeedForward(nn.Module):
    """SiLU feed-forward block (bias-free)."""

    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x))))


class Block(nn.Module):
    """Pre-norm transformer block with rotary attention and stochastic depth."""

    def __init__(self, cfg: TransformerConfig, drop_path: float):
        super().__init__()
        self.attention = Attention(cfg)
        self.feed_forward = FeedForward(
            cfg.d_model, round(cfg.mlp_ratio * cfg.d_model), cfg.dropout
        )
        self.attention_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, rot=None, mask=None):
        x = x + self.drop_path(self.attention(self.attention_norm(x), rot, mask))
        return x + self.drop_path(self.feed_forward(self.ffn_norm(x)))


class Transformer(nn.Module):
    """A stack of :class:`Block` sharing one rotary :class:`Rotation`, or
    taking one per block."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        rates = [
            cfg.drop_path_rate * i / max(cfg.depth - 1, 1) for i in range(cfg.depth)
        ]
        self.layers = nn.ModuleList(Block(cfg, r) for r in rates)

    def forward(
        self,
        x: torch.Tensor,
        rot: Rotation | Sequence[Rotation] | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run all blocks on ``x [B, L, d_model]``; ``rot`` is one rotation
        for every block or a sequence with exactly one per block."""
        if isinstance(rot, (list, tuple)):
            if len(rot) != len(self.layers):
                raise ValueError(f"{len(rot)} rotations for {len(self.layers)} blocks")
            rots = rot
        else:
            rots = [rot] * len(self.layers)
        for layer, r in zip(self.layers, rots):
            x = layer(x, r, mask)
        return x
