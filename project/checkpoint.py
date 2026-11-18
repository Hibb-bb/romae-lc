"""Save and load encoder checkpoints (``mae.pt``, ``last.pt``).

A checkpoint is one dict: the weights (``state_dict``), how to rebuild the
model (``backbone``, ``mae``, and ``jepa`` with ``kind == "jepa"``), how to
make tokens (``spec``, ``frames``, ``ladder``), the class names, the training
flags (``args``) and the recipe text (``recipe``, when the run used one).

    enc, meta = load_encoder("project/runs/enc_j1/mae.pt", device)
    z = enc.encode(batch["frames"])  # [B, T, 384]
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from romae_lc import FrameConfig, RoMAEForPreTraining
from romae_lc.model import pool_tokens

from project.backbone import Ladder
from project.jepa import TokenJEPA
from project.windows import TokenSpec, fuse_frames


# ------------------------------------------------------------------ save


def mae_state(
    model: RoMAEForPreTraining, spec, cfg, ladder, classes, args, step, metrics=None
):
    """A masked-pretraining checkpoint: the whole model and everything
    :func:`load_mae` needs to rebuild it and its encoder."""
    return dict(
        step=step,
        args=dict(vars(args)) if isinstance(args, argparse.Namespace) else dict(args),
        state_dict=model.state_dict(),
        backbone=dict(model.hparams, encoder=asdict(model.cfg)),
        mae=dict(
            decoder=asdict(model.dec_cfg),
            mask_ratio=model.mask_ratio,
            target_channels=model.target_channels,
            phasor=0 if getattr(model, "phasor", None) is None else model.phasor.harmonics,
            phase_rope=int(getattr(model, "phase_rope", 0)),
            phasor_init=float(getattr(model, "phasor_init", 0.02)),
            phase_freq=str(getattr(model, "phase_freq", "soft")),
        ),
        spec=spec.to_dict(),
        frames=asdict(cfg),
        ladder=ladder.to_dict(),
        classes=list(classes),
        metrics=metrics,
    )


def jepa_state(model: TokenJEPA, spec, cfg, ladder, classes, args, step, metrics=None) -> dict:
    """The stage-1 checkpoint of a token JEPA: ``mae_state``'s keys with
    ``kind="jepa"`` and the key ``jepa`` (EMA schedule, loss, which encoder
    is used downstream); :func:`load_mae` dispatches on ``kind``."""
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


def save_atomic(state: dict, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


# ------------------------------------------------------------------ load


@dataclass
class EncoderMeta:
    """Everything an encoder checkpoint (``mae.pt``) records besides the weights."""

    spec: TokenSpec
    cfg: FrameConfig
    ladder: Ladder
    classes: tuple[str, ...]
    args: dict
    step: int
    metrics: dict | None = None


def _read_ckpt(path) -> dict:
    """A checkpoint dict from its path, or the dict itself when already
    loaded (the loaders below take either, so a file is read once)."""
    if isinstance(path, dict):
        return path
    return torch.load(path, map_location="cpu", weights_only=False)


def _meta_from(ckpt: dict) -> EncoderMeta:
    return EncoderMeta(
        TokenSpec.from_dict(ckpt["spec"]),
        FrameConfig(**ckpt["frames"]),
        Ladder.from_dict(ckpt["ladder"]),
        tuple(ckpt["classes"]),
        ckpt["args"],
        int(ckpt["step"]),
        ckpt.get("metrics"),
    )


def load_mae(path, device="cpu") -> tuple[nn.Module, EncoderMeta]:
    """``(model in eval mode, meta)`` from a ``mae.pt`` (or its ``last.pt``,
    or the loaded dict). ``kind == "jepa"`` rebuilds a
    :class:`project.jepa.TokenJEPA`; a checkpoint without ``kind`` is a plain
    masked autoencoder (:class:`~romae_lc.RoMAEForPreTraining`). Both expose
    ``backbone(pool)``, ``encode``, ``use_cls``, ``rope_layout``,
    ``embed_dim`` and ``hparams``. Other kinds (bottleneck, GRU-ODE, world
    model) are legacy: check out the tag ``pre-cleanup`` to load them."""
    ckpt = _read_ckpt(path)
    kind = ckpt.get("kind")
    if kind == "jepa":
        model = TokenJEPA.from_checkpoint(ckpt)
    elif kind is None and "mae" in ckpt:
        model = RoMAEForPreTraining(**ckpt["mae"], **ckpt["backbone"])
        model.load_state_dict(ckpt["state_dict"])
    else:
        raise ValueError(f"not a JEPA or masked-autoencoder checkpoint (kind {kind!r}); its code is in legacy/, check out the tag pre-cleanup to load it")
    return model.to(device).eval(), _meta_from(ckpt)


# ------------------------------------------------------------------ the encoder as the probes and caches use it


def fused_features(backbone: nn.Module, frames: Sequence) -> torch.Tensor:
    """``[B, T, D]`` features of a ``(values, positions, pad_mask) -> [B, D]``
    backbone over ``T`` frames in one call (:func:`fuse_frames`)."""
    b, t = frames[0].values.shape[0], len(frames)
    return backbone(*fuse_frames(frames)).view(t, b, -1).transpose(0, 1)


class FrameEncoder(nn.Module):
    """A bare backbone with the ``encode(frames, project)`` interface that
    :func:`probe` and :func:`shuffle_score` call during pretraining."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def encode(self, frames: Sequence, project: bool = False) -> torch.Tensor:
        return fused_features(self.backbone, frames)


class PooledEncoder(nn.Module):
    """The CLS embedding of a stage-1 autoencoder's encoder
    (:class:`~romae_lc.RoMAEForPreTraining`, or any model with the same
    ``encode(values, positions, pad_mask) -> (tokens, pad_mask)`` and
    ``use_cls``), as the ``(values, positions, pad_mask) -> [B, D]`` backbone
    the probes and :class:`LatentEncoder` take. ``embed_dim`` and
    ``rope_layout`` are the model's, so a decoder can be built on it like on
    a :class:`~romae_lc.RoMAE`."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim

    @property
    def rope_layout(self):
        return self.model.rope_layout

    def forward(self, values, positions, pad_mask=None):
        x, pad = self.model.encode(values, positions, pad_mask)
        if getattr(self.model, "attn_pool", None) is not None:  # the attention-pooling head replaces the CLS
            return self.model.attn_pool(x, pad)
        return pool_tokens(x, pad, "cls", self.model.use_cls)


class LatentEncoder(nn.Module):
    """A frozen encoder loaded from a checkpoint. The latent of a window is
    the pooled CLS feature of the encoder (:class:`PooledEncoder`).
    ``encode(frames)`` returns ``[B, T, dim]`` float latents in one backbone
    call over the ``T`` frames. The module is always in eval mode with every
    parameter frozen (``train()`` keeps eval) and it is never saved.
    Attributes: ``kind`` (always ``"mae"``; the latent caches store it),
    ``dim`` (latent width), ``backbone`` (the ``(values, positions,
    pad_mask) -> [B, dim]`` encoder module) and ``model`` (the loaded
    checkpoint model).

        enc, meta = load_encoder("project/runs/enc_j1/mae.pt", device)
        z = enc.encode(batch["frames"])  # [B, T, 384]
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.kind, self.model = "mae", model
        self.backbone = PooledEncoder(model)
        self.dim = int(model.embed_dim)
        self.model.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "LatentEncoder":
        """A frozen encoder stays in eval mode (BatchNorm running statistics,
        no dropout) whatever the caller asks."""
        return super().train(False)

    def encode(self, frames: Sequence) -> torch.Tensor:
        """``[B, T, dim]`` latents of ``T`` frames."""
        return fused_features(self.backbone, frames).float()


def load_encoder(path, device="cpu") -> tuple[LatentEncoder, EncoderMeta]:
    """``(frozen LatentEncoder, meta)`` from a ``mae.pt`` (:func:`load_mae`)."""
    model, meta = load_mae(path, device)
    return LatentEncoder(model), meta
