"""Shared pieces of the latent world model project.

Everything builds on ``romae_lc``: records come from
:func:`~romae_lc.load_pc` with the fine ``class_str`` label, windows from
:class:`~romae_lc.FrameConfig` with ``with_err=True`` so that the per-point
error rides along, tokens from :func:`~romae_lc.collate_frames`, and
:func:`err_channel` turns the error into the second token channel
``(log sigma - mu) / sd`` that the encoder sees next to the magnitude. The
catalogue period never reaches a model: it is only a probe target
(:func:`probe`) and the reference of the period-recovery evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from romae_lc import (
    CLASSES,
    DEFAULT_SURVEYS,
    PC_BANDS,
    ARPredictor,
    FrameConfig,
    FrameDataset,
    LeWorldModel,
    Record,
    RoMAE,
    Tokens,
    collate_frames,
    frame_grid,
    lewm_mlp,
    load_pc,
    normalize,
    simulate,
    suggest_time_encoding,
    time_series,
    timescales_for,
    tokenize,
    wavelengths_for,
)

DATA_ROOT = "/projects/bfrf/data/PC_matches/ZTFxPC"
PROJECT_DIR = Path(__file__).resolve().parent
LABEL_FIELD = "class_str"


# --------------------------------------------------------------------------- data


@dataclass
class Data:
    """Per-band standardised records per split, the shared fine-class
    vocabulary (``record.label`` indexes it) and the tokenizer band table."""

    splits: dict[str, list[Record]]
    classes: tuple[str, ...]
    wavelengths: dict[int, float]

    def __getitem__(self, split: str) -> list[Record]:
        return self.splits[split]


def add_data_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("data")
    g.add_argument(
        "--data",
        default=DATA_ROOT,
        help="PC_matches sub-dataset directory, or 'sim' for the toy simulator",
    )
    g.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="keep only these class_str labels (default: every class)",
    )
    g.add_argument("--max-rows", type=int, default=None, help="cap rows per split")
    g.add_argument("--min-points", type=int, default=8, help="drop shorter records")
    g.add_argument("--n-sim", type=int, default=256, help="--data sim: stars")
    g.add_argument("--seed", type=int, default=0)


def class_vocabulary(root: str, field: str = LABEL_FIELD) -> tuple[str, ...]:
    """Sorted ``class_str`` values over every split: one vocabulary shared by
    train, validation and test."""
    import datasets

    dd = datasets.load_from_disk(root)
    vocab: set[str] = set()
    for split in dd:
        vocab |= set(dd[split][field])
    return tuple(sorted(vocab))


def superclass(record: Record) -> str:
    return str(record.meta.get("superclass_str", "?"))


def load_data(args, splits: Sequence[str] = ("train", "validation")) -> Data:
    """Load and standardise the requested splits (see :class:`Data`)."""
    if args.data == "sim":
        records = simulate(args.n_sim, seed=args.seed)
        for r in records:
            r.meta["class_str"] = r.meta["superclass_str"] = CLASSES[r.label]
        order = np.random.default_rng(args.seed).permutation(len(records))
        n_val = max(1, len(records) // 5)
        parts = {
            "validation": order[:n_val],
            "test": order[n_val : 2 * n_val],
            "train": order[2 * n_val :],
        }
        out = {s: [normalize(records[i]) for i in parts[s]] for s in splits}
        wl = {i: s.wavelength_nm for i, s in enumerate(DEFAULT_SURVEYS)}
        return Data(out, CLASSES, wl)
    classes = class_vocabulary(args.data)
    keep = None if not args.classes else set(args.classes)
    if keep and not keep <= set(classes):
        raise ValueError(f"unknown classes {sorted(keep - set(classes))}")
    out = {}
    for split in splits:
        recs = load_pc(
            args.data,
            split,
            label_field=LABEL_FIELD,
            classes=classes,
            max_rows=args.max_rows,
            min_points=args.min_points,
        )
        if keep:
            recs = [r for r in recs if classes[r.label] in keep]
        out[split] = [normalize(r) for r in recs]
    return Data(out, classes, wavelengths_for(PC_BANDS))


DATA_KEYS = ("data", "classes", "max_rows", "min_points", "n_sim", "seed")


def data_args_from(saved: dict, overrides=None) -> argparse.Namespace:
    """The data arguments a checkpoint was trained with, with any non-None
    attribute of ``overrides`` (a Namespace) taking precedence."""
    ns = argparse.Namespace(**{k: saved.get(k) for k in DATA_KEYS})
    for k in DATA_KEYS:
        v = getattr(overrides, k, None) if overrides is not None else None
        if v is not None:
            setattr(ns, k, v)
    return ns


def subset(records: Sequence, n: int | None, seed: int = 0) -> list:
    """A seeded random subset of at most ``n`` records (order kept)."""
    if n is None or n >= len(records):
        return list(records)
    idx = np.sort(np.random.default_rng(seed).choice(len(records), n, replace=False))
    return [records[i] for i in idx]


# ------------------------------------------------------------------ error channel


def err_stats(records: Sequence[Record]) -> tuple[float, float]:
    """Mean and std of ``log err`` over every point (standardised units)."""
    logs = np.concatenate([np.log(np.clip(r.err, 1e-12, None)) for r in records])
    return float(logs.mean()), float(logs.std() + 1e-6)


def err_channel(tokens: Tokens, stats: tuple[float, float]) -> Tokens:
    """Two-channel token values ``(m, (log sigma - mu) / sd)`` from the flux
    channel and ``Tokens.extras`` (the per-point sigma); 0 on padding."""
    if tokens.extras is None:
        raise ValueError("tokens carry no extras: cut frames with with_err=True")
    mu, sd = stats
    e = (tokens.extras.clamp_min(1e-12).log() - mu) / sd
    e = e.masked_fill(tokens.pad_mask, 0.0)
    values = torch.cat([tokens.values[..., :1], e[..., None]], -1)
    return Tokens(values, tokens.positions, tokens.pad_mask, tokens.extras)


def collate_err_frames(batch: list[dict], err_stats=None, **tokenize_kwargs) -> dict:
    """:func:`~romae_lc.collate_frames` followed by :func:`err_channel` on
    every frame when ``err_stats`` is given."""
    out = collate_frames(batch, **tokenize_kwargs)
    if err_stats is not None:
        out["frames"] = [err_channel(f, err_stats) for f in out["frames"]]
    return out


@dataclass
class TokenSpec:
    """How windows become tokens: the tokenizer keywords (band table and
    ``time_scale``) and the error-channel statistics (``None`` = single
    channel)."""

    tokenize: dict
    err_stats: tuple[float, float] | None = None

    @property
    def n_channels(self) -> int:
        return 1 if self.err_stats is None else 2

    def collate(self) -> Callable:
        return partial(collate_err_frames, err_stats=self.err_stats, **self.tokenize)

    def tokens(self, frames: Sequence[tuple]) -> Tokens:
        """Tokenize a list of ``(t, y, band[, err])`` windows, one per object."""
        cols = [
            [torch.from_numpy(np.ascontiguousarray(a)) for a in c] for c in zip(*frames)
        ]
        extras = cols[3] if len(cols) == 4 else None
        tok = tokenize(*cols[:3], extras=extras, **self.tokenize)
        return err_channel(tok, self.err_stats) if self.err_stats else tok

    def to_dict(self) -> dict:
        return dict(tokenize=dict(self.tokenize), err_stats=self.err_stats)

    @classmethod
    def from_dict(cls, d: dict) -> "TokenSpec":
        stats = d.get("err_stats")
        return cls(dict(d["tokenize"]), None if stats is None else tuple(stats))


# ------------------------------------------------------------------------ frames


def add_frame_args(
    parser, window: float = 500.0, min_tokens: int = 16, max_tokens: int = 256
) -> None:
    g = parser.add_argument_group("frames")
    g.add_argument("--window", type=float, default=window, help="window, days")
    g.add_argument(
        "--advance",
        type=float,
        nargs=2,
        default=(1.0, 2.0),
        metavar=("LO", "HI"),
        help="uniform start-to-start advance in window units",
    )
    g.add_argument("--n-frames", type=int, default=4, help="windows per sequence")
    g.add_argument("--min-tokens", type=int, default=min_tokens)
    g.add_argument("--max-tokens", type=int, default=max_tokens)


def frame_config(args, n_frames: int | None = None) -> FrameConfig:
    return FrameConfig(
        n_frames=n_frames or args.n_frames,
        window=args.window,
        advance=tuple(args.advance),
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        with_err=True,
    )


def frame_loader(
    records,
    cfg: FrameConfig,
    spec: TokenSpec,
    batch_size,
    train=False,
    workers=0,
    seed=0,
    persistent=False,
    prefetch=2,
    pin_memory=False,
) -> DataLoader:
    """Per-time-step padded token batches of frame sequences; a ``train``
    loader shuffles, drops the last batch and redraws windows every epoch.
    ``loader.dataset.indices`` says which records survived the span filter."""
    ds = FrameDataset(records, cfg, seed=seed, epoch_seed=train)
    kw = dict(
        persistent_workers=persistent and workers > 0,
        prefetch_factor=prefetch if workers > 0 else None,
        pin_memory=pin_memory,
    )
    return DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=train and len(ds) > batch_size,
        num_workers=workers,
        collate_fn=spec.collate(),
        **kw,
    )


def subsample_frame(frame: tuple, cap: int | None, rng: np.random.Generator):
    n = len(frame[0])
    if cap is None or n <= cap:
        return frame
    idx = np.sort(rng.choice(n, cap, replace=False))
    return tuple(a[idx] for a in frame)


def grid_starts(record: Record, cfg: FrameConfig, advance=None, start=None):
    """Window starts (days, record time) of ``frame_grid(..., fill=True)``."""
    t = record.t.astype(np.float64)
    advance = cfg.advance[0] if advance is None else float(advance)
    start = float(t.min()) if start is None else float(start)
    n = max(
        1, int(np.floor((t.max() - start - cfg.window) / (advance * cfg.window))) + 1
    )
    return start + advance * cfg.window * np.arange(n, dtype=np.float64)


def grid_item(
    record: Record,
    cfg: FrameConfig,
    index=0,
    advance=None,
    start=None,
    cap=None,
    seed=0,
) -> dict:
    """A :func:`~romae_lc.frame_grid` item (every window that fits) with
    windows above ``cap`` points deterministically subsampled, ready for
    ``TokenSpec.collate()``."""
    frames, actions = frame_grid(record, cfg, start=start, advance=advance, fill=True)
    if cap is not None:
        rng = np.random.default_rng([seed, index])
        frames = [subsample_frame(f, cap, rng) for f in frames]
    return dict(frames=frames, actions=actions, label=record.label, index=index)


# ------------------------------------------------------------------- time ladder


@dataclass
class Ladder:
    """A rotary time ladder: ``time_scale`` for the tokenizer, ``timescales``
    (position units) for ``RoMAE(rope_timescales=...)``, ``wavelengths`` in
    days for reading."""

    time_scale: float
    timescales: list[float]
    wavelengths: list[float]
    lam_min: float
    lam_max: float
    spacing: str
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Ladder":
        return cls(**d)


def add_ladder_args(parser) -> None:
    g = parser.add_argument_group(
        "time ladder (measured on the training curves, or explicit)"
    )
    g.add_argument(
        "--time-spacing", choices=("log", "linear", "quantile"), default="quantile"
    )
    g.add_argument("--time-mix", type=float, default=0.75, help="quantile-vs-log blend")
    g.add_argument("--max-freq", type=float, default=50.0, help="periodogram cap, 1/d")
    g.add_argument("--ladder-curves", type=int, default=128, help="curves sampled")
    g.add_argument("--lam-min", type=float, default=None, help="override, days")
    g.add_argument(
        "--lam-max", type=float, default=None, help="override, days (2 x window)"
    )
    g.add_argument("--ladder", default=None, help="reuse this ladder.json")
    g.add_argument(
        "--rope-wavelengths",
        type=float,
        nargs="+",
        default=None,
        help="explicit ladder in days (e.g. the PC_matches census: 0.01 0.053 "
        "0.107 0.218 0.428 0.835 1.64 3.43 6.98 14.3 34.9 6000); needs --time-scale",
    )
    g.add_argument(
        "--time-scale",
        type=float,
        default=None,
        help="days per position unit for --rope-wavelengths (census: 0.0015915)",
    )


def time_block_dim(width: int, heads: int, p: float) -> int:
    """Channels of the rotary time block of a backbone of this size."""
    return RoMAE(encoder=dict(d_model=width, nhead=heads, depth=1), p_rope=p).rope.dims[
        0
    ]


def measure_ladder(
    records,
    dim,
    p,
    window,
    spacing="quantile",
    mix=0.75,
    max_freq=50.0,
    max_curves=128,
    seed=0,
    lam_min=None,
    lam_max=None,
) -> Ladder:
    times, values, bands = time_series(records)
    report = suggest_time_encoding(
        times,
        values,
        dim,
        p=p,
        bands=bands,
        errors=[r.err for r in records],
        spacing=spacing,
        mix=mix,
        max_freq=max_freq,
        max_curves=max_curves,
        seed=seed,
        lam_min=lam_min,
        lam_max=2.0 * window if lam_max is None else lam_max,
    )
    return Ladder(
        float(report.time_scale),
        [float(x) for x in report.timescales],
        [float(x) for x in report.wavelengths],
        float(report.lam_min),
        float(report.lam_max),
        spacing,
        str(report),
    )


def get_ladder(args, records, out_dir: Path, dim: int, p: float) -> Ladder:
    """``--rope-wavelengths`` with ``--time-scale`` (explicit), else
    ``--ladder`` or ``out_dir/ladder.json`` when present, else measured on
    ``records``; always written to ``out_dir/ladder.json``."""
    out_dir = Path(out_dir)
    path = Path(args.ladder) if args.ladder else out_dir / "ladder.json"
    if args.rope_wavelengths:
        if not args.time_scale:
            raise ValueError("--rope-wavelengths needs --time-scale")
        w = [float(x) for x in args.rope_wavelengths]
        n_max = _n_active(dim, p)
        if len(w) > n_max:
            raise ValueError(
                f"{len(w)} wavelengths but the time block has {n_max} angles"
            )
        ladder = Ladder(
            float(args.time_scale),
            timescales_for(np.array(w), float(args.time_scale)),
            w,
            min(w),
            max(w),
            "explicit",
            f"explicit ladder of {len(w)} wavelengths, time_scale {args.time_scale}",
        )
    elif path.is_file():
        return Ladder.from_dict(json.load(open(path)))
    else:
        ladder = measure_ladder(
            records,
            dim,
            p,
            args.window,
            args.time_spacing,
            args.time_mix,
            args.max_freq,
            args.ladder_curves,
            args.seed,
            args.lam_min,
            args.lam_max,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(ladder.to_dict(), open(out_dir / "ladder.json", "w"), indent=1)
    return ladder


def _n_active(dim: int, p: float) -> int:
    return int(p * dim // 2)


# ----------------------------------------------------------------- fused encode


def fused_encode(self, frames: Sequence, project: bool = True) -> torch.Tensor:
    """Drop-in for :meth:`~romae_lc.LeWorldModel.encode` that runs the
    backbone once over all ``T`` frames (padded to a common length and
    stacked along the batch axis) instead of once per frame: the same
    latents, a quarter of the kernel launches of a 4-frame sequence.
    Padding positions are 0 and padded keys are masked, so real tokens see
    exactly what they see in a per-frame call."""
    frames = [f if isinstance(f, Tokens) else Tokens(*f) for f in frames]
    b, t = frames[0].values.shape[0], len(frames)
    n = max(f.values.shape[1] for f in frames)
    v0 = frames[0].values
    values = v0.new_zeros(b * t, n, v0.shape[-1])
    positions = frames[0].positions.new_zeros(b * t, frames[0].positions.shape[1], n)
    pad = frames[0].pad_mask.new_ones(b * t, n)
    for i, f in enumerate(frames):
        k = f.values.shape[1]
        rows = slice(i * b, (i + 1) * b)
        values[rows, :k] = f.values
        positions[rows, :, :k] = f.positions
        pad[rows, :k] = f.pad_mask
    feats = self.backbone(values, positions, pad).view(t, b, -1).transpose(0, 1)
    return self._project(feats) if project else feats


def use_fused_encode(model: LeWorldModel, enabled: bool = True) -> LeWorldModel:
    """Install :func:`fused_encode` on one model instance (or restore the
    per-frame method with ``enabled=False``)."""
    import types

    if enabled:
        model.encode = types.MethodType(fused_encode, model)
    elif "encode" in model.__dict__:
        del model.encode
    return model


# ------------------------------------------------------------------------ models


def add_model_args(parser) -> None:
    g = parser.add_argument_group(
        "model (light defaults; the design doc's full sizes are --depth 12 "
        "--pred-depth 6 --pred-heads 16 --pred-dim-head 64 --pred-mlp 2048 "
        "--proj-hidden 2048)"
    )
    g.add_argument("--width", type=int, default=192)
    g.add_argument("--depth", type=int, default=6, help="encoder blocks")
    g.add_argument("--heads", type=int, default=3)
    g.add_argument("--p-rope", type=float, default=0.75)
    g.add_argument("--attention", choices=("softmax", "linear"), default="softmax")
    g.add_argument("--pred-depth", type=int, default=3)
    g.add_argument("--pred-heads", type=int, default=4)
    g.add_argument("--pred-dim-head", type=int, default=48)
    g.add_argument("--pred-mlp", type=int, default=768)
    g.add_argument("--pred-dropout", type=float, default=0.1)
    g.add_argument("--proj-hidden", type=int, default=1024, help="projector width")
    g.add_argument("--history", type=int, default=3, help="predictor context")
    g.add_argument("--lamb", type=float, default=0.1, help="SIGReg weight")
    g.add_argument("--n-slices", type=int, default=1024, help="SIGReg projections")
    g.add_argument("--knots", type=int, default=17, help="Epps-Pulley nodes")
    g.add_argument("--t-max", type=float, default=3.0, help="Epps-Pulley bound")
    g.add_argument(
        "--no-err-channel",
        action="store_true",
        help="single-channel encoder (ablation)",
    )


def build_backbone(args, ladder: Ladder, n_channels: int) -> RoMAE:
    return RoMAE(
        encoder=dict(
            d_model=args.width,
            nhead=args.heads,
            depth=args.depth,
            attention=args.attention,
        ),
        n_channels=n_channels,
        n_axes=2,
        rope="axial",
        p_rope=args.p_rope,
        rope_timescales=list(ladder.timescales),
    )


def build_world_model(
    args, backbone: RoMAE, n_frames: int, fused: bool = True
) -> LeWorldModel:
    d = backbone.embed_dim
    predictor = ARPredictor(
        d,
        n_frames=max(args.history, n_frames - 1),
        depth=args.pred_depth,
        heads=args.pred_heads,
        dim_head=args.pred_dim_head,
        mlp_dim=args.pred_mlp,
        dropout=args.pred_dropout,
    )
    model = LeWorldModel(
        backbone,
        projector=lewm_mlp(d, args.proj_hidden),
        predictor=predictor,
        pred_proj=lewm_mlp(d, args.proj_hidden),
        history=args.history,
        lamb=args.lamb,
        n_slices=args.n_slices,
        t_max=args.t_max,
        n_points=args.knots,
    )
    return use_fused_encode(model, fused)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ------------------------------------------------------------------- checkpoints


@dataclass
class WMMeta:
    """Everything a stage-1 checkpoint records besides the weights."""

    spec: TokenSpec
    cfg: FrameConfig
    ladder: Ladder
    classes: tuple[str, ...]
    args: dict
    step: int
    metrics: dict | None = None


def wm_state(model: LeWorldModel, spec, cfg, ladder, classes, args, step, metrics=None):
    backbone = model.backbone
    return dict(
        step=step,
        args=dict(vars(args)) if isinstance(args, argparse.Namespace) else dict(args),
        state_dict=model.state_dict(),
        hparams=model.hparams,
        backbone=dict(
            backbone.hparams, encoder=asdict(backbone.cfg), pool=backbone.pool
        ),
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


def load_wm(path, device="cpu") -> tuple[LeWorldModel, WMMeta]:
    """``(model in eval mode, meta)`` from ``wm.pt`` or ``last.pt``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    backbone = RoMAE(**ckpt["backbone"])
    model = LeWorldModel.from_hparams(backbone, ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    use_fused_encode(model)
    meta = WMMeta(
        TokenSpec.from_dict(ckpt["spec"]),
        FrameConfig(**ckpt["frames"]),
        Ladder.from_dict(ckpt["ladder"]),
        tuple(ckpt["classes"]),
        ckpt["args"],
        int(ckpt["step"]),
        ckpt.get("metrics"),
    )
    return model.to(device).eval(), meta


# ------------------------------------------------------------------------ probes


def _standardize(z_tr, z_va):
    mu, sd = z_tr.mean(0), z_tr.std(0) + 1e-6
    return (z_tr - mu) / sd, (z_va - mu) / sd


def linear_probe(z_tr, y_tr, z_va, y_va, n_classes, steps=300, wd=1e-3) -> dict:
    """Multinomial logistic regression on standardised features (L-BFGS)."""
    x_tr, x_va = (torch.from_numpy(z).float() for z in _standardize(z_tr, z_va))
    y_tr, y_va = (torch.as_tensor(y, dtype=torch.long) for y in (y_tr, y_va))
    head = nn.Linear(x_tr.shape[1], n_classes)
    opt = torch.optim.LBFGS(
        head.parameters(), max_iter=steps, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(head(x_tr), y_tr) + wd * head.weight.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        acc = (head(x_va).argmax(1) == y_va).float().mean().item()
        train_acc = (head(x_tr).argmax(1) == y_tr).float().mean().item()
    return dict(acc=acc, train_acc=train_acc)


def ridge_r2(z_tr, t_tr, z_va, t_va, alpha=0.1) -> float:
    x_tr, x_va = (x.astype(np.float64) for x in _standardize(z_tr, z_va))
    t_tr, t_va = np.asarray(t_tr, dtype=np.float64), np.asarray(t_va, dtype=np.float64)
    n, d = x_tr.shape
    a = x_tr.T @ x_tr / n + alpha * np.eye(d)
    w = np.linalg.solve(a, x_tr.T @ (t_tr - t_tr.mean()) / n)
    resid = t_va - t_tr.mean() - x_va @ w
    return float(1.0 - (resid**2).sum() / ((t_va - t_va.mean()) ** 2).sum())


@torch.no_grad()
def embed_records(model, records, cfg, spec, batch_size, device, project=False, seed=0):
    """``(z [n, D], kept indices)``: mean over the windows of one deterministic
    sequence per record (records too short for a sequence are dropped)."""
    ds = FrameDataset(records, cfg, seed=seed, epoch_seed=False)
    loader = DataLoader(ds, batch_size, shuffle=False, collate_fn=spec.collate())
    was = model.training
    model.eval()
    zs = []
    for batch in loader:
        frames = [f.to(device) for f in batch["frames"]]
        zs.append(model.encode(frames, project=project).mean(1).float().cpu())
    model.train(was)
    return torch.cat(zs).numpy(), list(ds.indices)


def probe(
    model, train, val, cfg, spec, batch_size, device, n_classes, project=False, seed=0
):
    """Class accuracy of a linear probe and ridge R2 on log10 period, pooled
    and within every superclass with at least ``min_group`` records on both
    sides (``r2_by_superclass``)."""
    z_tr, i_tr = embed_records(
        model, train, cfg, spec, batch_size, device, project, seed
    )
    z_va, i_va = embed_records(model, val, cfg, spec, batch_size, device, project, seed)
    r_tr, r_va = [train[i] for i in i_tr], [val[i] for i in i_va]
    y_tr, y_va = (np.array([r.label for r in rs]) for rs in (r_tr, r_va))
    p_tr, p_va = (np.log10([r.period for r in rs]) for rs in (r_tr, r_va))
    out = linear_probe(z_tr, y_tr, z_va, y_va, n_classes)
    out["r2"] = ridge_r2(z_tr, p_tr, z_va, p_va)
    g_tr, g_va = (np.array([superclass(r) for r in rs]) for rs in (r_tr, r_va))
    by = {}
    for g in sorted(set(g_va.tolist())):
        m_tr, m_va = g_tr == g, g_va == g
        if m_tr.sum() >= 50 and m_va.sum() >= 20 and p_va[m_va].std() > 0:
            by[g] = ridge_r2(z_tr[m_tr], p_tr[m_tr], z_va[m_va], p_va[m_va])
    out["r2_by_superclass"] = by
    return out


# -------------------------------------------------------------------------- misc


def cosine_schedule(total: int, warmup_frac: float = 0.01):
    warmup = max(1, int(warmup_frac * total))

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    return multiplier


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def add_device_arg(parser) -> None:
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )


def get_device(args) -> torch.device:
    return torch.device(args.device)


def to_device(batch: dict, device):
    frames = [f.to(device) for f in batch["frames"]]
    return frames, batch["actions"].to(device)


class JsonlLog:
    """Append-only JSON lines log with a timestamp per line."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **fields) -> None:
        with open(self.path, "a") as f:
            f.write(
                json.dumps(dict(time=time.time(), **fields), default=_json_default)
                + "\n"
            )


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dump_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
