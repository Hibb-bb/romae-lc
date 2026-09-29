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

from contextlib import contextmanager

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
    RoMAEForPreTraining,
    Tokens,
    collapse_layout,
    collate_frames,
    frame_grid,
    lewm_mlp,
    load_pc,
    normalize,
    resample_ladder,
    simulate,
    suggest_time_encoding,
    time_series,
    timescales_for,
    tokenize,
    wavelengths_for,
)
from romae_lc import layout as rope_layout
from romae_lc.model import pool_tokens

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


def fine_class(record: Record) -> str:
    return str(record.meta.get("class_str", "?"))


#: Fine classes kept out of the class probe (decision of 2026-09-26): they are
#: anomaly classes (Blazhko modulation, O'Connell-effect eclipsing binaries),
#: the kind of object the anomaly scores are meant to find, so a classifier
#: must not be trained to name them. Their periods still count in every
#: period metric and they stay in the training data.
ANOMALY_CLASSES = ("RRab-Blazhko", "RRc-Blazhko", "EW/EB-OC")


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
    g = parser.add_argument_group(
        "frames (a record must span window * (1 + (n_frames - 1) * advance_lo) "
        "days to host a sequence: 2000 d for the defaults at window 250, which "
        "99% of the ZTF records do)"
    )
    g.add_argument("--window", type=float, default=window, help="window, days")
    g.add_argument(
        "--advance",
        type=float,
        nargs=2,
        default=(1.0, 1.5),
        metavar=("LO", "HI"),
        help="uniform start-to-start advance in window units (the final range "
        "of the advance curriculum, see --advance-start)",
    )
    g.add_argument("--n-frames", type=int, default=8, help="windows per sequence")
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
    (position units) for the time block of every head, ``wavelengths`` the
    same in days for reading.

    A *shared* ladder is a flat list (``layers == heads == 1``). A *dense*
    ladder is nested: ``[heads][n]`` when the heads of every layer tile it
    (``layers == 1``) or ``[layers][heads][n]`` when the layers do too, and
    ``deal`` says how the rungs were dealt (:func:`deal_ladder`). Old
    ``ladder.json`` files load as shared ladders.
    """

    time_scale: float
    timescales: list
    wavelengths: list
    lam_min: float
    lam_max: float
    spacing: str
    summary: str = ""
    layers: int = 1
    heads: int = 1
    deal: str = "shared"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Ladder":
        return cls(**d)

    @property
    def flat(self) -> list[float]:
        """Every distinct wavelength (days), ascending."""
        return [float(v) for v in np.unique(np.asarray(self.wavelengths, float))]

    @property
    def n_rungs(self) -> int:
        return len(self.flat)

    @property
    def per_head(self) -> int:
        """Active time angles per head."""
        return int(np.asarray(self.timescales, float).shape[-1])


@dataclass
class RopeGeometry:
    """How many rotary time angles a backbone has: ``layers`` x ``heads``
    heads of ``n_angles`` active angles each in a time block of ``time_dim``
    of the ``head_dim`` channels."""

    layers: int
    heads: int
    head_dim: int
    time_dim: int
    n_angles: int

    @property
    def n_rungs(self) -> int:
        """Distinct timescales a dense ladder gives this backbone."""
        return self.layers * self.heads * self.n_angles


def rope_geometry(args) -> RopeGeometry:
    """The rotary geometry of the backbone ``build_backbone(args, ...)``
    builds (``--width --heads --depth --time-frac --p-rope``)."""
    head_dim = args.width // args.heads
    blocks = rope_layout(head_dim, 2, time_frac=args.time_frac, p=args.p_rope)
    time_dim = blocks[0]["dim"]
    return RopeGeometry(
        args.depth, args.heads, head_dim, time_dim, _n_active(time_dim, args.p_rope)
    )


DEALS = ("bands", "comb")


def deal_ladder(wavelengths, layers: int, heads: int, n: int, deal: str = "bands"):
    """Deal ``layers * heads * n`` ascending wavelengths to the heads of every
    layer, ``n`` each, every rung to exactly one head, as a nested
    ``[layers][heads][n]`` list.

    ``"bands"``: rung ``i`` goes to layer ``i mod layers``, so every layer
    holds every ``layers``-th rung of the grid and the layers are offset by
    one grid step; within a layer the heads take contiguous bands of that
    sub-grid, so each head is a narrow-band filter over its own range of
    timescales and the heads of one layer cover the whole band between them.
    ``"comb"``: the same across layers, but within a layer the rungs are
    dealt round-robin over the heads, so every head spans the whole band at
    ``layers * heads`` times the grid spacing.
    """
    w = np.sort(np.asarray(wavelengths, dtype=np.float64).ravel())
    m = layers * heads * n
    if w.size != m:
        raise ValueError(f"{w.size} wavelengths for {layers} x {heads} x {n} slots")
    if deal not in DEALS:
        raise ValueError(f"deal must be one of {DEALS}, got {deal!r}")
    i = np.arange(m)
    layer, j = i % layers, i // layers
    if deal == "bands":
        head, angle = j // n, j % n
    else:
        head, angle = j % heads, j // heads
    out = np.empty((layers, heads, n), dtype=np.float64)
    out[layer, head, angle] = w
    return out.tolist()


def _nest(rows: np.ndarray, layers: int, heads: int):
    """Drop the axes of size 1 of a ``[layers][heads][n]`` array in the way
    :class:`Ladder` stores it: ``[n]``, ``[heads][n]`` or all three."""
    arr = np.asarray(rows, dtype=np.float64)
    if layers == 1:
        arr = arr[0]
        if heads == 1:
            arr = arr[0]
    return arr.tolist()


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
        "--ladder-mode",
        choices=("dense", "shared"),
        default="dense",
        help="dense: one distinct rung per active time angle of every head of "
        "every layer, so the encoder resolves layers x heads x angles "
        "timescales (the folding resolution a period needs is about 1 / (4 x "
        "cycles per window)); shared: the same ladder in every head and layer",
    )
    g.add_argument(
        "--ladder-deal",
        choices=DEALS,
        default="bands",
        help="dense ladders: bands = each head a contiguous band of timescales, "
        "comb = each head spans the whole band (see deal_ladder)",
    )
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


def time_block_dim(width: int, heads: int, p: float, time_frac=None) -> int:
    """Channels of the rotary time block of a backbone of this size."""
    return rope_layout(width // heads, 2, time_frac=time_frac, p=p)[0]["dim"]


def measure_wavelengths(
    records,
    n: int,
    window,
    spacing="quantile",
    mix=0.75,
    max_freq=50.0,
    max_curves=128,
    seed=0,
    lam_min=None,
    lam_max=None,
):
    """``n`` ascending wavelengths (days) measured on the periodograms of
    ``records`` by :func:`~romae_lc.suggest_time_encoding` (``lam_max``
    defaults to twice the window), with the report."""
    times, values, bands = time_series(records)
    report = suggest_time_encoding(
        times,
        values,
        2 * n,  # a time block of 2 n channels at p = 1 has n angles
        p=1.0,
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
    w = np.unique(np.asarray(report.wavelengths, dtype=np.float64))
    if w.size < n:  # rungs that piled up on the same timescale
        w = np.asarray(resample_ladder(w, n), dtype=np.float64)
    return w, report


def measure_ladder(records, n: int, window, *measure_args) -> Ladder:
    """A shared ladder of ``n`` measured wavelengths (see
    :func:`measure_wavelengths` for the remaining arguments)."""
    w, report = measure_wavelengths(records, n, window, *measure_args)
    return Ladder(
        float(report.time_scale),
        timescales_for(w, float(report.time_scale)),
        [float(x) for x in w],
        float(report.lam_min),
        float(report.lam_max),
        report.spacing,
        str(report),
    )


def dense_ladder(records, geo: RopeGeometry, window, deal="bands", *measure_args):
    """A dense ladder: ``geo.n_rungs`` measured wavelengths dealt to the
    heads of every layer (:func:`deal_ladder`), so no two heads share a
    timescale."""
    w, report = measure_wavelengths(records, geo.n_rungs, window, *measure_args)
    days = deal_ladder(w, geo.layers, geo.heads, geo.n_angles, deal)
    ts = np.asarray(days, dtype=np.float64) / (2 * np.pi * report.time_scale)
    return Ladder(
        float(report.time_scale),
        _nest(ts, geo.layers, geo.heads),
        _nest(days, geo.layers, geo.heads),
        float(report.lam_min),
        float(report.lam_max),
        report.spacing,
        f"dense ladder of {geo.n_rungs} wavelengths ({geo.layers} layers x "
        f"{geo.heads} heads x {geo.n_angles} angles, {deal}) from {report}",
        geo.layers,
        geo.heads,
        deal,
    )


def check_ladder(ladder: Ladder, geo: RopeGeometry) -> None:
    """Raise unless ``ladder`` fits the backbone geometry ``geo``."""
    if ladder.layers not in (1, geo.layers):
        raise ValueError(f"ladder has {ladder.layers} layers, the model {geo.layers}")
    if ladder.heads not in (1, geo.heads):
        raise ValueError(f"ladder has {ladder.heads} heads, the model {geo.heads}")
    if ladder.per_head > geo.time_dim // 2:
        raise ValueError(
            f"{ladder.per_head} timescales per head do not fit the time block "
            f"({geo.time_dim} channels, {geo.time_dim // 2} angles)"
        )


def get_ladder(args, records, out_dir: Path, geo: RopeGeometry) -> Ladder:
    """``--rope-wavelengths`` with ``--time-scale`` (an explicit shared
    ladder), else ``--ladder`` or ``out_dir/ladder.json`` when present, else
    measured on ``records`` (dense over the heads and layers of ``geo`` with
    ``--ladder-mode dense``, shared otherwise); always written to
    ``out_dir/ladder.json``."""
    out_dir = Path(out_dir)
    path = Path(args.ladder) if args.ladder else out_dir / "ladder.json"
    measure_args = (
        args.time_spacing,
        args.time_mix,
        args.max_freq,
        args.ladder_curves,
        args.seed,
        args.lam_min,
        args.lam_max,
    )
    if args.rope_wavelengths:
        if not args.time_scale:
            raise ValueError("--rope-wavelengths needs --time-scale")
        w = [float(x) for x in args.rope_wavelengths]
        if len(w) > geo.n_angles:
            raise ValueError(
                f"{len(w)} wavelengths but the time block has {geo.n_angles} angles"
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
        ladder = Ladder.from_dict(json.load(open(path)))
        check_ladder(ladder, geo)
        return ladder
    elif args.ladder_mode == "dense":
        ladder = dense_ladder(
            records, geo, args.window, args.ladder_deal, *measure_args
        )
    else:
        ladder = measure_ladder(records, geo.n_angles, args.window, *measure_args)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(ladder.to_dict(), open(out_dir / "ladder.json", "w"), indent=1)
    return ladder


def _n_active(dim: int, p: float) -> int:
    return int(p * dim // 2)


# ----------------------------------------------------------------- fused encode


def fuse_frames(frames: Sequence) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The ``T`` token batches of a frame sequence padded to a common length
    and stacked along the batch axis: ``(values [B T, N, C], positions [B T,
    A, N], pad_mask [B T, N])`` with frame ``t`` in rows ``t B .. (t + 1) B``.
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
    return values, positions, pad


def fused_encode(self, frames: Sequence, project: bool = True) -> torch.Tensor:
    """Drop-in for :meth:`~romae_lc.LeWorldModel.encode` that runs the
    backbone once over all ``T`` frames (:func:`fuse_frames`) instead of
    once per frame: the same latents, a quarter of the kernel launches of a
    4-frame sequence."""
    b, t = frames[0].values.shape[0], len(frames)
    feats = self.backbone(*fuse_frames(frames)).view(t, b, -1).transpose(0, 1)
    return self._project(feats) if project else feats


def fused_features(backbone: nn.Module, frames: Sequence) -> torch.Tensor:
    """``[B, T, D]`` features of a ``(values, positions, pad_mask) -> [B, D]``
    backbone over ``T`` frames in one call (:func:`fuse_frames`)."""
    b, t = frames[0].values.shape[0], len(frames)
    return backbone(*fuse_frames(frames)).view(t, b, -1).transpose(0, 1)


class FrameEncoder(nn.Module):
    """A bare backbone with the ``encode(frames, project)`` interface of
    :class:`~romae_lc.LeWorldModel`, so :func:`probe` and the diagnostics run
    on an encoder that has no world model yet (masked pretraining)."""

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
        return pool_tokens(x, pad, "cls", self.model.use_cls)


@contextmanager
def batch_stats(module: nn.Module):
    """Run the BatchNorm layers of ``module`` on batch statistics (train
    mode) inside the block, leaving their running statistics untouched."""
    bns = [
        m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]
    saved = [(m.training, m.momentum) for m in bns]
    for m in bns:
        m.train()
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, (training, momentum) in zip(bns, saved):
            m.train(training)
            m.momentum = momentum


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


PRESETS = {
    "light": dict(width=192, heads=3, depth=6),
    "wide": dict(width=384, heads=6, depth=6),
}


def add_model_args(parser) -> None:
    g = parser.add_argument_group(
        "model (--size light by default; the design doc's full sizes are "
        "--depth 12 --pred-depth 6 --pred-heads 16 --pred-dim-head 64 "
        "--pred-mlp 2048 --proj-hidden 2048)"
    )
    g.add_argument(
        "--size",
        choices=tuple(PRESETS),
        default="light",
        help="encoder preset: light = 192 wide, 3 heads, 6 deep (378 rungs in a "
        "dense ladder); wide = 384 wide, 6 heads, 6 deep (756 rungs); --width, "
        "--heads and --depth override it",
    )
    g.add_argument("--width", type=int, default=None, help="encoder width")
    g.add_argument("--depth", type=int, default=None, help="encoder blocks")
    g.add_argument("--heads", type=int, default=None)
    g.add_argument("--mlp-ratio", type=float, default=4.0, help="encoder MLP width")
    g.add_argument("--p-rope", type=float, default=0.75)
    g.add_argument(
        "--time-frac",
        type=float,
        default=0.875,
        help="fraction of every head's channels rotated by time, the rest by "
        "wavelength (ZTF has two bands, so 8 of 64 channels suffice for them)",
    )
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


MODEL_KEYS = (
    "size",
    "width",
    "depth",
    "heads",
    "mlp_ratio",
    "p_rope",
    "time_frac",
    "attention",
    "no_err_channel",
)


def resolve_model_args(args) -> argparse.Namespace:
    """Fill ``--width --heads --depth`` from ``--size`` where not given."""
    for k, v in PRESETS[args.size].items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


def encoder_config(args) -> dict:
    return dict(
        d_model=args.width,
        nhead=args.heads,
        depth=args.depth,
        attention=args.attention,
        mlp_ratio=getattr(args, "mlp_ratio", 4.0),
    )


def rope_layouts(args, ladder: Ladder):
    """The rotary layout (shared) or per-layer layouts of the backbone: the
    time block of ``--time-frac`` of every head carries the ladder, flat or
    per head, and the wavelength block takes the rest."""
    check_ladder(ladder, rope_geometry(args))
    head_dim = args.width // args.heads

    def lay(ts):
        return rope_layout(
            head_dim, 2, time_frac=args.time_frac, p=args.p_rope, time_timescales=ts
        )

    if ladder.layers > 1:
        return [lay(ts) for ts in ladder.timescales]
    return lay(ladder.timescales)


def build_backbone(args, ladder: Ladder, n_channels: int) -> RoMAE:
    return RoMAE(
        encoder=encoder_config(args),
        n_channels=n_channels,
        n_axes=2,
        rope=rope_layouts(args, ladder),
    )


def flat_layout(backbone: RoMAE) -> list[dict]:
    """The backbone's rotary layout with per-head and per-layer ladders
    collapsed to one shared ladder, for decoders with their own head count."""
    return collapse_layout(backbone.rope_layout)


def build_world_model(
    args,
    backbone: RoMAE,
    n_frames: int,
    fused: bool = True,
    projector: nn.Module | None = None,
) -> LeWorldModel:
    """The stage-2 world model of ``args`` over ``backbone``; ``projector``
    replaces the default ``lewm_mlp(d, args.proj_hidden)`` (an
    ``nn.Identity()`` makes the latent the backbone feature itself, which is
    what a frozen backbone wants)."""
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
        projector=lewm_mlp(d, args.proj_hidden) if projector is None else projector,
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
    """Everything a stage-1 (autoencoder, ``mae.pt``) or stage-2 (world
    model, ``wm.pt``) checkpoint records besides the weights."""

    spec: TokenSpec
    cfg: FrameConfig
    ladder: Ladder
    classes: tuple[str, ...]
    args: dict
    step: int
    metrics: dict | None = None


def wm_state(
    model: LeWorldModel, spec, cfg, ladder, classes, args, step, metrics=None, **extra
):
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
        **extra,
    )


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
        ),
        spec=spec.to_dict(),
        frames=asdict(cfg),
        ladder=ladder.to_dict(),
        classes=list(classes),
        metrics=metrics,
    )


def _read_ckpt(path) -> dict:
    """A checkpoint dict from its path, or the dict itself when already
    loaded (the loaders below take either, so a file is read once)."""
    if isinstance(path, dict):
        return path
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_kind(ckpt: dict) -> str:
    """``"mae"`` for a stage-1 autoencoder checkpoint (:func:`mae_state`, or
    a bottleneck autoencoder with ``kind == "bottleneck"``), ``"wm"`` for a
    stage-2 world-model checkpoint (:func:`wm_state`)."""
    if "mae" in ckpt or ckpt.get("kind") == "bottleneck":
        return "mae"
    return "wm"


def _meta_from(ckpt: dict) -> WMMeta:
    return WMMeta(
        TokenSpec.from_dict(ckpt["spec"]),
        FrameConfig(**ckpt["frames"]),
        Ladder.from_dict(ckpt["ladder"]),
        tuple(ckpt["classes"]),
        ckpt["args"],
        int(ckpt["step"]),
        ckpt.get("metrics"),
    )


def load_mae(path, device="cpu") -> tuple[nn.Module, WMMeta]:
    """``(model in eval mode, meta)`` from a stage-1 ``mae.pt`` (or its
    ``last.pt``, or the loaded dict); ``model.backbone()`` is the pretrained
    encoder. A checkpoint with ``kind == "bottleneck"`` rebuilds a
    :class:`project.bottleneck.BottleneckAE` instead of a
    :class:`~romae_lc.RoMAEForPreTraining`; both expose ``backbone(pool)``,
    ``encode``, ``use_cls``, ``rope_layout``, ``embed_dim`` and ``hparams``."""
    ckpt = _read_ckpt(path)
    if ckpt.get("kind") == "bottleneck":
        from project.bottleneck import BottleneckAE  # written by another agent

        model = BottleneckAE.from_checkpoint(ckpt)
    else:
        model = RoMAEForPreTraining(**ckpt["mae"], **ckpt["backbone"])
        model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), _meta_from(ckpt)


def save_atomic(state: dict, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_wm(path, device="cpu") -> tuple[LeWorldModel, WMMeta]:
    """``(model in eval mode, meta)`` from a stage-2 ``wm.pt`` (or
    ``last.pt``, or the loaded dict), with :func:`fused_encode` installed and
    ``model.encoder_only = False``.

    A stage-1 autoencoder checkpoint (:func:`checkpoint_kind` ``"mae"``) is
    accepted too: the result wraps its frozen CLS encoder
    (``mae.backbone("cls")``) in a :class:`~romae_lc.LeWorldModel` with an
    ``nn.Identity`` projector, ``history=3``, ``lamb=0`` and the default,
    untrained predictor and ``pred_proj``, and ``model.encoder_only = True``;
    ``meta.args`` is the autoencoder's args plus ``encoder_frozen=True`` and
    ``encoder_kind="mae"``. Its ``encode(frames)`` is then the stage-1 latent
    (the pooled CLS feature, what :class:`LatentEncoder` returns), so the
    stage-3 decoder, the gates and the probes run on frozen autoencoder
    latents through this one path; its ``predict()``, ``rollout()`` and
    ``surprise()`` are untrained noise, so a caller that needs a predictor
    must check ``encoder_only``. Such a model re-saves through
    :func:`wm_state` (``proj_hidden`` 0 stands for the Identity projector)
    as an ordinary stage-2 checkpoint whose predictor is untrained."""
    ckpt = _read_ckpt(path)
    if checkpoint_kind(ckpt) == "mae":
        mae, meta = load_mae(ckpt, device)
        model = LeWorldModel(
            mae.backbone("cls"), projector=nn.Identity(), history=3, lamb=0.0
        )
        use_fused_encode(model)
        model.encoder_only = True
        meta.args = dict(meta.args, encoder_frozen=True, encoder_kind="mae")
        return model.to(device).eval(), meta
    backbone = RoMAE(**ckpt["backbone"])
    model = LeWorldModel.from_hparams(backbone, ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    use_fused_encode(model)
    model.encoder_only = False
    return model.to(device).eval(), _meta_from(ckpt)


class LatentEncoder(nn.Module):
    """One frozen encoder over the two kinds of checkpoint the later stages
    read: a stage-1 autoencoder (``mae.pt``, ``kind == "mae"``: the latent is
    the pooled CLS feature of the RoMAE encoder, :class:`PooledEncoder`) or
    a stage-2 world model (``wm.pt``, ``kind == "wm"``: the post-projector
    latent of :meth:`~romae_lc.LeWorldModel.encode`). ``encode(frames)``
    returns ``[B, T, dim]`` float latents in one backbone call over the
    ``T`` frames; they feed the stage-2 predictor (cached), the stage-3
    decoder, the gates and the probes, so the module is always in eval mode
    with every parameter frozen (``train()`` keeps eval) and it is never
    saved. Attributes: ``kind``, ``dim`` (latent width), ``history`` (the
    world model's predictor context, 3 for an autoencoder), ``backbone``
    (the ``(values, positions, pad_mask) -> [B, dim]`` encoder module, with
    ``embed_dim`` and ``rope_layout`` for a decoder built on top) and
    ``model`` (the loaded checkpoint model).

        enc, meta = load_encoder("project/runs/mae_w250/mae.pt", device)
        z = enc.encode(batch["frames"])  # [B, T, 192]
    """

    KINDS = ("mae", "wm")

    def __init__(self, model: nn.Module, kind: str):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        self.kind, self.model = kind, model
        if kind == "wm":
            self.backbone = model.backbone
            self.history = int(model.history)
        else:
            self.backbone = PooledEncoder(model)
            self.history = 3
        self.dim = int(model.embed_dim)
        self.model.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "LatentEncoder":
        """A frozen encoder stays in eval mode (BatchNorm running statistics,
        no dropout) whatever the caller asks."""
        return super().train(False)

    def encode(self, frames: Sequence, project: bool | None = None) -> torch.Tensor:
        """``[B, T, dim]`` latents of ``T`` frames. ``project`` applies to a
        world model only (``False`` returns its pre-projector backbone
        features); an autoencoder has no projector and ignores it."""
        if self.kind == "wm":
            z = self.model.encode(frames, project=True if project is None else project)
        else:
            z = fused_features(self.backbone, frames)
        return z.float()


def load_encoder(path, device="cpu") -> tuple[LatentEncoder, WMMeta]:
    """``(frozen LatentEncoder, meta)`` from a stage-1 ``mae.pt`` or a
    stage-2 ``wm.pt`` (:func:`checkpoint_kind` tells them apart), via
    :func:`load_mae` or :func:`load_wm`."""
    ckpt = _read_ckpt(path)
    kind = checkpoint_kind(ckpt)
    model, meta = (load_mae if kind == "mae" else load_wm)(ckpt, device)
    return LatentEncoder(model, kind), meta


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
        pred_va, pred_tr = head(x_va).argmax(1).numpy(), head(x_tr).argmax(1).numpy()
    y_tr, y_va = y_tr.numpy(), y_va.numpy()
    majority = np.bincount(y_tr).argmax()
    return dict(
        acc=float((pred_va == y_va).mean()),
        train_acc=float((pred_tr == y_tr).mean()),
        macro_f1=macro_f1(y_va, pred_va),
        balanced_acc=balanced_accuracy(y_va, pred_va),
        majority_acc=float((y_va == majority).mean()),
        n_val=int(y_va.size),
    )


def macro_f1(y_true, y_pred) -> float:
    """Unweighted mean F1 over the classes present in ``y_true`` or
    ``y_pred`` (a class never predicted and never true does not count)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1 = []
    for c in np.unique(np.concatenate([y_true, y_pred])):
        tp = np.sum((y_pred == c) & (y_true == c))
        denom = (
            2 * tp
            + np.sum((y_pred == c) & (y_true != c))
            + np.sum((y_pred != c) & (y_true == c))
        )
        f1.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1))


def balanced_accuracy(y_true, y_pred) -> float:
    """Mean recall over the classes present in ``y_true``."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(
        np.mean([np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)])
    )


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


def probe_metrics(
    z_tr, r_tr, z_va, r_va, n_classes, min_train: int = 30, min_val: int = 10
) -> dict:
    """The probe metrics of features ``z`` of records ``r``: the linear
    class probe (accuracy, macro F1, balanced accuracy, the majority-class
    accuracy it must beat) and ridge R2 on log10 period pooled (``r2``),
    within superclasses (``r2_within``: the target is log10 period minus its
    superclass mean, so class-level period differences do not count) and
    per superclass with at least ``min_train`` / ``min_val`` records
    (``r2_by_superclass``, with every superclass's validation count in
    ``n_by_superclass``). Records of the :data:`ANOMALY_CLASSES` are left out
    of the class probe (fit and score; ``n_class_excluded`` counts them) and
    kept in every period metric."""
    y_tr, y_va = (np.array([r.label for r in rs]) for rs in (r_tr, r_va))
    p_tr, p_va = (np.log10([r.period for r in rs]) for rs in (r_tr, r_va))
    k_tr, k_va = (
        np.array([fine_class(r) not in ANOMALY_CLASSES for r in rs], dtype=bool)
        for rs in (r_tr, r_va)
    )
    out = linear_probe(z_tr[k_tr], y_tr[k_tr], z_va[k_va], y_va[k_va], n_classes)
    out["n_class_excluded"] = int((~k_tr).sum() + (~k_va).sum())
    out["r2"] = ridge_r2(z_tr, p_tr, z_va, p_va)
    g_tr, g_va = (np.array([superclass(r) for r in rs]) for rs in (r_tr, r_va))
    means = {g: float(p_tr[g_tr == g].mean()) for g in set(g_tr.tolist())}
    glob = float(p_tr.mean())
    c_tr = p_tr - np.array([means[g] for g in g_tr])
    c_va = p_va - np.array([means.get(g, glob) for g in g_va])
    out["r2_within"] = (
        ridge_r2(z_tr, c_tr, z_va, c_va) if c_va.std() > 0 else float("nan")
    )
    by, n_by = {}, {}
    for g in sorted(set(g_va.tolist())):
        m_tr, m_va = g_tr == g, g_va == g
        n_by[g] = int(m_va.sum())
        if m_tr.sum() >= min_train and m_va.sum() >= min_val and p_va[m_va].std() > 0:
            by[g] = ridge_r2(z_tr[m_tr], p_tr[m_tr], z_va[m_va], p_va[m_va])
    out["r2_by_superclass"], out["n_by_superclass"] = by, n_by
    return out


def probe(
    model, train, val, cfg, spec, batch_size, device, n_classes, project=False, seed=0
):
    """:func:`probe_metrics` on the frozen features of ``model`` (mean over
    the windows of one deterministic sequence per record)."""
    z_tr, i_tr = embed_records(
        model, train, cfg, spec, batch_size, device, project, seed
    )
    z_va, i_va = embed_records(model, val, cfg, spec, batch_size, device, project, seed)
    return probe_metrics(
        z_tr, [train[i] for i in i_tr], z_va, [val[i] for i in i_va], n_classes
    )


HEADLINE = ("ROT", "RR", "ECL", "CEP", "DSCT", "LPV")


def describe_by_class(by: dict, n_by: dict | None = None) -> str:
    """``ROT 0.12/112 RR 0.25/422 ...``: the per-superclass period R2 with the
    validation count, headline classes first, then by count."""
    n_by = n_by or {}
    order = [g for g in HEADLINE if g in by]
    order += sorted((g for g in by if g not in HEADLINE), key=lambda g: -n_by.get(g, 0))
    return " ".join(
        f"{g} {by[g]:.2f}" + (f"/{n_by[g]}" if g in n_by else "") for g in order
    )


def describe_probe(pr: dict) -> str:
    return (
        f"probe acc {pr['acc']:.3f} f1 {pr['macro_f1']:.3f} bal {pr['balanced_acc']:.3f} "
        f"(majority {pr['majority_acc']:.3f}, train acc {pr['train_acc']:.3f}) | logP R2 "
        f"within {pr['r2_within']:.3f} pooled {pr['r2']:.3f} by class "
        f"{describe_by_class(pr['r2_by_superclass'], pr['n_by_superclass'])}"
    )


def hand_features(record: Record, bands: Sequence[int]) -> np.ndarray:
    """Statistics a linear probe can read off a standardised record without
    any model: log points, span, median cadence, band count, and per band
    (zeros with a 0 flag for a band with under three points) a flag, log
    points, scatter, skewness, excess kurtosis, MAD over std, the fractions
    of points beyond +1 and -1, log median error and range over std. The
    per-band standardisation of :func:`~romae_lc.normalize` has already
    removed mean brightness and colour, so these are shape, scatter and
    noise statistics only, which is what the encoder sees too."""
    t = record.t.astype(np.float64)
    span = float(t.max() - t.min()) if record.n > 1 else 0.0
    gaps = np.diff(np.sort(t))
    gaps = gaps[gaps > 0]
    f = [
        np.log10(max(record.n, 1)),
        span / 1000.0,
        np.log10(np.median(gaps)) if gaps.size else 0.0,
        float(len(record.bands)),
    ]
    for b in bands:
        m = record.band == b
        n = int(m.sum())
        if n < 3:
            f += [0.0] * 10
            continue
        y, e = record.y[m].astype(np.float64), record.err[m].astype(np.float64)
        sd = float(y.std()) + 1e-6
        z = (y - y.mean()) / sd
        f += [
            1.0,
            np.log10(n),
            sd,
            float(np.mean(z**3)),
            float(np.mean(z**4) - 3.0),
            float(np.median(np.abs(y - np.median(y)))) / sd,
            float(np.mean(y > 1.0)),
            float(np.mean(y < -1.0)),
            np.log10(float(np.median(e)) + 1e-6),
            float(np.ptp(y)) / sd,
        ]
    return np.array(f, dtype=np.float64)


def baseline_probe(train, val, bands: Sequence[int], n_classes: int) -> dict:
    """:func:`probe_metrics` on :func:`hand_features`: the bar an encoder's
    probe has to clear."""
    bands = sorted(bands)
    x_tr = np.stack([hand_features(r, bands) for r in train])
    x_va = np.stack([hand_features(r, bands) for r in val])
    return probe_metrics(x_tr, train, x_va, val, n_classes)


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
