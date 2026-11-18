"""Records -> windows -> tokens.

A window is a stretch of ``--window`` days of one record. Windows are cut by
:class:`~romae_lc.FrameConfig` with ``with_err=True``, so the per-point error
rides along. :func:`err_channel` turns the error into the second token
channel ``(log sigma - mu) / sd`` next to the magnitude. :class:`TokenSpec`
holds everything needed to turn windows into tokens; it is saved in the
checkpoint.

Two ways to cut windows:

- training: random frame sequences (:func:`frame_loader`, fused into one
  batch by :func:`fuse_frames`);
- evaluation: a fixed grid of windows over every record
  (:func:`object_windows`, :class:`WindowDataset`, :func:`make_collate`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from romae_lc import FrameConfig, FrameDataset, Record, Tokens, collate_frames, frame_grid, tokenize


# ------------------------------------------------------------------ error channel and tokens


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


# ------------------------------------------------------------------ frame sequences (training)


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


# ------------------------------------------------------------------ the grid of windows (evaluation)


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


def object_windows(
    record: Record, cfg: FrameConfig, stride: float, cap: int | None, seed: int, index: int
):
    """The grid windows of one record as the cache stores them: ``(starts
    float64 [K], frames, n_tokens int [K])`` with ``frames`` the ``K``
    ``(t, y, band, err)`` windows after the deterministic cap (rng seeded
    from ``(seed, index)``) and
    ``n_tokens`` their point counts before it."""
    starts = grid_starts(record, cfg, advance=stride)
    frames, _ = frame_grid(record, cfg, advance=stride, fill=True)
    if len(frames) != len(starts):
        raise AssertionError(
            f"frame_grid gave {len(frames)} windows, grid_starts {len(starts)}"
        )
    n_tokens = [len(f[0]) for f in frames]
    if cap is not None:
        rng = np.random.default_rng([seed, index])
        frames = [subsample_frame(f, cap, rng) for f in frames]
    return starts, frames, n_tokens


class WindowDataset(Dataset):
    """Every window of every object of every split as one flat index. An
    item is ``(split_i, obj_i, win)`` resolved lazily: the object's grid is
    cut once and kept while consecutive items address it (the sampler is
    sequential, so a worker's batch mostly hits one or two objects)."""

    def __init__(
        self,
        splits: list[tuple[str, list[Record], list[int]]],
        cfg,
        stride,
        cap,
        seed,
    ):
        self.splits, self.cfg, self.stride, self.cap, self.seed = (
            splits,
            cfg,
            stride,
            cap,
            seed,
        )
        self.counts = []  # windows per object, per split
        self.offsets = []  # first flat index of every object, per split
        total = 0
        for _, records, _ in splits:
            n = np.array(
                [len(grid_starts(r, cfg, advance=stride)) for r in records],
                dtype=np.int64,
            )
            self.counts.append(n)
            self.offsets.append(total + np.concatenate([[0], np.cumsum(n)]))
            total += int(n.sum())
        self.total = total
        self._key, self._cached = None, None

    def __len__(self) -> int:
        return self.total

    def locate(self, i: int) -> tuple[int, int, int]:
        """``(split_i, obj_i, win)`` of flat index ``i``."""
        for s, off in enumerate(self.offsets):
            if i < off[-1]:
                o = int(np.searchsorted(off, i, side="right") - 1)
                return s, o, int(i - off[o])
        raise IndexError(i)

    def windows(self, s: int, o: int):
        if self._key != (s, o):
            _, records, indices = self.splits[s]
            self._cached = object_windows(
                records[o], self.cfg, self.stride, self.cap, self.seed, indices[o]
            )
            self._key = (s, o)
        return self._cached

    def __getitem__(self, i: int) -> dict:
        widx = int(i)
        s, o, w = self.locate(widx)
        starts, frames, n_tokens = self.windows(s, o)
        frame = frames[w]
        return dict(
            widx=widx,  # the flat window index (the cache row)
            split=s,
            obj=self.splits[s][2][o],  # the data[split] position, not the subset's
            win=w,
            start=float(starts[w]),
            n_tokens=int(n_tokens[w]),
            frame=frame,
        )


def make_collate(spec: TokenSpec):
    """Tokenise a list of single windows into one padded batch."""

    def collate(items: list[dict]) -> dict:
        return dict(
            tokens=spec.tokens([it["frame"] for it in items]),
            widx=torch.tensor([it["widx"] for it in items], dtype=torch.int64),
            split=torch.tensor([it["split"] for it in items], dtype=torch.int64),
            obj=torch.tensor([it["obj"] for it in items], dtype=torch.int64),
            win=torch.tensor([it["win"] for it in items], dtype=torch.int64),
            start=torch.tensor([it["start"] for it in items], dtype=torch.float32),
            n_tokens=torch.tensor([it["n_tokens"] for it in items], dtype=torch.int64),
        )

    return collate
