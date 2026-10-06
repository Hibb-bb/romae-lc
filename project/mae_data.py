"""Two changes to what the stage-1 autoencoder is asked to do. Neither one
uses the period of a star.

**Hide stretches of time, not single points** (:func:`block_mask`). With
random points hidden, every hidden point has close neighbours on both sides
and can be filled in from them. With a whole stretch hidden there are no
close neighbours: the pattern has to be carried across the gap, and that
needs the period and the phase. A few stretches are placed at random times
and grow until the wanted share of the points is hidden, so the share is
exact and the length of a stretch follows from it (half of a 250 d window in
one stretch is a 125 d gap, in six stretches about 20 d each).

**Windows of many lengths** (:class:`MultiWindowDataset`). Every training
window draws its own length, log-uniform between two bounds, so one encoder
sees short and long windows of every star. A short window holds the detail
of a fast star, a long one holds several cycles of a slow star. The rotary
encoding works on time differences in days, so nothing else changes. At use
every star is encoded at the same few lengths and the latents are joined,
which needs no period either.
"""

from __future__ import annotations

import math
import warnings
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from romae_lc import FrameConfig, Record
from romae_lc.model import gen_mask

MASK_MODES = ("random", "block", "mix", "blockplus")


def block_mask(
    t: torch.Tensor,
    pad_mask: torch.Tensor,
    mask_ratio: float,
    n_blocks: Sequence[int] = (1, 6),
    generator: torch.Generator | None = None,
    position: str = "random",
) -> torch.Tensor:
    """A token mask made of stretches of time; True = hidden.

    ``t [B, N]`` are the time positions and ``pad_mask [B, N]`` is True on
    padding. Every row draws a number of stretches in ``n_blocks`` (both ends
    included), a centre for each inside the row's time span and a relative
    width in (0.5, 1.5). Every point gets the distance to its nearest centre
    in units of that stretch's width, and the ``ceil(n_real * mask_ratio)``
    points with the smallest distance are hidden: the stretches grow together
    until the share is reached. Rows are then made equal in count with
    trailing padding tokens, exactly like :func:`romae_lc.model.gen_mask`.
    ``position="last"`` puts one stretch at the end of the row's span (the
    last ``mask_ratio`` of the points: a hidden last window).
    """
    if position not in ("random", "last"):
        raise ValueError(f"position must be random|last, got {position!r}")
    if not 0 <= mask_ratio <= 1:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
    lo_b, hi_b = int(n_blocks[0]), int(n_blocks[-1])
    if not 1 <= lo_b <= hi_b:
        raise ValueError(f"n_blocks must satisfy 1 <= lo <= hi, got {tuple(n_blocks)}")
    b, n = pad_mask.shape
    dev = pad_mask.device
    k = ((~pad_mask).sum(1) * mask_ratio).ceil().long()
    t = t.float()
    real = ~pad_mask
    big = torch.finfo(t.dtype).max
    t_lo = torch.where(real, t, torch.full_like(t, big)).min(1).values
    t_hi = torch.where(real, t, torch.full_like(t, -big)).max(1).values
    t_lo = torch.where(real.any(1), t_lo, torch.zeros_like(t_lo))
    span = torch.where(real.any(1), (t_hi - t_lo).clamp_min(1e-6), torch.ones_like(t_lo))
    m = torch.randint(lo_b, hi_b + 1, (b,), generator=generator).to(dev)
    centre = t_lo[:, None] + span[:, None] * torch.rand(b, hi_b, generator=generator).to(dev)
    width = 0.5 + torch.rand(b, hi_b, generator=generator).to(dev)
    if position == "last":
        m = torch.ones_like(m)
        centre = t_hi[:, None].expand(b, hi_b).clone()
    used = torch.arange(hi_b, device=dev)[None] < m[:, None]
    dist = (t[:, :, None] - centre[:, None, :]).abs() / width[:, None, :]
    dist = dist.masked_fill(~used[:, None, :], float("inf")).min(-1).values
    # a tiny random term breaks ties between points at the same time
    dist = dist + 1e-6 * span[:, None] * torch.rand(b, n, generator=generator).to(dev)
    dist = dist.masked_fill(pad_mask, float("inf"))
    rank = dist.argsort(1).argsort(1)
    mask = rank < k[:, None]
    extra = k.max() - k
    idx = torch.arange(n, device=dev)[None, :]
    return mask | (idx >= (n - extra)[:, None])


def make_mask(
    t: torch.Tensor,
    pad_mask: torch.Tensor,
    mask_ratio: float,
    mode: str = "random",
    n_blocks: Sequence[int] = (1, 6),
    block_prob: float = 0.5,
    generator: torch.Generator | None = None,
    position: str = "random",
    block_share: float = 0.5,
) -> torch.Tensor:
    """The mask of one batch: ``random`` points, ``block`` stretches,
    ``mix`` (every row is a stretch row with probability ``block_prob``), or
    ``blockplus`` (every row hides a stretch holding ``block_share`` of the
    hidden count and random points for the rest: a hidden window plus the
    easy signal). Every row hides the same number of tokens in every mode,
    so the modes can be mixed inside a batch. ``position`` places the
    stretches (``random`` | ``last``)."""
    if mode not in MASK_MODES:
        raise ValueError(f"mode must be one of {MASK_MODES}, got {mode!r}")
    if mode == "random":
        return gen_mask(mask_ratio, pad_mask, generator)
    if mode == "blockplus":
        # one stretch holding block_share of the hidden count, random points for the rest;
        # the block's own count is per row, the random top-up keeps every row at ceil(n_real * ratio)
        blocks = block_mask(t, pad_mask, mask_ratio * block_share, (1, 1), generator, position)
        real = ~pad_mask
        k = (real.sum(1).float() * mask_ratio).ceil().long()
        need = (k - (blocks & real).sum(1)).clamp_min(0)
        score = torch.rand(pad_mask.shape, generator=generator).to(pad_mask.device)
        score = score.masked_fill(blocks | pad_mask, 2.0)  # already hidden or padding: never picked
        rank = score.argsort(1).argsort(1)
        extra = rank < need[:, None]
        mask = blocks | extra
        n = pad_mask.shape[1]
        tot = (mask & real).sum(1)
        pad_extra = tot.max() - tot  # equalise the rows with trailing padding, like gen_mask
        idx = torch.arange(n, device=pad_mask.device)[None, :]
        return mask | (idx >= (n - pad_extra)[:, None])
    blocks = block_mask(t, pad_mask, mask_ratio, n_blocks, generator, position)
    if mode == "block":
        return blocks
    points = gen_mask(mask_ratio, pad_mask, generator)
    pick = (torch.rand(pad_mask.shape[0], generator=generator) < block_prob).to(pad_mask.device)
    return torch.where(pick[:, None], blocks, points)


def longest_gap(t: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """The longest stretch of time without a visible point, per row, as a
    share of the row's time span: what the hidden points leave open. ``t [B,
    N]`` must be sorted in time within a row; ``visible [B, N]`` is True on
    the points the encoder sees."""
    out = torch.zeros(t.shape[0])
    for i in range(t.shape[0]):
        tv = t[i][visible[i]].float()
        if tv.numel() < 2:
            continue
        span = float(tv.max() - tv.min())
        if span > 0:
            out[i] = float(tv.sort().values.diff().max()) / span
    return out


class MultiWindowDataset(Dataset):
    """Frame sequences whose windows have random lengths.

    Every item is ``dict(frames, actions, label, index)`` like
    :class:`romae_lc.FrameDataset`, so the same collate works, but each of
    the ``cfg.n_frames`` windows draws its own length, log-uniform in
    ``window_range`` (days), and its own start inside the record. A window
    with fewer than ``cfg.min_tokens`` points is redrawn up to
    ``cfg.max_tries`` times (the fullest draw is kept), and one with more
    than ``cfg.max_tokens`` points is thinned at random. The windows of an
    item are not consecutive and ``actions`` is zero: this is data for the
    autoencoder, which looks at one window at a time. Times are counted from
    the window start.

    Records shorter than the shortest window are dropped; ``indices`` says
    which ones were kept, as in ``FrameDataset``.
    """

    def __init__(
        self,
        records: Sequence[Record],
        cfg: FrameConfig,
        window_range: Sequence[float],
        seed: int = 0,
        epoch_seed: bool = True,
    ):
        lo, hi = float(window_range[0]), float(window_range[-1])
        if not 0 < lo <= hi:
            raise ValueError(f"window_range must satisfy 0 < lo <= hi, got {tuple(window_range)}")
        spans = np.array([float(r.t.max() - r.t.min()) if r.n else 0.0 for r in records])
        keep = [i for i, s in enumerate(spans) if s >= lo]
        if len(keep) < len(records):
            warnings.warn(f"{len(records) - len(keep)} of {len(records)} records shorter than {lo:g} d dropped")
        if not keep:
            raise ValueError(f"no record spans {lo:g} d")
        self.records = [records[i] for i in keep]
        self.indices = keep
        self.cfg, self.window_range = cfg, (lo, hi)
        self.seed, self.epoch_seed = seed, epoch_seed
        self.calls = [0] * len(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def window(self, record: Record, rng: np.random.Generator):
        """One window of ``record``: ``(start, length, point indices)``."""
        cfg = self.cfg
        t = record.t.astype(np.float64)
        t_min, span = float(t.min()), float(t.max() - t.min())
        lo, hi = self.window_range
        best = None
        for _ in range(cfg.max_tries):
            w = min(math.exp(rng.uniform(math.log(lo), math.log(hi))), span)
            s = t_min + rng.uniform(0.0, span - w) if span > w else t_min
            idx = np.flatnonzero((t >= s) & (t < s + w))
            if best is None or len(idx) > len(best[2]):
                best = (s, w, idx)
            if len(idx) >= cfg.min_tokens:
                break
        s, w, idx = best
        if len(idx) > cfg.max_tokens:
            idx = np.sort(rng.choice(idx, size=cfg.max_tokens, replace=False))
        return s, w, idx

    def __getitem__(self, i: int) -> dict:
        j = self.indices[i]
        entropy = [self.seed, j]
        if self.epoch_seed:
            self.calls[i] += 1
            entropy += [self.calls[i], torch.initial_seed()]
        rng = np.random.default_rng(entropy)
        record = self.records[i]
        t = record.t.astype(np.float64)
        frames, lengths = [], []
        for _ in range(self.cfg.n_frames):
            s, w, idx = self.window(record, rng)
            frame = (
                (t[idx] - s).astype(np.float32),
                record.y[idx].astype(np.float32),
                record.band[idx].astype(np.int64),
            )
            if self.cfg.with_err:
                frame = frame + (record.err[idx].astype(np.float32),)
            frames.append(frame)
            lengths.append(w)
        actions = np.zeros((self.cfg.n_frames, 1), dtype=np.float32)
        return dict(frames=frames, actions=actions, label=record.label, index=j,
                    lengths=np.asarray(lengths, dtype=np.float32))  # fmt: skip
