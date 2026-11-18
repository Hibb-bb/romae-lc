"""The masks of the pretraining: which tokens of a window are hidden.

**Hide stretches of time, not single points** (:func:`block_mask`). With
random points hidden, every hidden point has close neighbours on both sides
and can be filled in from them. With a whole stretch hidden there are no
close neighbours: the pattern has to be carried across the gap, and that
needs the period and the phase. A few stretches are placed at random times
and grow until the wanted share of the points is hidden, so the share is
exact and the length of a stretch follows from it (half of a 250 d window in
one stretch is a 125 d gap, in six stretches about 20 d each).

:func:`make_mask` picks between random points, stretches, and one stretch
plus random points (``blockplus``). :func:`phase_band_mask` hides a band of
phases of the catalogue period (training only). :func:`forecast_tail_mask`
also hides the last days of a window (``--forecast-tail``).
"""

from __future__ import annotations

from typing import Sequence

import torch

from romae_lc.model import gen_mask


MASK_MODES = ("random", "block", "blockplus", "phaseband")  # phaseband: phase_band_mask (needs the period)


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
    generator: torch.Generator | None = None,
    position: str = "random",
    block_share: float = 0.5,
) -> torch.Tensor:
    """The mask of one batch: ``random`` points, ``block`` stretches, or
    ``blockplus`` (every row hides a stretch holding ``block_share`` of the
    hidden count and random points for the rest: a hidden window plus the
    easy signal). Every row hides the same number of tokens in every mode.
    ``position`` places the stretches (``random`` | ``last``). The
    ``phaseband`` mode has its own function, :func:`phase_band_mask`."""
    if mode not in ("random", "block", "blockplus"):
        raise ValueError(f"mode must be random, block or blockplus, got {mode!r}")
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
    return block_mask(t, pad_mask, mask_ratio, n_blocks, generator, position)


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


def phase_band_mask(
    t_days: torch.Tensor,
    pad_mask: torch.Tensor,
    period: torch.Tensor,
    mask_ratio: float,
    band: Sequence[float] = (0.1, 0.3),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Hide a band of PHASE, then random points up to ``mask_ratio``.

    Every row hides the tokens whose phase ``(t / P + u) mod 1`` falls below a width
    drawn from ``band`` (``u`` a random offset per row, so the band sits anywhere in the
    cycle), capped at the row's hidden count, then random real tokens up to
    ``ceil(n_real * mask_ratio)``. The hidden band can only be filled by folding the
    other cycles, so the encoder must carry the period and the phase. ``period [B]``
    in days (the training-time catalogue period, never an input); a row without a
    finite period gets random points only. Rows are equalised like ``gen_mask``."""
    b, n = pad_mask.shape
    dev = pad_mask.device
    real = ~pad_mask
    k = (real.sum(1).float() * mask_ratio).ceil().long()
    p = period.to(dev).double()
    ok = torch.isfinite(p) & (p > 0)
    u = torch.rand(b, generator=generator).to(dev).double()
    w = (band[0] + (band[1] - band[0]) * torch.rand(b, generator=generator)).to(dev).double()
    phi = torch.remainder(t_days.double() / torch.where(ok, p, torch.ones_like(p))[:, None] + u[:, None], 1.0)
    in_band = (phi < w[:, None]) & real & ok[:, None]
    # keep at most k band tokens (a random subset when the band is larger)
    s_band = torch.rand(b, n, generator=generator).to(dev).masked_fill(~in_band, 2.0)
    band_m = in_band & (s_band.argsort(1).argsort(1) < k[:, None])
    need = (k - band_m.sum(1)).clamp_min(0)
    score = torch.rand(b, n, generator=generator).to(dev).masked_fill(band_m | pad_mask, 2.0)
    extra = score.argsort(1).argsort(1) < need[:, None]
    mask = band_m | extra
    tot = (mask & real).sum(1)
    pad_extra = tot.max() - tot
    idx = torch.arange(n, device=dev)[None, :]
    return mask | (idx >= (n - pad_extra)[:, None])


def forecast_tail_mask(mask, t_days, pad, window, h_range, share=1.0):
    """``--forecast-tail``: hide every real token in the last ``H`` days of each window (``H`` uniform in
    ``h_range`` per row; a ``share`` of the rows), keeping the same hidden count in every row, as the JEPA
    forward needs. Per row the hidden set is the tail first, then the tokens ``mask`` already hid, then random
    others (padding first), up to ``k``, the largest union of tail and ``mask`` over the rows. Returns ``(mask, share of hidden real tokens
    that are tail)``."""
    b, n = mask.shape
    lo, hi = h_range
    h = lo + (hi - lo) * torch.rand(b, device=mask.device)
    on = torch.rand(b, device=mask.device) < share
    tail = (~pad) & (t_days > (float(window) - h)[:, None]) & on[:, None]
    k = min(int((mask | tail).sum(1).max()), n - 1)
    # priority: the tail, then what mask hid, then padding (filler that costs no real token), then other real tokens
    score = torch.rand(b, n, device=mask.device) + 1.0 * pad.float() + 2.0 * mask.float() + 4.0 * tail.float()
    idx = score.topk(k, dim=1).indices
    out = torch.zeros_like(mask).scatter_(1, idx, True)
    hid_real = (out & ~pad).sum().clamp(min=1)
    return out, float((out & tail).sum() / hid_real)
