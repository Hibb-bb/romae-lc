"""Global and local views of a window, for the no-EMA recipe (``--jepa-no-ema``).

:func:`make_views` draws the keep masks: global views are the whole window
thinned, local views are a short stretch thinned, and every view may drop
one band. :func:`project.jepa.forward_views` encodes the views and computes
the loss.
"""

from __future__ import annotations

import torch


def pack(keep: torch.Tensor):
    """Indices that move the kept tokens of every row to the front (original order) and the padding of the
    packed rows: ``(order [b, k], pad [b, k])``."""
    cnt = keep.sum(1)
    k = max(int(cnt.max()), 1)
    order = torch.sort((~keep).to(torch.int8), dim=1, stable=True).indices[:, :k]
    pad = torch.arange(k, device=keep.device)[None] >= cnt[:, None]
    return order, pad


def take(values, positions, order):
    v = values.gather(1, order[..., None].expand(-1, -1, values.shape[-1]))
    p = positions.gather(2, order[:, None].expand(-1, positions.shape[1], -1))
    return v, p


def make_views(t_days, positions, pad, n_glob, n_loc, keep, local_span, band_drop, window, min_tokens=8):
    """Keep masks ``[(n_glob + n_loc), b, n]`` of the views of one window batch. Global views: the whole window,
    each real token kept with probability ``keep``. Local views: a stretch of ``local_span`` days around a random
    real token, thinned so the view holds about ``keep`` of the window's tokens (kept share of the stretch between
    0.3 and 0.9). Every view drops one band at random with probability ``band_drop``. A view with fewer than
    ``min_tokens`` tokens takes ``min_tokens`` real tokens instead: random ones for a global view, the nearest to
    the stretch's centre for a local view."""
    b, n = pad.shape
    dev = pad.device
    real = ~pad
    n_real = real.sum(1).float()
    out = []
    for v in range(n_glob + n_loc):
        r = torch.rand(b, n, device=dev)
        anchor_idx = torch.where(real, torch.rand(b, n, device=dev), torch.full((b, n), -1.0, device=dev)).argmax(1)
        anchor = t_days.gather(1, anchor_idx[:, None])
        if v < n_glob:
            kv = real & (r < keep)
            dist = torch.rand(b, n, device=dev)  # a sparse global view tops up with random tokens, not a clump
        else:
            L = local_span[0] + (local_span[1] - local_span[0]) * torch.rand(b, 1, device=dev)
            s = anchor - L * torch.rand(b, 1, device=dev)
            span = real & (t_days >= s) & (t_days < s + L)
            p_keep = (keep * n_real / span.sum(1).clamp(min=1)).clamp(0.3, 0.9)
            kv = span & (r < p_keep[:, None])
            dist = (t_days - (s + L / 2)).abs()
        drop = torch.rand(b, device=dev) < band_drop
        pick = torch.where(kv, torch.rand(b, n, device=dev), torch.full((b, n), -1.0, device=dev)).argmax(1)
        band_pick = positions[:, 1].gather(1, pick[:, None])
        kv = kv & ~(drop[:, None] & (positions[:, 1] == band_pick))
        few = kv.sum(1) < min_tokens
        if few.any():
            near = torch.where(real, dist, torch.full_like(dist, 1e9)).argsort(1).argsort(1) < min_tokens
            kv = torch.where(few[:, None], near & real, kv)
        out.append(kv)
    return torch.stack(out)
