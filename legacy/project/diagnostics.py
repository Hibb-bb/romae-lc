"""Stage-1 diagnostics: time-shuffle score, surprise along window grids.

``shuffle_score`` is the package's :func:`~romae_lc.time_shuffle_score` on
the first window of a deterministic sequence per record, standardised with
shared statistics so the batches are comparable (near 1 = the encoder is
blind to time). ``grid_surprise`` runs :meth:`~romae_lc.LeWorldModel.surprise`
on every window of one record (``frame_grid(fill=True)``), masking targets
with fewer than ``min_tokens`` points, and is what the injected-anomaly test
and the anomaly scores read.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from romae_lc import FrameDataset, time_shuffle_score

from project.common import grid_item, grid_starts, subset, superclass


@torch.no_grad()
def shuffle_score(
    model, records, cfg, spec, device, n=512, batch_size=64, seed=0
) -> float:
    recs = subset(records, n, seed)
    ds = FrameDataset(recs, cfg, seed=seed, epoch_seed=False)
    loader = DataLoader(ds, batch_size, shuffle=False, collate_fn=spec.collate())
    backbone = model.backbone
    was = backbone.training
    backbone.eval()
    toks = [b["frames"][0].to(device) for b in loader]
    z = torch.cat([backbone(*t).float() for t in toks])
    stats = (z.mean(0), z.std(0, unbiased=False) + 1e-6)
    gen = torch.Generator().manual_seed(seed)
    scores = [
        time_shuffle_score(backbone, t, generator=gen, stats=stats).cpu() for t in toks
    ]
    backbone.train(was)
    return float(torch.cat(scores).mean())


@torch.no_grad()
def grid_surprise(
    model,
    record,
    cfg,
    spec,
    device,
    advance=None,
    start=None,
    cap=None,
    seed=0,
    index=0,
) -> dict:
    """Per-window surprise of one record along the fill grid.

    Returns ``dict(scores [K-1] (NaN where the target window has fewer than
    cfg.min_tokens points), n_tokens [K], starts [K] (days), z [K, D])``;
    ``scores[k - 1]`` is the error of predicting window ``k``.
    """
    cap = cfg.max_tokens if cap is None else cap
    item = grid_item(record, cfg, index, advance, start, cap, seed)
    batch = spec.collate()([item])
    frames = [f.to(device) for f in batch["frames"]]
    n_tok = batch["n_tokens"][0].numpy()
    was = model.training
    model.eval()
    z = model.encode(frames).float().cpu().numpy()[0]
    if len(frames) >= 2:
        s = model.surprise(frames, batch["actions"].to(device))[0].float().cpu().numpy()
        s[n_tok[1:] < cfg.min_tokens] = np.nan
    else:
        s = np.zeros(0, dtype=np.float32)
    model.train(was)
    starts = grid_starts(record, cfg, advance, start)
    return dict(scores=s, n_tokens=n_tok, starts=starts, z=z)


@torch.no_grad()
def surprise_summary(
    model, records, cfg, spec, device, n=128, seed=0, cap=None
) -> dict:
    """Mean surprise per object over the valid windows, pooled and per
    superclass, on a seeded subset of ``records``."""
    per_obj, groups, n_valid, n_win = [], [], 0, 0
    for i, r in enumerate(subset(records, n, seed)):
        s = grid_surprise(model, r, cfg, spec, device, cap=cap, seed=seed, index=i)[
            "scores"
        ]
        n_win += s.size
        ok = np.isfinite(s)
        n_valid += int(ok.sum())
        if ok.any():
            per_obj.append(float(s[ok].mean()))
            groups.append(superclass(r))
    per_obj, groups = np.array(per_obj), np.array(groups)
    per_class = {
        str(g): float(per_obj[groups == g].mean()) for g in sorted(set(groups.tolist()))
    }
    return dict(
        mean=float(per_obj.mean()) if per_obj.size else float("nan"),
        median=float(np.median(per_obj)) if per_obj.size else float("nan"),
        n_objects=int(per_obj.size),
        frac_valid=n_valid / max(n_win, 1),
        per_superclass=per_class,
    )
