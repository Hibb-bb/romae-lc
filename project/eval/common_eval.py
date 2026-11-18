"""Helpers shared by the evaluation scripts. No command line here.

- device: ``eval_device``
- windows of a list of stars: ``window_dataset``, ``window_loader``
- the spectral layer of an encoder: ``spectral_layers``
- baselines and scores: ``fourier_forecast``, ``zstats``, ``period_hits``
- the star tables of a latent cache (train, validation, test): ``star_tables``
- small things: ``run_name``, ``write_json``, ``git_commit``
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from project.data import data_args_from, load_data
from project.windows import WindowDataset, make_collate

#: The data root the per-token and forecast jobs passed by hand (ZTFxPC).
ZTFXPC = "/projects/bfrf/data/PC_matches/ZTFxPC"
#: Period tolerances of the hit rates: within 10 %, 1 %, 0.1 %, 0.01 %.
TOL = (0.1, 0.01, 0.001, 1e-4)


def eval_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_name(ckpt) -> str:
    """``project/runs/NAME/mae.pt`` -> ``NAME``."""
    return os.path.basename(os.path.dirname(str(ckpt)))


def window_dataset(split: str, recs: list, cfg, stride: float, index=None) -> WindowDataset:
    """Every window of ``recs`` (one split) on the checkpoint's grid; ``stride`` in window lengths."""
    index = list(range(len(recs))) if index is None else list(index)
    return WindowDataset([(split, recs, index)], cfg, stride, cfg.max_tokens, 0)


def window_loader(ds, spec, batch_size: int) -> DataLoader:
    return DataLoader(ds, batch_size, shuffle=False, num_workers=0, collate_fn=make_collate(spec))


def spectral_layers(module, ran: bool = False) -> list:
    """The spectral layers inside ``module``. ``ran=True``: only those that ran in the last forward
    (a JEPA checkpoint holds two copies of the encoder, so one of them may be idle)."""
    layers = [m for m in module.modules() if type(m).__name__ == "SpectralLayer"]
    return [m for m in layers if getattr(m, "last_logits", None) is not None] if ran else layers


def fourier_forecast(t, m, sig, f, t_new, K):
    """Weighted least-squares fit of a constant and ``K`` harmonics of frequency ``f`` to (t, m, sig),
    evaluated at ``t_new``."""

    def design(tt):
        cols = [np.ones_like(tt)]
        for k in range(1, K + 1):
            cols += [np.cos(2 * np.pi * k * f * tt), np.sin(2 * np.pi * k * f * tt)]
        return np.stack(cols, 1)

    w = 1.0 / np.clip(sig, 1e-6, None)
    coef, *_ = np.linalg.lstsq(design(t) * w[:, None], m * w, rcond=None)
    return design(t_new) @ coef


def zstats(z):
    """Summary of z-scores ``(prediction - truth) / sigma``."""
    az = np.abs(z)
    return dict(n=int(len(z)), median_abs_z=float(np.median(az)) if len(z) else float("nan"),
                within_1=float(np.mean(az < 1)) if len(z) else float("nan"), within_2=float(np.mean(az < 2)) if len(z) else float("nan"),
                mean_z=float(np.mean(z)) if len(z) else float("nan"))


def period_hits(p_hat, p, tols=TOL):
    """Share of stars with the period within each tolerance. Returns (n, strict, alias): ``alias`` also
    accepts the double and the half period. Stars without an estimate or a true period are left out."""
    ok = np.isfinite(p) & np.isfinite(p_hat) & (p > 0)
    r = p_hat[ok] / p[ok]
    rel = np.abs(r - 1)
    rel_a = np.minimum(rel, np.minimum(np.abs(r / 2 - 1), np.abs(2 * r - 1)))
    return int(ok.sum()), [float(np.mean(rel < t)) for t in tols], [float(np.mean(rel_a < t)) for t in tols]


def star_tables(latents, test_latents, over=None):
    """Load a train + validation cache and a test cache (``project.eval.encode``) and build one star
    table per split (``project.period_probe.object_table``: window rows, class, log period, features).

    ``over``: a Namespace whose non-None data arguments replace the checkpoint's.
    Returns ``(cache, test cache, tables)``; ``cache["meta"]`` names the checkpoint.
    """
    from project.period_probe import object_table

    cache = torch.load(latents, map_location="cpu", weights_only=False)
    tcache = torch.load(test_latents, map_location="cpu", weights_only=False)
    meta = cache["meta"]
    ckpt = torch.load(meta["ckpt"], map_location="cpu", weights_only=False)
    data = load_data(data_args_from(ckpt["args"], over), splits=("train", "validation", "test"))
    bands = sorted({int(b) for r in data["train"][:200] for b in np.unique(r.band)})
    caches = {"train": cache, "validation": cache, "test": tcache}
    tables = {s: object_table(caches[s], s, data[s], bands, int(meta["min_tokens"])) for s in ("train", "validation", "test")}
    return cache, tcache, tables


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def git_commit() -> str | None:
    """The commit of the code that runs, or None outside a git checkout."""
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None
