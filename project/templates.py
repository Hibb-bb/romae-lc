"""Smooth targets: the Fourier fit of every window on a given period, in
place of the noisy observations, as the reconstruction target.

For every fused window row and band, a wave of ``harmonics`` harmonics on
the row's period is fitted to ALL the window's points (visible and hidden)
by error-weighted least squares, in one batched solve on the GPU. The fit's
value at every token is the smooth target; its adjusted R2 per band gates
it: where the wave does not fit (irregular stars, sparse bands) the target
falls back to the observation. This is the "fitted line" of the fold
figures, computed per window.

The period comes from the catalogue (teacher stage) or from a column of the
period table (``p_model_cands``: the model's own, label-free).
"""

from __future__ import annotations

import csv
import math

import numpy as np
import torch


def period_lookup(path: str, column: str) -> dict:
    """``{ztf_id: period}`` from ``periods.csv`` of ``project.eval.period_table``."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                v = float(row[column])
            except (KeyError, ValueError):
                continue
            if np.isfinite(v) and v > 0:
                out[str(row["ztf_id"])] = v
    return out


def template_periods(records, lookup: dict | None) -> torch.Tensor:
    """One period per record: the lookup's value by ztf id, else the catalogue's."""
    out = []
    for r in records:
        p = lookup.get(str(r.meta.get("id"))) if lookup else None
        out.append(float(p if p else (r.period or float("nan"))))
    return torch.tensor(out, dtype=torch.float64)


@torch.no_grad()
def window_templates(y, w, t_days, band, period, real, harmonics: int = 6, min_points: int = 20, ridge: float = 1e-6):
    """``(template [B, N], r2 [B, N])``: the weighted Fourier fit of every
    row and band at every token, and the adjusted R2 of that band's fit
    broadcast to its tokens (NaN where the band has fewer than
    ``min_points`` real points or the row has no period).

    Args:
        y, w: ``[B, N]`` values and weights (``1 / sigma^2``), float.
        t_days: ``[B, N]`` token times in days (any common origin per row).
        band: ``[B, N]`` integer band ids.
        period: ``[B]`` days (NaN = no period).
        real: ``[B, N]`` bool, True for real (non-padding) tokens.
    """
    b, n = y.shape
    y, w, t = y.double(), w.double(), t_days.double()
    k = torch.arange(1, harmonics + 1, dtype=torch.float64, device=y.device)
    ok_row = torch.isfinite(period) & (period > 0)
    p = period.double().clamp_min(1e-6).to(y.device)
    arg = (2.0 * math.pi) * t[..., None] / p[:, None, None] * k  # [B, N, H]
    x = torch.cat([torch.ones_like(t)[..., None], arg.cos(), arg.sin()], -1)  # [B, N, 2H+1]
    n_par = 2 * harmonics + 1
    template = torch.full((b, n), float("nan"), dtype=torch.float64, device=y.device)
    r2 = torch.full((b, n), float("nan"), dtype=torch.float64, device=y.device)
    eye = ridge * torch.eye(n_par, dtype=torch.float64, device=y.device)
    for bid in torch.unique(band[real]):
        m = (band == bid) & real & ok_row[:, None].to(y.device)
        wm = w * m
        cnt = m.sum(1)
        a = torch.einsum("bni,bn,bnj->bij", x, wm, x) + eye
        rhs = torch.einsum("bni,bn->bi", x, wm * y)
        beta = torch.linalg.solve(a, rhs[..., None])[..., 0]  # [B, 2H+1]
        fit = torch.einsum("bni,bi->bn", x, beta)
        res = ((y - fit) ** 2 * wm).sum(1)
        mean = (wm * y).sum(1) / wm.sum(1).clamp_min(1e-12)
        tot = ((y - mean[:, None]) ** 2 * wm).sum(1)
        dof = (cnt - n_par).clamp_min(1).double()
        adj = 1.0 - (res / dof) / (tot / (cnt - 1).clamp_min(1).double()).clamp_min(1e-12)
        good = (cnt >= min_points)[:, None] & m
        template = torch.where(good, fit, template)
        r2 = torch.where(good, adj[:, None].expand_as(r2), r2)
    return template.float(), r2.float()


def smooth_targets(values, positions, pad, period_rows, spec, harmonics=6, min_r2=0.5, min_points=20, mode="mix"):
    """The reconstruction target for a fused batch: the window's smooth fit
    where its band fits well (adjusted R2 >= ``min_r2``), the observation
    elsewhere (``mode="mix"``); ``"template"`` uses the fit wherever it
    exists. Returns ``(target [B, N], share_of_tokens_using_the_fit)``."""
    from project.phase_decoder import band_ids, band_table

    mu, sd = spec.err_stats
    y = values[..., 0].float()
    sigma = torch.exp(values[..., 1].float() * sd + mu)
    w = 1.0 / (sigma**2 + 1e-6)
    t_days = (positions[:, 0].float() - 1.0) * spec.tokenize["time_scale"]
    band = band_ids(positions[:, 1], band_table(spec))
    real = ~pad
    template, r2 = window_templates(y, w, t_days, band, period_rows.to(values.device), real, harmonics, min_points)
    use = torch.isfinite(template) & (r2 >= (min_r2 if mode == "mix" else -1e9)) & real
    target = torch.where(use, template, y)
    share = float(use.sum()) / max(float(real.sum()), 1.0)
    return target, share
