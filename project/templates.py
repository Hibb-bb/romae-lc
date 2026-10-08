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
def window_coefs(y, w, t_days, band, period, real, harmonics: int = 6, min_points: int = 20, ridge: float = 1e-6, n_bands: int | None = None):
    """``(beta [B, n_bands, 2H+1], r2 [B, n_bands])``: the weighted Fourier
    fit of every row and band on the row's period (NaN where the band has
    fewer than ``min_points`` real points or the row has no period) and the
    adjusted R2 of each fit."""
    b, n = y.shape
    y, w, t = y.double(), w.double(), t_days.double()
    n_bands = int(n_bands or int(band.max().item()) + 1)
    k = torch.arange(1, harmonics + 1, dtype=torch.float64, device=y.device)
    ok_row = (torch.isfinite(period) & (period > 0)).to(y.device)
    p = period.double().clamp_min(1e-6).to(y.device)
    arg = (2.0 * math.pi) * t[..., None] / p[:, None, None] * k
    x = torch.cat([torch.ones_like(t)[..., None], arg.cos(), arg.sin()], -1)  # [B, N, 2H+1]
    n_par = 2 * harmonics + 1
    beta_all = torch.full((b, n_bands, n_par), float("nan"), dtype=torch.float64, device=y.device)
    r2_all = torch.full((b, n_bands), float("nan"), dtype=torch.float64, device=y.device)
    eye = ridge * torch.eye(n_par, dtype=torch.float64, device=y.device)
    for bid in range(n_bands):
        m = (band == bid) & real & ok_row[:, None]
        cnt = m.sum(1)
        if not (cnt >= min_points).any():
            continue
        wm = w * m
        a = torch.einsum("bni,bn,bnj->bij", x, wm, x) + eye
        rhs = torch.einsum("bni,bn->bi", x, wm * y)
        beta = torch.linalg.solve(a, rhs[..., None])[..., 0]
        fit = torch.einsum("bni,bi->bn", x, beta)
        res = ((y - fit) ** 2 * wm).sum(1)
        mean = (wm * y).sum(1) / wm.sum(1).clamp_min(1e-12)
        tot = ((y - mean[:, None]) ** 2 * wm).sum(1)
        dof = (cnt - n_par).clamp_min(1).double()
        adj = 1.0 - (res / dof) / (tot / (cnt - 1).clamp_min(1).double()).clamp_min(1e-12)
        good = cnt >= min_points
        beta_all[good, bid] = beta[good]
        r2_all[good, bid] = adj[good]
    return beta_all, r2_all


@torch.no_grad()
def template_eval(beta, period, t_days, band, harmonics: int = 6):
    """The fit at any times: ``beta [B, n_bands, 2H+1]``, ``t_days [B, K]``,
    ``band [B, K]`` -> ``[B, K]`` (NaN where that band has no fit)."""
    k = torch.arange(1, harmonics + 1, dtype=torch.float64, device=beta.device)
    p = period.double().clamp_min(1e-6).to(beta.device)
    arg = (2.0 * math.pi) * t_days.double()[..., None] / p[:, None, None] * k
    x = torch.cat([torch.ones_like(t_days.double())[..., None], arg.cos(), arg.sin()], -1)  # [B, K, 2H+1]
    coef = torch.gather(beta, 1, band.long()[..., None].expand(-1, -1, beta.shape[-1]))  # [B, K, 2H+1]
    return (x * coef).sum(-1).float()


@torch.no_grad()
def window_templates(y, w, t_days, band, period, real, harmonics: int = 6, min_points: int = 20, ridge: float = 1e-6):
    """``(template [B, N], r2 [B, N])`` at the tokens' own times (see
    :func:`window_coefs`); the R2 of a token's band is broadcast to it."""
    beta, r2b = window_coefs(y, w, t_days, band, period, real, harmonics, min_points, ridge)
    template = template_eval(beta, period, t_days, band, harmonics)
    r2 = torch.gather(r2b, 1, band.long()).float()
    return template, r2


def sample_queries(positions, pad, band, k: int, delta_days: float, time_scale: float, generator=None):
    """``k`` query times per row near the real tokens: a random real token
    plus a uniform offset in ``[-delta, +delta]`` days, clipped to the
    window; the query takes that token's band. Returns ``(t_days [B, k],
    band [B, k])``."""
    b, n = pad.shape
    real = ~pad
    t = (positions[:, 0].float() - 1.0) * time_scale
    score = torch.rand(b, n, generator=generator).to(pad.device).masked_fill(pad, 2.0)
    pick = score.argsort(1)[:, :k] % n  # the k lowest scores = random real tokens (padding sorted last)
    pick = torch.where(real.gather(1, pick), pick, real.float().argmax(1, keepdim=True).expand_as(pick))  # rows with fewer than k real tokens repeat one
    t_pick = t.gather(1, pick)
    off = (torch.rand(b, k, generator=generator).to(pad.device) * 2.0 - 1.0) * delta_days
    t_max = (t * real).max(1, keepdim=True).values
    t_q = (t_pick + off).clamp(min=torch.zeros_like(t_max), max=t_max)
    return t_q, band.gather(1, pick)


def augment_with_queries(values, positions, pad, mask, target, weight, t_q, band_q, target_q, spec):
    """Append the query tokens as hidden tokens: zero brightness, the row's
    median error channel, positions from the query times and bands, masked
    for the encoder, with ``target_q`` as their target. Queries whose
    target is NaN are padded out (they do not count)."""
    from project.phase_decoder import band_table

    b, n, c = values.shape
    k = t_q.shape[1]
    ts = spec.tokenize["time_scale"]
    med_err = torch.stack([row[m].median() if m.any() else row.new_tensor(0.0) for row, m in zip(values[..., 1], ~pad)])
    v_q = values.new_zeros(b, k, c)
    if c > 1:
        v_q[..., 1] = med_err[:, None].to(values.dtype)
    table = band_table(spec).to(positions.device, positions.dtype)
    p_q = torch.stack([t_q.to(positions.dtype) / ts + 1.0, table[band_q.long()]], 1)  # [B, 2, k]
    bad = ~torch.isfinite(target_q)
    values = torch.cat([values, v_q], 1)
    positions = torch.cat([positions, p_q], 2)
    pad = torch.cat([pad, bad], 1)
    mask = torch.cat([mask, torch.ones(b, k, dtype=torch.bool, device=mask.device)], 1)
    target = torch.cat([target, torch.nan_to_num(target_q, nan=0.0).to(target.dtype)], 1)
    if weight is not None:
        weight = torch.cat([weight, torch.ones(b, k, dtype=weight.dtype, device=weight.device)], 1)
    return values, positions, pad, mask, target, weight


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
    period = period_rows.to(values.device)
    beta, r2b = window_coefs(y, w, t_days, band, period, real, harmonics, min_points, n_bands=len(spec.tokenize["band_wavelengths"]))
    template = template_eval(beta, period, t_days, band, harmonics)
    r2 = torch.gather(r2b, 1, band.long()).float()
    use = torch.isfinite(template) & (r2 >= (min_r2 if mode == "mix" else -1e9)) & real
    target = torch.where(use, template, y)
    share = float(use.sum()) / max(float(real.sum()), 1.0)
    smooth_targets.last = dict(beta=beta, period=period, band=band, t_days=t_days)
    return target, share
