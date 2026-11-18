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
def window_coefs(y, w, t_days, band, period, real, harmonics: int = 6, min_points: int = 20, ridge: float = 1e-3, n_bands: int | None = None):
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
        a = torch.einsum("bni,bn,bnj->bij", x, wm, x)
        a = a + eye * a.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)[:, None, None]  # ridge relative to the problem's scale
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


def sample_queries_cycles(positions, pad, band, k: int, period, lo_cycles: float, cap_cycles: float, time_scale: float,
                          random_share: float = 0.2, generator=None):
    """``k`` query times per row, their distance from a random real token
    drawn in CYCLES of the row's ``period [B]`` (days): log-uniform between
    ``lo_cycles`` and ``cap_cycles``, either sign, clipped to the window.
    A ``random_share`` of the ``k`` queries is instead uniform over the
    window, so the layout of the queries does not give the period away.
    Rows without a period (NaN or <= 0) get offsets as if the period were
    one day; their targets are NaN anyway and are padded out. Returns
    ``(t_days [B, k], band [B, k])``."""
    b, n = pad.shape
    real = ~pad
    t = (positions[:, 0].float() - 1.0) * time_scale
    w = real.float() + (~real.any(1, keepdim=True)).float()  # a row with no real token draws anything (its targets are NaN anyway)
    pick = torch.multinomial(w.cpu(), k, replacement=True, generator=generator).to(pad.device)  # k random real tokens, repeats allowed
    t_pick = t.gather(1, pick)
    per = torch.nan_to_num(period.to(t.device).float(), nan=1.0)
    per = torch.where(per > 0, per, torch.ones_like(per))
    lo, hi = math.log(max(lo_cycles, 1e-6)), math.log(max(cap_cycles, lo_cycles * 1.0001))
    cycles = torch.exp(lo + (hi - lo) * torch.rand(b, k, generator=generator).to(pad.device))
    sign = torch.where(torch.rand(b, k, generator=generator).to(pad.device) < 0.5, -1.0, 1.0)
    t_max = (t * real).max(1, keepdim=True).values
    t_q = (t_pick + sign * cycles * per[:, None]).clamp(min=torch.zeros_like(t_max), max=t_max)
    n_rand = int(round(random_share * k))
    if n_rand > 0:
        t_q[:, :n_rand] = torch.rand(b, n_rand, generator=generator).to(pad.device) * t_max
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


def bounded(template, y, real, margin: float = 1.0):
    """NaN where the fit leaves the window's observed range by more than
    ``margin`` times the range: a near-singular fit, not a target."""
    lo = torch.where(real, y, torch.full_like(y, float("inf"))).min(1, keepdim=True).values
    hi = torch.where(real, y, torch.full_like(y, float("-inf"))).max(1, keepdim=True).values
    span = (hi - lo).clamp_min(1e-3)
    ok = (template >= lo - margin * span) & (template <= hi + margin * span)
    return torch.where(ok, template, torch.full_like(template, float("nan")))


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
    template = bounded(template_eval(beta, period, t_days, band, harmonics), y, real)
    r2 = torch.gather(r2b, 1, band.long()).float()
    use = torch.isfinite(template) & (r2 >= (min_r2 if mode == "mix" else -1e9)) & real
    target = torch.where(use, template, y)
    share = float(use.sum()) / max(float(real.sum()), 1.0)
    smooth_targets.last = dict(beta=beta, period=period, band=band, t_days=t_days, y=y, real=real)
    return target, share


# ------------------------------------------------- whole-record templates

H_MAX = 8
H_CHOICES = (1, 2, 3, 4, 6, 8)


def period_lookup_best(path: str, cat_column: str = "p_catalogue_sharp", other_column: str = "p_model_cands") -> dict:
    """``{ztf_id: period}`` from the period table: the sharpened catalogue
    period, or the model's candidate period where the verdict says the
    catalogue is another period."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            col = other_column if row.get("verdict") == "other period" else cat_column
            try:
                v = float(row[col])
            except (KeyError, ValueError):
                continue
            if np.isfinite(v) and v > 0:
                out[str(row["ztf_id"])] = v
    return out


def _fit_one(args):
    i, t, y, err, band, period, n_bands = args
    coef = np.zeros((n_bands, 2 * H_MAX + 1))
    r2 = np.full(n_bands, np.nan)
    if not (np.isfinite(period) and period > 0):
        return i, coef, r2, 0
    from project import fold

    best = (-np.inf, 0, None, None)
    for h in H_CHOICES:
        n_par = 2 * h + 1
        ss_res = ss_tot = 0.0
        n_tot = p_tot = nb = 0
        betas, r2s = {}, {}
        for b in range(n_bands):
            m = band == b
            n = int(m.sum())
            if n < n_par + 5:
                continue
            w = 1.0 / (np.maximum(err[m], 0.0) ** 2 + 1e-6)
            sw = np.sqrt(w)
            x = fold.design(t[m].astype(np.float64), period, h)
            beta, *_ = np.linalg.lstsq(x * sw[:, None], y[m] * sw, rcond=None)
            res = y[m] - x @ beta
            mean = (w * y[m]).sum() / w.sum()
            sr, st = float((w * res**2).sum()), float((w * (y[m] - mean) ** 2).sum())
            betas[b] = beta
            r2s[b] = 1.0 - (sr / max(n - n_par, 1)) / max(st / max(n - 1, 1), 1e-12)
            ss_res, ss_tot, n_tot, p_tot, nb = ss_res + sr, ss_tot + st, n_tot + n, p_tot + n_par, nb + 1
        if not betas or n_tot - p_tot <= 0 or ss_tot <= 0:
            continue
        adj = 1.0 - (ss_res / (n_tot - p_tot)) / (ss_tot / (n_tot - nb))
        if adj > best[0] + 0.002:  # a higher harmonic count must earn its keep
            best = (adj, h, betas, r2s)
    adj, h, betas, r2s = best
    if betas is None:
        return i, coef, r2, 0
    for b, beta in betas.items():  # fold.design interleaves (1, cos1, sin1, cos2, sin2, ...); stored as (1, cos1..H, sin1..H)
        coef[b, 0] = beta[0]
        coef[b, 1 : 1 + h] = beta[1 : 1 + 2 * h : 2]
        coef[b, 1 + H_MAX : 1 + H_MAX + h] = beta[2 : 2 + 2 * h : 2]
        r2[b] = r2s[b]
    return i, coef, r2, h


def fit_record_templates(records, periods, n_bands: int = 3, workers: int = 8, log=print) -> dict:
    """One Fourier template per star, fitted per band on the WHOLE light
    curve (as the model sees it: the standardised records) on ``periods[i]``,
    with the harmonic count chosen per star by the pooled adjusted R2.
    Returns ``dict(coef [n, n_bands, 2 H_MAX + 1], period [n], harmonics [n],
    r2 [n, n_bands])`` as tensors (zero coefficients and NaN R2 where there
    is no fit)."""
    import time
    from multiprocessing import Pool

    t0 = time.time()
    jobs = [(i, r.t, r.y, r.err, r.band, float(periods[i]), n_bands) for i, r in enumerate(records)]
    coef = np.zeros((len(records), n_bands, 2 * H_MAX + 1))
    r2 = np.full((len(records), n_bands), np.nan)
    hs = np.zeros(len(records), dtype=np.int64)
    with Pool(max(workers, 1)) as pool:
        for k, (i, c, q, h) in enumerate(pool.imap_unordered(_fit_one, jobs, chunksize=64)):
            coef[i], r2[i], hs[i] = c, q, h
            if (k + 1) % 10000 == 0:
                log(f"  templates {k + 1} / {len(records)}, {time.time() - t0:.0f}s")
    ok = hs > 0
    log(f"templates: {int(ok.sum())} of {len(records)} stars fitted in {time.time() - t0:.0f}s; harmonics "
        + ", ".join(f"{h}: {int((hs == h).sum())}" for h in H_CHOICES) + f"; median R2 {np.nanmedian(r2[ok]):.3f}")
    return dict(coef=torch.tensor(coef, dtype=torch.float64), period=torch.tensor(np.asarray(periods, dtype=np.float64)),
                harmonics=torch.tensor(hs), r2=torch.tensor(r2, dtype=torch.float32))


class RecordTemplates:
    """The stored templates on a device: ``eval(rows, t_abs, band)`` gives
    the fit of record ``rows[b]`` at absolute record times ``t_abs [B, N]``
    (days) and bands ``band [B, N]`` -> ``[B, N]``; NaN where the band has
    no fit."""

    def __init__(self, fitted: dict, device):
        self.coef = fitted["coef"].to(device)
        self.period = fitted["period"].to(device)
        self.harmonics = fitted["harmonics"].to(device)
        self.r2 = fitted["r2"].to(device)

    @torch.no_grad()
    def eval(self, rows, t_abs, band):
        rows = rows.to(self.coef.device)
        beta = self.coef[rows]  # [B, n_bands, 2H+1]
        p = self.period[rows].clamp_min(1e-6)
        k = torch.arange(1, H_MAX + 1, dtype=torch.float64, device=self.coef.device)
        arg = (2.0 * math.pi) * t_abs.double().to(self.coef.device)[..., None] / p[:, None, None] * k
        x = torch.cat([torch.ones_like(arg[..., :1]), arg.cos(), arg.sin()], -1)  # [B, N, 2H+1]
        c = torch.gather(beta, 1, band.long().to(self.coef.device)[..., None].expand(-1, -1, beta.shape[-1]))
        out = (x * c).sum(-1).float()
        has = torch.isfinite(torch.gather(self.r2[rows], 1, band.long().to(self.coef.device)))
        return torch.where(has, out, torch.full_like(out, float("nan")))


def record_targets(templates, rows, starts, values, positions, pad, spec, mode="template", min_r2=0.5):
    """The reconstruction target of a fused batch from the stored
    whole-record templates: ``rows [B]`` record indices, ``starts [B]``
    window starts (record days). Returns ``(target [B, N], share)``."""
    from project.phase_decoder import band_ids, band_table

    y = values[..., 0].float()
    real = ~pad
    t_abs = starts.to(positions.device)[:, None].double() + (positions[:, 0].double() - 1.0) * spec.tokenize["time_scale"]
    band = band_ids(positions[:, 1], band_table(spec))
    tpl = bounded(templates.eval(rows, t_abs, band), y, real)
    r2 = torch.gather(templates.r2[rows.to(templates.r2.device)], 1, band.long().to(templates.r2.device)).to(y.device)
    use = torch.isfinite(tpl) & real & ((r2 >= min_r2) if mode == "mix" else torch.ones_like(real))
    record_targets.last = dict(t_abs=t_abs, band=band, y=y, real=real)
    return torch.where(use, tpl, y), float(use.sum()) / max(float(real.sum()), 1.0)
