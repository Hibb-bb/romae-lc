"""Brightness forecast, end to end: the first test of the whole chain.

Given the past windows of a star, what will it look like in a window that
starts one or two window lengths later, at the times it was really observed?
The chain is: the frozen encoder turns every past window into a latent, the
sequence predictor (``train_predictor.py --arch seq``) reads them and draws
latents for the future window, and the decoder turns each drawn latent into a
brightness at every observed time of that window. The forecast is a mixture
of the drawn curves. It is scored on the observed points under their known
errors, and every number sits next to yardsticks that use no model:

- ``persist``: decode the current latent at the future times ("the star
  looks the same as in the last window");
- ``constant``: the median brightness of the past, per band;
- ``gp``: a Gaussian process on the last window, run forward, per band;
- ``ls_fold``: fold the past on the Lomb-Scargle period of the past and read
  the template at the future times (the classic forecast, no labels);
- ``oracle_fold``: the same with the catalogue period, the upper bound of a
  fold, which uses a label the model never sees;
- ``encode_decode``: the decoder on the true latent of the future window
  (the encoder sees the window itself), the ceiling of the decoder. If this
  is good and the model is not, the fault is in the predicted latents.

The latent space is measured too, in standardised units per dimension:
the distance of each drawn latent, of their mean, of the current latent
("nothing changes") and of the training mean from the future window's true
latent, and the spread of the drawn latents around their mean.

Scores per star and horizon: the negative log-likelihood per point (a
Gaussian for every method, a mixture over drawn curves for the model), the
root mean squared error of the mean forecast, and the skill against
``constant`` (1 - mse / mse_constant). Tables pool them per class and
horizon (medians), with the share of stars where the model beats each
yardstick. A few typical stars are drawn, in time and phase-folded: the
model's forecast is also decoded on a fine time grid across the future
window and folded on the catalogue period and on the model's own period
(the bin read-out sharpened by the fine search, saved by
``period_probe.py`` in ``fold_r2.npz``; pass it with ``--model-periods``).

    python -m project.forecast --pred project/runs/pred_maew_seq2/pred.pt \\
        --dec project/runs/maew_w250/dec_mse/best.pt --out project/results/forecast_maew \\
        --model-periods project/results/period_maew_refine2/fold_r2.npz \\
        --periods-latents project/runs/maew_w250/latents_r4.pt
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from project import baselines as bl
from project.anomaly import states_long
from project.cache_latents import object_windows
from project.common import (
    add_device_arg,
    data_args_from,
    dump_json,
    get_device,
    load_data,
    load_encoder,
    subset,
    superclass,
)
from project.decoder import gaussian_var, load_decoder
from project.lookback import encode_context, is_lookback, load_lookback
from project.inject import band_templates, eval_template
from project.period_probe import ls_periods, pool_latents
from project.train_predictor import load_predictor

BASE_METHODS = ("model", "persist", "constant", "gp", "ls_fold", "oracle_fold", "encode_decode")
METHODS = BASE_METHODS  # plus "no_latent" when the decoder is a lookback decoder (set in run)
LATENT_KEYS = ("d_sample", "d_sample_mean", "d_persist", "d_train_mean", "spread")
HEADLINE = ("ECL", "RR", "ROT", "CEP", "DSCT", "LPV")


# ------------------------------------------------------------------- pieces


@torch.no_grad()
def decode(dec, z, tok, ctx=None, use_z=None):
    """``(mu, var)`` of the decoder at the tokens of one window, for every
    latent in ``z [S, D]``; the window's tokens are repeated ``S`` times.
    Both ``[S, N]`` float32, ``var`` including the point's error. A lookback
    decoder also takes ``ctx`` (one star's context, batch of 1, see
    :func:`project.lookback.encode_context`) and ``use_z`` (False = the "no
    latent" token)."""
    s = z.shape[0]
    rep = lambda x: x.expand(s, *x.shape[1:]).contiguous()
    from romae_lc import Tokens

    t = Tokens(rep(tok.values), rep(tok.positions), rep(tok.pad_mask), rep(tok.extras))
    if getattr(dec, "kind", "") == "lookback":
        if ctx is None:
            raise ValueError("a lookback decoder needs the context")
        flag = torch.full((s,), True if use_z is None else bool(use_z), device=z.device)
        mu, logvar = dec(z, t.positions, t.pad_mask, tuple(rep(x) for x in ctx), flag)
    else:
        mu, logvar = dec(z, t.positions, t.pad_mask, dec.log_sigma(t))
    return mu.float(), gaussian_var(t.extras, logvar)


def mixture_nll(y, mu, var):
    """Mean negative log-likelihood per point of ``y [N]`` under an equal
    mixture of Gaussians ``N(mu[s], var[s])``, ``mu, var [S, N]``."""
    lp = -0.5 * ((y[None] - mu) ** 2 / var + np.log(2 * np.pi * var))
    m = lp.max(0)
    log_mix = m + np.log(np.exp(lp - m).mean(0))
    return float(-log_mix.mean())


def fold_forecast(t_ctx, y_ctx, band_ctx, period, t_q, band_q):
    """Fold the past on ``period`` and read the template at the future times:
    ``(mu, var)`` with ``var`` the scatter of the past around the template."""
    from types import SimpleNamespace

    rec = SimpleNamespace(t=t_ctx, y=y_ctx, band=band_ctx, n=len(t_ctx))
    if not np.isfinite(period) or period <= 0:
        return np.full(len(t_q), np.median(y_ctx)), np.full(len(t_q), max(np.var(y_ctx), 1e-6))
    templates, resid = band_templates(rec, period)
    mu = np.full(len(t_q), np.median(y_ctx))
    var = np.full(len(t_q), max(np.var(y_ctx), 1e-6))
    for b, (edges, vals, _) in templates.items():
        q, c = band_q == b, band_ctx == b
        if q.any():
            mu[q] = eval_template(edges, vals, t_q[q] / period)
            var[q] = max(float(np.var(resid[c])), 1e-6)
    return mu, var


def load_model_periods(path, cache_path, split, min_tokens):
    """``{record index: period}`` from a ``fold_r2.npz`` of ``period_probe``:
    ``p_refined`` (the model's period after the fine search) per object of
    ``split``. Old files carry no ``index``; then the same latent cache the
    probe used rebuilds the object order (``pool_latents`` keeps the objects
    with a valid window, ``object_table`` then drops those without a
    catalogue period)."""
    d = np.load(path)
    p = d["p_refined"] if "p_refined" in d.files else d["p_model"]
    if "index" in d.files:
        index = d["index"]
    else:
        if cache_path is None:
            raise SystemExit("--model-periods has no 'index': pass --periods-latents (the cache the probe used)")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        pooled = pool_latents(cache, split, min_tokens)
        objs = cache["objects"][split]
        index = objs["index"].numpy()[pooled["keep"]]
        period = objs["period"].numpy().astype(np.float64)[pooled["keep"]]
        index = index[np.isfinite(period) & (period > 0)]
    if len(index) != len(p):
        raise SystemExit(f"{len(index)} objects in the cache order but {len(p)} periods in {path}")
    return {int(i): float(v) for i, v in zip(index, p)}


@torch.no_grad()
def dense_curves(dec, spec, zs, start, window, bands, err_med, n=200, period=None, ctx=None, use_z=None):
    """The decoder's curves on a fine time grid of the window, one per band,
    for every drawn latent in ``zs [S, D]``: ``dict(t, band, mean, lo, hi)``.
    With ``period`` the grid spans one period (at most the window) around
    the window's middle, so the folded curve covers every phase once."""
    if period is not None and np.isfinite(period) and period > 0:
        half = 0.5 * min(period, window)
        t = np.linspace(window / 2 - half, window / 2 + half, n, endpoint=False)
    else:
        t = np.linspace(0.0, window, n, endpoint=False)
    # unique, time-sorted query points, so the tokens keep this order
    t_all = np.concatenate([t + 1e-4 * k for k in range(len(bands))])
    b_all = np.concatenate([np.full(n, b) for b in bands]).astype(np.int64)
    order = np.argsort(t_all, kind="stable")
    t_all, b_all = t_all[order].astype(np.float32), b_all[order]
    frame = (t_all, np.zeros_like(t_all), b_all, np.full_like(t_all, err_med))
    tok = spec.tokens([frame]).to(zs.device)
    mu, _ = decode(dec, zs, tok, ctx, use_z)
    mu = mu.cpu().numpy()
    return dict(t=start + t_all.astype(np.float64), band=b_all, mean=mu.mean(0),
                lo=np.percentile(mu, 16, axis=0), hi=np.percentile(mu, 84, axis=0))  # fmt: skip


# -------------------------------------------------------------------- scoring


def score(y, err, mu, var, mse_const):
    """``dict(nll, rmse, skill)`` of one forecast; ``mu, var`` are ``[N]``
    (one curve) or ``[S, N]`` (drawn curves, scored as a mixture)."""
    mu, var = np.asarray(mu, dtype=np.float64), np.asarray(var, dtype=np.float64)
    if mu.ndim == 1:
        nll = float(bl.gaussian_nll(y, mu, var + err**2).mean())
        mean = mu
    else:
        nll = mixture_nll(y, mu, var + err[None] ** 2)
        mean = mu.mean(0)
    mse = float(((y - mean) ** 2).mean())
    return dict(nll=nll, rmse=math.sqrt(mse), skill=1.0 - mse / max(mse_const, 1e-12))


def forecast_star(enc, spec, model, dec, r, cfg, stride, args, index, device, rng, p_min):
    """Every forecast of one star: a list of dicts, one per target window and
    horizon, with the scores of every method and what is needed to draw it."""
    starts, frames, n_tokens = object_windows(r, cfg, stride, cfg.max_tokens, args.seed, index)
    n_tokens = np.asarray(n_tokens)
    valid = np.flatnonzero(n_tokens >= cfg.min_tokens)
    if len(valid) < args.min_hist + 1:
        return []
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    with torch.no_grad(), amp:
        toks = spec.tokens([frames[i] for i in valid]).to(device)
        z = enc.encode([toks])[:, 0].float()
    w = max(1, int(round(1.0 / stride)))
    wt = torch.as_tensor(valid, device=device)
    gaps = torch.zeros(len(valid), device=device)
    gaps[1:] = (wt[1:] - wt[:-1]).float() * stride
    with torch.no_grad():
        h = states_long(model, z, gaps)
    pos_of = {int(g): j for j, g in enumerate(valid)}  # grid index -> row in z
    # candidate (history end, target) pairs: the target starts h window
    # lengths after the history window, both valid, enough history before
    cands = []
    for j in range(args.min_hist, len(valid)):
        for hz in args.horizons:
            k = int(valid[j]) + hz * w
            if k in pos_of:
                cands.append((j, k, hz))
    if not cands:
        return []
    pick = rng.choice(len(cands), size=min(args.per_star, len(cands)), replace=False)
    t_all = r.t.astype(np.float64)
    out = []
    gen = torch.Generator(device=device).manual_seed(args.seed + index)
    for c in pick:
        j, k, hz = cands[c]
        tgt = frames[k]
        t_rel, y, band, err = (np.asarray(a) for a in tgt)
        t_abs = starts[k] + t_rel
        tok = spec.tokens([tgt]).to(device)
        real = ~tok.pad_mask[0].cpu().numpy()
        n = int(real.sum())
        ctx = None
        if getattr(dec, "kind", "") == "lookback":
            # the last n_ctx valid non-overlapping windows that end where the gap starts
            cw = [c for c in range(int(valid[j]), int(valid[j]) - dec.n_ctx * w, -w) if c >= 0 and c in pos_of]
            with torch.no_grad(), amp:
                ctx = encode_context(enc.model, [spec.tokens([frames[c]]).to(device) for c in cw],
                                     torch.tensor([[starts[c] - starts[k] for c in cw]], device=device, dtype=torch.float32),
                                     spec.tokenize["time_scale"])  # fmt: skip
                ctx = tuple(x.float() if x.is_floating_point() else x for x in ctx)
        # the past: every point before the target window
        past = t_all < starts[k]
        tp, yp, bp, ep = t_all[past], r.y[past].astype(np.float64), r.band[past], r.err[past].astype(np.float64)
        hist_win = (t_all >= starts[int(valid[j])]) & (t_all < starts[int(valid[j])] + cfg.window)
        mse_const = float(np.var(y)) if len(y) > 1 else 1.0
        res = dict(index=int(index), grid=int(k), horizon=int(hz), n_points=n,
                   start=float(starts[k]), hist_start=float(starts[int(valid[j])]), scores={})  # fmt: skip
        # the model: draw latents for the target, decode each at its points
        z_t, h_t = z[j : j + 1], h[j : j + 1]
        gap = torch.full((1,), float(hz), device=device)
        with torch.no_grad():
            zs = torch.cat([model.sample_next(h_t, gap, z_t, generator=gen) for _ in range(args.samples)])
            mu_s, var_s = decode(dec, zs, tok, ctx)
            mu_p, var_p = decode(dec, z_t, tok, ctx)
            z_true = z[pos_of[k] : pos_of[k] + 1]
            mu_e, var_e = decode(dec, z_true, tok, ctx)
            if ctx is not None:
                mu_n, var_n = decode(dec, z_t, tok, ctx, use_z=False)
                mu_n, var_n = mu_n.cpu().numpy()[0, real], np.maximum(var_n.cpu().numpy()[0, real] - err**2, 1e-6)
            zn, tn, sn = model.normalize(zs), model.normalize(z_true), model.normalize(z_t)
            rms = lambda x: float((x**2).mean(-1).sqrt().mean())
            res["latent"] = dict(d_sample=rms(zn - tn), d_sample_mean=rms(zn.mean(0, keepdim=True) - tn),
                                 d_persist=rms(sn - tn), d_train_mean=rms(tn), spread=float(zn.std(0).mean()))  # fmt: skip
        mu_e, var_e = mu_e.cpu().numpy()[0, real], np.maximum(var_e.cpu().numpy()[0, real] - err**2, 1e-6)
        mu_s, var_s = mu_s.cpu().numpy()[:, real], var_s.cpu().numpy()[:, real]
        var_s = np.maximum(var_s - err[None] ** 2, 1e-6)  # the decoder's own spread only
        mu_p, var_p = mu_p.cpu().numpy()[0, real], np.maximum(var_p.cpu().numpy()[0, real] - err**2, 1e-6)
        res["scores"]["model"] = score(y, err, mu_s, var_s, mse_const)
        res["scores"]["persist"] = score(y, err, mu_p, var_p, mse_const)
        res["scores"]["encode_decode"] = score(y, err, mu_e, var_e, mse_const)
        if ctx is not None:
            res["scores"]["no_latent"] = score(y, err, mu_n, var_n, mse_const)
        # yardsticks without a model
        mu_c, var_c = bl.per_band(bl.constant, tp, yp, ep, bp, t_abs, band)
        res["scores"]["constant"] = score(y, err, mu_c, var_c, mse_const)
        mu_g, var_g = bl.per_band(bl.gp_rbf, t_all[hist_win], r.y[hist_win].astype(np.float64),
                                  r.err[hist_win].astype(np.float64), r.band[hist_win], t_abs, band)  # fmt: skip
        res["scores"]["gp"] = score(y, err, mu_g, var_g, mse_const)
        ls = ls_periods(tp, yp, ep, bp, p_min, args.ls_oversample, args.ls_cap, 1, device)["best"]
        mu_l, var_l = fold_forecast(tp, yp, bp, ls, t_abs, band)
        res["scores"]["ls_fold"] = score(y, err, mu_l, var_l, mse_const)
        mu_o, var_o = fold_forecast(tp, yp, bp, float(r.period), t_abs, band)
        res["scores"]["oracle_fold"] = score(y, err, mu_o, var_o, mse_const)
        res["ls_period"] = float(ls)
        res["draw"] = dict(t=t_abs, y=y, err=err, band=band, model_mean=mu_s.mean(0),
                           model_lo=np.percentile(mu_s, 16, axis=0), model_hi=np.percentile(mu_s, 84, axis=0),
                           persist=mu_p, oracle=mu_o, ls=mu_l, zs=zs.cpu(), z_true=z_true.cpu(), err_med=float(np.median(err)),
                           ctx=None if ctx is None else tuple(x.cpu() for x in ctx),
                           hist_t=t_all[hist_win], hist_y=r.y[hist_win].astype(np.float64), hist_band=r.band[hist_win])  # fmt: skip
        out.append(res)
    return out


# --------------------------------------------------------------------- tables


def pool(rows, key):
    def med(vals):
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

    out = {}
    for m in METHODS:
        out[m] = {k: med([r["scores"][m][k] for r in rows]) for k in ("nll", "rmse", "skill")}
    out["n"] = len(rows)
    out["latent"] = {k: med([r["latent"][k] for r in rows if "latent" in r]) for k in LATENT_KEYS}
    for m in METHODS[1:]:
        both = [(r["scores"]["model"][key], r["scores"][m][key]) for r in rows]
        both = [(a, b) for a, b in both if np.isfinite(a) and np.isfinite(b)]
        out[f"beats_{m}"] = float(np.mean([a < b for a, b in both])) if both else float("nan")
    return out


def md_table(rows, cols):
    out = "| " + " | ".join(c for _, c in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    for r in rows:
        out += "| " + " | ".join(f"{r.get(k):.3f}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k, _ in cols) + " |\n"
    return out


def write_tables(path, res):
    lines = ["# Brightness forecast, end to end\n"]
    lines.append(
        f"{res['n_stars']} {res['split']} stars, up to {res['per_star']} forecasts each; a forecast is the "
        f"window that starts 1 or 2 window lengths ({res['window']:g} d) after the last window seen. "
        "Scores are medians over forecasts: NLL per observed point under its error (lower is better; the "
        "model's is a mixture over drawn curves), RMSE of the mean forecast in normalised brightness, "
        "and skill = 1 - mse / mse of the constant forecast (1 is perfect, 0 is no better than the mean). "
        "'beats' is the share of forecasts where the model's NLL is lower than the yardstick's.\n"
    )
    for name, groups in res["tables"].items():
        lines.append(f"\n## {name}\n")
        for stat in ("nll", "rmse", "skill"):
            lines.append(f"\n### {stat}\n")
            rows = [dict(group=g, n=v["n"], **{m: v[m][stat] for m in METHODS}) for g, v in groups.items()]
            lines.append(md_table(rows, [("group", "group"), ("n", "n")] + [(m, m) for m in METHODS]))
        lines.append("\n### share of forecasts where the model beats the yardstick (NLL)\n")
        rows = [dict(group=g, n=v["n"], **{m: v[f"beats_{m}"] for m in METHODS[1:]}) for g, v in groups.items()]
        lines.append(md_table(rows, [("group", "group"), ("n", "n")] + [(m, m) for m in METHODS[1:]]))
        lines.append(
            "\n### latent space (medians, standardised units per dimension)\n\n"
            "Distances to the future window's true latent: of each drawn latent, of the mean of the draws, "
            "of the current latent ('nothing changes') and of the training mean (a forecast of nothing); "
            "'spread' is the standard deviation of the draws around their mean.\n"
        )
        rows = [dict(group=g, n=v["n"], **v["latent"]) for g, v in groups.items()]
        lines.append(md_table(rows, [("group", "group"), ("n", "n")] + [(k, k) for k in LATENT_KEYS]))
    with open(path, "w") as f:
        f.write("\n".join(lines))


# -------------------------------------------------------------------- figures

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BAND_COL = ["#2a78d6", "#eb6834", "#1baf7a"]
BAND_NAME = {0: "g", 1: "r", 2: "i"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=7)


def _dense_lines(ax, dense, x_of, label_once=True):
    """The decoder's mean curve and its 16th to 84th percentile band, per band,
    against ``x_of(t)`` (time or phase); returns the handles for a legend."""
    handles = []
    for b in np.unique(dense["band"]):
        m = dense["band"] == b
        x = x_of(dense["t"][m])
        o = np.argsort(x)
        c = BAND_COL[int(b) % 3]
        ax.fill_between(x[o], dense["lo"][m][o], dense["hi"][m][o], color=c, alpha=0.18, linewidths=0)
        (h,) = ax.plot(x[o], dense["mean"][m][o], color=c, lw=1.5, label="model forecast" if label_once else None)
        handles.append(h)
        label_once = False
    return handles


def _true_lines(ax, dense, x_of):
    """The decoder's mean curve on the future window's own latent (the
    encoder saw the window): the decoder's ceiling, a dark dotted line."""
    first = True
    for b in np.unique(dense["band"]):
        m = dense["band"] == b
        x = x_of(dense["t"][m])
        o = np.argsort(x)
        ax.plot(x[o], dense["mean"][m][o], color=INK, lw=0.9, ls=(0, (1, 1.5)), alpha=0.8,
                label="decoder on the window's own latent" if first else None)  # fmt: skip
        first = False


def plot_examples(path, examples, window):
    """Time view: the past window (faint dots), the future window (bright
    dots), the decoder's forecast curves with their spread, and two
    yardsticks at the observed points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(examples), 1, figsize=(10.5, 2.7 * len(examples)), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (label, ex) in zip(axes[:, 0], examples):
        d = ex["draw"]
        _style(ax)
        for b in np.unique(np.concatenate([d["band"], d["hist_band"]])):
            c = BAND_COL[int(b) % 3]
            m, mh = d["band"] == b, d["hist_band"] == b
            ax.scatter(d["hist_t"][mh], d["hist_y"][mh], s=5, color=c, alpha=0.3, linewidths=0)
            ax.scatter(d["t"][m], d["y"][m], s=9, color=c, alpha=0.9, linewidths=0, zorder=4,
                       label=f"observed, {BAND_NAME.get(int(b), b)} band")  # fmt: skip
        _dense_lines(ax, d["dense"], lambda t: t)
        _true_lines(ax, d["dense_true"], lambda t: t)
        o = np.argsort(d["t"])
        ax.plot(d["t"][o], d["persist"][o], color="#7f7f7f", lw=0.9, ls=(0, (4, 3)), label="nothing changes")
        ax.plot(d["t"][o], d["oracle"][o], color="#d55e00", lw=0.9, ls=(0, (1, 2)), label="fold on the catalogue period")
        ax.axvspan(ex["hist_start"], ex["hist_start"] + window, color=GRID, alpha=0.45, lw=0)
        ax.axvspan(ex["start"], ex["start"] + window, color="#2a78d6", alpha=0.05, lw=0)
        s = ex["scores"]
        ax.set_title(
            f"{label}, P = {ex['catalogue_period']:.4g} d: {ex['horizon']} window ahead, {ex['n_points']} points  |  "
            f"NLL: model {s['model']['nll']:.2f}, nothing changes {s['persist']['nll']:.2f}, GP {s['gp']['nll']:.2f}, "
            f"catalogue fold {s['oracle_fold']['nll']:.2f}, own latent {s['encode_decode']['nll']:.2f}  |  skill: model {s['model']['skill']:.2f}, "
            f"catalogue fold {s['oracle_fold']['skill']:.2f}, own latent {s['encode_decode']['skill']:.2f}",
            fontsize=7.5, color=INK, loc="left",
        )  # fmt: skip
        ax.set_ylabel("brightness (normalised)", fontsize=7.5, color=INK2)
        ax.invert_yaxis()
    axes[-1, 0].set_xlabel("time (days)", fontsize=8, color=INK2)
    axes[0, 0].legend(fontsize=6.5, frameon=False, ncol=3, loc="upper left", labelcolor=INK2)
    fig.suptitle("Brightness forecasts in time: grey = the last window seen, blue = the window forecast", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_folded(path, examples):
    """Phase view: the future window's observed points and the decoder's
    forecast curve, folded on the catalogue period (left) and on the model's
    own period after the fine search (right); the past, folded, in grey."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(examples), 2, figsize=(10.5, 2.6 * len(examples)), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for row, (label, ex) in zip(axes, examples):
        d = ex["draw"]
        for ax, key, period, name in (
            (row[0], "dense_cat", ex["catalogue_period"], "catalogue period"),
            (row[1], "dense_mod", ex["model_period"], "model period + fine search"),
        ):
            _style(ax)
            if not (np.isfinite(period) and period > 0):
                ax.text(0.5, 0.5, f"no {name} for this star", transform=ax.transAxes, ha="center", color=INK2, fontsize=8)
                continue
            phase = lambda t, p=period: ((t - ex["start"]) / p) % 1.0
            for b in np.unique(np.concatenate([d["band"], d["hist_band"]])):
                c = BAND_COL[int(b) % 3]
                m, mh = d["band"] == b, d["hist_band"] == b
                ax.scatter(phase(d["hist_t"][mh]), d["hist_y"][mh], s=5, color="#9a9a9a", alpha=0.35, linewidths=0)
                ax.scatter(phase(d["t"][m]), d["y"][m], s=10, color=c, alpha=0.9, linewidths=0, zorder=4,
                           label=f"observed, {BAND_NAME.get(int(b), b)} band")  # fmt: skip
            _dense_lines(ax, d[key], phase)
            _true_lines(ax, d[key + "_true"], phase)
            ax.set_xlim(0, 1)
            ax.invert_yaxis()
            ax.set_title(f"{label}: {name}, P = {period:.6g} d", fontsize=7.5, color=INK, loc="left")
            ax.set_xlabel("phase", fontsize=7.5, color=INK2)
        row[0].set_ylabel("brightness (normalised)", fontsize=7.5, color=INK2)
    axes[0, 0].legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left", labelcolor=INK2)
    fig.suptitle("The same forecasts, phase-folded: grey = the past window, colour = the future window and the model's curve", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------ main


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True, help="pred.pt of a sequence predictor")
    p.add_argument("--dec", required=True, help="a decoder checkpoint (mse kind) on the same encoder")
    p.add_argument("--ckpt", default=None, help="the encoder (default: the predictor's)")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--n-objects", type=int, default=500)
    p.add_argument("--per-star", type=int, default=2, help="forecasts per star")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2], help="window lengths ahead")
    p.add_argument("--samples", type=int, default=16, help="drawn latents per forecast")
    p.add_argument("--min-hist", type=int, default=3)
    p.add_argument("--stride", type=float, default=None)
    p.add_argument("--ls-oversample", type=float, default=5.0)
    p.add_argument("--ls-cap", type=int, default=100_000)
    p.add_argument("--n-examples", type=int, default=6)
    p.add_argument("--example-min-points", type=int, default=40, help="points in a drawn future window")
    p.add_argument("--model-periods", default=None, help="fold_r2.npz of period_probe (the model's refined periods)")
    p.add_argument("--periods-latents", default=None, help="the cache that probe used (old npz without 'index')")
    p.add_argument("--n-dense", type=int, default=200, help="query times per band for the drawn curves")
    p.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("data (default: the checkpoint's)")
    g.add_argument("--data", default=None)
    g.add_argument("--max-rows", type=int, default=None)
    g.add_argument("--n-sim", type=int, default=None)
    add_device_arg(p)
    return p.parse_args(argv)


def run(args):
    t0 = time.time()
    dev = get_device(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, pmeta = load_predictor(args.pred, dev)
    if getattr(model, "arch", "mlp") != "seq":
        raise SystemExit("the forecast needs a sequence predictor (--arch seq)")
    global METHODS
    if is_lookback(args.dec):
        dec, _ = load_lookback(args.dec, dev)
        METHODS = BASE_METHODS + ("no_latent",)
    else:
        dec, _ = load_decoder(args.dec, dev)
        METHODS = BASE_METHODS
        if dec.kind != "mse":
            raise SystemExit("the forecast uses the mse decoder")
    lmeta = pmeta["latent_meta"]
    ckpt = args.ckpt or lmeta["ckpt"]
    stride = float(args.stride if args.stride is not None else lmeta["stride"])
    enc, meta = load_encoder(ckpt, dev)
    cfg, spec = meta.cfg, meta.spec
    if dec.z_dim != enc.dim:
        raise SystemExit(f"decoder latent width {dec.z_dim} does not match the encoder's {enc.dim}")
    over = argparse.Namespace(data=args.data, max_rows=args.max_rows, n_sim=args.n_sim)
    data = load_data(data_args_from(meta.args, over), splits=("train", args.split))
    p_min = float(min(r.period for r in data["train"] if r.period and r.period > 0))
    full = data[args.split]
    pick = np.arange(len(full))
    if args.n_objects is not None and args.n_objects < len(full):  # the same draw as common.subset
        pick = np.sort(np.random.default_rng(args.seed).choice(len(full), args.n_objects, replace=False))
    pick = [int(i) for i in pick if full[i].period and full[i].period > 0]
    recs = [full[i] for i in pick]
    model_periods = {}
    if args.model_periods:
        model_periods = load_model_periods(args.model_periods, args.periods_latents, args.split, cfg.min_tokens)
        print(f"model periods for {sum(i in model_periods for i in pick)} of {len(pick)} stars from {args.model_periods}")
    print(
        f"predictor {args.pred} ({model.kind}), decoder {args.dec}, encoder {ckpt} (dim {enc.dim}); "
        f"window {cfg.window:g} d, stride {stride:g}; {len(recs)} {args.split} stars, horizons {args.horizons}; {dev}",
        flush=True,
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    for i, r in enumerate(recs):
        for res in forecast_star(enc, spec, model, dec, r, cfg, stride, args, i, dev, rng, p_min):
            res["superclass"] = superclass(r)
            res["record"] = pick[i]
            res["catalogue_period"] = float(r.period)
            res["model_period"] = model_periods.get(pick[i], float("nan"))
            rows.append(res)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} / {len(recs)} stars, {len(rows)} forecasts, {time.time() - t0:.0f}s", flush=True)
    if not rows:
        raise SystemExit("no star gave a forecast")

    tables = {}
    tables["by horizon"] = {f"{hz} window ahead": pool([x for x in rows if x["horizon"] == hz], "nll") for hz in args.horizons}
    groups = [g for g in HEADLINE if any(x["superclass"] == g for x in rows)] or sorted({x["superclass"] for x in rows})
    tables["by superclass, 1 window ahead"] = {g: pool([x for x in rows if x["superclass"] == g and x["horizon"] == args.horizons[0]], "nll") for g in groups}
    res = dict(pred=str(args.pred), dec=str(args.dec), ckpt=str(ckpt), split=args.split, n_stars=len(recs),
               per_star=args.per_star, window=float(cfg.window), stride=stride, horizons=args.horizons,
               samples=args.samples, n_forecasts=len(rows), tables=tables, args=dict(vars(args)),
               seconds=time.time() - t0)  # fmt: skip
    dump_json(res, out / "summary.json")
    write_tables(out / "tables.md", res)
    np.savez_compressed(
        out / "forecasts.npz",
        **{f"{k}_{m}": np.array([x["scores"][m][k] for x in rows]) for m in METHODS for k in ("nll", "rmse", "skill")},
        horizon=np.array([x["horizon"] for x in rows]),
        superclass=np.array([x["superclass"] for x in rows]),
        index=np.array([x["index"] for x in rows]),
        n_points=np.array([x["n_points"] for x in rows]),
        **{f"latent_{k}": np.array([x["latent"][k] for x in rows]) for k in LATENT_KEYS},
    )
    # typical examples: per class, the 1-window forecast nearest the class median skill
    examples = []
    for g in groups:
        first = [x for x in rows if x["superclass"] == g and x["horizon"] == args.horizons[0]]
        cand = [x for x in first if x["n_points"] >= args.example_min_points] or first
        if not cand:
            continue
        sk = np.array([x["scores"]["model"]["skill"] for x in cand])
        examples.append((g, cand[int(np.argmin(np.abs(sk - np.median(sk))))]))
    examples = examples[: args.n_examples]
    if examples:
        for _, ex in examples:  # the decoder's curves on a fine grid, in time and per period
            d = ex["draw"]
            zs, bands = d["zs"].to(dev), [int(b) for b in np.unique(d["band"])]
            ctx = None if d["ctx"] is None else tuple(x.to(dev) for x in d["ctx"])
            W = float(cfg.window)
            d["dense"] = dense_curves(dec, spec, zs, ex["start"], W, bands, d["err_med"], args.n_dense, ctx=ctx)
            d["dense_cat"] = dense_curves(dec, spec, zs, ex["start"], W, bands, d["err_med"], args.n_dense, ex["catalogue_period"], ctx)
            d["dense_mod"] = dense_curves(dec, spec, zs, ex["start"], W, bands, d["err_med"], args.n_dense, ex["model_period"], ctx)
            zt = d["z_true"].to(dev)
            d["dense_true"] = dense_curves(dec, spec, zt, ex["start"], W, bands, d["err_med"], args.n_dense, ctx=ctx)
            d["dense_cat_true"] = dense_curves(dec, spec, zt, ex["start"], W, bands, d["err_med"], args.n_dense, ex["catalogue_period"], ctx)
            d["dense_mod_true"] = dense_curves(dec, spec, zt, ex["start"], W, bands, d["err_med"], args.n_dense, ex["model_period"], ctx)
        plot_examples(out / "examples.png", examples, float(cfg.window))
        plot_folded(out / "examples_folded.png", examples)
    b = tables["by horizon"][f"{args.horizons[0]} window ahead"]
    print(
        f"1 window ahead ({b['n']} forecasts), median NLL per point: model {b['model']['nll']:.3f} | nothing changes "
        f"{b['persist']['nll']:.3f} | constant {b['constant']['nll']:.3f} | GP {b['gp']['nll']:.3f} | LS fold "
        f"{b['ls_fold']['nll']:.3f} | catalogue fold {b['oracle_fold']['nll']:.3f}; skill model {b['model']['skill']:.3f}, "
        f"LS fold {b['ls_fold']['skill']:.3f}, catalogue fold {b['oracle_fold']['skill']:.3f}, own latent {b['encode_decode']['skill']:.3f} "
        f"(NLL {b['encode_decode']['nll']:.3f}); latent distances: draws {b['latent']['d_sample']:.3f}, mean of draws "
        f"{b['latent']['d_sample_mean']:.3f}, nothing changes {b['latent']['d_persist']:.3f}, train mean {b['latent']['d_train_mean']:.3f}, "
        f"spread {b['latent']['spread']:.3f}; the model beats "
        f"'nothing changes' in {b['beats_persist']:.0%}, the GP in {b['beats_gp']:.0%}, the LS fold in {b['beats_ls_fold']:.0%}"
    )
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return res


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
