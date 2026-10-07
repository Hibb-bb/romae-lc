"""Period regression from the frozen latents of a whole light curve, next to
Lomb-Scargle, with the failure cases mined.

The question: why is the period still not fully learnable from the stage-1
latents, and where does it fail? The script pools the cached window latents
of every object (``cache_latents.py``, realisation 0 only), fits regressors
on log10 period (train split) and scores them on the validation split with
the numbers astronomers use (the share of objects whose period is recovered
within 1 % and 10 %, the alias rate at 2 P and P / 2, the residual scatter in
dex) next to the ridge R2 the probes report. A generalised Lomb-Scargle
(floating mean, all bands together) on one window of the checkpoint's length
and on the whole light curve is the classic reference. The residuals of the
best model and of the window Lomb-Scargle are then binned by class, period,
cycles per window, points per window, number of windows and amplitude to
show where each one fails.

Feature sets per object:

- ``mean``: the mean of the object's valid window latents (``n_tokens >=
  meta.min_tokens``); ``meanmax``: the mean and the max over windows;
- ``hand``: :func:`project.common.hand_features` of the record (no model);
- ``seq_last``, ``seq_mean`` (with ``--pred``): the hidden states of a
  sequence predictor (``train_predictor.load_predictor``, ``model.states``)
  at the last valid window and averaged over the valid windows. A predictor
  without a sequence trunk skips these with a message.

Regressors, all on log10 period: ridge (alpha 0.1, standardised features), a
small MLP, and a bin classifier over ``--bins`` log-period bins spanning the
training range (the prediction is the centre of the best bin; the second
best bin is kept as the alias candidate).

Lomb-Scargle: ``--n-ls`` seeded validation objects. Frequencies run from
``1 / (2 span)`` to ``2 / P_min`` (``P_min`` the shortest training period, or
``--ls-pmin``; twice its frequency so the half-period alias of an eclipsing
binary is inside the grid) with ``--ls-oversample`` points per ``1 / span``,
capped at ``--ls-cap`` frequencies. The estimate is the highest peak;
``ls_top5`` picks the one of the five highest peaks closest to the catalogue
period (an oracle: is the true period among the candidates at all?). An
eclipsing binary has two dips per orbit, so Lomb-Scargle usually finds P / 2;
that counts as an alias, not a hit.

Outputs in ``--out``: ``results.json`` (everything), ``tables.md`` (the
tables), ``worst.csv`` (the 100 worst validation objects of the best model),
``log.jsonl`` and the figures ``pred_vs_true_model.png``,
``pred_vs_true_ls_window.png``, ``resid_vs_cycles.png``,
``resid_vs_points.png``, ``ratio_hist.png``.

    python -m project.period_probe --latents project/runs/maew_w250/latents_r4.pt \\
        --out project/results/period_maew
    python -m project.period_probe --latents project/runs/maew_w250/latents_r4.pt \\
        --out project/results/period_maew --pred project/runs/pred_maew_seq/pred.pt \\
        --n-ls 2000 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from project import fold
from project.common import (
    HEADLINE,
    JsonlLog,
    add_device_arg,
    data_args_from,
    dump_json,
    fine_class,
    get_device,
    hand_features,
    load_data,
    load_encoder,
    seed_all,
    superclass,
)

FEATURE_SETS = ("hand", "mean", "meanmax", "multi", "seq_last", "seq_mean")
REGRESSORS = ("ridge", "mlp", "bins", "joint", "winjoint")
LS_METHODS = ("ls_window", "ls_full", "ls_top5")

# Fixed colours per headline superclass (colour-blind safe, never cycled);
# every other superclass shares the grey.
COLOURS = {
    "ROT": "#2a78d6",
    "RR": "#eb6834",
    "ECL": "#1baf7a",
    "CEP": "#eda100",
    "DSCT": "#e87ba4",
    "LPV": "#008300",
}
OTHER_COLOUR = "#8a8a85"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--latents", required=True, help="latents.pt of cache_latents")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--pred", default=None, help="pred.pt of a sequence predictor")
    p.add_argument(
        "--extra-latents",
        nargs="*",
        default=None,
        help="caches of the same encoder at other window lengths; the feature "
        "set 'multi' joins the mean latent of every length",
    )
    p.add_argument(
        "--ckpt", default=None, help="the encoder checkpoint (default: the cache's)"
    )
    g = p.add_argument_group("data (default: the checkpoint's)")
    g.add_argument("--data", default=None)
    g.add_argument("--max-rows", type=int, default=None)
    g.add_argument("--n-sim", type=int, default=None)
    g = p.add_argument_group("regressors")
    g.add_argument("--mlp-steps", type=int, default=3000)
    g.add_argument("--mlp-hidden", type=int, default=256)
    g.add_argument("--mlp-act", choices=tuple(MLP_ACT), default="silu", help="hidden activation of every MLP read-out")
    g.add_argument("--batch-size", type=int, default=1024)
    g.add_argument("--bins", type=int, default=240, help="log-period bins")
    g.add_argument("--alpha", type=float, default=0.1, help="ridge penalty")
    g = p.add_argument_group("Lomb-Scargle reference")
    g.add_argument("--n-ls", type=int, default=2000, help="validation objects")
    g.add_argument("--ls-oversample", type=float, default=10.0)
    g.add_argument("--ls-cap", type=int, default=200_000, help="max frequencies")
    g.add_argument(
        "--ls-pmin",
        type=float,
        default=None,
        help="shortest period searched (default: the shortest training period)",
    )
    g.add_argument("--ls-peaks", type=int, default=5, help="peaks kept for the oracle")
    g = p.add_argument_group("phase-fold test (project.fold)")
    g.add_argument("--fold-harmonics", type=int, default=3, help="harmonics of the wave")
    g.add_argument("--no-fold", action="store_true", help="skip the phase-fold test")
    g.add_argument("--no-refine", action="store_true", help="skip the fine search near the model's period")
    g.add_argument("--refine-rel", type=float, default=0.1, help="the search range, as a share of the period")
    g.add_argument("--refine-oversample", type=float, default=5.0, help="trial periods per 1 / baseline")
    p.add_argument("--worst", type=int, default=100, help="rows of worst.csv")
    p.add_argument("--seed", type=int, default=0)
    add_device_arg(p)
    return p.parse_args(argv)


# --------------------------------------------------------------------- objects


def class_order(counts: dict) -> list:
    """Headline superclasses first, then the rest by count."""
    order = [g for g in HEADLINE if g in counts]
    return order + sorted((g for g in counts if g not in HEADLINE), key=lambda g: -counts[g])


def pool_latents(cache: dict, split: str, min_tokens: int) -> dict:
    """Per object of ``split``: the mean and max of its valid window latents
    (realisation 0), the number of valid windows, the mean points per valid
    window, the start of the first valid window and the valid rows
    themselves. Objects without a valid window are left out (``keep`` says
    which cache objects survived)."""
    ptr = cache["ptr"][split].numpy()
    n_obj = len(ptr) - 1
    z = cache["z"].numpy()
    n_tok = cache["n_tokens"].numpy().astype(np.int64)
    start = cache["start"].numpy().astype(np.float64)
    keep = np.zeros(n_obj, dtype=bool)
    mean, mx, n_valid, pts, first, rows = [], [], [], [], [], []
    for i in range(n_obj):
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        ok = np.flatnonzero(n_tok[lo:hi] >= min_tokens) + lo
        if ok.size == 0:
            continue
        keep[i] = True
        zi = z[ok].astype(np.float32)
        mean.append(zi.mean(0))
        mx.append(zi.max(0))
        n_valid.append(ok.size)
        pts.append(float(n_tok[ok].mean()))
        first.append(start[ok[0]])
        rows.append(ok)
    return dict(
        keep=keep,
        mean=np.stack(mean) if mean else np.zeros((0, z.shape[1]), np.float32),
        max=np.stack(mx) if mx else np.zeros((0, z.shape[1]), np.float32),
        n_valid=np.array(n_valid, dtype=np.int64),
        points=np.array(pts, dtype=np.float64),
        first_start=np.array(first, dtype=np.float64),
        rows=rows,
    )


def amplitude(record) -> float:
    """Robust range of the normalised magnitudes: 90th minus 10th percentile."""
    y = record.y.astype(np.float64)
    return float(np.percentile(y, 90) - np.percentile(y, 10)) if y.size else 0.0


def object_table(cache: dict, split: str, records, bands, min_tokens: int) -> dict:
    """Everything the regressors and the tables need for one split, one row
    per kept object: the pooled latents, the hand features, the target and
    the binning variables."""
    pooled = pool_latents(cache, split, min_tokens)
    objs = cache["objects"][split]
    keep = pooled["keep"]
    index = objs["index"].numpy()[keep]
    period = objs["period"].numpy().astype(np.float64)[keep]
    recs = [records[int(i)] for i in index]
    good = np.isfinite(period) & (period > 0)
    if not good.all():  # an object without a catalogue period cannot be scored
        sel = np.flatnonzero(good)
        recs = [recs[i] for i in sel]
        for k in ("mean", "max", "n_valid", "points", "first_start"):
            pooled[k] = pooled[k][sel]
        pooled["rows"] = [pooled["rows"][i] for i in sel]
        index, period = index[sel], period[sel]
    return dict(
        split=split,
        index=index,
        records=recs,
        period=period,
        logp=np.log10(period),
        superclass=np.array([superclass(r) for r in recs]),
        fine=np.array([fine_class(r) for r in recs]),
        ids=[str(r.meta.get("id") or f"{split}:{int(i)}") for r, i in zip(recs, index)],
        n_valid=pooled["n_valid"],
        points=pooled["points"],
        first_start=pooled["first_start"],
        rows=pooled["rows"],
        amp=np.array([amplitude(r) for r in recs]),
        features={
            "mean": pooled["mean"],
            "meanmax": np.concatenate([pooled["mean"], pooled["max"]], 1),
            "hand": np.stack([hand_features(r, bands) for r in recs]).astype(np.float32),
        },
        n_dropped=int((~keep).sum() + (~good).sum()),
    )


# ------------------------------------------------------- many window lengths


def multi_features(paths, tables: dict) -> None:
    """Add the feature set ``multi`` to every table: the mean latent of the
    main cache joined with the mean latent of every cache in ``paths`` (the
    same encoder at other window lengths). A star without a valid window at
    some length gets zeros there, and one extra column per length says
    whether the star had a window of that length."""
    for tab in tables.values():
        tab["features"]["multi"] = tab["features"]["mean"]
    for path in paths:
        extra = torch.load(path, map_location="cpu", weights_only=False)
        mt = int(extra["meta"]["min_tokens"])
        for split, tab in tables.items():
            pooled = pool_latents(extra, split, mt)
            idx = extra["objects"][split]["index"].numpy()[pooled["keep"]]
            where = {int(i): j for j, i in enumerate(idx)}
            d = pooled["mean"].shape[1]
            rows = np.zeros((len(tab["index"]), d + 1), dtype=np.float32)
            for r, i in enumerate(tab["index"]):
                j = where.get(int(i))
                if j is not None:
                    rows[r, :d], rows[r, d] = pooled["mean"][j], 1.0
            tab["features"]["multi"] = np.concatenate([tab["features"]["multi"], rows], 1)
        print(f"  joined {path} (window {float(extra['meta']['window']):g} d)")


# --------------------------------------------------------- sequence features


@torch.no_grad()
def seq_features(pred_path, cache: dict, tables: dict, window: float, device) -> bool:
    """Add ``seq_last`` and ``seq_mean`` to every table from the hidden
    states of a sequence predictor. Returns False (with a message) when the
    predictor cannot provide them."""
    try:
        from project.train_predictor import crop_last, load_predictor

        model, _ = load_predictor(pred_path, device)
    except (ImportError, AttributeError) as e:
        print(f"sequence features skipped: cannot load {pred_path} ({e})")
        return False
    if getattr(model, "arch", "mlp") != "seq" or not hasattr(model, "states"):
        print(
            f"sequence features skipped: {pred_path} is not a sequence predictor "
            f"(arch {getattr(model, 'arch', 'mlp')!r}); only cache features are used"
        )
        return False
    z_all = cache["z"]
    start = cache["start"].numpy().astype(np.float64)
    for tab in tables.values():
        rows = tab["rows"]
        n = len(rows)
        last, mean = [None] * n, [None] * n
        order = np.argsort([len(r) for r in rows])
        for lo in range(0, n, 256):
            idx = order[lo : lo + 256]
            T = max(len(rows[i]) for i in idx)
            z = torch.zeros(len(idx), T, z_all.shape[1])
            gaps = torch.zeros(len(idx), T)
            mask = torch.zeros(len(idx), T, dtype=torch.bool)
            for b, i in enumerate(idx):
                r = rows[i]
                z[b, : len(r)] = z_all[r].float()
                s = start[r]
                gaps[b, 1 : len(r)] = torch.from_numpy((s[1:] - s[:-1]) / window).float()
                mask[b, : len(r)] = True
            # the core accepts at most max_len windows: keep the last ones
            z, gaps, mask = crop_last(z, gaps, mask, int(model.max_len))
            h = model.states(z.to(device), gaps.to(device), mask.to(device)).float().cpu()
            m = mask[..., None].float()
            hm = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
            n_real = mask.sum(1)
            for b, i in enumerate(idx):
                last[i] = h[b, int(n_real[b]) - 1].numpy()
                mean[i] = hm[b].numpy()
        tab["features"]["seq_last"] = np.stack(last).astype(np.float32)
        tab["features"]["seq_mean"] = np.stack(mean).astype(np.float32)
    return True


# ------------------------------------------------------------------ regressors


def _standardize(x_tr, x_va):
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    return (x_tr - mu) / sd, (x_va - mu) / sd


def fit_ridge(x_tr, y_tr, x_va, alpha: float) -> np.ndarray:
    """Ridge on standardised features (the probe's regressor); returns the
    validation predictions."""
    x_tr, x_va = (x.astype(np.float64) for x in _standardize(x_tr, x_va))
    y_tr = np.asarray(y_tr, dtype=np.float64)
    n, d = x_tr.shape
    a = x_tr.T @ x_tr / n + alpha * np.eye(d)
    w = np.linalg.solve(a, x_tr.T @ (y_tr - y_tr.mean()) / n)
    return y_tr.mean() + x_va @ w


MLP_ACT = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}
_MLP_ACT = ["silu"]  # set by --mlp-act; a module-level switch so every regressor builds the same way


class Mlp(nn.Module):
    """Two hidden layers of ``hidden`` with the activation of ``--mlp-act``
    (SiLU by default; ReLU makes the network piecewise linear, so it
    extrapolates along straight lines beyond the training range). The
    output layer is always linear."""

    def __init__(self, d_in: int, d_out: int, hidden: int):
        super().__init__()
        act = MLP_ACT[_MLP_ACT[0]]
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            act(),
            nn.Linear(hidden, hidden),
            act(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def _train_mlp(x_tr, target, d_out, loss_kind, steps, batch, hidden, device, seed):
    """Fit an :class:`Mlp` with AdamW 1e-3 on random batches; ``target`` is
    a float tensor (mse) or a long tensor (ce)."""
    torch.manual_seed(seed)
    x = torch.as_tensor(x_tr, dtype=torch.float32, device=device)
    t = target.to(device)
    model = Mlp(x.shape[1], d_out, hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=gen).to(device)
        out = model(x[idx])
        if loss_kind == "mse":
            loss = F.mse_loss(out[:, 0], t[idx])
        else:
            loss = F.cross_entropy(out, t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def fit_mlp(x_tr, y_tr, x_va, steps, batch, hidden, device, seed) -> np.ndarray:
    """The MLP regressor on standardised inputs and target; validation
    predictions in log10 period."""
    x_tr, x_va = _standardize(x_tr, x_va)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    mu, sd = y_tr.mean(), y_tr.std() + 1e-6
    target = torch.as_tensor((y_tr - mu) / sd, dtype=torch.float32)
    with torch.enable_grad():
        model = _train_mlp(x_tr, target, 1, "mse", steps, batch, hidden, device, seed)
    out = model(torch.as_tensor(x_va, dtype=torch.float32, device=device))[:, 0]
    return out.cpu().numpy().astype(np.float64) * sd + mu


@torch.no_grad()
def fit_bins(x_tr, y_tr, x_va, n_bins, steps, batch, hidden, device, seed, k: int = 5):
    """The bin classifier: ``n_bins`` equal log-period bins over the training
    range. Returns ``(best, second, top)``: the centres of the best and the
    second best bins of every validation object (plus the offset model's
    shift), and the ``k`` best as ``[n, k]``."""
    x_tr, x_va = _standardize(x_tr, x_va)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    lo, hi = y_tr.min(), y_tr.max()
    if hi <= lo:
        hi = lo + 1e-3
    edges = np.linspace(lo, hi, n_bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    b = np.clip(np.searchsorted(edges, y_tr, side="right") - 1, 0, n_bins - 1)
    with torch.enable_grad():
        model = _train_mlp(
            x_tr, torch.as_tensor(b, dtype=torch.long), n_bins, "ce", steps, batch,
            hidden, device, seed,
        )  # fmt: skip
        # A bin is a few percent wide, so its centre cannot be right to 1 %.
        # A second small model predicts the offset of the period inside its
        # bin, in bin widths, from the features.
        width = float(edges[1] - edges[0])
        offset = torch.as_tensor((y_tr - centres[b]) / width, dtype=torch.float32)
        fine = _train_mlp(x_tr, offset, 1, "mse", steps, batch, hidden, device, seed + 1)
    xv = torch.as_tensor(x_va, dtype=torch.float32, device=device)
    logits = model(xv).cpu()
    shift = fine(xv)[:, 0].clamp(-0.5, 0.5).cpu().numpy().astype(np.float64) * width
    top = logits.topk(min(k, n_bins), dim=1).indices.numpy()
    return centres[top[:, 0]] + shift, centres[top[:, 1]] + shift, centres[top] + shift[:, None]


def _joint_train(x_tr, coord_tr, n_bins, steps, batch, hidden, device, seed):
    """One head with a logit and an offset per bin: cross-entropy on the
    bin plus the squared offset error at the true bin. ``coord_tr`` is the
    continuous bin coordinate of the target."""
    torch.manual_seed(seed)
    x = torch.as_tensor(x_tr, dtype=torch.float32, device=device)
    c = torch.as_tensor(coord_tr, dtype=torch.float32, device=device)
    b = c.round().long().clamp(0, n_bins - 1)
    off = c - b.float()
    model = Mlp(x.shape[1], 2 * n_bins, hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    with torch.enable_grad():
        for _ in range(steps):
            idx = torch.randint(0, n, (min(batch, n),), generator=gen).to(device)
            out = model(x[idx])
            logits, offsets = out[:, :n_bins], out[:, n_bins:]
            loss = F.cross_entropy(logits, b[idx]) + F.mse_loss(offsets.gather(1, b[idx, None])[:, 0], off[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def _joint_predict(model, x, n_bins, device, groups=None, k=5):
    """Bin coordinate per row (``groups`` None) or per group: the group's
    log-probabilities summed over its rows, the offset averaged at the
    chosen bin. ``_joint_predict.top`` keeps the ``k`` best coordinates
    per row (with their offsets) of the last call, best first."""
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    outs = torch.cat([model(x[i : i + 65536]) for i in range(0, len(x), 65536)])
    lp, off = F.log_softmax(outs[:, :n_bins], 1), outs[:, n_bins:].clamp(-0.5, 0.5)
    if groups is None:
        best = lp.argmax(1)
        topk = lp.topk(min(k, n_bins), dim=1).indices
        _joint_predict.top = (topk.float() + off.gather(1, topk)).cpu().numpy().astype(np.float64)
        return (best.float() + off.gather(1, best[:, None])[:, 0]).cpu().numpy().astype(np.float64)
    coords = np.full(len(groups), np.nan)
    for g, rows in enumerate(groups):
        if len(rows) == 0:
            continue
        r = torch.as_tensor(rows, device=device)
        best = int(lp[r].sum(0).argmax())
        coords[g] = best + float(off[r, best].mean())
    return coords


def fit_joint(x_tr, y_tr, x_va, n_bins, steps, batch, hidden, device, seed):
    """The joint read-out on pooled features: one head gives the bin
    distribution and the offset inside every bin; the chosen bin carries
    its own offset. Returns predictions in log10 period."""
    x_tr, x_va = _standardize(x_tr, x_va)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    lo, hi = y_tr.min(), y_tr.max()
    width = (hi - lo) / n_bins if hi > lo else 1e-3
    model = _joint_train(x_tr, (y_tr - lo) / width - 0.5, n_bins, steps, batch, hidden, device, seed)
    pred = lo + (_joint_predict(model, x_va, n_bins, device) + 0.5) * width
    fit_joint.top = lo + (_joint_predict.top + 0.5) * width  # [n, k] log10 periods, best first
    return pred


def fit_winjoint(z_all, rows_tr, y_tr, rows_va, n_bins, steps, batch, hidden, device, seed, max_rows=400_000):
    """The joint read-out trained per WINDOW (every valid window of a
    training object, the object's period as its target) and read out per
    object by summing the windows' log-probabilities; the pooled latent is
    never formed. ``rows_*`` are the cache rows of every object's valid
    windows."""
    rng = np.random.default_rng(seed)
    r_tr = np.concatenate(rows_tr)
    t_tr = np.concatenate([np.full(len(r), y, dtype=np.float64) for r, y in zip(rows_tr, y_tr)])
    if len(r_tr) > max_rows:
        sel = rng.choice(len(r_tr), max_rows, replace=False)
        r_tr, t_tr = r_tr[sel], t_tr[sel]
    r_va = np.concatenate(rows_va)
    groups, start = [], 0
    for r in rows_va:
        groups.append(np.arange(start, start + len(r)))
        start += len(r)
    x_tr = z_all[torch.as_tensor(r_tr)].float().numpy()
    x_va = z_all[torch.as_tensor(r_va)].float().numpy()
    x_tr, x_va = _standardize(x_tr, x_va)
    lo, hi = t_tr.min(), t_tr.max()
    width = (hi - lo) / n_bins if hi > lo else 1e-3
    model = _joint_train(x_tr, (t_tr - lo) / width - 0.5, n_bins, steps, batch, hidden, device, seed)
    return lo + (_joint_predict(model, x_va, n_bins, device, groups) + 0.5) * width


# --------------------------------------------------------------------- scoring


def recovery(p_pred, p_true):
    """``(ratio, hit1, hit10, alias)``: the ratio ``P_pred / P_true``, hits
    within 1 % and 10 %, and the alias flag (within 10 % of ``2 P`` or ``P /
    2`` but not of ``P``)."""
    ratio = np.asarray(p_pred, dtype=np.float64) / np.asarray(p_true, dtype=np.float64)
    ratio = np.where(np.isfinite(ratio), ratio, np.inf)
    hit1 = np.abs(ratio - 1) < 0.01
    hit10 = np.abs(ratio - 1) < 0.10
    alias = ~hit10 & ((np.abs(ratio / 2 - 1) < 0.1) | (np.abs(ratio * 2 - 1) < 0.1))
    return ratio, hit1, hit10, alias


def hit_within(p_pred, p_true, tol: float) -> np.ndarray:
    """Hits within ``tol`` (relative) of the true period."""
    ratio = np.asarray(p_pred, dtype=np.float64) / np.asarray(p_true, dtype=np.float64)
    return np.abs(np.where(np.isfinite(ratio), ratio, np.inf) - 1) < tol


def _r2(resid, y) -> float:
    ss = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - (resid**2).sum() / ss) if ss > 0 else float("nan")


def score(y_pred, y_true, groups, class_means, order, min_n: int = 10) -> dict:
    """Recovery rates, residual scatter and R2 of predictions ``y_pred`` of
    log10 period. ``r2_within`` measures the residual against the spread of
    the target around its superclass mean (``class_means``, from the train
    split): the null is predicting every object's superclass mean, as in
    ``probe_metrics``. Per superclass with at least ``min_n`` objects."""
    y_pred = np.where(np.isfinite(y_pred), y_pred, y_true.mean())
    r = y_pred - y_true
    _, h1, h10, alias = recovery(10.0**y_pred, 10.0**y_true)
    h20 = hit_within(10.0**y_pred, 10.0**y_true, 0.20)
    glob = float(np.mean(list(class_means.values()))) if class_means else 0.0
    c = y_true - np.array([class_means.get(g, glob) for g in groups])

    def block(m, need=2):
        return dict(
            n=int(m.sum()),
            r2=_r2(r[m], y_true[m]) if m.sum() >= need else float("nan"),
            med_abs=float(np.median(np.abs(r[m]))),
            rec1=float(h1[m].mean()),
            rec10=float(h10[m].mean()),
            rec20=float(h20[m].mean()),
            alias=float(alias[m].mean()),
        )

    out = block(np.ones_like(r, dtype=bool))
    out["r2_within"] = _r2(r, c) if c.std() > 0 else float("nan")
    out["by_superclass"] = {
        g: block(groups == g, min_n) for g in order if (groups == g).any()
    }
    return out


def describe(s: dict, order) -> str:
    by = s["by_superclass"]
    per = " ".join(
        f"{g} {by[g]['rec10']:.2f}/{by[g]['n']}" for g in order if g in by
    )
    return (
        f"rec1 {s['rec1']:.3f} rec10 {s['rec10']:.3f} alias {s['alias']:.3f} "
        f"med|r| {s['med_abs']:.3f} dex R2 {s['r2']:.3f} within {s['r2_within']:.3f} "
        f"| rec10 by class {per}"
    )


# ---------------------------------------------------------------- Lomb-Scargle


def gls_power(t, y, w, freqs, device="cpu", chunk_elems: int = 4_000_000):
    """Generalised Lomb-Scargle power (floating mean, weights ``w``) at
    ``freqs`` (cycles per day), in ``[0, 1]``; frequencies in chunks so the
    ``n_freq x n_points`` arrays fit. float64 throughout: the phase of a
    2000 d baseline at 40 cycles per day needs it."""
    t = torch.as_tensor(np.asarray(t, dtype=np.float64), device=device)
    y = torch.as_tensor(np.asarray(y, dtype=np.float64), device=device)
    w = torch.as_tensor(np.asarray(w, dtype=np.float64), device=device)
    w = w / w.sum()
    y = y - (w * y).sum()
    yy = (w * y * y).sum()
    freqs = torch.as_tensor(np.asarray(freqs, dtype=np.float64), device=device)
    n = t.numel()
    step = max(1, chunk_elems // max(n, 1))
    out = []
    for lo in range(0, freqs.numel(), step):
        ph = (2 * math.pi) * freqs[lo : lo + step, None] * t[None, :]
        c, s = torch.cos(ph), torch.sin(ph)
        wc, ws = w * c, w * s
        C, S = wc.sum(1), ws.sum(1)
        yc, ys = (wc * y).sum(1), (ws * y).sum(1)
        cc = (wc * c).sum(1) - C * C
        ss = (ws * s).sum(1) - S * S
        cs = (wc * s).sum(1) - C * S
        d = cc * ss - cs * cs
        p = (ss * yc * yc + cc * ys * ys - 2 * cs * yc * ys) / (yy * d)
        out.append(torch.where(torch.isfinite(p), p, torch.zeros_like(p)))
    return torch.cat(out).cpu().numpy()


def freq_grid(span: float, p_min: float, oversample: float, cap: int) -> np.ndarray:
    """From ``1 / (2 span)`` to ``2 / p_min`` with ``oversample`` points per
    ``1 / span``, at most ``cap`` frequencies (then coarser)."""
    f_lo, f_hi = 1.0 / (2.0 * span), 2.0 / p_min
    if f_hi <= f_lo:
        f_hi = 2.0 * f_lo
    n = int((f_hi - f_lo) * oversample * span) + 1
    return np.linspace(f_lo, f_hi, min(max(n, 2), cap))


def top_peaks(power: np.ndarray, freqs: np.ndarray, k: int) -> np.ndarray:
    """The frequencies of the ``k`` highest local maxima, highest first
    (padded with the global maximum when there are fewer peaks)."""
    if power.size < 3:
        return np.repeat(freqs[np.argmax(power)], k)
    inner = (power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])
    idx = np.flatnonzero(inner) + 1
    if idx.size == 0:
        idx = np.array([int(np.argmax(power))])
    idx = idx[np.argsort(-power[idx])][:k]
    if idx.size < k:
        idx = np.concatenate([idx, np.repeat(int(np.argmax(power)), k - idx.size)])
    return freqs[idx]


def ls_periods(t, y, err, band, p_min, oversample, cap, k, device) -> dict:
    """Lomb-Scargle on one series: every band's median subtracted, all points
    together, weights ``1 / err^2``. Returns ``best`` (days), ``top`` (the
    ``k`` best peaks, days), ``n`` points and ``n_freq``; NaN when the series
    is too short."""
    t, y = np.asarray(t, dtype=np.float64), np.asarray(y, dtype=np.float64).copy()
    err, band = np.asarray(err, dtype=np.float64), np.asarray(band)
    span = float(t.max() - t.min()) if t.size else 0.0
    if t.size < 4 or span <= 0:
        return dict(best=float("nan"), top=[float("nan")] * k, n=int(t.size), n_freq=0)
    for b in np.unique(band):
        m = band == b
        y[m] -= np.median(y[m])
    w = np.where(err > 0, 1.0 / np.maximum(err, 1e-12) ** 2, 0.0)
    if not (w > 0).any():
        w = np.ones_like(t)
    freqs = freq_grid(span, p_min, oversample, cap)
    power = gls_power(t, y, w, freqs, device)
    peaks = top_peaks(power, freqs, k)
    return dict(best=float(1.0 / peaks[0]), top=(1.0 / peaks).tolist(), n=int(t.size), n_freq=int(freqs.size))


def ls_reference(tab: dict, sel: np.ndarray, window: float, p_min: float, args, device) -> dict:
    """Lomb-Scargle on the first valid window and on the whole light curve of
    the selected validation objects: the predictions in log10 period of
    ``ls_window``, ``ls_full`` and ``ls_top5`` (the oracle over the top peaks
    of the full curve), plus the grid sizes."""
    n = len(sel)
    out = {m: np.full(n, np.nan) for m in LS_METHODS}
    top_window = np.full(n, np.nan)
    n_freq = dict(ls_window=[], ls_full=[])
    t0 = time.time()
    for j, i in enumerate(sel):
        r = tab["records"][int(i)]
        t = r.t.astype(np.float64)
        s = tab["first_start"][i]
        m = (t >= s) & (t < s + window)
        win = ls_periods(t[m], r.y[m], r.err[m], r.band[m], p_min, args.ls_oversample, args.ls_cap, args.ls_peaks, device)
        full = ls_periods(t, r.y, r.err, r.band, p_min, args.ls_oversample, args.ls_cap, args.ls_peaks, device)
        p_true = tab["period"][i]
        out["ls_window"][j] = np.log10(win["best"])
        out["ls_full"][j] = np.log10(full["best"])
        top = np.asarray(full["top"], dtype=np.float64)
        out["ls_top5"][j] = np.log10(top[np.argmin(np.abs(np.log10(top / p_true)))])
        topw = np.asarray(win["top"], dtype=np.float64)
        top_window[j] = np.log10(topw[np.argmin(np.abs(np.log10(topw / p_true)))])
        n_freq["ls_window"].append(win["n_freq"])
        n_freq["ls_full"].append(full["n_freq"])
        if (j + 1) % 200 == 0:
            print(f"  Lomb-Scargle {j + 1} / {n} objects, {time.time() - t0:.0f}s", flush=True)
    out["ls_window_top5"] = top_window
    return dict(
        pred=out,
        n_freq_median={k: float(np.median(v)) if v else float("nan") for k, v in n_freq.items()},
        seconds=time.time() - t0,
    )


# ------------------------------------------------------------ failure mining


PERIOD_EDGES = [0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
CYCLE_EDGES = [10, 30, 100, 300, 1000]
POINT_EDGES = [30, 60, 120]
WINDOW_EDGES = [10, 20, 40]


def edge_labels(edges, unit: str = "") -> list[str]:
    labels = [f"<{edges[0]:g}{unit}"]
    labels += [f"{a:g}-{b:g}{unit}" for a, b in zip(edges[:-1], edges[1:])]
    labels.append(f">{edges[-1]:g}{unit}")
    return labels


def bin_by_edges(x, edges, unit: str = ""):
    """``(labels per value, ordered label list)`` for numeric bins."""
    labels = edge_labels(edges, unit)
    idx = np.searchsorted(np.asarray(edges, dtype=np.float64), x, side="right")
    return np.array([labels[i] for i in idx]), labels


def bin_by_quartile(x, ref):
    """Quartile bins of ``x`` with the cuts taken from ``ref``."""
    q = np.percentile(ref, [25, 50, 75])
    labels = [f"Q1 <{q[0]:.2f}", f"Q2 {q[0]:.2f}-{q[1]:.2f}", f"Q3 {q[1]:.2f}-{q[2]:.2f}", f"Q4 >{q[2]:.2f}"]
    idx = np.searchsorted(q, x, side="right")
    return np.array([labels[i] for i in idx]), labels


def binned_table(r, h1, h10, labels, order) -> list[dict]:
    """One row per bin: n, median |r|, the 1 % and 10 % recovery rates."""
    rows = []
    for lab in order:
        m = labels == lab
        if not m.any():
            continue
        rows.append(
            dict(bin=lab, n=int(m.sum()), med_abs=float(np.median(np.abs(r[m]))),
                 rec1=float(h1[m].mean()), rec10=float(h10[m].mean()))
        )  # fmt: skip
    return rows


def failure_tables(tab: dict, sel: np.ndarray, y_pred: np.ndarray, window: float, amp_ref) -> dict:
    """The binned tables of one method on the objects ``sel`` of ``tab``."""
    y_true = tab["logp"][sel]
    y_pred = np.where(np.isfinite(y_pred), y_pred, y_true.mean())
    r = y_pred - y_true
    _, h1, h10, _ = recovery(10.0**y_pred, 10.0**y_true)
    period = tab["period"][sel]
    groups = tab["superclass"][sel]
    counts = {g: int((groups == g).sum()) for g in set(groups.tolist())}
    fine = tab["fine"][sel]
    fine_counts = {g: int((fine == g).sum()) for g in set(fine.tolist())}
    specs = [
        ("superclass", groups, class_order(counts)),
        ("fine_class", fine, sorted(fine_counts, key=lambda g: -fine_counts[g])),
        ("true_period_d", *bin_by_edges(period, PERIOD_EDGES)),
        ("cycles_per_window", *bin_by_edges(window / period, CYCLE_EDGES)),
        ("points_per_window", *bin_by_edges(tab["points"][sel], POINT_EDGES)),
        ("n_valid_windows", *bin_by_edges(tab["n_valid"][sel], WINDOW_EDGES)),
        ("amplitude", *bin_by_quartile(tab["amp"][sel], amp_ref)),
    ]
    return {name: binned_table(r, h1, h10, labels, order) for name, labels, order in specs}


TABLE_TEXT = {
    "superclass": "Residuals by superclass: where a whole class of stars fails.",
    "fine_class": "Residuals by fine class (class_str), largest classes first.",
    "true_period_d": "Residuals by the catalogue period in days.",
    "cycles_per_window": "Residuals by cycles per window (window / P): how many periods one window holds.",
    "points_per_window": "Residuals by the mean number of points in a valid window.",
    "n_valid_windows": "Residuals by the number of valid windows of the object.",
    "amplitude": "Residuals by amplitude (90th minus 10th percentile of the normalised magnitudes), validation quartiles.",
}


def worst_bins(tables: dict, k: int = 2, min_n: int = 10) -> list[dict]:
    """The ``k`` bins (over every table) with the lowest 10 % recovery."""
    rows = [
        dict(table=name, **row)
        for name, t in tables.items()
        for row in t
        if row["n"] >= min_n
    ]
    rows.sort(key=lambda d: (d["rec10"], -d["med_abs"]))
    return rows[:k]


# --------------------------------------------------------------------- writing


def md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(c for _, c in cols) + " |\n"
    head += "|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k, "")
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def write_tables(path: Path, res: dict, order: list) -> None:
    cols = [("name", "method"), ("n", "n"), ("rec1", "rec 1%"), ("rec10", "rec 10%"), ("rec20", "rec 20%"),
            ("alias", "alias"), ("med_abs", "median abs residual (dex)"), ("r2", "R2"), ("r2_within", "R2 within")]  # fmt: skip
    lines = ["# Period probe\n"]
    lines.append(
        f"Latents `{res['latents']}` (window {res['window']:g} d), "
        f"{res['n_train']} train / {res['n_val']} validation objects "
        f"({res['n_dropped']} without a valid window or period). "
        f"Recovery: |P_pred / P_true - 1| below 1 % and 10 %; alias: within 10 % of "
        f"2 P or P / 2 but not of P. Lomb-Scargle on {res['n_ls']} validation objects.\n"
    )
    lines.append("## Regressors on the validation split\n")
    lines.append("Every feature set and regressor, fitted on the train split. The best model is marked.\n")
    rows = [dict(name=("**" + k + "**" if k == res["best"] else k), **v) for k, v in res["models"].items()]
    lines.append(md_table(rows, cols))
    lines.append("\n## Lomb-Scargle reference\n")
    lines.append(
        "One window of the checkpoint's length (the first valid one), the whole light curve, "
        "and the oracle that picks the best of the top 5 peaks of the whole curve. "
        "Eclipsing binaries have two dips per orbit, so Lomb-Scargle usually finds P / 2 for "
        "them; that lands in the alias column, not in the recovery columns.\n"
    )
    rows = [dict(name=k, **v) for k, v in res["ls"].items()]
    lines.append(md_table(rows, cols))
    lines.append("\n## Recovery per superclass\n")
    lines.append("The 10 % recovery rate (and n) per superclass of the best model, the baseline and Lomb-Scargle.\n")
    methods = {res["best"]: res["models"][res["best"]], "hand/ridge": res["models"].get("hand/ridge", {})}
    methods.update(res["ls"])
    per = []
    for name, s in methods.items():
        by = s.get("by_superclass", {})
        per.append(dict(name=name, **{g: f"{by[g]['rec10']:.2f} ({by[g]['n']})" if g in by else "-" for g in order}))
    lines.append(md_table(per, [("name", "method")] + [(g, g) for g in order]))
    for who, key in (("best model " + res["best"], "failures_model"), ("Lomb-Scargle window", "failures_ls_window")):
        lines.append(f"\n## Failure mining: {who}\n")
        for name, t in res[key].items():
            lines.append(f"\n### {name}\n\n{TABLE_TEXT[name]}\n")
            lines.append(md_table(t, [("bin", "bin"), ("n", "n"), ("med_abs", "median abs residual (dex)"), ("rec1", "rec 1%"), ("rec10", "rec 10%")]))
    if res.get("fold"):
        lines.append("\n## Phase-fold test\n")
        lines.append(
            "Each light curve is folded on a period and a wave (a mean plus "
            f"{res['fold']['harmonics']} harmonics per band) is fitted. R2 is the share of the "
            "brightness change the wave explains. It is computed on the catalogue period, on "
            "the model's period, and on a period that is wrong on purpose (the floor). "
            "'as good' is the share of stars whose fold on the model's period keeps at least "
            "90 % of the R2 of the fold on the catalogue period.\n"
        )
        fcols = [("bin", "bin"), ("n", "n"), ("r2_catalogue", "R2 catalogue"), ("r2_model", "R2 model"),
                 ("r2_wrong", "R2 wrong period"), ("as_good", "as good")]  # fmt: skip
        lines.append("\n### by superclass\n")
        lines.append(md_table(res["fold"]["tables"]["superclass"], fcols))
        lines.append("\n### by the size of the model's period error\n")
        lines.append(md_table(res["fold"]["tables"]["period_error"], fcols))
        ref = res["fold"].get("refined")
        if ref:
            lines.append("\n## Phase-fold test after a fine search\n")
            lines.append(
                "The model's period is a rough one. A fine search tries every period within "
                f"{ref['rel']:.0%} of it (about {ref['trials_median']:.0f} trial periods per star) and keeps "
                "the one that folds best. The range is too narrow to reach half or twice the "
                "period, so that choice stays the model's. 'R2 model' is now the fold on the "
                "refined period. 'R2 wrong period' is the floor after the same search: the "
                "best fold found near a period that is wrong on purpose.\n"
            )
            lines.append("\n### by superclass\n")
            lines.append(md_table(ref["tables"]["superclass"], fcols))
            lines.append("\n### by the size of the refined period's error\n")
            lines.append(md_table(ref["tables"]["period_error"], fcols))
            lines.append("\n### how sharp the period is, before and after the fine search\n")
            rows = [
                dict(bin=a["bin"], n=a["n"], rough=a["share"], refined=b["share"])
                for a, b in zip(ref["precision_rough"], ref["precision_refined"])
            ]
            lines.append(md_table(rows, [("bin", "period is"), ("n", "n"), ("rough", "model alone"), ("refined", "after the fine search")]))
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_worst(path: Path, tab: dict, y_pred: np.ndarray, k: int) -> None:
    r = y_pred - tab["logp"]
    order = np.argsort(-np.abs(r))[:k]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "class", "superclass", "true_period", "pred_period", "ratio", "n_windows", "mean_points", "amplitude"])
        for i in order:
            w.writerow([
                tab["ids"][i], tab["fine"][i], tab["superclass"][i], f"{tab['period'][i]:.6g}",
                f"{10.0 ** y_pred[i]:.6g}", f"{10.0 ** r[i]:.4g}", int(tab["n_valid"][i]),
                f"{tab['points'][i]:.1f}", f"{tab['amp'][i]:.3f}",
            ])  # fmt: skip


# --------------------------------------------------------------------- figures


def _colour(g: str) -> str:
    return COLOURS.get(g, OTHER_COLOUR)


def plot_pred_vs_true(path, y_true, y_pred, groups, order, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    lo, hi = float(np.nanmin(y_true)) - 0.2, float(np.nanmax(y_true)) + 0.2
    ax.plot([lo, hi], [lo, hi], color="#8a8a85", lw=1, label="P_pred = P_true")
    for s, lab in ((np.log10(2), "2 P, P / 2"), (-np.log10(2), None)):
        ax.plot([lo, hi], [lo + s, hi + s], color="#8a8a85", lw=1, ls="--", label=lab)
    for g in order:
        m = groups == g
        if m.any():
            ax.scatter(y_true[m], y_pred[m], s=6, alpha=0.5, lw=0, color=_colour(g), label=f"{g} ({m.sum()})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo - 1, hi + 1)
    ax.set_xlabel("log10 true period (d)")
    ax.set_ylabel("log10 predicted period (d)")
    ax.set_title(title)
    ax.legend(fontsize=8, markerscale=2, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_resid_vs(path, x, r, groups, order, xlabel, title, logx=True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for g in order:
        m = groups == g
        if m.any():
            ax.scatter(x[m], np.abs(r[m]), s=6, alpha=0.4, lw=0, color=_colour(g), label=g)
    # a running median over log bins, the summary the eye needs
    xs = np.log10(np.maximum(x, 1e-9)) if logx else x
    edges = np.linspace(np.nanmin(xs), np.nanmax(xs) + 1e-9, 13)
    mids, meds = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (xs >= a) & (xs < b)
        if m.sum() >= 5:
            mids.append(10 ** (0.5 * (a + b)) if logx else 0.5 * (a + b))
            meds.append(np.median(np.abs(r[m])))
    if mids:
        ax.plot(mids, meds, color="#1a1a19", lw=2, label="median")
    if logx:
        ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("|log10 P_pred - log10 P_true| (dex)")
    ax.set_title(title)
    ax.legend(fontsize=8, markerscale=2, frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ratio_hist(path, ratios: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.logspace(-1.5, 1.5, 121)
    colours = ["#2a78d6", "#eb6834", "#1baf7a"]
    for (name, ratio), c in zip(ratios.items(), colours):
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        ax.hist(np.clip(ratio, bins[0], bins[-1]), bins=bins, histtype="step", lw=1.5, color=c, label=name)
    for v in (0.5, 1.0, 2.0):
        ax.axvline(v, color="#8a8a85", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("P_pred / P_true (aliases peak at 2 and 0.5)")
    ax.set_ylabel("validation objects")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- driver


def _top_candidates(tops: dict, models: dict, best: str) -> dict:
    """``p_top`` for predictions.npz: the best read-out's candidate bins, or,
    when it has none (winjoint), those of the best read-out that has some
    (the sweep and the candidate search need seeds)."""
    if best in tops:
        return {"p_top": 10.0 ** tops[best], "p_top_from": best}
    have = [k for k in tops if k in models]
    if not have:
        return {}
    src = max(have, key=lambda k: (models[k]["rec10"], -models[k]["med_abs"]))
    return {"p_top": 10.0 ** tops[src], "p_top_from": src}


def run(args: argparse.Namespace) -> dict:
    _MLP_ACT[0] = args.mlp_act
    seed_all(args.seed)
    dev = get_device(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = JsonlLog(out / "log.jsonl")
    t0 = time.time()
    cache = torch.load(args.latents, map_location="cpu", weights_only=False)
    meta = cache["meta"]
    window, min_tokens = float(meta["window"]), int(meta["min_tokens"])
    ckpt = args.ckpt or meta["ckpt"]
    _, emeta = load_encoder(ckpt, "cpu")  # only its data arguments are needed
    overrides = argparse.Namespace(data=args.data, max_rows=args.max_rows, n_sim=args.n_sim)
    data = load_data(data_args_from(emeta.args, overrides), splits=("train", "validation"))
    bands = sorted(data.wavelengths.keys())
    tables = {
        s: object_table(cache, s, data[s], bands, min_tokens) for s in ("train", "validation")
    }
    tr, va = tables["train"], tables["validation"]
    print(
        f"latents {args.latents} (dim {meta['dim']}, window {window:g} d, stride "
        f"{meta['stride']:g}, min_tokens {min_tokens}): {len(tr['records'])} train / "
        f"{len(va['records'])} validation objects with a valid window "
        f"({tr['n_dropped'] + va['n_dropped']} dropped); loaded in {time.time() - t0:.0f}s"
    )
    if args.extra_latents:
        multi_features(args.extra_latents, tables)
    if args.pred:
        seq_features(args.pred, cache, tables, window, dev)
    else:
        print("no --pred: sequence features skipped")

    counts = {g: int((va["superclass"] == g).sum()) for g in set(va["superclass"].tolist())}
    order = class_order(counts)
    class_means = {g: float(tr["logp"][tr["superclass"] == g].mean()) for g in set(tr["superclass"].tolist())}

    # --- regressors
    models, preds = {}, {}
    tops = {}  # the k best bins of every bin read-out, for candidate searches
    for feat in FEATURE_SETS:
        if feat not in tr["features"]:
            continue
        x_tr, x_va = tr["features"][feat], va["features"][feat]
        for reg in REGRESSORS:
            t1 = time.time()
            extra = {}
            if reg == "ridge":
                y_pred = fit_ridge(x_tr, tr["logp"], x_va, args.alpha)
            elif reg == "mlp":
                y_pred = fit_mlp(x_tr, tr["logp"], x_va, args.mlp_steps, args.batch_size, args.mlp_hidden, dev, args.seed)
            elif reg == "joint":
                y_pred = fit_joint(x_tr, tr["logp"], x_va, args.bins, args.mlp_steps, args.batch_size, args.mlp_hidden, dev, args.seed)
                tops[f"{feat}/{reg}"] = fit_joint.top
            elif reg == "winjoint":
                if feat != "mean":
                    continue  # the per-window read-out reads the cache rows, one feature set
                y_pred = fit_winjoint(cache["z"], tr["rows"], tr["logp"], va["rows"], args.bins, args.mlp_steps * 2, args.batch_size,
                                      args.mlp_hidden, dev, args.seed)  # fmt: skip
            else:
                y_pred, second, top_k = fit_bins(x_tr, tr["logp"], x_va, args.bins, args.mlp_steps, args.batch_size, args.mlp_hidden, dev, args.seed)
                tops[f"{feat}/{reg}"] = top_k
                _, _, h10_1, _ = recovery(10.0**y_pred, va["period"])
                _, _, h10_2, _ = recovery(10.0**second, va["period"])
                extra = dict(
                    second_bin_rec10=float(h10_2.mean()),
                    second_bin_rescues=float((~h10_1 & h10_2).mean()),
                    best_of_two_rec10=float((h10_1 | h10_2).mean()),
                )
            name = f"{feat}/{reg}"
            s = score(y_pred, va["logp"], va["superclass"], class_means, order)
            s.update(extra, dim=int(x_tr.shape[1]), seconds=time.time() - t1)
            models[name], preds[name] = s, y_pred
            log.write(kind="model", name=name, **{k: v for k, v in s.items() if k != "by_superclass"})
            print(f"  {name:16s} {describe(s, order)}")
    latent = [k for k in models if not k.startswith("hand/")] or list(models)
    best = max(latent, key=lambda k: (models[k]["rec10"], -models[k]["med_abs"]))

    # --- Lomb-Scargle
    n_ls = min(args.n_ls, len(va["records"]))
    sel = np.sort(np.random.default_rng(args.seed).choice(len(va["records"]), n_ls, replace=False))
    p_min = args.ls_pmin or float(tr["period"].min())
    print(f"Lomb-Scargle on {n_ls} validation objects, periods down to {p_min:.4g} d")
    ls = ls_reference(va, sel, window, p_min, args, dev)
    ls_scores = {}
    for m in LS_METHODS:
        ls_scores[m] = score(ls["pred"][m], va["logp"][sel], va["superclass"][sel], class_means, order)
        log.write(kind="ls", name=m, **{k: v for k, v in ls_scores[m].items() if k != "by_superclass"})
        print(f"  {m:16s} {describe(ls_scores[m], order)}")
    ls_scores["ls_window_top5"] = score(ls["pred"]["ls_window_top5"], va["logp"][sel], va["superclass"][sel], class_means, order)
    print(f"  {'ls_window_top5':16s} {describe(ls_scores['ls_window_top5'], order)}")
    # the best model on the same objects, for a like-for-like comparison
    best_on_ls = score(preds[best][sel], va["logp"][sel], va["superclass"][sel], class_means, order)

    # --- failure mining
    fail_model = failure_tables(va, np.arange(len(va["records"])), preds[best], window, va["amp"])
    fail_ls = failure_tables(va, sel, ls["pred"]["ls_window"], window, va["amp"])
    worst_model, worst_ls = worst_bins(fail_model), worst_bins(fail_ls)

    # --- phase-fold test: fold on the model's period and on the catalogue's
    fold_res = None
    if not args.no_fold:
        t1 = time.time()
        p_model = 10.0 ** preds[best]
        r2 = fold.fold_compare(va["records"], p_model, va["period"], args.fold_harmonics)
        fold_res = dict(
            harmonics=args.fold_harmonics,
            tables=fold.fold_tables(r2, va["superclass"], p_model, va["period"], order),
            seconds=time.time() - t1,
        )
        extra = {}
        if not args.no_refine:
            # the model's period is a rough one: search near it for the period
            # that folds best, then fold on that
            ref = fold.refine_all(
                va["records"], p_model, args.refine_rel, args.refine_oversample,
                args.fold_harmonics, dev,
            )  # fmt: skip
            # The best of thousands of trial periods folds better than one
            # period picked blind, even where there is nothing to find. So
            # the floor must be searched too: the same fine search around a
            # period that is wrong on purpose.
            floor = fold.refine_all(
                va["records"], va["period"] * fold.WRONG_FACTOR, args.refine_rel,
                args.refine_oversample, args.fold_harmonics, dev,
            )  # fmt: skip
            r2_ref = dict(model=ref["r2"], catalogue=r2["catalogue"], wrong=floor["r2"])
            fold_res["refined"] = dict(
                rel=args.refine_rel,
                oversample=args.refine_oversample,
                trials_median=float(np.median(ref["trials"])),
                seconds=ref["seconds"],
                tables=fold.fold_tables(r2_ref, va["superclass"], ref["period"], va["period"], order),
                precision_rough=fold.precision_table(p_model, va["period"]),
                precision_refined=fold.precision_table(ref["period"], va["period"]),
            )
            extra = dict(p_refined=ref["period"], r2_refined=ref["r2"], r2_searched_floor=floor["r2"])
            fold.plot_fold(out / "fold_r2_refined.png", r2_ref, va["superclass"], order, _colour, f"phase-fold test after the fine search, {best}")
            a = fold_res["refined"]["tables"]["superclass"][0]
            pr = {r["bin"]: r["share"] for r in fold_res["refined"]["precision_refined"]}
            print(
                f"after a fine search within {args.refine_rel:.0%} of the model's period: median R2 "
                f"{a['r2_model']:.3f} (catalogue {a['r2_catalogue']:.3f}); as good as the catalogue's "
                f"fold for {a['as_good']:.1%} of the stars; period within 0.1 % for "
                f"{pr['within 0.1 %']:.1%}, within 0.01 % for {pr['within 0.01 %']:.1%} | {ref['seconds']:.0f}s"
            )
        np.savez_compressed(out / "fold_r2.npz", index=va["index"], p_model=p_model, p_catalogue=va["period"], **r2, **extra)
        fold.plot_fold(out / "fold_r2.png", r2, va["superclass"], order, _colour, f"phase-fold test, {best}")
        a = fold_res["tables"]["superclass"][0]
        print(
            f"phase-fold test ({args.fold_harmonics} harmonics, {a['n']} stars): median R2 on the "
            f"catalogue period {a['r2_catalogue']:.3f}, on the model's period {a['r2_model']:.3f}, "
            f"on a wrong period {a['r2_wrong']:.3f}; the model's fold is as good as the "
            f"catalogue's for {a['as_good']:.1%} of the stars | {time.time() - t1:.0f}s"
        )

    res = dict(
        latents=str(args.latents),
        ckpt=str(ckpt),
        pred=args.pred,
        window=window,
        min_tokens=min_tokens,
        dim=int(meta["dim"]),
        n_train=len(tr["records"]),
        n_val=len(va["records"]),
        n_dropped=tr["n_dropped"] + va["n_dropped"],
        n_by_superclass={g: counts[g] for g in order},
        class_means_train=class_means,
        models=models,
        best=best,
        best_on_ls_objects=best_on_ls,
        ls=ls_scores,
        n_ls=n_ls,
        ls_pmin=p_min,
        ls_n_freq_median=ls["n_freq_median"],
        ls_seconds=ls["seconds"],
        failures_model=fail_model,
        failures_ls_window=fail_ls,
        worst_bins_model=worst_model,
        worst_bins_ls_window=worst_ls,
        fold=fold_res,
        args=dict(vars(args)),
        seconds=time.time() - t0,
    )
    dump_json(res, out / "results.json")
    write_tables(out / "tables.md", res, order)
    write_worst(out / "worst.csv", va, preds[best], args.worst)
    np.savez_compressed(  # per-star periods of the best read-out, for later questions
        out / "predictions.npz", index=va["index"], p_model=10.0 ** preds[best], p_catalogue=va["period"],
        superclass=va["superclass"], n_windows=va["n_valid"], points=va["points"], best=best,
        **_top_candidates(tops, models, best),  # [n, k] candidate periods, best first
    )  # fmt: skip

    # --- figures
    plot_pred_vs_true(out / "pred_vs_true_model.png", va["logp"], preds[best], va["superclass"], order, f"best model: {best}")
    plot_pred_vs_true(out / "pred_vs_true_ls_window.png", va["logp"][sel], ls["pred"]["ls_window"], va["superclass"][sel], order, f"Lomb-Scargle, one {window:g} d window")
    r_best = preds[best] - va["logp"]
    plot_resid_vs(out / "resid_vs_cycles.png", window / va["period"], r_best, va["superclass"], order, "cycles per window (window / P)", f"{best}: residual vs cycles per window")
    plot_resid_vs(out / "resid_vs_points.png", va["points"], r_best, va["superclass"], order, "mean points per valid window", f"{best}: residual vs points per window")
    plot_ratio_hist(
        out / "ratio_hist.png",
        {best: 10.0**r_best, "ls_window": 10.0 ** (ls["pred"]["ls_window"] - va["logp"][sel]), "ls_full": 10.0 ** (ls["pred"]["ls_full"] - va["logp"][sel])},
    )

    # --- summary
    b, w, f = models[best], ls_scores["ls_window"], ls_scores["ls_full"]
    print(
        f"best feature set {best}: rec1 {b['rec1']:.3f} rec10 {b['rec10']:.3f} alias {b['alias']:.3f} "
        f"med|r| {b['med_abs']:.3f} dex (on the LS objects rec10 {best_on_ls['rec10']:.3f}) | "
        f"LS window rec1 {w['rec1']:.3f} rec10 {w['rec10']:.3f} alias {w['alias']:.3f} | "
        f"LS full rec1 {f['rec1']:.3f} rec10 {f['rec10']:.3f} alias {f['alias']:.3f} | "
        f"LS top5 oracle rec10 {ls_scores['ls_top5']['rec10']:.3f}"
    )
    hand = models.get("hand/ridge")
    if hand:
        print(f"hand-feature baseline (ridge): rec10 {hand['rec10']:.3f} med|r| {hand['med_abs']:.3f} dex")
    for wb in worst_model:
        print(f"model fails most: {wb['table']} = {wb['bin']} (n {wb['n']}, rec10 {wb['rec10']:.3f}, med|r| {wb['med_abs']:.3f} dex)")
    print("note: eclipsing binaries have two dips per orbit, so Lomb-Scargle finds P / 2 and lands in the alias rate")
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return res


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
