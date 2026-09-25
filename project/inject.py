"""Injected-anomaly test (M2): does the surprise spike where a curve changes?

Every object is scored twice with :func:`~project.diagnostics.grid_surprise`,
clean and with one anomaly injected at a random time ``t*``. The anomalies
are built on a phase-folded template of the curve at its catalogue period,
which is evaluation-side knowledge only (the model never sees the period):
``y = T_b(phi(t)) + r(t)`` with ``T_b`` the binned template of band ``b`` and
``r`` the residual of the fold, so the injected curve equals the original
before ``t*`` and keeps its cadence and noise after it.

Kinds (applied for ``t >= t*`` unless said otherwise):

- ``phase``: template shifted by U(0.25, 0.5) cycles.
- ``amp``: template amplitude scaled by U(1.5, 2.0), or its inverse.
- ``period``: template re-folded at ``P (1 + eps)``, ``|eps|`` in U(0.01, 0.05),
  phase continuous at ``t*``.
- ``bump``: Gaussian bump of height U(1, 3) standardised units and width
  U(5, 30) d centred at ``t*`` in every band (a transient).
- ``color``: constant offset U(0.5, 1.0) on the bluest band only.

Metrics per kind, pooled and per superclass, with ``k*`` the grid window
holding ``t*`` and ``Delta = surprise(injected) - surprise(clean)``:
localization hit rate (argmax of ``Delta`` within ``[k* - 1, k* + 2]``),
object AUROC (max surprise of the injected curve against the clean one),
window AUROC (windows ``[k*, k* + 2]`` against the other windows of the
injected curve, on the injected surprise and on ``Delta``) and the mean
``Delta`` profile aligned at ``k*``.

    python -m project.inject --ckpt project/runs/wm_w500/wm.pt --n-objects 300 \
        --out project/results/inject_w500
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from romae_lc import Record

from project.common import (
    add_device_arg,
    data_args_from,
    dump_json,
    get_device,
    grid_starts,
    load_data,
    load_wm,
    subset,
    superclass,
)
from project.diagnostics import grid_surprise

KINDS = ("phase", "amp", "period", "bump", "color")


# ----------------------------------------------------------------- templates


def fold_template(t, y, period, n_bins=32):
    """Periodic binned-median template of one band: ``(edges [n_bins + 1],
    values [n_bins])`` over phase, empty bins filled by periodic linear
    interpolation."""
    phase = np.mod(t / period, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.minimum((phase * n_bins).astype(int), n_bins - 1)
    vals = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            vals[b] = np.median(y[m])
    ok = np.isfinite(vals)
    if not ok.any():
        vals[:] = float(np.median(y))
    elif not ok.all():
        centers = (edges[:-1] + edges[1:]) / 2
        x = np.concatenate([centers[ok] - 1, centers[ok], centers[ok] + 1])
        v = np.tile(vals[ok], 3)
        vals[~ok] = np.interp(centers[~ok], x, v)
    return edges, vals


def eval_template(edges, vals, phase):
    """Periodic linear interpolation of a template at phases in [0, 1)."""
    n = len(vals)
    centers = (edges[:-1] + edges[1:]) / 2
    x = np.concatenate([centers - 1, centers, centers + 1])
    v = np.tile(vals, 3)
    return np.interp(np.mod(phase, 1.0), x, v)


def band_templates(record: Record, period: float, min_points=8, n_bins=32):
    """``{band: (edges, vals, mean)}`` for bands with enough points and the
    residual ``r = y - T_b(phi)`` of every point (0 for bands without one)."""
    templates, resid = {}, np.zeros_like(record.y)
    for b in np.unique(record.band):
        m = record.band == b
        if m.sum() < min_points:
            continue
        nb = int(min(n_bins, max(4, m.sum() // 8)))
        edges, vals = fold_template(record.t[m], record.y[m], period, nb)
        templates[int(b)] = (edges, vals, float(vals.mean()))
        resid[m] = record.y[m] - eval_template(edges, vals, record.t[m] / period)
    return templates, resid


# ----------------------------------------------------------------- injections


def inject(
    record: Record, kind: str, t_star: float, rng: np.random.Generator, period=None
):
    """A copy of ``record`` with one anomaly of ``kind`` injected at ``t_star``;
    returns ``(record, params)``."""
    period = record.period if period is None else period
    t, y = record.t.astype(np.float64), record.y.astype(np.float64).copy()
    after = t >= t_star
    params: dict = dict(kind=kind, t_star=float(t_star), period=float(period))
    if kind in ("phase", "amp", "period"):
        templates, resid = band_templates(record, period)
        if kind == "phase":
            params["shift"] = float(rng.uniform(0.25, 0.5))
        elif kind == "amp":
            a = float(rng.uniform(1.5, 2.0))
            params["factor"] = a if rng.uniform() < 0.5 else 1.0 / a
        else:
            eps = float(rng.uniform(0.01, 0.05)) * (1 if rng.uniform() < 0.5 else -1)
            params["eps"] = eps
        for b, (edges, vals, mean) in templates.items():
            m = (record.band == b) & after
            if not m.any():
                continue
            phase = t[m] / period
            if kind == "phase":
                base = eval_template(edges, vals, phase + params["shift"])
            elif kind == "amp":
                base = mean + params["factor"] * (
                    eval_template(edges, vals, phase) - mean
                )
            else:
                p_new = period * (1 + params["eps"])
                phase = t_star / period + (t[m] - t_star) / p_new
                base = eval_template(edges, vals, phase)
            y[m] = base + resid[m]
    elif kind == "bump":
        amp, width = float(rng.uniform(1.0, 3.0)), float(rng.uniform(5.0, 30.0))
        params.update(amp=amp, width=width)
        y += amp * np.exp(-0.5 * ((t - t_star) / width) ** 2)
    elif kind == "color":
        offset, band = float(rng.uniform(0.5, 1.0)), int(record.band.min())
        params.update(offset=offset, band=band)
        y[(record.band == band) & after] += offset
    else:
        raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")
    out = replace(record, y=y.astype(np.float32), meta=dict(record.meta, inject=params))
    return out, params


# -------------------------------------------------------------------- metrics


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney), NaN when a side is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return _auroc(pos, neg)


def _auroc(pos, neg):
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, float)
    sorted_v = allv[order]
    i = 0
    while i < allv.size:  # average ranks for ties
        j = i
        while j + 1 < allv.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def score_object(model, record, cfg, spec, device, kind, rng, seed=0, index=0):
    """Clean and injected surprise of one object along the fill grid, or
    ``None`` when the record has fewer than three windows."""
    starts = grid_starts(record, cfg)
    k_n = len(starts)
    if k_n < 3:
        return None
    lo, hi = starts[1], starts[k_n - 1] + cfg.window
    t_star = float(rng.uniform(lo, hi))
    k_star = int(np.searchsorted(starts, t_star, side="right") - 1)
    injected, params = inject(record, kind, t_star, rng)
    s0 = grid_surprise(model, record, cfg, spec, device, seed=seed, index=index)[
        "scores"
    ]
    s1 = grid_surprise(model, injected, cfg, spec, device, seed=seed, index=index)[
        "scores"
    ]
    return dict(
        s0=s0, s1=s1, k_star=k_star, t_star=t_star, params=params, n_windows=k_n
    )


def summarize(results: list[dict], offsets=range(-3, 4)) -> dict:
    """Pooled metrics of a list of :func:`score_object` results."""
    hits, pos_obj, neg_obj, win_pos, win_neg, d_pos, d_neg = [], [], [], [], [], [], []
    profile = {o: [] for o in offsets}
    for r in results:
        s0, s1, ks = r["s0"], r["s1"], r["k_star"]
        k = np.arange(1, len(s1) + 1)  # target window of each score
        ok = np.isfinite(s0) & np.isfinite(s1)
        if not ok.any():
            continue
        d = s1 - s0
        best = k[ok][np.nanargmax(d[ok])]
        hits.append(ks - 1 <= best <= ks + 2)
        pos_obj.append(np.nanmax(s1[ok]))
        neg_obj.append(np.nanmax(s0[ok]))
        anom = (k >= ks) & (k <= ks + 2) & ok
        rest = ok & ~anom
        win_pos += s1[anom].tolist()
        win_neg += s1[rest].tolist()
        d_pos += d[anom].tolist()
        d_neg += d[rest].tolist()
        for o in offsets:
            j = ks + o - 1
            if 0 <= j < len(d) and ok[j]:
                profile[o].append(d[j])
    return dict(
        n=len(hits),
        hit_rate=float(np.mean(hits)) if hits else float("nan"),
        auroc_object=(
            _auroc(np.array(pos_obj), np.array(neg_obj)) if hits else float("nan")
        ),
        auroc_window=_auroc(np.array(win_pos), np.array(win_neg)),
        auroc_delta=_auroc(np.array(d_pos), np.array(d_neg)),
        delta_profile={
            str(o): (float(np.mean(v)) if v else None) for o, v in profile.items()
        },
        mean_clean=float(np.mean(neg_obj)) if neg_obj else float("nan"),
        mean_injected=float(np.mean(pos_obj)) if pos_obj else float("nan"),
    )


# ------------------------------------------------------------------------ main


def run(model, meta, records, kinds, n_objects, device, seed=0, out=None, log=print):
    cfg, spec = meta.cfg, meta.spec
    recs = subset(records, n_objects, seed)
    summary, raw = {}, {}
    for kind in kinds:
        t0 = time.time()
        rng = np.random.default_rng([seed, KINDS.index(kind)])
        results, groups = [], []
        for i, r in enumerate(recs):
            res = score_object(
                model, r, cfg, spec, device, kind, rng, seed=seed, index=i
            )
            if res is None:
                continue
            res["superclass"], res["class"] = superclass(r), r.meta.get("class_str")
            results.append(res)
            groups.append(res["superclass"])
        pooled = summarize(results)
        per = {
            g: summarize([x for x in results if x["superclass"] == g])
            for g in sorted(set(groups))
        }
        summary[kind] = dict(pooled, per_superclass=per, seconds=time.time() - t0)
        raw[kind] = results
        log(
            f"{kind:7s} n={pooled['n']:4d}  hit {pooled['hit_rate']:.3f}  AUROC object "
            f"{pooled['auroc_object']:.3f}  window {pooled['auroc_window']:.3f}  delta "
            f"{pooled['auroc_delta']:.3f}  ({time.time() - t0:.0f}s)"
        )
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        dump_json(
            dict(
                ckpt_step=meta.step,
                n_objects=len(recs),
                kinds=list(kinds),
                results=summary,
            ),
            out / "summary.json",
        )
        np.savez_compressed(
            out / "objects.npz",
            **{
                f"{kind}/{key}": np.array([r[key] for r in res], dtype=object)
                for kind, res in raw.items()
                for key in ("s0", "s1", "k_star", "t_star", "superclass", "class")
            },
        )
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--ckpt", required=True, help="stage-1 checkpoint (wm.pt or last.pt)"
    )
    p.add_argument("--split", default="validation")
    p.add_argument("--n-objects", type=int, default=300)
    p.add_argument("--kinds", nargs="*", default=list(KINDS), choices=KINDS)
    p.add_argument("--data", default=None, help="override the checkpoint's data root")
    p.add_argument(
        "--classes", nargs="*", default=None, help="override the class filter"
    )
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="injection seed (data seed comes from the checkpoint)",
    )
    p.add_argument(
        "--out", default=None, help="results directory (default next to the checkpoint)"
    )
    add_device_arg(p)
    args = p.parse_args(argv)
    dev = get_device(args)
    model, meta = load_wm(args.ckpt, dev)
    data = load_data(data_args_from(meta.args, args), splits=(args.split,))
    if tuple(data.classes) != tuple(meta.classes):
        print("warning: class vocabulary differs from the checkpoint's")
    out = args.out or (Path(args.ckpt).parent / f"inject_{args.split}_{meta.step}")
    print(
        f"{len(data[args.split])} {args.split} records; checkpoint step {meta.step}; out {out}"
    )
    run(model, meta, data[args.split], args.kinds, args.n_objects, dev, args.seed, out)


if __name__ == "__main__":
    main()
