"""The period report on every validation star.

For one period read-out (``predictions.npz`` of ``period_probe``) this
scores the whole validation split, whole light curves:

* ``model``: the read-out's period alone;
* ``model_refined``: a fine search within ``--refine-rel`` of it, the fold
  (adjusted R2 of a 3-harmonic wave) picks;
* ``model_cands``: the same search around every candidate (the read-out's
  top-k bins, the double and the half of its period), the fold picks;
* ``ls_full@B``: our GPU Lomb-Scargle at B trial frequencies;
* ``astropy_mb@B``: the collaborator's multiband astropy search at B trials;
* ``--compare name=predictions.npz``: other read-outs, scored alone (used
  for the ReLU against SiLU question).

Hits within 20, 10, 1, 0.1 and 0.01 % (strict and alias-tolerant), overall
and per superclass; hit rate against the true period (line plot); predicted
against true period with the binned median ratio (the "bend"); phase-fold
demos with the fitted wave; and ``better_than_catalogue.csv``: the stars
whose best model period folds clearly better than the catalogue's.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project import fold
from project.common import add_device_arg, data_args_from, dump_json, get_device, load_data, load_encoder, superclass, fine_class
from project.eval.ls_benchmark import astropy_search, ls_full, run_lomb_scargle, star_inputs
from project.plot_period_examples import BAND_COLOUR, CLASS_COLOUR, CLASS_NAME, GRID, INK, INK2, OTHER, SURFACE, style

TOLS = (0.2, 0.1, 0.01, 0.001, 0.0001)
TOL_NAME = {0.2: "20 %", 0.1: "10 %", 0.01: "1 %", 0.001: "0.1 %", 0.0001: "0.01 %"}
HEADLINE = ("ECL", "RR", "ROT", "CEP", "DSCT", "LPV", "ELL", "PCEB")
P_EDGES = np.array([0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 1000.0])
METHOD_COLOUR = {"model": "#2a78d6", "model_refined": "#1baf7a", "model_cands": "#0b0b0b", "ls": "#eb6834", "astropy": "#b07cd6"}


# -------------------------------------------------------------------- scoring


def rel_error(p_hat, p_true, alias: bool = False) -> np.ndarray:
    r = np.abs(p_hat / p_true - 1.0)
    if alias:
        r = np.minimum.reduce([r, np.abs(p_hat / p_true / 2.0 - 1.0), np.abs(p_hat / p_true * 2.0 - 1.0)])
    return r


def hit_table(p_hat, p_true) -> dict:
    out = {}
    for alias in (False, True):
        r = rel_error(p_hat, p_true, alias)
        for t in TOLS:
            out[f"{'alias_' if alias else 'within_'}{t:g}"] = float(np.mean(r < t))  # NaN predictions count as misses
    return out


def alias_kind(ratio: float) -> str:
    if abs(ratio - 1.0) < 0.01:
        return "same period, sharper"
    if abs(ratio - 2.0) < 0.02:
        return "double"
    if abs(ratio - 0.5) < 0.005:
        return "half"
    for k in (3.0, 1.5, 2.0 / 3.0, 1.0 / 3.0):
        if abs(ratio / k - 1.0) < 0.01:
            return f"{k:.3g} x"
    return "different"


# -------------------------------------------------------------------- figures


def fold_panel(ax, r, period: float, title: str, r2: float, template: bool = True, harmonics: int = 3):
    """One phase-folded panel: the points of every band, twice over, and the
    fitted wave of ``harmonics`` harmonics (the fold's own model) on top."""
    style(ax)
    t0 = float(r.t.min())
    lo, hi = np.percentile(r.y, [0.5, 99.5])
    for b in np.unique(r.band):
        m = r.band == b
        ph = np.mod((r.t[m] - t0) / period, 1.0)
        col = BAND_COLOUR[int(b) % 3]
        ax.scatter(np.concatenate([ph, ph + 1]), np.concatenate([r.y[m], r.y[m]]), s=3, color=col, alpha=0.45, linewidths=0)
        if template and m.sum() >= 2 * harmonics + 4:
            w = 1.0 / (np.maximum(r.err[m], 0.0) ** 2 + 1e-6)
            sw = np.sqrt(w)[:, None]
            x = fold.design(r.t[m].astype(np.float64), period, harmonics)
            beta, *_ = np.linalg.lstsq(x * sw, r.y[m] * sw[:, 0], rcond=None)
            grid = np.linspace(0, 2, 400)
            xg = fold.design(t0 + grid * period, period, harmonics)
            ax.plot(grid, xg @ beta, color=col, lw=1.1, alpha=0.95)
    pad = 0.15 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, 2)
    ax.set_title(f"{title}\nP = {period:.6g} d, fold R2 {r2:.2f}", fontsize=7.5, color=INK, loc="left")


def fold_figure(path, rows, heading: str):
    """``rows``: ``(record, label, [(period, title, r2), ...])`` with three
    periods per star: catalogue, model alone, best model period."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(10.5, 2.4 * len(rows)), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax_row, (r, label, cols) in zip(axes, rows):
        for ax, (p, title, r2) in zip(ax_row, cols):
            fold_panel(ax, r, p, title, r2)
        ax_row[0].set_ylabel(f"{label}\n\nbrightness", fontsize=7.5, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("phase (shown twice)", fontsize=8, color=INK2)
    fig.suptitle(heading, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def binned_rate(p_true, hit, edges):
    """Hit rate and count per log-period bin."""
    idx = np.digitize(p_true, edges) - 1
    rate, n = np.full(len(edges) - 1, np.nan), np.zeros(len(edges) - 1, dtype=int)
    for b in range(len(edges) - 1):
        m = idx == b
        n[b] = m.sum()
        if n[b] >= 15:
            rate[b] = float(np.mean(hit[m]))
    return rate, n


def plot_hit_vs_period(path, p_true, series: dict, sup, title: str):
    """``series``: ``{label: (hit bool array, colour, linestyle)}``; the
    count of stars per bin as faint bars behind."""
    edges = P_EDGES
    mid = np.sqrt(edges[:-1] * edges[1:])
    fig, ax = plt.subplots(figsize=(9.2, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax2 = ax.twinx()
    _, n = binned_rate(p_true, np.ones_like(p_true, dtype=bool), edges)
    ax2.bar(mid, n, width=np.diff(edges) * 0.9, color=GRID, alpha=0.6, zorder=0)
    ax2.set_ylabel("stars per bin", fontsize=8, color=INK2)
    ax2.tick_params(labelsize=7, colors=INK2)
    ax2.set_ylim(0, n.max() * 3.2)
    ax2.spines[:].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    for label, (hit, colour, ls) in series.items():
        rate, _ = binned_rate(p_true, hit, edges)
        ax.plot(mid, rate, color=colour, lw=1.4, ls=ls, marker="o", ms=3, label=label)
    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("catalogue period (days), log-spaced bins; bins with fewer than 15 stars are left out", fontsize=8, color=INK2)
    ax.set_ylabel("hit rate", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2, labelcolor=INK2)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_scatter_bend(path, p_true, preds: dict, sup, title: str):
    """Top: predicted against true period per read-out (log axes). Bottom:
    the ratio predicted / true against the true period, with the binned
    median and the 16-84 % band, which shows a bend at long periods."""
    names = list(preds)
    fig, axes = plt.subplots(2, len(names), figsize=(4.4 * len(names), 8.2), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    groups = [("other", ~np.isin(sup, list(CLASS_COLOUR)), OTHER)] + [(g, sup == g, c) for g, c in CLASS_COLOUR.items()]
    edges = np.exp(np.linspace(np.log(0.03), np.log(1000.0), 40))
    for j, name in enumerate(names):
        p = preds[name]
        ok = np.isfinite(p) & (p > 0)
        ax = axes[0][j]
        style(ax)
        for g, m, c in groups:
            mm = m & ok
            ax.scatter(p_true[mm], p[mm], s=2.5, color=c, alpha=0.4, linewidths=0, label=f"{CLASS_NAME[g]} ({int(m.sum())})")
        lim = np.array([0.03, 1000.0])
        for k in (1.0, 2.0, 0.5):
            ax.plot(lim, lim * k, color=INK2, lw=0.7, ls="-" if k == 1 else (0, (4, 3)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")
        hit = np.mean(np.abs(p / p_true - 1) < 0.1)
        ax.set_title(f"{name}: predicted against catalogue period\n{hit:.1%} within 10 %, {np.mean(np.abs(p / p_true - 1) < 0.01):.1%} within 1 %", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("catalogue period (days)", fontsize=8, color=INK2)
        ax.set_ylabel("predicted period (days)", fontsize=8, color=INK2)
        if j == 0:
            ax.legend(fontsize=6.5, frameon=False, loc="upper left", markerscale=4, labelcolor=INK2)
        ax = axes[1][j]
        style(ax)
        ratio = p / p_true
        for g, m, c in groups:
            mm = m & ok
            ax.scatter(p_true[mm], ratio[mm], s=2.5, color=c, alpha=0.3, linewidths=0)
        idx = np.digitize(p_true, edges) - 1
        mid = np.sqrt(edges[:-1] * edges[1:])
        med, lo, hi = (np.full(len(mid), np.nan) for _ in range(3))
        for b in range(len(mid)):
            m = (idx == b) & ok
            if m.sum() >= 10:
                med[b], lo[b], hi[b] = np.percentile(ratio[m], [50, 16, 84])
        ax.fill_between(mid, lo, hi, color=INK, alpha=0.12, linewidth=0, label="16-84 % of the stars")
        ax.plot(mid, med, color=INK, lw=1.4, label="median ratio per bin")
        ax.axhline(1.0, color=INK2, lw=0.7)
        for k in (2.0, 0.5):
            ax.axhline(k, color=INK2, lw=0.7, ls=(0, (4, 3)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.03, 1000.0)
        ax.set_ylim(0.2, 5.0)
        ax.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0])
        ax.set_yticklabels(["1/4", "1/2", "1", "2", "4"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("catalogue period (days)", fontsize=8, color=INK2)
        ax.set_ylabel("predicted / catalogue", fontsize=8, color=INK2)
        ax.set_title(f"{name}: the ratio against the catalogue period", fontsize=8.5, color=INK, loc="left")
        if j == 0:
            ax.legend(fontsize=6.5, frameon=False, loc="upper left", labelcolor=INK2)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# -------------------------------------------------------------------- main


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="the encoder checkpoint (for the data arguments)")
    p.add_argument("--predictions", required=True, help="predictions.npz of period_probe (index, p_model, p_top)")
    p.add_argument("--compare", nargs="*", default=[], help="name=predictions.npz of other read-outs, scored alone")
    p.add_argument("--out", required=True)
    p.add_argument("--n-objects", type=int, default=0, help="0 = every validation star with a period")
    p.add_argument("--ls-budgets", type=int, nargs="*", default=[200000, 500000], help="our GPU Lomb-Scargle grids; none to skip")
    p.add_argument("--astropy-budget", type=int, default=100000, help="the collaborator's multiband search; 0 to skip")
    p.add_argument("--refine-rel", type=float, default=0.1, help="fine-search range around the read-out's period")
    p.add_argument("--cand-rel", type=float, default=0.03, help="fine-search range around every candidate")
    p.add_argument("--gain", type=float, default=0.1, help="fold R2 gain over the catalogue period to call a period better")
    p.add_argument("--min-r2", type=float, default=0.5, help="fold R2 the better period must reach")
    p.add_argument("--p-min", type=float, default=None, help="default: the shortest training period")
    p.add_argument("--harmonics", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--n-sim", type=int, default=None)
    add_device_arg(p)
    return p.parse_args(argv)


def run(args):
    t_start = time.time()
    dev = get_device(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _, meta = load_encoder(args.ckpt, "cpu")
    over = argparse.Namespace(data=args.data, max_rows=args.max_rows, n_sim=args.n_sim)
    data = load_data(data_args_from(meta.args, over), splits=("train", "validation"))
    p_min = args.p_min or float(min(r.period for r in data["train"] if r.period and r.period > 0))
    pred = np.load(args.predictions, allow_pickle=True)
    rough = {int(i): float(p) for i, p in zip(pred["index"], pred["p_model"])}
    tops = {int(i): [float(v) for v in row] for i, row in zip(pred["index"], pred["p_top"])} if "p_top" in pred.files else {}
    compare = {}
    for item in args.compare:
        name, path = item.split("=", 1)
        c = np.load(path, allow_pickle=True)
        compare[name] = {int(i): float(p) for i, p in zip(c["index"], c["p_model"])}
    if args.astropy_budget and run_lomb_scargle is None:
        raise SystemExit("--astropy-budget needs lomb_scargle.py at the repository root and astropy")
    idx = [int(i) for i in pred["index"] if data["validation"][int(i)].period and data["validation"][int(i)].period > 0]
    if args.n_objects and len(idx) > args.n_objects:
        idx = sorted(np.random.default_rng(args.seed).choice(idx, args.n_objects, replace=False).tolist())
    n = len(idx)
    methods = ["model", "model_refined", "model_cands"] + [f"ls_full@{b}" for b in args.ls_budgets]
    if args.astropy_budget:
        methods.append(f"astropy_mb@{args.astropy_budget}")
    methods += list(compare)
    periods = {m: np.full(n, np.nan) for m in methods}
    trials = {m: np.zeros(n) for m in methods}
    seconds = {m: 0.0 for m in methods}
    r2 = {k: np.full(n, np.nan) for k in ("catalogue", "model", "model_refined", "model_cands")}
    p_true, sup, fine, ids = np.zeros(n), np.empty(n, dtype=object), np.empty(n, dtype=object), np.empty(n, dtype=object)
    n_points, span = np.zeros(n, dtype=int), np.zeros(n)
    print(f"{n} validation stars; p_min {p_min:.4g} d; methods {methods}; {dev}", flush=True)
    for j, i in enumerate(idx):
        r = data["validation"][i]
        t, y, err, band, w = star_inputs(r)
        p_true[j], sup[j], fine[j], ids[j] = float(r.period), superclass(r), fine_class(r), str(r.meta.get("id", i))
        n_points[j], span[j] = t.size, float(t.max() - t.min())
        pm = rough.get(i, np.nan)
        periods["model"][j] = pm
        r2["catalogue"][j] = fold.fold_r2(t, y, err, band, p_true[j], args.harmonics)
        r2["model"][j] = fold.fold_r2(t, y, err, band, pm, args.harmonics)
        t0 = time.time()
        periods["model_refined"][j], r2["model_refined"][j], trials["model_refined"][j] = fold.refine_period(t, y, err, band, pm, args.refine_rel, harmonics=args.harmonics, device=dev)
        seconds["model_refined"] += time.time() - t0
        t0 = time.time()
        seeds = [s for s in tops.get(i, [pm]) if np.isfinite(s) and s > 0]
        seeds += [2.0 * pm, 0.5 * pm]
        best_p, best_r2, nt = periods["model_refined"][j], r2["model_refined"][j], 0
        for s in seeds:
            if not (np.isfinite(s) and s > 0):
                continue
            p, q, k = fold.refine_period(t, y, err, band, float(s), args.cand_rel, harmonics=args.harmonics, device=dev)
            nt += int(k)
            if np.isfinite(q) and (not np.isfinite(best_r2) or q > best_r2):
                best_p, best_r2 = float(p), float(q)
        periods["model_cands"][j], r2["model_cands"][j], trials["model_cands"][j] = best_p, best_r2, trials["model_refined"][j] + nt
        seconds["model_cands"] += time.time() - t0
        for b in args.ls_budgets:
            m = f"ls_full@{b}"
            t0 = time.time()
            periods[m][j], trials[m][j] = ls_full(t, y, w, span[j], p_min, b, dev)
            seconds[m] += time.time() - t0
        if args.astropy_budget:
            m = f"astropy_mb@{args.astropy_budget}"
            t0 = time.time()
            periods[m][j], trials[m][j] = astropy_search(r, args.astropy_budget, True, p_min)
            seconds[m] += time.time() - t0
        for name, table in compare.items():
            periods[name][j] = table.get(i, np.nan)
        if (j + 1) % 250 == 0:
            print(f"  {j + 1} / {n} stars, {time.time() - t_start:.0f}s", flush=True)
    sup = sup.astype(str)
    fine = fine.astype(str)
    seconds["model_cands"] += seconds["model_refined"]  # the candidate search includes the plain fine search

    # --- tables
    res = dict(n_stars=n, p_min=p_min, predictions=str(args.predictions), compare=args.compare, refine_rel=args.refine_rel, cand_rel=args.cand_rel, methods={})
    groups = [g for g in HEADLINE if (sup == g).sum() >= 5]
    for m in methods:
        res["methods"][m] = dict(trials_median=float(np.median(trials[m])), seconds_per_star=seconds[m] / n, **hit_table(periods[m], p_true),
                                 by_superclass={g: hit_table(periods[m][sup == g], p_true[sup == g]) for g in groups})  # fmt: skip
    res["fold"] = {k: dict(median=float(np.nanmedian(v)), mean=float(np.nanmean(v))) for k, v in r2.items()}
    res["fold"]["as_good_refined"] = float(np.mean(r2["model_refined"] >= r2["catalogue"] - 0.05))
    res["fold"]["as_good_cands"] = float(np.mean(r2["model_cands"] >= r2["catalogue"] - 0.05))

    # --- the better-than-catalogue list
    ratio = periods["model_cands"] / p_true
    gain = r2["model_cands"] - r2["catalogue"]
    better = np.flatnonzero(np.isfinite(gain) & (gain > args.gain) & (r2["model_cands"] > args.min_r2))
    better = better[np.argsort(-gain[better])]
    worse = int(np.sum(np.isfinite(gain) & (gain < -args.gain)))
    kinds = np.array([alias_kind(float(x)) if np.isfinite(x) else "?" for x in ratio])
    with open(out / "better_than_catalogue.csv", "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["ztf_id", "val_index", "class", "superclass", "p_catalogue_d", "p_model_d", "p_best_d", "best_over_catalogue", "kind",
                      "r2_catalogue", "r2_best", "gain", "n_points", "baseline_d"])  # fmt: skip
        for j in better:
            wtr.writerow([ids[j], idx[j], fine[j], sup[j], f"{p_true[j]:.6f}", f"{periods['model'][j]:.6f}", f"{periods['model_cands'][j]:.6f}",
                          f"{ratio[j]:.5f}", kinds[j], f"{r2['catalogue'][j]:.3f}", f"{r2['model_cands'][j]:.3f}", f"{gain[j]:.3f}", n_points[j], f"{span[j]:.0f}"])  # fmt: skip
    res["better"] = dict(n=int(len(better)), worse=worse, gain=args.gain, min_r2=args.min_r2,
                         by_kind={k: int(np.sum(kinds[better] == k)) for k in np.unique(kinds[better])},
                         by_superclass={g: int(np.sum(sup[better] == g)) for g in groups})  # fmt: skip
    dump_json(res, out / "results.json")
    np.savez_compressed(out / "per_star.npz", index=np.array(idx), ztf_id=ids.astype(str), p_true=p_true, superclass=sup, fine=fine, n_points=n_points, span=span,
                        **{f"p_{m}": periods[m] for m in methods}, **{f"trials_{m}": trials[m] for m in methods}, **{f"r2_{k}": v for k, v in r2.items()})  # fmt: skip

    md = [f"# Period report: {n} validation stars, whole light curves\n",
          "Hits = share of stars whose period is within the tolerance of the catalogue period; a missing prediction counts as a miss. "
          "Alias-tolerant also accepts the double and the half. Trials = frequencies evaluated per star (median); s/star = wall-clock per star.\n",
          "| method | trials | s/star | 20 % | 10 % | 1 % | 0.1 % | 0.01 % | alias 1 % | alias 0.01 % |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        v = res["methods"][m]
        md.append(f"| {m} | {v['trials_median']:.0f} | {v['seconds_per_star']:.3f} | " + " | ".join(f"{v[f'within_{t:g}']:.3f}" for t in TOLS)
                  + f" | {v['alias_0.01']:.3f} | {v['alias_0.0001']:.3f} |")  # fmt: skip
    for key, name in (("within_0.1", "within 10 %"), ("within_0.01", "within 1 %"), ("within_0.0001", "within 0.01 %"), ("alias_0.0001", "alias-tolerant 0.01 %")):
        md += [f"\n## {name} by superclass\n", "| method | " + " | ".join(f"{g} ({(sup == g).sum()})" for g in groups) + " |", "|---|" + "---|" * len(groups)]
        for m in methods:
            md.append(f"| {m} | " + " | ".join(f"{res['methods'][m]['by_superclass'][g][key]:.3f}" for g in groups) + " |")
    md += ["\n## phase-fold test (adjusted R2 of a 3-harmonic wave, whole curve)\n", "| period | median R2 | mean R2 |", "|---|---|---|"]
    md += [f"| {k} | {v['median']:.3f} | {v['mean']:.3f} |" for k, v in res["fold"].items() if isinstance(v, dict)]
    md.append(f"\nmodel_refined folds at least as well as the catalogue (within 0.05) for {res['fold']['as_good_refined']:.1%} of the stars; model_cands for {res['fold']['as_good_cands']:.1%}.")
    md.append(f"\n## better than the catalogue\n\n{len(better)} stars where the best model period folds better than the catalogue period by more than {args.gain:g} in R2 and reaches R2 > {args.min_r2:g} "
              f"({worse} stars where it is worse by the same margin). By kind: " + ", ".join(f"{k}: {v}" for k, v in res["better"]["by_kind"].items()) + ". See better_than_catalogue.csv.")
    (out / "tables.md").write_text("\n".join(md) + "\n")

    # --- figures
    ls_name = f"ls_full@{max(args.ls_budgets)}" if args.ls_budgets else None
    series = {"model alone, within 20 %": (rel_error(periods["model"], p_true) < 0.2, METHOD_COLOUR["model"], (0, (1, 1))),
              "model alone, within 10 %": (rel_error(periods["model"], p_true) < 0.1, METHOD_COLOUR["model"], (0, (4, 2))),
              "model alone, within 1 %": (rel_error(periods["model"], p_true) < 0.01, METHOD_COLOUR["model"], "-"),
              "model + fine search, within 0.01 %": (rel_error(periods["model_refined"], p_true) < 1e-4, METHOD_COLOUR["model_refined"], "-"),
              "model candidates + fine search, within 0.01 %": (rel_error(periods["model_cands"], p_true) < 1e-4, METHOD_COLOUR["model_cands"], "-"),
              "model candidates + fine search, alias-tolerant 0.01 %": (rel_error(periods["model_cands"], p_true, True) < 1e-4, METHOD_COLOUR["model_cands"], (0, (4, 2)))}  # fmt: skip
    if ls_name:
        series[f"Lomb-Scargle {ls_name.split('@')[1]} trials, within 0.01 %"] = (rel_error(periods[ls_name], p_true) < 1e-4, METHOD_COLOUR["ls"], "-")
        series[f"Lomb-Scargle {ls_name.split('@')[1]} trials, alias-tolerant 0.01 %"] = (rel_error(periods[ls_name], p_true, True) < 1e-4, METHOD_COLOUR["ls"], (0, (4, 2)))
    plot_hit_vs_period(out / "hit_vs_period.png", p_true, series, sup, "Hit rate against the catalogue period, every validation star")
    for g in ("ECL", "RR"):
        m = sup == g
        if m.sum() >= 100:
            plot_hit_vs_period(out / f"hit_vs_period_{g}.png", p_true[m], {k: (v[0][m], v[1], v[2]) for k, v in series.items()}, sup[m],
                               f"Hit rate against the catalogue period, {CLASS_NAME.get(g, g)} only")  # fmt: skip
    preds = {"model alone (the read-out)": periods["model"], **{f"{k} (alone)": periods[k] for k in compare}, "model candidates + fine search": periods["model_cands"]}
    plot_scatter_bend(out / "scatter_bend.png", p_true, preds, sup, "Predicted against catalogue period and the ratio, every validation star")

    # fold demos: typical sharp hits per fine class, and the top gainers
    err_c = rel_error(periods["model_cands"], p_true)
    rows = []
    for cls, note in (("EW/EB", "contact binary"), ("EA", "detached binary"), ("RRAB", "RR Lyrae ab"), ("RRC", "RR Lyrae c"), ("ROT", "rotation"),
                      ("CEP", "Cepheid"), ("DSCT", "delta Scuti"), ("LPV", "long-period variable")):  # fmt: skip
        m = (fine == cls) & (err_c < 1e-4) & (n_points >= 300)
        if m.sum() == 0:
            m = (fine == cls) & (err_c < 1e-2)
        if m.sum() == 0:
            continue
        cand = np.flatnonzero(m)
        j = int(cand[np.argsort(np.abs(r2["model_cands"][cand] - np.median(r2["model_cands"][cand])))[0]])
        rows.append(j)
    def three(j):
        pm, pb = periods["model"][j], periods["model_cands"][j]
        return [(p_true[j], "catalogue period", r2["catalogue"][j]),
                (pm, f"model alone ({pm / p_true[j] - 1:+.2%} off)", r2["model"][j]),
                (pb, f"model + search ({pb / p_true[j] - 1:+.4%} off)" if abs(pb / p_true[j] - 1) < 0.1 else f"model + search ({pb / p_true[j]:.3f} x catalogue)", r2["model_cands"][j])]  # fmt: skip
    if rows:
        fold_figure(out / "fold_demo.png", [(data["validation"][idx[j]], f"{fine[j]}\n{ids[j]}", three(j)) for j in rows],
                    "Phase-folded light curves with the fitted 3-harmonic wave: a typical hit of every class")  # fmt: skip
    top = [int(j) for j in better[:8]]
    if top:
        fold_figure(out / "fold_better.png", [(data["validation"][idx[j]], f"{fine[j]}\n{ids[j]}\n{kinds[j]}", three(j)) for j in top],
                    "Stars where the model's period folds better than the catalogue period (largest gains)")  # fmt: skip
    print("\n".join(md[3:3 + len(methods) + 2]))
    print(md[-1])
    print(f"wrote {out} in {time.time() - t_start:.0f}s")
    return res


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
