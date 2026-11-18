"""Period search efficiency: the model's prior against Lomb-Scargle.

The encoder's rough period (and the period head's fine one) shrink the
Lomb-Scargle search. This benchmark puts that on one axis, the number of
trial frequencies evaluated per star, which is the cost unit of every
Lomb-Scargle variant (points times trials), with wall-clock on the same
GPU next to it. Methods, on the WHOLE light curve of every validation star:

- ``ls_full@B``: one Lomb-Scargle grid of ``B`` trials from ``1 / (2 span)``
  to ``2 / p_min`` (the brute force), at several budgets ``B``;
- ``ls_two_stage@B``: the search astronomers use: a coarse grid of ``B``
  trials, its ``k`` best peaks each sharpened by a fine local search; the
  candidate whose fold scores best wins; trials counted in full;
- ``model_refined``: the model's rough period sharpened by the fine search
  within 10 % (about 7,500 trials);
- ``model_top5``: the model's five best bins, each sharpened, the fold
  picks (about 37,000 trials);
- ``head``: the period head's output as is (``--head`` predictions), a
  fixed 20,000-bin spectrum per window and no search;
- ``astropy_1b@B`` / ``astropy_mb@B``: the collaborator's reference search
  (``lomb_scargle.run_lomb_scargle``, astropy's fast Lomb-Scargle, one
  sinusoid, a UNIFORM frequency grid of ``B`` points from one cycle per
  baseline to ``1 / 0.05 d``) on the band with the most points and on all
  bands (the multiband model), at the same budgets as ``ls_full``.

Scores: the share of stars whose period is within 10 %, 1 %, 0.1 % and
0.01 % of the catalogue, the alias-tolerant share (within 0.01 % of the
period, its double or its half), per superclass. The crossing: the budget
at which ``ls_full`` and ``ls_two_stage`` match the model-seeded methods
within 0.01 %. Figures: hit rate against trials, and heat maps of the hit
rate over the data properties (points per window x windows, cadence x
baseline) for the model-seeded search and for Lomb-Scargle.

    python -m project.eval.ls_benchmark --ckpt project/runs/mae_w250/mae.pt \\
        --predictions project/results/period_mae20/predictions.npz --out project/results/ls_benchmark
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from project import fold
from project.common import add_device_arg, data_args_from, dump_json, get_device, load_data, load_encoder, superclass
from project.period_probe import freq_grid, gls_power, top_peaks

try:  # the collaborator's reference search (astropy); optional
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from project.eval.lomb_scargle import run_lomb_scargle  # the collaborator's reference search
    except ImportError:
        from lomb_scargle import run_lomb_scargle  # its old place, the repository root
except Exception:  # pragma: no cover
    run_lomb_scargle = None

TOLS = (0.1, 0.01, 0.001, 0.0001)
HEADLINE = ("ECL", "RR", "ROT", "CEP", "DSCT", "LPV")


# -------------------------------------------------------------------- scoring


def hit_table(p_hat, p_true):
    r = np.abs(p_hat / p_true - 1.0)
    out = {f"within_{t:g}": float(np.nanmean(r < t)) for t in TOLS}
    alias = np.minimum.reduce([np.abs(p_hat / p_true / m - 1.0) for m in (1.0, 2.0, 0.5)])
    out["alias_tolerant_0.0001"] = float(np.nanmean(alias < 1e-4))
    out["alias_tolerant_0.1"] = float(np.nanmean(alias < 0.1))
    return out


def star_inputs(r):
    t, y = r.t.astype(np.float64), r.y.astype(np.float64).copy()
    err, band = r.err.astype(np.float64), r.band
    for b in np.unique(band):
        m = band == b
        y[m] -= np.median(y[m])
    w = np.where(err > 0, 1.0 / np.maximum(err, 1e-12) ** 2, 0.0)
    return t, y, err, band, w


def ls_full(t, y, w, span, p_min, budget, device):
    """The brute-force grid at ``budget`` trials: ``(period, trials)``."""
    freqs = freq_grid(span, p_min, 1e9, int(budget))  # the cap sets the budget
    power = gls_power(t, y, w, freqs, device)
    return float(1.0 / freqs[int(np.argmax(power))]), int(freqs.size)


def ls_two_stage(t, y, err, band, w, span, p_min, budget, k, device, oversample=5.0):
    """Coarse grid of ``budget`` trials, its ``k`` best peaks each refined
    within the coarse grid's spacing, the fold picks: ``(period, trials)``."""
    freqs = freq_grid(span, p_min, 1e9, int(budget))
    power = gls_power(t, y, w, freqs, device)
    peaks = top_peaks(power, freqs, k)
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0 / span
    best, best_r2, trials = float("nan"), -np.inf, int(freqs.size)
    for f in peaks:
        rel = max(2.0 * df / f, 1e-4)
        p, r2, n = fold.refine_period(t, y, err, band, 1.0 / f, rel, oversample, device=device)
        trials += int(n)
        if np.isfinite(r2) and r2 > best_r2:
            best, best_r2 = float(p), float(r2)
    return best, trials


def astropy_search(r, budget, multiband: bool, p_min: float, p_max: float = 400.0):
    """The collaborator's search on the record (normalised magnitudes, which
    leave a per-band sinusoid fit unchanged): ``(period, trials)``."""
    if run_lomb_scargle is None:
        return float("nan"), 0
    bands = {}
    for b in np.unique(r.band):
        m = r.band == b
        bands[str(int(b))] = dict(mjd=r.t[m].astype(float), mag=r.y[m].astype(float), mag_unc=r.err[m].astype(float))
    if not multiband:
        best = max(bands, key=lambda k: len(bands[k]["mjd"]))
        bands = {best: bands[best]}
    try:
        out = run_lomb_scargle(bands, autopower=False, n_grid=int(budget), min_period=p_min, max_period=p_max, n_peaks=0)
    except Exception:
        return float("nan"), int(budget)
    if out is None:
        return float("nan"), int(budget)
    return float(out["best_period"]), int(budget)


def model_seeded(t, y, err, band, seeds, rel, device, oversample=5.0):
    """Each seed sharpened within ``rel``; the fold picks: ``(period, trials)``."""
    best, best_r2, trials = float("nan"), -np.inf, 0
    for s in seeds:
        if not (np.isfinite(s) and s > 0):
            continue
        p, r2, n = fold.refine_period(t, y, err, band, float(s), rel, oversample, device=device)
        trials += int(n)
        if np.isfinite(r2) and r2 > best_r2:
            best, best_r2 = float(p), float(r2)
    return best, trials


# -------------------------------------------------------------------- figures


def plot_crossing(path, curves, lines):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    for name, (xs, ys) in curves.items():
        ax.plot(xs, ys, marker="o", lw=1.5, label=name)
    for name, (x, y) in lines.items():
        ax.scatter([x], [y], s=70, marker="*", zorder=5, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("trial frequencies per star")
    ax.set_ylabel("share of stars within 0.01 % of the catalogue period")
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Period search: hit rate against the search budget", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def heat(path, hits_by_method, x, y, x_edges, y_edges, x_label, y_label, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(hits_by_method)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.2), dpi=150, squeeze=False)
    xi = np.clip(np.digitize(x, x_edges) - 1, 0, len(x_edges) - 2)
    yi = np.clip(np.digitize(y, y_edges) - 1, 0, len(y_edges) - 2)
    for ax, name in zip(axes[0], names):
        h = hits_by_method[name].astype(float)
        grid = np.full((len(y_edges) - 1, len(x_edges) - 1), np.nan)
        counts = np.zeros_like(grid)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                m = (xi == j) & (yi == i)
                counts[i, j] = m.sum()
                if m.sum() >= 8:
                    grid[i, j] = np.nanmean(h[m])
        im = ax.imshow(grid, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if counts[i, j] >= 8:
                    ax.text(j, i, f"{grid[i, j]:.2f}\n({int(counts[i, j])})", ha="center", va="center", fontsize=6.5,
                            color="white" if grid[i, j] < 0.6 else "black")  # fmt: skip
        ax.set_xticks(range(len(x_edges) - 1), [f"{x_edges[j]:g}-{x_edges[j + 1]:g}" for j in range(len(x_edges) - 1)], fontsize=7, rotation=30)
        ax.set_yticks(range(len(y_edges) - 1), [f"{y_edges[i]:g}-{y_edges[i + 1]:g}" for i in range(len(y_edges) - 1)], fontsize=7)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.set_title(name, fontsize=9, loc="left")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="share within 0.01 %")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def scatter(path, x, y, hit, x_label, y_label, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.6), dpi=150)
    ax.scatter(x[~hit], y[~hit], s=6, color="#c0504d", alpha=0.5, linewidths=0, label="miss")
    ax.scatter(x[hit], y[hit], s=6, color="#2b6cb0", alpha=0.5, linewidths=0, label="within 0.01 %")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8, frameon=False, markerscale=2)
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------ main


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="the encoder checkpoint (for the data arguments)")
    p.add_argument("--predictions", required=True, help="predictions.npz of period_probe (index, p_model, p_top)")
    p.add_argument("--head", default=None, help="predictions.npz of period_head (index, p_model), optional")
    p.add_argument("--out", required=True)
    p.add_argument("--n-objects", type=int, default=1000)
    p.add_argument("--budgets", type=int, nargs="+", default=[2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000])
    p.add_argument("--two-stage-budgets", type=int, nargs="+", default=[1000, 2000, 5000, 10000, 20000, 50000])
    p.add_argument("--peaks", type=int, default=5)
    p.add_argument("--astropy", action="store_true", help="also run the collaborator's astropy search at --budgets (1 band and multiband)")
    p.add_argument("--refine-rel", type=float, default=0.1)
    p.add_argument("--p-min", type=float, default=None, help="default: the shortest training period")
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
    n_win = {int(i): int(n) for i, n in zip(pred["index"], pred["n_windows"])} if "n_windows" in pred.files else {}
    pts = {int(i): float(n) for i, n in zip(pred["index"], pred["points"])} if "points" in pred.files else {}
    head = None
    if args.head:
        h = np.load(args.head, allow_pickle=True)
        head = {int(i): float(p) for i, p in zip(h["index"], h["p_model"])}
    idx = [int(i) for i in pred["index"] if data["validation"][int(i)].period and data["validation"][int(i)].period > 0]
    rng = np.random.default_rng(args.seed)
    if args.n_objects and len(idx) > args.n_objects:
        idx = sorted(rng.choice(idx, args.n_objects, replace=False).tolist())
    methods = [f"ls_full@{b}" for b in args.budgets] + [f"ls_two_stage@{b}" for b in args.two_stage_budgets] + ["model_refined", "model_top5"]
    if args.astropy:
        if run_lomb_scargle is None:
            raise SystemExit("--astropy needs lomb_scargle.py at the repository root and astropy")
        methods += [f"astropy_1b@{b}" for b in args.budgets] + [f"astropy_mb@{b}" for b in args.budgets]
    if head is not None:
        methods.append("head")
    periods = {m: np.full(len(idx), np.nan) for m in methods}
    trials = {m: np.zeros(len(idx)) for m in methods}
    seconds = {m: 0.0 for m in methods}
    props = dict(n_points=np.zeros(len(idx)), span=np.zeros(len(idx)), cadence=np.zeros(len(idx)), per_year=np.zeros(len(idx)),
                 n_windows=np.zeros(len(idx)), points_per_window=np.zeros(len(idx)), p_true=np.zeros(len(idx)), superclass=np.empty(len(idx), dtype=object))  # fmt: skip
    print(f"{len(idx)} validation stars; p_min {p_min:.4g} d; methods {methods}; {dev}", flush=True)
    for j, i in enumerate(idx):
        r = data["validation"][i]
        t, y, err, band, w = star_inputs(r)
        span = float(t.max() - t.min())
        props["n_points"][j], props["span"][j], props["p_true"][j], props["superclass"][j] = t.size, span, float(r.period), superclass(r)
        ts = np.sort(t)
        props["cadence"][j] = float(np.median(np.diff(ts))) if ts.size > 1 else np.nan
        props["per_year"][j] = t.size / max(span, 1.0) * 365.25
        props["n_windows"][j], props["points_per_window"][j] = n_win.get(i, np.nan), pts.get(i, np.nan)
        for b in args.budgets:
            m = f"ls_full@{b}"
            t0 = time.time()
            periods[m][j], trials[m][j] = ls_full(t, y, w, span, p_min, b, dev)
            seconds[m] += time.time() - t0
        for b in args.two_stage_budgets:
            m = f"ls_two_stage@{b}"
            t0 = time.time()
            periods[m][j], trials[m][j] = ls_two_stage(t, y, err, band, w, span, p_min, b, args.peaks, dev)
            seconds[m] += time.time() - t0
        if args.astropy:
            for b in args.budgets:
                for tag, mb in (("1b", False), ("mb", True)):
                    m = f"astropy_{tag}@{b}"
                    t0 = time.time()
                    periods[m][j], trials[m][j] = astropy_search(r, b, mb, p_min)
                    seconds[m] += time.time() - t0
        t0 = time.time()
        periods["model_refined"][j], trials["model_refined"][j] = model_seeded(t, y, err, band, [rough.get(i, np.nan)], args.refine_rel, dev)
        seconds["model_refined"] += time.time() - t0
        t0 = time.time()
        periods["model_top5"][j], trials["model_top5"][j] = model_seeded(t, y, err, band, tops.get(i, [rough.get(i, np.nan)]), args.refine_rel, dev)
        seconds["model_top5"] += time.time() - t0
        if head is not None:
            periods["head"][j], trials["head"][j] = head.get(i, np.nan), 20000  # the spectrum's bins; the wall-clock is the head's own forward, not timed here
        if (j + 1) % 100 == 0:
            print(f"  {j + 1} / {len(idx)} stars, {time.time() - t_start:.0f}s", flush=True)
    p_true = props["p_true"]
    sup = props["superclass"].astype(str)
    res = dict(n_stars=len(idx), p_min=p_min, predictions=str(args.predictions), head=str(args.head), methods={})
    for m in methods:
        res["methods"][m] = dict(trials_median=float(np.median(trials[m])), seconds_per_star=seconds[m] / len(idx), **hit_table(periods[m], p_true),
                                 by_superclass={g: hit_table(periods[m][sup == g], p_true[sup == g]) for g in HEADLINE if (sup == g).sum() >= 5})  # fmt: skip
    # the crossing: the budget at which the Lomb-Scargle curves reach the model-seeded hit rate
    def curve(prefix, key):
        ms = [m for m in methods if m.startswith(prefix)]
        return np.array([res["methods"][m]["trials_median"] for m in ms]), np.array([res["methods"][m][key] for m in ms])

    model_methods = ("model_refined", "model_top5") + (("head",) if head else ())
    res["crossing"] = {}
    for key, label in (("within_0.0001", "within 0.01%"), ("alias_tolerant_0.0001", "alias-tolerant 0.01%")):
        curves = {"Lomb-Scargle, one grid": curve("ls_full@", key), "Lomb-Scargle, coarse grid + local search": curve("ls_two_stage@", key)}
        for cname, (xs, ys) in curves.items():
            for m in model_methods:
                target = res["methods"][m][key]
                above = np.flatnonzero(ys >= target)
                res["crossing"][f"[{label}] {cname} to match {m}"] = float(xs[above[0]]) if above.size else float("inf")
    curves = {"Lomb-Scargle, one grid": curve("ls_full@", "within_0.0001"), "Lomb-Scargle, coarse grid + local search": curve("ls_two_stage@", "within_0.0001")}
    if args.astropy:
        curves["astropy, best band"] = curve("astropy_1b@", "within_0.0001")
        curves["astropy, multiband"] = curve("astropy_mb@", "within_0.0001")
        for key, label in (("within_0.0001", "within 0.01%"), ("alias_tolerant_0.0001", "alias-tolerant 0.01%")):
            for cname, prefix in (("astropy, best band", "astropy_1b@"), ("astropy, multiband", "astropy_mb@")):
                xs, ys = curve(prefix, key)
                for m in model_methods:
                    target = res["methods"][m][key]
                    above = np.flatnonzero(ys >= target)
                    res["crossing"][f"[{label}] {cname} to match {m}"] = float(xs[above[0]]) if above.size else float("inf")
    lines = {m: (res["methods"][m]["trials_median"], res["methods"][m]["within_0.0001"]) for m in model_methods}
    dump_json(res, out / "results.json")
    np.savez_compressed(out / "per_star.npz", index=np.array(idx), p_true=p_true, superclass=sup, **{f"p_{m}": periods[m] for m in methods},
                        **{f"trials_{m}": trials[m] for m in methods}, **{k: v for k, v in props.items() if k not in ("superclass", "p_true")})  # fmt: skip
    # tables
    lines_md = ["# Period search: the model's prior against Lomb-Scargle\n",
                f"{len(idx)} validation stars, whole light curves. Trials = frequencies evaluated per star (median); seconds = GPU wall-clock per star. "
                "Hits = share of stars within the tolerance of the catalogue period; alias-tolerant also accepts the double and the half.\n",
                "| method | trials | s/star | within 10% | 1% | 0.1% | 0.01% | alias-tolerant 0.01% |", "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        v = res["methods"][m]
        lines_md.append(f"| {m} | {v['trials_median']:.0f} | {v['seconds_per_star']:.3f} | {v['within_0.1']:.3f} | {v['within_0.01']:.3f} | {v['within_0.001']:.3f} | {v['within_0.0001']:.3f} | {v['alias_tolerant_0.0001']:.3f} |")
    lines_md.append("\n## crossing: trials Lomb-Scargle needs to match the model-seeded search (strict, then accepting the double and the half)\n")
    for k, v in res["crossing"].items():
        lines_md.append(f"- {k}: {'never within the budgets tried' if not np.isfinite(v) else f'{v:.0f} trials'}")
    lines_md.append("\n## within 0.01% by superclass\n")
    groups = [g for g in HEADLINE if (sup == g).sum() >= 5]
    lines_md.append("| method | " + " | ".join(f"{g} ({(sup == g).sum()})" for g in groups) + " |")
    lines_md.append("|---|" + "---|" * len(groups))
    for m in methods:
        lines_md.append(f"| {m} | " + " | ".join(f"{res['methods'][m]['by_superclass'][g]['within_0.0001']:.2f}" for g in groups) + " |")
    (out / "tables.md").write_text("\n".join(lines_md) + "\n")
    # figures
    plot_crossing(out / "crossing.png", curves, lines)
    best_ls = max((m for m in methods if m.startswith("ls_")), key=lambda m: res["methods"][m]["within_0.0001"])
    hit = {"model top-5 + fine search": np.abs(periods["model_top5"] / p_true - 1) < 1e-4, best_ls: np.abs(periods[best_ls] / p_true - 1) < 1e-4}
    ok = np.isfinite(props["points_per_window"]) & np.isfinite(props["n_windows"])
    heat(out / "heat_points_windows.png", {k: v[ok] for k, v in hit.items()}, props["points_per_window"][ok], props["n_windows"][ok],
         np.array([0, 30, 60, 120, 1000]), np.array([0, 10, 20, 40, 200]), "points per window", "valid windows", "hit rate within 0.01 %")  # fmt: skip
    heat(out / "heat_cadence_span.png", hit, props["cadence"], props["span"], np.array([0, 1, 2, 4, 10, 1000]), np.array([0, 1000, 2000, 2500, 3000, 5000]),
         "median gap between observations (days)", "baseline (days)", "hit rate within 0.01 %")  # fmt: skip
    scatter(out / "scatter_points_windows.png", props["n_points"], props["n_windows"].clip(1), hit["model top-5 + fine search"], "points in the light curve",
            "valid windows", "model top-5 + fine search: hits over the data")  # fmt: skip
    print("\n".join(lines_md[3:3 + len(methods) + 2]))
    print(f"wrote {out} in {time.time() - t_start:.0f}s")
    return res


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
