"""The model-seeded search over its own budget, on the benchmark's stars.

The Lomb-Scargle benchmark (``ls_benchmark``) gives Lomb-Scargle a growing
budget of trial frequencies. This script gives the model the same axis: the
fine search around the encoder's period is run with a growing width
(``--rels``, as a share of the period) and a growing number of seed peaks
(``--tops``, the read-out's best bins), and every setting records its
trial count and its hit rates. Candidates are compared the way
``period_report`` does it: the wave's harmonics scale with the period so
every candidate reaches the same highest frequency, the first seed (the
read-out's best bin) is the reference, and another seed's period replaces
it only when its fold beats it by more than ``--cand-margin``.

Writes ``results.json`` (per setting: trials, hits) and ``per_star.npz``;
``plot_budget`` draws the three-panel figure against the Lomb-Scargle
curves of a benchmark folder.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from project import fold
from project.common import add_device_arg, data_args_from, dump_json, get_device, load_data, load_encoder, superclass
from project.eval.ls_benchmark import hit_table, star_inputs
from project.eval.period_report import h_of


def model_search(t, y, err, band, seeds, rel, harmonics, margin, device):
    """``(period, trials)``: every seed sharpened within ``rel``; harmonics
    scaled to the first seed, which is the reference; another seed's period
    replaces it only when its fold beats it by more than ``margin``."""
    found, trials, p_ref = [], 0, None
    for s in seeds:
        if not (np.isfinite(s) and s > 0):
            continue
        p_ref = p_ref or float(s)
        p, q, n = fold.refine_period(t, y, err, band, float(s), rel, harmonics=h_of(float(s), p_ref, harmonics), device=device)
        trials += int(n)
        if np.isfinite(q):
            found.append((float(p), float(q)))
    if not found:
        return float("nan"), trials
    ref_p, ref_r2 = found[0]
    better = [f for f in found[1:] if f[1] > ref_r2 + margin]
    return (max(better, key=lambda f: f[1])[0] if better else ref_p), trials


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--predictions", required=True, help="predictions.npz with p_top")
    p.add_argument("--out", required=True)
    p.add_argument("--n-objects", type=int, default=1000, help="the benchmark's subset (same seed, same draw)")
    p.add_argument("--rels", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    p.add_argument("--tops", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--harmonics", type=int, default=6)
    p.add_argument("--cand-margin", type=float, default=0.02)
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
    data = load_data(data_args_from(meta.args, over), splits=("validation",))
    pred = np.load(args.predictions, allow_pickle=True)
    if "p_top" not in pred.files:
        raise SystemExit("the predictions need p_top (the joint or bins read-out saves it)")
    tops = {int(i): [float(v) for v in row] for i, row in zip(pred["index"], pred["p_top"])}
    idx = [int(i) for i in pred["index"] if data["validation"][int(i)].period and data["validation"][int(i)].period > 0]
    rng = np.random.default_rng(args.seed)
    if args.n_objects and len(idx) > args.n_objects:
        idx = sorted(rng.choice(idx, args.n_objects, replace=False).tolist())
    settings = [(rel, k) for k in args.tops for rel in args.rels]
    names = [f"top{k}_rel{rel:g}" for rel, k in settings]
    periods = {m: np.full(len(idx), np.nan) for m in names}
    trials = {m: np.zeros(len(idx)) for m in names}
    p_true, sup = np.zeros(len(idx)), np.empty(len(idx), dtype=object)
    print(f"{len(idx)} stars; settings {names}; {dev}", flush=True)
    for j, i in enumerate(idx):
        r = data["validation"][i]
        t, y, err, band, _ = star_inputs(r)
        p_true[j], sup[j] = float(r.period), superclass(r)
        for (rel, k), m in zip(settings, names):
            periods[m][j], trials[m][j] = model_search(t, y, err, band, tops[i][:k], rel, args.harmonics, args.cand_margin, dev)
        if (j + 1) % 100 == 0:
            print(f"  {j + 1} / {len(idx)} stars, {time.time() - t_start:.0f}s", flush=True)
    sup = sup.astype(str)
    res = dict(n_stars=len(idx), predictions=str(args.predictions), harmonics=args.harmonics, cand_margin=args.cand_margin, settings={})
    for (rel, k), m in zip(settings, names):
        res["settings"][m] = dict(rel=rel, top=k, trials_median=float(np.median(trials[m])), trials_mean=float(np.mean(trials[m])), **hit_table(periods[m], p_true))
    dump_json(res, out / "results.json")
    np.savez_compressed(out / "per_star.npz", index=np.array(idx), p_true=p_true, superclass=sup, **{f"p_{m}": periods[m] for m in names}, **{f"trials_{m}": trials[m] for m in names})
    for m in names:
        v = res["settings"][m]
        print(f"{m:14s} trials {v['trials_median']:8.0f}  within 10% {v['within_0.1']:.3f}  1% {v['within_0.01']:.3f}  0.01% {v['within_0.0001']:.3f}  alias {v['alias_tolerant_0.0001']:.3f}")
    print(f"wrote {out} in {time.time() - t_start:.0f}s")
    return res


def plot_budget(path, sweep: dict, bench: dict, title: str = "Hit rate against the search budget, 1,000 validation stars"):
    """Three panels (within 10 %, 1 %, 0.01 %) of hit rate against trial
    frequencies per star: the Lomb-Scargle variants of ``bench`` in a muted
    colour, the model's search in a strong one, one line per number of
    seeds over the widths; labels at the line ends, no legend box."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    LS, MODEL = "#b9a99a", ["#1d6fd1", "#0b4fa0", "#062f63"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURF)
    ls_methods = {"multiband": sorted((m for m in bench["methods"] if m.startswith("astropy_mb@")), key=lambda m: int(m.split("@")[1])),
                  "best band": sorted((m for m in bench["methods"] if m.startswith("astropy_1b@")), key=lambda m: int(m.split("@")[1]))}
    tops = sorted({v["top"] for v in sweep["settings"].values()})
    for ax, (key, lab) in zip(axes, (("within_0.1", "within 10 %"), ("within_0.01", "within 1 %"), ("within_0.0001", "within 0.01 %"))):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
        ax.tick_params(colors=INK2, labelsize=7.5)
        for name, ms in ls_methods.items():
            if not ms:
                continue
            x = [bench["methods"][m]["trials_median"] for m in ms]
            yv = [bench["methods"][m][key] for m in ms]
            ax.plot(x, yv, color=LS, lw=1.2, alpha=0.8, marker="o", ms=3, ls="-" if name == "multiband" else (0, (3, 2)))
            ax.annotate(f"Lomb-Scargle, {name}", (x[-1], yv[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5, color=INK2, va="center")
        for c, k in zip(MODEL, tops):
            items = sorted((v for v in sweep["settings"].values() if v["top"] == k), key=lambda v: v["trials_median"])
            x = [v["trials_median"] for v in items]
            yv = [v[key] for v in items]
            ax.plot(x, yv, color=c, lw=1.4, alpha=0.85, marker="o", ms=3.5)
            ax.annotate(f"model, top {k} seed{'s' if k > 1 else ''}", (x[-1], yv[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5, color=c, va="center")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"hit rate {lab}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("trial frequencies per star", fontsize=8, color=INK2)
    axes[0].set_ylabel("share of the stars", fontsize=8, color=INK2)
    lo = min(v["trials_median"] for v in sweep["settings"].values())
    hi = max(bench["methods"][m]["trials_median"] for ms in ls_methods.values() for m in ms)
    axes[0].set_xlim(lo * 0.7, hi * 6)
    fig.suptitle(title + "  (the model's points: search width 1, 3, 10, 30 % of its period)", fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
