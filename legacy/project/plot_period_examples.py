"""Figures for the period results of :mod:`project.period_probe`.

Two kinds of figure, from the ``fold_r2.npz`` a period probe run wrote:

- Folded light curves of a few stars, good and bad. Every star gets three
  panels: folded on the catalogue period, on the model's own period, and on
  the model's period after the fine search. The stars are typical ones, not
  the best ones: for every group the star nearest the group's median fold R2.
- Predicted against true period for every validation star, in days and in
  log period, for the model alone and after the fine search.

    python -m project.plot_period_examples --results project/results/period_maew_refine2 \\
        --latents project/runs/maew_w250/latents_r4.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from project.common import data_args_from, load_data

# surfaces and ink
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# the first three categorical slots are safe together in a scatter; the rest
# of the classes fold into a neutral "other"
CLASS_COLOUR = {"ECL": "#2a78d6", "RR": "#eb6834", "ROT": "#1baf7a"}
OTHER = "#8a8983"
CLASS_NAME = {"ECL": "eclipsing binaries", "RR": "RR Lyrae", "ROT": "rotation", "other": "other classes"}
BAND_COLOUR = ["#2a78d6", "#eb6834", "#1baf7a"]
BAND_NAME = {0: "g band", 1: "r band", 2: "i band"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7, length=2)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def load_stars(latents: str, n_expected: int, p_cat: np.ndarray):
    """The validation records in the order of the probe's arrays, with their
    superclass and fine class."""
    cache = torch.load(latents, map_location="cpu", weights_only=False)
    obj, ptr = cache["objects"]["validation"], cache["ptr"]["validation"]
    nt, mt = cache["n_tokens"].numpy(), int(cache["meta"]["min_tokens"])
    per = obj["period"].numpy()
    keep = np.array(
        [
            (nt[int(ptr[i]) : int(ptr[i + 1])] >= mt).any() and np.isfinite(per[i]) and per[i] > 0
            for i in range(len(per))
        ]
    )
    index = obj["index"].numpy()[keep]
    assert len(index) == n_expected and np.allclose(per[keep], p_cat, rtol=1e-5)
    ckpt = torch.load(cache["meta"]["ckpt"], map_location="cpu", weights_only=False)
    data = load_data(data_args_from(ckpt["args"]), splits=("validation",))
    recs = [data["validation"][int(i)] for i in index]
    sup = np.array(obj["superclass"])[keep]
    fine = np.array([str(r.meta.get("class_str", "?")) for r in recs])
    return recs, sup, fine


def typical(mask: np.ndarray, value: np.ndarray, n_points: np.ndarray, k: int = 1):
    """The ``k`` stars of ``mask`` nearest the median of ``value``, among the
    ones with a usual number of points (so the panel is readable)."""
    idx = np.flatnonzero(mask & np.isfinite(value) & (n_points >= 300))
    if len(idx) == 0:
        idx = np.flatnonzero(mask & np.isfinite(value))
    if len(idx) == 0:
        return []
    med = np.median(value[idx])
    return idx[np.argsort(np.abs(value[idx] - med))[:k]].tolist()


def fold_panel(ax, r, period, title, r2):
    style(ax)
    t0 = float(r.t.min())
    for b in np.unique(r.band):
        m = r.band == b
        ph = np.mod((r.t[m] - t0) / period, 1.0)
        ph2, y2 = np.concatenate([ph, ph + 1.0]), np.concatenate([r.y[m], r.y[m]])
        ax.scatter(ph2, y2, s=3, color=BAND_COLOUR[int(b) % 3], alpha=0.55, linewidths=0)
    lo, hi = np.percentile(r.y, [0.5, 99.5])
    pad = 0.15 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, 2)
    ax.set_title(f"{title}\nP = {period:.6g} d, fold R2 {r2:.2f}", fontsize=7.5, color=INK, loc="left")


def fold_figure(path, rows, recs, fine, d, heading):
    """One row per star: catalogue period, model alone, model after the search."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(10.5, 2.35 * len(rows)), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax_row, (i, note) in zip(axes, rows):
        r = recs[i]
        pc, pm, pr = d["p_catalogue"][i], d["p_model"][i], d["p_refined"][i]
        fold_panel(ax_row[0], r, pc, "catalogue period", d["catalogue"][i])
        fold_panel(ax_row[1], r, pm, f"model alone ({pm / pc - 1:+.2%} off)", d["model"][i])
        off = pr / pc - 1
        how = f"{off:+.4%} off" if abs(off) < 0.1 else f"{pr / pc:.2f} x catalogue"
        fold_panel(ax_row[2], r, pr, f"model + fine search ({how})", d["r2_refined"][i])
        ax_row[0].set_ylabel(f"{fine[i]}\n{note}\n\nbrightness", fontsize=7.5, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("phase (two cycles shown)", fontsize=7.5, color=INK2)
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=5, color=BAND_COLOUR[j], label=BAND_NAME[j])
        for j in range(3)
    ]
    fig.legend(handles=handles, fontsize=8, frameon=False, ncol=3, loc="upper right",
               bbox_to_anchor=(0.99, 0.995), labelcolor=INK2)  # fmt: skip
    fig.suptitle(heading, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def scatter_panel(ax, x, y, sup, log, lim):
    style(ax)
    groups = [("other", ~np.isin(sup, list(CLASS_COLOUR)), OTHER)] + [
        (g, sup == g, c) for g, c in CLASS_COLOUR.items()
    ]
    for g, m, c in groups:
        ax.scatter(x[m], y[m], s=3, color=c, alpha=0.45, linewidths=0,
                   label=f"{CLASS_NAME[g]} ({int(m.sum())})")  # fmt: skip
    a = np.array(lim, dtype=float)
    for k in (1.0, 2.0, 0.5):
        ax.plot(a, a * k, color=INK2, lw=0.7, ls="-" if k == 1 else (0, (4, 3)))
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")


def scatter_figure(path, d, sup):
    pc = d["p_catalogue"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for row, (key, name) in enumerate((("p_model", "model alone"), ("p_refined", "model after the fine search"))):
        p = d[key]
        ok = np.isfinite(p) & (p > 0)
        hit = np.abs(p / pc - 1) < 0.1
        inside = (pc < 1.5) & (p < 1.5)
        scatter_panel(axes[row][0], pc[ok], p[ok], sup[ok], False, (0.0, 1.5))
        axes[row][0].set_title(
            f"{name}: period in days, periods under 1.5 days\n({inside.mean():.0%} of the stars are in this range)",
            fontsize=8.5, color=INK, loc="left",
        )  # fmt: skip
        scatter_panel(axes[row][1], pc[ok], p[ok], sup[ok], True, (0.03, 1500.0))
        axes[row][1].set_title(
            f"{name}: every star, log axes\n{hit.mean():.0%} within 10 % of the catalogue period",
            fontsize=8.5, color=INK, loc="left",
        )  # fmt: skip
        for ax in axes[row]:
            ax.set_xlabel("catalogue period (days)", fontsize=8, color=INK2)
            ax.set_ylabel("predicted period (days)", fontsize=8, color=INK2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    order = [1, 2, 3, 0]  # the three classes, then "other"
    handles = [handles[i] for i in order] + [
        plt.Line2D([], [], color=INK2, lw=0.9),
        plt.Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 3))),
    ]
    labels = [labels[i] for i in order] + ["same period", "twice or half the period"]
    fig.suptitle("Predicted against catalogue period, validation stars", fontsize=10, color=INK, x=0.01, ha="left")
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=6, markerscale=4, loc="upper left",
               bbox_to_anchor=(0.005, 0.972), labelcolor=INK2, handletextpad=0.4, columnspacing=1.4)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="a period_probe output directory")
    p.add_argument("--latents", required=True, help="the latents.pt the probe used")
    args = p.parse_args(argv)
    res = Path(args.results)
    out = res / "figures"
    out.mkdir(exist_ok=True)
    d = dict(np.load(res / "fold_r2.npz"))
    pc, pm, pr = d["p_catalogue"], d["p_model"], d["p_refined"]
    recs, sup, fine = load_stars(args.latents, len(pc), pc)
    n_pts = np.array([r.n for r in recs])

    off = np.abs(pr / pc - 1)
    sharp = off < 1e-4
    ratio = pr / pc
    alias = ((np.abs(ratio - 2) < 0.2) | (np.abs(ratio - 0.5) < 0.05)) & (off >= 0.1)
    df = np.abs(1 / pr - 1 / pc)
    daily = (off >= 0.1) & ~alias & (np.abs(df - 1.0027) < 0.03)
    plain = (off >= 0.1) & ~alias & ~daily
    r2 = d["r2_refined"]

    good = []
    for cls, note in (("EW/EB", "contact binary"), ("EA", "detached binary"), ("RRAB", "RR Lyrae"),
                      ("RRC", "RR Lyrae"), ("ROT", "rotation")):  # fmt: skip
        good += [(i, note) for i in typical(sharp & (fine == cls), r2, n_pts)]
    bad = []
    bad += [(i, "found half or twice the period") for i in typical(alias & (sup == "ECL"), r2, n_pts)]
    bad += [(i, "daily alias") for i in typical(daily, r2, n_pts)]
    bad += [(i, "plain miss") for i in typical(plain & (sup == "ECL"), r2, n_pts)]
    bad += [(i, "plain miss") for i in typical(plain & (sup == "ROT"), r2, n_pts)]
    bad += [(i, "rare class, plain miss") for i in typical(plain & np.isin(sup, ["LPV", "CEP", "DSCT"]), r2, n_pts)]

    fold_figure(out / "fold_good.png", good, recs, fine, d,
                "Stars the model gets right: folded on three periods (typical stars, not the best ones)")  # fmt: skip
    fold_figure(out / "fold_bad.png", bad, recs, fine, d,
                "Stars the model gets wrong: folded on three periods (typical stars of every kind of miss)")  # fmt: skip
    scatter_figure(out / "period_scatter.png", d, sup)
    print(f"sharp {sharp.sum()}, alias {alias.sum()}, daily alias {daily.sum()}, plain miss {plain.sum()} of {len(pc)}")
    for name, rows in (("good", good), ("bad", bad)):
        for i, note in rows:
            print(f"  {name:4s} {fine[i]:12s} {note:32s} P_cat {pc[i]:.6f} model {pm[i]:.6f} refined {pr[i]:.6f} "
                  f"R2 cat {d['catalogue'][i]:.2f} refined {r2[i]:.2f} points {n_pts[i]}")  # fmt: skip
    print(f"wrote {out}/fold_good.png, fold_bad.png, period_scatter.png")


if __name__ == "__main__":
    main()
