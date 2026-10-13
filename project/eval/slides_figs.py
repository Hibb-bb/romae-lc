"""Figures for the talk: (1) one raw ZTF light curve over its whole baseline
with a 30-day zoom; (2) the hit rate within 0.01 % by class with the talk's
method names (model alone, model -> LS, model + aliases -> LS, Lomb-Scargle
500k trials), in the artifact's colour scheme.

    python -m project.eval.slides_figs --out project/results/slides
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project.common import DATA_ROOT
from project.eval.period_report import rel_error
from project.plot_period_examples import INK, INK2, SURFACE, style
from romae_lc.data import load_pc

C = dict(model="#6baed6", refined="#2171b5", cands="#08306b", ls="#e6550d", ls_alias="#fd8d3c")
BAND = {0: ("g", "#2a9d3f"), 1: ("r", "#d6452a"), 2: ("i", "#7b3fb8")}
CLS = {"ECL": "eclipsing", "RR": "RR Lyrae", "ROT": "rotation", "CEP": "Cepheid", "DSCT": "δ Scuti", "LPV": "long-period"}


def raw_light_curve(out: Path, cls: str = "RRAB", min_points: int = 800, min_span: float = 2000.0, zoom_days: float = 30.0):
    recs = load_pc(DATA_ROOT, "validation", label_field="class_str", classes=(cls,), max_rows=4000)
    recs = [r for r in recs if r.t.size >= min_points and r.t.max() - r.t.min() >= min_span]
    r = min(recs, key=lambda r: abs(r.t.size - 1200))  # a typical star, not the densest
    fig, (ax, az) = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw=dict(width_ratios=[2.6, 1]))
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, az):
        style(a)
    # a 30-day stretch with the most points, for the zoom
    t = r.t
    best, lo = 0, 0.0
    for s in np.arange(0, t.max() - zoom_days, 5.0):
        n = int(((t >= s) & (t < s + zoom_days)).sum())
        if n > best:
            best, lo = n, s
    for b in np.unique(r.band):
        m = r.band == b
        name, col = BAND.get(int(b), (str(b), INK2))
        ax.errorbar(t[m], -r.y[m], yerr=r.err[m], fmt="o", ms=2, color=col, ecolor=col, elinewidth=0.5, alpha=0.75, label=f"ZTF {name}")
        mz = m & (t >= lo) & (t < lo + zoom_days)
        az.errorbar(t[mz], -r.y[mz], yerr=r.err[mz], fmt="o", ms=3.5, color=col, ecolor=col, elinewidth=0.7, alpha=0.9)
    for a in (ax, az):
        a.invert_yaxis()
        a.set_xlabel("days since the first observation", fontsize=9)
    ax.set_ylabel("magnitude", fontsize=9)
    ax.set_title(f"one ZTF light curve: {r.meta.get('class_str')}, {r.t.size} points over {t.max() - t.min():.0f} days, period {r.period:.4f} d", fontsize=9.5, loc="left", color=INK)
    az.set_title(f"{zoom_days:.0f} days, from day {lo:.0f}", fontsize=9.5, loc="left", color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "raw_light_curve.png", dpi=160, facecolor=SURFACE)
    return r


def bars_by_class(out: Path, report: str, ls_extra: str, tol: float = 1e-4, label: str = "0.01 %"):
    d = np.load(Path(report) / "per_star.npz", allow_pickle=True)
    e = np.load(Path(ls_extra) / "per_star.npz", allow_pickle=True)
    p_true, sup = d["p_true"], d["superclass"].astype(str)
    methods = [("model alone", d["p_model"], C["model"], None, False),
               ("model → LS", d["p_model_refined"], C["refined"], None, False),
               ("model + aliases → LS", d["p_model_cands"], C["cands"], None, False),
               ("Lomb-Scargle, 500k trials", e["p_astropy_mb@500000"], C["ls"], "//", False),
               ("Lomb-Scargle, 500k trials, double and half accepted", e["p_astropy_mb@500000"], C["ls_alias"], "..", True)]
    groups = [g for g in CLS if (sup == g).sum() >= 20]
    counts = {g: int((sup == g).sum()) for g in groups}
    fig, ax = plt.subplots(figsize=(9.6, 3.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    x = np.arange(len(groups))
    w = 0.8 / len(methods)
    for i, (name, p, col, hatch, alias) in enumerate(methods):
        hit = np.where(np.isfinite(p), rel_error(p, p_true, alias) < tol, False)
        ax.bar(x + (i - len(methods) / 2) * w + w / 2, [hit[sup == g].mean() for g in groups], w, color=col, hatch=hatch, edgecolor=SURFACE if hatch else col, lw=0.5, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{CLS[g]}\n({counts[g]})" for g in groups], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of the stars", fontsize=8.5, color=INK2)
    ax.set_title(f"Period within {label} of the catalogue, by class ({len(p_true):,} validation stars)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out / f"bars_by_class_{tol:g}.png", facecolor=SURFACE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="project/results/slides")
    ap.add_argument("--report", default="project/results/period_report_spec4")
    ap.add_argument("--ls-extra", default="project/results/ls_allstars_spec")
    ap.add_argument("--cls", default="RRAB")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bars_by_class(out, a.report, a.ls_extra)
    r = raw_light_curve(out, a.cls)
    print(f"wrote {out}: raw_light_curve.png ({r.meta.get('id')}), bars_by_class_0.0001.png")


if __name__ == "__main__":
    main()
