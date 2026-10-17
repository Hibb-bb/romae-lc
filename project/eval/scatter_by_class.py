"""Predicted against catalogue period, every validation star, coloured by
fine class (all classes), delta Scuti drawn last and larger. Two panels: the
model alone (the read-out on the latent) and model + aliases -> LS.

    python -m project.eval.scatter_by_class --out project/results/slides
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project.plot_period_examples import INK, INK2, SURFACE, style

FS = dict(title=20, label=18, tick=16, legend=14)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#8e44ad", "#c0392b", "#d4a017", "#16a085", "#7f8c8d", "#e84393", "#2c3e50", "#a0522d", "#00a8cc", "#6c5ce7", "#b2bec3", "#e17055", "#00b894"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default="project/results/period_report_spec4")
    ap.add_argument("--out", default="project/results/slides")
    ap.add_argument("--highlight", default="DSCT")
    a = ap.parse_args(argv)
    d = np.load(Path(a.report) / "per_star.npz", allow_pickle=True)
    p_true, fine = d["p_true"], d["fine"].astype(str)
    classes = [c for c, _ in sorted(zip(*np.unique(fine, return_counts=True)), key=lambda x: -x[1])]
    colour = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}
    panels = [("model alone (the read-out on the latent)", d["p_model"]), ("(model + aliases) → LS", d["p_model_cands"])]
    fig, axes = plt.subplots(1, 2, figsize=(19, 9), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    lo, hi = 0.03, 600
    for ax, (title, p) in zip(axes, panels):
        style(ax)
        for f in (1.0, 2.0, 0.5):
            ax.plot([lo, hi], [lo * f, hi * f], color=INK2, lw=0.8, ls="-" if f == 1 else (0, (3, 3)), alpha=0.6)
        order = [c for c in classes if c != a.highlight] + [a.highlight]
        for c in order:
            m = (fine == c) & np.isfinite(p)
            hl = c == a.highlight
            ax.scatter(p_true[m], p[m], s=34 if hl else 9, color=colour[c], alpha=0.95 if hl else 0.55, edgecolors=INK if hl else "none", linewidths=0.6, label=f"{c} ({m.sum()})", zorder=3 if hl else 2)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("catalogue period (days)", fontsize=FS["label"], color=INK)
        ax.set_title(title, fontsize=FS["title"], color=INK, loc="left")
        ax.tick_params(labelsize=FS["tick"], labelcolor=INK)
    axes[0].set_ylabel("predicted period (days)", fontsize=FS["label"], color=INK)
    axes[1].legend(fontsize=FS["legend"], frameon=False, loc="lower right", ncol=2, markerscale=2.2, labelcolor=INK)
    fig.suptitle(f"Predicted against catalogue period, {len(p_true):,} validation stars by class; dashed lines: the double and the half", fontsize=FS["title"], color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(Path(a.out) / f"scatter_by_class_{a.highlight}.png", facecolor=SURFACE)
    # the numbers behind the question: the highlighted class by period
    m = fine == a.highlight
    for name, p in panels:
        rel = np.abs(p[m] / p_true[m] - 1)
        print(f"{a.highlight}, {name}: n {m.sum()}, within 10 % {np.mean(rel < .1):.2f}; periods {np.round(np.sort(p_true[m]), 3).tolist()[:12]} ...")
        far = p[m] / p_true[m]
        print("   ratio predicted/true for the misses:", np.round(np.sort(far[rel >= .1]), 2).tolist())
    print(f"wrote {Path(a.out) / f'scatter_by_class_{a.highlight}.png'}")


if __name__ == "__main__":
    main()
