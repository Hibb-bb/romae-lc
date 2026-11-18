"""One figure per class: predicted against catalogue period for every
validation star in grey, the class in colour; stars with sparse coverage
(fewer than ``--min-points`` observations or fewer than ``--min-windows``
valid 250-day windows) get a red edge. Two panels: the model alone and
(model + aliases) -> LS.

    python -m project.eval.scatter_per_class --out project/results/slides/scatter_per_class
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from project.plot_period_examples import INK, INK2, SURFACE, style

FS = dict(title=20, label=18, tick=16, legend=15)
GREY, RED = "#c3c7cc", "#d62728"


def valid_windows(latents: str, split: str, index: np.ndarray, min_tokens: int = 16) -> np.ndarray:
    """Valid 250-day windows per object of ``split`` (by record index) from the latent cache."""
    c = torch.load(latents, map_location="cpu", weights_only=False)
    ptr, n_tok = c["ptr"][split].numpy(), c["n_tokens"].numpy().astype(int)
    obj_index = c["objects"][split]["index"].numpy()
    counts = {int(obj_index[i]): int((n_tok[ptr[i]:ptr[i + 1]] >= min_tokens).sum()) for i in range(len(ptr) - 1)}
    return np.array([counts.get(int(i), 0) for i in index])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default="project/results/period_report_spec4")
    ap.add_argument("--latents", default="project/runs/maew_spec/latents100k.pt")
    ap.add_argument("--out", default="project/results/slides/scatter_per_class")
    ap.add_argument("--min-points", type=int, default=300)
    ap.add_argument("--min-windows", type=int, default=10)
    ap.add_argument("--min-class", type=int, default=8, help="skip classes with fewer stars")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = np.load(Path(a.report) / "per_star.npz", allow_pickle=True)
    p_true, fine, n_points = d["p_true"], d["fine"].astype(str), d["n_points"]
    n_win = valid_windows(a.latents, "validation", d["index"])
    sparse = (n_points < a.min_points) | (n_win < a.min_windows)
    panels = [("model alone", d["p_model"]), ("(model + aliases) → LS", d["p_model_cands"])]
    lo, hi = 0.03, 600
    classes = [c for c, n in zip(*np.unique(fine, return_counts=True)) if n >= a.min_class]
    for cls in classes:
        m = fine == cls
        colour = "#2a78d6"
        fig, axes = plt.subplots(1, 2, figsize=(19, 9), dpi=140)
        fig.patch.set_facecolor(SURFACE)
        for ax, (title, p) in zip(axes, panels):
            style(ax)
            for f in (1.0, 2.0, 0.5):
                ax.plot([lo, hi], [lo * f, hi * f], color=INK2, lw=0.8, ls="-" if f == 1 else (0, (3, 3)), alpha=0.6)
            ok = np.isfinite(p)
            ax.scatter(p_true[~m & ok], p[~m & ok], s=7, color=GREY, alpha=0.6, edgecolors="none", label="every other class", zorder=1)
            dense, thin = m & ok & ~sparse, m & ok & sparse
            ax.scatter(p_true[dense], p[dense], s=40, color=colour, alpha=0.9, edgecolors="none", label=f"{cls} ({m.sum()})", zorder=3)
            ax.scatter(p_true[thin], p[thin], s=46, color=colour, alpha=0.9, edgecolors=RED, linewidths=1.6, label=f"{cls}, sparse: under {a.min_points} points or {a.min_windows} valid windows ({thin.sum()})", zorder=4)
            rel = np.abs(p[m & ok] / p_true[m & ok] - 1)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel("catalogue period (days)", fontsize=FS["label"], color=INK)
            ax.set_title(f"{title}\n{cls}: within 10 % {np.mean(rel < .1):.2f}, within 0.01 % {np.mean(rel < 1e-4):.2f}", fontsize=FS["title"] - 2, color=INK, loc="left")
            ax.tick_params(labelsize=FS["tick"], labelcolor=INK)
        axes[0].set_ylabel("predicted period (days)", fontsize=FS["label"], color=INK)
        axes[1].legend(fontsize=FS["legend"], frameon=False, loc="lower right", markerscale=1.6, labelcolor=INK)
        fig.suptitle(f"{cls}: predicted against catalogue period, {len(p_true):,} validation stars; dashed lines: the double and the half", fontsize=FS["title"], color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out / f"scatter_{cls.replace('/', '-')}.png", facecolor=SURFACE)
        plt.close(fig)
        print(f"{cls:14s} n {m.sum():5d} sparse {int((m & sparse).sum()):4d}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
