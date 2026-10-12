"""Two plots and one table: hit rate against the tolerance for every method,
strict (the period itself) and with the double and the half accepted.

    python -m project.eval.hit_curves --report project/results/period_report_spec4 \
        --ls-extra project/results/ls_allstars_spec --out project/results/hit_curves
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project.eval.period_report import rel_error
from project.plot_period_examples import INK, SURFACE, style

TOLS = [0.2, 0.1, 0.01, 0.001, 0.0001]
LABEL = ["20 %", "10 %", "1 %", "0.1 %", "0.01 %"]
STY = {"model alone": ("#6baed6", "-", "o"), "model + fine search": ("#2171b5", "-", "o"), "model candidates + search": ("#08306b", "-", "o"),
       "Lomb-Scargle, 100k trials": ("#fdae6b", (0, (5, 2.5)), "s"), "Lomb-Scargle, 200k trials": ("#f16913", (0, (5, 2.5)), "s"), "Lomb-Scargle, 500k trials": ("#a63603", (0, (5, 2.5)), "s")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default="project/results/period_report_spec4")
    ap.add_argument("--ls-extra", default="project/results/ls_allstars_spec")
    ap.add_argument("--out", default="project/results/hit_curves")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = np.load(Path(a.report) / "per_star.npz", allow_pickle=True)
    e = np.load(Path(a.ls_extra) / "per_star.npz", allow_pickle=True)
    assert np.allclose(d["p_true"], e["p_true"])
    p_true = d["p_true"]
    methods = {"model alone": d["p_model"], "model + fine search": d["p_model_refined"], "model candidates + search": d["p_model_cands"],
               "Lomb-Scargle, 100k trials": d["p_astropy_mb@100000"], "Lomb-Scargle, 200k trials": e["p_astropy_mb@200000"], "Lomb-Scargle, 500k trials": e["p_astropy_mb@500000"]}
    def rate(p, tol, alias):
        return float(np.mean(np.where(np.isfinite(p), rel_error(p, p_true, alias) < tol, False)))
    table = {m: {"strict": [rate(p, t, False) for t in TOLS], "alias": [rate(p, t, True) for t in TOLS]} for m, p in methods.items()}
    json.dump(dict(tolerances=TOLS, n_stars=int(p_true.size), methods=table), open(out / "hit_curves.json", "w"), indent=1)
    lines = ["| method | " + " | ".join(LABEL) + " |", "|---|" + "---|" * len(TOLS)]
    for kind in ("strict", "alias"):
        lines.append(f"| **{'the period itself' if kind == 'strict' else 'double and half accepted'}** |" + " |" * len(TOLS))
        for m in methods:
            lines.append(f"| {m} | " + " | ".join(f"{v:.3f}" for v in table[m][kind]) + " |")
    (out / "hit_curves.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(TOLS))
    for ax, kind, title in zip(axes, ("strict", "alias"), ("Hit rate: the period itself", "Hit rate with the double and the half accepted")):
        style(ax)
        for m in methods:
            c, ls, mk = STY[m]
            ax.plot(x, table[m][kind], color=c, ls=ls, marker=mk, ms=5, lw=1.6, label=m)
        ax.set_xticks(x); ax.set_xticklabels(["within " + l for l in LABEL], fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(title, fontsize=10, loc="left", color=INK); ax.tick_params(labelsize=8)
        ax.set_ylabel("share of the 5,346 validation stars" if kind == "strict" else "")
    axes[1].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "hit_curves.png", dpi=140, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
