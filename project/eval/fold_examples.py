"""Side-by-side folds for the talk: validation stars where the model's
period folds the light curve much more cleanly than the catalogue's. Picks
``--n`` stars from the report's ``model_fine_search.csv`` with the largest
R2 gain, at most two per class and a mix of the two verdicts (another
period; the same period made sharper), and draws each as one row:
catalogue period on the left, model period on the right, points only. One
figure with every row and one PNG per star, at slide-sized text.

    python -m project.eval.fold_examples --report project/results/period_report_spec4 --out project/results/slides/folds
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project.common import data_args_from, load_data, load_encoder
from project.eval.period_report import fold_panel
from project.plot_period_examples import INK, INK2, SURFACE


def pick(rows, n, per_class=2, min_points=400):
    rows = [r for r in rows if int(r["n_points"]) >= min_points and float(r["r2_best"]) >= 0.7]
    rows.sort(key=lambda r: -float(r["gain"]))
    out, seen = [], {}
    want = {"different period": (n + 1) // 2, "catalogue imprecise": n // 2}
    for r in rows:
        c, v = r["class"], r["verdict"]
        if seen.get(c, 0) >= per_class or want.get(v, 0) <= 0:
            continue
        out.append(r); seen[c] = seen.get(c, 0) + 1; want[v] -= 1
        if len(out) == n:
            break
    return out


def draw_row(axes, r, row, fs):
    cat = (float(row["p_catalogue_d"]), "catalogue period", float(row["r2_catalogue"]))
    mod = (float(row["p_best_d"]), "model period" + (f" ({float(row['best_over_catalogue']):.3g} × the catalogue)" if row["verdict"] == "different period" else " (same, sharpened)"), float(row["r2_best"]))
    for ax, (p, title, r2) in zip(axes, (cat, mod)):
        fold_panel(ax, r, p, title, r2, template=False)
        ax.set_title(f"{title}\nP = {p:.6g} d, fold R2 {r2:.2f}", fontsize=fs["title"], color=INK, loc="left")
        ax.tick_params(labelsize=fs["tick"])
        for line in ax.get_lines():
            line.set_markersize(max(line.get_markersize(), 3.5))
    axes[0].set_ylabel(f"{row['class']}\n{row['ztf_id']}\n\nbrightness", fontsize=fs["label"], color=INK)
    for ax in axes:
        ax.set_xlabel("phase (shown twice)", fontsize=fs["label"], color=INK2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default="project/results/period_report_spec4")
    ap.add_argument("--ckpt", default="project/runs/maew_spec/mae.pt", help="for the data arguments")
    ap.add_argument("--out", default="project/results/slides/folds")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fs = dict(title=14, label=13, tick=12)
    rows = pick(list(csv.DictReader(open(Path(a.report) / "model_fine_search.csv"))), a.n)
    _, meta = load_encoder(a.ckpt, "cpu")
    val = load_data(data_args_from(meta.args), splits=("validation",))["validation"]
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 3.6 * len(rows)), dpi=140, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax_row, row in zip(axes, rows):
        r = val[int(row["val_index"])]
        draw_row(ax_row, r, row, fs)
        f2, ax2 = plt.subplots(1, 2, figsize=(12, 4.2), dpi=140)
        f2.patch.set_facecolor(SURFACE)
        draw_row(ax2, r, row, fs)
        f2.tight_layout(); f2.savefig(out / f"fold_{row['class'].replace('/', '-')}_{row['ztf_id']}.png", facecolor=SURFACE); plt.close(f2)
        print(f"{row['class']:10s} {row['ztf_id']}  verdict {row['verdict']:20s} catalogue {float(row['p_catalogue_d']):.5g} d (R2 {float(row['r2_catalogue']):.2f}) -> model {float(row['p_best_d']):.5g} d (R2 {float(row['r2_best']):.2f}), {row['n_points']} points")
    fig.suptitle("Stars where the model's period folds the light curve more cleanly than the catalogue's", fontsize=fs["title"] + 2, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out / "folds_all.png", facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
