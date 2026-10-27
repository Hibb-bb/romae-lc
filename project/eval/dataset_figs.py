"""Three figures for the paper / slides, written to project/results/figures/:

  ztfxpc_labels.{png,pdf}     the 13 fine classes of ZTFxPC (the three anomaly classes left out), all splits
  ztfxpc_periods.{png,pdf}    the catalogue period distribution of every ZTFxPC star with a period
  classification.{png,pdf}    fine-class random forest on frozen features, 13 classes, test split:
                              "RoPE Transformer + Masked Autoencoder" (the wide encoder's pooled latent)
                              against hand-crafted features; macro F1 (5 seeds) and the F1 of every class

Inputs: project/results/figures/ztfxpc_summary.npz (labels and periods per split; made once from the data)
and project/results/classify_rf_spec5 (results.json, report_mean.csv, report_hand.csv).

    python -m project.eval.dataset_figs
"""
import csv, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "project/results/figures"
ANOMALY = ("RRab-Blazhko", "RRc-Blazhko", "EW/EB-OC")
from project.eval.dataset_figs_style import BLUE, ORANGE, INK, INK2, MUTED, GRID, AXIS, NAME_ENC, NAME_HAND, tidy, save

d = np.load(f"{FIG}/ztfxpc_summary.npz")
labels = np.concatenate([d[f"{s}_label"] for s in ("train", "validation", "test")])
periods = np.concatenate([d[f"{s}_period"] for s in ("train", "validation", "test")])

# ---------------------------------------------------------------- 1. labels
keep = ~np.isin(labels, ANOMALY)
u, c = np.unique(labels[keep], return_counts=True)
order = np.argsort(c)
u, c = u[order], c[order]
fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
bars = ax.barh(u, c, color=BLUE, height=0.72)
for b, v in zip(bars, c):
    ax.text(b.get_width() + c.max() * 0.01, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", ha="left", fontsize=8.5, color=INK2)
tidy(ax, "x")
ax.set_xlim(0, c.max() * 1.13)
ax.set_xlabel("stars")
ax.set_title(f"ZTFxPC: {len(u)} classes, {int(c.sum()):,} stars", pad=10)
ax.text(0, 1.005, "", transform=ax.transAxes)
save(fig, "ztfxpc_labels")

# ---------------------------------------------------------------- 2. periods
p = periods[keep & np.isfinite(periods) & (periods > 0)]  # the same 13 classes as the label figure
lp = np.log10(p)
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
edges = np.linspace(np.floor(lp.min() * 10) / 10, np.ceil(lp.max() * 10) / 10, 71)
ax.hist(lp, bins=edges, color=BLUE, edgecolor="white", linewidth=0.6)
ticks = [-1, 0, 1, 2, 3]
ticks = [t for t in ticks if edges[0] - 0.3 <= t <= edges[-1] + 0.3]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{10 ** t:g}" for t in ticks])
ax.set_xlabel("catalogue period (days, log axis)")
ax.set_ylabel("stars")
tidy(ax, "y")
med = np.median(p)
ax.axvline(np.log10(med), color=INK2, linewidth=1, linestyle=(0, (3, 2)))
ax.text(np.log10(med) + 0.05, ax.get_ylim()[1] * 0.93, f"median {med:.2f} d", color=INK2, fontsize=8.5)
ax.set_title(f"ZTFxPC: catalogue periods of {len(p):,} stars (13 classes)", pad=10)
save(fig, "ztfxpc_periods")

# ---------------------------------------------------------------- 3. classification
res = json.load(open("project/results/classify_rf_spec5/results.json"))
m, h = res["results"]["mean"], res["results"]["hand"]


def report(name):
    rows = list(csv.DictReader(open(f"project/results/classify_rf_spec5/report_{name}.csv")))
    return {r["class"]: (float(r["f1"]), int(r["support"])) for r in rows}


rm, rh = report("mean"), report("hand")
classes = [k for k in rm if rm[k][1] > 0]
classes.sort(key=lambda k: -rm[k][1])
fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150, gridspec_kw=dict(width_ratios=[1, 3.2], wspace=0.28))
# macro F1
vals = [m["f1_mean"], h["f1_mean"]]
errs = [m["f1_std"], h["f1_std"]]
b = a0.bar([0, 1], vals, color=[BLUE, ORANGE], width=0.62, yerr=errs, error_kw=dict(ecolor=INK2, capsize=3, lw=1))
for x, v in zip([0, 1], vals):
    a0.text(x, v + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK)
a0.set_xticks([0, 1])
a0.set_xticklabels(["RoPE Transformer\n+ Masked\nAutoencoder", "Hand-crafted\nfeatures"], fontsize=8.5)
a0.set_ylim(0, 0.75)
a0.set_ylabel("macro F1 (test split, 5 seeds)")
a0.set_title("Macro F1, 13 classes", pad=30)
tidy(a0, "y")
# per class
x = np.arange(len(classes))
w = 0.38
a1.bar(x - w / 2, [rm[k][0] for k in classes], width=w - 0.02, color=BLUE, label=NAME_ENC)
a1.bar(x + w / 2, [rh[k][0] for k in classes], width=w - 0.02, color=ORANGE, label=NAME_HAND)
a1.set_xticks(x)
a1.set_xticklabels([f"{k}\nn={rm[k][1]}" for k in classes], fontsize=7.5, rotation=45, ha="right", rotation_mode="anchor")
a1.set_ylim(0, 1.0)
a1.set_ylabel("F1 per class (test split)")
a1.set_title("F1 of every class, sorted by test support", pad=30)
tidy(a1, "y")
a1.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
fig.suptitle("Fine-class classification with a random forest on frozen features", x=0.01, ha="left", fontsize=11.5, color=INK, y=1.10)
save(fig, "classification")
