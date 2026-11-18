"""nepa_class.{png,pdf}: macro F1 on the 13 ZTFxPC classes (test split, 3 seeds), star-level probe (mean head)
on frozen latents. Best NEPA run (J3: prediction and SIGReg both weight 1, i.e. lambda = 0.5) against the best
masked autoencoder (maew_spec) and hand-crafted features (0.578, value given by the user).

    python -m project.eval.nepa_class_fig
"""
import json
from project.eval.dataset_figs_style import BLUE, ORANGE, INK, INK2, tidy, save, plt

GREEN = "#1baf7a"
R = "project/results/attn_probe_{}/results.json"
rows = []
for run, name, col in [("enc_j3", "NEPA (J3)\nλ = 0.5", BLUE), ("maew_spec", "Masked\nautoencoder", ORANGE)]:
    m, s = json.load(open(R.format(run)))["class/mean"]["mean_std"]["macro_f1"]
    rows.append((name, m, s, col))
rows.append(("Hand-crafted\nfeatures", 0.578, float("nan"), GREEN))

fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=150)
x = range(len(rows))
ax.bar(x, [r[1] for r in rows], color=[r[3] for r in rows], width=0.62,
       yerr=[r[2] for r in rows], error_kw=dict(ecolor=INK2, capsize=3, lw=1))
for i, r in enumerate(rows):
    ax.text(i, r[1] + (r[2] if r[2] == r[2] else 0) + 0.012, f"{r[1]:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK)
ax.set_xticks(list(x))
ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_ylim(0, 0.7)
ax.set_ylabel("macro F1 (test split, 3 seeds)")
ax.set_title("Classification, 13 classes", pad=12)
tidy(ax, "y")
save(fig, "nepa_class")
