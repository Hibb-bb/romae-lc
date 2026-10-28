"""Training curves of the dynamic-state add-on of one run (default enc_j3): prediction loss, SIGReg,
contrastive loss and target hit rate, with the largest allowed gap (it ramps) in its own panel.
Writes project/results/figures/dyn_curves_NAME.{png,pdf}.

    python -m project.eval.dyn_curves [NAME]
"""
import sys, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from project.eval.dataset_figs_style import BLUE, INK, INK2, MUTED, tidy, save

name = sys.argv[1] if len(sys.argv) > 1 else "enc_j3"
rows = [json.loads(l) for l in open(f"project/runs/{name}/log.jsonl")]
tr = [r for r in rows if r.get("kind") == "train" and "dyn_pred" in r]
step = np.array([r["step"] for r in tr])
# The log holds the MEAN SINCE THE LAST RESET of every auxiliary term, not the mean of the last
# log interval. Resets: every evaluation (the terms are cleared after it) and every link restart
# (a fresh process; seen as a long pause between log rows). Undo the running mean per interval.
t = np.array([r["time"] for r in tr])
resets = {r["step"] for r in rows if r.get("kind") == "eval"}
dt_row = np.diff(t)
resets |= {int(step[k]) for k in np.flatnonzero(dt_row > 1800)}  # a restart waits for a node; slow steps pause minutes at most
last = np.array([max([e for e in resets if e < x] + [0]) for x in step])


def per_interval(m):
    out = np.empty_like(m)
    for k in range(len(m)):
        if k == 0 or step[k - 1] <= last[k]:
            out[k] = m[k]
        else:
            c1, c0 = step[k] - last[k], step[k - 1] - last[k]
            out[k] = (c1 * m[k] - c0 * m[k - 1]) / (c1 - c0)
    return out


panels = [("dyn_pred", "Prediction loss", "squared error of the predicted state"),
          ("dyn_sig", "SIGReg", "distance of the states from a Gaussian"),
          ("dyn_nce", "Contrastive loss", "cross-entropy over the star's tokens"),
          ("dyn_hit", "Right target picked", "share of pairs"),
          ("dyn_dt_max", "Largest gap allowed", "days")]
if "dyn_pred_rel" in tr[0]:  # runs from 10-27 on: the scale-free prediction error (1 = no better than predicting zero)
    panels.insert(1, ("dyn_pred_rel", "Prediction error relative to predicting zero", "ratio"))
    panels.insert(2, ("dyn_state_ms", "State scale", "mean square of the targets"))
fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 1.95 * len(panels)), dpi=150, sharex=True, gridspec_kw=dict(hspace=0.45))


def smooth(y, k=9):
    if len(y) < k:
        return y
    pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


for ax, (key, title, unit) in zip(axes, panels):
    y = per_interval(np.array([r[key] for r in tr], float))
    if key != "dyn_dt_max":
        ax.plot(step, y, color=BLUE, lw=0.8, alpha=0.3)
        ax.plot(step, smooth(y), color=BLUE, lw=2)
    else:
        ax.plot(step, y, color=MUTED, lw=2)
    ax.set_title(f"{title}  ({unit}, mean of every 100 steps)" if key != "dyn_dt_max" else f"{title}  ({unit})", pad=6, fontsize=10)
    if key == "dyn_sig":
        ax.set_yscale("log")
    tidy(ax, "y")
axes[-1].set_xlabel("training step")
fig.suptitle(f"{name}: dynamic-state add-on, pairs across windows", x=0.01, ha="left", fontsize=11.5, color=INK, y=0.93)
save(fig, f"dyn_curves_{name}")
