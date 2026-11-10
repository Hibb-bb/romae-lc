"""forecast_analysis.{png,pdf}: why the forecasts are bad. Training curves of the Phase 1 runs and the phase
content of their token latents.

(a) the add-on's prediction error over the state's mean square (1 = no better than predicting zero), against
    "no change" for J3; (b) the add-on's contrast: share of pairs whose nearest token is the true target;
(c) the masked JEPA loss with and without the forecast tail; (d) the spread of the predictions over the spread
of the targets (the collapse watch: low = the predictor outputs nearly the same latent everywhere);
(e) per-token phase probe (project.eval.token_phase_probe): R2 of the catalogue phase from a token's latent,
    against the same probe on the raw input (brightness, error, band) of the token.

    python -m project.eval.forecast_analysis
"""
import json
import os

import numpy as np

from project.eval.dataset_figs_style import BLUE, ORANGE, INK, INK2, MUTED, GRID, tidy, save, plt

AQUA = "#1baf7a"


def curves(name, keys):
    """Per-interval values of logged running means (they reset at every eval and every restart)."""
    rows = [json.loads(l) for l in open(f"project/runs/{name}/log.jsonl")]
    tr = [r for r in rows if r.get("kind") == "train" and all(k in r for k in keys)]
    step = np.array([r["step"] for r in tr])
    t = np.array([r["time"] for r in tr])
    resets = {r["step"] for r in rows if r.get("kind") == "eval"} | {int(step[k]) for k in np.flatnonzero(np.diff(t) > 1800)}
    last = np.array([max([e for e in resets if e < x] + [0]) for x in step])
    out = {}
    for key in keys:
        m = np.array([r[key] for r in tr], float)
        y = np.empty_like(m)
        for k in range(len(m)):
            if k == 0 or step[k - 1] <= last[k]:
                y[k] = m[k]
            else:
                c1, c0 = step[k] - last[k], step[k - 1] - last[k]
                y[k] = (c1 * m[k] - c0 * m[k - 1]) / (c1 - c0)
        out[key] = y
    return step, out


def smooth(y, k=9):
    if len(y) < k:
        return y
    return np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def line(ax, x, y, c, lab, ls="-"):
    ax.plot(x, y, color=c, lw=0.8, alpha=0.25, ls=ls)
    ax.plot(x, smooth(y), color=c, lw=2, ls=ls, label=lab)


fig = plt.figure(figsize=(14, 9), dpi=150)
gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.34)

# (a) add-on prediction error
ax = fig.add_subplot(gs[0, 0])
for name, lab, c in (("enc_p1_j3t", "J3t", BLUE), ("enc_p1_sig0", "SIGReg off", ORANGE), ("enc_p1_nce0", "no contrast", AQUA)):  # J3 predates the logged ratio
    s, v = curves(name, ["dyn_pred_rel", "dyn_stay_rel"])
    line(ax, s, v["dyn_pred_rel"], c, f"{lab}: predicted state")
s, v = curves("enc_p1_sig0", ["dyn_stay_rel"])
line(ax, s, v["dyn_stay_rel"], MUTED, "no change (source state as the guess)", ls="--")
ax.axhline(1, color=INK, lw=1, ls=":")
ax.text(200, 1.03, "predicting zero", fontsize=8, color=INK2, va="bottom")
ax.set_ylim(0, 3.2)
ax.set_title("(a) Add-on: prediction error", pad=8)
ax.axvline(12000, color=GRID, lw=1.2)
ax.text(12200, 1.75, "largest gap\nreaches 30 d", fontsize=8, color=INK2, va="top")
ax.set_xlabel("training step")
ax.set_ylabel("error / mean square of the target state")
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
tidy(ax, "y")

# (b) contrast hit rate
ax = fig.add_subplot(gs[0, 1])
for name, lab, c in (("enc_j3", "J3", BLUE), ("enc_p1_sig0", "SIGReg off", ORANGE), ("enc_p1_nce0", "no contrast (not trained on it)", AQUA)):
    s, v = curves(name, ["dyn_hit"])
    line(ax, s, v["dyn_hit"], c, lab)
ax.set_title("(b) Add-on: target picked out", pad=8)
ax.axvline(12000, color=GRID, lw=1.2)
ax.set_xlabel("training step")
ax.set_ylabel("share of pairs (about 1,000 candidates)")
ax.set_ylim(0, 0.5)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
tidy(ax, "y")

# (c) masked JEPA loss
ax = fig.add_subplot(gs[0, 2])
for name, lab, c in (("enc_p1_noadd", "no tail", BLUE), ("enc_p1_fc", "tail 5-30 d", ORANGE), ("enc_p1_fc75", "tail + 70-80 % hidden", AQUA)):
    s, v = curves(name, ["loss"])
    line(ax, s, v["loss"], c, lab)
ax.set_title("(c) Masked JEPA: training loss", pad=8)
ax.set_xlabel("training step")
ax.set_ylabel("loss")
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
tidy(ax, "y")

# (d) prediction spread over target spread
ax = fig.add_subplot(gs[1, 0])
for name, lab, c in (("enc_p1_noadd", "no tail", BLUE), ("enc_p1_fc", "tail 5-30 d", ORANGE), ("enc_p1_fc75", "tail + 70-80 % hidden", AQUA)):
    s, v = curves(name, ["jepa_pred_std", "jepa_target_std"])
    line(ax, s, v["jepa_pred_std"] / v["jepa_target_std"], c, lab)
ax.axhline(1, color=INK, lw=1, ls=":")
ax.set_ylim(0, 1.1)
ax.set_title("(d) Masked JEPA: spread of predictions", pad=8)
ax.set_xlabel("training step")
ax.set_ylabel("prediction std / target std")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
tidy(ax, "y")

# (e) per-token phase content
ax = fig.add_subplot(gs[1, 1:])
runs = [("enc_j3", "J3"), ("enc_p1_j3t", "J3t"), ("enc_p1_sig0", "SIGReg off"), ("enc_p1_sigbatch", "SIGReg across"),
        ("enc_p1_norot", "rotation off"), ("enc_p1_nce0", "no contrast"), ("enc_p1_mask75", "70-80 % hidden"),
        ("enc_p1_noadd", "no add-on"), ("enc_p1_fc", "forecast tail"), ("enc_p1_fc75", "tail + 70-80 %")]
vals, raw, labs = [], [], []
for name, lab in runs:
    p = f"project/results/token_phase/{name}.json"
    if not os.path.exists(p):
        continue
    d = list(json.load(open(p)).values())[0]
    vals.append(d["h | mlp"])  # [cos, sin]
    raw.append(d["raw | mlp"])
    labs.append(lab)
vals, raw = np.array(vals), np.array(raw)
x = np.arange(len(vals))
w = 0.38
ax.bar(x - w / 2, vals[:, 0], width=w - 0.04, color=BLUE, label="cos of phase (tracks brightness; the token sees its own)")
ax.bar(x + w / 2, vals[:, 1], width=w - 0.04, color=ORANGE, label="sin of phase (brightening or fading: needs context)")
for k in range(len(vals)):
    for o, v in ((-w / 2, vals[k, 0]), (w / 2, vals[k, 1])):
        ax.text(k + o, max(v, 0) + 0.004, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
ax.axhline(raw[:, 0].mean(), color=BLUE, lw=1.2, ls="--")
ax.text(len(vals) - 0.45, raw[:, 0].mean() + 0.003, f"raw input only, cos {raw[:, 0].mean():.2f}", ha="left", fontsize=8, color=INK2)
ax.axhline(raw[:, 1].mean(), color=ORANGE, lw=1.2, ls="--")
ax.text(len(vals) - 0.45, raw[:, 1].mean() + 0.003, f"raw input only, sin {raw[:, 1].mean():.2f}", ha="left", fontsize=8, color=INK2)
ax.set_xlim(-0.6, len(vals) + 1.6)
ax.set_xticks(x)
ax.set_xticklabels(labs, rotation=25, ha="right", fontsize=8.5)
ax.set_ylim(-0.01, 0.25)
ax.set_ylabel("R2 from the token's latent, held-out stars")
ax.set_title("(e) Phase in a token's latent (R2, 1 = all of it)", pad=8)
ax.legend(frameon=False, fontsize=8, loc="upper left")
tidy(ax, "y")
save(fig, "forecast_analysis")
