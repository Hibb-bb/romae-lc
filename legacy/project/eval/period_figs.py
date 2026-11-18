"""Period hit rates of the wide encoder against Lomb-Scargle, on the same ZTFxPC validation stars:
project/results/figures/period_hits.{png,pdf}.

Encoder = the spectral layer's peak (local centroid per window, median over a star's windows with
>= 32 tokens): the raw estimate, no search after it. Lomb-Scargle = astropy multiband, 200k trial
frequencies (project/results/ls_allstars_spec). A star the encoder gives no estimate for counts as a
miss. Left: strict (within the tolerance of the catalogue period). Right: the double and the half
also accepted (Lomb-Scargle finds half the period of most eclipsing binaries).

    python -m project.eval.period_figs
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from project.eval.dataset_figs_style import BLUE, ORANGE, INK, INK2, tidy, save, NAME_ENC

PEAK = "project/results/figures/peak_maew_spec_val.npz"
LS = "project/results/ls_allstars_spec/per_star.npz"
TOL = np.array([0.2, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4])

e, l = np.load(PEAK), np.load(LS, allow_pickle=True)
key = [k for k in e.files if k.startswith("p_hat:")][0]
pe = dict(zip(e["index"].tolist(), e[key]))
idx = [int(i) for i in l["index"] if int(i) in pe]
p_true = np.array([dict(zip(l["index"].tolist(), l["p_true"]))[i] for i in idx])
p_enc = np.array([pe[i] for i in idx])
p_ls = np.array([dict(zip(l["index"].tolist(), l["p_astropy_mb@200000"]))[i] for i in idx])


def rates(p_hat, alias):
    r = p_hat / p_true
    rel = np.abs(r - 1)
    if alias:
        rel = np.minimum(rel, np.minimum(np.abs(r / 2 - 1), np.abs(2 * r - 1)))
    rel = np.where(np.isfinite(rel), rel, np.inf)  # no estimate = a miss
    return np.array([np.mean(rel < t) for t in TOL])


out = {"n_stars": len(idx), "n_enc_missing": int((~np.isfinite(p_enc)).sum()), "tolerances": TOL.tolist()}
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=150, sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, alias, title in zip(axes, (False, True), ("Strict: the catalogue period", "Double and half also accepted")):
    for p, c, name in ((p_enc, BLUE, NAME_ENC), (p_ls, ORANGE, "Lomb-Scargle (200k trial frequencies)")):
        h = rates(p, alias)
        out[f"{name} | {'alias' if alias else 'strict'}"] = h.tolist()
        ax.plot(TOL * 100, h, color=c, lw=2, marker="o", ms=5, label=name)
        ax.text(TOL[-1] * 100 * 0.8, h[-1], f"{h[-1]:.2f}", color=INK2, fontsize=8.5, ha="left", va="center")
    ax.set_xscale("log")
    ax.set_xlim(TOL[0] * 100 * 1.4, TOL[-1] * 100 * 0.45)  # reversed: tighter to the right; room for the end labels
    ax.minorticks_off()
    ax.set_xticks(TOL * 100)
    ax.set_xticklabels([f"{t * 100:g}" for t in TOL], fontsize=8.5)
    ax.set_xlabel("tolerance (% of the period)")
    ax.set_ylim(0, 1.0)
    ax.set_title(title, pad=10)
    tidy(ax, "y")
axes[0].set_ylabel("share of stars within the tolerance")
axes[0].legend(frameon=False, loc="lower left", fontsize=9)
fig.suptitle(f"Period recovery on {len(idx):,} ZTFxPC validation stars (no search after the encoder's estimate)",
             x=0.01, ha="left", fontsize=11.5, color=INK, y=1.03)
save(fig, "period_hits")
json.dump(out, open("project/results/figures/period_hits.json", "w"), indent=1)
print(json.dumps(out, indent=1))
