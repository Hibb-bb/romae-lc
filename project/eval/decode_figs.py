"""decode_zscore.{png,pdf}: is NEPA's predicted latent decodable? From state_decode_probe --save-npz.

(a) share of forecasts with |z| < x, z = (decoded - observed) / sigma_obs, against the noise-only ideal
(|z| of a unit Gaussian); (b) share within 2 sigma by gap; (c) example forecasts on held-out stars:
the end of window A, the next 10 days of observations, NEPA's decoded prediction and the Fourier fit.

    python -m project.eval.decode_figs project/results/state_decode/pairs_enc_j3.npz [--decoder mlp]
"""
import argparse
import math

import numpy as np

from project.eval.dataset_figs_style import BLUE, ORANGE, INK, INK2, MUTED, GRID, tidy, save, plt

GREEN = "#1baf7a"
ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--decoder", default="mlp")
ap.add_argument("--examples", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--name", default="decode_zscore")
a = ap.parse_args()
d = np.load(a.npz, allow_pickle=True)
names = list(d["row_names"])
yhat, m, s, dt = d["yhat"], d["m_j"], d["sig_j"], d["dt"]
rows = [  # (label, row name, colour)
    ("NEPA: decoded predicted state", f"predicted state [{a.decoder}]", BLUE),
    ("No change: decoded source state", f"no change [{a.decoder}]", MUTED),
    ("Fourier fit to window A", next(n for n in names if n.startswith("Fourier")), ORANGE),
]
z = {lab: (yhat[names.index(r)] - m) / s for lab, r, _ in rows}
z_ceil = (yhat[names.index(f"target state, saw m_j [{a.decoder}]")] - m) / s  # the target's own state: the ceiling

fig = plt.figure(figsize=(12, 7.6), dpi=150)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.55, wspace=0.3)
# (a) CDF of |z|
ax = fig.add_subplot(gs[0, 0:2])
x = np.linspace(0, 6, 300)
ax.plot(x, [math.erf(v / math.sqrt(2)) for v in x], color=INK, lw=1.2, ls="--", label="ideal: error = noise only")
for lab, _, c in rows:
    az = np.sort(np.abs(z[lab]))
    ax.plot(x, np.searchsorted(az, x) / len(az), color=c, lw=2, label=f"{lab} (median |z| {np.median(az):.1f})")
azc = np.sort(np.abs(z_ceil))
ax.plot(x, np.searchsorted(azc, x) / len(azc), color=BLUE, lw=1.2, ls=":", label=f"ceiling: decoded target's own state (median |z| {np.median(azc):.1f})")
for v in (1, 2):
    ax.axvline(v, color=GRID, lw=0.8, zorder=0)
ax.set_xlim(0, 6)
ax.set_ylim(0, 1)
ax.set_xlabel("|z| = |decoded − observed| / σ_obs")
ax.set_ylabel("share of forecasts below |z|")
ax.set_title(f"(a) Forecast error in units of the observation's σ ({len(m):,} forecasts, held-out stars)", pad=8)
tidy(ax, "y")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
# (b) within 2 sigma by gap
ax = fig.add_subplot(gs[0, 2])
bins = [(0.5, 2), (2, 5), (5, 10)]
w = 0.26
for k, (lab, _, c) in enumerate(rows):
    v = [np.mean(np.abs(z[lab][(dt >= lo) & (dt < hi)]) < 2) for lo, hi in bins]
    ax.bar(np.arange(3) + (k - 1) * w, v, width=w - 0.03, color=c)
ax.axhline(0.954, color=INK, lw=1.2, ls="--")
ax.text(2.45, 0.954, "ideal", ha="right", va="bottom", fontsize=8, color=INK2)
ax.set_xticks(range(3))
ax.set_xticklabels([f"{lo:g}–{hi:g} d" for lo, hi in bins])
ax.set_xlabel("gap from source to target")
ax.set_ylabel("share with |z| < 2")
ax.set_ylim(0, 1)
ax.set_title("(b) Within 2σ, by gap", pad=8)
tidy(ax, "y")
# (c) examples, folded at the star's spectral frequency: window A's observations (seen), the next 10 days
# (not seen), NEPA's decoded forecast from the last token of A over the 10 days, and the Fourier fit
for col, e in enumerate(a.examples[:3]):
    ax = fig.add_subplot(gs[1, col])
    o, f, endA, ti = d[f"ex{e}_info"]
    ph = lambda tt: np.remainder((tt - ti) * f, 1.0)
    At, Am, As = d[f"ex{e}_A_t"], d[f"ex{e}_A_m"], d[f"ex{e}_A_s"]
    ax.errorbar(ph(At), Am, As, fmt="o", ms=2.5, color=MUTED, ecolor=GRID, elinewidth=0.8, label="window A, 250 d (seen)", zorder=1)
    g = d[f"ex{e}_grid_t"]
    k = np.argsort(ph(g))
    ax.plot(ph(g)[k], d[f"ex{e}_four"][k], color=ORANGE, lw=1.6, label="Fourier fit", zorder=2)
    ax.plot(ph(g)[k], d[f"ex{e}_pred_{a.decoder}"][k], color=BLUE, lw=1.8, label="NEPA decoded forecast", zorder=3)
    ax.errorbar(ph(d[f"ex{e}_B_t"]), d[f"ex{e}_B_m"], d[f"ex{e}_B_s"], fmt="o", ms=5, color=INK, mfc="white", ecolor=INK2,
                elinewidth=1, label="next 10 days (not seen)", zorder=4)
    lo, hi = np.percentile(Am, [1, 99])
    pad_ = 0.25 * (hi - lo)
    ax.set_ylim(hi + pad_, lo - pad_)  # brighter up
    ax.set_xlim(0, 1)
    ax.set_xlabel("phase (cycles since the source token)")
    if col == 0:
        ax.set_ylabel("magnitude (scaled per band)")
    ax.set_title(f"({'cde'[col]}) held-out star, period {1 / f:.2f} d", pad=8, fontsize=10)
    tidy(ax, "y")
    if col == 0:
        ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.13), ncol=4)
save(fig, a.name)
