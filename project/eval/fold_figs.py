"""Raw light curve beside its phase fold for ZTFxPC validation stars with clean folds:
project/results/figures/fold_<class>.{png,pdf}.

Fold quality = share of the magnitude variance explained by the binned folded curve (50 phase
bins, median per bin), per band, the weaker band counting; only clean points; at least 300 points
in each of g and r. The best star of each class is drawn. The fold uses the catalogue period;
phase 0 is the brightest bin of the r curve, or
the faintest (the primary eclipse) for eclipsing binaries.

    python -m project.eval.fold_figs [--classes RRAB EW/EB LPV]
"""
import argparse, json, numpy as np
from datasets import load_from_disk
from project.eval.dataset_figs_style import BLUE, ORANGE, INK, INK2, tidy, save, plt

ap = argparse.ArgumentParser()
ap.add_argument("--classes", nargs="+", default=["RRAB", "EW/EB", "LPV"])
ap.add_argument("--min-points", type=int, default=300)
a = ap.parse_args()
NAMES = {"RRAB": "RR Lyrae (ab type)", "RRC": "RR Lyrae (c type)", "EW/EB": "contact eclipsing binary", "EA": "detached eclipsing binary",
         "LPV": "long-period variable", "DSCT": "delta Scuti", "T2CEP": "type II Cepheid", "ROT": "rotational variable"}
BANDS = (("g_ZTF", "g band", BLUE), ("r_ZTF", "r band", ORANGE))
NB = 50

d = load_from_disk("/projects/bfrf/data/PC_matches/ZTFxPC/validation")
cls = np.array(d["class_str"])
per = np.array(d["period"], float)


def bands(lc):
    out = {}
    for b, _, _ in BANDS:
        v = lc.get(b)
        if not v or not v["mjd"]:
            return None
        t, m, e = (np.asarray(v[k], float) for k in ("mjd", "mag", "mag_unc"))
        ok = np.asarray(v["clean"], bool) & np.isfinite(m) & np.isfinite(t)
        out[b] = (t[ok], m[ok], e[ok])
    return out


def score(t, m, p):
    ph = np.mod(t / p, 1.0)
    k = np.minimum((ph * NB).astype(int), NB - 1)
    med = np.full(NB, np.nan)
    for i in range(NB):
        if (k == i).sum() >= 2:
            med[i] = np.median(m[k == i])
    fit = med[k]
    good = np.isfinite(fit)
    if good.sum() < 0.8 * len(m):
        return -1.0, med
    return 1 - np.var(m[good] - fit[good]) / np.var(m[good]), med


picked = {}
for c in a.classes:
    best = None
    for i in np.flatnonzero((cls == c) & np.isfinite(per) & (per > 0)):
        bb = bands(d[int(i)]["lightcurve"])
        if bb is None or min(len(bb[b][0]) for b, _, _ in BANDS) < a.min_points:
            continue
        s = min(score(bb[b][0], bb[b][1], per[i])[0] for b, _, _ in BANDS)
        if best is None or s > best[0]:
            best = (s, int(i), bb)
    picked[c] = best
    print(c, "best fold score", round(best[0], 4), "row", best[1], "period", per[best[1]], flush=True)

for c, (s, i, bb) in picked.items():
    p, row = per[i], d[i]
    t_ref = np.floor(min(bb[b][0].min() for b, _, _ in BANDS))
    # phase 0 at the brightest bin of the r band
    _, med = score(bb["r_ZTF"][0], bb["r_ZTF"][1], p)
    phi0 = ((np.nanargmax(med) if c in ("EW/EB", "EA") else np.nanargmin(med)) + 0.5) / NB  # magnitudes: max = faintest
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=150, gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.18))
    for b, lab, col in BANDS:
        t, m, e = bb[b]
        a0.errorbar(t - t_ref, m, yerr=e, fmt="o", ms=2.2, color=col, ecolor=col, elinewidth=0.5, alpha=0.75, label=f"{lab} ({len(t)} points)")
        ph = np.mod(t / p - phi0, 1.0)
        for off in (0, 1):
            a1.errorbar(ph + off, m, yerr=e, fmt="o", ms=2.2, color=col, ecolor=col, elinewidth=0.5, alpha=0.75)
    for ax in (a0, a1):
        ax.invert_yaxis()
        tidy(ax, "y")
    a0.set_xlabel(f"days since MJD {t_ref:.0f}")
    a0.set_ylabel("magnitude")
    a0.set_title("Raw light curve", pad=8)
    a0.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.32, 1.0), ncol=2, markerscale=2, borderaxespad=0.2)
    a1.set_xlim(0, 2)
    a1.set_xticks([0, 0.5, 1, 1.5, 2])
    a1.set_xlabel("phase (two cycles shown)")
    a1.set_title(f"Folded at P = {p:.6g} d", pad=8)
    name = NAMES.get(c, c)
    fig.suptitle(f"{c}: {name}, ZTFxPC validation star {row['sourceid']}", x=0.01, ha="left", fontsize=11.5, color=INK, y=1.04)
    save(fig, "fold_" + c.replace("/", ""))
    plt.close(fig)
json.dump({c: dict(row=v[1], sourceid=d[v[1]]["sourceid"], period=float(per[v[1]]), fold_score=float(v[0])) for c, v in picked.items()},
          open("project/results/figures/fold_stars.json", "w"), indent=1)
