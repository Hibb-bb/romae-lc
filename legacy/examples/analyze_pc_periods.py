"""Period and cadence census of the PC_matches light-curve datasets.

Reads the catalogue ``period`` of every object (all splits of every
sub-dataset), the sampling of every survey (gaps, spans, points per band) and,
on a random subsample of curves per survey, the Lomb-Scargle spectral support
(shortest timescale carrying signal, harmonics included) with
:mod:`romae_lc.analysis`. Writes ``summary.json``, ``arrays.npz`` and a
figure into ``--out``. Run it as a batch job (``jobs/analyze_pc_periods.sh``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from romae_lc.analysis import _gaps, _standardize, periodogram, spectral_support

ROOT = Path("/projects/bfrf/data/PC_matches")
SPLITS = ("train", "validation", "test")
TIME_KEYS = ("mjd", "hjd", "time", "jd")
VALUE_KEYS = ("mag", "flux")
ERR_KEYS = ("mag_unc", "mag_err", "flux_unc", "flux_err", "flux_error")
QS = (0.001, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.999)


def q(a, qs=QS) -> dict:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
    d = {f"q{int(x * 1000):04d}": float(v) for x, v in zip(qs, np.quantile(a, qs))}
    d.update(n=int(a.size), min=float(a.min()), max=float(a.max()))
    return d


def lc_datasets() -> list[str]:
    out = []
    for d in sorted(ROOT.iterdir()):
        info = d / "train" / "dataset_info.json"
        if info.exists() and "lightcurve" in json.load(open(info))["features"]:
            out.append(d.name)
    return out


def pick(d: dict, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return k
    return None


def flatten(lc: dict):
    """One row's ``lightcurve`` dict -> (t, y, err, band, band_names)."""
    ts, ys, es, bs, names = [], [], [], [], []
    for b, (name, arr) in enumerate(lc.items()):
        tk, vk, ek = pick(arr, TIME_KEYS), pick(arr, VALUE_KEYS), pick(arr, ERR_KEYS)
        if tk is None or vk is None:
            continue
        t = np.asarray(arr[tk], dtype=np.float64)
        y = np.asarray(arr[vk], dtype=np.float64)
        e = np.asarray(arr[ek], dtype=np.float64) if ek else np.ones_like(y)
        keep = np.isfinite(t) & np.isfinite(y)
        if "clean" in arr and arr["clean"] is not None and len(arr["clean"]) == len(t):
            c = np.asarray(arr["clean"], dtype=bool)
            if c.sum() >= 0.1 * keep.sum():
                keep &= c
        if "quality_flag" in arr and arr["quality_flag"] is not None:
            qf = np.asarray(arr["quality_flag"])
            if len(qf) == len(t) and (qf == 0).sum() >= 0.1 * keep.sum():
                keep &= qf == 0
        e = np.where(np.isfinite(e) & (e > 0), e, np.nan)
        if not np.isfinite(e).any():
            e = np.ones_like(y)
        else:
            e = np.where(np.isfinite(e), e, np.nanmedian(e))
        ts.append(t[keep]), ys.append(y[keep]), es.append(e[keep])
        bs.append(np.full(keep.sum(), b))
        names.append(name)
    if not ts:
        return None
    return (
        np.concatenate(ts),
        np.concatenate(ys),
        np.concatenate(es),
        np.concatenate(bs),
        names,
    )


def analyse_curve(args):
    """Periodogram support of one curve; returns a dict of scalars."""
    t, y, e, band, period, max_freq, max_evals = args
    out = dict(n=int(t.size), span=float(np.ptp(t)) if t.size else 0.0)
    g = _gaps(t)
    out.update(
        dt_min=float(g.min()) if g.size else np.nan,
        dt_q05=float(np.quantile(g, 0.05)) if g.size else np.nan,
        dt_median=float(np.median(g)) if g.size else np.nan,
        period=float(period) if period is not None else np.nan,
    )
    out.update(detected=False, f_lo=np.nan, f_hi=np.nan, f_peak=np.nan)
    if np.unique(t).size < 3:
        return out
    y, e = _standardize(y, e, band)
    fmax = min(0.5 / np.quantile(g, 0.05), max_freq)
    fmin = 0.5 / out["span"]
    if not fmin < fmax:
        return out
    freqs, power = periodogram(
        t, y, e, min_freq=fmin, max_freq=fmax, max_evals=max_evals
    )
    s = spectral_support(freqs, power, rel=0.1)
    out.update(detected=bool(s.detected), f_lo=s.f_lo, f_hi=s.f_hi, f_peak=s.f_peak)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/pc_period")
    ap.add_argument("--per-survey", type=int, default=512, help="curves sampled")
    ap.add_argument("--max-freq", type=float, default=100.0, help="cycles / day")
    ap.add_argument("--max-evals", type=float, default=2e7)
    ap.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    import datasets as hfd

    hfd.disable_progress_bars()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary, arrays = {}, {}

    # 1. catalogue periods of every object (PC_metadata holds them all)
    t0 = time.time()
    periods, classes, superclasses, splits = [], [], [], []
    for split in SPLITS:
        ds = hfd.load_from_disk(str(ROOT / "PC_metadata" / split))
        d = ds.select_columns(["period", "class_str", "superclass_str"]).to_dict()
        periods += d["period"]
        classes += d["class_str"]
        superclasses += d["superclass_str"]
        splits += [split] * len(d["period"])
    P = np.array([np.nan if p is None else p for p in periods], dtype=np.float64)
    cls = np.array([str(c) for c in classes], dtype=object)
    sup = np.array([str(c) for c in superclasses], dtype=object)
    ok = np.isfinite(P) & (P > 0)
    arrays["period"] = P
    arrays["class_str"] = cls.astype(str)
    arrays["superclass_str"] = sup.astype(str)
    arrays["split"] = np.array(splits)
    cat = dict(n_objects=int(P.size), n_with_period=int(ok.sum()), all=q(P[ok]))
    cat["by_superclass"] = {s: q(P[ok & (sup == s)]) for s in sorted(set(sup[ok]))}
    cat["by_class"] = {c: q(P[ok & (cls == c)]) for c in sorted(set(cls[ok]))}
    lo = np.floor(np.log10(P[ok].min()) * 100) / 100
    edges = np.arange(lo, np.log10(P[ok].max()) + 0.02, 0.01)
    arrays["logP_hist"], arrays["logP_hist_edges"] = np.histogram(
        np.log10(P[ok]), edges
    )
    summary["catalogue"] = cat
    print(
        f"catalogue: {P.size} objects, {ok.sum()} with period ({time.time()-t0:.0f}s)",
        flush=True,
    )
    print(json.dumps(cat["all"], indent=1), flush=True)

    # 2. per survey: sampling + periodogram support on a subsample
    names = args.datasets or lc_datasets()
    with ProcessPoolExecutor(args.workers) as pool:
        for name in names:
            t0 = time.time()
            ds = hfd.load_from_disk(str(ROOT / name / "train"))
            n = len(ds)
            idx = np.sort(rng.choice(n, min(args.per_survey, n), replace=False))
            rows = ds.select(idx).select_columns(["lightcurve", "period", "class_str"])
            jobs, band_names, per_band_n = [], None, {}
            for row in rows:
                fl = flatten(row["lightcurve"])
                if fl is None:
                    continue
                t, y, e, b, bn = fl
                band_names = bn
                for i, nm in enumerate(bn):
                    per_band_n.setdefault(nm, []).append(int((b == i).sum()))
                jobs.append((t, y, e, b, row["period"], args.max_freq, args.max_evals))
            res = list(pool.map(analyse_curve, jobs, chunksize=4))
            R = {k: np.array([r[k] for r in res], dtype=float) for k in res[0]}
            det = R["detected"] > 0
            short, long, peak = (1 / R[k][det] for k in ("f_hi", "f_lo", "f_peak"))
            harm = R["period"][det] / short  # highest harmonic order seen
            entry = dict(
                n_objects=n,
                n_sampled=len(res),
                n_detected=int(det.sum()),
                bands=band_names,
                points_per_band={k: q(v) for k, v in per_band_n.items()},
                n_points=q(R["n"]),
                span=q(R["span"]),
                dt_min=q(R["dt_min"]),
                dt_q05=q(R["dt_q05"]),
                dt_median=q(R["dt_median"]),
                catalogue_period=q(R["period"]),
                shortest_timescale=q(short),
                longest_timescale=q(long),
                peak_period=q(peak),
                harmonic_order=q(harm),
                peak_over_catalogue=q(peak / R["period"][det]),
            )
            summary[name] = entry
            for k, v in R.items():
                arrays[f"{name}/{k}"] = v
            print(
                f"{name}: {n} objects, {len(res)} sampled, {det.sum()} detected; "
                f"dt_q05 med {entry['dt_q05'].get('q0500', np.nan):.4g} d, span med "
                f"{entry['span'].get('q0500', np.nan):.4g} d, shortest timescale "
                f"q02 {entry['shortest_timescale'].get('q0020', np.nan):.4g} d, "
                f"harmonic order med {entry['harmonic_order'].get('q0500', np.nan):.3g} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )
            json.dump(summary, open(out / "summary.json", "w"), indent=1)
            np.savez_compressed(out / "arrays.npz", **arrays)

    # 3. figure
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("done (no matplotlib)", flush=True)
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    ax = axes[0]
    bins = np.geomspace(P[ok].min(), P[ok].max(), 200)
    ax.hist(P[ok], bins, color="0.3", label=f"all ({ok.sum()})")
    for s in sorted(set(sup[ok])):
        m = ok & (sup == s)
        if m.sum() > 200:
            ax.hist(P[m], bins, histtype="step", label=f"{s} ({m.sum()})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("catalogue period [d]")
    ax.set_ylabel("objects")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.3)
    ax = axes[1]
    bins = np.geomspace(1e-4, 1e5, 180)
    for name in names:
        if f"{name}/f_hi" not in arrays:
            continue
        d = arrays[f"{name}/detected"] > 0
        ax.hist(1 / arrays[f"{name}/f_hi"][d], bins, histtype="step", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("shortest timescale with signal (1 / f_hi) [d]")
    ax.set_ylabel("curves")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "periods.png", dpi=150)
    print("done", flush=True)


if __name__ == "__main__":
    main()
