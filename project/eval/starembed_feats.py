"""The StarEmbed hand-crafted features on our stars, for a like-for-like
forest baseline: per band, the 34 FATS features (Lomb-Scargle period and its
fit, Fourier harmonic amplitudes and relative phases at three frequencies,
CAR parameters, autocorrelation length, Con, PairSlopeTrend, Psi_CS,
Psi_eta) and the 33 ``light-curve`` extractor features, exactly as
``src/model/handcrafted_features/extract_feats.py`` of
github.com/skai-institute/StarEmbed computes them: raw magnitudes, times and
errors sorted by time, a band with four points or fewer -> -1, a band whose
FATS call fails -> -2. Bands are concatenated band-major.

    python -m project.eval.starembed_feats --out project/results/starembed_feats --workers 30
"""

from __future__ import annotations

import argparse
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

FATS_NAMES = (["PeriodLS", "Period_fit", "Psi_CS", "Psi_eta", "Autocor_length", "Con", "PairSlopeTrend"]
              + [f"Freq{i}_harmonics_amplitude_{j}" for i in (1, 2, 3) for j in range(4)]
              + [f"Freq{i}_harmonics_rel_phase_{j}" for i in (1, 2, 3) for j in range(4)]
              + ["CAR_sigma", "CAR_tau", "CAR_mean"])
_EXT = {}


def _extractors():
    """The FATS feature space and the light-curve extractor, built once per process."""
    if not _EXT:
        warnings.filterwarnings("ignore")
        import FATS
        import light_curve as lc

        _EXT["fats"] = FATS.FeatureSpace(Data=["magnitude", "time", "error"], featureList=FATS_NAMES)
        _EXT["lc"] = lc.Extractor(
            lc.Amplitude(), lc.AndersonDarlingNormal(), lc.BeyondNStd(nstd=1), lc.BeyondNStd(nstd=2), lc.BeyondNStd(nstd=3), lc.Cusum(), lc.Eta(), lc.EtaE(),
            lc.InterPercentileRange(0.25), lc.InterPercentileRange(0.1), lc.Kurtosis(), lc.LinearFit(), lc.LinearTrend(), lc.MagnitudePercentageRatio(),
            lc.MaximumSlope(), lc.Mean(), lc.Median(), lc.MedianAbsoluteDeviation(), lc.MedianBufferRangePercentage(), lc.OtsuSplit(), lc.PercentAmplitude(),
            lc.ReducedChi2(), lc.Skew(), lc.StandardDeviation(), lc.StetsonK(), lc.WeightedMean())
        _EXT["names"] = [f"fats_{n}" for n in FATS_NAMES] + [f"lc_{n}" for n in _EXT["lc"].names]
    return _EXT


def feature_names(bands=("g", "r", "i")) -> list[str]:
    ex = _extractors()
    return [f"{b}_{n}" for b in bands for n in ex["names"]]


def band_features(t, mag, err) -> np.ndarray:
    """One band: the FATS and light-curve features of (time, magnitude,
    error) sorted by time; -1 for four points or fewer, -2 when FATS fails."""
    ex = _extractors()
    n_f, n_l = len(FATS_NAMES), len(ex["lc"].names)
    if t is None or len(t) <= 4:
        return np.full(n_f + n_l, -1.0)
    order = np.argsort(t)
    t, mag, err = np.asarray(t, dtype=np.float64)[order], np.asarray(mag, dtype=np.float64)[order], np.asarray(err, dtype=np.float64)[order]
    try:
        ex["fats"].calculateFeature(np.array([mag, t, err]))
        f = np.asarray(ex["fats"].result(method="array"), dtype=np.float64)
    except Exception:
        f = np.full(n_f, -2.0)
    try:
        l = np.asarray(ex["lc"](t, mag, err, sorted=True, check=False), dtype=np.float64)
    except Exception:
        l = np.full(n_l, -2.0)
    return np.concatenate([f, l])


def record_features(args):
    i, t, y, err, band, n_bands = args
    out = [band_features(t[band == b], y[band == b], err[band == b]) if (band == b).any() else band_features(None, None, None) for b in range(n_bands)]
    return i, np.concatenate(out)


def main(argv=None):
    from project.common import DATA_ROOT, LABEL_FIELD, class_vocabulary, dump_json
    from romae_lc.data import load_pc

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DATA_ROOT)
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--min-points", type=int, default=8, help="as the training loader")
    p.add_argument("--n-objects", type=int, default=0, help="per split, for a quick run")
    args = p.parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = class_vocabulary(args.data)
    names = feature_names()
    for split in args.splits:
        recs = load_pc(args.data, split, label_field=LABEL_FIELD, classes=classes, min_points=args.min_points)  # raw magnitudes, not standardised
        if args.n_objects:
            recs = recs[: args.n_objects]
        jobs = [(i, r.t, r.y, r.err, r.band, 3) for i, r in enumerate(recs)]
        feats = np.zeros((len(recs), len(names)))
        print(f"{split}: {len(recs)} stars, {len(names)} features, {args.workers} workers", flush=True)
        with Pool(args.workers) as pool:
            for k, (i, f) in enumerate(pool.imap_unordered(record_features, jobs, chunksize=8)):
                feats[i] = f
                if (k + 1) % 2000 == 0:
                    print(f"  {split} {k + 1} / {len(recs)}, {time.time() - t0:.0f}s", flush=True)
        ids = np.array([str(r.meta.get("id") or f"{split}:{i}") for i, r in enumerate(recs)])
        np.savez_compressed(out / f"{split}.npz", ids=ids, index=np.arange(len(recs)), features=feats.astype(np.float32), names=np.array(names))
        print(f"wrote {out / f'{split}.npz'} ({time.time() - t0:.0f}s)", flush=True)
    dump_json(dict(names=names, splits=args.splits, seconds=time.time() - t0), out / "info.json")


if __name__ == "__main__":
    main()
