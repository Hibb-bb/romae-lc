"""The 800k-star Chen ZTF set as training data.

A compact cache per shard (``shard_XX.npz``: the standardised points of
every star, its id, Chen's class string, OUR period and its split), built
once from the arrow shards, then loaded in seconds by every training link.
The period is the one the pretraining's auxiliary loss and templates use,
and it comes from our own table (``project/results/period_chen/periods.csv``):
the model's candidate period where its fold beats the sharpened Chen period
by more than ``GAIN``, otherwise the sharpened Chen period, and none (NaN:
reconstruction only) where neither folds above ``MIN_R2``. Splits are by a
hash of the source id: 2 % validation, 2 % test, the rest training. Chen's
class labels are carried only for the read-only class probe.

    python -m project.chen_data build --root /work/hdd/bfrf/hibb/chen20_ztf_dr24 --out /work/hdd/bfrf/hibb/chen_cache
    python -m project.pretrain_mae --data /work/hdd/bfrf/hibb/chen_cache ...
"""
from __future__ import annotations

import argparse
import csv
import time
import zlib
from pathlib import Path

import numpy as np

from romae_lc import Record

GAIN, MIN_R2 = 0.1, 0.3
DEFAULT_CACHE = "/work/hdd/bfrf/hibb/chen_cache"


def period_rule(p_model, r2_model, p_chen_sharp, r2_chen_sharp) -> float:
    """Our period for a Chen star (NaN = none clean enough)."""
    pm, rm, pc, rc = (float(v) if v not in ("", None) else float("nan") for v in (p_model, r2_model, p_chen_sharp, r2_chen_sharp))
    if np.isfinite(rm) and rm > MIN_R2 and (not np.isfinite(rc) or rm - rc > GAIN) and pm > 0:
        return pm
    if np.isfinite(rc) and rc > MIN_R2 and pc > 0:
        return pc
    return float("nan")


def split_of(sourceid: str) -> int:
    """0 train, 1 validation, 2 test, by a stable hash of the id."""
    h = zlib.crc32(sourceid.encode()) % 50
    return 1 if h == 0 else (2 if h == 1 else 0)


def is_cache(path: str) -> bool:
    return (Path(path) / "shard_00.npz").is_file()


def build_cache(root: str, out: str, periods_csv: str, shards=None, max_rows: int = 0, log=print) -> None:
    from project.eval.period_chen import load_shard

    out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)
    table = {}
    with open(periods_csv) as f:
        for row in csv.DictReader(f):
            table[row["sourceid"]] = period_rule(row["p_model_cands"], row["r2_model_cands"], row["p_chen_sharp"], row["r2_chen_sharp"])
    shards = shards or sorted(p.name for p in Path(root).glob("shard_*") if p.is_dir())
    for shard in shards:
        dst = out_dir / f"{shard}.npz"
        if dst.is_file():
            log(f"{shard}: cached"); continue
        t0 = time.time()
        recs = load_shard(str(Path(root) / shard), min_points=8, max_rows=max_rows)  # standardised per band, like the training set
        n = np.array([r.t.size for r in recs], dtype=np.int64)
        off = np.concatenate([[0], np.cumsum(n)])
        np.savez(dst, t=np.concatenate([r.t for r in recs]).astype(np.float32), y=np.concatenate([r.y for r in recs]).astype(np.float32),
                 err=np.concatenate([r.err for r in recs]).astype(np.float32), band=np.concatenate([r.band for r in recs]).astype(np.int8),
                 offsets=off, ids=np.array([r.meta["id"] for r in recs]), cls=np.array([r.meta.get("class_str") or "" for r in recs]),
                 period=np.array([table.get(r.meta["id"], float("nan")) for r in recs], dtype=np.float64),
                 split=np.array([split_of(r.meta["id"]) for r in recs], dtype=np.int8))
        log(f"{shard}: {len(recs)} stars, {int(n.sum())} points, {time.time() - t0:.0f}s -> {dst}")


def load_chen(cache: str, splits=("train", "validation"), max_rows: int | None = None, min_points: int = 8):
    """``(records per split, class vocabulary)`` from the cache. The records
    are already standardised; ``period`` is our period or None."""
    want = {"train": 0, "validation": 1, "test": 2}
    out = {s: [] for s in splits}
    classes = set()
    for npz in sorted(Path(cache).glob("shard_*.npz")):
        z = np.load(npz, allow_pickle=False)
        t, y, err, band, off = z["t"], z["y"], z["err"], z["band"], z["offsets"]
        ids, cls, per, sp = z["ids"], z["cls"], z["period"], z["split"]
        classes.update(c for c in cls.tolist() if c)
        for i in range(len(ids)):
            s = [k for k, v in want.items() if v == int(sp[i])][0]
            if s not in out or (max_rows and len(out[s]) >= max_rows):
                continue
            lo, hi = int(off[i]), int(off[i + 1])
            if hi - lo < min_points:
                continue
            p = float(per[i]) if np.isfinite(per[i]) else None
            out[s].append(Record(t[lo:hi], y[lo:hi], err[lo:hi], band[lo:hi].astype(np.int64), -1, p,
                                 dict(id=str(ids[i]), class_str=str(cls[i]), superclass_str=None, source=str(npz.parent), period=p)))
        if max_rows and all(len(v) >= max_rows for v in out.values()):
            break
    vocab = tuple(sorted(classes))
    lookup = {c: i for i, c in enumerate(vocab)}
    for recs in out.values():
        for r in recs:
            r.label = lookup.get(r.meta["class_str"], -1)
    return out, vocab


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", default="/work/hdd/bfrf/hibb/chen20_ztf_dr24")
    b.add_argument("--out", default=DEFAULT_CACHE)
    b.add_argument("--periods", default="project/results/period_chen/periods.csv")
    b.add_argument("--shards", nargs="*", default=None)
    b.add_argument("--max-rows", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build_cache(a.root, a.out, a.periods, a.shards, a.max_rows)


if __name__ == "__main__":
    main()
