"""The collaborator's astropy Lomb-Scargle search on every validation star at
given budgets, in parallel on the CPU (one process per core), so the all-star
hit table can show Lomb-Scargle at more than one budget.

    python -m project.eval.ls_allstars --ckpt mae.pt --predictions predictions.npz --budgets 200000 500000 --out DIR

Writes ``results.json`` (``methods``: ``astropy_mb@B`` with the same hit keys
as ``period_report``) and ``per_star.npz`` (index, p_true, periods).
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from project.common import data_args_from, dump_json, load_data, load_encoder, superclass
from project.eval.ls_benchmark import astropy_search
from project.eval.period_report import HEADLINE, hit_table

_REC = {}


def _init(records, budgets, p_min):
    _REC["records"], _REC["budgets"], _REC["p_min"] = records, budgets, p_min


def _one(i):
    r = _REC["records"][i]
    return i, [astropy_search(r, b, True, _REC["p_min"])[0] for b in _REC["budgets"]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--predictions", required=True, help="predictions.npz: the stars to score (its index)")
    p.add_argument("--budgets", type=int, nargs="+", default=[200000, 500000])
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--n-objects", type=int, default=0)
    p.add_argument("--p-min", type=float, default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--n-sim", type=int, default=None)
    args = p.parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _, meta = load_encoder(args.ckpt, "cpu")
    over = argparse.Namespace(data=args.data, max_rows=args.max_rows, n_sim=args.n_sim)
    data = load_data(data_args_from(meta.args, over), splits=("train", "validation"))
    p_min = args.p_min or float(min(r.period for r in data["train"] if r.period and r.period > 0))
    pred = np.load(args.predictions, allow_pickle=True)
    idx = [int(i) for i in pred["index"] if data["validation"][int(i)].period and data["validation"][int(i)].period > 0]
    if args.n_objects:
        idx = idx[: args.n_objects]
    recs = {i: data["validation"][i] for i in idx}
    p_true = np.array([float(recs[i].period) for i in idx])
    sup = np.array([superclass(recs[i]) for i in idx])
    periods = {b: np.full(len(idx), np.nan) for b in args.budgets}
    pos = {i: j for j, i in enumerate(idx)}
    print(f"{len(idx)} stars, budgets {args.budgets}, {args.workers} workers", flush=True)
    with Pool(args.workers, initializer=_init, initargs=(recs, args.budgets, p_min)) as pool:
        for k, (i, ps) in enumerate(pool.imap_unordered(_one, idx, chunksize=4)):
            for b, v in zip(args.budgets, ps):
                periods[b][pos[i]] = v
            if (k + 1) % 500 == 0:
                print(f"  {k + 1} / {len(idx)} stars, {time.time() - t0:.0f}s", flush=True)
    groups = [g for g in HEADLINE if (sup == g).sum() >= 5]
    res = dict(n_stars=len(idx), p_min=p_min, budgets=args.budgets, seconds=time.time() - t0, methods={})
    for b in args.budgets:
        m = f"astropy_mb@{b}"
        res["methods"][m] = dict(trials_median=float(b), seconds_per_star=float("nan"), **hit_table(periods[b], p_true),
                                 by_superclass={g: hit_table(periods[b][sup == g], p_true[sup == g]) for g in groups})
        v = res["methods"][m]
        print(f"{m}: within 10/1/0.01 % {v['within_0.1']:.3f}/{v['within_0.01']:.3f}/{v['within_0.0001']:.3f}, alias 0.01 % {v['alias_0.0001']:.3f}")
    dump_json(res, out / "results.json")
    np.savez_compressed(out / "per_star.npz", index=np.array(idx), p_true=p_true, superclass=sup, **{f"p_astropy_mb@{b}": periods[b] for b in args.budgets})
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return res


if __name__ == "__main__":
    main()
