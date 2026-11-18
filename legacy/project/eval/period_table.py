"""A period for every star of every split, from the model and from the
catalogue, in one table (step 0 of the phase-coordinate plan).

Read-out: the joint bin head of ``period_probe`` fitted on the training
latents (so training-split periods are in-sample for the read-out; the
validation and test splits are not). Then per star, on the whole light
curve: the catalogue period sharpened by the fine search within
``--cat-rel``; the model's period sharpened within ``--refine-rel``; the
candidate search (top-k bins, double, half; harmonics scaled to equal reach;
a candidate replaces the refined period only when it beats it by
``--cand-margin``); the fold R2 of each. Writes ``periods.csv``,
``per_star.npz`` and ``summary.json``.

    python -m project.eval.period_table --latents latents.pt [--test-latents latents_test.pt] --out DIR
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from project import fold
from project.common import add_device_arg, data_args_from, dump_json, get_device, load_data
from project.eval.ls_benchmark import star_inputs
from project.eval.period_report import alias_kind, h_of, hit_table
from project.period_probe import fit_joint, object_table


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latents", required=True, help="cache with the train and validation splits")
    p.add_argument("--test-latents", default=None, help="cache with the test split (same encoder)")
    p.add_argument("--out", required=True)
    p.add_argument("--bins", type=int, default=240)
    p.add_argument("--mlp-steps", type=int, default=3000)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--k", type=int, default=5, help="candidate bins per star")
    p.add_argument("--refine-rel", type=float, default=0.1)
    p.add_argument("--cand-rel", type=float, default=0.03)
    p.add_argument("--cand-margin", type=float, default=0.02)
    p.add_argument("--cat-rel", type=float, default=0.002)
    p.add_argument("--harmonics", type=int, default=6)
    p.add_argument("--gain", type=float, default=0.1, help="R2 gain that calls the model's period the cleaner one")
    p.add_argument("--n-objects", type=int, default=0, help="per split, for a quick run; 0 = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--n-sim", type=int, default=None)
    add_device_arg(p)
    return p.parse_args(argv)


def search_star(r, p_cat, p_model, tops, args, dev):
    """Every period and fold R2 of one star: ``dict``."""
    t, y, err, band, _ = star_inputs(r)
    H = args.harmonics
    out = dict(n_points=int(t.size), baseline=float(t.max() - t.min()) if t.size else 0.0)
    out["r2_catalogue"] = fold.fold_r2(t, y, err, band, p_cat, H)
    out["p_catalogue_sharp"], out["r2_catalogue_sharp"], _ = fold.refine_period(t, y, err, band, p_cat, args.cat_rel, harmonics=H, device=dev)
    out["r2_model"] = fold.fold_r2(t, y, err, band, p_model, h_of(p_model, p_cat, H))
    pr, r2r, n = fold.refine_period(t, y, err, band, p_model, args.refine_rel, harmonics=H, device=dev)
    out["p_model_refined"], out["r2_model_refined"], trials = pr, r2r, int(n)
    p_ref = pr if np.isfinite(pr) else p_model
    found = [(pr, r2r)] if np.isfinite(r2r) else []
    for s in [v for v in tops if np.isfinite(v) and v > 0] + [2.0 * p_ref, 0.5 * p_ref]:
        if not (np.isfinite(s) and s > 0) or abs(s / p_ref - 1.0) < args.cand_rel:
            continue
        pc, q, n = fold.refine_period(t, y, err, band, float(s), args.cand_rel, harmonics=h_of(float(s), p_ref, H), device=dev)
        trials += int(n)
        if np.isfinite(q):
            found.append((float(pc), float(q)))
    if found:
        ref_p, ref_r2 = found[0]
        better = [f for f in found[1:] if f[1] > ref_r2 + args.cand_margin]
        out["p_model_cands"], out["r2_model_cands"] = max(better, key=lambda f: f[1]) if better else (ref_p, ref_r2)
    else:
        out["p_model_cands"], out["r2_model_cands"] = float("nan"), float("nan")
    out["trials"] = trials
    return out


def run(args):
    t0 = time.time()
    dev = get_device(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.latents, map_location="cpu", weights_only=False)
    meta = cache["meta"]
    ckpt = torch.load(meta["ckpt"], map_location="cpu", weights_only=False)
    over = argparse.Namespace(data=args.data, max_rows=args.max_rows, n_sim=args.n_sim)
    caches = {s: cache for s in cache["splits"]}
    if args.test_latents:
        tc = torch.load(args.test_latents, map_location="cpu", weights_only=False)
        for s in tc["splits"]:
            caches[s] = tc
    splits = [s for s in ("train", "validation", "test") if s in caches]
    data = load_data(data_args_from(ckpt["args"], over), splits=tuple(splits))
    bands = sorted({int(b) for s in splits for r in data[s][:200] for b in np.unique(r.band)})
    tables = {s: object_table(caches[s], s, data[s], bands, int(meta["min_tokens"])) for s in splits}
    print(f"splits {[(s, len(tables[s]['records'])) for s in splits]}; loaded in {time.time() - t0:.0f}s", flush=True)

    # the read-out, fitted on the training latents, applied to every split
    tr = tables["train"]
    x_all = np.concatenate([tables[s]["features"]["mean"] for s in splits])
    pred = fit_joint(tr["features"]["mean"], tr["logp"], x_all, args.bins, args.mlp_steps, args.batch_size, args.mlp_hidden, dev, args.seed)
    p_model_all, p_top_all = 10.0 ** pred, 10.0 ** fit_joint.top[:, : args.k]
    offsets = np.cumsum([0] + [len(tables[s]["records"]) for s in splits])
    print(f"read-out fitted on {len(tr['records'])} training stars in {time.time() - t0:.0f}s", flush=True)

    rows, arrays = [], {}
    for si, s in enumerate(splits):
        tab = tables[s]
        n = len(tab["records"])
        sel = np.arange(n)
        if args.n_objects and n > args.n_objects:
            sel = np.sort(np.random.default_rng(args.seed).choice(n, args.n_objects, replace=False))
        for j in sel:
            r = tab["records"][j]
            g = offsets[si] + j
            res = search_star(r, float(tab["period"][j]), float(p_model_all[g]), p_top_all[g], args, dev)
            ratio = res["p_model_cands"] / tab["period"][j]
            gain, gain_sharp = res["r2_model_cands"] - res["r2_catalogue"], res["r2_model_cands"] - res["r2_catalogue_sharp"]
            if np.isfinite(gain) and gain > args.gain and res["r2_model_cands"] > 0.5:
                verdict = "catalogue imprecise" if (gain_sharp <= args.gain or abs(ratio - 1) < 0.01) else "other period"
            elif np.isfinite(gain_sharp) and gain_sharp < -args.gain:
                verdict = "catalogue cleaner"
            else:
                verdict = "same"
            rows.append(dict(ztf_id=tab["ids"][j], split=s, val_index=int(tab["index"][j]), cls=tab["fine"][j], superclass=tab["superclass"][j],
                             p_catalogue=float(tab["period"][j]), p_model=float(p_model_all[g]), ratio=float(ratio), kind=alias_kind(float(ratio)) if np.isfinite(ratio) else "?",
                             verdict=verdict, **res))
            if len(rows) % 1000 == 0:
                print(f"  {len(rows)} stars, {time.time() - t0:.0f}s", flush=True)

    keys = ["ztf_id", "split", "val_index", "cls", "superclass", "n_points", "baseline", "p_catalogue", "p_catalogue_sharp", "p_model", "p_model_refined", "p_model_cands",
            "ratio", "kind", "verdict", "r2_catalogue", "r2_catalogue_sharp", "r2_model", "r2_model_refined", "r2_model_cands", "trials"]
    with open(out / "periods.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{row[k]:.7g}" if isinstance(row[k], float) else row[k]) for k in keys})
    for k in keys:
        v = [row[k] for row in rows]
        arrays[k] = np.array(v, dtype=object) if isinstance(v[0], str) else np.array(v)
    np.savez_compressed(out / "per_star.npz", **arrays)
    summary = dict(n=len(rows), seconds=time.time() - t0, latents=args.latents, test_latents=args.test_latents, args=dict(vars(args)), splits={})
    for s in splits:
        m = arrays["split"] == s
        if not m.any():
            continue
        pt = arrays["p_catalogue"][m]
        summary["splits"][s] = dict(n=int(m.sum()), in_sample_readout=s == "train",
                                    **{k: hit_table(arrays[f"p_{k}"][m], pt) for k in ("model", "model_refined", "model_cands")},
                                    fold_median={k: float(np.nanmedian(arrays[f"r2_{k}"][m])) for k in ("catalogue", "catalogue_sharp", "model_refined", "model_cands")},
                                    verdicts={v: int(np.sum(arrays["verdict"][m] == v)) for v in np.unique(arrays["verdict"][m])})
        v = summary["splits"][s]
        print(f"{s}: {v['n']} stars; model within 10/1/0.01 % {v['model']['within_0.1']:.3f}/{v['model']['within_0.01']:.3f}/{v['model']['within_0.0001']:.3f}; "
              f"+search {v['model_refined']['within_0.01']:.3f}/{v['model_refined']['within_0.0001']:.3f}; candidates {v['model_cands']['within_0.01']:.3f}/{v['model_cands']['within_0.0001']:.3f}; "
              f"fold medians cat {v['fold_median']['catalogue']:.3f} sharp {v['fold_median']['catalogue_sharp']:.3f} model {v['fold_median']['model_cands']:.3f}; verdicts {v['verdicts']}", flush=True)
    dump_json(summary, out / "summary.json")
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
