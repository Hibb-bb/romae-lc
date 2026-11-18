"""Model periods for the 800k-star Chen ZTF set (step 5 of docs/PLAN.md):
no labels enter. Shard by shard: the records (same cleaning and per-band
standardisation as the ZTF training data), the encoder's window latents
pooled per star, the joint bin read-out (fitted ONCE on the ZTFxPC training
latents of ``--latents`` and saved in ``--out/readout.pt``), then per star
the fine search within ``--refine-rel`` of the model's period and the
candidate search (top-k bins, double, half; the fold decides). Chen's own
period, when present, is scored only as a weak check (``p_chen``,
``r2_chen``, ``kind``). One CSV per shard, rows appended as they are done,
so a link that stops at its time budget resumes where it left off.

    python -m project.eval.period_chen --root /work/hdd/bfrf/hibb/chen20_ztf_dr24 \
        --latents project/runs/maew_spec/latents100k.pt --out project/results/period_chen [--time-budget 6600]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from project import fold
from project.cache_latents import WindowDataset, make_collate
from project.common import add_device_arg, dump_json, get_device, load_encoder
from project.eval.ls_benchmark import star_inputs
from project.eval.period_report import alias_kind, h_of
from project.period_probe import _joint_predict, _joint_train, pool_latents
from romae_lc import Record
from romae_lc.data import PC_BANDS, _pc_list_column, normalize

KEYS = ["sourceid", "shard", "class_chen", "n_points", "baseline", "p_chen", "p_model", "p_model_refined", "p_model_cands", "ratio", "kind",
        "r2_chen", "r2_chen_sharp", "p_chen_sharp", "r2_model", "r2_model_refined", "r2_model_cands", "trials", "n_windows"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/work/hdd/bfrf/hibb/chen20_ztf_dr24")
    p.add_argument("--shards", nargs="*", default=None, help="shard directory names (default: every shard_* under --root)")
    p.add_argument("--latents", default="project/runs/maew_spec/latents100k.pt", help="ZTFxPC cache: the encoder and the training latents for the read-out")
    p.add_argument("--out", required=True)
    p.add_argument("--bins", type=int, default=240)
    p.add_argument("--mlp-steps", type=int, default=3000)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--mlp-batch", type=int, default=1024)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--refine-rel", type=float, default=0.1)
    p.add_argument("--cand-rel", type=float, default=0.03)
    p.add_argument("--cand-margin", type=float, default=0.02)
    p.add_argument("--cat-rel", type=float, default=0.002)
    p.add_argument("--harmonics", type=int, default=6)
    p.add_argument("--min-points", type=int, default=8)
    p.add_argument("--stride", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=512, help="windows per encoder batch")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-stars", type=int, default=0, help="per shard, for a quick run; 0 = all")
    p.add_argument("--time-budget", type=float, default=0.0, help="seconds; stop cleanly (rows are saved as they go) when exceeded")
    p.add_argument("--seed", type=int, default=0)
    add_device_arg(p)
    return p.parse_args(argv)


def load_shard(path: str, min_points: int = 8, max_rows: int = 0) -> list[Record]:
    """The records of one shard (a ``Dataset`` saved to disk): like
    ``romae_lc.data.load_pc`` but no label filter and rows without a period
    kept; ``meta`` has ``id`` (sourceid), ``class_str`` (Chen's label, never
    used here beyond the CSV) and ``period`` (Chen's, the weak check)."""
    import datasets

    ds = datasets.load_from_disk(path)
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    n = len(ds)
    ids, cls = ds["sourceid"], ds["class_str"]
    periods = np.array([np.nan if p is None else float(p) for p in ds["period"]], dtype=np.float64)
    lc = ds.data.column("lightcurve").combine_chunks()
    per_band = []
    for b in [b for b in PC_BANDS if b in lc.type.names]:
        band = lc.field(b)
        t, off = _pc_list_column(band, "mjd")
        y, _ = _pc_list_column(band, "mag")
        e, _ = _pc_list_column(band, "mag_unc")
        keep = np.isfinite(t) & np.isfinite(y)
        if "clean" in band.type.names:
            c, _ = _pc_list_column(band, "clean", bool, True)
            keep &= c
        e = np.where(np.isfinite(e) & (e > 0), e, np.nan)
        per_band.append((PC_BANDS.index(b), t, y, e, keep, off))
    records = []
    for i in range(n):
        ts, ys, es, bs = [], [], [], []
        for bid, t, y, e, keep, off in per_band:
            lo, hi = off[i], off[i + 1]
            m = keep[lo:hi]
            if not m.any():
                continue
            ts.append(t[lo:hi][m]); ys.append(-y[lo:hi][m]); es.append(e[lo:hi][m]); bs.append(np.full(int(m.sum()), bid, dtype=np.int64))
        if not ts:
            continue
        t = np.concatenate(ts)
        if t.size < min_points:
            continue
        err = np.concatenate(es)
        if np.isnan(err).any():
            fill = np.nanmedian(err) if np.isfinite(err).any() else 1.0
            err = np.where(np.isfinite(err), err, fill)
        t0 = float(t.min())
        order = np.lexsort((t, np.concatenate(bs)))
        meta = dict(id=str(ids[i]), class_str=cls[i], superclass_str=None, t0=t0, source=str(path), period=float(periods[i]) if periods[i] > 0 else None)
        records.append(Record((t - t0).astype(np.float32)[order], np.concatenate(ys).astype(np.float32)[order], err.astype(np.float32)[order],
                              np.concatenate(bs)[order], -1, float(periods[i]) if periods[i] > 0 else None, meta))
    return [normalize(r) for r in records]


def pooled_latents(records, enc, cfg, spec, args, dev) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean latent of every star's valid windows: ``(mean [n, D], n_valid [n], keep [n])``."""
    ds = WindowDataset([("chen", records, list(range(len(records))))], cfg, args.stride, cfg.max_tokens, args.seed, 1, 0.5)
    loader = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=make_collate(spec))
    z = torch.empty(ds.total, enc.dim, dtype=torch.float16)
    n_tok = torch.empty(ds.total, dtype=torch.int64)
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    with torch.no_grad():
        for batch in loader:
            with amp:
                zb = enc.encode([batch["tokens"].to(dev)])[:, 0]
            widx = batch["widx"]
            z[widx] = zb.float().cpu().half()
            n_tok[widx] = batch["n_tokens"]
    off = ds.offsets[0]
    mean, n_valid, keep = np.zeros((len(records), enc.dim), np.float32), np.zeros(len(records), np.int64), np.zeros(len(records), bool)
    zf, nt = z.numpy(), n_tok.numpy()
    for i in range(len(records)):
        lo, hi = int(off[i]), int(off[i + 1])
        ok = np.flatnonzero(nt[lo:hi] >= cfg.min_tokens) + lo
        if ok.size:
            keep[i], n_valid[i] = True, ok.size
            mean[i] = zf[ok].astype(np.float32).mean(0)
    return mean, n_valid, keep


def readout(args, cache, dev, path: Path) -> dict:
    """The joint read-out fitted on the ZTFxPC training latents (or loaded)."""
    if path.is_file():
        return torch.load(path, map_location=dev, weights_only=False)
    pooled = pool_latents(cache, "train", int(cache["meta"]["min_tokens"]))
    period = cache["objects"]["train"]["period"].numpy().astype(np.float64)[pooled["keep"]]
    good = np.isfinite(period) & (period > 0)
    x, y = pooled["mean"][good], np.log10(period[good])
    mu, sd = x.mean(0), x.std(0) + 1e-6
    lo, hi = y.min(), y.max()
    width = (hi - lo) / args.bins
    model = _joint_train((x - mu) / sd, (y - lo) / width - 0.5, args.bins, args.mlp_steps, args.mlp_batch, args.mlp_hidden, dev, args.seed)
    state = dict(model=model, mu=mu, sd=sd, lo=float(lo), width=float(width), n_bins=args.bins, n_train=int(good.sum()))
    torch.save(state, path)
    return state


def predict(ro: dict, x: np.ndarray, dev, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``(p_model [n], p_top [n, k])`` in days."""
    pred = _joint_predict(ro["model"], (x - ro["mu"]) / ro["sd"], ro["n_bins"], dev, k=k)
    p = 10.0 ** (ro["lo"] + (np.asarray(pred) + 0.5) * ro["width"])
    top = 10.0 ** (ro["lo"] + (np.asarray(_joint_predict.top)[:, :k] + 0.5) * ro["width"])
    return p, top


def search_star(r, p_model: float, tops, args, dev) -> dict:
    t, y, err, band, _ = star_inputs(r)
    H = args.harmonics
    out = dict(n_points=int(t.size), baseline=float(t.max() - t.min()) if t.size else 0.0)
    p_chen = r.meta.get("period")
    if p_chen:
        out["r2_chen"] = fold.fold_r2(t, y, err, band, p_chen, H)
        out["p_chen_sharp"], out["r2_chen_sharp"], _ = fold.refine_period(t, y, err, band, p_chen, args.cat_rel, harmonics=H, device=dev)
    else:
        out["r2_chen"] = out["p_chen_sharp"] = out["r2_chen_sharp"] = float("nan")
    out["r2_model"] = fold.fold_r2(t, y, err, band, p_model, H)
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


def fmt(v):
    return f"{v:.7g}" if isinstance(v, float) else ("" if v is None else v)


def run(args):
    t0 = time.time()
    dev = get_device(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.latents, map_location="cpu", weights_only=False)
    enc, meta = load_encoder(cache["meta"]["ckpt"], dev)
    cfg, spec = meta.cfg, meta.spec
    ro = readout(args, cache, dev, out / "readout.pt")
    print(f"encoder {cache['meta']['ckpt']} (step {meta.step}); read-out on {ro['n_train']} ZTFxPC training stars, {ro['n_bins']} bins; {time.time() - t0:.0f}s", flush=True)
    del cache
    shards = args.shards or sorted(p.name for p in Path(args.root).glob("shard_*") if p.is_dir())
    budget_hit = False
    for shard in shards:
        csv_path = out / f"{shard}.csv"
        done = set()
        if csv_path.is_file():
            with open(csv_path) as f:
                done = {row["sourceid"] for row in csv.DictReader(f)}
        if (out / f"{shard}.done").is_file():
            print(f"{shard}: done ({len(done)} stars)", flush=True)
            continue
        ts = time.time()
        records = load_shard(str(Path(args.root) / shard), args.min_points, args.max_stars)
        todo = [r for r in records if r.meta["id"] not in done]
        print(f"{shard}: {len(records)} stars, {len(todo)} to do, loaded in {time.time() - ts:.0f}s", flush=True)
        if todo:
            mean, n_valid, keep = pooled_latents(todo, enc, cfg, spec, args, dev)
            p_model, p_top = predict(ro, mean, dev, args.k)
            print(f"{shard}: latents of {int(keep.sum())} stars in {time.time() - ts:.0f}s", flush=True)
            new = not csv_path.is_file()
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=KEYS)
                if new:
                    w.writeheader()
                for j, r in enumerate(todo):
                    if not keep[j]:
                        res = dict(n_points=int(r.t.size), baseline=float(r.t.max() - r.t.min()), p_model_refined=float("nan"), p_model_cands=float("nan"),
                                   r2_chen=float("nan"), p_chen_sharp=float("nan"), r2_chen_sharp=float("nan"), r2_model=float("nan"), r2_model_refined=float("nan"), r2_model_cands=float("nan"), trials=0)
                        pm = float("nan")
                    else:
                        pm = float(p_model[j])
                        res = search_star(r, pm, p_top[j], args, dev)
                    p_chen = r.meta.get("period")
                    ratio = res["p_model_cands"] / p_chen if p_chen else float("nan")
                    w.writerow({k: fmt(v) for k, v in dict(sourceid=r.meta["id"], shard=shard, class_chen=r.meta.get("class_str") or "", p_chen=p_chen if p_chen else float("nan"),
                                                           p_model=pm, ratio=ratio, kind=alias_kind(float(ratio)) if np.isfinite(ratio) else "", n_windows=int(n_valid[j]), **res).items()})
                    if (j + 1) % 500 == 0:
                        f.flush()
                        print(f"  {shard}: {j + 1} / {len(todo)} stars, {time.time() - t0:.0f}s", flush=True)
                        if args.time_budget and time.time() - t0 > args.time_budget:
                            budget_hit = True
                            break
        if budget_hit:
            print(f"time budget reached inside {shard}; rows saved, resume later", flush=True)
            break
        (out / f"{shard}.done").write_text(f"{len(records)} stars\n")
        print(f"{shard}: finished, {time.time() - t0:.0f}s total", flush=True)
    if not budget_hit:
        merge(out, shards)
    return budget_hit


def merge(out: Path, shards):
    rows = []
    for shard in shards:
        p = out / f"{shard}.csv"
        if p.is_file():
            rows += list(csv.DictReader(open(p)))
    with open(out / "periods.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEYS)
        w.writeheader()
        w.writerows(rows)
    pc = np.array([float(r["p_chen"] or "nan") for r in rows])
    has = np.isfinite(pc) & (pc > 0)
    summary = dict(n=len(rows), with_chen_period=int(has.sum()))
    for key in ("p_model", "p_model_refined", "p_model_cands"):
        pm = np.array([float(r[key] or "nan") for r in rows])
        rel = np.abs(pm[has] / pc[has] - 1)
        summary[key] = {f"within_{t:g}": float(np.mean(rel < t)) for t in (0.1, 0.01, 0.0001)}
    r2 = {k: float(np.nanmedian([float(r[k] or "nan") for r in rows])) for k in ("r2_chen", "r2_chen_sharp", "r2_model_cands")}
    summary["fold_median"] = r2
    dump_json(summary, out / "summary.json")
    print(json.dumps(summary, indent=1))


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
