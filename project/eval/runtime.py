"""Wall-clock per star of every step of the model's period path, next to
Lomb-Scargle at the budgets of the search benchmark.

The model path, one star at a time (the whole light curve):

* ``cut``: the window grid and the tokens (CPU);
* ``encode``: the encoder forward over the star's valid windows, one batch
  per star (GPU, bf16);
* ``read``: mean pooling and the read-out MLP (the probe's shape);
* ``search``: the fine search within ``--refine-rel`` of the read-out's
  period (GPU);

and, batched, the encoder over many stars' windows at once (``encode
batched``: the survey-scale cost). Against: our GPU Lomb-Scargle at each
budget and the collaborator's astropy search (CPU; single band and
multiband). Every timing is the wall-clock of that step on the record,
after warm-up, with the GPU synchronised.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from project import fold
from project.cache_latents import object_windows
from project.common import add_device_arg, data_args_from, dump_json, get_device, load_data, load_encoder
from project.eval.ls_benchmark import astropy_search, ls_full, run_lomb_scargle, star_inputs
from project.period_probe import Mlp
from project.plot_period_examples import INK, INK2, SURFACE, style


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--predictions", required=True, help="predictions.npz of period_probe (the seeds of the fine search)")
    p.add_argument("--out", required=True)
    p.add_argument("--n-objects", type=int, default=300)
    p.add_argument("--stride", type=float, default=0.25)
    p.add_argument("--bins", type=int, default=240)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--refine-rel", type=float, default=0.1)
    p.add_argument("--ls-budgets", type=int, nargs="*", default=[50000, 200000, 500000])
    p.add_argument("--astropy-budgets", type=int, nargs="*", default=[50000, 200000, 500000], help="none to skip")
    p.add_argument("--astropy-objects", type=int, default=100, help="stars for the astropy timings (CPU, slow)")
    p.add_argument("--batch-windows", type=int, default=512)
    p.add_argument("--p-min", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--n-sim", type=int, default=None)
    add_device_arg(p)
    return p.parse_args(argv)


def run(args):
    t_start = time.time()
    dev = get_device(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    enc, meta = load_encoder(args.ckpt, dev)
    enc.eval()
    cfg, spec = meta.cfg, meta.spec
    over = argparse.Namespace(data=args.data, max_rows=args.max_rows, n_sim=args.n_sim)
    data = load_data(data_args_from(meta.args, over), splits=("train", "validation"))
    p_min = args.p_min or float(min(r.period for r in data["train"] if r.period and r.period > 0))
    pred = np.load(args.predictions, allow_pickle=True)
    rough = {int(i): float(p) for i, p in zip(pred["index"], pred["p_model"])}
    idx = [int(i) for i in pred["index"] if data["validation"][int(i)].period and data["validation"][int(i)].period > 0]
    rng = np.random.default_rng(args.seed)
    if args.n_objects and len(idx) > args.n_objects:
        idx = sorted(rng.choice(idx, args.n_objects, replace=False).tolist())
    n = len(idx)
    if args.astropy_budgets and run_lomb_scargle is None:
        raise SystemExit("--astropy-budgets needs lomb_scargle.py at the repository root and astropy")
    astro_idx = set(idx[: min(n, args.astropy_objects)])
    mlp = Mlp(enc.dim, 2 * args.bins, args.mlp_hidden).to(dev).eval()  # the joint read-out's shape: a logit and an offset per bin
    seed_cap = int(meta.args.get("seed", 0) or 0)
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    steps = ["cut", "encode", "read", "search"] + [f"ls_full@{b}" for b in args.ls_budgets]
    steps += [f"astropy_{tag}@{b}" for b in args.astropy_budgets for tag in ("1b", "mb")]
    times = {s: np.full(n, np.nan) for s in steps}
    n_points, n_windows = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    print(f"{n} validation stars; encoder {enc.kind} dim {enc.dim}; window {cfg.window} d stride {args.stride}; {dev}", flush=True)

    def model_path(i, j=None):
        r = data["validation"][i]
        t0 = time.time()
        _, frames, n_tok = object_windows(r, cfg, args.stride, cfg.max_tokens, seed_cap, i)
        frames = [f for f, k in zip(frames, n_tok) if k >= cfg.min_tokens]
        if not frames:
            return None
        toks = [spec.tokens(frames[lo : lo + args.batch_windows]) for lo in range(0, len(frames), args.batch_windows)]
        t1 = time.time()
        with torch.no_grad(), amp:
            z = torch.cat([enc.encode([tok.to(dev)])[:, 0].float() for tok in toks])
        sync(dev)
        t2 = time.time()
        with torch.no_grad():
            y = mlp(z.mean(0, keepdim=True))
        sync(dev)
        t3 = time.time()
        if j is not None:
            times["cut"][j], times["encode"][j], times["read"][j] = t1 - t0, t2 - t1, t3 - t2
            n_windows[j] = len(frames)
        return z

    # warm-up: kernels, allocator, the data loader's first touch
    for i in idx[:3]:
        model_path(i)
        r = data["validation"][i]
        t, y, err, band, w = star_inputs(r)
        for b in args.ls_budgets:
            ls_full(t, y, w, float(t.max() - t.min()), p_min, b, dev)
        fold.refine_period(t, y, err, band, rough.get(i, float(r.period)), args.refine_rel, device=dev)

    for j, i in enumerate(idx):
        r = data["validation"][i]
        model_path(i, j)
        t, y, err, band, w = star_inputs(r)
        n_points[j] = t.size
        span = float(t.max() - t.min())
        t0 = time.time()
        fold.refine_period(t, y, err, band, rough.get(i, np.nan), args.refine_rel, device=dev)
        sync(dev)
        times["search"][j] = time.time() - t0
        for b in args.ls_budgets:
            t0 = time.time()
            ls_full(t, y, w, span, p_min, b, dev)
            sync(dev)
            times[f"ls_full@{b}"][j] = time.time() - t0
        if i in astro_idx:
            for b in args.astropy_budgets:
                for tag, mb in (("1b", False), ("mb", True)):
                    t0 = time.time()
                    astropy_search(r, b, mb, p_min)
                    times[f"astropy_{tag}@{b}"][j] = time.time() - t0
        if (j + 1) % 50 == 0:
            print(f"  {j + 1} / {n} stars, {time.time() - t_start:.0f}s", flush=True)

    # the encoder batched over many stars: windows of all the stars in batches
    toks, counts = [], []
    for i in idx:
        r = data["validation"][i]
        _, frames, n_tok = object_windows(r, cfg, args.stride, cfg.max_tokens, seed_cap, i)
        frames = [f for f, k in zip(frames, n_tok) if k >= cfg.min_tokens]
        if frames:
            toks.append(frames)
    flat = [f for fr in toks for f in fr]
    t0 = time.time()
    with torch.no_grad(), amp:
        for lo in range(0, len(flat), args.batch_windows):
            tok = spec.tokens(flat[lo : lo + args.batch_windows])
            enc.encode([tok.to(dev)])[:, 0]
    sync(dev)
    batched = (time.time() - t0) / max(len(toks), 1)
    model_total = times["cut"] + times["encode"] + times["read"] + times["search"]
    model_total_batched = batched + np.nanmedian(times["read"] + times["search"])

    def stats(v):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return dict(n=0)
        return dict(n=int(v.size), median=float(np.median(v)), mean=float(np.mean(v)), p16=float(np.percentile(v, 16)), p84=float(np.percentile(v, 84)))

    res = dict(n_stars=n, device=str(dev), gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else None, cpu_threads=torch.get_num_threads(),
               window_days=cfg.window, stride=args.stride, batch_windows=args.batch_windows, points=stats(n_points.astype(float)), windows=stats(n_windows.astype(float)),
               steps={s: stats(times[s]) for s in steps}, model_total=stats(model_total),
               encode_batched_per_star=batched, model_total_batched_per_star=float(model_total_batched))  # fmt: skip
    dump_json(res, out / "results.json")
    np.savez_compressed(out / "per_star.npz", index=np.array(idx), n_points=n_points, n_windows=n_windows, **{f"t_{s}": times[s] for s in steps})

    names = {"cut": "model: windows and tokens (CPU)", "encode": "model: encoder forward, one star per batch (GPU)", "read": "model: pooling + read-out MLP",
             "search": "model: fine search within 10 % (GPU)"}  # fmt: skip
    for b in args.ls_budgets:
        names[f"ls_full@{b}"] = f"our Lomb-Scargle, {b:,} trials (GPU)"
    for b in args.astropy_budgets:
        names[f"astropy_1b@{b}"] = f"astropy, best band, {b:,} trials (CPU)"
        names[f"astropy_mb@{b}"] = f"astropy, multiband, {b:,} trials (CPU)"
    md = [f"# Runtime per star: {n} validation stars, whole light curves\n",
          f"{res['gpu'] or 'no GPU'}, {res['cpu_threads']} CPU threads. Median points per star {res['points']['median']:.0f}, valid windows {res['windows']['median']:.0f}. "
          "Milliseconds per star, median and the 16-84 % range.\n", "| step | stars | median ms | 16 % | 84 % | mean ms |", "|---|---|---|---|---|---|"]
    rows = [(names[s], res["steps"][s]) for s in steps] + [("model: whole path, one star at a time", res["model_total"])]
    for name, v in rows:
        if v["n"]:
            md.append(f"| {name} | {v['n']} | {1e3 * v['median']:.1f} | {1e3 * v['p16']:.1f} | {1e3 * v['p84']:.1f} | {1e3 * v['mean']:.1f} |")
    md.append(f"\nEncoder batched over many stars ({args.batch_windows} windows per batch): {1e3 * batched:.1f} ms per star; whole model path at that rate about {1e3 * model_total_batched:.0f} ms per star.")
    (out / "tables.md").write_text("\n".join(md) + "\n")

    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(rows) + 1.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    labels, med, lo, hi, cols = [], [], [], [], []
    for name, v in rows:
        if not v["n"]:
            continue
        labels.append(name)
        med.append(1e3 * v["median"])
        lo.append(1e3 * (v["median"] - v["p16"]))
        hi.append(1e3 * (v["p84"] - v["median"]))
        cols.append("#2a78d6" if name.startswith("model") else "#eb6834" if "our Lomb" in name else "#b07cd6")
    ypos = np.arange(len(labels))[::-1]
    ax.barh(ypos, med, xerr=[lo, hi], color=cols, height=0.62, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("milliseconds per star (median, 16-84 % range)", fontsize=8, color=INK2)
    ax.set_title(f"Runtime per star, {n} validation stars, whole light curves", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "runtime.png", facecolor=SURFACE)
    plt.close(fig)
    print("\n".join(md[3:]))
    print(f"wrote {out} in {time.time() - t_start:.0f}s")
    return res


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
