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
batched``: the survey-scale cost) at every batch size of ``--batch-sweep``,
GPU time only on pre-built batches and end to end with a loader of
``--workers`` tokenising in parallel. Against: our GPU Lomb-Scargle at each
budget and the collaborator's astropy search (CPU; single band and
multiband). Every timing is the wall-clock of that step on the record,
after warm-up, with the GPU synchronised.

``--quant int8wo | int8dq | fp8`` quantises the transformer's linear layers
with torchao (weight-only int8, dynamic int8 activations and weights, or
float8). ``--ref-latents`` (the ``latents.pt`` an unquantised run writes)
gives the accuracy check: the relative change of the window latents and,
for an encoder with the spectral layer, the hit rate of the layer's own
peak period against the catalogue, before and after.
"""

from __future__ import annotations

import argparse
import math
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
from romae_lc.tokenize import Tokens
from project.plot_period_examples import INK, INK2, SURFACE, style


def quantize_encoder(enc, kind: str) -> None:
    """torchao quantisation of every ``nn.Linear`` of the encoder's transformer."""
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig, quantize_

    cfg = {"int8wo": Int8WeightOnlyConfig, "int8dq": Int8DynamicActivationInt8WeightConfig, "fp8": Float8DynamicActivationFloat8WeightConfig}[kind]()
    quantize_(enc.backbone, cfg)


def spectral_peak_period(enc):
    """The period at the spectral layer's own peak for the last encoded
    batch (``[B]`` days), or ``None`` without the layer."""
    model = getattr(enc.backbone, "model", None)
    layer = getattr(model, "spectral", None)
    if layer is None or layer.last_logits is None:
        return None
    return layer.period_of(layer.last_logits.argmax(-1).float()).float().cpu()


class _Frames(torch.utils.data.Dataset):
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return self.frames[i]


def pad_tokens(tok: Tokens, n_tokens: int, batch_multiple: int) -> Tokens:
    """The batch padded to ``n_tokens`` positions and a multiple of
    ``batch_multiple`` rows (zero values and positions, masked), so a
    compiled encoder sees a few fixed shapes instead of one per star."""
    b, n, c = tok.values.shape
    bp = int(math.ceil(b / batch_multiple) * batch_multiple)
    values = tok.values.new_zeros(bp, n_tokens, c)
    positions = tok.positions.new_zeros(bp, tok.positions.shape[1], n_tokens)
    pad = tok.pad_mask.new_ones(bp, n_tokens)
    values[:b, :n], positions[:b, :, :n], pad[:b, :n] = tok.values, tok.positions, tok.pad_mask
    extras = None
    if tok.extras is not None:
        extras = tok.extras.new_zeros(bp, n_tokens)
        extras[:b, :n] = tok.extras
    return Tokens(values, positions, pad, extras)


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
    p.add_argument("--compile", default="none", choices=["none", "default", "max-autotune-no-cudagraphs", "reduce-overhead"], help="torch.compile the encoder's backbone")
    p.add_argument("--dynamic", action="store_true", help="compile with dynamic shapes")
    p.add_argument("--static-pad", action="store_true", help="pad every batch to max_tokens positions and a multiple of --batch-multiple rows")
    p.add_argument("--batch-multiple", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3, help="stars run before timing (more for a compiled encoder)")
    p.add_argument("--batch-sweep", type=int, nargs="*", default=[256, 512, 1024, 2048, 4096, 8192], help="window batch sizes of the batched throughput test")
    p.add_argument("--workers", type=int, default=8, help="loader workers tokenising for the end-to-end batched test")
    p.add_argument("--quant", default="none", choices=["none", "int8wo", "int8dq", "fp8"], help="torchao quantisation of the transformer's linear layers")
    p.add_argument("--ref-latents", default=None, help="latents.pt of an unquantised run of the same stars, for the accuracy check")
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
    if args.quant != "none":
        quantize_encoder(enc, args.quant)
    if args.compile != "none":
        enc.backbone = torch.compile(enc.backbone, mode=args.compile, dynamic=args.dynamic)
    prep = (lambda tok: pad_tokens(tok, cfg.max_tokens, args.batch_multiple)) if args.static_pad else (lambda tok: tok)
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
        toks = [prep(spec.tokens(frames[lo : lo + args.batch_windows])) for lo in range(0, len(frames), args.batch_windows)]
        t1 = time.time()
        with torch.no_grad(), amp:
            z = torch.cat([enc.encode([tok.to(dev)])[:, 0].float()[: tok.pad_mask.shape[0]] for tok in toks])
            z = z[: len(frames)]
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
    for i in idx[: max(3, args.warmup)]:
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
    star_of = np.concatenate([np.full(len(fr), k) for k, fr in enumerate(toks)]) if toks else np.zeros(0, dtype=int)
    n_stars_b = max(len(toks), 1)
    sweep = {}
    z_all, peak_all = [], []
    sizes = args.batch_sweep or [args.batch_windows]
    for bs in sizes:
        t0 = time.time()
        batches = [prep(spec.tokens(flat[lo : lo + bs])) for lo in range(0, len(flat), bs)]
        tokenise = time.time() - t0
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
        keep = bs == sizes[0]  # the latents and the spectral peaks for the accuracy check, from the smallest batch
        try:
            with torch.no_grad(), amp:
                enc.encode([batches[0].to(dev)])  # warm the shape
                sync(dev)
                t0 = time.time()
                for tok in batches:
                    z = enc.encode([tok.to(dev)])[:, 0].float()
                    if keep:
                        z_all.append(z[: tok.pad_mask.shape[0]].cpu())
                        pk = spectral_peak_period(enc)
                        if pk is not None:
                            peak_all.append(pk[: tok.pad_mask.shape[0]])
                sync(dev)
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {bs}: out of GPU memory", flush=True)
            sweep[bs] = dict(windows=len(flat), oom=True)
            torch.cuda.empty_cache()
            continue
        gpu = time.time() - t0
        # end to end: a loader tokenises in parallel while the GPU encodes
        loader = torch.utils.data.DataLoader(_Frames(flat), bs, shuffle=False, num_workers=args.workers, collate_fn=lambda fr: prep(spec.tokens(fr)), prefetch_factor=4 if args.workers else None)
        t0 = time.time()
        with torch.no_grad(), amp:
            for tok in loader:
                enc.encode([tok.to(dev, non_blocking=True)])[:, 0]
            sync(dev)
        e2e = time.time() - t0
        sweep[bs] = dict(windows=len(flat), tokenise_s_per_star=tokenise / n_stars_b, gpu_s_per_star=gpu / n_stars_b, e2e_s_per_star=e2e / n_stars_b,
                         stars_per_s=n_stars_b / e2e, windows_per_s=len(flat) / gpu,
                         peak_mem_gb=(torch.cuda.max_memory_allocated(dev) / 1e9) if dev.type == "cuda" else None)
        print(f"  batch {bs}: tokenise {1e3 * tokenise / n_stars_b:.1f} ms/star, GPU {1e3 * gpu / n_stars_b:.2f} ms/star ({len(flat) / gpu:.0f} windows/s), end to end {1e3 * e2e / n_stars_b:.1f} ms/star ({n_stars_b / e2e:.0f} stars/s)", flush=True)
    best_bs = min((b for b in sweep if "e2e_s_per_star" in sweep[b]), key=lambda b: sweep[b]["e2e_s_per_star"])
    batched = sweep[best_bs]["e2e_s_per_star"]
    model_total = times["cut"] + times["encode"] + times["read"] + times["search"]
    model_total_batched = batched + np.nanmedian(times["read"] + times["search"])
    # the accuracy check: latents and the spectral layer's own period, against a reference run
    acc = {}
    if z_all:
        z_cat = torch.cat(z_all)
        torch.save(dict(index=np.array(idx), star_of=star_of, z=z_cat.half(), peak=torch.cat(peak_all) if peak_all else None, quant=args.quant), out / "latents.pt")
        if peak_all:
            pk = torch.cat(peak_all).numpy()
            p_star = np.array([np.exp(np.median(np.log(pk[star_of == k]))) if (star_of == k).any() else np.nan for k in range(n_stars_b)])
            pt = np.array([float(data["validation"][i].period) for i in idx])[: n_stars_b]
            r = np.abs(p_star / pt - 1)
            acc["spectral_peak_within_10pct"] = float(np.nanmean(r < 0.1))
            acc["spectral_peak_within_1pct"] = float(np.nanmean(r < 0.01))
        if args.ref_latents:
            ref = torch.load(args.ref_latents, map_location="cpu", weights_only=False)
            zr = ref["z"].float()
            if zr.shape == z_cat.shape:
                rel = ((z_cat - zr).norm(dim=1) / zr.norm(dim=1).clamp_min(1e-6)).numpy()
                cos = torch.nn.functional.cosine_similarity(z_cat, zr, dim=1).numpy()
                acc["latent_rel_change_median"] = float(np.median(rel))
                acc["latent_rel_change_p95"] = float(np.percentile(rel, 95))
                acc["latent_cosine_median"] = float(np.median(cos))
                if peak_all and ref.get("peak") is not None:
                    acc["spectral_peak_same_bin_share"] = float(np.mean(np.isclose(torch.cat(peak_all).numpy(), ref["peak"].numpy(), rtol=1e-3)))
            else:
                acc["note"] = f"reference latents have shape {tuple(zr.shape)}, this run {tuple(z_cat.shape)}: different stars or windows"

    def stats(v):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return dict(n=0)
        return dict(n=int(v.size), median=float(np.median(v)), mean=float(np.mean(v)), p16=float(np.percentile(v, 16)), p84=float(np.percentile(v, 84)))

    res = dict(n_stars=n, device=str(dev), gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else None, cpu_threads=torch.get_num_threads(),
               compile=args.compile, dynamic=args.dynamic, static_pad=args.static_pad, batch_multiple=args.batch_multiple,
               window_days=cfg.window, stride=args.stride, batch_windows=args.batch_windows, points=stats(n_points.astype(float)), windows=stats(n_windows.astype(float)),
               steps={s: stats(times[s]) for s in steps}, model_total=stats(model_total),
               encode_batched_per_star=batched, model_total_batched_per_star=float(model_total_batched), batch_sweep=sweep, best_batch=best_bs,
               quant=args.quant, accuracy=acc)  # fmt: skip
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
    md += [f"\n## batched over many stars ({'quantised ' + args.quant if args.quant != 'none' else 'bf16'}{', compiled ' + args.compile if args.compile != 'none' else ''})\n",
           "GPU = encoder time on pre-built batches; end to end = a loader tokenising with " + f"{args.workers} workers while the GPU encodes.\n",
           "| windows per batch | tokenise ms/star (1 thread) | GPU ms/star | windows/s | end to end ms/star | stars/s | peak GPU memory GB |", "|---|---|---|---|---|---|---|"]
    for bs, v in sweep.items():
        if v.get("oom"):
            md.append(f"| {bs} | out of GPU memory | | | | | |")
            continue
        md.append(f"| {bs} | {1e3 * v['tokenise_s_per_star']:.1f} | {1e3 * v['gpu_s_per_star']:.2f} | {v['windows_per_s']:,.0f} | {1e3 * v['e2e_s_per_star']:.1f} | {v['stars_per_s']:,.0f} | {'' if v['peak_mem_gb'] is None else round(v['peak_mem_gb'], 1)} |")
    md.append(f"\nBest batch {best_bs}: {1e3 * batched:.1f} ms per star end to end; the whole model path at that rate about {1e3 * model_total_batched:.0f} ms per star.")
    if acc:
        md.append("\nAccuracy check: " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in acc.items()))
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
