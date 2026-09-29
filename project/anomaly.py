"""The anomaly test on the frozen pipeline: does the model notice when a
light curve changes?

A fake event is added to a real light curve at a random time ``t*``
(:func:`project.inject.inject`: a phase jump, an amplitude change, a period
change, a bump, a colour offset). The curve is scored twice, clean and with
the event. The model should be more surprised near ``t*`` on the second one.

The pipeline is the frozen one: the stage-1 encoder turns every window of the
star into a latent, and the stage-2 sequence predictor (``train_predictor.py
--arch seq``) reads all the windows so far and says how likely the next one
is. The windows lie on the grid the predictor was trained on (one start every
``stride`` window lengths). Four scores are kept for every window:

- ``nll``: minus the log density the flow gives the window's latent, per
  dimension. The main score.
- ``mse``: the squared error of the flow's mean prediction.
- ``persist``: the squared change from the window before. No model: the
  yardstick in latent space.
- ``hand``: the change of simple numbers of the window (the median and the
  scatter of every band) from the window before. No model and no encoder:
  the yardstick in data space.

The model is only worth its cost where ``nll`` or ``mse`` beat both
yardsticks.

Numbers per kind of event, pooled and per superclass, with ``k*`` the last
window that starts before ``t*`` and ``w`` the number of grid steps in one
window length:

- ``auroc_object``: the largest score of the curve with the event against
  the largest score of the clean curve. 0.5 is chance.
- ``auroc_window``: on the curve with the event, the windows near ``t*``
  (``k* - w + 1`` to ``k* + w``: the ones that hold ``t*`` and the first ones
  after it) against all the others.
- ``auroc_delta``: the same with the score difference, event minus clean.
- ``hit``: the window with the largest difference is near ``t*``;
  ``hit_chance`` is what picking a window blindly would give.

    python -m project.anomaly --pred project/runs/pred_maew_seq2/pred.pt \\
        --out project/results/anomaly_maew --n-objects 300
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from project.cache_latents import object_windows
from project.common import (
    add_device_arg,
    data_args_from,
    dump_json,
    get_device,
    load_data,
    load_encoder,
    subset,
    superclass,
)
from project.inject import KINDS, _auroc, inject
from project.train_predictor import load_predictor

SCORES = ("nll", "mse", "persist", "hand")
MODEL_SCORES = ("nll", "mse")


# ------------------------------------------------------------------- encoding


@torch.no_grad()
def encode_grid(enc, spec, record, cfg, stride, seed, index, device, batch=256):
    """The valid windows of one record on the grid: ``(win [V] grid indices,
    z [V, D] raw latents on the device, feats [V, F] simple numbers of every
    window, n_grid)``."""
    starts, frames, n_tokens = object_windows(record, cfg, stride, cfg.max_tokens, seed, index)
    win = np.flatnonzero(np.asarray(n_tokens) >= cfg.min_tokens)
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    zs = []
    for lo in range(0, len(win), batch):
        tok = spec.tokens([frames[i] for i in win[lo : lo + batch]]).to(device)
        with amp:
            zs.append(enc.encode([tok])[:, 0].float())
    z = torch.cat(zs) if zs else torch.zeros(0, enc.dim, device=device)
    bands = np.unique(record.band)[:2]
    feats = np.full((len(win), 2 * len(bands)), np.nan)
    for j, i in enumerate(win):
        _, y, b = frames[i][:3]
        for c, band in enumerate(bands):
            m = b == band
            if m.sum() >= 3:
                feats[j, 2 * c] = np.median(y[m])
                feats[j, 2 * c + 1] = np.subtract(*np.percentile(y[m], [84, 16])) / 2
    return win, z, feats, len(starts)


# --------------------------------------------------------------------- scores


@torch.no_grad()
def states_long(model, z, gaps):
    """The states of one sequence of any length. The core reads at most
    ``max_len`` windows, so a longer sequence is read in overlapping pieces
    and every position takes its state from the piece where it has the most
    history."""
    v, m = z.shape[0], int(model.max_len)
    if v <= m:
        mask = torch.ones(1, v, dtype=torch.bool, device=z.device)
        return model.states(z[None], gaps[None], mask)[0]
    hop = max(1, m // 2)
    h = None
    for lo in range(0, v - m + hop, hop):
        lo = min(lo, v - m)
        g = gaps[lo : lo + m].clone()
        g[0] = 0.0
        mask = torch.ones(1, m, dtype=torch.bool, device=z.device)
        part = model.states(z[None, lo : lo + m], g[None], mask)[0]
        if h is None:
            h = torch.zeros(v, part.shape[-1], device=z.device, dtype=part.dtype)
            h[:m] = part
        else:
            h[lo + hop : lo + m] = part[hop:]
    return h


def seq_scores(model, win, z, feats, feat_scale, n_grid, stride, args, seed) -> dict:
    """The four scores of one sequence, on the grid: arrays of length
    ``n_grid`` with NaN where a window is not scored (too few points, or fewer
    than ``min_hist`` valid windows before it)."""
    out = {k: np.full(n_grid, np.nan) for k in SCORES}
    v = z.shape[0]
    if v < args.min_hist + 1:
        return out
    dev = z.device
    wt = torch.as_tensor(win, device=dev)
    gaps = torch.zeros(v, device=dev)
    gaps[1:] = (wt[1:] - wt[:-1]).float() * stride
    h = states_long(model, z, gaps)
    t = torch.arange(args.min_hist, v - 1, device=dev)
    if len(t) == 0:
        return out
    gen = torch.Generator(device=dev).manual_seed(seed)  # the same draws for clean and event
    z_t, z_n, g_n, h_t = z[t], z[t + 1], gaps[t + 1], h[t]
    zs_t, zs_n = model.normalize(z_t), model.normalize(z_n)
    pred = model.normalize(model.predict_mean(h_t, g_n, z_t, args.eval_samples, generator=gen))
    target = win[(t + 1).cpu().numpy()]
    out["mse"][target] = (pred - zs_n).square().mean(-1).cpu().numpy()
    out["persist"][target] = (zs_t - zs_n).square().mean(-1).cpu().numpy()
    if model.kind == "flow":
        lp = model.log_prob(z_n, h_t, g_n, z_t, n_steps=args.nll_steps, generator=gen)
        lp = lp + model.sd.log().sum()  # the density of the standardised latent
        out["nll"][target] = (-lp / model.dim).cpu().numpy()
    tn = t.cpu().numpy()
    d = ((feats[tn + 1] - feats[tn]) / feat_scale) ** 2
    seen = np.isfinite(d)  # a band missing from a window gives no number
    out["hand"][target] = np.sqrt(np.where(seen, d, 0.0).sum(1) / np.maximum(seen.sum(1), 1))
    out["hand"][target[seen.sum(1) == 0]] = np.nan
    return out


def feature_scale(feats: np.ndarray) -> np.ndarray:
    """How much every simple number moves from one window to the next on
    the clean curve (median absolute change), the unit of the hand score."""
    if len(feats) < 2:
        return np.ones(feats.shape[1])
    d = np.abs(np.diff(feats, axis=0))
    with np.errstate(all="ignore"):
        s = np.nanmedian(d, axis=0)
    return np.where(np.isfinite(s) & (s > 0), s, 1.0) + 1e-3


# -------------------------------------------------------------------- metrics


def summarize(results: list[dict], key: str, w: int, offsets=None) -> dict:
    """The numbers of the module docstring for one score over a list of
    objects."""
    offsets = range(-2 * w, 3 * w + 1) if offsets is None else offsets
    hits, chance, pos_obj, neg_obj = [], [], [], []
    win_pos, win_neg, d_pos, d_neg = [], [], [], []
    profile = {o: [] for o in offsets}
    for r in results:
        s0, s1, ks = r["clean"][key], r["event"][key], r["k_star"]
        k = np.arange(len(s1))
        ok = np.isfinite(s0) & np.isfinite(s1)
        if not ok.any():
            continue
        lo, hi = ks - (w - 1), ks + w
        anom = (k >= lo) & (k <= hi) & ok
        if not anom.any():
            continue  # the event fell where no window can be scored
        d = s1 - s0
        best = k[ok][np.argmax(d[ok])]
        near = (k >= lo - 1) & (k <= hi + 1) & ok
        hits.append(bool(lo - 1 <= best <= hi + 1))
        chance.append(near.sum() / ok.sum())
        pos_obj.append(np.max(s1[ok]))
        neg_obj.append(np.max(s0[ok]))
        rest = ok & ~anom
        win_pos += s1[anom].tolist()
        win_neg += s1[rest].tolist()
        d_pos += d[anom].tolist()
        d_neg += d[rest].tolist()
        for o in offsets:
            j = ks + o
            if 0 <= j < len(d) and ok[j]:
                profile[o].append(d[j])
    n = len(hits)
    nan = float("nan")

    def au(a, b):
        return _auroc(np.array(a), np.array(b)) if len(a) and len(b) else nan

    return dict(
        n=n,
        hit=float(np.mean(hits)) if n else nan,
        hit_chance=float(np.mean(chance)) if n else nan,
        auroc_object=au(pos_obj, neg_obj),
        auroc_window=au(win_pos, win_neg),
        auroc_delta=au(d_pos, d_neg),
        delta_profile={str(o): (float(np.mean(v)) if v else None) for o, v in profile.items()},
    )


# ------------------------------------------------------------------- outputs


def md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    out = "| " + " | ".join(c for _, c in cols) + " |\n"
    out += "|" + "|".join("---" for _ in cols) + "|\n"
    for r in rows:
        cells = [f"{r.get(k):.3f}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k, _ in cols]
        out += "| " + " | ".join(cells) + " |\n"
    return out


def write_tables(path: Path, res: dict) -> None:
    cols = [("score", "score"), ("n", "n"), ("auroc_object", "AUROC object"),
            ("auroc_window", "AUROC window"), ("auroc_delta", "AUROC difference"),
            ("hit", "hit"), ("hit_chance", "hit by chance")]  # fmt: skip
    lines = ["# Anomaly test\n"]
    lines.append(
        f"Predictor `{res['pred']}`, encoder `{res['ckpt']}`, {res['n_objects']} "
        f"{res['split']} stars, one window start every {res['stride']:g} window lengths. "
        "A fake event is added at a random time. AUROC 0.5 is chance, 1 is perfect. "
        "`nll` and `mse` are the model's scores; `persist` and `hand` need no model and "
        "are the yardsticks.\n"
    )
    for kind, s in res["results"].items():
        lines.append(f"\n## {kind}\n")
        lines.append(md_table([dict(score=k, **s["pooled"][k]) for k in SCORES], cols))
        lines.append("\nAUROC of the score difference per superclass:\n")
        groups = sorted(s["per_superclass"], key=lambda g: -s["per_superclass"][g]["nll"]["n"])
        rows = [
            dict(superclass=g, n=s["per_superclass"][g]["nll"]["n"],
                 **{k: s["per_superclass"][g][k]["auroc_delta"] for k in SCORES})
            for g in groups
        ]  # fmt: skip
        lines.append(md_table(rows, [("superclass", "superclass"), ("n", "n")] + [(k, k) for k in SCORES]))
    with open(path, "w") as f:
        f.write("\n".join(lines))


def plot_profiles(path: Path, res: dict, w: int) -> None:
    """The mean score difference (event minus clean) around ``t*``, one panel
    per kind of event, the model's density score next to the yardsticks.
    Every curve is divided by its own largest value so the shapes compare."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = list(res["results"])
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.2 * len(kinds), 3.0), dpi=140, squeeze=False)
    colours = dict(nll="#0072B2", mse="#009E73", persist="#D55E00", hand="#7F7F7F")
    for ax, kind in zip(axes[0], kinds):
        for k in SCORES:
            prof = res["results"][kind]["pooled"][k]["delta_profile"]
            x = np.array([int(o) for o in prof])
            y = np.array([np.nan if v is None else v for v in prof.values()], dtype=float)
            top = np.nanmax(np.abs(y)) if np.isfinite(y).any() else 1.0
            ax.plot(x / w, y / (top or 1.0), color=colours[k], lw=1.4, label=k)
        ax.axvline(0.0, color="0.5", lw=0.8)
        ax.set_title(kind, fontsize=9)
        ax.set_xlabel("window start minus event time (window lengths)")
    axes[0][0].set_ylabel("score difference (scaled)")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------ main


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True, help="pred.pt of a sequence predictor (--arch seq)")
    p.add_argument("--ckpt", default=None, help="the encoder checkpoint (default: the predictor's)")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--n-objects", type=int, default=300)
    p.add_argument("--kinds", nargs="*", default=list(KINDS), choices=KINDS)
    p.add_argument("--stride", type=float, default=None, help="default: the predictor's grid")
    p.add_argument("--min-hist", type=int, default=3, help="valid windows before a scored one")
    p.add_argument("--nll-steps", type=int, default=50, help="reverse steps of the density")
    p.add_argument("--eval-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="the events and the subset (not the data)")
    g = p.add_argument_group("data (default: the checkpoint's)")
    g.add_argument("--data", default=None)
    g.add_argument("--classes", nargs="*", default=None)
    g.add_argument("--max-rows", type=int, default=None)
    g.add_argument("--n-sim", type=int, default=None)
    add_device_arg(p)
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    dev = get_device(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, pmeta = load_predictor(args.pred, dev)
    if getattr(model, "arch", "mlp") != "seq":
        raise SystemExit(f"{args.pred} is not a sequence predictor (train with --arch seq)")
    lmeta = pmeta["latent_meta"]
    ckpt = args.ckpt or lmeta["ckpt"]
    stride = float(args.stride if args.stride is not None else lmeta["stride"])
    enc, meta = load_encoder(ckpt, dev)
    cfg, spec = meta.cfg, meta.spec
    over = argparse.Namespace(data=args.data, classes=args.classes, max_rows=args.max_rows, n_sim=args.n_sim)
    data = load_data(data_args_from(meta.args, over), splits=(args.split,))
    recs = [r for r in subset(data[args.split], args.n_objects, args.seed) if r.period and r.period > 0]
    w = max(1, int(round(1.0 / stride)))
    print(
        f"predictor {args.pred} ({model.kind}, step {pmeta.get('step')}), encoder {ckpt} "
        f"({enc.kind}, dim {enc.dim}); window {cfg.window:g} d, stride {stride:g} "
        f"({w} grid steps per window); {len(recs)} {args.split} stars; {dev}"
    )

    # the clean curves are scored once, every kind of event reuses them
    clean = []
    for i, r in enumerate(recs):
        win, z, feats, n_grid = encode_grid(enc, spec, r, cfg, stride, args.seed, i, dev)
        scale = feature_scale(feats)
        s = seq_scores(model, win, z, feats, scale, n_grid, stride, args, args.seed + i)
        clean.append(dict(scores=s, scale=scale, n_grid=n_grid, scored=np.isfinite(s["mse"])))
        if (i + 1) % 50 == 0:
            print(f"  clean {i + 1} / {len(recs)} stars, {time.time() - t0:.0f}s", flush=True)

    results, raw = {}, {}
    for kind in args.kinds:
        t1 = time.time()
        rng = np.random.default_rng([args.seed, KINDS.index(kind)])
        rows = []
        for i, r in enumerate(recs):
            c = clean[i]
            scored = np.flatnonzero(c["scored"])
            if len(scored) < 2 * w:
                continue
            starts = r.t.min() + stride * cfg.window * np.arange(c["n_grid"])
            # the event falls where windows are scored on both sides of it
            t_star = float(rng.uniform(starts[scored[0]] + cfg.window, starts[scored[-1]]))
            k_star = int(np.searchsorted(starts, t_star, side="right") - 1)
            event, params = inject(r, kind, t_star, rng)
            win, z, feats, n_grid = encode_grid(enc, spec, event, cfg, stride, args.seed, i, dev)
            s = seq_scores(model, win, z, feats, c["scale"], n_grid, stride, args, args.seed + i)
            rows.append(dict(clean=c["scores"], event=s, k_star=k_star, t_star=t_star,
                             params=params, superclass=superclass(r)))  # fmt: skip
        groups = sorted({x["superclass"] for x in rows})
        pooled = {k: summarize(rows, k, w) for k in SCORES}
        per = {g: {k: summarize([x for x in rows if x["superclass"] == g], k, w) for k in SCORES} for g in groups}
        results[kind] = dict(pooled=pooled, per_superclass=per, seconds=time.time() - t1)
        raw[kind] = rows
        line = "  ".join(
            f"{k} obj {pooled[k]['auroc_object']:.3f} win {pooled[k]['auroc_window']:.3f} "
            f"diff {pooled[k]['auroc_delta']:.3f} hit {pooled[k]['hit']:.2f}"
            for k in SCORES
        )
        print(f"{kind:7s} n={pooled['nll']['n']:4d} (hit by chance {pooled['nll']['hit_chance']:.2f}) | {line} | {time.time() - t1:.0f}s", flush=True)

    res = dict(
        pred=str(args.pred), ckpt=str(ckpt), split=args.split, n_objects=len(recs),
        stride=stride, window=float(cfg.window), grid_steps_per_window=w,
        kinds=list(args.kinds), results=results, args=dict(vars(args)),
        seconds=time.time() - t0,
    )  # fmt: skip
    dump_json(res, out / "summary.json")
    write_tables(out / "tables.md", res)
    plot_profiles(out / "profiles.png", res, w)
    np.savez_compressed(
        out / "objects.npz",
        **{
            f"{kind}/{part}_{k}": np.array([x[part][k] for x in rows], dtype=object)
            for kind, rows in raw.items()
            for part in ("clean", "event")
            for k in SCORES
        },
        **{f"{kind}/k_star": np.array([x["k_star"] for x in rows]) for kind, rows in raw.items()},
        **{f"{kind}/superclass": np.array([x["superclass"] for x in rows]) for kind, rows in raw.items()},
    )
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return res


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
