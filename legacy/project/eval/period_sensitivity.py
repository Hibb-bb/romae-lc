"""The smallest period change an encoder can tell from noise.

The shuffle score catches a time-blind model; this is its partner: how
FINE a change in the period moves the latent. For every validation star,
the catalogue period and the star's folded template (an error-weighted
Fourier fit, as in ``phase_decoder``) rebuild one 250-day window at the
window's own observation times, with fresh noise drawn from the errors.
The same window is rebuilt with the period shifted by a factor ``1 + e``
for ``e`` on a log grid from 0.01 % to 30 %, both directions. Each version
goes through the frozen encoder.

Scores per star:

- the noise floor: the latent distance between two noise replicates at
  the true period;
- the sensitivity curve: the distance between the shifted and the
  unshifted window, divided by the floor, as a function of ``|e|``;
- the threshold: the smallest ``|e|`` at which the curve exceeds
  ``--margin`` (2 = twice the noise), by interpolation; the same number in
  absolute frequency, ``|e| / P`` cycles per day.

Optionally (``--latents`` + a period read-out is not needed) the bin
probe is skipped: this test is about the encoder alone. Tables: median
threshold per superclass and per period range, relative and absolute;
the median curve per superclass; a figure. Runs on every checkpoint in
minutes (encoder forward passes only).

    python -m project.eval.period_sensitivity --ckpt project/runs/mae_w250/mae.pt --out project/results/sens_mae
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from project.cache_latents import object_windows
from project.common import add_device_arg, data_args_from, dump_json, get_device, load_data, load_encoder, subset, superclass
from project.phase_decoder import fourier_fit, template_at

HEADLINE = ("ECL", "RR", "ROT", "CEP", "DSCT", "LPV")
PERIOD_EDGES = (0.0, 0.3, 1.0, 3.0, 10.0, 100.0, 1e9)


def shifted_window(coefs, t_abs, band, err, period, shift, rng, harmonics, start):
    """The window's brightness rebuilt from the template at the period
    ``period * (1 + shift)`` with fresh noise, the phase ANCHORED at the
    window's start: a shift stretches the cycle inside the window and
    leaves the phase at ``start`` unchanged, so the latent distance
    measures sensitivity to the period, not to a slid phase (anchoring at
    the record's first epoch would slide the window by half a cycle at a
    0.01 % shift for a 0.3 d star 1500 d in)."""
    t_eq = start + (t_abs - start) / (1.0 + shift)  # the time whose phase on `period` equals the shifted one
    y = template_at(coefs, t_eq, band, period, harmonics)
    y = (y + rng.standard_normal(len(y)) * err).astype(np.float32)
    return y


@torch.no_grad()
def star_curve(enc, spec, record, cfg, shifts, rng, device, harmonics, index, n_floor=4):
    """Distances for one star on its densest valid window: ``(floor, curve
    [n_shifts] of distance / floor, n_points)`` or None."""
    starts, frames, n_tokens = object_windows(record, cfg, 1.0, cfg.max_tokens, 0, index)
    n_tokens = np.asarray(n_tokens)
    if (n_tokens >= cfg.min_tokens).sum() == 0:
        return None
    k = int(np.argmax(n_tokens))
    t_rel, _, band, err = (np.asarray(a) for a in frames[k])
    t_abs = starts[k] + t_rel.astype(np.float64)
    p = float(record.period)
    coefs = fourier_fit(record, p, harmonics)
    versions = []
    # n_floor replicates at the true period, then every shift once
    for _ in range(n_floor):
        versions.append((t_rel, shifted_window(coefs, t_abs, band, err, p, 0.0, rng, harmonics, starts[k]), band, err))
    for s in shifts:
        versions.append((t_rel, shifted_window(coefs, t_abs, band, err, p, s, rng, harmonics, starts[k]), band, err))
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    tok = spec.tokens(versions).to(device)
    with amp:
        z = enc.encode([tok])[:, 0].float()
    base = z[:n_floor]
    floor = float(torch.cdist(base, base).sum() / (n_floor * (n_floor - 1)))  # mean pairwise replicate distance
    centre = base.mean(0, keepdim=True)
    d = torch.cdist(z[n_floor:], centre)[:, 0].cpu().numpy()
    return floor, d / max(floor, 1e-8), int(len(t_rel))


def threshold(abs_shifts, ratio, margin):
    """The smallest shift at which ``ratio`` first exceeds ``margin``, by
    log interpolation; inf when it never does."""
    above = np.flatnonzero(ratio >= margin)
    if above.size == 0:
        return float("inf")
    j = int(above[0])
    if j == 0:
        return float(abs_shifts[0])
    x0, x1, y0, y1 = np.log(abs_shifts[j - 1]), np.log(abs_shifts[j]), ratio[j - 1], ratio[j]
    f = (margin - y0) / max(y1 - y0, 1e-9)
    return float(np.exp(x0 + f * (x1 - x0)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-objects", type=int, default=600)
    p.add_argument("--shift-min", type=float, default=1e-4)
    p.add_argument("--shift-max", type=float, default=0.3)
    p.add_argument("--n-shifts", type=int, default=14, help="per direction, log-spaced")
    p.add_argument("--margin", type=float, default=2.0, help="distance / replicate floor that counts as distinguished")
    p.add_argument("--harmonics", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--n-sim", type=int, default=None)
    add_device_arg(p)
    return p.parse_args(argv)


def run(args):
    t0 = time.time()
    dev = get_device(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    enc, meta = load_encoder(args.ckpt, dev)
    cfg, spec = meta.cfg, meta.spec
    over = argparse.Namespace(data=args.data, max_rows=args.max_rows, n_sim=args.n_sim)
    data = load_data(data_args_from(meta.args, over), splits=("validation",))
    recs = [r for r in subset(data["validation"], args.n_objects, args.seed) if r.period and r.period > 0]
    abs_shifts = np.logspace(np.log10(args.shift_min), np.log10(args.shift_max), args.n_shifts)
    shifts = np.concatenate([abs_shifts, -abs_shifts])
    rng = np.random.default_rng(args.seed)
    rows = []
    print(f"{len(recs)} validation stars; shifts {args.shift_min:g} to {args.shift_max:g} ({args.n_shifts} per direction); margin {args.margin}; {dev}", flush=True)
    for i, r in enumerate(recs):
        res = star_curve(enc, spec, r, cfg, shifts, rng, dev, args.harmonics, i)
        if res is None:
            continue
        floor, ratio, n = res
        ratio = 0.5 * (ratio[: args.n_shifts] + ratio[args.n_shifts :])  # both directions averaged
        thr = threshold(abs_shifts, ratio, args.margin)
        rows.append(dict(superclass=superclass(r), period=float(r.period), floor=floor, ratio=ratio, threshold_rel=thr,
                         threshold_freq=thr / float(r.period) if np.isfinite(thr) else float("inf"), n_points=n))  # fmt: skip
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} / {len(recs)} stars, {time.time() - t0:.0f}s", flush=True)
    if not rows:
        raise SystemExit("no star gave a window")
    sup = np.array([x["superclass"] for x in rows])
    per = np.array([x["period"] for x in rows])
    thr = np.array([x["threshold_rel"] for x in rows])
    thr_f = np.array([x["threshold_freq"] for x in rows])
    curves = np.stack([x["ratio"] for x in rows])

    def block(m):
        t = thr[m]
        fin = np.isfinite(t)
        return dict(n=int(m.sum()), median_rel=float(np.median(t[fin])) if fin.any() else float("inf"),
                    median_freq_cpd=float(np.median(thr_f[m][fin])) if fin.any() else float("inf"),
                    share_resolved_1pct=float(np.mean(t < 0.01)), share_resolved_0_1pct=float(np.mean(t < 0.001)),
                    never=float(np.mean(~fin)), curve=np.median(curves[m], 0).tolist())  # fmt: skip

    res = dict(ckpt=str(args.ckpt), n_stars=len(rows), shifts=abs_shifts.tolist(), margin=args.margin,
               all=block(np.ones(len(rows), dtype=bool)),
               by_superclass={g: block(sup == g) for g in HEADLINE if (sup == g).sum() >= 5},
               by_period={f"{lo:g}-{hi:g} d": block((per >= lo) & (per < hi)) for lo, hi in zip(PERIOD_EDGES[:-1], PERIOD_EDGES[1:]) if ((per >= lo) & (per < hi)).sum() >= 5},
               seconds=time.time() - t0)  # fmt: skip
    dump_json(res, out / "results.json")
    np.savez_compressed(out / "per_star.npz", superclass=sup, period=per, threshold_rel=thr, threshold_freq=thr_f, curves=curves, shifts=abs_shifts)
    lines = [f"# Period sensitivity: {args.ckpt}\n",
             f"{len(rows)} validation stars, one window each, rebuilt from the star's template at the true period and at periods shifted by "
             f"a factor 1 + e. Threshold = the smallest |e| at which the latent moves by {args.margin}x the replicate-noise distance. "
             "'resolved' = share of stars whose threshold is below 1 % / 0.1 %; 'never' = the latent never moves that much up to 30 %.\n",
             "| group | n | median threshold (relative) | median threshold (cycles/day) | resolved 1% | resolved 0.1% | never |", "|---|---|---|---|---|---|---|"]
    for name, b in [("all", res["all"])] + list(res["by_superclass"].items()) + list(res["by_period"].items()):
        lines.append(f"| {name} | {b['n']} | {b['median_rel']:.4g} | {b['median_freq_cpd']:.3g} | {b['share_resolved_1pct']:.2f} | {b['share_resolved_0_1pct']:.2f} | {b['never']:.2f} |")
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.4), dpi=150)
        for g, b in res["by_superclass"].items():
            ax.plot(abs_shifts, b["curve"], marker="o", ms=3, lw=1.3, label=f"{g} ({b['n']})")
        ax.axhline(args.margin, color="k", lw=0.8, ls="--", label=f"margin {args.margin:g}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("relative period shift |e|")
        ax.set_ylabel("latent distance / replicate-noise distance")
        ax.grid(True, lw=0.4, alpha=0.5)
        ax.legend(fontsize=8, frameon=False)
        ax.set_title("How fine a period change moves the latent", fontsize=10, loc="left")
        fig.tight_layout()
        fig.savefig(out / "curves.png")
        plt.close(fig)
    except Exception as e:  # the figure is optional
        print(f"figure skipped: {e}")
    a = res["all"]
    print(f"median threshold {a['median_rel']:.4g} relative ({a['median_freq_cpd']:.3g} cycles/day); resolved within 1%: {a['share_resolved_1pct']:.2f}, "
          f"within 0.1%: {a['share_resolved_0_1pct']:.2f}; never: {a['never']:.2f} | "
          + " ".join(f"{g} {b['median_rel']:.3g}" for g, b in res["by_superclass"].items()))
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return res


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
