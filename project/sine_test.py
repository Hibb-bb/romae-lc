"""The sine test: can the masked autoencoder's decoder recover a clean
sinusoid far from the seen points, and down to which period?

Every sample is one window of ``--n-points`` times (uniform random or a
regular grid over ``--window`` days), a sinusoid of log-uniform period in
``--p-range`` with a random phase (``--shape`` adds a second harmonic), no
noise unless ``--noise``. The same encoder (with the spectral layer), the
same block masking and the same decoder as the real runs, at a small size.
Variants: ``plain`` (the decoder as it is), ``phasor`` (sine/cosine of time
at the spectral layer's own frequency estimate on every decoder token,
``--phasor`` harmonics), ``oracle`` (the same features at the TRUE frequency:
the ceiling of what the decoder can do with the right coordinate).

The report (``results.json``, ``summary.png``) answers two questions on
fresh samples: (1) explained variance of the hidden points by the distance
to the nearest seen point in cycles = how far the decoder can predict;
(2) by the period, for near points (within 2 cycles) and far points, with
the spectral layer's frequency error = the shortest period the model
recovers.

    python -m project.sine_test --variant plain --out project/results/sine_test/plain
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from project.mae_data import make_mask
from romae_lc import RoMAEForPreTraining
from romae_lc.spectral import spectral_aux_loss


def layout(head_dim: int, p_lo: float, p_hi: float, band_dim: int = 8) -> list[dict]:
    """A log ladder of rotary timescales covering periods ``p_lo .. p_hi``
    days (positions are days), the band axis on ``band_dim`` channels."""
    n = (head_dim - band_dim) // 2
    ts = np.exp(np.linspace(math.log(p_lo / (2 * math.pi)), math.log(p_hi / (2 * math.pi)), n)).tolist()
    return [{"kind": "axial", "axes": [0], "dim": head_dim - band_dim, "timescales": ts}, {"kind": "axial", "axes": [1], "dim": band_dim}]


def sample(b: int, n: int, window: float, p_range, sampling: str, shape: float, noise: float, generator: torch.Generator):
    """``(values [B, N, 2], positions [B, 2, N], period [B], t [B, N], y [B, N])``."""
    lo, hi = math.log(p_range[0]), math.log(p_range[1])
    period = torch.exp(lo + (hi - lo) * torch.rand(b, generator=generator))
    phase = 2 * math.pi * torch.rand(b, generator=generator)
    if sampling == "regular":
        t = torch.linspace(0, window, n)[None].expand(b, n).clone()
    else:
        t = torch.sort(torch.rand(b, n, generator=generator) * window, 1).values
    arg = 2 * math.pi * t / period[:, None] + phase[:, None]
    y = torch.sin(arg) + shape * torch.sin(2 * arg)
    if noise > 0:
        y = y + noise * torch.randn(b, n, generator=generator)
    values = torch.stack([y, torch.zeros_like(y)], -1)
    positions = torch.stack([t + 1.0, torch.full_like(t, 0.6)], 1)
    return values, positions, period, t, y


def build(a) -> RoMAEForPreTraining:
    enc = dict(d_model=a.d_model, nhead=a.heads, depth=a.depth)
    dec = dict(d_model=a.dec_width, nhead=a.heads, depth=a.dec_depth)
    head_dim = a.d_model // a.heads
    spectral = dict(time_scale=1.0, p_min=a.p_range[0] / 2, p_max=a.p_range[1] * 2, rel=a.spectral_rel, channels=4, reader=16, depth=2, after_layer=min(2, a.depth))
    return RoMAEForPreTraining(decoder=dec, mask_ratio=a.mask_ratio, target_channels=1, encoder=enc, n_channels=2, n_axes=2,
                               rope=layout(head_dim, a.p_range[0] / 2, a.p_range[1] * 4), spectral=spectral,
                               phasor=a.phasor if a.variant in ("phasor", "oracle") else 0)


def mask_for(a, positions, generator):
    pad = torch.zeros(positions.shape[0], positions.shape[2], dtype=torch.bool)
    return make_mask(positions[:, 0], pad, a.mask_ratio, a.mask, (1, 1), 1.0, generator, "random", a.block_share), pad


def explained(se, tv):
    return float(1 - se.mean() / tv.var()) if len(se) > 1 else float("nan")


@torch.no_grad()
def evaluate(a, model, dev, n_batches: int, seed: int) -> dict:
    model.eval()
    g = torch.Generator().manual_seed(seed)
    rows = dict(se=[], tv=[], cyc=[], per=[], ferr_soft=[], ferr_peak=[], per_s=[])
    for _ in range(n_batches):
        values, positions, period, t, y = sample(a.batch, a.n_points, a.window, a.p_range, a.sampling, a.shape, a.noise, g)
        mask, pad = mask_for(a, positions, g)
        if a.variant == "oracle":
            model.phasor.override = (1.0 / period).to(dev)
        out = model(values.to(dev), positions.to(dev), pad.to(dev), mask.to(dev))
        se = ((out.pred.float() - out.target.float()) ** 2)[..., 0].cpu()  # [B, k] in mask order
        tgt = out.target[..., 0].float().cpu()
        tm = t[mask].reshape(t.shape[0], -1)
        for i in range(t.shape[0]):
            vis = t[i][~mask[i]]
            d = (tm[i][:, None] - vis[None]).abs().amin(1)
            rows["se"].append(se[i]); rows["tv"].append(tgt[i]); rows["cyc"].append(d / period[i]); rows["per"].append(period[i].expand(len(d)))
        # the spectral layer's frequency estimate (from the visible tokens)
        sp = model.spectral
        p = torch.softmax(sp.last_logits.float(), -1)
        coord = (p * torch.arange(p.shape[-1], device=p.device)).sum(-1)
        f_soft = torch.exp(sp.log_f0 + sp.step * coord).cpu()
        f_peak = torch.exp(sp.log_f0 + sp.step * sp.last_logits.argmax(-1).float()).cpu()
        f_true = 1.0 / period
        rows["ferr_soft"].append((f_soft / f_true - 1).abs()); rows["ferr_peak"].append((f_peak / f_true - 1).abs()); rows["per_s"].append(period)
    if a.variant == "oracle":
        model.phasor.override = None
    R = {k: torch.cat(v) for k, v in rows.items()}
    cyc_edges = [0, 0.1, 0.5, 2, 5, 10, 30, 100, 300, 1e9]
    by_cycles = []
    for lo, hi in zip(cyc_edges[:-1], cyc_edges[1:]):
        m = (R["cyc"] >= lo) & (R["cyc"] < hi)
        if m.sum() > 20:
            by_cycles.append(dict(lo=lo, hi=hi, n=int(m.sum()), explained=explained(R["se"][m], R["tv"][m]), mse=float(R["se"][m].mean())))
    p_edges = np.exp(np.linspace(math.log(a.p_range[0]), math.log(a.p_range[1]), 9)).tolist()
    by_period = []
    for lo, hi in zip(p_edges[:-1], p_edges[1:]):
        m = (R["per"] >= lo) & (R["per"] < hi)
        ms = (R["per_s"] >= lo) & (R["per_s"] < hi)
        near, far = m & (R["cyc"] < 2), m & (R["cyc"] >= 5)
        by_period.append(dict(lo=lo, hi=hi, n_samples=int(ms.sum()),
                              explained_near=explained(R["se"][near], R["tv"][near]) if near.sum() > 20 else None,
                              explained_far=explained(R["se"][far], R["tv"][far]) if far.sum() > 20 else None,
                              freq_err_soft_median=float(R["ferr_soft"][ms].median()) if ms.any() else None,
                              freq_err_peak_median=float(R["ferr_peak"][ms].median()) if ms.any() else None,
                              freq_within_1pct=float((R["ferr_peak"][ms] < 0.01).float().mean()) if ms.any() else None,
                              peak_width_in_grid_steps=float(math.sqrt(lo * hi) / a.window / a.spectral_rel)))
    model.train()
    return dict(explained_all=explained(R["se"], R["tv"]), by_cycles=by_cycles, by_period=by_period)


def plot(res: dict, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bc = res["by_cycles"]
    ax[0].plot([math.sqrt(max(r["lo"], 0.05) * min(r["hi"], 1000)) for r in bc], [r["explained"] for r in bc], "o-")
    ax[0].set_xscale("log"); ax[0].set_ylim(-0.1, 1.05); ax[0].set_xlabel("distance to the nearest seen point (cycles)"); ax[0].set_ylabel("variance of the target explained"); ax[0].set_title("how far")
    bp = res["by_period"]
    x = [math.sqrt(r["lo"] * r["hi"]) for r in bp]
    ax[1].plot(x, [r["explained_near"] if r["explained_near"] is not None else float("nan") for r in bp], "o-", label="hidden points within 2 cycles")
    ax[1].plot(x, [r["explained_far"] if r["explained_far"] is not None else float("nan") for r in bp], "s-", label="hidden points beyond 5 cycles")
    ax[1].plot(x, [r["freq_within_1pct"] for r in bp], "^--", label="spectral peak within 1 % of the frequency")
    ax[1].set_xscale("log"); ax[1].set_ylim(-0.1, 1.05); ax[1].set_xlabel("period (days)"); ax[1].set_title("which periods"); ax[1].legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["plain", "phasor", "oracle"], default="plain")
    ap.add_argument("--phasor", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-points", type=int, default=256)
    ap.add_argument("--window", type=float, default=250.0)
    ap.add_argument("--p-range", type=float, nargs=2, default=[0.05, 50.0])
    ap.add_argument("--sampling", choices=["random", "regular"], default="random")
    ap.add_argument("--shape", type=float, default=0.0, help="amplitude of a second harmonic (0 = pure sine)")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--mask", choices=["random", "block", "blockplus"], default="blockplus")
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--block-share", type=float, default=0.5)
    ap.add_argument("--spectral-rel", type=float, default=5e-4)
    ap.add_argument("--spectral-aux", type=float, default=0.5, help="weight of the cross-entropy on the true period's bin (as in the real runs); 0 = label-free")
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dec-width", type=int, default=192)
    ap.add_argument("--dec-depth", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    model = build(a).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{a.variant}: {n_par / 1e6:.2f}M parameters, spectral grid {model.spectral.n_bins} bins, {a.sampling} sampling, periods {a.p_range}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 100) * 0.5 * (1 + math.cos(math.pi * min(s / a.steps, 1.0))))
    g = torch.Generator().manual_seed(a.seed + 1)
    log, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        values, positions, period, t, y = sample(a.batch, a.n_points, a.window, a.p_range, a.sampling, a.shape, a.noise, g)
        mask, pad = mask_for(a, positions, g)
        if a.variant == "oracle":
            model.phasor.override = (1.0 / period).to(dev)
        out_ = model(values.to(dev), positions.to(dev), pad.to(dev), mask.to(dev))
        loss = out_.loss
        if a.spectral_aux > 0:
            loss = loss + a.spectral_aux * spectral_aux_loss(model.spectral, period.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0:
            print(f"step {step} loss {out_.loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate(a, model, dev, a.eval_batches, seed=1000)
            r["step"] = step; log.append(r)
            print(f"  eval {step}: explained all {r['explained_all']:.3f} | by cycles " + ", ".join(f"{x['lo']:g}-{x['hi'] if x['hi'] < 1e8 else 'inf'}: {x['explained']:.2f}" for x in r["by_cycles"]), flush=True)
    final = log[-1]
    final["args"] = vars(a); final["history"] = [dict(step=r["step"], explained_all=r["explained_all"]) for r in log]
    json.dump(final, open(out / "results.json", "w"), indent=1)
    plot(final, out / "summary.png", f"sine test, {a.variant}, {a.sampling} sampling, {a.mask} mask")
    torch.save(model.state_dict(), out / "model.pt")
    print("by period:")
    f = lambda v: "   -" if v is None else f"{v:4.2f}"
    for r in final["by_period"]:
        print(f"  {r['lo']:7.3f}-{r['hi']:<7.3f} d  near {f(r['explained_near'])}  far {f(r['explained_far'])}  peak within 1 % {f(r['freq_within_1pct'])}  width/grid {r['peak_width_in_grid_steps']:.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
