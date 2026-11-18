"""Draw what the sine-test models do: for a few periods, one window with its
last part hidden, each saved model's prediction at dense queries against the
true sinusoid, near the start of the hidden stretch and far into it.

    python -m project.eval.sine_demo --model plain=project/results/sine_test10k/plain \
        --model phasor=project/results/sine_test/phasor --out project/results/sine_demo
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from project.plot_period_examples import INK, INK2, SURFACE, style
from project.sine_test import build, parse_args as sine_args
from romae_lc.model import gen_mask
from project.mae_data import make_mask

COLOUR = ["#8c8c8c", "#eb6834", "#1baf7a", "#6a51a3", "#2a78d6"]


def load(folder: Path):
    res = json.load(open(folder / "results.json"))
    a = sine_args(["--out", str(folder)])
    for k, v in res["args"].items():
        setattr(a, k, v)
    model = build(a)
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
    return model.eval(), a


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True, help="NAME=folder of a sine_test run, repeatable")
    p.add_argument("--periods", type=float, nargs="*", default=[0.1, 0.35, 1.3, 6.0, 25.0])
    p.add_argument("--far-cycles", type=float, default=100.0, help="the far view starts this many cycles into the hidden stretch")
    p.add_argument("--n-points", type=int, default=256)
    p.add_argument("--window", type=float, default=250.0)
    p.add_argument("--hide", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    models = {}
    for item in args.model:
        name, folder = item.split("=", 1)
        models[name] = load(Path(folder))
    g = torch.Generator().manual_seed(args.seed)
    fig, axes = plt.subplots(len(args.periods), 2, figsize=(13, 2.5 * len(args.periods)))
    fig.patch.set_facecolor(SURFACE)
    for row, period in enumerate(args.periods):
        t = torch.sort(torch.rand(1, args.n_points, generator=g) * args.window, 1).values
        phase = 2 * math.pi * torch.rand(1, generator=g)
        y = torch.sin(2 * math.pi * t / period + phase)
        values = torch.stack([y, torch.zeros_like(y)], -1)
        positions = torch.stack([t + 1.0, torch.full_like(t, 0.6)], 1)
        pad = torch.zeros(1, args.n_points, dtype=torch.bool)
        mask = make_mask(positions[:, 0], pad, args.hide, "block", (1, 1), 1.0, torch.Generator().manual_seed(1), "last", 1.0)
        t_hide = float(t[0][mask[0]].min())
        far = min(args.far_cycles, (args.window - t_hide) / period - 6.0)  # stay inside the window
        views = [(t_hide - 2 * period, t_hide + 6 * period, "near the hidden start"), (t_hide + far * period, t_hide + (far + 6) * period, f"{far:.0f} cycles into the hidden stretch")]
        for col, (lo, hi, label) in enumerate(views):
            ax = axes[row, col]; style(ax)
            hi = min(hi, args.window); lo = max(lo, 0.0)
            tt = torch.linspace(lo, hi, 400)
            ax.plot(tt, torch.sin(2 * math.pi * tt / period + phase), "-", color=INK2, lw=0.9, alpha=0.6, label="true" if row == 0 and col == 0 else None)
            seen = ~mask[0]
            ax.plot(t[0][seen], y[0][seen], "o", ms=3, color=INK, label="seen" if row == 0 and col == 0 else None)
            ax.plot(t[0][mask[0]], y[0][mask[0]], "o", ms=3.5, mfc="none", color=INK, label="hidden" if row == 0 and col == 0 else None)
            if col == 0:
                ax.axvline(t_hide, color=INK2, lw=0.8, ls=":")
            # dense queries over the view as extra hidden tokens
            t_q = tt[None]
            v2 = torch.cat([values, torch.zeros(1, 400, 2)], 1)
            p2 = torch.cat([positions, torch.stack([t_q + 1.0, torch.full_like(t_q, 0.6)], 1)], 2)
            pad2 = torch.cat([pad, torch.zeros(1, 400, dtype=torch.bool)], 1)
            mask2 = torch.cat([mask, torch.ones(1, 400, dtype=torch.bool)], 1)
            for k, (name, (model, a)) in enumerate(models.items()):
                if model.phasor is not None:
                    model.phasor.override = (torch.tensor([1.0 / period]) if a.variant in ("oracle", "phaserope_oracle") else None)
                model.phase_override = torch.tensor([1.0 / period]) if a.variant in ("oracle", "phaserope_oracle") else None
                with torch.no_grad():
                    o = model(v2, p2, pad2, mask2)
                pred = o.pred[0, :, 0]
                tm = (p2[0, 0] - 1.0)[mask2[0]]
                isq = (torch.arange(v2.shape[1]) >= args.n_points)[mask2[0]]
                ax.plot(tm[isq], pred[isq], "-", color=COLOUR[k % len(COLOUR)], lw=1.2, label=name if row == 0 and col == 0 else None)
            ax.set_xlim(lo, hi); ax.set_ylim(-1.6, 1.6)
            ax.set_title(f"P = {period:g} d, {label}", fontsize=8, loc="left", color=INK)
            ax.tick_params(labelsize=7)
            if row == len(args.periods) - 1:
                ax.set_xlabel("days in the window", fontsize=8)
    axes[0, 0].legend(fontsize=6.5, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "sine_demo.png", dpi=130, facecolor=SURFACE)
    print(f"wrote {out / 'sine_demo.png'}")


if __name__ == "__main__":
    main()
