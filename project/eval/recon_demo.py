"""Does the autoencoder recover the light curve? Masked reconstruction shown
for a few validation stars: one window, half the points hidden at random
(the training task) and the last quarter hidden as a block (a forecast
inside the window); the decoder's predictions at the hidden times against
the truth, in time and folded on the catalogue period.

    python -m project.eval.recon_demo --ckpt project/runs/maew_spec/mae.pt --out project/results/recon_demo_spec
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from project.common import add_device_arg, data_args_from, fine_class, get_device, grid_item, load_data, load_mae
from project.mae_data import make_mask
from project.plot_period_examples import INK, INK2, SURFACE, style

HIDDEN, PRED, SEEN = "#0b0b0b", "#2171b5", "#b5b3ae"


def pick_stars(val, classes, min_points, per_class=1):
    out = []
    for cls in classes:
        found = [i for i, r in enumerate(val) if fine_class(r) == cls and r.period and r.period > 0 and r.t.size >= min_points]
        out += found[:per_class]
    return out


def best_window(r, cfg, index):
    """The window of the grid with the most points (capped like the cache)."""
    item = grid_item(r, cfg, index=index, cap=cfg.max_tokens)
    frames = item["frames"]
    k = int(np.argmax([len(f[0]) for f in frames]))
    return frames[k]


def reconstruct(model, spec, frame, mask_kind, ratio, dev, seed=0):
    tok = spec.tokens([frame]).to(dev)
    values, positions, pad = tok.values, tok.positions, tok.pad_mask
    g = torch.Generator(device="cpu").manual_seed(seed)
    if mask_kind == "random":
        mask = make_mask(positions[:, 0].cpu(), pad.cpu(), ratio, "random", generator=g).to(dev)
    else:
        mask = make_mask(positions[:, 0].cpu(), pad.cpu(), ratio, "block", n_blocks=(1, 1), block_prob=1.0, generator=g, position="last", block_share=1.0).to(dev)
    with torch.no_grad():
        out = model(values, positions, pad, mask)
    m = mask[0].cpu().numpy()
    pred = out.pred[0, :, 0].float().cpu().numpy()
    truth = out.target[0, :, 0].float().cpu().numpy()
    return m, pred, truth


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--classes", nargs="*", default=["EW/EB", "EA", "RRAB", "RRC", "ROT", "LPV"])
    p.add_argument("--min-points", type=int, default=400)
    p.add_argument("--ratio", type=float, default=0.5, help="hidden share for the random mask")
    p.add_argument("--block", type=float, default=0.25, help="hidden share for the block at the end")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--n-sim", type=int, default=None)
    add_device_arg(p)
    args = p.parse_args(argv)
    t0 = time.time()
    dev = get_device(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, meta = load_mae(args.ckpt, dev)
    cfg, spec = meta.cfg, meta.spec
    over = argparse.Namespace(data=args.data, max_rows=args.max_rows, n_sim=args.n_sim)
    val = load_data(data_args_from(meta.args, over), splits=("validation",))["validation"]
    stars = pick_stars(val, args.classes, args.min_points)
    kinds = [("random", args.ratio, f"{args.ratio:.0%} of the points hidden at random"), ("block", args.block, f"the last {args.block:.0%} of the window hidden")]
    fig, axes = plt.subplots(len(stars) * len(kinds), 2, figsize=(11, 2.4 * len(stars) * len(kinds)), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    scores = []
    row = 0
    for i in stars:
        r = val[i]
        frame = best_window(r, cfg, i)
        t, y = frame[0].astype(np.float64), frame[1].astype(np.float64)
        for kind, ratio, label in kinds:
            m, pred, truth = reconstruct(model, spec, frame, kind, ratio, dev, args.seed)
            th, yh = t[m], y[m]
            order = np.argsort(th)  # the model's masked tokens follow the (time-sorted) token order; the frame may not
            th_sorted = np.sort(th)
            assert np.allclose(th_sorted, th[order])
            r2 = 1.0 - np.sum((truth - pred) ** 2) / max(np.sum((truth - truth.mean()) ** 2), 1e-9)
            r2_seen = 1.0 - np.sum((truth - y[~m].mean()) ** 2) / max(np.sum((truth - truth.mean()) ** 2), 1e-9)
            scores.append(dict(ztf_id=str(r.meta.get("id")), cls=fine_class(r), kind=kind, r2=float(r2), r2_mean_of_seen=float(r2_seen), n_hidden=int(m.sum()), n_seen=int((~m).sum())))
            # the model's token order is the tokenizer's (sorted by time); map predictions to the sorted hidden times
            for ax, fold in zip(axes[row], (False, True)):
                style(ax)
                x_seen, x_hid = t[~m] - t.min(), th_sorted - t.min()
                if fold:
                    P = float(r.period)
                    x_seen, x_hid = np.mod(x_seen / P, 1.0), np.mod(x_hid / P, 1.0)
                ax.scatter(x_seen, y[~m], s=7, color=SEEN, linewidths=0, label="seen")
                ax.scatter(x_hid, truth, s=9, color=HIDDEN, linewidths=0, label="hidden, true")
                ax.scatter(x_hid, pred, s=9, color=PRED, marker="x", linewidths=0.8, label="hidden, predicted")
                ax.set_xlabel("phase on the catalogue period" if fold else "days since the window start", fontsize=7.5, color=INK2)
                if not fold:
                    ax.set_title(f"{fine_class(r)}  {r.meta.get('id')}   P = {r.period:.5g} d   {label}   R2 on the hidden points {r2:.2f} (mean of the seen points {r2_seen:.2f})",
                                 fontsize=7.5, color=INK, loc="left")
            axes[row][0].set_ylabel("brightness (standardised)", fontsize=7.5, color=INK2)
            row += 1
    axes[0][0].legend(fontsize=6.5, frameon=False, loc="upper right", labelcolor=INK2)
    fig.suptitle("Masked reconstruction by the autoencoder: one window per star, hidden points predicted from the seen ones", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out / "recon_demo.png", facecolor=SURFACE)
    plt.close(fig)
    json.dump(scores, open(out / "scores.json", "w"), indent=1)
    for s in scores:
        print(f"{s['cls']:7s} {s['ztf_id']:20s} {s['kind']:6s} hidden {s['n_hidden']:3d} seen {s['n_seen']:3d}  R2 {s['r2']:.2f}  (mean of seen {s['r2_mean_of_seen']:.2f})")
    print(f"wrote {out}/recon_demo.png in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
