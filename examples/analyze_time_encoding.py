"""Choose the rotary time band for a dataset and check a trained backbone.

Prints the ``suggest_time_encoding`` report for the time block of the model
described by ``--width/--heads/--rope``, optionally plots the detected
periods against the recommended rotary ladder, and with ``--ckpt``
prints the ``time_shuffle_score`` of a saved backbone on the validation
curves (1 = blind to time, lower is better). Examples::

    python examples/analyze_time_encoding.py --n 512 --plot runs/te.png
    python examples/analyze_time_encoding.py --ckpt runs/lejepa.pt
"""

from __future__ import annotations

import argparse

import torch

from common import (
    add_data_args,
    add_model_args,
    build_records,
    device,
    embed,
    load_backbone,
    make_loader,
    time_report,
)
from romae_lc import plot_time_encoding, time_shuffle_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    add_model_args(parser)
    parser.add_argument("--plot", help="save the periodogram figure here (.png)")
    parser.add_argument("--ckpt", help="saved backbone to score on the val curves")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    train, val, _ = build_records(args)
    report = time_report(args, train)
    print(report)
    if args.plot:
        ax = plot_time_encoding(report)
        ax.figure.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"saved {args.plot}")
    if args.ckpt:
        dev = device(args)
        backbone, tok = load_backbone(args.ckpt, dev)
        base = backbone.rope.blocks[0].base
        print(f"{args.ckpt}: time_scale = {tok['time_scale']:.4g}, base = {base:.4g}")
        g = torch.Generator().manual_seed(args.seed)
        # One reference mean/std for every batch, so the scores share a scale
        # (time_shuffle_score standardises over its batch otherwise).
        z = torch.as_tensor(embed(backbone, val, tok, args.batch_size, dev))
        stats = (z.mean(0), z.std(0, unbiased=False) + 1e-6)
        scores = [
            time_shuffle_score(backbone, batch["full"].to(dev), g, stats=stats).cpu()
            for batch in make_loader(val, tok, args.batch_size)
        ]
        score = torch.cat(scores).mean().item()
        print(f"time_shuffle_score on {len(val)} val curves: {score:.3f} (1 = blind)")


if __name__ == "__main__":
    main()
