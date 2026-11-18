"""Pretrain RoMAE with the paper's masked autoencoding on full light curves.

Half of the tokens of every curve are masked at random each step and a
light decoder reconstructs their fluxes from MASK tokens at their positions.
The decoder has its own ``--decoder-width/--decoder-depth/--decoder-heads``
(the paper's tiny-shallow by default) whatever the encoder size. The probes
run on the encoder extracted with ``model.backbone()``. Examples::

    python examples/train_mae.py --n 2048 --epochs 20 --auto-time
    python examples/train_mae.py --auto-time --mask-ratio 0.75 --decoder-width 96
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.optim.lr_scheduler import LambdaLR

from common import (
    add_data_args,
    add_model_args,
    add_train_args,
    auto_time,
    build_records,
    cosine_schedule,
    device,
    evaluate,
    make_loader,
    model_kwargs,
    save_backbone,
)
from romae_lc import RoMAEForPreTraining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    add_model_args(parser)
    add_train_args(parser, out="runs/mae.pt")
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument(
        "--decoder-width", type=int, default=180, help="tiny-shallow: 180"
    )
    parser.add_argument("--decoder-depth", type=int, default=2, help="tiny-shallow: 2")
    parser.add_argument("--decoder-heads", type=int, default=3, help="tiny-shallow: 3")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dev = device(args)

    train, val, wavelengths = build_records(args)
    auto_time(args, train)
    tok = dict(band_wavelengths=wavelengths, time_scale=args.time_scale)
    loader = make_loader(
        train, tok, args.batch_size, train=True, workers=args.workers, seed=args.seed
    )
    print(
        f"{len(train)} train / {len(val)} val curves, {len(loader)} steps/epoch, {dev}"
    )

    decoder = dict(
        d_model=args.decoder_width,
        nhead=args.decoder_heads,
        depth=args.decoder_depth,
        attention=args.attention,
    )
    model = RoMAEForPreTraining(decoder, args.mask_ratio, **model_kwargs(args)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = LambdaLR(opt, cosine_schedule(args.epochs * len(loader)))
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, t0 = 0.0, time.time()
        for batch in loader:
            with amp:
                out = model(*batch["full"].to(dev))
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            opt.step()
            sched.step()
            total += out.loss.item()
        print(
            f"epoch {epoch:3d}  loss {total / len(loader):.4f}"
            f"  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s"
        )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m = evaluate(
                model.backbone().to(dev), train, val, tok, args.batch_size, dev
            )
            print(
                f"  probe acc {m['acc']:.3f} (train {m['train_acc']:.3f})"
                f"  log-period R2 {m['r2']:.3f}"
            )
    save_backbone(args.out, model.backbone(), tok)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
