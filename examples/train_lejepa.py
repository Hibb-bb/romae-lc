"""Pretrain a RoMAE backbone with LeJEPA on light-curve views.

Every star yields two global and four local time windows (see
``romae_lc.data.ViewConfig``); the loss is view invariance plus SIGReg. A
linear probe on the frozen embeddings tracks class accuracy and the R2 of
log period. Examples::

    python examples/train_lejepa.py --n 2048 --epochs 20 --auto-time
    python examples/train_lejepa.py --auto-time --rope simplex --attention linear
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from common import (
    add_data_args,
    add_model_args,
    add_train_args,
    auto_time,
    build_backbone,
    build_records,
    cosine_schedule,
    device,
    evaluate,
    make_loader,
    save_backbone,
)
from romae_lc import LeJEPA, ViewConfig, mlp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    add_model_args(parser)
    add_train_args(parser, out="runs/lejepa.pt")
    parser.add_argument("--lamb", type=float, default=0.02, help="SIGReg weight")
    parser.add_argument("--proj-dim", type=int, default=128, help="projector output")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dev = device(args)

    train, val, wavelengths = build_records(args)
    auto_time(args, train)
    tok = dict(band_wavelengths=wavelengths, time_scale=args.time_scale)
    loader = make_loader(
        train,
        tok,
        args.batch_size,
        train=True,
        view_cfg=ViewConfig(),
        workers=args.workers,
        seed=args.seed,
    )
    print(
        f"{len(train)} train / {len(val)} val curves, {len(loader)} steps/epoch, {dev}"
    )

    backbone = build_backbone(args)
    d = backbone.embed_dim
    model = LeJEPA(backbone, proj=mlp([d, 4 * d, 4 * d, args.proj_dim]), lamb=args.lamb)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = LambdaLR(opt, cosine_schedule(args.epochs * len(loader)))
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums, t0 = np.zeros(3), time.time()
        for batch in loader:
            views = [[v.to(dev) for v in batch[k]] for k in ("global", "local")]
            with amp:
                out = model(global_views=views[0], local_views=views[1])
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            opt.step()
            sched.step()
            sums += [out.loss.item(), out.inv_loss.item(), out.sigreg_loss.item()]
        loss, inv, sig = sums / len(loader)
        print(
            f"epoch {epoch:3d}  loss {loss:.4f}  inv {inv:.4f}  sigreg {sig:.4f}"
            f"  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s"
        )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m = evaluate(backbone, train, val, tok, args.batch_size, dev)
            print(
                f"  probe acc {m['acc']:.3f} (train {m['train_acc']:.3f})"
                f"  log-period R2 {m['r2']:.3f}"
            )
    save_backbone(args.out, backbone, tok)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
