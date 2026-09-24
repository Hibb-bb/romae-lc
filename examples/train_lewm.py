"""Pretrain a RoMAE backbone as a LeWorldModel on light-curve windows.

Every star yields ``--n-frames`` consecutive time windows of ``--window``
days (see ``romae_lc.frames.FrameConfig``), the action after each window
being the advance to the next one in window units, drawn from
``--advance lo hi``. The loss is the next-window latent prediction of a
causal AdaLN-zero predictor plus SIGReg on the latents of every window, with
no stop-gradient or EMA (Maes, Le Lidec et al. 2026, arXiv:2603.19312).
Model and optimiser defaults are those of the paper and the released
``config/train/lewm.yaml`` (width 192, encoder depth 12, predictor 6 x 16
heads with 10 % dropout, lambda 0.1, AdamW 5e-5 / 1e-3, gradient clipping
1.0, 1 % linear warmup then cosine, stepped per optimizer step); ``--width``,
``--advance`` and the token bounds are the light-curve translation. A linear
probe on the frozen backbone features (mean over the windows of a fixed
sequence per star; ``--probe-latent`` probes the post-projector latent
instead) tracks class accuracy and the R2 of log period. The validation
``pred`` / ``sigreg`` are computed in eval mode (BatchNorm running
statistics, no predictor dropout, as the official validation step), so in
very short runs ``val pred`` lags the train-mode ``pred`` until the running
statistics catch up; SIGReg is the Epps-Pulley statistic on a batch of
latents and carries a factor of the batch size, so ``val sigreg`` (printed
with its batch size) is on the training scale only when the validation
batches have ``--batch-size`` rows. Examples::

    python examples/train_lewm.py --n 2048 --epochs 20 --auto-time
    python examples/train_lewm.py --window 30 --advance 1 2 --history 3 --n-frames 4 --lamb 0.1
    python examples/train_lewm.py --rope simplex --attention linear --no-actions
    python examples/train_lewm.py --n 256 --epochs 2 --width 96 --depth 2 --pred-depth 1 --pred-heads 2 --n-slices 64   # CPU smoke run
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

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
    save_backbone,
)
from romae_lc import (
    ARPredictor,
    FrameConfig,
    FrameDataset,
    LeWorldModel,
    collate_frames,
    lewm_mlp,
)


def add_world_model_args(parser: argparse.ArgumentParser) -> None:
    """Frame geometry, loss, predictor and optimisation extras of LeWM."""
    g = parser.add_argument_group("world model")
    g.add_argument("--history", type=int, default=3, help="predictor context")
    g.add_argument(
        "--n-frames",
        type=int,
        default=4,
        help="windows per sequence (history + 1); more trains every position on "
        "next-window prediction (paper Alg. 3), the released num_preds > 1 "
        "k-step mode is not implemented",
    )
    g.add_argument("--window", type=float, default=30.0, help="window length, days")
    g.add_argument(
        "--advance",
        type=float,
        nargs=2,
        default=(1.0, 2.0),
        metavar=("LO", "HI"),
        help="uniform start-to-start advance in window units (1 = contiguous)",
    )
    g.add_argument("--min-tokens", type=int, default=8, help="points per window")
    g.add_argument("--max-tokens", type=int, default=512, help="subsample cap")
    g.add_argument(
        "--lamb",
        type=float,
        default=0.1,
        help="SIGReg weight (paper 0.1, released config 0.09)",
    )
    g.add_argument("--n-slices", type=int, default=1024, help="SIGReg projections")
    g.add_argument("--knots", type=int, default=17, help="Epps-Pulley nodes")
    g.add_argument("--t-max", type=float, default=3.0, help="Epps-Pulley bound")
    g.add_argument("--pred-depth", type=int, default=6)
    g.add_argument("--pred-heads", type=int, default=16)
    g.add_argument("--pred-dim-head", type=int, default=64)
    g.add_argument("--pred-mlp", type=int, default=2048)
    g.add_argument("--pred-dropout", type=float, default=0.1)
    g.add_argument(
        "--proj-hidden", type=int, default=2048, help="projector / pred_proj width"
    )
    g.add_argument("--clip", type=float, default=1.0, help="gradient norm clip")
    g.add_argument(
        "--warmup", type=float, default=0.01, help="warmup fraction of the steps"
    )
    g.add_argument(
        "--no-actions",
        action="store_true",
        help="unconditional predictor (actions=None; an extension, the official "
        "model is always action-conditioned)",
    )
    g.add_argument(
        "--probe-latent",
        action="store_true",
        help="probe the post-projector latent z instead of the backbone features",
    )


def make_frame_loader(
    records, cfg, tokenize_kwargs, batch_size, train=False, workers=0, seed=0
) -> DataLoader:
    """Per-time-step padded ``Tokens`` batches of frame sequences; a ``train``
    loader shuffles, drops the last batch and re-draws the windows every
    epoch. Records too short for a sequence are dropped by ``FrameDataset``,
    so ``loader.dataset.indices`` says which ones remain."""
    ds = FrameDataset(records, cfg, seed=seed, epoch_seed=train)
    return DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=train and len(ds) > batch_size,
        num_workers=workers,
        collate_fn=partial(collate_frames, **tokenize_kwargs),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    add_model_args(
        parser, depth_help="encoder blocks (paper: 12; use 4 for CPU smoke runs)"
    )
    add_train_args(parser, out="runs/lewm.pt")
    add_world_model_args(parser)
    # C lewm.yaml optimiser (lr 5e-5, weight decay 1e-3); P Sec. 3.1 encoder
    # (ViT-Tiny: 12 layers, 3 heads, width 192 = the --width default).
    parser.set_defaults(lr=5e-5, wd=1e-3, depth=12, heads=3)
    args = parser.parse_args()
    if args.n_frames < 2:
        parser.error("--n-frames must be >= 2 (history plus one target)")
    if args.width < 184:
        print(
            f"warning: --width {args.width} < 184; paper Fig. 15 / App. G report "
            "a performance drop below ~184 latent dimensions"
        )
    torch.manual_seed(args.seed)
    dev = device(args)

    train, val, wavelengths = build_records(args)
    # The encoder only ever sees window-long frames, so the slowest rotary
    # channel need not turn over the whole baseline: half a turn per window.
    auto_time(args, train, lam_max=2 * args.window)
    tok = dict(band_wavelengths=wavelengths, time_scale=args.time_scale)
    cfg = FrameConfig(
        n_frames=args.n_frames,
        window=args.window,
        advance=tuple(args.advance),
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    loader = make_frame_loader(
        train,
        cfg,
        tok,
        args.batch_size,
        train=True,
        workers=args.workers,
        seed=args.seed,
    )
    val_loader = make_frame_loader(val, cfg, tok, args.batch_size, seed=args.seed)
    # Probe rows must match the FrameDataset rows (short records are dropped).
    train_kept = [train[i] for i in loader.dataset.indices]
    val_kept = [val[i] for i in val_loader.dataset.indices]
    print(
        f"{len(train_kept)} train / {len(val_kept)} val curves, "
        f"{len(loader)} steps/epoch, {dev}"
    )

    backbone = build_backbone(args)
    d = backbone.embed_dim
    predictor = ARPredictor(
        d,
        n_frames=max(args.history, args.n_frames - 1),
        depth=args.pred_depth,
        heads=args.pred_heads,
        dim_head=args.pred_dim_head,
        mlp_dim=args.pred_mlp,
        dropout=args.pred_dropout,
    )
    model = LeWorldModel(
        backbone,
        projector=lewm_mlp(d, args.proj_hidden),
        predictor=predictor,
        pred_proj=lewm_mlp(d, args.proj_hidden),
        history=args.history,
        lamb=args.lamb,
        n_slices=args.n_slices,
        t_max=args.t_max,
        n_points=args.knots,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # Official: LinearWarmupCosineAnnealingLR from 0 over 1 % of the steps to
    # 0 at the end; cosine_schedule starts its ramp at 1/warmup instead of 0.
    sched = LambdaLR(opt, cosine_schedule(args.epochs * len(loader), args.warmup))
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")

    def to_dev(batch):
        frames = [f.to(dev) for f in batch["frames"]]
        actions = None if args.no_actions else batch["actions"].to(dev)
        return frames, actions

    @torch.no_grad()
    def embed_frames(records):
        """``[len(records), D]`` mean over the windows of one deterministic
        sequence per star (no record is dropped: already filtered)."""
        ds = FrameDataset(records, cfg, seed=args.seed, epoch_seed=False)
        fn = partial(collate_frames, **tok)
        probe_loader = DataLoader(ds, args.batch_size, shuffle=False, collate_fn=fn)
        was = model.training
        model.eval()
        zs = []
        for batch in probe_loader:
            frames = [f.to(dev) for f in batch["frames"]]
            z = model.encode(frames, project=args.probe_latent).mean(1)
            zs.append(z.float().cpu())
        model.train(was)
        return torch.cat(zs).numpy()

    @torch.no_grad()
    def validate():
        """``(pred, sigreg, batch size)`` over the fixed validation windows in
        eval mode (BatchNorm running statistics, no dropout), per-batch means
        weighted by their batch size. SIGReg carries a factor of the batch
        size (``EppsPulley``), so the value is on the training scale only
        when the validation batches are ``--batch-size`` long; the largest
        batch size is returned for the print."""
        was = model.training
        model.eval()
        sums, n, b_max = np.zeros(2), 0, 0
        for batch in val_loader:
            b = batch["actions"].shape[0]
            with amp:
                out = model(*to_dev(batch))
            sums += b * np.array([out.pred_loss.item(), out.sigreg_loss.item()])
            n, b_max = n + b, max(b_max, b)
        model.train(was)
        return sums / max(1, n), b_max

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums, t0 = np.zeros(4), time.time()
        for batch in loader:
            frames, actions = to_dev(batch)
            with amp:
                out = model(frames, actions)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            sums += [
                out.loss.item(),
                out.pred_loss.item(),
                out.sigreg_loss.item(),
                out.straightness.item(),
            ]
        loss, pred, sig, straight = sums / len(loader)
        print(
            f"epoch {epoch:3d}  loss {loss:.4f}  pred {pred:.4f}  sigreg {sig:.4f}"
            f"  straight {straight:.3f}  lr {sched.get_last_lr()[0]:.2e}"
            f"  {time.time() - t0:.0f}s"
        )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            (v_pred, v_sig), v_b = validate()
            m = evaluate(
                backbone,
                train_kept,
                val_kept,
                tok,
                args.batch_size,
                dev,
                embed_fn=embed_frames,
            )
            print(
                f"  val (B={v_b}) pred {v_pred:.4f}  sigreg {v_sig:.4f}"
                f"  probe acc {m['acc']:.3f} (train {m['train_acc']:.3f})"
                f"  log-period R2 {m['r2']:.3f}"
            )
    save_backbone(args.out, backbone, tok)
    wm_path = Path(args.out).with_name(Path(args.out).stem + "_wm.pt")
    torch.save(
        dict(
            state_dict=model.state_dict(),
            hparams=model.hparams,
            backbone=dict(
                backbone.hparams, encoder=asdict(backbone.cfg), pool=backbone.pool
            ),
            tokenize=tok,
            frames=asdict(cfg),
            no_actions=args.no_actions,
        ),
        wm_path,
    )
    print(f"saved {args.out} and {wm_path}")


if __name__ == "__main__":
    main()
