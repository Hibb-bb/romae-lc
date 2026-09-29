"""Stage 1: the autoencoder. Masked pretraining of the window encoder (the
RoMAE recipe), or the bottleneck autoencoder with ``--bottleneck``.

Every window of a frame sequence is a sample: half of its tokens are hidden
at random and a light decoder reconstructs the magnitudes of the hidden ones
from MASK tokens at their positions (:class:`~romae_lc.RoMAEForPreTraining`).
Next-latent prediction is satisfied by static per-object statistics;
predicting a held-out magnitude at an arbitrary time inside a window is not:
it needs period, phase, shape and amplitude in the encoder output. The
encoder, ladder, token spec and frame settings are those of
:mod:`project.train_wm` (same arguments), so the checkpoint ``mae.pt`` is
what ``train_wm.py --init-backbone`` starts from and what the stage-2
predictor and the stage-3 decoder read as the frozen encoder.

``--bottleneck`` trains a :class:`project.bottleneck.BottleneckAE` instead:
the same encoder, but the decoder is a :class:`~project.decoder.QueryDecoder`
that sees only the pooled CLS latent (plus the query positions and errors),
so the reconstruction demand acts on the latent the later stages read, not
on the visible tokens' outputs. ``--mask-ratio`` is then the fraction of
points hidden from the encoder, ``--dec-width/--dec-heads/--dec-depth`` the
query decoder's size (``dec-width / dec-heads`` must equal the encoder's
head dimension) and ``--bottleneck-loss all|hidden`` which points are scored.
It needs the error channel (no ``--no-err-channel``). The checkpoint has
``kind == "bottleneck"``; every loader and probe treats it like ``mae.pt``.

Resumable like ``train_wm`` (``last.pt``, ``--time-budget``); ``mae.pt`` and
``DONE`` appear when ``--steps`` is reached. Evaluations at step 0 and every
``--eval-every`` steps: the validation reconstruction loss, the linear class
probe and log-period R2 (:func:`project.common.probe` on the encoder) and the
time-shuffle score; the hand-feature baseline probe once at the start.

    python -m project.pretrain_mae --out project/runs/mae_w250 --window 250 --wandb
    python -m project.pretrain_mae --bottleneck --out project/runs/bn_w250 --window 250
    python -m project.pretrain_mae --data sim --n-sim 128 --steps 20 --width 48 \\
        --depth 1 --heads 2 --window 30 --min-tokens 4 --batch-size 8 \\
        --eval-every 10 --probe-train 64 --probe-val 32 --shuffle-objects 16 \\
        --device cpu --out /tmp/mae_smoke
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from romae_lc import RoMAEForPreTraining

from project.bottleneck import BottleneckAE, bottleneck_state, fuse_tokens
from project.common import (
    FrameEncoder,
    JsonlLog,
    Ladder,
    PooledEncoder,
    TokenSpec,
    add_data_args,
    add_device_arg,
    add_frame_args,
    add_ladder_args,
    add_model_args,
    baseline_probe,
    cosine_schedule,
    describe_probe,
    dump_json,
    encoder_config,
    err_stats,
    frame_config,
    frame_loader,
    fuse_frames,
    get_device,
    get_ladder,
    load_data,
    mae_state,
    n_params,
    probe,
    resolve_model_args,
    rope_geometry,
    rope_layouts,
    save_atomic,
    seed_all,
    subset,
)
from project.diagnostics import shuffle_score
from project.tracking import StepTimer, Tracker, add_wandb_args, gpu_stats

#: Arguments read from the command line even when resuming a checkpoint.
RUN_CONTROL = (
    "steps",
    "time_budget",
    "workers",
    "device",
    "eval_every",
    "ckpt_every",
    "log_every",
    "out",
    "no_resume",
    "no_eval_at_start",
    "probe_train",
    "probe_val",
    "val_objects",
    "shuffle_objects",
    "persistent_workers",
    "prefetch",
    "pin_memory",
    "wandb",
    "wandb_project",
    "wandb_entity",
    "wandb_name",
    "wandb_group",
    "wandb_tags",
)


def add_train_args(parser) -> None:
    g = parser.add_argument_group("training")
    g.add_argument("--steps", type=int, default=50_000, help="optimizer steps")
    g.add_argument(
        "--batch-size", type=int, default=64, help="sequences (x n_frames windows)"
    )
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--wd", type=float, default=0.05)
    g.add_argument("--warmup", type=float, default=0.02, help="fraction of steps")
    g.add_argument("--clip", type=float, default=1.0)
    g.add_argument("--mask-ratio", type=float, default=0.5)
    g.add_argument("--dec-width", type=int, default=192, help="MAE decoder width")
    g.add_argument("--dec-depth", type=int, default=2)
    g.add_argument("--dec-heads", type=int, default=3)
    g.add_argument(
        "--bottleneck",
        action="store_true",
        help="bottleneck autoencoder: a query decoder reads only the pooled "
        "latent (project.bottleneck); --dec-* size it, --mask-ratio hides points",
    )
    g.add_argument(
        "--bottleneck-loss",
        choices=("all", "hidden"),
        default="all",
        help="score every real point of the window or only the hidden ones",
    )
    g.add_argument(
        "--bottleneck-var",
        choices=("learned", "known", "unit"),
        default="unit",
        help="bottleneck: the variance the points are scored under: 'unit' "
        "(plain squared error, the masked-pretraining loss that learns "
        "period; default), 'known' (the reported error, a 1/sigma^2 weight), "
        "'learned' (known plus a learned extra variance: lets the decoder "
        "explain an oscillation as scatter and stalls period learning)",
    )
    g.add_argument(
        "--denoise",
        action="store_true",
        help="bottleneck: the encoder sees magnitudes redrawn from N(m, sigma) "
        "while the decoder is scored on the observed ones (measurement-noise "
        "invariance at the known level)",
    )
    g.add_argument("--eval-every", type=int, default=2500)
    g.add_argument("--ckpt-every", type=int, default=1000)
    g.add_argument("--log-every", type=int, default=100)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--persistent-workers", action="store_true")
    g.add_argument("--prefetch", type=int, default=2, help="batches per worker")
    g.add_argument("--pin-memory", action="store_true")
    g.add_argument("--out", default="project/runs/mae")
    g.add_argument(
        "--no-resume", action="store_true", help="ignore an existing last.pt"
    )
    g.add_argument(
        "--time-budget", type=float, default=0.0, help="stop and checkpoint after s"
    )
    g.add_argument("--probe-train", type=int, default=4000, help="probe fit records")
    g.add_argument("--probe-val", type=int, default=2000)
    g.add_argument("--val-objects", type=int, default=1000, help="for the val loss")
    g.add_argument("--shuffle-objects", type=int, default=512)
    g.add_argument(
        "--no-eval-at-start",
        action="store_true",
        help="skip the step-0 evaluation of the untrained encoder",
    )
    add_device_arg(g)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_args(parser)
    add_frame_args(parser)
    add_ladder_args(parser)
    add_model_args(parser)
    add_train_args(parser)
    add_wandb_args(parser)
    args = parser.parse_args(argv)
    resolve_model_args(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    last = out / "last.pt"
    ckpt = None
    if last.is_file() and not args.no_resume:
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        for k, v in ckpt["args"].items():
            if k not in RUN_CONTROL:
                setattr(args, k, v)
        print(f"resuming {last} at step {ckpt['step']}")
    seed_all(args.seed)
    dev = get_device(args)
    t_start = time.time()

    data = load_data(args)
    train, val = data["train"], data["validation"]
    print(
        f"{len(train)} train / {len(val)} val records, {len(data.classes)} classes, "
        f"loaded in {time.time() - t_start:.0f}s"
    )
    cfg = frame_config(args)
    if ckpt is not None:
        ladder = Ladder.from_dict(ckpt["ladder"])
        spec = TokenSpec.from_dict(ckpt["spec"])
    else:
        t0 = time.time()
        ladder = get_ladder(args, train, out, rope_geometry(args))
        print(ladder.summary or "ladder reused", f"({time.time() - t0:.0f}s)")
        stats = None if args.no_err_channel else err_stats(train)
        spec = TokenSpec(
            dict(band_wavelengths=data.wavelengths, time_scale=ladder.time_scale), stats
        )
    if args.bottleneck and spec.err_stats is None:
        parser.error("--bottleneck needs the error channel (drop --no-err-channel)")
    dump_json(vars(args), out / "args.json")
    days = ladder.flat
    print(
        f"ladder: {ladder.n_rungs} distinct wavelengths from {days[0]:.4g} to "
        f"{days[-1]:.4g} d, {ladder.per_head} per head, {ladder.layers} layer "
        f"ladder(s) x {ladder.heads} head ladder(s), deal {ladder.deal}"
    )

    loader = frame_loader(
        train,
        cfg,
        spec,
        args.batch_size,
        train=True,
        workers=args.workers,
        seed=args.seed,
        persistent=args.persistent_workers,
        prefetch=args.prefetch,
        pin_memory=args.pin_memory,
    )
    train_kept = [train[i] for i in loader.dataset.indices]
    val_sub = subset(val, args.val_objects, args.seed)
    val_loader = frame_loader(val_sub, cfg, spec, args.batch_size, seed=args.seed)
    val_kept = [val_sub[i] for i in val_loader.dataset.indices]
    print(
        f"{len(train_kept)} train / {len(val_kept)} val sequences, "
        f"{len(loader)} steps per epoch of {cfg.n_frames} windows each, {cfg}, {dev}"
    )

    decoder = dict(
        d_model=args.dec_width,
        nhead=args.dec_heads,
        depth=args.dec_depth,
        attention=args.attention,
    )
    if ckpt is not None and ckpt.get("kind") == "bottleneck":
        model = BottleneckAE.from_checkpoint(ckpt)
    elif ckpt is not None:
        model = RoMAEForPreTraining(**ckpt["mae"], **ckpt["backbone"])
    elif args.bottleneck:
        model = BottleneckAE(
            encoder=encoder_config(args),
            n_channels=spec.n_channels,
            err_stats=spec.err_stats,
            n_axes=2,
            rope=rope_layouts(args, ladder),
            decoder=decoder,
            mask_ratio=args.mask_ratio,
            loss_on=args.bottleneck_loss,
            denoise=args.denoise,
            learned_var={"learned": True, "known": False, "unit": "unit"}[args.bottleneck_var],
        )
    else:
        model = RoMAEForPreTraining(
            decoder=decoder,
            mask_ratio=args.mask_ratio,
            target_channels=1,
            encoder=encoder_config(args),
            n_channels=spec.n_channels,
            n_axes=2,
            rope=rope_layouts(args, ladder),
        )
    model = model.to(dev)
    encoder = FrameEncoder(PooledEncoder(model))
    sizes = dict(
        params_encoder=n_params(model.transformer) + n_params(model.projection),
        params_total=n_params(model),
    )
    state_fn = bottleneck_state if isinstance(model, BottleneckAE) else mae_state

    def forward(frames):
        """The reconstruction loss of one fused window batch; the bottleneck
        model takes the Tokens with the per-point errors."""
        if isinstance(model, BottleneckAE):
            return model(fuse_tokens(frames)).loss
        return model(*fuse_frames(frames)).loss

    print(
        f"params: encoder {sizes['params_encoder'] / 1e6:.2f}M, "
        f"total {sizes['params_total'] / 1e6:.2f}M"
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = LambdaLR(opt, cosine_schedule(args.steps, args.warmup))
    step, epoch, elapsed, last_metrics = 0, 0, 0.0, None
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        step, epoch, elapsed = ckpt["step"], ckpt["epoch"], ckpt["elapsed"]
        last_metrics = ckpt.get("metrics")
    tracker = Tracker(
        args,
        config=dict(
            vars(args),
            **sizes,
            ladder_rungs=ladder.n_rungs,
            ladder_days=days,
            frames=str(cfg),
            n_train=len(train_kept),
            n_classes=len(data.classes),
        ),
        run_id=ckpt.get("wandb_id") if ckpt else None,
    )
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    log = JsonlLog(out / "log.jsonl")
    probe_tr = subset(train_kept, args.probe_train, args.seed)
    probe_va = subset(val_kept, args.probe_val, args.seed)

    def save(path, metrics, with_opt=True):
        state = state_fn(model, spec, cfg, ladder, data.classes, args, step, metrics)
        state.update(epoch=epoch, elapsed=elapsed, wandb_id=tracker.id)
        if with_opt:
            state.update(opt=opt.state_dict(), sched=sched.state_dict())
        save_atomic(state, path)

    @torch.no_grad()
    def evaluate(train_stats: dict) -> dict:
        t0 = time.time()
        model.eval()
        total, n = 0.0, 0
        for batch in val_loader:
            frames = [f.to(dev) for f in batch["frames"]]
            with amp:
                loss = forward(frames)
            rows = len(frames) * frames[0].values.shape[0]
            total += loss.item() * rows
            n += rows
        v_loss = total / max(n, 1)
        pr = probe(
            encoder,
            probe_tr,
            probe_va,
            cfg,
            spec,
            args.batch_size,
            dev,
            len(data.classes),
            seed=args.seed,
        )
        sh = shuffle_score(
            encoder,
            val_kept,
            cfg,
            spec,
            dev,
            n=args.shuffle_objects,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        model.train()
        m = dict(
            step=step,
            elapsed=elapsed,
            **train_stats,
            val_loss=v_loss,
            probe_acc=pr["acc"],
            probe_train_acc=pr["train_acc"],
            probe_macro_f1=pr["macro_f1"],
            probe_balanced_acc=pr["balanced_acc"],
            probe_majority_acc=pr["majority_acc"],
            logP_r2=pr["r2"],
            logP_r2_within=pr["r2_within"],
            logP_r2_by_superclass=pr["r2_by_superclass"],
            n_by_superclass=pr["n_by_superclass"],
            shuffle_score=sh,
            eval_seconds=time.time() - t0,
        )
        print(
            f"  eval @ {step}: val recon {v_loss:.4f} | {describe_probe(pr)} | "
            f"shuffle {sh:.3f} | {time.time() - t0:.0f}s",
            flush=True,
        )
        log.write(kind="eval", **m)
        tracker.log(
            {
                "val/loss": v_loss,
                "val/probe_acc": pr["acc"],
                "val/probe_train_acc": pr["train_acc"],
                "val/probe_macro_f1": pr["macro_f1"],
                "val/probe_balanced_acc": pr["balanced_acc"],
                "val/probe_majority_acc": pr["majority_acc"],
                "val/logP_r2": pr["r2"],
                "val/logP_r2_within": pr["r2_within"],
                "val/logP_r2_by_superclass": pr["r2_by_superclass"],
                "val/shuffle_score": sh,
                "val/eval_seconds": time.time() - t0,
                **{f"train_mean/{k}": v for k, v in train_stats.items()},
            },
            step=step,
        )
        return m

    if ckpt is None:
        t0 = time.time()
        bl = baseline_probe(
            probe_tr, probe_va, data.wavelengths.keys(), len(data.classes)
        )
        print(
            f"  hand-feature baseline (no model): {describe_probe(bl)} | "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
        log.write(kind="baseline", step=step, **bl)
        tracker.summary(
            **{f"baseline_{k}": v for k, v in bl.items() if isinstance(v, (int, float))}
        )
        if not args.no_eval_at_start:
            last_metrics = evaluate(dict(loss=float("nan")))

    model.train()
    timer = StepTimer(dev)
    total, n_acc, t_last, t_run, stop = 0.0, 0, time.time(), time.time(), False
    while step < args.steps and not stop:
        torch.manual_seed(args.seed + epoch)  # fresh window draws every epoch
        timer.reset()
        for batch in loader:
            timer.got_batch()
            frames = [f.to(dev) for f in batch["frames"]]
            with amp:
                loss_t = forward(frames)
            opt.zero_grad(set_to_none=True)
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            timer.done_step()
            step += 1
            total += loss_t.item()
            n_acc += 1
            if step % args.log_every == 0:
                now = time.time()
                rate, t_last = (now - t_last) / args.log_every, now
                loss = total / n_acc
                lr = sched.get_last_lr()[0]
                perf, gpu = timer.report(), gpu_stats(dev)
                timer.reset()
                print(
                    f"step {step:6d}  loss {loss:.4f}  lr {lr:.2e}  {rate:.3f} s/step"
                    f"  (data {perf['data_frac']:.0%}, gpu {gpu.get('gpu_util', float('nan')):.0f}%)",
                    flush=True,
                )
                log.write(
                    kind="train",
                    step=step,
                    loss=loss,
                    lr=lr,
                    s_per_step=rate,
                    **perf,
                    **gpu,
                )
                tracker.log(
                    {
                        "train/loss": loss,
                        "train/lr": lr,
                        "perf/s_per_step": rate,
                        "perf/data_s": perf["data_s"],
                        "perf/compute_s": perf["compute_s"],
                        "perf/data_frac": perf["data_frac"],
                        "perf/seq_per_s": args.batch_size / max(rate, 1e-9),
                        **{f"sys/{k}": v for k, v in gpu.items()},
                    },
                    step=step,
                )
            if step % args.eval_every == 0 or step == args.steps:
                loss = total / max(n_acc, 1)
                total, n_acc = 0.0, 0
                elapsed, t_run = elapsed + time.time() - t_run, time.time()
                last_metrics = evaluate(dict(loss=loss))
                t_run = time.time()
                timer.reset()
            budget = (
                bool(args.time_budget) and (time.time() - t_start) > args.time_budget
            )
            if step % args.ckpt_every == 0 or step == args.steps or budget:
                elapsed, t_run = elapsed + time.time() - t_run, time.time()
                save(last, last_metrics)
            if budget and step < args.steps:
                print(
                    f"time budget reached at step {step}; checkpoint saved", flush=True
                )
                stop = True
                break
            if step >= args.steps:
                break
        else:
            epoch += 1
    if step >= args.steps:
        save(out / "mae.pt", last_metrics, with_opt=False)
        (out / "DONE").write_text(f"{step} steps, {elapsed / 3600:.2f} h\n")
        print(f"done: {step} steps in {elapsed / 3600:.2f} h; saved {out / 'mae.pt'}")
        if last_metrics:
            tracker.summary(
                **{k: v for k, v in last_metrics.items() if isinstance(v, (int, float))}
            )
    tracker.finish()


if __name__ == "__main__":
    main()
