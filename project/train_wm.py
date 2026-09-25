"""Stage 1 (M1): LeWorldModel on light-curve windows with the error channel.

Step-based training of :class:`~romae_lc.LeWorldModel` (next-window latent
prediction plus SIGReg, no stop-gradient, no EMA) with the design doc's
light-curve translation: a frame is a ``--window``-day window, the action
the advance to the next window in window units. Every token carries
``(m, log sigma)`` (drop the error channel with ``--no-err-channel``). The
rotary time ladder is measured on the training curves once per run
(``ladder.json`` in ``--out``) with ``lam_max = 2 * window``, or given
explicitly (``--rope-wavelengths ... --time-scale ...``).

Resumable: ``--out/last.pt`` is written every ``--ckpt-every`` steps and when
``--time-budget`` seconds have elapsed; running the same command again
continues from it (model, data, frame and ladder arguments are then taken
from the checkpoint, only the run-control arguments from the command line).
``wm.pt`` and a ``DONE`` marker appear when ``--steps`` is reached. Every
``--eval-every`` steps: validation losses in eval mode, a linear probe on the
fine class and ridge R2 on log period (frozen features), the time-shuffle
score and the surprise along window grids of validation objects.

With ``--wandb`` the same numbers go to Weights & Biases (``train/loss``,
``train/pred_loss`` for the predictor, ``train/sigreg_loss`` for the encoder
regulariser, ``val/...``, ``perf/...`` data-wait and compute time per step,
``sys/gpu_util`` and memory), one run per ``--out`` across resumed links.

    python -m project.train_wm --out project/runs/wm_w500 --window 500 --wandb
    python -m project.train_wm --data sim --n-sim 128 --steps 20 --width 48 \\
        --depth 1 --heads 2 --pred-depth 1 --pred-heads 2 --pred-dim-head 16 \\
        --pred-mlp 64 --proj-hidden 64 --n-slices 32 --window 30 --min-tokens 4 \\
        --batch-size 8 --eval-every 10 --probe-train 64 --probe-val 32 \\
        --shuffle-objects 16 --surprise-objects 4 --device cpu --out /tmp/wm_smoke
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from project.common import (
    JsonlLog,
    Ladder,
    TokenSpec,
    add_data_args,
    add_device_arg,
    add_frame_args,
    add_ladder_args,
    add_model_args,
    build_backbone,
    build_world_model,
    cosine_schedule,
    dump_json,
    err_stats,
    frame_config,
    frame_loader,
    get_device,
    get_ladder,
    load_data,
    n_params,
    probe,
    save_atomic,
    seed_all,
    subset,
    time_block_dim,
    to_device,
    wm_state,
)
from project.diagnostics import shuffle_score, surprise_summary
from project.tracking import StepTimer, Tracker, add_wandb_args, gpu_stats

#: Arguments read from the command line even when resuming a checkpoint.
RUN_CONTROL = (
    "advance_start",
    "advance_ramp",
    "steps",
    "time_budget",
    "workers",
    "device",
    "eval_every",
    "ckpt_every",
    "log_every",
    "out",
    "no_resume",
    "probe_train",
    "probe_val",
    "val_objects",
    "shuffle_objects",
    "surprise_objects",
    "probe_latent",
    "persistent_workers",
    "prefetch",
    "pin_memory",
    "compile",
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
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--lr", type=float, default=5e-5)
    g.add_argument("--wd", type=float, default=1e-3)
    g.add_argument("--warmup", type=float, default=0.01, help="fraction of steps")
    g.add_argument("--clip", type=float, default=1.0)
    g.add_argument("--eval-every", type=int, default=2500)
    g.add_argument("--ckpt-every", type=int, default=1000)
    g.add_argument("--log-every", type=int, default=100)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--persistent-workers", action="store_true")
    g.add_argument("--prefetch", type=int, default=2, help="batches per worker")
    g.add_argument("--pin-memory", action="store_true")
    g.add_argument("--compile", action="store_true", help="torch.compile the encoder")
    g.add_argument("--out", default="project/runs/wm")
    g.add_argument(
        "--no-resume", action="store_true", help="ignore an existing last.pt"
    )
    g.add_argument(
        "--time-budget", type=float, default=0.0, help="stop and checkpoint after s"
    )
    g.add_argument("--probe-train", type=int, default=4000, help="probe fit records")
    g.add_argument("--probe-val", type=int, default=2000)
    g.add_argument("--val-objects", type=int, default=2000, help="for val losses")
    g.add_argument("--shuffle-objects", type=int, default=512)
    g.add_argument("--surprise-objects", type=int, default=128)
    g.add_argument("--probe-latent", action="store_true", help="probe z, not features")
    g.add_argument(
        "--advance-start",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="curriculum: the advance range at step 0, ramped linearly to --advance "
        "over the first --advance-ramp of the steps (small advances make the next "
        "window start inside the current one, so its phase is readable locally "
        "before extrapolation is required)",
    )
    g.add_argument("--advance-ramp", type=float, default=0.6, help="fraction of steps")
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
    dump_json(vars(args), out / "args.json")
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
        dim = time_block_dim(args.width, args.heads, args.p_rope)
        t0 = time.time()
        ladder = get_ladder(args, train, out, dim, args.p_rope)
        print(ladder.summary or "ladder reused", f"({time.time() - t0:.0f}s)")
        stats = None if args.no_err_channel else err_stats(train)
        spec = TokenSpec(
            dict(band_wavelengths=data.wavelengths, time_scale=ladder.time_scale), stats
        )
    print(f"ladder (days): {np.round(ladder.wavelengths, 4).tolist()}")
    print(f"error channel stats (mu, sd of log sigma): {spec.err_stats}")

    if args.advance_start and args.persistent_workers:
        print("advance curriculum needs workers re-forked every epoch: persistent off")
        args.persistent_workers = False
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
        f"{len(loader)} steps per epoch, {cfg}, {dev}"
    )

    backbone = build_backbone(args, ladder, spec.n_channels)
    model = build_world_model(args, backbone, args.n_frames).to(dev)
    sizes = dict(
        params_backbone=n_params(backbone),
        params_predictor=n_params(model.predictor),
        params_total=n_params(model),
    )
    print(
        f"params: backbone {sizes['params_backbone'] / 1e6:.2f}M, predictor "
        f"{sizes['params_predictor'] / 1e6:.2f}M, total {sizes['params_total'] / 1e6:.2f}M"
    )
    if args.compile:
        model.backbone.forward = torch.compile(model.backbone.forward, dynamic=True)
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
            ladder_days=ladder.wavelengths,
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
        state = wm_state(model, spec, cfg, ladder, data.classes, args, step, metrics)
        state.update(epoch=epoch, elapsed=elapsed, wandb_id=tracker.id)
        if with_opt:
            state.update(opt=opt.state_dict(), sched=sched.state_dict())
        save_atomic(state, path)

    @torch.no_grad()
    def evaluate(train_stats: dict) -> dict:
        t0 = time.time()
        model.eval()
        sums, n = np.zeros(2), 0
        for batch in val_loader:
            frames, actions = to_device(batch, dev)
            with amp:
                o = model(frames, actions)
            b = actions.shape[0]
            sums += b * np.array([o.pred_loss.item(), o.sigreg_loss.item()])
            n += b
        v_pred, v_sig = sums / max(n, 1)
        pr = probe(
            model,
            probe_tr,
            probe_va,
            cfg,
            spec,
            args.batch_size,
            dev,
            len(data.classes),
            project=args.probe_latent,
            seed=args.seed,
        )
        sh = shuffle_score(
            model,
            val_kept,
            cfg,
            spec,
            dev,
            n=args.shuffle_objects,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        su = surprise_summary(
            model, val_kept, cfg, spec, dev, n=args.surprise_objects, seed=args.seed
        )
        model.train()
        m = dict(
            step=step,
            elapsed=elapsed,
            **train_stats,
            val_pred=v_pred,
            val_sigreg=v_sig,
            val_batch=args.batch_size,
            probe_acc=pr["acc"],
            probe_train_acc=pr["train_acc"],
            logP_r2=pr["r2"],
            logP_r2_by_superclass=pr["r2_by_superclass"],
            shuffle_score=sh,
            surprise=su,
            eval_seconds=time.time() - t0,
        )
        print(
            f"  eval @ {step}: val pred {v_pred:.4f} sigreg {v_sig:.4f} | probe acc "
            f"{pr['acc']:.3f} (train {pr['train_acc']:.3f}) logP R2 {pr['r2']:.3f} | "
            f"shuffle {sh:.3f} | surprise {su['mean']:.4f} ({su['frac_valid']:.2f} valid) "
            f"| R2 by class {({k: round(v, 2) for k, v in pr['r2_by_superclass'].items()})} "
            f"| {time.time() - t0:.0f}s",
            flush=True,
        )
        log.write(kind="eval", **m)
        tracker.log(
            {
                "val/pred_loss": v_pred,
                "val/sigreg_loss": v_sig,
                "val/loss": v_pred + args.lamb * v_sig,
                "val/probe_acc": pr["acc"],
                "val/probe_train_acc": pr["train_acc"],
                "val/logP_r2": pr["r2"],
                "val/logP_r2_by_superclass": pr["r2_by_superclass"],
                "val/shuffle_score": sh,
                "val/surprise_mean": su["mean"],
                "val/surprise_median": su["median"],
                "val/surprise_frac_valid": su["frac_valid"],
                "val/surprise_by_superclass": su["per_superclass"],
                "val/eval_seconds": time.time() - t0,
                **{f"train_mean/{k}": v for k, v in train_stats.items()},
            },
            step=step,
        )
        return m

    def current_advance(at_step: int) -> tuple[float, float]:
        """The curriculum's advance range at ``at_step`` (``cfg.advance`` at the
        end of the ramp and without ``--advance-start``)."""
        if not args.advance_start:
            return tuple(cfg.advance)
        t = min(1.0, at_step / max(1.0, args.advance_ramp * args.steps))
        return tuple(
            float(a + t * (b - a)) for a, b in zip(args.advance_start, cfg.advance)
        )

    model.train()
    timer = StepTimer(dev)
    sums, n_acc, t_last, t_run, stop = np.zeros(4), 0, time.time(), time.time(), False
    while step < args.steps and not stop:
        torch.manual_seed(args.seed + epoch)  # fresh window draws every epoch
        adv = current_advance(step)
        loader.dataset.cfg = replace(cfg, advance=adv)  # records were kept for cfg
        if args.advance_start and epoch % 5 == 0:
            print(
                f"epoch {epoch}: advance range {adv[0]:.2f} - {adv[1]:.2f}", flush=True
            )
        timer.reset()
        for batch in loader:
            timer.got_batch()
            frames, actions = to_device(batch, dev)
            with amp:
                o = model(frames, actions)
            opt.zero_grad(set_to_none=True)
            o.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            timer.done_step()
            step += 1
            sums += [
                o.loss.item(),
                o.pred_loss.item(),
                o.sigreg_loss.item(),
                float(np.nan_to_num(o.straightness.item())),
            ]
            n_acc += 1
            if step % args.log_every == 0:
                now = time.time()
                rate, t_last = (now - t_last) / args.log_every, now
                loss, pred, sig, st = sums / n_acc
                lr = sched.get_last_lr()[0]
                perf, gpu = timer.report(), gpu_stats(dev)
                timer.reset()
                print(
                    f"step {step:6d}  loss {loss:.4f}  pred {pred:.4f}  sigreg {sig:.4f}"
                    f"  straight {st:.3f}  lr {lr:.2e}  {rate:.3f} s/step"
                    f"  (data {perf['data_frac']:.0%}, gpu {gpu.get('gpu_util', float('nan')):.0f}%)",
                    flush=True,
                )
                log.write(
                    kind="train",
                    step=step,
                    loss=loss,
                    pred=pred,
                    sigreg=sig,
                    straight=st,
                    lr=lr,
                    s_per_step=rate,
                    **perf,
                    **gpu,
                )
                tracker.log(
                    {
                        "train/loss": loss,
                        "train/pred_loss": pred,
                        "train/sigreg_loss": sig,
                        "train/straightness": st,
                        "train/advance_lo": current_advance(step)[0],
                        "train/advance_hi": current_advance(step)[1],
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
                loss, pred, sig, st = sums / max(n_acc, 1)
                sums[:], n_acc = 0, 0
                elapsed, t_run = elapsed + time.time() - t_run, time.time()
                last_metrics = evaluate(
                    dict(loss=loss, pred=pred, sigreg=sig, straight=st)
                )
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
        save(out / "wm.pt", last_metrics, with_opt=False)
        (out / "DONE").write_text(f"{step} steps, {elapsed / 3600:.2f} h\n")
        print(f"done: {step} steps in {elapsed / 3600:.2f} h; saved {out / 'wm.pt'}")
        if last_metrics:
            tracker.summary(
                **{k: v for k, v in last_metrics.items() if isinstance(v, (int, float))}
            )
    tracker.finish()


if __name__ == "__main__":
    main()
