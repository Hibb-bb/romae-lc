"""Stage 2 (the predictor): LeWorldModel on light-curve windows, MSE path.

Stage 1 is the autoencoder (:mod:`project.pretrain_mae`, ``mae.pt``), stage
2 the predictor over its latents, stage 3 the decoder
(:mod:`project.train_decoder`). This script is stage 2 with the LeWorldModel
objective: step-based training of :class:`~romae_lc.LeWorldModel`
(next-window latent MSE plus SIGReg, no stop-gradient, no EMA) with the
design doc's light-curve translation: a frame is a ``--window``-day window,
the action the advance to the next window in window units. Every token
carries ``(m, log sigma)`` (drop the error channel with ``--no-err-channel``).

``--freeze-backbone`` (with ``--init-backbone mae.pt``) is the stage-2 MSE
ablation on frozen stage-1 latents: the encoder's parameters are excluded
from the optimizer and it stays in eval mode, the projector is an identity
(the latent the predictor sees is the frozen pooled CLS feature itself; a
trainable projector under the prediction loss alone would collapse), SIGReg
is dropped from the loss (``--lamb`` forced to 0; the values are still
logged) and ``--recon-weight`` is refused (nothing to train on the encoder
side). Why: trained jointly, the next-latent loss plus SIGReg erased the
period information the stage-1 autoencoder had learned (within-superclass
log-period R2 fell from 0.50 to 0.24 in 50k steps while the prediction loss
kept improving), because a static per-object latent is its easiest
solution. The main stage-2 path is the conditional flow predictor on cached
frozen latents (:mod:`project.train_predictor`); this flag is the MSE
ablation next to it.

Every evaluation prints the prediction loss next to two nulls computed on
the same validation batches: persistence (the next latent equals the current
one) and the history mean (the mean of the last ``--history`` latents), and
``ratio = val_pred / val_pred_persist``. A ratio near 1 means the predictor
learned nothing beyond persistence.

The rotary time ladder is dense by default: one distinct measured wavelength
per active time angle of every head of every encoder layer (``--ladder-mode
dense``, ``ladder.json`` in ``--out``), so the light encoder resolves 378
timescales and a ``--size wide`` one 756, against the 12 of a shared ladder
(``--ladder-mode shared``, or ``--rope-wavelengths ... --time-scale ...`` for
an explicit one); ``--time-frac`` gives time 56 of the 64 channels of a head.
The folding resolution a period needs is about 1 / (4 x cycles per window).

Sequences are 8 windows and the advance curriculum is on by default: the
start-to-start advance begins at ``--advance-start`` (0.25 to 0.5 window
lengths, so the next window starts inside the current one and its content is
readable locally) and ramps linearly to ``--advance`` over ``--advance-ramp``
of the steps (``--no-advance-curriculum`` turns it off).

Two ways to put a reconstruction demand on the encoder, the objective that
needs period, phase and shape in the latent where next-latent prediction is
satisfied by static per-object statistics: ``--init-backbone mae.pt`` starts
from a :mod:`project.pretrain_mae` checkpoint (its encoder, ladder and token
spec replace the ones the arguments would build) and ``--recon-weight w`` adds
``w`` times a masked reconstruction loss (:class:`~romae_lc.MaskedDecoder` on
the same backbone) to every step.

Resumable: ``--out/last.pt`` is written every ``--ckpt-every`` steps and when
``--time-budget`` seconds have elapsed; running the same command again
continues from it (model, data, frame and ladder arguments are then taken
from the checkpoint, only the run-control arguments from the command line).
``wm.pt`` and a ``DONE`` marker appear when ``--steps`` is reached.

Evaluations, at step 0 (the untrained encoder, the reference every later
line is measured against; ``--no-eval-at-start`` skips it) and every
``--eval-every`` steps: validation losses in eval mode (SIGReg also with the
projector's BatchNorm on batch statistics, ``val_sigreg_bn_train``, which
exposes a running-statistics gap), the linear class probe (accuracy, macro
F1, balanced accuracy and the majority-class accuracy it has to beat), ridge
R2 on log period within superclasses (the headline; ``ROT`` is printed
first), pooled and per superclass with validation counts, the time-shuffle
score and the surprise along window grids of validation objects. A probe on
hand features (shape, scatter and noise statistics, no model) is printed once
at the start as the bar to clear.

With ``--wandb`` the same numbers go to Weights & Biases (``train/loss``,
``train/pred_loss`` for the predictor, ``train/sigreg_loss`` for the encoder
regulariser, ``train/recon_loss``, ``val/...``, ``baseline_...`` in the run
summary, ``perf/...`` data-wait and compute time per step, ``sys/gpu_util``
and memory), one run per ``--out`` across resumed links.

    python -m project.train_wm --out project/runs/wm_w250 --window 250 --wandb
    python -m project.train_wm --out project/runs/wm_w250_mae --window 250 \\
        --init-backbone project/runs/mae_w250/mae.pt --wandb
    python -m project.train_wm --out project/runs/wm_w250_frozen --window 250 \\
        --init-backbone project/runs/mae_w250/mae.pt --freeze-backbone --wandb
    python -m project.train_wm --data sim --n-sim 128 --steps 20 --width 48 \\
        --depth 1 --heads 2 --pred-depth 1 --pred-heads 2 --pred-dim-head 16 \\
        --pred-mlp 64 --proj-hidden 64 --n-slices 32 --window 30 --min-tokens 4 \\
        --batch-size 8 --eval-every 10 --probe-train 64 --probe-val 32 \\
        --shuffle-objects 16 --surprise-objects 4 --device cpu --out /tmp/wm_smoke
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from romae_lc import MaskedDecoder, RoMAE

from project.common import (
    MODEL_KEYS,
    JsonlLog,
    Ladder,
    TokenSpec,
    add_data_args,
    add_device_arg,
    add_frame_args,
    add_ladder_args,
    add_model_args,
    baseline_probe,
    batch_stats,
    build_backbone,
    build_world_model,
    cosine_schedule,
    describe_probe,
    dump_json,
    err_stats,
    frame_config,
    frame_loader,
    fuse_frames,
    get_device,
    get_ladder,
    load_data,
    load_mae,
    n_params,
    probe,
    resolve_model_args,
    rope_geometry,
    save_atomic,
    seed_all,
    subset,
    to_device,
    wm_state,
)
from project.diagnostics import shuffle_score, surprise_summary
from project.tracking import StepTimer, Tracker, add_wandb_args, gpu_stats

#: Arguments read from the command line even when resuming a checkpoint.
RUN_CONTROL = (
    "advance_start",
    "advance_ramp",
    "no_advance_curriculum",
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
        "--no-eval-at-start",
        action="store_true",
        help="skip the step-0 evaluation of the untrained encoder",
    )
    g.add_argument(
        "--advance-start",
        type=float,
        nargs=2,
        default=(0.25, 0.5),
        metavar=("LO", "HI"),
        help="curriculum: the advance range at step 0, ramped linearly to --advance "
        "over the first --advance-ramp of the steps (small advances make the next "
        "window start inside the current one, so its content is readable locally "
        "before extrapolation is required)",
    )
    g.add_argument("--advance-ramp", type=float, default=0.6, help="fraction of steps")
    g.add_argument(
        "--no-advance-curriculum",
        action="store_true",
        help="train at --advance from step 0",
    )
    g.add_argument(
        "--init-backbone",
        default=None,
        metavar="MAE_PT",
        help="start from the encoder of a project.pretrain_mae checkpoint (its "
        "ladder, token spec and model size replace the ones given here)",
    )
    g.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="stage-2 MSE ablation on frozen stage-1 latents: needs --init-backbone; "
        "the encoder is excluded from the optimizer and kept in eval mode, the "
        "projector is an identity, SIGReg is dropped from the loss and "
        "--recon-weight is refused (a model argument: remembered on resume)",
    )
    g.add_argument(
        "--recon-weight",
        type=float,
        default=0.0,
        help="weight of a masked reconstruction loss on the backbone (0 = off)",
    )
    g.add_argument("--recon-mask-ratio", type=float, default=0.5)
    g.add_argument("--recon-width", type=int, default=192, help="recon decoder width")
    g.add_argument("--recon-depth", type=int, default=2)
    g.add_argument("--recon-heads", type=int, default=3)
    add_device_arg(g)


def prediction_baselines(z: torch.Tensor, history: int) -> tuple[float, float]:
    """Null prediction losses on latents ``z [B, T, D]``, in float32, with
    the shape of the model's next-latent MSE: persistence (``z[:, t]`` for
    ``z[:, t + 1]``) and the history mean (the mean of the last
    ``min(history, t + 1)`` latents up to ``t``), each a mean over
    ``(b, t < T - 1, d)``."""
    z = z.detach().float()
    if z.shape[1] < 2:
        return float("nan"), float("nan")
    target = z[:, 1:]
    persist = (z[:, :-1] - target).square().mean().item()
    means = torch.stack(
        [z[:, max(0, t + 1 - history) : t + 1].mean(1) for t in range(z.shape[1] - 1)],
        dim=1,
    )
    histmean = (means - target).square().mean().item()
    return persist, histmean


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
    if args.no_advance_curriculum:
        args.advance_start = None
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
    if args.freeze_backbone:
        if not args.init_backbone:
            parser.error(
                "--freeze-backbone needs --init-backbone MAE_PT (a stage-1 encoder "
                "to freeze)"
            )
        if args.recon_weight > 0:
            parser.error(
                "--recon-weight with --freeze-backbone: nothing to train on the "
                "encoder side"
            )
        if args.lamb != 0.0:
            print(
                f"--freeze-backbone: SIGReg dropped from the loss (--lamb {args.lamb} "
                "-> 0): it regularises the encoder, which is frozen; the values are "
                "still logged"
            )
            args.lamb = 0.0
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
    init = None
    if ckpt is not None:
        ladder = Ladder.from_dict(ckpt["ladder"])
        spec = TokenSpec.from_dict(ckpt["spec"])
    elif args.init_backbone:
        init, meta = load_mae(args.init_backbone)
        ladder, spec = meta.ladder, meta.spec
        for k in MODEL_KEYS:
            if k in meta.args:
                setattr(args, k, meta.args[k])
        if abs(float(meta.cfg.window) - float(cfg.window)) > 1e-9:
            print(
                f"warning: the pretrained encoder saw {meta.cfg.window} d windows, "
                f"this run uses {cfg.window} d"
            )
        print(
            f"encoder, ladder and token spec from {args.init_backbone} "
            f"(step {meta.step}, {ladder.summary or 'ladder'})"
        )
        dump_json(ladder.to_dict(), out / "ladder.json")
    else:
        t0 = time.time()
        ladder = get_ladder(args, train, out, rope_geometry(args))
        print(ladder.summary or "ladder reused", f"({time.time() - t0:.0f}s)")
        stats = None if args.no_err_channel else err_stats(train)
        spec = TokenSpec(
            dict(band_wavelengths=data.wavelengths, time_scale=ladder.time_scale), stats
        )
    dump_json(vars(args), out / "args.json")
    days = ladder.flat
    print(
        f"ladder: {ladder.n_rungs} distinct wavelengths from {days[0]:.4g} to "
        f"{days[-1]:.4g} d, {ladder.per_head} per head, {ladder.layers} layer "
        f"ladder(s) x {ladder.heads} head ladder(s), deal {ladder.deal}"
    )
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

    if ckpt is not None:
        backbone = RoMAE(**ckpt["backbone"])
    elif init is not None:
        backbone = init.backbone("cls")
    else:
        backbone = build_backbone(args, ladder, spec.n_channels)
    # Identity projector on a frozen encoder: the latent is the frozen feature.
    projector = torch.nn.Identity() if args.freeze_backbone else None
    model = build_world_model(args, backbone, args.n_frames, projector=projector)
    model = model.to(dev)
    if args.freeze_backbone:
        model.backbone.requires_grad_(False)
    recon = None
    if args.recon_weight > 0:
        recon = MaskedDecoder(
            backbone,
            decoder=dict(
                d_model=args.recon_width,
                nhead=args.recon_heads,
                depth=args.recon_depth,
                attention=args.attention,
            ),
            mask_ratio=args.recon_mask_ratio,
            target_channels=1,
        ).to(dev)
    params = [p for p in model.parameters() if p.requires_grad] + (
        list(recon.parameters()) if recon is not None else []
    )
    sizes = dict(
        params_backbone=n_params(backbone),
        params_predictor=n_params(model.predictor),
        params_total=n_params(model),
        params_recon=n_params(recon) if recon is not None else 0,
        params_trainable=sum(p.numel() for p in params),
    )
    print(
        f"params: backbone {sizes['params_backbone'] / 1e6:.2f}M"
        + (" (frozen)" if args.freeze_backbone else "")
        + f", predictor {sizes['params_predictor'] / 1e6:.2f}M, total "
        f"{sizes['params_total'] / 1e6:.2f}M"
        + (f", recon head {sizes['params_recon'] / 1e6:.2f}M" if recon else "")
        + f", trainable {sizes['params_trainable'] / 1e6:.2f}M"
    )
    if args.compile:
        model.backbone.forward = torch.compile(model.backbone.forward, dynamic=True)

    def set_train() -> None:
        """Training mode, except a frozen backbone, which stays in eval mode
        (no dropout) so its latents are the same ones the probes see."""
        model.train()
        if recon is not None:
            recon.train()
        if args.freeze_backbone:
            model.backbone.eval()

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = LambdaLR(opt, cosine_schedule(args.steps, args.warmup))
    step, epoch, elapsed, last_metrics = 0, 0, 0.0, None
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"])
        if recon is not None and ckpt.get("recon"):
            recon.load_state_dict(ckpt["recon"]["state_dict"])
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
        extra = {}
        if recon is not None:
            extra["recon"] = dict(
                hparams=dict(
                    decoder=asdict(recon.dec_cfg),
                    mask_ratio=recon.mask_ratio,
                    target_channels=recon.target_channels,
                ),
                state_dict=recon.state_dict(),
            )
        state = wm_state(
            model, spec, cfg, ladder, data.classes, args, step, metrics, **extra
        )
        state.update(epoch=epoch, elapsed=elapsed, wandb_id=tracker.id)
        if with_opt:
            state.update(opt=opt.state_dict(), sched=sched.state_dict())
        save_atomic(state, path)

    @torch.no_grad()
    def evaluate(train_stats: dict) -> dict:
        t0 = time.time()
        model.eval()
        if recon is not None:
            recon.eval()
        sums, n = np.zeros(6), 0
        for batch in val_loader:
            frames, actions = to_device(batch, dev)
            with amp:
                o = model(frames, actions)
                with batch_stats(model.projector):
                    o_bn = model(frames, actions)
                rec = (
                    recon(model.backbone, *fuse_frames(frames)).loss.item()
                    if recon is not None
                    else 0.0
                )
            persist, histmean = prediction_baselines(o.embedding, model.history)
            b = actions.shape[0]
            sums += b * np.array(
                [
                    o.pred_loss.item(),
                    o.sigreg_loss.item(),
                    o_bn.sigreg_loss.item(),
                    rec,
                    persist,
                    histmean,
                ]
            )
            n += b
        v_pred, v_sig, v_sig_bn, v_rec, v_persist, v_hist = sums / max(n, 1)
        v_ratio = v_pred / max(v_persist, 1e-12)
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
        set_train()
        m = dict(
            step=step,
            elapsed=elapsed,
            **train_stats,
            val_pred=v_pred,
            val_pred_persist=v_persist,
            val_pred_histmean=v_hist,
            val_pred_ratio=v_ratio,
            val_sigreg=v_sig,
            val_sigreg_bn_train=v_sig_bn,
            val_recon=v_rec if recon is not None else None,
            val_batch=args.batch_size,
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
            surprise=su,
            eval_seconds=time.time() - t0,
        )
        print(
            f"  eval @ {step}: val pred {v_pred:.4f} (persist {v_persist:.4f}, "
            f"hist-mean {v_hist:.4f}, ratio {v_ratio:.2f}) sigreg {v_sig:.4f} "
            f"(bn-train {v_sig_bn:.4f})"
            + (f" recon {v_rec:.4f}" if recon is not None else "")
            + f" | {describe_probe(pr)} | shuffle {sh:.3f} | surprise {su['mean']:.4f} "
            f"({su['frac_valid']:.2f} valid) | {time.time() - t0:.0f}s",
            flush=True,
        )
        log.write(kind="eval", **m)
        tracker.log(
            {
                "val/pred_loss": v_pred,
                "val/pred_persist": v_persist,
                "val/pred_histmean": v_hist,
                "val/pred_ratio": v_ratio,
                "val/sigreg_loss": v_sig,
                "val/sigreg_bn_train": v_sig_bn,
                "val/loss": v_pred + args.lamb * v_sig,
                "val/recon_loss": v_rec if recon is not None else None,
                "val/probe_acc": pr["acc"],
                "val/probe_train_acc": pr["train_acc"],
                "val/probe_macro_f1": pr["macro_f1"],
                "val/probe_balanced_acc": pr["balanced_acc"],
                "val/probe_majority_acc": pr["majority_acc"],
                "val/logP_r2": pr["r2"],
                "val/logP_r2_within": pr["r2_within"],
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
            nan = float("nan")
            last_metrics = evaluate(dict(loss=nan, pred=nan, sigreg=nan, straight=nan))

    set_train()
    timer = StepTimer(dev)
    sums, n_acc, t_last, t_run, stop = np.zeros(5), 0, time.time(), time.time(), False
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
                loss = o.loss
                rec = None
                if recon is not None:
                    rec = recon(model.backbone, *fuse_frames(frames)).loss
                    loss = loss + args.recon_weight * rec
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            sched.step()
            timer.done_step()
            step += 1
            sums += [
                o.loss.item(),
                o.pred_loss.item(),
                o.sigreg_loss.item(),
                float(np.nan_to_num(o.straightness.item())),
                rec.item() if rec is not None else 0.0,
            ]
            n_acc += 1
            if step % args.log_every == 0:
                now = time.time()
                rate, t_last = (now - t_last) / args.log_every, now
                loss_m, pred, sig, st, rc = sums / n_acc
                lr = sched.get_last_lr()[0]
                perf, gpu = timer.report(), gpu_stats(dev)
                timer.reset()
                print(
                    f"step {step:6d}  loss {loss_m:.4f}  pred {pred:.4f}  sigreg {sig:.4f}"
                    f"  straight {st:.3f}"
                    + (f"  recon {rc:.4f}" if recon is not None else "")
                    + f"  lr {lr:.2e}  {rate:.3f} s/step"
                    f"  (data {perf['data_frac']:.0%}, gpu {gpu.get('gpu_util', float('nan')):.0f}%)",
                    flush=True,
                )
                log.write(
                    kind="train",
                    step=step,
                    loss=loss_m,
                    pred=pred,
                    sigreg=sig,
                    straight=st,
                    recon=rc if recon is not None else None,
                    lr=lr,
                    s_per_step=rate,
                    **perf,
                    **gpu,
                )
                tracker.log(
                    {
                        "train/loss": loss_m,
                        "train/pred_loss": pred,
                        "train/sigreg_loss": sig,
                        "train/straightness": st,
                        "train/recon_loss": rc if recon is not None else None,
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
                loss_m, pred, sig, st, rc = sums / max(n_acc, 1)
                sums[:], n_acc = 0, 0
                elapsed, t_run = elapsed + time.time() - t_run, time.time()
                stats = dict(loss=loss_m, pred=pred, sigreg=sig, straight=st)
                if recon is not None:
                    stats["recon"] = rc
                last_metrics = evaluate(stats)
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
