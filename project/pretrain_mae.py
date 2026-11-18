"""Pretraining of the window encoder: the token-level JEPA (``--jepa``), or the
plain masked autoencoder without it.

Every window of a frame sequence is a sample. Part of its tokens are hidden
(``--mask-mode``, ``--mask-ratio``, ``--forecast-tail``). The JEPA predicts the
latents of the hidden tokens (:mod:`project.jepa`); the plain masked
autoencoder (:class:`~romae_lc.RoMAEForPreTraining`) predicts their
magnitudes. ``--jepa-no-ema`` trains the JEPA on views of a window with SIGReg
and no EMA target (:func:`project.jepa.forward_views`). ``--spectral-aux``
adds the cross-entropy of the spectral layer against the catalogue period (a
training target only).

A run resumes from ``last.pt`` in ``--out``. Model and data flags then come
from the checkpoint; only the flags in ``RUN_CONTROL`` are read from the
command line. ``--time-budget`` stops and saves. ``mae.pt`` and ``DONE``
appear when ``--steps`` is reached.

Validation, at step 0 and every ``--eval-every`` steps: the validation loss
with random points hidden and with stretches hidden, the linear class probe
and the log-period ridge (:func:`project.common.probe`), and the time-shuffle
score. The hand-feature baseline probe runs once at the start.

    python -m project.pretrain_mae --jepa --spectral --spectral-aux 0.5 --size wide \\
        --window 250 --out project/runs/NAME --wandb
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
from romae_lc.model import gen_mask
from romae_lc.spectral import spectral_aux_loss

from project.jepa import TokenJEPA, forward_views, jepa_state
from project.mae_data import MASK_MODES, make_mask, phase_band_mask
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
    shuffle_score,
    subset,
)
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


#: Options of older code that this script no longer has, with the value that means "off". A checkpoint
#: saved with one of them switched on cannot be resumed here (use the tag ``pre-cleanup``).
REMOVED_OPTIONS = dict(
    bottleneck=False, encoder="transformer", target="obs", input="obs", phase_loss=0.0, fold_loss=0.0, dyn=0,
    cls_inv=0.0, cls_sig=0.0, attn_pool=0, abs_time=False, phasor=0, window_range=None, mask_ratio_range=None,
    loss_weight="none", query_extra=0, jepa_clean_target=False,
)  # fmt: skip


def check_resumable(ckpt: dict) -> None:
    """Stop when ``ckpt`` was trained with an option this script no longer has."""
    saved = ckpt["args"]
    on = [k for k, off in REMOVED_OPTIONS.items() if saved.get(k, off) != off]
    if saved.get("mask_mode") == "mix":
        on.append("mask_mode=mix")
    if ckpt.get("kind") not in (None, "jepa"):
        on.append(f"kind={ckpt['kind']}")
    if on:
        raise SystemExit(f"this checkpoint uses options that moved to legacy/ ({', '.join(on)}); check out the tag pre-cleanup to resume it")


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
    g.add_argument("--forecast-tail", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="forecast mode of the masked JEPA: every training window also hides ALL its tokens in the last H days, "
                        "H uniform in [LO, HI] per window, so the predictor must forecast past the context's end (the target encoder "
                        "sees the whole window, so targets and queries share one clock); the hidden count per row stays fixed")
    g.add_argument("--forecast-share", type=float, default=1.0, help="--forecast-tail: share of the windows that get a tail")
    g.add_argument("--phase-band", type=float, nargs=2, default=(0.1, 0.3), metavar=("LO", "HI"),
                   help="--mask-mode phaseband: width range, in cycles, of the hidden phase band (catalogue period, training only)")
    g.add_argument("--dec-width", type=int, default=192, help="MAE decoder width")
    g.add_argument("--dec-depth", type=int, default=2)
    g.add_argument("--dec-heads", type=int, default=3)
    g.add_argument(
        "--mask-mode",
        choices=MASK_MODES,
        default="random",
        help="what is hidden: random points, block = whole stretches of time "
        "(the pattern must be carried across a gap, which needs the period), "
        "blockplus = one stretch plus random points, phaseband = a band of phases",
    )
    g.add_argument(
        "--mask-blocks",
        type=int,
        nargs=2,
        default=(1, 6),
        metavar=("LO", "HI"),
        help="stretches per window (half of a 250 d window in one stretch is a "
        "125 d gap, in six about 20 d each)",
    )
    g.add_argument("--mask-block-pos", choices=("random", "last"), default="random",
                   help="where a hidden stretch sits: anywhere, or at the end of the input (a hidden last window)")  # fmt: skip
    g.add_argument("--mask-block-share", type=float, default=0.5,
                   help="blockplus: share of the hidden points that form the stretch (the rest are random points)")  # fmt: skip
    g.add_argument("--spectral", action="store_true", help="the spectral layer (a learned periodogram over the tokens) in the encoder")
    g.add_argument("--spectral-rel", type=float, default=5e-4, help="relative frequency step of its grid")
    g.add_argument("--spectral-channels", type=int, default=8)
    g.add_argument("--spectral-reader", type=int, default=32)
    g.add_argument("--spectral-depth", type=int, default=3)
    g.add_argument("--spectral-after", type=int, default=2, help="encoder block after which the layer runs")
    g.add_argument("--spectral-p", type=float, nargs=2, default=(0.02, 500.0), metavar=("PMIN", "PMAX"), help="period range of the grid, days")
    g.add_argument(
        "--spectral-aux",
        type=float,
        default=0.0,
        help="weight of the auxiliary cross-entropy of the layer's logits against the catalogue period's bin (target only); 0 = off",
    )
    g.add_argument("--phase-freq", choices=["soft", "peak", "local"], default="soft", help="how the phase rope reads the frequency off the spectral logits: the softmax expectation, the argmax bin, or the centroid within 5 bins of the argmax")
    g.add_argument("--phase-rope", type=int, default=0, help="the phase at the spectral layer's own frequency as a rotary axis of the decoder, this many harmonics (0 = off; needs --spectral)")
    g.add_argument("--jepa", action="store_true", help="token-level JEPA (project.jepa): predict the EMA target encoder's latents of the hidden tokens instead of their values")
    g.add_argument("--jepa-ema", type=float, default=0.996, help="target encoder momentum at the start")
    g.add_argument("--jepa-ema-end", type=float, default=1.0, help="momentum at the end (linear schedule)")
    g.add_argument("--jepa-loss", choices=["smoothl1", "mse"], default="smoothl1")
    g.add_argument("--jepa-use-context", action="store_true", help="use the context encoder downstream (default: the EMA target encoder)")
    g.add_argument("--jepa-recon-weight", type=float, default=0.0, help="hybrid loss: this weight times the MSE of a brightness head on the predictor's outputs")
    g.add_argument("--jepa-proj", type=int, default=0, help="alignment loss in a projection of this width (0 = the encoder space); the probes read the pre-projection latent")
    g.add_argument("--jepa-proj-hidden", type=int, default=256)
    # LeJEPA at the token level (project.jepa.forward_views): no EMA, SIGReg on the projected token latents,
    # 2 global + 4 local views per window with point and band dropout
    g.add_argument("--jepa-no-ema", action="store_true", help="one encoder for both sides, no EMA, SIGReg against collapse, multi-view training")
    g.add_argument("--sig-lambda", type=float, default=0.05, help="--jepa-no-ema: loss = (1 - lambda) prediction + lambda SIGReg (+ spectral aux)")
    g.add_argument("--sig-slices", type=int, default=1024, help="--jepa-no-ema: SIGReg random projections")
    g.add_argument("--views", type=int, nargs=2, default=(2, 4), metavar=("GLOBAL", "LOCAL"), help="--jepa-no-ema: views per window")
    g.add_argument("--view-keep", type=float, default=0.15, help="--jepa-no-ema: share of the window's tokens in each view")
    g.add_argument("--local-span", type=float, nargs=2, default=(30.0, 80.0), metavar=("LO", "HI"), help="--jepa-no-ema: local view length, days")
    g.add_argument("--band-drop", type=float, default=0.2, help="--jepa-no-ema: chance that a view drops one band")
    g.add_argument("--jepa-init", default=None, help="warm start the JEPA encoder (and the target copy) from this masked-autoencoder mae.pt")
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


def forecast_tail_mask(mask, t_days, pad, window, h_range, share=1.0):
    """``--forecast-tail``: hide every real token in the last ``H`` days of each window (``H`` uniform in
    ``h_range`` per row; a ``share`` of the rows), keeping the same hidden count in every row, as the JEPA
    forward needs. Per row the hidden set is the tail first, then the tokens ``mask`` already hid, then random
    others (padding first), up to ``k``, the largest union of tail and ``mask`` over the rows. Returns ``(mask, share of hidden real tokens
    that are tail)``."""
    b, n = mask.shape
    lo, hi = h_range
    h = lo + (hi - lo) * torch.rand(b, device=mask.device)
    on = torch.rand(b, device=mask.device) < share
    tail = (~pad) & (t_days > (float(window) - h)[:, None]) & on[:, None]
    k = min(int((mask | tail).sum(1).max()), n - 1)
    # priority: the tail, then what mask hid, then padding (filler that costs no real token), then other real tokens
    score = torch.rand(b, n, device=mask.device) + 1.0 * pad.float() + 2.0 * mask.float() + 4.0 * tail.float()
    idx = score.topk(k, dim=1).indices
    out = torch.zeros_like(mask).scatter_(1, idx, True)
    hid_real = (out & ~pad).sum().clamp(min=1)
    return out, float((out & tail).sum() / hid_real)


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
        check_resumable(ckpt)
        # flags of older code that this parser no longer has are set too, so they stay in args.json and in the checkpoint
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
    spectral_kw = dict(time_scale=ladder.time_scale, p_min=args.spectral_p[0], p_max=args.spectral_p[1], rel=args.spectral_rel,
                       channels=args.spectral_channels, reader=args.spectral_reader, depth=args.spectral_depth,
                       after_layer=args.spectral_after) if args.spectral else None  # fmt: skip
    if ckpt is not None and ckpt.get("kind") == "jepa":
        model = TokenJEPA.from_checkpoint(ckpt)
    elif ckpt is not None:
        model = RoMAEForPreTraining(**ckpt["mae"], **ckpt["backbone"])
    elif args.jepa:
        model = TokenJEPA(
            decoder=decoder, mask_ratio=args.mask_ratio, encoder=encoder_config(args), n_channels=spec.n_channels, n_axes=2,
            rope=rope_layouts(args, ladder), spectral=spectral_kw,
            ema=args.jepa_ema, ema_end=args.jepa_ema_end, loss=args.jepa_loss, use_target=not args.jepa_use_context, recon_weight=args.jepa_recon_weight,
            proj_dim=args.jepa_proj, proj_hidden=args.jepa_proj_hidden,
            phase_rope=args.phase_rope, phase_freq=args.phase_freq, no_ema=args.jepa_no_ema, sig_slices=args.sig_slices,
        )  # fmt: skip
        if args.jepa_no_ema and (args.jepa_proj <= 0 or args.jepa_recon_weight > 0 or args.forecast_tail is not None):
            parser.error("--jepa-no-ema needs --jepa-proj > 0 and no reconstruction or forecast tail")
        if args.jepa_init:
            model.init_from_mae(torch.load(args.jepa_init, map_location="cpu", weights_only=False))
    else:
        model = RoMAEForPreTraining(
            decoder=decoder,
            mask_ratio=args.mask_ratio,
            target_channels=1,
            encoder=encoder_config(args),
            n_channels=spec.n_channels,
            n_axes=2,
            rope=rope_layouts(args, ladder),
            spectral=spectral_kw,
            phase_rope=args.phase_rope,
            phase_freq=args.phase_freq,
        )
    if args.phase_rope and not args.spectral:
        parser.error("--phase-rope needs --spectral")
    model = model.to(dev)
    encoder = FrameEncoder(PooledEncoder(model))
    sizes = dict(
        params_encoder=n_params(model.transformer) + n_params(model.projection),
        params_total=n_params(model),
    )
    state_fn = jepa_state if isinstance(model, TokenJEPA) else mae_state

    # the spectral aux loss needs the catalogue period of every training row (a training target only)
    aux = args.spectral_aux > 0
    if aux and not args.spectral:
        parser.error("--spectral-aux needs --spectral")
    if aux and spec.err_stats is None:
        parser.error("--spectral-aux needs the error channel")
    train_periods = torch.tensor([float(r.period or float("nan")) for r in train], dtype=torch.float64)
    aux_stats = {"spectral": []}

    def forward(frames, mode=None, periods_rows=None):
        """The loss of one fused window batch. ``mode`` is the mask mode
        (default: the run's). With ``periods_rows`` (the catalogue period of
        every fused row) the spectral aux loss is added."""
        values, positions, pad = fuse_frames(frames)
        mode = args.mask_mode if mode is None else mode
        mask = None  # None: the model draws random points at its own mask ratio (--mask-ratio)
        if mode == "phaseband":
            if periods_rows is None:
                raise ValueError("--mask-mode phaseband needs the period of every row (--spectral-aux on)")
            t_days_all = (positions[:, 0].float() - 1.0) * spec.tokenize["time_scale"]
            mask = phase_band_mask(t_days_all, pad, periods_rows.to(pad.device), args.mask_ratio, args.phase_band)
        elif mode != "random":
            mask = make_mask(
                positions[:, 0], pad, args.mask_ratio, mode, args.mask_blocks,
                position=args.mask_block_pos, block_share=args.mask_block_share,
            )  # fmt: skip
        if args.forecast_tail is not None and model.training:
            t_days_all = (positions[:, 0].float() - 1.0) * spec.tokenize["time_scale"]
            if mask is None:
                mask = gen_mask(args.mask_ratio, pad)
            mask, tail_share = forecast_tail_mask(mask, t_days_all, pad, args.window, args.forecast_tail, args.forecast_share)
            aux_stats.setdefault("forecast_tail_share", []).append(tail_share)
        loss = model(values, positions, pad, mask).loss
        if aux and periods_rows is not None:
            l_aux = spectral_aux_loss(model.spectral, periods_rows.to(dev))
            aux_stats["spectral"].append(float(l_aux))
            loss = loss + args.spectral_aux * l_aux
        return loss

    print(
        f"params: encoder {sizes['params_encoder'] / 1e6:.2f}M, "
        f"total {sizes['params_total'] / 1e6:.2f}M"
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
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
        state.update(epoch=epoch, elapsed=elapsed, wandb_id=tracker.id, aux={})  # aux: always empty now; the key stays for old readers
        if with_opt:
            state.update(opt=opt.state_dict(), sched=sched.state_dict())
        save_atomic(state, path)

    @torch.no_grad()
    def evaluate(train_stats: dict) -> dict:
        t0 = time.time()
        model.eval()
        total, total_b, n = 0.0, 0.0, 0
        for batch in val_loader:
            frames = [f.to(dev) for f in batch["frames"]]
            with amp:
                # random points hidden: the same task in every run, so the
                # number compares across runs; stretches hidden: the harder task
                loss = forward(frames, "random")
                loss_b = forward(frames, "block")
            rows = len(frames) * frames[0].values.shape[0]
            total += loss.item() * rows
            total_b += loss_b.item() * rows
            n += rows
        v_loss = total / max(n, 1)
        v_loss_block = total_b / max(n, 1)
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
            val_loss_block=v_loss_block,
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
            f"  eval @ {step}: val recon {v_loss:.4f} (stretches hidden {v_loss_block:.4f}) | {describe_probe(pr)} | "
            f"shuffle {sh:.3f} | {time.time() - t0:.0f}s",
            flush=True,
        )
        log.write(kind="eval", **m)
        tracker.log(
            {
                "val/loss": v_loss,
                "val/loss_block": v_loss_block,
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
            periods_rows = train_periods[batch["index"]].repeat(len(frames)) if aux else None
            with amp:
                if args.jepa_no_ema:
                    values_, positions_, pad_ = fuse_frames(frames)
                    l_pred, l_sig, l_aux, st_ = forward_views(
                        model, values_, positions_, pad_, periods_rows.to(dev) if periods_rows is not None else None, args.views[0], args.views[1],
                        args.view_keep, args.local_span, args.band_drop, args.window)  # fmt: skip
                    loss_t = (1 - args.sig_lambda) * l_pred + args.sig_lambda * l_sig
                    if l_aux is not None and args.spectral_aux > 0:
                        loss_t = loss_t + args.spectral_aux * l_aux
                        aux_stats["spectral"].append(float(l_aux.detach()))
                    aux_stats.setdefault("lejepa_pred", []).append(float(l_pred.detach()))
                    aux_stats.setdefault("lejepa_sig", []).append(float(l_sig.detach()))
                    for k_, v_ in st_.items():
                        aux_stats.setdefault(f"lejepa_{k_}", []).append(v_)
                else:
                    loss_t = forward(frames, periods_rows=periods_rows)
            opt.zero_grad(set_to_none=True)
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            if isinstance(model, TokenJEPA):
                model.ema_update(step, args.steps)
                for k_, v_ in model.last_stats.items():
                    aux_stats.setdefault(f"jepa_{k_}", []).append(v_)
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
                    **{k: float(np.mean(v)) for k, v in aux_stats.items() if v},  # the auxiliary terms so far in this eval interval
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
                extra = {k: float(np.mean(v)) for k, v in aux_stats.items() if v}
                for v in aux_stats.values():
                    v.clear()
                last_metrics = evaluate(dict(loss=loss, **extra))
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
