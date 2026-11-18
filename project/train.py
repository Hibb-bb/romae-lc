"""Pretraining of the window encoder: the token-level JEPA (``--jepa``), or the
plain masked autoencoder without it.

Every window of a frame sequence is a sample. Part of its tokens are hidden
(``--mask-mode``, ``--mask-ratio``, ``--forecast-tail``; :mod:`project.masks`).
The JEPA predicts the latents of the hidden tokens (:mod:`project.jepa`); the
plain masked autoencoder (:class:`~romae_lc.RoMAEForPreTraining`) predicts
their magnitudes. ``--jepa-no-ema`` trains the JEPA on views of a window with
SIGReg and no EMA target (:func:`project.jepa.forward_views`).
``--spectral-aux`` adds the cross-entropy of the spectral layer against the
catalogue period (a training target only).

The flags of a run come from a recipe file (``--recipe``,
:mod:`project.recipe`), and any flag on the command line beats the recipe.

A run resumes from ``last.pt`` in ``--out``. Model and data flags then come
from the checkpoint; only the flags in ``RUN_CONTROL`` are read from the
command line. ``--time-budget`` stops and saves. ``mae.pt`` and ``DONE``
appear when ``--steps`` is reached.

Validation, at step 0 and every ``--eval-every`` steps: the validation loss
with random points hidden and with stretches hidden, the linear class probe
and the log-period ridge (:func:`project.probes.probe`), and the time-shuffle
score. The hand-feature baseline probe runs once at the start.

This file reads top-down: flags, data, model, loop, validation, checkpoint,
and ``main`` at the end ties them together.

    python -m project.train --recipe project/recipes/enc_o3_ph.toml --out project/runs/NAME --wandb
    python -m project.train --data sim --n-sim 128 --steps 20 --width 48 \\
        --depth 1 --heads 2 --window 30 --min-tokens 4 --batch-size 8 \\
        --eval-every 10 --probe-train 64 --probe-val 32 --shuffle-objects 16 \\
        --device cpu --out /tmp/mae_smoke
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from romae_lc import RoMAEForPreTraining
from romae_lc.model import gen_mask
from romae_lc.spectral import spectral_aux_loss

from project.backbone import (
    Ladder,
    add_ladder_args,
    add_model_args,
    encoder_config,
    get_ladder,
    n_params,
    resolve_model_args,
    rope_geometry,
    rope_layouts,
)
from project.checkpoint import FrameEncoder, PooledEncoder, jepa_state, mae_state, save_atomic
from project.data import add_data_args, load_data, subset
from project.jepa import TokenJEPA, forward_views
from project.masks import MASK_MODES, forecast_tail_mask, make_mask, phase_band_mask
from project.probes import baseline_probe, describe_probe, probe, shuffle_score
from project.recipe import apply_recipe, recipe_text
from project.tracking import StepTimer, Tracker, add_wandb_args, gpu_stats
from project.utils import JsonlLog, add_device_arg, cosine_schedule, dump_json, get_device, seed_all
from project.windows import TokenSpec, add_frame_args, err_stats, frame_config, frame_loader, fuse_frames

# ============================================================================================ flags

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


def build_parser() -> argparse.ArgumentParser:
    """Every flag of the training script (``--recipe`` is added by :func:`project.recipe.apply_recipe`)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser)
    add_frame_args(parser)
    add_ladder_args(parser)
    add_model_args(parser)
    add_train_args(parser)
    add_wandb_args(parser)
    return parser


def read_flags(argv=None) -> tuple[argparse.Namespace, dict | None, Callable]:
    """``(args, ckpt, fail)``. ``ckpt`` is ``last.pt`` of ``--out`` when the run resumes, else None. On a
    resume the checkpoint's flags replace the given ones, except the ``RUN_CONTROL`` flags. ``fail(message)``
    stops the script like a bad flag does."""
    parser = build_parser()
    args = apply_recipe(parser, argv)
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
        args.recipe = ckpt["args"].get("recipe")  # the recipe of the run is the one it started with
        print(f"resuming {last} at step {ckpt['step']}")
    return args, ckpt, parser.error


# ============================================================================================= data


def build_tokens(args, ckpt, data, out: Path) -> tuple[Ladder, TokenSpec]:
    """The time ladder and the token spec: the checkpoint's on a resume, else measured on the train records."""
    if ckpt is not None:
        ladder = Ladder.from_dict(ckpt["ladder"])
        spec = TokenSpec.from_dict(ckpt["spec"])
    else:
        t0 = time.time()
        ladder = get_ladder(args, data["train"], out, rope_geometry(args))
        print(ladder.summary or "ladder reused", f"({time.time() - t0:.0f}s)")
        stats = None if args.no_err_channel else err_stats(data["train"])
        spec = TokenSpec(dict(band_wavelengths=data.wavelengths, time_scale=ladder.time_scale), stats)
    return ladder, spec


def build_loaders(args, data, cfg, spec):
    """``(train loader, validation loader, train records kept, validation records kept)``. The train loader
    draws new windows of the TRAIN records every epoch. The validation loader holds one fixed window sequence
    of ``--val-objects`` VALIDATION records. "Kept" records are long enough to host a sequence."""
    train, val = data["train"], data["validation"]
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
    return loader, val_loader, train_kept, val_kept


# ============================================================================================ model


def build_model(args, ckpt, spec: TokenSpec, ladder: Ladder, fail: Callable) -> torch.nn.Module:
    """The model on the CPU: rebuilt from the checkpoint on a resume (weights are loaded later, in
    :func:`restore`), else a new :class:`TokenJEPA` (``--jepa``) or a plain masked autoencoder."""
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
            fail("--jepa-no-ema needs --jepa-proj > 0 and no reconstruction or forecast tail")
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
        fail("--phase-rope needs --spectral")
    return model


def check_spectral_aux(args, spec: TokenSpec, fail: Callable) -> None:
    if args.spectral_aux > 0 and not args.spectral:
        fail("--spectral-aux needs --spectral")
    if args.spectral_aux > 0 and spec.err_stats is None:
        fail("--spectral-aux needs the error channel")


# ============================================================================================= loop


@dataclass
class Run:
    """Everything one training run holds. The functions below read it and update its counters."""

    args: argparse.Namespace
    out: Path
    dev: torch.device
    t_start: float  # wall-clock start, for --time-budget
    data: object  # project.data.Data
    cfg: object  # FrameConfig
    spec: TokenSpec
    ladder: Ladder
    recipe: str | None  # the recipe text, stored in the checkpoint
    model: torch.nn.Module
    encoder: FrameEncoder  # the model's pooled encoder, as the probes call it
    loader: object  # train windows
    val_loader: object  # validation windows
    val_kept: list  # validation records of val_loader
    probe_tr: list  # train records the probes are fitted on
    probe_va: list  # validation records the probes are scored on
    train_periods: torch.Tensor  # catalogue period per train record (a training target only)
    opt: torch.optim.Optimizer
    sched: LambdaLR
    tracker: Tracker
    log: JsonlLog
    amp: torch.autocast
    step: int = 0
    epoch: int = 0
    elapsed: float = 0.0  # training seconds so far, over all job links
    last_metrics: dict | None = None
    aux_stats: dict = field(default_factory=lambda: {"spectral": []})  # per-step extra terms since the last evaluation

    @property
    def aux(self) -> bool:
        """The spectral aux loss is on."""
        return self.args.spectral_aux > 0


def window_loss(run: Run, frames, mode=None, periods_rows=None) -> torch.Tensor:
    """The masked loss of one fused window batch. ``mode`` is the mask mode
    (default: the run's). With ``periods_rows`` (the catalogue period of
    every fused row) the spectral aux loss is added."""
    args, model, spec = run.args, run.model, run.spec
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
        run.aux_stats.setdefault("forecast_tail_share", []).append(tail_share)
    loss = model(values, positions, pad, mask).loss
    if run.aux and periods_rows is not None:
        l_aux = spectral_aux_loss(model.spectral, periods_rows.to(run.dev))
        run.aux_stats["spectral"].append(float(l_aux))
        loss = loss + args.spectral_aux * l_aux
    return loss


def views_loss(run: Run, frames, periods_rows=None) -> torch.Tensor:
    """The loss of the no-EMA recipe (``--jepa-no-ema``): prediction between views plus SIGReg (plus spectral aux)."""
    args, aux_stats = run.args, run.aux_stats
    values_, positions_, pad_ = fuse_frames(frames)
    l_pred, l_sig, l_aux, st_ = forward_views(
        run.model, values_, positions_, pad_, periods_rows.to(run.dev) if periods_rows is not None else None, args.views[0], args.views[1],
        args.view_keep, args.local_span, args.band_drop, args.window)  # fmt: skip
    loss_t = (1 - args.sig_lambda) * l_pred + args.sig_lambda * l_sig
    if l_aux is not None and args.spectral_aux > 0:
        loss_t = loss_t + args.spectral_aux * l_aux
        aux_stats["spectral"].append(float(l_aux.detach()))
    aux_stats.setdefault("lejepa_pred", []).append(float(l_pred.detach()))
    aux_stats.setdefault("lejepa_sig", []).append(float(l_sig.detach()))
    for k_, v_ in st_.items():
        aux_stats.setdefault(f"lejepa_{k_}", []).append(v_)
    return loss_t


def train_step(run: Run, batch) -> torch.Tensor:
    """One optimizer step on one batch. Returns the loss."""
    args, model = run.args, run.model
    frames = [f.to(run.dev) for f in batch["frames"]]
    periods_rows = run.train_periods[batch["index"]].repeat(len(frames)) if run.aux else None
    with run.amp:
        if args.jepa_no_ema:
            loss_t = views_loss(run, frames, periods_rows)
        else:
            loss_t = window_loss(run, frames, periods_rows=periods_rows)
    run.opt.zero_grad(set_to_none=True)
    loss_t.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    run.opt.step()
    run.sched.step()
    if isinstance(model, TokenJEPA):
        model.ema_update(run.step, args.steps)
        for k_, v_ in model.last_stats.items():
            run.aux_stats.setdefault(f"jepa_{k_}", []).append(v_)
    return loss_t


def log_train(run: Run, loss: float, rate: float, timer: StepTimer) -> None:
    """One ``train`` row: the mean loss since the last evaluation, the learning rate and the speed."""
    lr = run.sched.get_last_lr()[0]
    perf, gpu = timer.report(), gpu_stats(run.dev)
    timer.reset()
    print(
        f"step {run.step:6d}  loss {loss:.4f}  lr {lr:.2e}  {rate:.3f} s/step"
        f"  (data {perf['data_frac']:.0%}, gpu {gpu.get('gpu_util', float('nan')):.0f}%)",
        flush=True,
    )
    run.log.write(
        kind="train",
        step=run.step,
        loss=loss,
        lr=lr,
        s_per_step=rate,
        **perf,
        **gpu,
        **{k: float(np.mean(v)) for k, v in run.aux_stats.items() if v},  # the auxiliary terms so far in this eval interval
    )
    run.tracker.log(
        {
            "train/loss": loss,
            "train/lr": lr,
            "perf/s_per_step": rate,
            "perf/data_s": perf["data_s"],
            "perf/compute_s": perf["compute_s"],
            "perf/data_frac": perf["data_frac"],
            "perf/seq_per_s": run.args.batch_size / max(rate, 1e-9),
            **{f"sys/{k}": v for k, v in gpu.items()},
        },
        step=run.step,
    )


def train_loop(run: Run) -> None:
    """Train until ``--steps`` or until the time budget is used up. Every ``--log-every`` steps a train row,
    every ``--eval-every`` steps the validation (:func:`evaluate`), every ``--ckpt-every`` steps ``last.pt``."""
    args = run.args
    last = run.out / "last.pt"
    run.model.train()
    timer = StepTimer(run.dev)
    total, n_acc, t_last, t_run, stop = 0.0, 0, time.time(), time.time(), False
    while run.step < args.steps and not stop:
        torch.manual_seed(args.seed + run.epoch)  # fresh window draws every epoch
        timer.reset()
        for batch in run.loader:
            timer.got_batch()
            loss_t = train_step(run, batch)
            timer.done_step()
            run.step += 1
            total += loss_t.item()
            n_acc += 1
            if run.step % args.log_every == 0:
                now = time.time()
                rate, t_last = (now - t_last) / args.log_every, now
                log_train(run, total / n_acc, rate, timer)
            if run.step % args.eval_every == 0 or run.step == args.steps:
                loss = total / max(n_acc, 1)
                total, n_acc = 0.0, 0
                run.elapsed, t_run = run.elapsed + time.time() - t_run, time.time()
                extra = {k: float(np.mean(v)) for k, v in run.aux_stats.items() if v}
                for v in run.aux_stats.values():
                    v.clear()
                run.last_metrics = evaluate(run, dict(loss=loss, **extra))
                t_run = time.time()
                timer.reset()
            budget = bool(args.time_budget) and (time.time() - run.t_start) > args.time_budget
            if run.step % args.ckpt_every == 0 or run.step == args.steps or budget:
                run.elapsed, t_run = run.elapsed + time.time() - t_run, time.time()
                save(run, last)
            if budget and run.step < args.steps:
                print(f"time budget reached at step {run.step}; checkpoint saved", flush=True)
                stop = True
                break
            if run.step >= args.steps:
                break
        else:
            run.epoch += 1


# ======================================================================================= validation


@torch.no_grad()
def validation_losses(run: Run) -> tuple[float, float]:
    """The loss on the validation windows, per row: with random points hidden (the same task in every run,
    so the number compares across runs) and with stretches hidden (the harder task)."""
    total, total_b, n = 0.0, 0.0, 0
    for batch in run.val_loader:
        frames = [f.to(run.dev) for f in batch["frames"]]
        with run.amp:
            loss = window_loss(run, frames, "random")
            loss_b = window_loss(run, frames, "block")
        rows = len(frames) * frames[0].values.shape[0]
        total += loss.item() * rows
        total_b += loss_b.item() * rows
        n += rows
    return total / max(n, 1), total_b / max(n, 1)


@torch.no_grad()
def evaluate(run: Run, train_stats: dict) -> dict:
    """The periodic validation: the two validation losses, the class and period probes (fitted on train
    records, scored on validation records) and the time-shuffle score. Writes an ``eval`` row and returns it.
    ``train_stats`` are the training means since the last evaluation; they go into the same row."""
    args, model = run.args, run.model
    t0 = time.time()
    model.eval()
    v_loss, v_loss_block = validation_losses(run)
    pr = probe(
        run.encoder,
        run.probe_tr,
        run.probe_va,
        run.cfg,
        run.spec,
        args.batch_size,
        run.dev,
        len(run.data.classes),
        seed=args.seed,
    )
    sh = shuffle_score(
        run.encoder,
        run.val_kept,
        run.cfg,
        run.spec,
        run.dev,
        n=args.shuffle_objects,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model.train()
    m = dict(
        step=run.step,
        elapsed=run.elapsed,
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
        f"  eval @ {run.step}: val recon {v_loss:.4f} (stretches hidden {v_loss_block:.4f}) | {describe_probe(pr)} | "
        f"shuffle {sh:.3f} | {time.time() - t0:.0f}s",
        flush=True,
    )
    run.log.write(kind="eval", **m)
    run.tracker.log(
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
        step=run.step,
    )
    return m


def baseline(run: Run) -> None:
    """The hand-feature probe (no model), once at the start of a run: the bar the encoder's probe has to clear."""
    t0 = time.time()
    bl = baseline_probe(run.probe_tr, run.probe_va, run.data.wavelengths.keys(), len(run.data.classes))
    print(
        f"  hand-feature baseline (no model): {describe_probe(bl)} | "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )
    run.log.write(kind="baseline", step=run.step, **bl)
    run.tracker.summary(**{f"baseline_{k}": v for k, v in bl.items() if isinstance(v, (int, float))})


# ======================================================================================= checkpoint


def save(run: Run, path: Path, with_opt: bool = True) -> None:
    """Write the checkpoint (:mod:`project.checkpoint`) plus the run's counters, the wandb id and the recipe
    text. ``last.pt`` also holds the optimizer and the schedule, so the run can resume."""
    state_fn = jepa_state if isinstance(run.model, TokenJEPA) else mae_state
    state = state_fn(run.model, run.spec, run.cfg, run.ladder, run.data.classes, run.args, run.step, run.last_metrics)
    state.update(epoch=run.epoch, elapsed=run.elapsed, wandb_id=run.tracker.id, aux={})  # aux: always empty now; the key stays for old readers
    state.update(recipe=run.recipe)
    if with_opt:
        state.update(opt=run.opt.state_dict(), sched=run.sched.state_dict())
    save_atomic(state, path)


def restore(ckpt: dict, model, opt, sched) -> None:
    """Load the weights, the optimizer and the schedule of ``last.pt``."""
    model.load_state_dict(ckpt["state_dict"])
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])


def finish(run: Run) -> None:
    """At ``--steps``: write ``mae.pt`` (no optimizer) and the ``DONE`` file."""
    out = run.out
    save(run, out / "mae.pt", with_opt=False)
    (out / "DONE").write_text(f"{run.step} steps, {run.elapsed / 3600:.2f} h\n")
    print(f"done: {run.step} steps in {run.elapsed / 3600:.2f} h; saved {out / 'mae.pt'}")
    if run.last_metrics:
        run.tracker.summary(**{k: v for k, v in run.last_metrics.items() if isinstance(v, (int, float))})


# ============================================================================================= main


def main(argv=None):
    # flags
    args, ckpt, fail = read_flags(argv)
    out = Path(args.out)
    seed_all(args.seed)
    dev = get_device(args)
    t_start = time.time()

    # data
    data = load_data(args)
    train, val = data["train"], data["validation"]
    print(
        f"{len(train)} train / {len(val)} val records, {len(data.classes)} classes, "
        f"loaded in {time.time() - t_start:.0f}s"
    )
    cfg = frame_config(args)
    ladder, spec = build_tokens(args, ckpt, data, out)
    dump_json(vars(args), out / "args.json")
    days = ladder.flat
    print(
        f"ladder: {ladder.n_rungs} distinct wavelengths from {days[0]:.4g} to "
        f"{days[-1]:.4g} d, {ladder.per_head} per head, {ladder.layers} layer "
        f"ladder(s) x {ladder.heads} head ladder(s), deal {ladder.deal}"
    )
    loader, val_loader, train_kept, val_kept = build_loaders(args, data, cfg, spec)
    print(
        f"{len(train_kept)} train / {len(val_kept)} val sequences, "
        f"{len(loader)} steps per epoch of {cfg.n_frames} windows each, {cfg}, {dev}"
    )

    # model
    model = build_model(args, ckpt, spec, ladder, fail).to(dev)
    sizes = dict(
        params_encoder=n_params(model.transformer) + n_params(model.projection),
        params_total=n_params(model),
    )
    check_spectral_aux(args, spec, fail)
    print(
        f"params: encoder {sizes['params_encoder'] / 1e6:.2f}M, "
        f"total {sizes['params_total'] / 1e6:.2f}M"
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    sched = LambdaLR(opt, cosine_schedule(args.steps, args.warmup))
    if ckpt is not None:
        restore(ckpt, model, opt, sched)

    # the run: everything the loop, the validation and the checkpoint need
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
    run = Run(
        args=args, out=out, dev=dev, t_start=t_start, data=data, cfg=cfg, spec=spec, ladder=ladder,
        recipe=ckpt.get("recipe") if ckpt else (recipe_text(args.recipe) if args.recipe else None),
        model=model, encoder=FrameEncoder(PooledEncoder(model)), loader=loader, val_loader=val_loader, val_kept=val_kept,
        probe_tr=subset(train_kept, args.probe_train, args.seed), probe_va=subset(val_kept, args.probe_val, args.seed),
        # the spectral aux loss needs the catalogue period of every training row (a training target only)
        train_periods=torch.tensor([float(r.period or float("nan")) for r in train], dtype=torch.float64),
        opt=opt, sched=sched, tracker=tracker, log=JsonlLog(out / "log.jsonl"),
        amp=torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"),
    )  # fmt: skip
    if ckpt is not None:
        run.step, run.epoch, run.elapsed = ckpt["step"], ckpt["epoch"], ckpt["elapsed"]
        run.last_metrics = ckpt.get("metrics")

    # validation at the start of a new run, then the loop
    if ckpt is None:
        baseline(run)
        if not args.no_eval_at_start:
            run.last_metrics = evaluate(run, dict(loss=float("nan")))
    train_loop(run)
    if run.step >= args.steps:
        finish(run)
    tracker.finish()


if __name__ == "__main__":
    main()
