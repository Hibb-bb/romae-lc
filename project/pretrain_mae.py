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
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from romae_lc import RoMAEForPreTraining

from project.bottleneck import BottleneckAE, bottleneck_state, fuse_tokens
from project.gruode import GRUODEForPreTraining, gruode_state
from project.jepa import TokenJEPA, jepa_state
from project.mae_data import MASK_MODES, MultiWindowDataset, make_mask
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
    g.add_argument("--forecast-tail", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="forecast mode of the masked JEPA: every training window also hides ALL its tokens in the last H days, "
                        "H uniform in [LO, HI] per window, so the predictor must forecast past the context's end (the target encoder "
                        "sees the whole window, so targets and queries share one clock); the hidden count per row stays fixed")
    g.add_argument("--forecast-share", type=float, default=1.0, help="--forecast-tail: share of the windows that get a tail")
    g.add_argument("--mask-ratio-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="draw the mask ratio of every training step from [LO, HI] (one ratio per batch); overrides --mask-ratio in training")
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
        "mix = a stretch row with probability --mask-block-prob",
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
    g.add_argument("--mask-block-prob", type=float, default=0.5)
    g.add_argument("--mask-block-pos", choices=("random", "last"), default="random",
                   help="where a hidden stretch sits: anywhere, or at the end of the input (a hidden last window)")  # fmt: skip
    g.add_argument("--mask-block-share", type=float, default=0.5,
                   help="blockplus: share of the hidden points that form the stretch (the rest are random points)")  # fmt: skip
    g.add_argument(
        "--loss-weight",
        choices=("none", "err"),
        default="none",
        help="err = every hidden point's squared error divided by its sigma^2 (weights "
        "capped at --loss-weight-cap times the window's median, mean 1 per window)",
    )
    g.add_argument("--loss-weight-cap", type=float, default=20.0)
    g.add_argument(
        "--phase-loss",
        type=float,
        default=0.0,
        help="weight of the per-token PHASE objective: a head on every visible token's encoder output "
        "predicts (cos, sin) of the point's phase from the window's start on the catalogue period "
        "(the period is a training target only); 0 = off",
    )
    g.add_argument(
        "--fold-loss",
        type=float,
        default=0.0,
        help="weight of the phase-folded reconstruction objective: a phase decoder reads the CLS latent only "
        "and predicts the window's observed points at their phase (window start, catalogue period) and "
        "band, Gaussian likelihood under the known errors; the model is asked to phase-fold; 0 = off",
    )
    g.add_argument("--fold-harm", type=int, default=8, help="phase harmonics of the fold decoder")
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
    # the dynamic-state add-on (project.dyn): a per-token state advanced by a rotation at the spectral layer's own
    # frequency, with a same-window contrastive term and SIGReg in time; label-free
    g.add_argument("--dyn", type=int, default=0, help="rotation blocks K of the dynamic-state add-on (needs --spectral); 0 = off")
    g.add_argument("--dyn-weight", type=float, default=1.0, help="weight of the add-on's loss in the total")
    g.add_argument("--dyn-nce", type=float, default=0.2, help="weight of the same-window contrastive term")
    g.add_argument("--dyn-sig", type=float, default=1.0, help="weight of SIGReg on the states centred per window")
    g.add_argument("--dyn-mean", type=float, default=1.0, help="weight of the squared per-window mean state")
    g.add_argument("--dyn-tau", type=float, default=1.0, help="temperature of the contrastive logits (squared distances)")
    g.add_argument("--dyn-pairs", type=int, default=128, help="source-target pairs per window and step")
    g.add_argument("--dyn-dt", type=float, nargs=2, default=(0.5, 60.0), metavar=("MIN", "MAX"),
                   help="gap range in days; the largest gap ramps geometrically from MIN to MAX over --dyn-ramp of the steps")  # fmt: skip
    g.add_argument("--dyn-ramp", type=float, default=0.6)
    g.add_argument("--dyn-freq", choices=["soft", "peak", "local"], default="local", help="how the clock speed is read off the spectral logits")
    # LeJEPA's static part on the CLS (the "LeNEPA" variant): the CLS of a star's windows pulled together, SIGReg across the batch
    g.add_argument("--cls-inv", type=float, default=0.0, help="weight of the variance of the CLS across a star's windows in the batch; 0 = off")
    g.add_argument("--cls-sig", type=float, default=0.0, help="weight of SIGReg (LeJEPA) on the CLS across the batch; 0 = off")
    g.add_argument("--attn-pool", type=int, default=0,
                   help="learned queries of an attention-pooling head over the token outputs (romae_lc.model.AttnPool); it replaces the CLS "
                        "as the window vector: --cls-inv / --cls-sig act on it, and the latents and probes read it; 0 = the CLS")  # fmt: skip
    g.add_argument("--dyn-lambda", type=float, default=None,
                   help="LeJEPA's weighting: (1 - lambda) (prediction + dyn-nce x contrastive) + lambda SIGReg (LeJEPA's default 0.05); "
                        "unset = the old weights (prediction 1, SIGReg --dyn-sig)")  # fmt: skip
    g.add_argument("--dyn-phase-pos", type=float, default=None, metavar="TOL",
                   help="--dyn-cross: tokens of the star (outside the source window) within TOL cycles of the target's phase also count "
                        "as matches in the contrastive term (training-time catalogue period); unset = only the target token")  # fmt: skip
    g.add_argument("--dyn-ham", action="store_true",
                   help="Hamiltonian predictor (project.dyn.HamHead, after LeHami): exact harmonic rotation + a learned "
                        "symplectic term, no damping; advances the fraction of the elapsed cycles; needs --dyn-cross")  # fmt: skip
    g.add_argument("--dyn-ham-steps", type=int, default=16, help="--dyn-ham: substeps per cycle")
    g.add_argument("--dyn-ham-close", type=float, default=1.0, help="--dyn-ham: weight of the one-cycle closure loss")
    g.add_argument("--dyn-cross", action="store_true",
                   help="pairs across a star's windows (source in one window, target in another; negatives = every token of the star): "
                        "a clock built from the time since the window start cannot pass this test, only the phase can")  # fmt: skip
    # ablations and the JEPA-style target (11-03)
    g.add_argument("--dyn-target", choices=["context", "ema"], default="context",
                   help="--dyn-cross: where the target states and negatives come from: the context encoder's visible tokens of the same "
                        "masked forward, with gradient on both sides (J3), or the JEPA target encoder on the full window through the "
                        "state head with no gradient (a JEPA target)")  # fmt: skip
    g.add_argument("--dyn-pred", type=float, default=1.0, help="weight of the add-on's prediction loss when --dyn-lambda is unset (ablation: 0)")
    g.add_argument("--dyn-no-rot", action="store_true", help="ablation: the frequency is set to 0, so the predictor only damps (no clock)")
    g.add_argument("--dyn-decode", type=float, default=0.0,
                   help="--dyn-cross: weight of a magnitude decoder on [predicted state, source CLS], Huber on the z-score against "
                        "the target's observed magnitude and error; 0 = off")  # fmt: skip
    g.add_argument("--dyn-sig-floor", type=float, default=0.05, help="--dyn-decode: floor on the error (scaled magnitude units)")
    g.add_argument("--dyn-dec-band", action="store_true", help="--dyn-decode: the decoder is told the target's band")
    g.add_argument("--dyn-dec-nll", action="store_true",
                   help="--dyn-decode: Gaussian likelihood with a learned model spread (variance sigma_obs^2 + sigma_model^2) instead of Huber on z")
    g.add_argument("--dyn-sig-mode", choices=["star", "batch"], default="star",
                   help="--dyn-cross: SIGReg on the states centred per star (default) or on the raw states pooled over stars")
    g.add_argument("--dyn-freeze", action="store_true",
                   help="freeze the encoder (needs --init or a resume): only the state head and the predictor train, on the full window "
                        "without masking and without the reconstruction loss; tells whether the phase is already in the frozen tokens")  # fmt: skip
    g.add_argument(
        "--abs-time",
        action="store_true",
        help="absolute time features in the tokens (NeRF-style sines and cosines of the time "
        "since the window's start at every rung of the ladder), so the latent can hold the phase",
    )
    g.add_argument(
        "--window-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="train on windows of random length, log-uniform between LO and HI "
        "days (the evaluations stay at --window); the ladder then reaches 2 HI",
    )
    g.add_argument("--target", choices=["obs", "template", "mix"], default="obs",
                   help="reconstruction target: the observations, the window's smooth Fourier fit on the star's period (project.templates), "
                   "or the fit where its band fits well (adjusted R2 >= --template-min-r2) and the observation elsewhere")
    g.add_argument("--template-harmonics", type=int, default=6)
    g.add_argument("--template-min-r2", type=float, default=0.5)
    g.add_argument("--template-min-points", type=int, default=20, help="points a band needs in the window for a fit")
    g.add_argument("--template-periods", default=None, help="periods.csv of project.eval.period_table; default: the catalogue period")
    g.add_argument("--template-column", default="p_catalogue_sharp", help="its column, e.g. p_model_cands for the model's own periods")
    g.add_argument("--template-fit", choices=["record", "window"], default="record",
                   help="record: one template per star fitted on its whole light curve (harmonics chosen per star), stored and read at absolute times; "
                   "window: refitted on every window's points (the first version)")
    g.add_argument("--template-workers", type=int, default=8, help="processes for the whole-record fits at startup")
    g.add_argument("--query-extra", type=int, default=0, help="with --target template/mix: extra hidden query tokens per window at sampled times, the fit as their target")
    g.add_argument("--query-delta", type=float, nargs=2, default=[0.5, 30.0], metavar=("START", "END"),
                   help="days around a random real token the queries are drawn from, ramped linearly from START to END over --query-ramp of the steps")
    g.add_argument("--query-ramp", type=float, default=0.6, help="share of the steps over which the query distance ramps")
    g.add_argument("--query-cycles", type=float, nargs=2, default=None, metavar=("START", "END"),
                   help="draw the query distance in CYCLES of the star's template period instead of days: log-uniform between "
                        "--query-cycles-min and a cap that ramps geometrically from START to END over --query-ramp of the steps "
                        "(needs --template-fit record)")
    g.add_argument("--query-cycles-min", type=float, default=0.1, help="the smallest query distance in cycles")
    g.add_argument("--query-random-share", type=float, default=0.2, help="share of the queries at uniform random times in the window (so their layout does not reveal the period)")
    g.add_argument("--phasor", type=int, default=0, help="decoder tokens get sine/cosine features of their time at the spectral layer's own peak frequency and this many harmonics (0 = off; needs --spectral)")
    g.add_argument("--input", choices=["obs", "template"], default="obs",
                   help="template: the encoder sees the template value at every real observation time plus Gaussian noise at the window's residual level, in place of the observed brightness (everything real except the non-periodic part; needs --target template --template-fit record)")
    g.add_argument("--phasor-init", type=float, default=0.02, help="start scale of the phasor map")
    g.add_argument("--phase-freq", choices=["soft", "peak", "local"], default="soft", help="how the phase rope / phasor read the frequency off the spectral logits: the softmax expectation, the argmax bin, or the centroid within 5 bins of the argmax")
    g.add_argument("--phase-rope", type=int, default=0, help="the phase at the spectral layer's own frequency as a rotary axis of the decoder, this many harmonics (0 = off; needs --spectral)")
    g.add_argument("--init", default=None, help="warm start the autoencoder from this mae.pt (every tensor whose name and shape match; a fresh optimiser)")
    g.add_argument("--encoder", choices=["transformer", "gruode"], default="transformer",
                   help="gruode: the collaborators' GRU-ODE encoder (project.gruode) in place of the transformer, everything else the same")
    g.add_argument("--rnn-band", choices=["centre", "percentiles"], default="centre", help="gruode: the band as its central wavelength or its 10/50/90 %% filter wavelengths")
    g.add_argument("--rnn-ode-hidden", type=int, default=128)
    g.add_argument("--rnn-time-unit", type=float, default=100.0, help="gruode: days per ODE time unit")
    g.add_argument("--rnn-steps", type=int, default=1, help="gruode: fixed solver steps per gap")
    g.add_argument("--jepa", action="store_true", help="token-level JEPA (project.jepa): predict the EMA target encoder's latents of the hidden tokens instead of their values")
    g.add_argument("--jepa-ema", type=float, default=0.996, help="target encoder momentum at the start")
    g.add_argument("--jepa-ema-end", type=float, default=1.0, help="momentum at the end (linear schedule)")
    g.add_argument("--jepa-loss", choices=["smoothl1", "mse"], default="smoothl1")
    g.add_argument("--jepa-use-context", action="store_true", help="use the context encoder downstream (default: the EMA target encoder)")
    g.add_argument("--jepa-recon-weight", type=float, default=0.0, help="hybrid loss: this weight times the MSE of a brightness head on the predictor's outputs")
    g.add_argument("--jepa-proj", type=int, default=0, help="alignment loss in a projection of this width (0 = the encoder space); the probes read the pre-projection latent")
    g.add_argument("--jepa-proj-hidden", type=int, default=256)
    g.add_argument("--jepa-clean-target", action="store_true", help="denoising JEPA: the target encoder sees the smooth fit (needs --target template/mix), the context encoder the raw points")
    g.add_argument("--jepa-init", default=None, help="warm start the JEPA encoder (and the target copy) from this masked-autoencoder mae.pt")
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


def abs_timescales(ladder) -> list[float]:
    """Every distinct finite rung of the ladder (position units), sorted:
    the timescales of the absolute time features (``--abs-time``)."""
    vals = np.asarray(ladder.timescales, dtype=float).ravel()
    return sorted({float(v) for v in vals if np.isfinite(v) and v > 0})


def error_weights(values, pad, err_stats, cap: float = 20.0) -> torch.Tensor:
    """``1 / sigma^2`` per token from the standardised log-sigma channel
    (``values[..., 1]``, see :func:`project.common.err_channel`), capped at
    ``cap`` times the window's median weight and scaled to mean 1 over the
    real tokens of every window, so the loss keeps its scale and a few very
    precise points cannot take it over. 0 on padding."""
    if values.shape[-1] < 2 or err_stats is None:
        raise ValueError("--loss-weight err needs the error channel (no --no-err-channel)")
    mu, sd = err_stats
    log_sigma = values[..., 1].float() * sd + mu
    w = torch.exp(-2.0 * log_sigma).masked_fill(pad, 0.0)
    real = (~pad).float()
    med = torch.stack([row[m].median() if m.any() else row.new_tensor(1.0) for row, m in zip(w, ~pad)])
    w = torch.minimum(w, cap * med[:, None])
    w = w * real.sum(1, keepdim=True) / (w * real).sum(1, keepdim=True).clamp(min=1e-12)
    return w * real



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
    if args.window_range and args.lam_max is None:
        args.lam_max = 2.0 * float(args.window_range[1])  # the longest window must fit
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
    if args.window_range:
        ds = MultiWindowDataset(train, cfg, args.window_range, seed=args.seed)
        loader = DataLoader(
            ds,
            args.batch_size,
            shuffle=True,
            drop_last=len(ds) > args.batch_size,
            num_workers=args.workers,
            collate_fn=spec.collate(),
            persistent_workers=args.persistent_workers and args.workers > 0,
            prefetch_factor=args.prefetch if args.workers > 0 else None,
            pin_memory=args.pin_memory,
        )
        print(f"training windows of {args.window_range[0]:g} to {args.window_range[1]:g} d (log-uniform)")
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
    if ckpt is not None and ckpt.get("kind") == "bottleneck":
        model = BottleneckAE.from_checkpoint(ckpt)
    elif ckpt is not None and ckpt.get("kind") == "jepa":
        model = TokenJEPA.from_checkpoint(ckpt)
    elif ckpt is not None and ckpt.get("kind") == "gruode":
        model = GRUODEForPreTraining.from_checkpoint(ckpt)
    elif args.encoder == "gruode":
        model = GRUODEForPreTraining(
            decoder=decoder, mask_ratio=args.mask_ratio, target_channels=1, encoder=encoder_config(args), n_channels=spec.n_channels, n_axes=2,
            rope=rope_layouts(args, ladder), abs_timescales=None,
            rnn=dict(band=args.rnn_band, ode_hidden=args.rnn_ode_hidden, steps=args.rnn_steps, time_unit=args.rnn_time_unit, time_scale=ladder.time_scale,
                     band_wavelengths=[float(spec.tokenize["band_wavelengths"][k]) for k in sorted(spec.tokenize["band_wavelengths"])]),
        )  # fmt: skip
        print(f"GRU-ODE encoder: hidden {model.cfg.d_model}, {model.rnn_params / 1e6:.2f}M parameters, band features {model.n_band_feats}", flush=True)
    elif ckpt is not None:
        model = RoMAEForPreTraining(**ckpt["mae"], **ckpt["backbone"])
    elif args.jepa:
        model = TokenJEPA(
            decoder=decoder, mask_ratio=args.mask_ratio, encoder=encoder_config(args), n_channels=spec.n_channels, n_axes=2,
            rope=rope_layouts(args, ladder), abs_timescales=abs_timescales(ladder) if args.abs_time else None, spectral=spectral_kw,
            ema=args.jepa_ema, ema_end=args.jepa_ema_end, loss=args.jepa_loss, use_target=not args.jepa_use_context, recon_weight=args.jepa_recon_weight,
            clean_target=args.jepa_clean_target, proj_dim=args.jepa_proj, proj_hidden=args.jepa_proj_hidden, attn_pool=args.attn_pool,
        )  # fmt: skip
        if args.jepa_clean_target and args.target == "obs":
            parser.error("--jepa-clean-target needs --target template or mix")
        if args.jepa_init:
            model.init_from_mae(torch.load(args.jepa_init, map_location="cpu", weights_only=False))
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
            abs_timescales=abs_timescales(ladder) if args.abs_time else None,
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
            abs_timescales=abs_timescales(ladder) if args.abs_time else None,
            spectral=dict(time_scale=ladder.time_scale, p_min=args.spectral_p[0], p_max=args.spectral_p[1], rel=args.spectral_rel,
                          channels=args.spectral_channels, reader=args.spectral_reader, depth=args.spectral_depth,
                          after_layer=args.spectral_after) if args.spectral else None,  # fmt: skip
            phasor=args.phasor,
            phase_rope=args.phase_rope,
            phasor_init=args.phasor_init,
            phase_freq=args.phase_freq,
            attn_pool=args.attn_pool,
        )
        if args.init:
            from project.common import warm_start

            w = warm_start(model, torch.load(args.init, map_location="cpu", weights_only=False))
            print(f"warm start from {args.init}: {w['copied']} tensors copied, {w['left']} left as initialised"
                  f"{' (' + ', '.join(w['left_names'][:6]) + ('...' if w['left'] > 6 else '') + ')' if w['left'] else ''}", flush=True)
    if args.attn_pool > 0 and args.cls_inv <= 0 and args.cls_sig <= 0:
        parser.error("--attn-pool needs --cls-inv and/or --cls-sig: nothing else trains the pooling head")
    if (args.phasor or args.phase_rope) and not args.spectral:
        parser.error("--phasor and --phase-rope need --spectral")
    if args.query_cycles and args.template_fit != "record":
        parser.error("--query-cycles needs --template-fit record (the template period of every star)")
    if args.input == "template" and (args.target == "obs" or args.template_fit != "record"):
        parser.error("--input template needs --target template/mix with --template-fit record")
    model = model.to(dev)
    encoder = FrameEncoder(PooledEncoder(model))
    sizes = dict(
        params_encoder=n_params(model.transformer) + n_params(model.projection),
        params_total=n_params(model),
    )
    state_fn = bottleneck_state if isinstance(model, BottleneckAE) else jepa_state if isinstance(model, TokenJEPA) else gruode_state if isinstance(model, GRUODEForPreTraining) else mae_state

    if args.mask_mode != "random" and (args.bottleneck or isinstance(model, BottleneckAE)):
        parser.error("--mask-mode block / mix is for the plain masked autoencoder")

    aux = (args.phase_loss > 0 or args.fold_loss > 0 or args.spectral_aux > 0) and not args.bottleneck
    if args.spectral_aux > 0 and not args.spectral:
        parser.error("--spectral-aux needs --spectral")
    if aux and spec.err_stats is None:
        parser.error("--phase-loss / --fold-loss need the error channel")
    phase_head = nn.Linear(model.embed_dim, 2).to(dev) if args.phase_loss > 0 and not args.bottleneck else None
    fold_dec = None
    if args.fold_loss > 0 and not args.bottleneck:
        from project.phase_decoder import PhaseDecoder

        fold_dec = PhaseDecoder(model.embed_dim, model.embed_dim, args.dec_width, args.dec_heads, args.dec_depth, args.fold_harm,
                                len(spec.tokenize["band_wavelengths"]), latent_drop=0.0).to(dev)  # fmt: skip
    dyn = None
    if args.dyn > 0 and not args.bottleneck:
        if not args.spectral:
            parser.error("--dyn needs --spectral (the clock speed comes from the spectral layer)")
        from project.dyn import DynHead

        if args.dyn_ham:
            from project.dyn import HamHead

            if not args.dyn_cross:
                parser.error("--dyn-ham needs --dyn-cross")
            dyn = HamHead(model.embed_dim, args.dyn, n_sub=args.dyn_ham_steps).to(dev)
        else:
            dyn = DynHead(model.embed_dim, args.dyn).to(dev)
        if args.dyn_decode > 0:
            if args.dyn_target != "ema" or spec.err_stats is None:
                parser.error("--dyn-decode needs --dyn-target ema and the error channel")
            dyn.add_decoder(model.embed_dim, n_bands=len(spec.tokenize["band_wavelengths"]) if args.dyn_dec_band else 0,
                            learn_sig=args.dyn_dec_nll)
            dyn.to(dev)
        if args.dyn_target == "ema" and not (args.dyn_cross and args.jepa):
            parser.error("--dyn-target ema needs --dyn-cross and --jepa")
        if args.dyn_freeze:
            if not args.init and ckpt is None:
                parser.error("--dyn-freeze needs --init (a trained encoder to freeze) or a resume")
            model.requires_grad_(False)
            print("encoder frozen: only the dynamic-state head and predictor train", flush=True)
    aux_modules = [m for m in (phase_head, fold_dec, dyn) if m is not None]

    def dyn_term(tok, cl, pd, t_days, step_now, n_fr=None, n_b=None, start_rows=None, periods_rows=None, tgt=None):
        """The add-on's loss on token outputs ``tok [R, N, D]`` (no CLS), ``cl [R, D]``, padding ``pd [R, N]``,
        ``t_days [R, N]``; the clock speed is the spectral layer's own estimate from the same forward."""
        from romae_lc.model import spectral_frequency

        frac = min(step_now / max(args.dyn_ramp * args.steps, 1), 1.0)
        dt_max = args.dyn_dt[0] * (args.dyn_dt[1] / args.dyn_dt[0]) ** frac
        f = spectral_frequency(model.spectral, None, args.dyn_freq)
        if args.dyn_no_rot:
            f = torch.zeros_like(f)
        if args.dyn_lambda is None:
            w_pred, w_nce, w_sig = args.dyn_pred, args.dyn_nce, args.dyn_sig
        else:
            lam = args.dyn_lambda
            w_pred, w_nce, w_sig = 1.0 - lam, (1.0 - lam) * args.dyn_nce, lam
        if args.dyn_cross:
            # absolute token time: window start (days, relative to the star's first window) + time in the window
            t_abs = start_rows.to(tok.device).float()[:, None] + t_days
            kw = {}
            if tgt is not None:  # tgt: dict(tokens, pad, t_days, mag, sig) over the target set
                if "tokens" in tgt:
                    kw.update(tgt_tokens=tgt["tokens"], tgt_pad=tgt["pad"], tgt_t_abs=start_rows.to(tok.device).float()[:, None] + tgt["t_days"])
                if args.dyn_decode > 0:
                    kw.update(tgt_mag=tgt["mag"], tgt_sig=tgt["sig"], w_dec=args.dyn_decode, sig_floor=args.dyn_sig_floor)
                    if args.dyn_dec_band:
                        kw.update(tgt_band=tgt["band"])
            l_dyn, st = dyn.forward_cross(tok, cl, pd, t_abs, f, n_fr, n_b, dt_max, args.dyn_pairs, args.dyn_dt[0], args.dyn_tau,
                                          w_nce, w_sig, args.dyn_mean, w_pred, periods_rows, args.dyn_phase_pos,
                                          args.dyn_ham_close if args.dyn_ham else 0.0, sig_mode=args.dyn_sig_mode, **kw)  # fmt: skip
        else:
            l_dyn, st = dyn(tok, cl, pd, t_days, f, dt_max, args.dyn_pairs, args.dyn_dt[0], args.dyn_tau, w_nce, w_sig, args.dyn_mean, w_pred)
        for k_, v_ in st.items():
            aux_stats.setdefault(f"dyn_{k_}", []).append(v_)
        return args.dyn_weight * l_dyn
    train_periods = torch.tensor([float(r.period or float("nan")) for r in train], dtype=torch.float64)
    smooth = args.target != "obs"
    record_templates = None
    if smooth and isinstance(model, BottleneckAE):
        parser.error("--target template/mix is for the masked autoencoder and the JEPA")
    if smooth:
        from project.templates import period_lookup, template_periods

        from project.templates import RecordTemplates, fit_record_templates, period_lookup_best

        if args.template_periods and args.template_column == "p_catalogue_sharp":
            lookup = period_lookup_best(args.template_periods)  # the sharpened catalogue, or the model's period where the catalogue is another period
        else:
            lookup = period_lookup(args.template_periods, args.template_column) if args.template_periods else None
        train_template_periods = template_periods(train, lookup)
        if args.template_fit == "record":
            tpl_file = out / "templates.pt"
            if tpl_file.is_file():
                fitted = torch.load(tpl_file, map_location="cpu", weights_only=False)
                print(f"templates: loaded {tpl_file}", flush=True)
            else:
                fitted = fit_record_templates(train, train_template_periods.numpy(), n_bands=len(spec.tokenize["band_wavelengths"]), workers=args.template_workers)
                torch.save(fitted, tpl_file)
            record_templates = RecordTemplates(fitted, dev)
        print(f"smooth targets ({args.target}): {args.template_harmonics} harmonics on "
              f"{'the catalogue period' if lookup is None else args.template_column + ' of ' + args.template_periods}"
              f"{'' if lookup is None else f' ({sum(1 for r in train if str(r.meta.get(chr(105)+chr(100))) in lookup)} of {len(train)} training stars found)'}", flush=True)
    aux_stats = {"phase": [], "fold": [], "spectral": [], "template_share": []}

    def aux_losses(out, values, positions, pad, periods_rows):
        """The per-token phase loss on the encoder's visible-token outputs
        and the fold loss on the CLS, for rows with a period."""
        from project.phase_decoder import band_ids, band_table

        from project.decoder import gaussian_var

        ts = spec.tokenize["time_scale"]
        period = periods_rows.to(dev)
        ok = torch.isfinite(period) & (period > 0)
        total = values.new_zeros(())
        if phase_head is not None:
            tok, pos, pd = out.enc_tokens[:, 1:], out.enc_positions[:, 0, 1:], out.enc_pad[:, 1:]
            phi = torch.remainder((pos.double() - 1.0) * ts / period.clamp_min(1e-6)[:, None], 1.0).float()
            target = torch.stack([torch.cos(2 * math.pi * phi), torch.sin(2 * math.pi * phi)], -1)
            pred = phase_head(tok.float())
            pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            m = (~pd) & ok[:, None]
            lp = ((pred - target) ** 2).sum(-1)
            lp = (lp * m).sum() / m.sum().clamp(min=1)
            aux_stats["phase"].append(float(lp))
            total = total + args.phase_loss * lp
        if fold_dec is not None:
            z = out.enc_tokens[:, 0].float()
            phi = torch.remainder((positions[:, 0].double() - 1.0) * ts / period.clamp_min(1e-6)[:, None], 1.0).float()
            band = band_ids(positions[:, 1], band_table(spec))
            mu_s, sd_s = spec.err_stats
            sigma = torch.exp(values[..., 1].float() * sd_s + mu_s)
            mu, logvar = fold_dec(z, phi, band, pad, None)
            var = gaussian_var(sigma, logvar)
            y = values[..., 0].float()
            nll = 0.5 * ((y - mu.float()).square() / var + var.log())
            m = (~pad) & ok[:, None]
            lf = (nll * m).sum() / m.sum().clamp(min=1)
            aux_stats["fold"].append(float(lf))
            total = total + args.fold_loss * lf
        if args.spectral_aux > 0:
            from romae_lc.spectral import spectral_aux_loss

            ls_ = spectral_aux_loss(model.spectral, period)
            aux_stats["spectral"].append(float(ls_))
            total = total + args.spectral_aux * ls_
        return total

    def forward(frames, mode=None, periods_rows=None, template_rows=None, step_now=0, index_rows=None, start_rows=None, dyn_starts=None):
        """The reconstruction loss of one fused window batch; the bottleneck
        model takes the Tokens with the per-point errors. ``mode`` is the
        mask mode (default: the run's). With ``periods_rows`` (the catalogue
        period of every fused row) the auxiliary phase and fold losses are
        added."""
        if isinstance(model, BottleneckAE):
            return model(fuse_tokens(frames)).loss
        values, positions, pad = fuse_frames(frames)
        if dyn is not None and args.dyn_freeze and model.training:
            # frozen encoder: the full window, no masking, no reconstruction; only the add-on's loss
            with torch.no_grad():
                x, _ = model.encode(values, positions, pad)
            t_days = (positions[:, 0].float() - 1.0) * spec.tokenize["time_scale"]
            return dyn_term(x[:, 1:].float(), x[:, 0].float(), pad, t_days, step_now, len(frames), frames[0].values.shape[0], dyn_starts, periods_rows)
        mode = args.mask_mode if mode is None else mode
        mask = None
        ratio = args.mask_ratio
        if args.mask_ratio_range is not None and model.training:
            lo, hi = args.mask_ratio_range
            ratio = lo + (hi - lo) * float(torch.rand(()))
        if mode == "phaseband":
            from project.mae_data import phase_band_mask

            if periods_rows is None:
                raise ValueError("--mask-mode phaseband needs the period of every row (the aux losses on)")
            t_days_all = (positions[:, 0].float() - 1.0) * spec.tokenize["time_scale"]
            mask = phase_band_mask(t_days_all, pad, periods_rows.to(pad.device), ratio, args.phase_band)
        elif mode == "random" and ratio != args.mask_ratio:
            from romae_lc.model import gen_mask

            mask = gen_mask(ratio, pad)
        elif mode != "random":
            mask = make_mask(
                positions[:, 0], pad, ratio, mode, args.mask_blocks, args.mask_block_prob,
                position=args.mask_block_pos, block_share=args.mask_block_share,
            )
        if args.forecast_tail is not None and model.training:
            t_days_all = (positions[:, 0].float() - 1.0) * spec.tokenize["time_scale"]
            if mask is None:
                from romae_lc.model import gen_mask

                mask = gen_mask(ratio, pad)
            mask, tail_share = forecast_tail_mask(mask, t_days_all, pad, args.window, args.forecast_tail, args.forecast_share)
            aux_stats.setdefault("forecast_tail_share", []).append(tail_share)
        weight = error_weights(values, pad, spec.err_stats, args.loss_weight_cap) if args.loss_weight == "err" else None
        target_values = None
        if smooth and template_rows is not None:
            from project.templates import augment_with_queries, bounded, record_targets, sample_queries, smooth_targets, template_eval

            use_record = record_templates is not None and index_rows is not None and start_rows is not None
            if use_record:
                target_values, share = record_targets(record_templates, index_rows, start_rows, values, positions, pad, spec, args.target, args.template_min_r2)
                last = record_targets.last
                if args.input == "template":
                    # the template as the input too, with noise at the window's own residual level
                    y0 = values[..., 0].float()
                    real = ~pad
                    tv = target_values.float()
                    have = torch.isfinite(tv) & real & (tv != y0)
                    resid = torch.where(have, y0 - tv, torch.zeros_like(y0))
                    n_have = have.float().sum(1, keepdim=True).clamp_min(1.0)
                    sigma = (resid.pow(2).sum(1, keepdim=True) / n_have).sqrt()
                    noisy = tv + sigma * torch.randn_like(y0)
                    values = values.clone()
                    values[..., 0] = torch.where(have, noisy, y0).to(values.dtype)
            else:
                target_values, share = smooth_targets(values, positions, pad, template_rows, spec, args.template_harmonics, args.template_min_r2,
                                                      args.template_min_points, args.target)  # fmt: skip
                last = smooth_targets.last
            aux_stats["template_share"].append(share)
            if args.query_extra > 0:
                # extra hidden tokens at sampled times, the fit as their target: the distance to a seen point ramps up over training
                from romae_lc.model import gen_mask

                frac = min(step_now / max(args.query_ramp * args.steps, 1), 1.0)
                if mask is None:
                    mask = gen_mask(args.mask_ratio, pad)
                if args.query_cycles and use_record:
                    from project.templates import sample_queries_cycles

                    lo_c, hi_c = args.query_cycles
                    cap = lo_c * (hi_c / lo_c) ** frac  # geometric ramp of the cap, in cycles
                    t_q, band_q = sample_queries_cycles(positions, pad, last["band"], args.query_extra, record_templates.period[index_rows.to(record_templates.period.device)],
                                                        args.query_cycles_min, cap, spec.tokenize["time_scale"], args.query_random_share)  # fmt: skip
                    delta = cap
                    aux_stats.setdefault("query_cap_cycles", []).append(float(cap))
                else:
                    delta = args.query_delta[0] + (args.query_delta[1] - args.query_delta[0]) * frac
                    t_q, band_q = sample_queries(positions, pad, last["band"], args.query_extra, delta, spec.tokenize["time_scale"])
                    aux_stats.setdefault("query_delta_days", []).append(float(delta))
                if use_record:
                    t_abs_q = start_rows.to(t_q.device)[:, None].double() + t_q.double()
                    target_q = bounded(record_templates.eval(index_rows, t_abs_q, band_q), last["y"], last["real"])
                else:
                    target_q = bounded(template_eval(last["beta"], last["period"], t_q, band_q, args.template_harmonics), last["y"], last["real"])
                values, positions, pad, mask, target_values, weight = augment_with_queries(values, positions, pad, mask, target_values, weight, t_q, band_q, target_q, spec)
        out = model(values, positions, pad, mask, weight, target_values=target_values)
        loss = out.loss
        if aux and periods_rows is not None:
            loss = loss + aux_losses(out, values, positions, pad, periods_rows)
        if dyn is not None and model.training:
            # the dynamic-state add-on on the visible-token outputs of the same masked forward
            t_days = (out.enc_positions[:, 0, 1:].float() - 1.0) * spec.tokenize["time_scale"]
            tgt = None
            if args.dyn_cross and args.dyn_target == "ema":
                # the target encoder's full-window tokens (no CLS), stashed by TokenJEPA.forward, and the window's values
                mu_e, sd_e = spec.err_stats if spec.err_stats is not None else (0.0, 1.0)
                xf, pdf = model.last_target_tokens
                tgt = dict(tokens=xf, pad=pdf, t_days=(positions[:, 0].float() - 1.0) * spec.tokenize["time_scale"],
                           mag=values[..., 0], sig=torch.exp(values[..., 1].float() * sd_e + mu_e))
                if args.dyn_dec_band:
                    from project.phase_decoder import band_ids, band_table

                    tgt["band"] = band_ids(positions[:, 1], band_table(spec))
            loss = loss + dyn_term(out.enc_tokens[:, 1:], out.enc_tokens[:, 0], out.enc_pad[:, 1:], t_days, step_now,
                                   len(frames), frames[0].values.shape[0], dyn_starts, periods_rows, tgt)  # fmt: skip
        if (args.cls_inv > 0 or args.cls_sig > 0) and model.training:
            # LeJEPA's static part: fused rows are frame t of every star in rows t B .. (t + 1) B, so a star's windows are rows b, B + b, ...
            from project.dyn import sigreg

            # the window vector: the attention-pooling head over the visible tokens when there is one, else the CLS
            enc_net = model.context if isinstance(model, TokenJEPA) else model
            z = enc_net.pool_vector(out.enc_tokens, out.enc_pad).float() if args.attn_pool > 0 else out.enc_tokens[:, 0].float()
            n_fr, n_b = len(frames), frames[0].values.shape[0]
            zs = z.view(n_fr, n_b, -1)
            if args.cls_inv > 0:
                l_inv = (zs - zs.mean(0, keepdim=True)).pow(2).mean()
                aux_stats.setdefault("cls_inv", []).append(float(l_inv.detach()))
                loss = loss + args.cls_inv * l_inv
            if args.cls_sig > 0:
                l_sig = sigreg(z)
                aux_stats.setdefault("cls_sig", []).append(float(l_sig.detach()))
                loss = loss + args.cls_sig * l_sig
        return loss

    print(
        f"params: encoder {sizes['params_encoder'] / 1e6:.2f}M, "
        f"total {sizes['params_total'] / 1e6:.2f}M"
    )
    # the phasor map gets no weight decay: with nothing using it yet, decay shrank it to nothing (cycph, 2026-10-12)
    ph = [p for n, p in model.named_parameters() if n.startswith("phasor.") and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("phasor.") and p.requires_grad] + [p for m in aux_modules for p in m.parameters()]
    opt = torch.optim.AdamW([dict(params=rest), dict(params=ph, weight_decay=0.0)] if ph else rest, lr=args.lr, weight_decay=args.wd)
    sched = LambdaLR(opt, cosine_schedule(args.steps, args.warmup))
    step, epoch, elapsed, last_metrics = 0, 0, 0.0, None
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"])
        for name, m in (("phase_head", phase_head), ("fold_dec", fold_dec), ("dyn", dyn)):
            if m is not None and name in ckpt.get("aux", {}):
                m.load_state_dict(ckpt["aux"][name])
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
        state.update(epoch=epoch, elapsed=elapsed, wandb_id=tracker.id,
                     aux={n: m.state_dict() for n, m in (("phase_head", phase_head), ("fold_dec", fold_dec), ("dyn", dyn)) if m is not None})  # fmt: skip
        if with_opt:
            state.update(opt=opt.state_dict(), sched=sched.state_dict())
        save_atomic(state, path)

    @torch.no_grad()
    def evaluate(train_stats: dict) -> dict:
        t0 = time.time()
        model.eval()
        total, total_b, n = 0.0, 0.0, 0
        plain = not isinstance(model, BottleneckAE)
        for batch in val_loader:
            frames = [f.to(dev) for f in batch["frames"]]
            with amp:
                # random points hidden: the same task in every run, so the
                # number compares across runs; stretches hidden: the harder task
                loss = forward(frames, "random")
                loss_b = forward(frames, "block") if plain else loss
            rows = len(frames) * frames[0].values.shape[0]
            total += loss.item() * rows
            total_b += loss_b.item() * rows
            n += rows
        v_loss = total / max(n, 1)
        v_loss_block = total_b / max(n, 1) if plain else float("nan")
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
            template_rows = train_template_periods[batch["index"]].repeat(len(frames)) if smooth else None
            index_rows = batch["index"].repeat(len(frames)) if smooth else None  # fused rows: frame t of every item in rows t B .. (t + 1) B
            start_rows = batch["starts"].T.reshape(-1) if (smooth and "starts" in batch) else None
            dyn_starts = None
            if dyn is not None and args.dyn_cross:
                st0 = batch["starts"]  # [B, T] days; relative to the star's first window keeps float32 exact enough
                dyn_starts = (st0 - st0[:, :1]).T.reshape(-1)
            with amp:
                loss_t = forward(frames, periods_rows=periods_rows, template_rows=template_rows, step_now=step, index_rows=index_rows,
                                 start_rows=start_rows, dyn_starts=dyn_starts)  # fmt: skip
            opt.zero_grad(set_to_none=True)
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [p for m in aux_modules for p in m.parameters()], args.clip)
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
