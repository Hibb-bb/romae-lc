"""The period head: a fine period from the raw window, in one forward pass.

The encoder's rotary rungs sit 2 to 4 % apart, so its pooled latent holds
the period to a few per cent at best, while a fold needs one part in ten
thousand. This head reads the fine information where it is, in the raw
points, and uses the latent only to choose among peaks:

1. a fixed spectrum, no parameters: the generalised Lomb-Scargle power of
   the window (errors as weights, every band's mean removed) on a dense
   log grid of ``K`` trial frequencies (0.05 % steps from 0.02 to 500 d,
   about 20,000 bins, finer than the peak of a 0.3 d star in a 250 d
   window), plus the same power read at twice and at half the frequency
   as extra channels, so a harmonic sits next to its fundamental;
2. learned peak picking: a 1-D convolutional network along the frequency
   axis, modulated by the pooled latent (feature-wise scale and shift), that
   outputs a logit and a sub-bin offset per bin;
3. the read-out: the best bin's centre plus its offset is the period; the
   softmax is the distribution over periods; the top bins are candidates.

The catalogue period is the training TARGET of the logits (cross-entropy,
softened over the neighbouring bins) and of the offset; it is never an
input. At inference the head sees the raw window and the latent only. The
plain Lomb-Scargle peak of the same spectrum is reported as the baseline:
the gap between the two is what the learned picking (aliases, harmonics,
the shape prior from the latent) adds.

    python -m project.period_head --ckpt project/runs/mae_w250/mae.pt --out project/runs/mae_w250/period_head
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from project.common import (
    JsonlLog,
    add_device_arg,
    cosine_schedule,
    data_args_from,
    dump_json,
    frame_loader,
    fuse_frames,
    get_device,
    load_data,
    load_encoder,
    n_params,
    save_atomic,
    seed_all,
    subset,
    superclass,
)
from project.phase_decoder import band_ids, band_table
from project.tracking import StepTimer, Tracker, add_wandb_args, gpu_stats

KIND = "period_head"
TOLERANCES = (0.10, 0.01, 0.001, 0.0001)


# ------------------------------------------------------------------ spectrum


class FrequencyGrid:
    """A log-spaced grid of trial frequencies (cycles per day) with a
    relative step ``rel``: bin ``k`` is ``f0 (1 + rel)^k``; ``shift``
    bins move by a factor of two."""

    def __init__(self, p_min: float = 0.02, p_max: float = 500.0, rel: float = 5e-4):
        self.rel, self.step = float(rel), math.log1p(rel)
        f_lo, f_hi = 1.0 / p_max, 1.0 / p_min
        self.n = int(math.ceil(math.log(f_hi / f_lo) / self.step)) + 1
        self.log_f0 = math.log(f_lo)
        self.shift = int(round(math.log(2.0) / self.step))

    @property
    def freqs(self) -> np.ndarray:
        return np.exp(self.log_f0 + self.step * np.arange(self.n))

    def bin_of(self, period) -> np.ndarray:
        """Continuous bin coordinate of a period (days)."""
        return (np.log(1.0 / np.asarray(period, dtype=np.float64)) - self.log_f0) / self.step

    def period_of(self, coord) -> np.ndarray:
        return 1.0 / np.exp(self.log_f0 + self.step * np.asarray(coord, dtype=np.float64))

    def to_dict(self) -> dict:
        return dict(p_min=1.0 / math.exp(self.log_f0 + self.step * (self.n - 1)), p_max=1.0 / math.exp(self.log_f0), rel=self.rel)


@torch.no_grad()
def window_spectrum(t, y, w, band, freqs, n_bands: int, chunk: int = 2048) -> torch.Tensor:
    """Generalised Lomb-Scargle power ``[B, K]`` of ``B`` windows at the
    ``K`` frequencies ``freqs`` (cycles per day): ``t, y, w, band [B, N]``
    with ``w`` the per-point weights (0 on padding), every band's weighted
    mean removed first. float32; ``chunk`` frequencies at a time."""
    b, n = t.shape
    w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
    # per band weighted mean
    bw = torch.zeros(b, n_bands, device=t.device).scatter_add_(1, band, w)
    by = torch.zeros(b, n_bands, device=t.device).scatter_add_(1, band, w * y)
    mean = (by / bw.clamp_min(1e-12)).gather(1, band)
    y = (y - mean) * (w > 0)
    y = y - (w * y).sum(1, keepdim=True)
    yy = (w * y * y).sum(1).clamp_min(1e-12)
    out = torch.empty(b, len(freqs), device=t.device)
    two_pi = 2 * math.pi
    for lo in range(0, len(freqs), chunk):
        f = freqs[lo : lo + chunk]
        ph = two_pi * t[:, :, None] * f[None, None, :]  # [B, N, k]
        c, s = torch.cos(ph), torch.sin(ph)
        wc, ws = w[..., None] * c, w[..., None] * s
        C, S = wc.sum(1), ws.sum(1)
        yc, ys = (wc * y[..., None]).sum(1), (ws * y[..., None]).sum(1)
        cc = (wc * c).sum(1) - C * C
        ss = (ws * s).sum(1) - S * S
        cs = (wc * s).sum(1) - C * S
        d = cc * ss - cs * cs
        p = (ss * yc * yc + cc * ys * ys - 2 * cs * yc * ys) / (yy[:, None] * d)
        out[:, lo : lo + chunk] = torch.where(torch.isfinite(p), p, torch.zeros_like(p)).clamp(0, 1)
    return out


def spectrum_channels(power: torch.Tensor, shift: int) -> torch.Tensor:
    """``[B, 6, K]``: the power, the power at twice and at half the
    frequency (shifted copies, zero outside the grid), each as is and
    log-scaled."""
    b, k = power.shape
    twice = torch.zeros_like(power)
    twice[:, : k - shift] = power[:, shift:]
    half = torch.zeros_like(power)
    half[:, shift:] = power[:, : k - shift]
    lin = torch.stack([power, twice, half], 1)
    return torch.cat([lin, torch.log(lin + 1e-3) * 0.25], 1)


# --------------------------------------------------------------------- model


class FiLMBlock(nn.Module):
    """A dilated convolution along the frequency axis, modulated by the
    latent (feature-wise scale and shift, zero at initialisation so the
    block starts as a plain convolution)."""

    def __init__(self, ch: int, dilation: int, z_dim: int, kernel: int = 9):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel, padding=dilation * (kernel // 2), dilation=dilation)
        self.mix = nn.Conv1d(ch, ch, 1)
        self.norm = nn.GroupNorm(8, ch)
        self.film = nn.Linear(z_dim, 2 * ch)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, x, z):
        scale, shift = self.film(z).chunk(2, -1)
        h = self.norm(self.conv(x)) * (1 + scale[..., None]) + shift[..., None]
        return x + self.mix(F.gelu(h))


class PeakNet(nn.Module):
    """Logit and sub-bin offset per frequency bin from the spectrum
    channels ``[B, 6, K]`` and the latent ``z [B, z_dim]``."""

    def __init__(self, z_dim: int, ch: int = 64, depth: int = 4, kernel: int = 9, in_ch: int = 6):
        super().__init__()
        self.z_dim, self.ch, self.depth, self.kernel, self.in_ch = int(z_dim), int(ch), int(depth), int(kernel), int(in_ch)
        self.inp = nn.Conv1d(in_ch, ch, kernel, padding=kernel // 2)
        self.z_norm = nn.LayerNorm(z_dim)  # the raw latent has a norm of ~70; the modulation wants unit scale
        self.blocks = nn.ModuleList(FiLMBlock(ch, 2**i, z_dim, kernel) for i in range(depth))
        self.out = nn.Conv1d(ch, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @property
    def hparams(self) -> dict:
        return dict(z_dim=self.z_dim, ch=self.ch, depth=self.depth, kernel=self.kernel, in_ch=self.in_ch)

    def forward(self, spec, z):
        x = self.inp(spec)
        z = self.z_norm(z.float()).to(x.dtype)
        for blk in self.blocks:
            x = blk(x, z)
        out = self.out(x)
        return out[:, 0], out[:, 1]  # logits [B, K], offset [B, K]


def soft_target(coord: torch.Tensor, k: int, sigma: float = 1.0) -> torch.Tensor:
    """``[B, K]`` target distribution: a Gaussian of width ``sigma`` bins
    around the continuous bin coordinate of the true period."""
    idx = torch.arange(k, device=coord.device, dtype=torch.float32)
    logits = -0.5 * ((idx[None] - coord[:, None]) / sigma) ** 2
    return torch.softmax(logits, 1)


def head_loss(logits, offset, coord, sigma: float = 1.0):
    """Cross-entropy against the softened target plus the squared offset
    error at the true bin; rows with a non-finite coordinate are skipped."""
    ok = torch.isfinite(coord)
    if ok.sum() == 0:
        return logits.sum() * 0.0
    logits, offset, coord = logits[ok], offset[ok], coord[ok]
    target = soft_target(coord, logits.shape[1], sigma)
    ce = -(target * F.log_softmax(logits.float(), 1)).sum(1).mean()
    idx = coord.round().long().clamp(0, logits.shape[1] - 1)
    off = (offset.float().gather(1, idx[:, None])[:, 0] - (coord - idx.float())).square().mean()
    return ce + off


# ------------------------------------------------------------------ batches


@torch.no_grad()
def batch_spectra(enc, spec, frames, device, grid: FrequencyGrid, freqs_t, err_floor: float = 1e-3, no_latent: bool = False):
    """Latents, spectrum channels and raw inputs of the fused windows of a
    batch: ``(z [R, D], chans [R, 6, K])`` for the ``R = B T`` windows."""
    values, positions, pad = fuse_frames(frames)
    ts = spec.tokenize["time_scale"]
    t = ((positions[:, 0].float() - 1.0) * ts).masked_fill(pad, 0.0)
    y = values[..., 0].float().masked_fill(pad, 0.0)
    mu, sd = spec.err_stats
    sigma = torch.exp(values[..., 1].float() * sd + mu)
    w = (1.0 / (sigma * sigma + err_floor * err_floor)).masked_fill(pad, 0.0)
    band = band_ids(positions[:, 1], band_table(spec)).clamp(0, len(spec.tokenize["band_wavelengths"]) - 1)
    power = window_spectrum(t, y, w, band, freqs_t, len(spec.tokenize["band_wavelengths"]))
    chans = spectrum_channels(power, grid.shift)
    if no_latent:
        return torch.zeros(values.shape[0], enc.dim, device=device), chans, power
    x, _ = enc.model.encode(values, positions, pad)
    return x[:, 0].float(), chans, power


def hits(p_hat: np.ndarray, p_true: np.ndarray) -> dict:
    r = np.abs(p_hat / p_true - 1.0)
    out = {f"within_{tol:g}": float(np.mean(r < tol)) for tol in TOLERANCES}
    alias = (np.abs(p_hat / p_true / 2 - 1) < 0.1) | (np.abs(p_hat / p_true * 2 - 1) < 0.1)
    out["alias"] = float(np.mean(alias & (r >= 0.1)))
    return out


@torch.no_grad()
def evaluate(enc, head, spec, loader, records, device, grid, freqs_t, log=print, keep_predictions=False, no_latent=False):
    """Per window and per object (the object's windows' log-probabilities
    added), the head's period and the plain Lomb-Scargle peak against the
    catalogue period: hit rates within 10 %, 1 %, 0.1 % and 0.01 %, the
    alias rate, per superclass."""
    head.eval()
    win_h, win_l, win_p, obj_rows = [], [], [], {}
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for batch in loader:
        frames = [f.to(device) for f in batch["frames"]]
        b, tt = frames[0].values.shape[0], len(frames)
        idx = batch["index"].repeat(tt)
        with amp:
            z, chans, power = batch_spectra(enc, spec, frames, device, grid, freqs_t, no_latent=no_latent)
            logits, offset = head(chans, z)
        logits, offset = logits.float(), offset.float()
        best = logits.argmax(1)
        coord = best.float() + offset.gather(1, best[:, None])[:, 0].clamp(-0.5, 0.5)
        p_h = grid.period_of(coord.cpu().numpy())
        p_l = grid.period_of(power.argmax(1).cpu().numpy())
        p_t = np.array([float(records[int(i)].period or np.nan) for i in idx])
        win_h.append(p_h), win_l.append(p_l), win_p.append(p_t)
        lp = F.log_softmax(logits, 1)
        for r in range(b * tt):
            i = int(idx[r])
            o = obj_rows.setdefault(i, dict(lp=torch.zeros(logits.shape[1], device=device), lpow=torch.zeros_like(power[0]), n=0, p=p_t[r]))
            o["lp"] += lp[r]
            o["lpow"] += torch.log(power[r] + 1e-3)
            o["n"] += 1
    win_h, win_l, win_p = np.concatenate(win_h), np.concatenate(win_l), np.concatenate(win_p)
    ok = np.isfinite(win_p) & (win_p > 0)
    res = dict(n_windows=int(ok.sum()), window=dict(head=hits(win_h[ok], win_p[ok]), ls=hits(win_l[ok], win_p[ok])))
    obj_i, obj_h, obj_l, obj_p, obj_top = [], [], [], [], []
    for i, o in obj_rows.items():
        if not (np.isfinite(o["p"]) and o["p"] > 0):
            continue
        lp = o["lp"]
        best = int(lp.argmax())
        obj_i.append(i), obj_h.append(float(grid.period_of(best))), obj_l.append(float(grid.period_of(int(o["lpow"].argmax())))), obj_p.append(o["p"])
        obj_top.append(grid.period_of(lp.topk(5).indices.cpu().numpy()))
    obj_h, obj_l, obj_p = np.array(obj_h), np.array(obj_l), np.array(obj_p)
    sup = np.array([superclass(records[i]) for i in obj_i])
    res["n_objects"] = len(obj_i)
    res["object"] = dict(head=hits(obj_h, obj_p), ls=hits(obj_l, obj_p))
    res["object"]["by_superclass"] = {g: dict(n=int((sup == g).sum()), head=hits(obj_h[sup == g], obj_p[sup == g]), ls=hits(obj_l[sup == g], obj_p[sup == g]))
                                      for g in ("ECL", "RR", "ROT", "CEP", "DSCT", "LPV") if (sup == g).sum() >= 5}  # fmt: skip
    head.train()
    w, o = res["window"], res["object"]
    log(f"  period head ({res['n_windows']} windows, {res['n_objects']} objects): per window within 10%/1%/0.1%/0.01%: head "
        f"{w['head']['within_0.1']:.2f}/{w['head']['within_0.01']:.2f}/{w['head']['within_0.001']:.2f}/{w['head']['within_0.0001']:.2f} | LS peak "
        f"{w['ls']['within_0.1']:.2f}/{w['ls']['within_0.01']:.2f}/{w['ls']['within_0.001']:.2f}/{w['ls']['within_0.0001']:.2f} || per object: head "
        f"{o['head']['within_0.1']:.2f}/{o['head']['within_0.01']:.2f}/{o['head']['within_0.001']:.2f}/{o['head']['within_0.0001']:.2f} (alias {o['head']['alias']:.2f}) | LS "
        f"{o['ls']['within_0.1']:.2f}/{o['ls']['within_0.01']:.2f}/{o['ls']['within_0.001']:.2f}/{o['ls']['within_0.0001']:.2f} (alias {o['ls']['alias']:.2f}) | "
        + " ".join(f"{g} {v['head']['within_0.1']:.2f}/{v['ls']['within_0.1']:.2f}" for g, v in res["object"]["by_superclass"].items()))
    if keep_predictions:
        res["predictions"] = dict(index=np.array(obj_i), p_model=obj_h, p_ls=obj_l, p_catalogue=obj_p, superclass=sup, p_top=np.stack(obj_top))
    return res


# ------------------------------------------------------------------ training

RUN_CONTROL = ("steps", "time_budget", "workers", "device", "eval_every", "ckpt_every", "log_every", "no_resume", "wandb",
               "wandb_name", "wandb_project", "wandb_entity", "wandb_group", "wandb_tags", "val_objects", "pin_memory")  # fmt: skip


def add_args(p):
    p.add_argument("--ckpt", required=True, help="stage-1 checkpoint (the frozen encoder)")
    p.add_argument("--out", required=True)
    p.add_argument("--p-min", type=float, default=0.02)
    p.add_argument("--p-max", type=float, default=500.0)
    p.add_argument("--grid-rel", type=float, default=5e-4, help="relative frequency step of the grid")
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--kernel", type=int, default=9)
    p.add_argument("--target-sigma", type=float, default=1.0, help="softening of the target, in bins")
    p.add_argument("--no-latent", action="store_true", help="zeros in place of the latent: the spectrum and the picking alone")
    p.add_argument("--n-frames", type=int, default=4, help="windows per object per batch")
    p.add_argument("--steps", type=int, default=15_000)
    p.add_argument("--batch-size", type=int, default=16, help="objects per step (x n-frames windows)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.02)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--time-budget", type=float, default=0.0)
    p.add_argument("--val-objects", type=int, default=1000)
    p.add_argument("--train-objects", type=int, default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--n-sim", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    add_device_arg(p)
    add_wandb_args(p)


def load_period_head(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("kind") != KIND:
        raise ValueError(f"{path} is not a period head checkpoint")
    head = PeakNet(**ckpt["hparams"])
    head.load_state_dict(ckpt["state_dict"])
    return head.to(device).eval(), FrequencyGrid(**ckpt["grid"]), dict(ckpt.get("meta", {}), step=ckpt.get("step"), metrics=ckpt.get("metrics"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    args = p.parse_args(argv)
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
    args.out = str(out)
    dump_json(vars(args), out / "args.json")
    seed_all(args.seed)
    dev = get_device(args)
    t_start = time.time()
    enc, meta = load_encoder(args.ckpt, dev)
    cfg, spec = meta.cfg, meta.spec
    cfg = type(cfg)(**{**cfg.__dict__, "n_frames": args.n_frames})
    over = argparse.Namespace(data=args.data, max_rows=args.max_rows, n_sim=args.n_sim)
    data = load_data(data_args_from(meta.args, over))
    train = subset(data["train"], args.train_objects, args.seed)
    val = subset(data["validation"], args.val_objects, args.seed)
    val_index = subset(list(range(len(data["validation"]))), args.val_objects, args.seed)  # the same draw: positions in the full split
    grid = FrequencyGrid(args.p_min, args.p_max, args.grid_rel)
    freqs_t = torch.as_tensor(grid.freqs, dtype=torch.float32, device=dev)
    loader = frame_loader(train, cfg, spec, args.batch_size, train=True, workers=args.workers, seed=args.seed, pin_memory=args.pin_memory)
    val_loader = frame_loader(val, cfg, spec, args.batch_size, train=False, workers=min(args.workers, 4), seed=args.seed)
    head = PeakNet(enc.dim, args.ch, args.depth, args.kernel).to(dev)
    print(f"{len(train)} train / {len(val)} val records; grid {grid.n} bins ({args.p_min} to {args.p_max} d, step {args.grid_rel:.1e}, "
          f"x2 = {grid.shift} bins); head {n_params(head) / 1e6:.2f}M params; {len(loader)} steps per epoch; {dev}; loaded in {time.time() - t_start:.0f}s", flush=True)  # fmt: skip
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = LambdaLR(opt, cosine_schedule(args.steps, args.warmup))
    step, epoch, elapsed, last_metrics, n_skipped = 0, 0, 0.0, None, 0
    if ckpt is not None:
        head.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        step, epoch, elapsed = ckpt["step"], ckpt["epoch"], ckpt["elapsed"]
        last_metrics = ckpt.get("metrics")
    tracker = Tracker(args, config=dict(vars(args), params_head=n_params(head), grid_bins=grid.n), run_id=ckpt.get("wandb_id") if ckpt else None, job_type="period-head")
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    log = JsonlLog(out / "log.jsonl")
    hmeta = dict(ckpt=str(args.ckpt), wm_step=meta.step, kind=KIND, frames=cfg.__dict__, spec=spec.to_dict(), no_latent=args.no_latent)
    train_periods = torch.tensor([float(r.period or float("nan")) for r in train], dtype=torch.float64)

    def save(path, metrics, with_opt=True):
        state = dict(kind=KIND, state_dict=head.state_dict(), hparams=head.hparams, grid=grid.to_dict(), meta=hmeta, step=step,
                     metrics=metrics, args=dict(vars(args)), epoch=epoch, elapsed=elapsed, wandb_id=tracker.id)  # fmt: skip
        if with_opt:
            state.update(opt=opt.state_dict(), sched=sched.state_dict())
        save_atomic(state, path)

    def run_eval(train_loss, keep=False):
        t0 = time.time()
        m = evaluate(enc, head, spec, val_loader, val, dev, grid, freqs_t, keep_predictions=keep, no_latent=args.no_latent)
        preds = m.pop("predictions", None)
        m.update(step=step, elapsed=elapsed, train_loss=train_loss, eval_seconds=time.time() - t0)
        log.write(kind="eval", **m)
        flat = {"train_mean/loss": train_loss}
        for lvl in ("window", "object"):
            for who in ("head", "ls"):
                for k, v in m[lvl][who].items():
                    flat[f"val/{lvl}_{who}_{k}"] = v
        tracker.log(flat, step=step)
        return m, preds

    head.train()
    timer = StepTimer(dev)
    run, n_acc, t_last, t_run, stop = 0.0, 0, time.time(), time.time(), False
    while step < args.steps and not stop:
        torch.manual_seed(args.seed + epoch)
        timer.reset()
        for batch in loader:
            timer.got_batch()
            frames = [f.to(dev) for f in batch["frames"]]
            periods = train_periods[batch["index"]].repeat(len(frames))
            coord = torch.as_tensor(grid.bin_of(periods.numpy()), dtype=torch.float32, device=dev)
            with torch.no_grad(), amp:
                z, chans, _ = batch_spectra(enc, spec, frames, dev, grid, freqs_t, no_latent=args.no_latent)
            with amp:
                logits, offset = head(chans, z)
                loss = head_loss(logits, offset, coord, args.target_sigma)
            opt.zero_grad(set_to_none=True)
            finite = bool(torch.isfinite(loss))
            if finite:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(head.parameters(), args.clip)
                finite = bool(torch.isfinite(gn))
            if finite:
                opt.step()
            else:
                opt.zero_grad(set_to_none=True)
                n_skipped += 1
                if n_skipped > 1000:
                    raise RuntimeError("more than 1000 non-finite updates: stopping")
            sched.step()
            timer.done_step()
            step += 1
            if finite:
                run += loss.item()
                n_acc += 1
            if step % args.log_every == 0:
                now = time.time()
                rate, t_last = (now - t_last) / args.log_every, now
                perf, gpu = timer.report(), gpu_stats(dev)
                timer.reset()
                lr = sched.get_last_lr()[0]
                print(f"step {step:6d}  loss {run / max(n_acc, 1):.4f}  lr {lr:.2e}  {rate:.3f} s/step"
                      f"  (data {perf['data_frac']:.0%}, gpu {gpu.get('gpu_util', float('nan')):.0f}%)", flush=True)  # fmt: skip
                log.write(kind="train", step=step, loss=run / max(n_acc, 1), lr=lr, skipped=n_skipped, s_per_step=rate, **perf, **gpu)
                tracker.log({"train/loss": run / max(n_acc, 1), "train/lr": lr, "perf/s_per_step": rate, **{f"sys/{k}": v for k, v in gpu.items()}}, step=step)
            if step % args.eval_every == 0 or step == args.steps:
                elapsed, t_run = elapsed + time.time() - t_run, time.time()
                last_metrics, _ = run_eval(run / max(n_acc, 1))
                run, n_acc = 0.0, 0
                t_run = time.time()
                timer.reset()
            budget = bool(args.time_budget) and (time.time() - t_start) > args.time_budget
            if step % args.ckpt_every == 0 or step == args.steps or budget:
                elapsed, t_run = elapsed + time.time() - t_run, time.time()
                save(last, last_metrics)
            if budget and step < args.steps:
                print(f"time budget reached at step {step}; checkpoint saved", flush=True)
                stop = True
                break
            if step >= args.steps:
                break
        else:
            epoch += 1
    if step < args.steps:
        return
    last_metrics, preds = run_eval(float("nan"), keep=True)
    save(out / "period_head.pt", last_metrics, with_opt=False)
    if preds is not None:
        preds["index"] = np.array([val_index[int(i)] for i in preds["index"]])  # indices into the full validation split
        np.savez_compressed(out / "predictions.npz", **preds)
    dump_json(last_metrics, out / "results.json")
    (out / "DONE").write_text(f"{step} steps, {elapsed / 3600:.2f} h\n")
    print(f"done: {step} steps in {elapsed / 3600:.2f} h; saved {out / 'period_head.pt'}")
    if hasattr(tracker, "finish"):
        tracker.finish()


if __name__ == "__main__":
    main()
