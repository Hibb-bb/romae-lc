"""The token read-out: a fine period from the encoder's own token features.

The pooled vector holds the period to about 1 %. The encoder also outputs
one feature vector per observation, and pooling to the CLS token is where
fine time structure may be lost. This head reads those token features at
full resolution, with no periodogram anywhere: a few learned queries attend
over the window's tokens (each token = the encoder's feature plus the
sines and cosines of its time since the window's start at the ladder's
rungs, so the head can relate features to times), the pooled result goes
through an MLP, and the output is a logit and a sub-bin offset per bin of
a fine log-period grid (``--grid-rel``, 0.4 % by default, 2,500 bins; the
catalogue period is the training target, softened over neighbouring bins).

If this reaches 0.1 % where the pooled read-out stops at 1 %, the
information is in the network's tokens and the pooling was the loss. If it
stops at 1 % too, the encoder itself does not resolve the period finer,
and a layer that can (a spectral layer over fine rungs) is required.

    python -m project.token_head --ckpt project/runs/mae_w250/mae.pt --out project/runs/mae_w250/token_head
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from romae_lc.transformer import attention_mask

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
from project.period_head import TOLERANCES, FrequencyGrid, head_loss, hits
from project.tracking import StepTimer, Tracker, add_wandb_args, gpu_stats

KIND = "token_head"


class TokenPeakNet(nn.Module):
    """Learned queries attend over the window's tokens; the pooled vector
    gives a logit and an offset per bin of the fine grid."""

    def __init__(self, enc_dim: int, n_bins: int, timescales, d_model: int = 192, n_queries: int = 4, nhead: int = 4, hidden: int = 1024):
        super().__init__()
        self.enc_dim, self.n_bins, self.d_model, self.n_queries, self.nhead, self.hidden = int(enc_dim), int(n_bins), int(d_model), int(n_queries), int(nhead), int(hidden)
        ts = torch.as_tensor([float(v) for v in timescales], dtype=torch.float32)
        self.register_buffer("timescales", ts)
        self.inp = nn.Linear(enc_dim + 2 * ts.numel(), d_model)
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(n_queries * d_model, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.logit = nn.Linear(hidden, n_bins)
        self.offset = nn.Linear(hidden, n_bins)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    @property
    def hparams(self) -> dict:
        return dict(enc_dim=self.enc_dim, n_bins=self.n_bins, timescales=self.timescales.tolist(), d_model=self.d_model,
                    n_queries=self.n_queries, nhead=self.nhead, hidden=self.hidden)  # fmt: skip

    def forward(self, tokens, t_pos, pad):
        """``tokens [B, N, enc_dim]`` (the encoder's features), ``t_pos [B,
        N]`` the time axis in position units (origin 1.0 = window start),
        ``pad [B, N]``. Returns ``(logits, offset)`` each ``[B, n_bins]``."""
        ang = (t_pos.float() - 1.0)[..., None] / self.timescales
        feats = torch.cat([tokens.float(), torch.sin(ang), torch.cos(ang)], -1)
        x = self.inp(feats.to(self.inp.weight.dtype))
        q = self.queries[None].expand(x.shape[0], -1, -1).to(x.dtype)
        pooled, _ = self.attn(q, x, x, key_padding_mask=pad)
        h = self.mlp(self.norm(pooled).flatten(1))
        return self.logit(h).float(), self.offset(h).float()


@torch.no_grad()
def batch_tokens(enc, frames):
    """``(tokens [R, N, D], t_pos [R, N], pad [R, N], empty [R])``: the
    encoder's token features of the fused windows; ``empty`` marks rows
    with no real token (a window of another object's longer sequence),
    which get one unmasked slot so the attention stays finite and are
    dropped from the loss."""
    values, positions, pad = fuse_frames(frames)
    x, pad_all = enc.model.encode(values, positions, pad)
    pad = pad_all[:, 1:].clone()
    empty = pad.all(1)
    pad[empty, 0] = False
    return x[:, 1:].float(), positions[:, 0], pad, empty


@torch.no_grad()
def evaluate(enc, head, loader, records, device, grid, log=print, keep_predictions=False):
    head.eval()
    win_h, win_p, obj_rows = [], [], {}
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for batch in loader:
        frames = [f.to(device) for f in batch["frames"]]
        b, tt = frames[0].values.shape[0], len(frames)
        idx = batch["index"].repeat(tt)
        with amp:
            toks, t_pos, pad, empty = batch_tokens(enc, frames)
            logits, offset = head(toks, t_pos, pad)
        logits[empty] = float("nan")
        best = logits.nan_to_num(0.0).argmax(1)
        coord = best.float() + offset.gather(1, best[:, None])[:, 0].clamp(-0.5, 0.5)
        p_h = grid.period_of(coord.cpu().numpy())
        p_t = np.array([float(records[int(i)].period or np.nan) for i in idx])
        p_t[empty.cpu().numpy()] = np.nan
        win_h.append(p_h), win_p.append(p_t)
        lp = F.log_softmax(logits.nan_to_num(0.0), 1)
        for r in range(b * tt):
            if bool(empty[r]):
                continue
            i = int(idx[r])
            o = obj_rows.setdefault(i, dict(lp=torch.zeros(logits.shape[1], device=device), n=0, p=p_t[r]))
            o["lp"] += lp[r]
            o["n"] += 1
    win_h, win_p = np.concatenate(win_h), np.concatenate(win_p)
    ok = np.isfinite(win_p) & (win_p > 0)
    res = dict(n_windows=int(ok.sum()), window=hits(win_h[ok], win_p[ok]))
    obj_i, obj_h, obj_p, obj_top = [], [], [], []
    for i, o in obj_rows.items():
        if not (np.isfinite(o["p"]) and o["p"] > 0):
            continue
        obj_i.append(i), obj_h.append(float(grid.period_of(int(o["lp"].argmax())))), obj_p.append(o["p"])
        obj_top.append(grid.period_of(o["lp"].topk(5).indices.cpu().numpy()))
    obj_h, obj_p = np.array(obj_h), np.array(obj_p)
    sup = np.array([superclass(records[i]) for i in obj_i])
    res["n_objects"] = len(obj_i)
    res["object"] = hits(obj_h, obj_p)
    res["object"]["by_superclass"] = {g: dict(n=int((sup == g).sum()), **hits(obj_h[sup == g], obj_p[sup == g])) for g in ("ECL", "RR", "ROT", "CEP", "DSCT", "LPV") if (sup == g).sum() >= 5}
    head.train()
    w, o = res["window"], res["object"]
    log(f"  token head ({res['n_windows']} windows, {res['n_objects']} objects): per window within 10%/1%/0.1%/0.01%: "
        f"{w['within_0.1']:.2f}/{w['within_0.01']:.2f}/{w['within_0.001']:.2f}/{w['within_0.0001']:.2f} || per object: "
        f"{o['within_0.1']:.2f}/{o['within_0.01']:.2f}/{o['within_0.001']:.2f}/{o['within_0.0001']:.2f} (alias {o['alias']:.2f}) | "
        + " ".join(f"{g} {v['within_0.01']:.2f}" for g, v in o["by_superclass"].items()))
    if keep_predictions:
        res["predictions"] = dict(index=np.array(obj_i), p_model=obj_h, p_catalogue=obj_p, superclass=sup, p_top=np.stack(obj_top))
    return res


RUN_CONTROL = ("steps", "time_budget", "workers", "device", "eval_every", "ckpt_every", "log_every", "no_resume", "wandb",
               "wandb_name", "wandb_project", "wandb_entity", "wandb_group", "wandb_tags", "val_objects", "pin_memory")  # fmt: skip


def add_args(p):
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--p-min", type=float, default=0.02)
    p.add_argument("--p-max", type=float, default=500.0)
    p.add_argument("--grid-rel", type=float, default=4e-3, help="relative bin width of the period grid")
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-queries", type=int, default=4)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--target-sigma", type=float, default=1.0)
    p.add_argument("--n-frames", type=int, default=4)
    p.add_argument("--steps", type=int, default=15_000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
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
    val_index = subset(list(range(len(data["validation"]))), args.val_objects, args.seed)
    grid = FrequencyGrid(args.p_min, args.p_max, args.grid_rel)
    # the ladder's rungs (position units) for the time features: the backbone's flat layout
    from project.common import flat_layout

    timescales = sorted({float(v) for b in flat_layout(enc.backbone) for v in b.get("timescales", []) if np.isfinite(v)}) or [1.0]
    loader = frame_loader(train, cfg, spec, args.batch_size, train=True, workers=args.workers, seed=args.seed, pin_memory=args.pin_memory)
    val_loader = frame_loader(val, cfg, spec, args.batch_size, train=False, workers=min(args.workers, 4), seed=args.seed)
    head = TokenPeakNet(enc.dim, grid.n, timescales, args.d_model, args.n_queries, 4, args.hidden).to(dev)
    print(f"{len(train)} train / {len(val)} val records; grid {grid.n} bins (step {args.grid_rel:.1e}); {len(timescales)} time features; "
          f"head {n_params(head) / 1e6:.2f}M params; {len(loader)} steps per epoch; {dev}; loaded in {time.time() - t_start:.0f}s", flush=True)  # fmt: skip
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = LambdaLR(opt, cosine_schedule(args.steps, args.warmup))
    step, epoch, elapsed, last_metrics, n_skipped = 0, 0, 0.0, None, 0
    if ckpt is not None:
        head.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        step, epoch, elapsed = ckpt["step"], ckpt["epoch"], ckpt["elapsed"]
        last_metrics = ckpt.get("metrics")
    tracker = Tracker(args, config=dict(vars(args), params_head=n_params(head), grid_bins=grid.n), run_id=ckpt.get("wandb_id") if ckpt else None, job_type="token-head")
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    log = JsonlLog(out / "log.jsonl")
    hmeta = dict(ckpt=str(args.ckpt), wm_step=meta.step, kind=KIND, frames=cfg.__dict__, spec=spec.to_dict())
    train_periods = torch.tensor([float(r.period or float("nan")) for r in train], dtype=torch.float64)

    def save(path, metrics, with_opt=True):
        state = dict(kind=KIND, state_dict=head.state_dict(), hparams=head.hparams, grid=grid.to_dict(), meta=hmeta, step=step,
                     metrics=metrics, args=dict(vars(args)), epoch=epoch, elapsed=elapsed, wandb_id=tracker.id)  # fmt: skip
        if with_opt:
            state.update(opt=opt.state_dict(), sched=sched.state_dict())
        save_atomic(state, path)

    def run_eval(train_loss, keep=False):
        t0 = time.time()
        m = evaluate(enc, head, val_loader, val, dev, grid, keep_predictions=keep)
        preds = m.pop("predictions", None)
        m.update(step=step, elapsed=elapsed, train_loss=train_loss, eval_seconds=time.time() - t0)
        log.write(kind="eval", **m)
        flat = {"train_mean/loss": train_loss}
        for lvl in ("window", "object"):
            for k, v in m[lvl].items():
                if isinstance(v, float):
                    flat[f"val/{lvl}_{k}"] = v
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
                toks, t_pos, pad, empty = batch_tokens(enc, frames)
            coord = coord.masked_fill(empty, float("nan"))  # empty rows carry no window: out of the loss
            with amp:
                logits, offset = head(toks, t_pos, pad)
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
                print(f"step {step:6d}  loss {run / max(n_acc, 1):.4f}  lr {lr:.2e}  {rate:.3f} s/step  (data {perf['data_frac']:.0%}, gpu {gpu.get('gpu_util', float('nan')):.0f}%)", flush=True)
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
    save(out / "token_head.pt", last_metrics, with_opt=False)
    if preds is not None:
        preds["index"] = np.array([val_index[int(i)] for i in preds["index"]])
        np.savez_compressed(out / "predictions.npz", **preds)
    dump_json(last_metrics, out / "results.json")
    (out / "DONE").write_text(f"{step} steps, {elapsed / 3600:.2f} h\n")
    print(f"done: {step} steps in {elapsed / 3600:.2f} h; saved {out / 'token_head.pt'}")
    if hasattr(tracker, "finish"):
        tracker.finish()


if __name__ == "__main__":
    main()
