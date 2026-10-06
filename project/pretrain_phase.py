"""Stage 1 with a phase bottleneck: an encoder whose pooled latent must
hold the phase of the window as well as its shape.

The masked autoencoder never stores the phase in its pooled vector, because
its decoder reads the visible tokens and their time differences
(``phase_probe``: exactly chance, 2026-09-30). Here the decoder is the
:class:`project.phase_decoder.PhaseDecoder` with no context: it sees only
the pooled latent and, per query point of the SAME window, the point's
phase counted from the window's start (``phi = (t_rel / P) mod 1``, ``P``
the catalogue period, a label used only in the coordinate) and its band.
"0.37 cycles after the window started" says nothing about where in the
star's cycle the window started, so the only way to reconstruct the window
is for the latent to carry that offset next to the shape, amplitude and
level. The encoder is trained end to end through this decoder; a fraction
of the points is hidden from the encoder (``--encoder-drop``) so it cannot
simply memorise the batch.

The checkpoint is a stage-1 ``mae.pt`` like the MAE's (the encoder in a
:class:`~romae_lc.RoMAEForPreTraining` shell with an untrained MAE head),
so ``cache_latents``, the probes and every later stage read it as usual;
the phase decoder's weights sit next to it in ``phase_decoder.pt``. The
checks afterwards: the phase probe must leave chance, the period probe
must keep what the MAE had.

    python -m project.pretrain_phase --out project/runs/phase_bn --window 250 --min-tokens 16 --max-tokens 256
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from romae_lc import RoMAEForPreTraining, Tokens

from project.common import (
    JsonlLog,
    Ladder,
    TokenSpec,
    add_data_args,
    add_device_arg,
    add_frame_args,
    add_ladder_args,
    add_model_args,
    cosine_schedule,
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
    resolve_model_args,
    rope_geometry,
    rope_layouts,
    save_atomic,
    seed_all,
    subset,
)
from project.decoder import drop_tokens
from project.phase_decoder import PhaseDecoder, band_ids, band_table
from project.tracking import StepTimer, Tracker, add_wandb_args, gpu_stats

RUN_CONTROL = ("steps", "time_budget", "workers", "device", "eval_every", "ckpt_every", "log_every", "persistent_workers",
               "prefetch", "pin_memory", "no_resume", "wandb", "wandb_name", "wandb_project", "wandb_entity", "wandb_group",
               "wandb_tags", "val_objects")  # fmt: skip


def add_train_args(p):
    g = p.add_argument_group("training")
    g.add_argument("--steps", type=int, default=15_000)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--wd", type=float, default=0.05)
    g.add_argument("--warmup", type=float, default=0.02)
    g.add_argument("--clip", type=float, default=1.0)
    g.add_argument("--encoder-drop", type=float, default=0.25, help="points hidden from the encoder, decoded anyway")
    g.add_argument("--init", default=None, help="start the encoder from this stage-1 mae.pt (its ladder and token spec are reused)")
    g.add_argument("--dec-width", type=int, default=192)
    g.add_argument("--dec-depth", type=int, default=3)
    g.add_argument("--dec-heads", type=int, default=3)
    g.add_argument("--n-harm", type=int, default=8, help="phase harmonics of the decoder")
    g.add_argument("--mask-ratio", type=float, default=0.5, help="of the (unused) MAE head in the saved shell")
    g.add_argument("--eval-every", type=int, default=2500)
    g.add_argument("--ckpt-every", type=int, default=500)
    g.add_argument("--log-every", type=int, default=100)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--persistent-workers", action="store_true")
    g.add_argument("--prefetch", type=int, default=2)
    g.add_argument("--pin-memory", action="store_true")
    g.add_argument("--no-resume", action="store_true")
    g.add_argument("--time-budget", type=float, default=0.0)
    g.add_argument("--val-objects", type=int, default=1000)
    g.add_argument("--out", required=True)
    add_device_arg(p)


def phase_targets(frames, periods_t, spec, device):
    """Fused tokens of ``T`` frames of ``B`` objects (``[B T, N, C]`` rows
    frame-major) with, per row: the catalogue period, the phase of every
    token from the window's start, the band id and sigma."""
    values, positions, pad = fuse_frames(frames)
    b, t = frames[0].values.shape[0], len(frames)
    period = periods_t.to(device).repeat(t)  # rows t B .. (t + 1) B hold frame t
    ts = spec.tokenize["time_scale"]
    t_rel = (positions[:, 0].double() - 1.0) * ts
    phase = torch.remainder(t_rel / period.clamp_min(1e-6)[:, None], 1.0).float()
    band = band_ids(positions[:, 1], band_table(spec))
    mu, sd = spec.err_stats
    sigma = torch.exp(values[..., 1].float() * sd + mu)
    return values, positions, pad, phase, band, sigma, period


def step_loss(model, dec, values, positions, pad, phase, band, sigma, period, drop, gen):
    """The window reconstructed from its own pooled latent at the phases of
    its points: Gaussian NLL with variance ``sigma^2 + exp(logvar)`` over
    the real points of rows with a period."""
    tok_in = drop_tokens(Tokens(values, positions, pad), drop, gen) if drop > 0 else Tokens(values, positions, pad)
    x, _ = model.encode(tok_in.values, tok_in.positions, tok_in.pad_mask)
    z = x[:, 0].float()
    mu, logvar = dec(z, phase, band, pad, None)
    var = sigma.square() + torch.exp(logvar.float().clamp(-14.0, 10.0))
    m = values[..., 0].float()
    nll = 0.5 * ((m - mu.float()).square() / var + var.log())
    mask = (~pad) & torch.isfinite(period)[:, None] & (period > 0)[:, None]
    return (nll * mask).sum() / mask.sum().clamp(min=1), mu, mask


@torch.no_grad()
def evaluate(model, dec, loader, records, spec, device, drop):
    """Validation reconstruction: NLL per point and skill against the
    window's own median (1 = perfect, 0 = a flat line at the level)."""
    model.eval()
    dec.eval()
    nlls, skills = [], []
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    gen = torch.Generator(device=device).manual_seed(0)
    for batch in loader:
        frames = [f.to(device) for f in batch["frames"]]
        periods = torch.tensor([float(records[int(i)].period or float("nan")) for i in batch["index"]], dtype=torch.float64)
        with amp:
            values, positions, pad, phase, band, sigma, period = phase_targets(frames, periods, spec, device)
            loss, mu, mask = step_loss(model, dec, values, positions, pad, phase, band, sigma, period, drop, gen)
        nlls.append(float(loss))
        m = values[..., 0].float()
        for r in range(m.shape[0]):
            k = mask[r]
            if k.sum() < 4:
                continue
            y, p = m[r, k], mu[r, k].float()
            skills.append(float(1 - ((y - p) ** 2).mean() / ((y - y.median()) ** 2).mean().clamp_min(1e-12)))
    model.train()
    dec.train()
    return dict(val_nll=float(np.mean(nlls)) if nlls else float("nan"), val_skill=float(np.median(skills)) if skills else float("nan"), n_windows=len(skills))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    train, val = data["train"], subset(data["validation"], args.val_objects, args.seed)
    cfg = frame_config(args)
    init = torch.load(args.init, map_location="cpu", weights_only=False) if args.init else None
    if ckpt is not None:
        ladder = Ladder.from_dict(ckpt["ladder"])
        spec = TokenSpec.from_dict(ckpt["spec"])
    elif init is not None:  # the same ladder and token spec as the encoder we start from
        ladder = Ladder.from_dict(init["ladder"])
        spec = TokenSpec.from_dict(init["spec"])
        print(f"encoder initialised from {args.init} (step {init.get('step')}); its ladder and token spec reused")
    else:
        ladder = get_ladder(args, train, out, rope_geometry(args))
        print(ladder.summary or "ladder reused")
        spec = TokenSpec(dict(band_wavelengths=data.wavelengths, time_scale=ladder.time_scale), err_stats(train))
    if spec.err_stats is None:
        parser.error("the phase bottleneck needs the error channel")
    dump_json(vars(args), out / "args.json")
    loader = frame_loader(train, cfg, spec, args.batch_size, train=True, workers=args.workers, seed=args.seed,
                          persistent=args.persistent_workers, prefetch=args.prefetch, pin_memory=args.pin_memory)  # fmt: skip
    val_loader = frame_loader(val, cfg, spec, args.batch_size, train=False, workers=min(args.workers, 4), seed=args.seed)
    model = RoMAEForPreTraining(
        decoder=dict(d_model=args.dec_width, nhead=args.dec_heads, depth=1, attention=args.attention),
        mask_ratio=args.mask_ratio, target_channels=1, encoder=encoder_config(args), n_channels=spec.n_channels,
        n_axes=2, rope=rope_layouts(args, ladder),
    ).to(dev)  # fmt: skip
    n_bands = len(spec.tokenize["band_wavelengths"])
    dec = PhaseDecoder(model.embed_dim, model.embed_dim, args.dec_width, args.dec_heads, args.dec_depth, args.n_harm, n_bands,
                       latent_drop=0.0).to(dev)  # fmt: skip
    params = [p for n, p in model.named_parameters() if not n.startswith(("decoder", "head", "encoder_to_decoder", "mask_token"))]
    if init is not None and ckpt is None:
        own = model.state_dict()
        take = {k: v for k, v in init["state_dict"].items() if k in own and own[k].shape == v.shape
                and k.startswith(("projection", "transformer", "cls", "abs_proj", "rope"))}  # fmt: skip
        missing = [k for k in own if k.startswith(("projection", "transformer", "cls")) and k not in take]
        if missing:
            raise SystemExit(f"--init does not match this encoder: missing {missing[:5]}")
        model.load_state_dict(take, strict=False)
        print(f"loaded {len(take)} encoder tensors from --init")
    print(f"{len(train)} train / {len(val)} val records; encoder {sum(p.numel() for p in params) / 1e6:.2f}M params, "
          f"phase decoder {n_params(dec) / 1e6:.2f}M; {len(loader)} steps per epoch; {dev}; loaded in {time.time() - t_start:.0f}s", flush=True)  # fmt: skip
    opt = torch.optim.AdamW(params + list(dec.parameters()), lr=args.lr, weight_decay=args.wd)
    sched = LambdaLR(opt, cosine_schedule(args.steps, args.warmup))
    step, epoch, elapsed, last_metrics, n_skipped = 0, 0, 0.0, None, 0
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"])
        dec.load_state_dict(ckpt["phase_decoder"]["state_dict"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        step, epoch, elapsed = ckpt["step"], ckpt["epoch"], ckpt["elapsed"]
        last_metrics = ckpt.get("metrics")
    tracker = Tracker(args, config=dict(vars(args), params_encoder=sum(p.numel() for p in params)), run_id=ckpt.get("wandb_id") if ckpt else None,
                      job_type="phase-bottleneck")  # fmt: skip
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    log = JsonlLog(out / "log.jsonl")
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1)
    train_periods = torch.tensor([float(r.period or float("nan")) for r in train], dtype=torch.float64)

    def save(path, metrics, with_opt=True):
        state = mae_state(model, spec, cfg, ladder, data.classes, args, step, metrics)
        state.update(kind="mae", stage1="phase", phase_decoder=dict(state_dict=dec.state_dict(), hparams=dec.hparams),
                     epoch=epoch, elapsed=elapsed, wandb_id=tracker.id)  # fmt: skip
        if with_opt:
            state.update(opt=opt.state_dict(), sched=sched.state_dict())
        save_atomic(state, path)

    def run_eval(train_loss):
        t0 = time.time()
        m = evaluate(model, dec, val_loader, val, spec, dev, args.encoder_drop)
        m.update(step=step, elapsed=elapsed, train_loss=train_loss, eval_seconds=time.time() - t0)
        log.write(kind="eval", **m)
        tracker.log({"train_mean/loss": train_loss, "val/nll": m["val_nll"], "val/skill": m["val_skill"]}, step=step)
        print(f"  eval @ {step}: val NLL {m['val_nll']:.4f} | skill vs the window's level {m['val_skill']:.3f} ({m['n_windows']} windows) | {time.time() - t0:.0f}s", flush=True)
        return m

    model.train()
    dec.train()
    timer = StepTimer(dev)
    run, n_acc, t_last, t_run, stop = 0.0, 0, time.time(), time.time(), False
    if ckpt is None:
        last_metrics = run_eval(float("nan"))
    while step < args.steps and not stop:
        torch.manual_seed(args.seed + epoch)
        timer.reset()
        for batch in loader:
            timer.got_batch()
            frames = [f.to(dev) for f in batch["frames"]]
            periods = train_periods[batch["index"]]
            with amp:
                values, positions, pad, phase, band, sigma, period = phase_targets(frames, periods, spec, dev)
                loss, _, _ = step_loss(model, dec, values, positions, pad, phase, band, sigma, period, args.encoder_drop, gen)
            opt.zero_grad(set_to_none=True)
            finite = bool(torch.isfinite(loss))
            if finite:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params + list(dec.parameters()), args.clip)
                finite = bool(torch.isfinite(gn))
            if finite:
                opt.step()
            else:
                opt.zero_grad(set_to_none=True)
                n_skipped += 1
                if n_skipped in (1, 10) or n_skipped % 100 == 0:
                    print(f"non-finite loss or gradient at step {step}: update skipped ({n_skipped} so far)", flush=True)
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
                last_metrics = run_eval(run / max(n_acc, 1))
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
    if step >= args.steps:
        save(out / "mae.pt", last_metrics, with_opt=False)
        save_atomic(dict(kind="phase", state_dict=dec.state_dict(), hparams=dec.hparams, step=step), out / "phase_decoder.pt")
        (out / "DONE").write_text(f"{step} steps, {elapsed / 3600:.2f} h\n")
        print(f"done: {step} steps in {elapsed / 3600:.2f} h; saved {out / 'mae.pt'}")
    if hasattr(tracker, "finish"):
        tracker.finish()


if __name__ == "__main__":
    main()
