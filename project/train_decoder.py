"""Stage 3 (M5): a :class:`~project.decoder.QueryDecoder` on frozen latents,
and the imputation evaluation.

``--ckpt`` is the stage-1 autoencoder ``mae.pt`` (the frozen pooled CLS
feature of the MAE encoder; :func:`project.common.load_wm` wraps it in a
world model with an identity projector) or a stage-2 ``wm.pt`` of
``train_wm.py`` (its post-projector latent); either way the encoder gets no
gradient. Every window of the checkpoint's frame sequences is a training
example: the frozen encoder embeds it with ``--encoder-drop`` of its points
hidden, the
decoder is trained on all its points. Evaluation hides ``--holdout-frac`` of
the points of every validation window (``random`` points or one contiguous
``gap`` in time), encodes the rest and decodes the hidden ones; the headline
number is the NLL of the hidden magnitudes under the known errors,
``mean_j (m_j - mu_hat_j)^2 / (2 sigma_j^2) + log sigma_j + log(2 pi) / 2``,
next to the RMSE, both compared with the per-band baselines of
:mod:`project.baselines` (constant, linear interpolation, RBF Gaussian
process, and the periodic GP oracle that uses the catalogue period with
``--oracle``). Resumable like ``train_wm``; ``--wandb`` logs ``train/loss``
(the decoder loss), the validation imputation metrics, timing and GPU stats.

    python -m project.train_decoder --ckpt project/runs/mae_w250/mae.pt --kind flow \\
        --out project/runs/mae_w250/dec_flow --baselines --wandb
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from romae_lc import FrameDataset

from project import baselines as bl
from project.common import (
    JsonlLog,
    add_device_arg,
    cosine_schedule,
    data_args_from,
    dump_json,
    flat_layout,
    frame_loader,
    get_device,
    load_data,
    load_wm,
    n_params,
    save_atomic,
    seed_all,
    subset,
    superclass,
    to_device,
)
from project.decoder import (
    QueryDecoder,
    decode_mean,
    decoder_loss,
    decoder_state,
    drop_tokens,
    hide_tokens,
)
from project.tracking import StepTimer, Tracker, add_wandb_args, gpu_stats

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
    "val_objects",
    "baselines",
    "oracle",
    "sample_steps",
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


def choose_holdout(tokens, mode, frac, window, time_scale, generator):
    """``hide [B, N]``: a random fraction of the real tokens, or the real
    tokens inside one random gap of ``frac * window`` days."""
    real = ~tokens.pad_mask
    if mode == "random":
        u = torch.rand(real.shape, generator=generator)
        u[~real] = 2.0
        n_real = real.sum(1)
        n_hide = torch.where(
            n_real >= 2,
            (n_real.float() * frac).floor().clamp(min=1).long(),
            torch.zeros_like(n_real),
        )
        rank = u.argsort(1).argsort(1)
        return rank < n_hide[:, None]
    t = (tokens.positions[:, 0] - 1.0) * time_scale
    g = torch.rand(real.shape[0], generator=generator) * window * (1 - frac)
    return real & (t >= g[:, None]) & (t < (g + frac * window)[:, None])


@torch.no_grad()
def imputation_eval(
    model,
    dec,
    records,
    meta,
    device,
    holdout="random",
    frac=0.2,
    sample_steps=20,
    use_baselines=True,
    oracle=False,
    batch_size=64,
    seed=0,
    n_objects=1000,
    log=print,
):
    cfg, spec = meta.cfg, meta.spec
    recs = subset(records, n_objects, seed)
    ds = FrameDataset(recs, cfg, seed=seed, epoch_seed=False)
    loader = DataLoader(ds, batch_size, shuffle=False, collate_fn=spec.collate())
    gen = torch.Generator().manual_seed(seed)
    methods = ["decoder"] + (list(bl.BASELINES) if use_baselines else [])
    methods += ["gp_periodic"] if oracle else []
    acc = {m: dict(nll=[], rmse=[], nll_pred=[]) for m in methods}
    groups = []
    ts = spec.tokenize["time_scale"]
    model.eval()
    dec.eval()
    for batch in loader:
        for tok in batch["frames"]:
            hide = choose_holdout(tok, holdout, frac, cfg.window, ts, gen)
            rows = hide.any(1) & (~tok.pad_mask & ~hide).any(1)
            if not rows.any():
                continue
            enc_in = hide_tokens(tok, hide).to(device)
            z = model.encode([enc_in])[:, 0]
            tok_d = tok.to(device)
            mu = decode_mean(dec, z, tok_d, n_steps=sample_steps).cpu()
            var_pred = None
            if dec.kind == "mse":
                _, logvar = dec(
                    z, tok_d.positions, tok_d.pad_mask, dec.log_sigma(tok_d)
                )
                var_pred = tok.extras.float().square() + logvar.exp().cpu()
            m, sig = tok.values[..., 0].float(), tok.extras.float()
            for b in torch.nonzero(rows).flatten().tolist():
                h = hide[b]
                keep = ~tok.pad_mask[b] & ~h
                y, s = m[b, h].numpy(), sig[b, h].numpy()
                preds = {
                    "decoder": (
                        mu[b, h].numpy(),
                        None if var_pred is None else var_pred[b, h].numpy(),
                    )
                }
                rec = recs[batch["index"][b].item()]
                if use_baselines or oracle:
                    t_all = ((tok.positions[b, 0] - 1.0) * ts).numpy()
                    band = tok.positions[b, 1].numpy()
                    kn, hn = keep.numpy(), h.numpy()
                    ctx = (
                        t_all[kn],
                        m[b][keep].numpy(),
                        sig[b][keep].numpy(),
                        band[kn],
                    )
                    q = (t_all[hn], band[hn])
                    for name in methods:
                        if name == "decoder":
                            continue
                        if name == "gp_periodic":
                            mu_b, var_b = bl.per_band(
                                bl.gp_periodic, *ctx, *q, period=float(rec.period)
                            )
                        else:
                            mu_b, var_b = bl.per_band(bl.BASELINES[name], *ctx, *q)
                        preds[name] = (mu_b, var_b)
                for name, (mu_b, var_b) in preds.items():
                    acc[name]["nll"].append(
                        float(bl.gaussian_nll(y, mu_b, s**2).mean())
                    )
                    acc[name]["rmse"].append(float(np.sqrt(((y - mu_b) ** 2).mean())))
                    if var_b is not None:
                        acc[name]["nll_pred"].append(
                            float(bl.gaussian_nll(y, mu_b, var_b + s**2).mean())
                        )
                groups.append(superclass(rec))
    groups = np.array(groups)
    out = {}
    for name, v in acc.items():
        nll = np.array(v["nll"])
        out[name] = dict(
            nll=float(nll.mean()) if nll.size else None,
            rmse=float(np.mean(v["rmse"])) if v["rmse"] else None,
            nll_pred=float(np.mean(v["nll_pred"])) if v["nll_pred"] else None,
            per_superclass=(
                {
                    g: float(nll[groups == g].mean())
                    for g in sorted(set(groups.tolist()))
                }
                if nll.size
                else {}
            ),
        )
    out["n_windows"] = int(len(acc["decoder"]["nll"]))
    out["holdout"], out["frac"] = holdout, frac
    log(
        f"  imputation ({holdout}, {frac:.0%} hidden, {out['n_windows']} windows): "
        + "  ".join(
            f"{k} NLL {v['nll']:.3f} RMSE {v['rmse']:.3f}"
            for k, v in out.items()
            if isinstance(v, dict) and v.get("nll") is not None
        )
    )
    return out


def add_args(p):
    p.add_argument("--ckpt", required=True, help="stage-1 checkpoint")
    p.add_argument("--kind", choices=("flow", "mse"), default="flow")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--encoder-drop", type=float, default=0.25)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--out", default=None, help="default <ckpt dir>/dec_<kind>")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--time-budget", type=float, default=0.0)
    p.add_argument("--val-objects", type=int, default=1000)
    p.add_argument("--holdout", choices=("random", "gap"), default="random")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--sample-steps", type=int, default=20, help="flow ODE steps")
    p.add_argument("--baselines", action="store_true", help="also score the baselines")
    p.add_argument("--oracle", action="store_true", help="periodic-GP oracle (uses P)")
    p.add_argument("--data", default=None)
    p.add_argument("--classes", nargs="*", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--train-objects", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    add_device_arg(p)
    add_wandb_args(p)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_args(p)
    args = p.parse_args(argv)
    out = Path(args.out or (Path(args.ckpt).parent / f"dec_{args.kind}"))
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
    model, meta = load_wm(args.ckpt, dev)
    data = load_data(data_args_from(meta.args, args))
    train = subset(data["train"], args.train_objects, args.seed)
    val = subset(data["validation"], args.val_objects, args.seed)
    cfg, spec = meta.cfg, meta.spec
    err = spec.err_stats or (0.0, 1.0)
    print(
        f"{len(train)} train / {len(val)} val records; stage-1 step {meta.step}; "
        f"window {cfg.window}; loaded in {time.time() - t_start:.0f}s"
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
    dec = QueryDecoder(
        model.embed_dim,
        flat_layout(model.backbone),
        err,
        args.kind,
        args.width,
        args.heads,
        args.depth,
    ).to(dev)
    print(
        f"decoder {args.kind}: {n_params(dec) / 1e6:.2f}M params; "
        f"{len(loader)} steps per epoch; {dev}"
    )
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = LambdaLR(opt, cosine_schedule(args.steps, args.warmup))
    step, epoch, elapsed, last_metrics = 0, 0, 0.0, None
    if ckpt is not None:
        dec.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        step, epoch, elapsed = ckpt["step"], ckpt["epoch"], ckpt["elapsed"]
        last_metrics = ckpt.get("metrics")
    tracker = Tracker(
        args,
        config=dict(
            vars(args),
            params_decoder=n_params(dec),
            wm_step=meta.step,
            window=cfg.window,
        ),
        run_id=ckpt.get("wandb_id") if ckpt else None,
        job_type="decoder",
    )
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    log = JsonlLog(out / "log.jsonl")
    dmeta = dict(
        wm=str(args.ckpt),
        wm_step=meta.step,
        kind=args.kind,
        frames=asdict(cfg),
        spec=spec.to_dict(),
        encoder_drop=args.encoder_drop,
    )
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1)

    def save(path, metrics, with_opt=True):
        state = decoder_state(dec, dmeta, step, metrics)
        state.update(
            args=dict(vars(args)), epoch=epoch, elapsed=elapsed, wandb_id=tracker.id
        )
        if with_opt:
            state.update(opt=opt.state_dict(), sched=sched.state_dict())
        save_atomic(state, path)

    def evaluate(train_loss):
        t0 = time.time()
        dec.eval()
        m = imputation_eval(
            model,
            dec,
            val,
            meta,
            dev,
            args.holdout,
            args.holdout_frac,
            args.sample_steps,
            args.baselines,
            args.oracle,
            args.batch_size,
            args.seed,
        )
        dec.train()
        m.update(
            step=step,
            elapsed=elapsed,
            train_loss=train_loss,
            eval_seconds=time.time() - t0,
        )
        log.write(kind="eval", **m)
        flat = {"train_mean/loss": train_loss, "val/eval_seconds": m["eval_seconds"]}
        for name, v in m.items():
            if isinstance(v, dict) and v.get("nll") is not None:
                flat[f"val/nll_{name}"] = v["nll"]
                flat[f"val/rmse_{name}"] = v["rmse"]
                if v.get("nll_pred") is not None:
                    flat[f"val/nll_pred_{name}"] = v["nll_pred"]
                flat[f"val/nll_{name}_by_superclass"] = v["per_superclass"]
        tracker.log(flat, step=step)
        return m

    model.eval()
    dec.train()
    timer = StepTimer(dev)
    run, n_acc, t_last, t_run, stop = 0.0, 0, time.time(), time.time(), False
    while step < args.steps and not stop:
        torch.manual_seed(args.seed + epoch)
        timer.reset()
        for batch in loader:
            timer.got_batch()
            frames, _ = to_device(batch, dev)
            with torch.no_grad(), amp:
                z = model.encode(
                    [drop_tokens(f, args.encoder_drop, gen) for f in frames]
                )
            with amp:
                loss = sum(
                    decoder_loss(dec, z[:, t].float(), frames[t], gen)
                    for t in range(len(frames))
                ) / len(frames)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), args.clip)
            opt.step()
            sched.step()
            timer.done_step()
            step += 1
            run += loss.item()
            n_acc += 1
            if step % args.log_every == 0:
                now = time.time()
                rate, t_last = (now - t_last) / args.log_every, now
                perf, gpu = timer.report(), gpu_stats(dev)
                timer.reset()
                lr = sched.get_last_lr()[0]
                print(
                    f"step {step:6d}  loss {run / n_acc:.4f}  lr {lr:.2e}  {rate:.3f} s/step"
                    f"  (data {perf['data_frac']:.0%}, gpu {gpu.get('gpu_util', float('nan')):.0f}%)",
                    flush=True,
                )
                log.write(
                    kind="train",
                    step=step,
                    loss=run / n_acc,
                    lr=lr,
                    s_per_step=rate,
                    **perf,
                    **gpu,
                )
                tracker.log(
                    {
                        "train/loss": run / n_acc,
                        "train/lr": lr,
                        "perf/s_per_step": rate,
                        "perf/data_frac": perf["data_frac"],
                        "perf/data_s": perf["data_s"],
                        "perf/compute_s": perf["compute_s"],
                        **{f"sys/{k}": v for k, v in gpu.items()},
                    },
                    step=step,
                )
            if step % args.eval_every == 0 or step == args.steps:
                elapsed, t_run = elapsed + time.time() - t_run, time.time()
                last_metrics = evaluate(run / max(n_acc, 1))
                run, n_acc = 0.0, 0
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
        save(out / "dec.pt", last_metrics, with_opt=False)
        (out / "DONE").write_text(f"{step} steps, {elapsed / 3600:.2f} h\n")
        print(f"done: {step} steps in {elapsed / 3600:.2f} h; saved {out / 'dec.pt'}")
    tracker.finish()


if __name__ == "__main__":
    main()
