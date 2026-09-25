"""Throughput benchmark of stage-1 training: which loader settings keep the
GPU busy? Each configuration builds a fresh loader (and model), warms up,
then times ``--steps`` optimizer steps while a thread samples the GPU
utilisation; the table reports seconds per step, sequences per second, the
fraction of the step spent waiting for data, and the mean GPU utilisation.
With ``--wandb`` every configuration is one run in the ``bench`` group.

    python -m project.bench --rope-wavelengths ... --time-scale 0.0015915 --wandb
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from project.common import (
    TokenSpec,
    add_data_args,
    add_device_arg,
    add_frame_args,
    add_ladder_args,
    add_model_args,
    build_backbone,
    build_world_model,
    err_stats,
    frame_config,
    frame_loader,
    get_device,
    get_ladder,
    load_data,
    seed_all,
    time_block_dim,
    to_device,
)
from project.tracking import StepTimer, Tracker, add_wandb_args, gpu_stats

CONFIGS = {
    "w8_unfused": dict(workers=8, fused=False),
    "w8": dict(workers=8),
    "w8_pin": dict(workers=8, pin_memory=True),
    "w16_pin": dict(workers=16, pin_memory=True),
    "w8_persist_pin": dict(workers=8, persistent=True, prefetch=4, pin_memory=True),
    "w8_pin_b128": dict(workers=8, pin_memory=True, batch_size=128),
    "w8_pin_compile": dict(workers=8, pin_memory=True, compile=True),
}
DEFAULT_CONFIGS = [k for k in CONFIGS if "compile" not in k]


class GpuSampler(threading.Thread):
    def __init__(self, device, interval=0.5):
        super().__init__(daemon=True)
        self.device, self.interval, self.samples, self.stop = (
            device,
            interval,
            [],
            False,
        )

    def run(self):
        while not self.stop:
            g = gpu_stats(self.device)
            if "gpu_util" in g:
                self.samples.append(g["gpu_util"])
            time.sleep(self.interval)


def run_config(name, conf, args, train, cfg, spec, ladder, dev):
    bs = conf.get("batch_size", args.batch_size)
    loader = frame_loader(
        train,
        cfg,
        spec,
        bs,
        train=True,
        workers=conf.get("workers", 0),
        seed=args.seed,
        persistent=conf.get("persistent", False),
        prefetch=conf.get("prefetch", 2),
        pin_memory=conf.get("pin_memory", False),
    )
    backbone = build_backbone(args, ladder, spec.n_channels)
    model = build_world_model(
        args, backbone, args.n_frames, fused=conf.get("fused", True)
    ).to(dev)
    if conf.get("compile"):
        model.backbone.forward = torch.compile(model.backbone.forward, dynamic=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    model.train()
    timer = StepTimer(dev)
    sampler = GpuSampler(dev)
    it = iter(loader)
    n_warm, n_time, step = args.warmup, args.steps, 0
    t0 = time.time()
    while step < n_warm + n_time:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if step == n_warm:
            timer.reset()
            sampler.start()
        timer.got_batch()
        frames, actions = to_device(batch, dev)
        with amp:
            o = model(frames, actions)
        opt.zero_grad(set_to_none=True)
        o.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        timer.done_step()
        step += 1
    sampler.stop = True
    sampler.join(timeout=2)
    rep = timer.report()
    s_per_step = rep["data_s"] + rep["compute_s"]
    out = dict(
        config=name,
        batch_size=bs,
        **{k: v for k, v in conf.items() if k != "batch_size"},
        s_per_step=s_per_step,
        seq_per_s=bs / s_per_step,
        data_frac=rep["data_frac"],
        data_s=rep["data_s"],
        compute_s=rep["compute_s"],
        gpu_util=float(np.mean(sampler.samples)) if sampler.samples else float("nan"),
        gpu_mem_gb=gpu_stats(dev).get("gpu_mem_used_gb", float("nan")),
        wall_s=time.time() - t0,
    )
    print(
        f"{name:22s} bs {bs:4d}  {s_per_step:.4f} s/step  {out['seq_per_s']:7.1f} seq/s  "
        f"data {rep['data_frac']:.0%}  gpu {out['gpu_util']:.0f}%  mem {out['gpu_mem_gb']:.1f} GB",
        flush=True,
    )
    del loader, it, model, opt
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_args(p)
    add_frame_args(p)
    add_ladder_args(p)
    add_model_args(p)
    add_wandb_args(p)
    p.add_argument(
        "--configs", nargs="*", default=DEFAULT_CONFIGS, choices=list(CONFIGS)
    )
    p.add_argument("--steps", type=int, default=150, help="timed steps per config")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--train-objects", type=int, default=None)
    p.add_argument("--out", default="project/results/bench")
    add_device_arg(p)
    args = p.parse_args(argv)
    seed_all(args.seed)
    dev = get_device(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(args, splits=("train",))
    train = (
        data["train"]
        if args.train_objects is None
        else data["train"][: args.train_objects]
    )
    cfg = frame_config(args)
    dim = time_block_dim(args.width, args.heads, args.p_rope)
    ladder = get_ladder(args, train, out, dim, args.p_rope)
    spec = TokenSpec(
        dict(band_wavelengths=data.wavelengths, time_scale=ladder.time_scale),
        None if args.no_err_channel else err_stats(train),
    )
    print(
        f"{len(train)} train records, {cfg}, {dev}; {len(args.configs)} configs x {args.steps} steps"
    )
    rows = []
    for name in args.configs:
        row = run_config(name, CONFIGS[name], args, train, cfg, spec, ladder, dev)
        rows.append(row)
        if args.wandb:
            a = argparse.Namespace(**vars(args))
            a.wandb_name, a.wandb_group, a.out = (
                f"bench-{name}",
                args.wandb_group or "bench",
                str(out),
            )
            tr = Tracker(
                a,
                config=dict(row, window=cfg.window, max_tokens=cfg.max_tokens),
                job_type="bench",
            )
            tr.log(
                {
                    f"bench/{k}": v
                    for k, v in row.items()
                    if isinstance(v, (int, float))
                },
                step=0,
            )
            tr.summary(
                **{k: v for k, v in row.items() if isinstance(v, (int, float, str))}
            )
            tr.finish()
    json.dump(rows, open(out / "bench.json", "w"), indent=1)
    best = min(rows, key=lambda r: r["s_per_step"] / r["batch_size"])
    print(
        f"best sequences per second: {best['config']} ({best['seq_per_s']:.1f} seq/s, gpu {best['gpu_util']:.0f}%)"
    )


if __name__ == "__main__":
    main()
