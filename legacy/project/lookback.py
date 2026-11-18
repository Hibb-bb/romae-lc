"""Stage 3b: a decoder trained on the forecast task, with a lookback.

The plain :class:`~project.decoder.QueryDecoder` gets one latent and the
query times, and the frozen latent holds no phase, so it draws a flat
line (2026-09-29). This decoder also sees the past: the frozen encoder's
per-token features of the last ``n_ctx`` windows before the target window,
placed on the same time axis as the queries (time relative to the target
window's start, so the rotary encoding relates every past point to every
future time). From the past it can fix the period and the phase; the
latent of the target window is there to supply what the past cannot: the
shape, amplitude and level of *that* window.

It is trained on forecasting, not on filling in: the target is a window
that starts one or two window lengths after the last context window, the
latent is the target's true latent (teacher forcing; at inference the
predictor's draw), and the loss is the Gaussian likelihood of the target's
observed magnitudes with variance ``sigma_j^2 + exp(logvar_j)``. The
query's own error is never an input. With ``--latent-drop`` the latent is
replaced by a learned "no latent" token for a share of the training
samples, so the same model also answers "what does the past alone give",
which measures what the latent adds.

    python -m project.lookback --ckpt project/runs/mae_w250/mae.pt --out project/runs/mae_w250/lookback
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from romae_lc import Tokens
from romae_lc.model import _init_weights
from romae_lc.rope import BlockRope
from romae_lc.transformer import Transformer, attention_mask, config

from project.cache_latents import object_windows
from project.common import (
    JsonlLog,
    add_device_arg,
    cosine_schedule,
    data_args_from,
    dump_json,
    flat_layout,
    get_device,
    load_data,
    load_encoder,
    n_params,
    save_atomic,
    seed_all,
    subset,
    superclass,
)
from project.decoder import gaussian_var
from project.tracking import StepTimer, Tracker, add_wandb_args, gpu_stats

KIND = "lookback"


# -------------------------------------------------------------------- model


class LookbackDecoder(nn.Module):
    """``(z, target queries, context tokens) -> (mu, logvar)`` per query.

    One rotary transformer over ``[latent token, context tokens, query
    tokens]`` with the encoder's own time ladder. Context tokens are the
    frozen encoder's token features plus the token's value channels
    (magnitude and standardised log sigma), projected to ``d_model``; query
    tokens are one learned vector, told apart only by their rotary position
    and band; the latent token sits at position 0 like the encoder's CLS.
    ``use_z`` (bool ``[B]``) swaps the latent token for the learned "no
    latent" token where False."""

    def __init__(
        self,
        z_dim: int,
        enc_dim: int,
        rope_layout: list[dict],
        err_stats: tuple[float, float],
        d_model: int = 192,
        nhead: int = 3,
        depth: int = 3,
        n_ctx: int = 3,
        latent_drop: float = 0.3,
        val_channels: int = 2,
    ):
        super().__init__()
        self.z_dim, self.enc_dim, self.n_ctx = int(z_dim), int(enc_dim), int(n_ctx)
        self.latent_drop, self.val_channels = float(latent_drop), int(val_channels)
        self.rope_layout = [dict(b) for b in rope_layout]
        self.cfg = config(dict(d_model=d_model, nhead=nhead, depth=depth))
        dims = sum(b["dim"] for b in rope_layout)
        if dims != self.cfg.head_dim:
            raise ValueError(
                f"decoder head_dim {self.cfg.head_dim} must equal the encoder's "
                f"rotary layout width {dims} (same d_model / nhead ratio)"
            )
        self.rope = BlockRope(self.cfg.head_dim, nhead, self.rope_layout)
        self.register_buffer("err_stats", torch.tensor(list(err_stats), dtype=torch.float32))
        self.z_proj = nn.Sequential(nn.Linear(z_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.no_z = nn.Parameter(torch.zeros(1, d_model))
        self.q_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.ctx_proj = nn.Linear(enc_dim + val_channels, d_model)
        self.type_emb = nn.Embedding(3, d_model)  # latent, context, query
        self.transformer = Transformer(self.cfg)
        self.norm = nn.RMSNorm(d_model, eps=self.cfg.norm_eps)
        self.head = nn.Linear(d_model, 2)
        self.apply(_init_weights)
        nn.init.normal_(self.no_z, std=0.02)
        nn.init.normal_(self.q_token, std=0.02)

    @property
    def kind(self) -> str:
        return KIND

    @property
    def hparams(self) -> dict:
        return dict(
            z_dim=self.z_dim, enc_dim=self.enc_dim, rope_layout=self.rope_layout,
            err_stats=self.err_stats.tolist(), d_model=self.cfg.d_model, nhead=self.cfg.nhead,
            depth=self.cfg.depth, n_ctx=self.n_ctx, latent_drop=self.latent_drop,
            val_channels=self.val_channels,
        )  # fmt: skip

    def forward(self, z, positions, pad_mask, ctx, use_z=None):
        """``z [B, Dz]``; query ``positions [B, n_axes, N]``, ``pad_mask [B,
        N]``; ``ctx`` = ``(feats [B, M, enc_dim + val_channels], positions
        [B, n_axes, M], pad [B, M])`` on the target's time axis (see
        :func:`encode_context`); ``use_z`` bool ``[B]`` or None (all True).
        Returns ``(mu, logvar)`` each ``[B, N]`` float32."""
        b, n = pad_mask.shape
        c_feats, c_pos, c_pad = ctx
        dt = self.q_token.dtype
        c = self.z_proj(z.to(dt))
        if use_z is not None:
            c = torch.where(use_z[:, None], c, self.no_z.to(dt).expand_as(c))
        x_c = self.ctx_proj(c_feats.to(dt)) + self.type_emb.weight[1]
        x_q = self.q_token.expand(b, n, -1) + self.type_emb.weight[2]
        x = torch.cat([(c + self.type_emb.weight[0])[:, None], x_c, x_q], 1)
        pos = torch.cat([positions.new_zeros(b, positions.shape[1], 1), c_pos.to(positions.dtype), positions], 2)
        pad = torch.cat([pad_mask.new_zeros(b, 1), c_pad, pad_mask], 1)
        x = self.transformer(x, self.rope.prepare(pos), attention_mask(pad))
        out = self.head(self.norm(x[:, -n:])).float()
        return out[..., 0], out[..., 1]


@torch.no_grad()
def encode_context(model, ctx_tokens: list[Tokens], offsets: torch.Tensor, time_scale: float, slot_real=None):
    """The context of a batch: the frozen encoder's token features of the
    ``K`` context windows (``ctx_tokens[i]`` = the batch's i-th context
    window, tokens on that window's own time axis), moved onto the target's
    time axis with ``offsets [B, K]`` (window start minus target start,
    days). ``slot_real [B, K]`` (bool) marks the slots that hold a window;
    the others are all padding. Returns ``(feats [B, M, D + C], positions
    [B, n_axes, M], pad [B, M])`` with ``M`` the windows' token counts
    summed; the CLS token is dropped."""
    feats, poss, pads = [], [], []
    for i, tok in enumerate(ctx_tokens):
        x, pad = model.encode(tok.values, tok.positions, tok.pad_mask)
        x, pad = x[:, 1:].float(), pad[:, 1:]  # drop CLS
        pos = tok.positions.clone().float()
        pos[:, 0] = pos[:, 0] + (offsets[:, i] / time_scale)[:, None]
        if slot_real is not None:
            pad = pad | ~slot_real[:, i][:, None]
        feats.append(torch.cat([x, tok.values.float()], -1))
        poss.append(pos)
        pads.append(pad)
    return torch.cat(feats, 1), torch.cat(poss, 2), torch.cat(pads, 1)


def lookback_loss(dec, z, tok: Tokens, ctx, use_z=None):
    """Gaussian NLL of the target's observed magnitudes, variance ``sigma^2
    + exp(logvar)``, mean over the real tokens."""
    m, pad = tok.values[..., 0].float(), tok.pad_mask
    mu, logvar = dec(z, tok.positions, pad, ctx, use_z)
    var = gaussian_var(tok.extras, logvar)
    nll = 0.5 * ((m - mu.float()).square() / var + var.log())
    mask = ~pad
    return (nll * mask).sum() / mask.sum().clamp(min=1)


# --------------------------------------------------------------------- data


class ForecastDataset(Dataset):
    """One forecast sample per record: a target window on the grid and the
    ``n_ctx`` non-overlapping windows that end where the target starts, minus
    a gap of ``horizon - 1`` window lengths. Windows are the grid of
    :func:`~project.cache_latents.object_windows` (stride, cap, seed as the
    latent cache); the target and the nearest context window must be valid
    (``cfg.min_tokens`` points), farther context windows are used when
    valid. A record with no valid pair gives ``None`` (dropped by the
    collate). With ``epoch_seed`` the draw changes every epoch."""

    def __init__(self, records, cfg, stride: float, cap, n_ctx: int, horizons, seed: int = 0, epoch_seed: bool = True):
        self.records, self.cfg, self.stride, self.cap = list(records), cfg, float(stride), cap
        self.n_ctx, self.horizons, self.seed, self.epoch_seed = int(n_ctx), [int(h) for h in horizons], int(seed), bool(epoch_seed)
        self.w = max(1, int(round(1.0 / self.stride)))  # grid steps per window
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        starts, frames, n_tokens = object_windows(r, self.cfg, self.stride, self.cap, self.seed, i)
        valid = np.asarray(n_tokens) >= self.cfg.min_tokens
        pairs = [(k, hz) for hz in self.horizons for k in range(len(starts))
                 if valid[k] and k - hz * self.w >= 0 and valid[k - hz * self.w]]  # fmt: skip
        if not pairs:
            return None
        rng = np.random.default_rng([self.seed, i, self.epoch if self.epoch_seed else 0])
        k, hz = pairs[int(rng.integers(len(pairs)))]
        j = k - hz * self.w
        ctx = [(frames[c], float(starts[c] - starts[k])) for c in range(j, j - self.n_ctx * self.w, -self.w) if c >= 0 and valid[c]]
        return dict(index=i, target=frames[k], start=float(starts[k]), horizon=hz, ctx=ctx, period=float(r.period or 0.0), superclass=superclass(r))


def make_collate(spec, n_ctx: int):
    """Tokens of the target and of every context slot, offsets and slot
    flags; ``None`` when the batch is empty."""
    dummy = (np.zeros(1, np.float32), np.zeros(1, np.float32), np.zeros(1, np.int64), np.ones(1, np.float32))

    def collate(items):
        items = [x for x in items if x is not None]
        if not items:
            return None
        b = len(items)
        target = spec.tokens([x["target"] for x in items])
        offsets = torch.zeros(b, n_ctx)
        real = torch.zeros(b, n_ctx, dtype=torch.bool)
        ctx = []
        for s in range(n_ctx):
            frames = []
            for bi, x in enumerate(items):
                if s < len(x["ctx"]):
                    frames.append(x["ctx"][s][0])
                    offsets[bi, s], real[bi, s] = x["ctx"][s][1], True
                else:
                    frames.append(dummy)
            ctx.append(spec.tokens(frames))
        return dict(
            target=target, ctx=ctx, offsets=offsets, slot_real=real,
            index=torch.tensor([x["index"] for x in items]), horizon=torch.tensor([x["horizon"] for x in items]),
            start=torch.tensor([x["start"] for x in items], dtype=torch.float64),
            superclass=[x["superclass"] for x in items], period=torch.tensor([x["period"] for x in items], dtype=torch.float64),
        )  # fmt: skip

    return collate


def batch_context(enc, spec, batch, device, target_z=True):
    """Move a collated batch to ``device`` and encode it: ``(target tokens,
    z [B, D] of the target (or None), ctx)``."""
    tgt = batch["target"].to(device)
    ctx_tok = [t.to(device) for t in batch["ctx"]]
    ctx = encode_context(enc.model, ctx_tok, batch["offsets"].to(device), spec.tokenize["time_scale"], batch["slot_real"].to(device))
    z = enc.encode([tgt])[:, 0].float() if target_z else None
    return tgt, z, ctx


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def forecast_eval(enc, dec, spec, loader, device, log=print) -> dict:
    """Forecast scores on the validation draws with the target's true
    latent: NLL per point under the known error plus the decoder's own
    variance, RMSE of the mean, skill against the constant made from the
    context (the median of the context magnitudes per band; 1 = perfect, 0
    = the constant), each with the latent and without it (the "no latent"
    token), per horizon and per superclass."""
    dec.eval()
    rows = []
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for batch in loader:
        if batch is None:
            continue
        with amp:
            tgt, z, ctx = batch_context(enc, spec, batch, device)
        m, pad, sig = tgt.values[..., 0].float(), tgt.pad_mask, tgt.extras.float()
        band = tgt.positions[:, 1]
        out = {}
        for name, use in (("model", True), ("no_latent", False)):
            flag = torch.full((z.shape[0],), use, device=device)
            with amp:
                mu, logvar = dec(z, tgt.positions, pad, ctx, flag)
            var = gaussian_var(sig, logvar)
            out[name] = (mu.float(), var)
        # the constant from the context: per band median of the context values
        c_feats, c_pos, c_pad = ctx
        c_val, c_band = c_feats[..., -2], c_pos[:, 1]
        for b in range(m.shape[0]):
            real = ~pad[b]
            if real.sum() == 0:
                continue
            y, s = m[b, real], sig[b, real]
            const = torch.zeros_like(y)
            for bd in band[b, real].unique():
                cm = (~c_pad[b]) & (c_band[b] == bd)
                const[band[b, real] == bd] = c_val[b, cm].median() if cm.any() else c_val[b, ~c_pad[b]].median()
            mse_c = ((y - const) ** 2).mean().clamp_min(1e-12)
            nll_c = (0.5 * ((y - const) ** 2 / (s**2 + (y - const).var().clamp_min(1e-6)) + (s**2 + (y - const).var().clamp_min(1e-6)).log() + np.log(2 * np.pi))).mean()
            row = dict(horizon=int(batch["horizon"][b]), superclass=batch["superclass"][b], n=int(real.sum()),
                       nll_const=float(nll_c), rmse_const=float(mse_c.sqrt()))  # fmt: skip
            for name, (mu, var) in out.items():
                mu_b, var_b = mu[b, real], var[b, real]
                nll = (0.5 * ((y - mu_b) ** 2 / var_b + var_b.log() + np.log(2 * np.pi))).mean()
                mse = ((y - mu_b) ** 2).mean()
                row[f"nll_{name}"], row[f"rmse_{name}"], row[f"skill_{name}"] = float(nll), float(mse.sqrt()), float(1 - mse / mse_c)
            rows.append(row)
    dec.train()
    if not rows:
        return dict(n=0)

    def med(key, sel):
        v = [r[key] for r in sel if np.isfinite(r[key])]
        return float(np.median(v)) if v else float("nan")

    keys = [k for k in rows[0] if k.startswith(("nll_", "rmse_", "skill_"))]

    def block(sel):
        return dict(n=len(sel), **{k: med(k, sel) for k in keys},
                    beats_const=float(np.mean([r["nll_model"] < r["nll_const"] for r in sel])),
                    beats_no_latent=float(np.mean([r["nll_model"] < r["nll_no_latent"] for r in sel])))  # fmt: skip

    res = block(rows)
    res["by_horizon"] = {str(h): block([r for r in rows if r["horizon"] == h]) for h in sorted({r["horizon"] for r in rows})}
    groups = sorted({r["superclass"] for r in rows}, key=lambda g: -sum(r["superclass"] == g for r in rows))
    res["by_superclass"] = {g: block([r for r in rows if r["superclass"] == g]) for g in groups if sum(r["superclass"] == g for r in rows) >= 10}
    log(
        f"  forecast ({res['n']} draws): NLL model {res['nll_model']:.3f} | no latent {res['nll_no_latent']:.3f} | "
        f"constant {res['nll_const']:.3f}; RMSE {res['rmse_model']:.3f} / {res['rmse_no_latent']:.3f} / {res['rmse_const']:.3f}; "
        f"skill {res['skill_model']:.3f} / {res['skill_no_latent']:.3f}; beats constant {res['beats_const']:.0%}, "
        f"beats no-latent {res['beats_no_latent']:.0%} | by horizon "
        + " ".join(f"h{h} skill {v['skill_model']:.3f}/{v['skill_no_latent']:.3f} ({v['n']})" for h, v in res["by_horizon"].items())
    )
    return res


# --------------------------------------------------------------- checkpoint


def lookback_state(dec, meta, step, metrics=None) -> dict:
    return dict(kind=KIND, state_dict=dec.state_dict(), hparams=dec.hparams, meta=meta, step=step, metrics=metrics)


def load_lookback(path, device="cpu") -> tuple[LookbackDecoder, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("kind") != KIND:
        raise ValueError(f"{path} is not a lookback decoder checkpoint")
    dec = LookbackDecoder(**ckpt["hparams"])
    dec.load_state_dict(ckpt["state_dict"])
    meta = dict(ckpt.get("meta", {}), step=ckpt.get("step"), metrics=ckpt.get("metrics"))
    return dec.to(device).eval(), meta


def is_lookback(path) -> bool:
    return torch.load(path, map_location="cpu", weights_only=False).get("kind") == KIND


# ------------------------------------------------------------------ training

RUN_CONTROL = ("steps", "time_budget", "workers", "device", "eval_every", "ckpt_every", "log_every",
               "persistent_workers", "prefetch", "pin_memory", "no_resume", "wandb", "wandb_name",
               "wandb_project", "wandb_entity", "wandb_group", "wandb_tags", "val_objects")  # fmt: skip


def add_args(p):
    p.add_argument("--ckpt", required=True, help="stage-1 checkpoint (the frozen encoder)")
    p.add_argument("--out", required=True)
    p.add_argument("--n-ctx", type=int, default=3, help="context windows")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2], help="window lengths ahead")
    p.add_argument("--stride", type=float, default=0.25, help="window grid stride (as the latent cache)")
    p.add_argument("--latent-drop", type=float, default=0.3, help="share of samples trained without the latent")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.02)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--prefetch", type=int, default=2)
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
    over = argparse.Namespace(data=args.data, max_rows=args.max_rows, n_sim=args.n_sim)
    data = load_data(data_args_from(meta.args, over))
    train = subset(data["train"], args.train_objects, args.seed)
    val = subset(data["validation"], args.val_objects, args.seed)
    err = spec.err_stats or (0.0, 1.0)
    collate = make_collate(spec, args.n_ctx)
    kw = dict(persistent_workers=args.persistent_workers and args.workers > 0,
              prefetch_factor=args.prefetch if args.workers > 0 else None, pin_memory=args.pin_memory)  # fmt: skip
    train_ds = ForecastDataset(train, cfg, args.stride, cfg.max_tokens, args.n_ctx, args.horizons, args.seed, epoch_seed=True)
    val_ds = ForecastDataset(val, cfg, args.stride, cfg.max_tokens, args.n_ctx, args.horizons, args.seed, epoch_seed=False)
    loader = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True, collate_fn=collate, num_workers=args.workers, **kw)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate, num_workers=min(args.workers, 4),
                            **dict(kw, persistent_workers=False))  # fmt: skip
    print(f"{len(train)} train / {len(val)} val records; stage-1 step {meta.step}; window {cfg.window:g}, "
          f"stride {args.stride:g}, {args.n_ctx} context windows, horizons {args.horizons}; loaded in {time.time() - t_start:.0f}s")  # fmt: skip
    dec = LookbackDecoder(enc.dim, enc.backbone.embed_dim, flat_layout(enc.backbone), err, args.width, args.heads,
                          args.depth, args.n_ctx, args.latent_drop).to(dev)  # fmt: skip
    print(f"lookback decoder: {n_params(dec) / 1e6:.2f}M params; {len(loader)} steps per epoch; {dev}")
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = LambdaLR(opt, cosine_schedule(args.steps, args.warmup))
    step, epoch, elapsed, last_metrics = 0, 0, 0.0, None
    best_nll, n_skipped = float("inf"), 0
    if ckpt is not None:
        dec.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        step, epoch, elapsed = ckpt["step"], ckpt["epoch"], ckpt["elapsed"]
        last_metrics = ckpt.get("metrics")
        best_nll = float(ckpt.get("best_nll", float("inf")))
        n_skipped = int(ckpt.get("skipped", 0))
    tracker = Tracker(args, config=dict(vars(args), params_decoder=n_params(dec), wm_step=meta.step, window=cfg.window),
                      run_id=ckpt.get("wandb_id") if ckpt else None, job_type="lookback")  # fmt: skip
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    log = JsonlLog(out / "log.jsonl")
    dmeta = dict(ckpt=str(args.ckpt), wm_step=meta.step, kind=KIND, frames=asdict(cfg), spec=spec.to_dict(),
                 stride=args.stride, n_ctx=args.n_ctx, horizons=list(args.horizons))  # fmt: skip
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1)

    def save(path, metrics, with_opt=True):
        state = lookback_state(dec, dmeta, step, metrics)
        state.update(args=dict(vars(args)), epoch=epoch, elapsed=elapsed, wandb_id=tracker.id, best_nll=best_nll, skipped=n_skipped)
        if with_opt:
            state.update(opt=opt.state_dict(), sched=sched.state_dict())
        save_atomic(state, path)

    def evaluate(train_loss):
        t0 = time.time()
        m = forecast_eval(enc, dec, spec, val_loader, dev)
        m.update(step=step, elapsed=elapsed, train_loss=train_loss, eval_seconds=time.time() - t0)
        log.write(kind="eval", **m)
        flat = {"train_mean/loss": train_loss, "val/eval_seconds": m["eval_seconds"]}
        for k in ("nll_model", "nll_no_latent", "nll_const", "rmse_model", "rmse_no_latent", "rmse_const",
                  "skill_model", "skill_no_latent", "beats_const", "beats_no_latent"):  # fmt: skip
            if k in m:
                flat[f"val/{k}"] = m[k]
        tracker.log(flat, step=step)
        return m

    dec.train()
    timer = StepTimer(dev)
    run, n_acc, t_last, t_run, stop = 0.0, 0, time.time(), time.time(), False
    while step < args.steps and not stop:
        torch.manual_seed(args.seed + epoch)
        train_ds.set_epoch(epoch)
        timer.reset()
        for batch in loader:
            timer.got_batch()
            if batch is None:
                continue
            with torch.no_grad(), amp:
                tgt, z, ctx = batch_context(enc, spec, batch, dev)
            use_z = torch.rand(z.shape[0], device=dev, generator=gen) >= args.latent_drop
            with amp:
                loss = lookback_loss(dec, z, tgt, ctx, use_z)
            opt.zero_grad(set_to_none=True)
            finite = bool(torch.isfinite(loss))
            if finite:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(dec.parameters(), args.clip)
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
                tracker.log({"train/loss": run / max(n_acc, 1), "train/lr": lr, "perf/s_per_step": rate,
                             "perf/data_frac": perf["data_frac"], **{f"sys/{k}": v for k, v in gpu.items()}}, step=step)  # fmt: skip
            if step % args.eval_every == 0 or step == args.steps:
                elapsed, t_run = elapsed + time.time() - t_run, time.time()
                last_metrics = evaluate(run / max(n_acc, 1))
                run, n_acc = 0.0, 0
                nll = last_metrics.get("nll_model")
                if nll is not None and nll == nll and nll < best_nll:
                    best_nll = float(nll)
                    save(out / "best.pt", last_metrics, with_opt=False)
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
        save(out / "dec.pt", last_metrics, with_opt=False)
        (out / "DONE").write_text(f"{step} steps, {elapsed / 3600:.2f} h\n")
        print(f"done: {step} steps in {elapsed / 3600:.2f} h; saved {out / 'dec.pt'}")
    tracker.finish() if hasattr(tracker, "finish") else None


if __name__ == "__main__":
    main()
