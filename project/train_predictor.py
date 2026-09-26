"""Stage 2: a conditional flow-matching predictor on cached frozen latents.

The stage-1 autoencoder (``pretrain_mae.py``, ``mae.pt``) learns period and
shape; training the encoder jointly with a next-latent loss erased them
(``SESSION-2026-09-25.md``). So the encoder is frozen, its latents are cached
once by :mod:`project.cache_latents` (``latents.pt``: one ``[D]`` row per
window of a grid with ``stride`` window lengths between starts) and this
script trains only the transition on them, the recipe of the molecular
world-model kit: the predictor is a conditional flow in latent space
anchored at the last latent, the training budget (hundreds of thousands of
steps on latents that live on the GPU) matters more than width, and every
number is reported next to a null.

Sequences: an object is drawn uniformly over its valid start windows (so
weighted by its number of valid windows, the analogue of drawing windows
uniformly), then ``--history`` advances ``a_1..a_H ~ U(--advance)`` each
rounded to the grid (``k = round(a / stride) >= 1`` steps; the model sees
the realised ``k * stride``); the rows ``i, i + k_1, i + k_1 + k_2, ...``
form the history and the target, and a sequence counts only when all its
rows are inside the object and valid (``n_tokens >= --min-tokens``).
``--advance-start LO HI`` with ``--advance-ramp`` is the optional curriculum
of ``train_wm``: the range ramps linearly from the start range to
``--advance`` over the first fraction of the steps.

Model (``--kind flow``, :class:`FlowPredictor`): latents are standardised
per dimension with the training mean and sd (buffers ``mu``, ``sd``); the
condition is the ``H`` history latents, the log of the ``H`` realised
advances through a 2-layer MLP and a sinusoidal embedding of the flow time;
the velocity network is a residual MLP (LayerNorm, Linear, SiLU, Linear
blocks with skips, zero-initialised output). The flow runs from ``x_0 = z_H
+ sigma0 eps`` to ``x_1 = z_{H+1}`` along ``x_s = (1 - s) x_0 + s x_1``,
so it only has to learn the change over one step; ``sample`` integrates it
with ``--n-euler`` Euler steps, ``predict_mean`` averages samples and
``log_prob`` is the exact density (:func:`anchored_log_density`: reverse
flow with a Hutchinson divergence and the Gaussian base at the anchor).
``--reproject norm`` rescales every sample to the mean training-latent
norm, the analogue of the kit's sphere re-projection; it is off by default
because this latent is not on a sphere, and the rollout norm drift is
logged instead. ``--kind mse`` (:class:`MsePredictor`) is the ablation: the
same conditioning and trunk regress the change ``z_{H+1} - z_H``.

Evaluation on a fixed set of validation sequences, every model number next
to its null: the loss; the one-step MSE in standardised units against
persistence (``z_H``) and the history mean; the NLL per dimension (flow)
against a Gaussian ``N(z_H + m, diag v)`` fitted to training changes; a
rollout repeating the last advance for ``--eval-horizons`` steps (MSE
against persistence, and the norm of the rolled latent over the norm of the
true one: a drift from 1 means the chain leaves the encoder's manifold);
and the sample spread against the spread of the true changes (a flow whose
samples do not spread has collapsed to the mean).

Resumable like ``train_wm``: ``--out/last.pt`` every ``--ckpt-every`` steps
and at ``--time-budget``; ``pred.pt`` and ``DONE`` at ``--steps``. With
``--wandb`` the train and val numbers go to Weights & Biases.

    python -m project.train_predictor --latents project/runs/mae_w250/latents.pt \\
        --out project/runs/pred_w250 --wandb
    python -m project.train_predictor --latents latents.pt --out /tmp/pred --kind mse \\
        --steps 20 --batch-size 8 --hidden 16 --depth 1 --device cpu
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
    dump_json,
    get_device,
    n_params,
    save_atomic,
    seed_all,
)
from project.flow import time_embedding
from project.tracking import Tracker, add_wandb_args, gpu_stats

#: Arguments read from the command line even when resuming a checkpoint.
RUN_CONTROL = (
    "advance_start",
    "advance_ramp",
    "steps",
    "time_budget",
    "device",
    "eval_every",
    "ckpt_every",
    "log_every",
    "out",
    "no_resume",
    "val_sequences",
    "eval_samples",
    "eval_horizons",
    "eval_batch",
    "wandb",
    "wandb_project",
    "wandb_entity",
    "wandb_name",
    "wandb_group",
    "wandb_tags",
)


# ------------------------------------------------------------------------ flow


def round_advance(a: torch.Tensor, stride: float) -> torch.Tensor:
    """Grid steps of advances ``a`` (window units): ``round(a / stride)``, at
    least 1."""
    return torch.round(a / stride).clamp(min=1).long()


def anchored_log_density(
    v_fn,
    x1: torch.Tensor,
    anchor: torch.Tensor,
    sigma0: float,
    n_steps: int = 20,
    n_probes: int = 1,
    generator=None,
) -> torch.Tensor:
    """``log p(x1) [B]`` under the flow ``dx/ds = v(x, s)`` whose base is
    ``x_0 ~ N(anchor, sigma0^2 I)`` rather than ``N(0, I)``: the reverse
    flow is integrated from ``s = 1`` to 0 with Euler steps and a Hutchinson
    estimate of ``div v`` (``n_probes`` Rademacher probes), like
    :func:`project.flow.log_density`, and the Gaussian log density of the
    endpoint ``x_0`` at the anchor is added. ``v_fn(x, s)`` takes ``x [B,
    D]`` and ``s [B]``; a field that ignores ``x`` has zero divergence, so on
    a zero field the result is the Gaussian log density of ``x1`` itself.
    Runs with autograd enabled inside."""
    x = x1.detach().clone()
    b, d = x.shape
    ds = 1.0 / n_steps
    logdet = torch.zeros(b, device=x.device)
    for i in range(n_steps, 0, -1):
        s = torch.full((b,), i * ds, device=x.device, dtype=torch.float32)
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            v = v_fn(x_req, s)
            div = torch.zeros(b, device=x.device)
            for _ in range(n_probes if v.requires_grad else 0):
                probe = (
                    torch.randint(0, 2, x.shape, device=x.device, generator=generator)
                    .float()
                    .mul(2)
                    .sub(1)
                )
                (jvp,) = torch.autograd.grad(
                    (v * probe).sum(), x_req, retain_graph=True, allow_unused=True
                )
                if jvp is not None:
                    div = div + (jvp * probe).sum(-1)
            div = div / n_probes
        x = x - ds * v.detach()
        # log p_1(x_1) = log p_0(x_0) - int_0^1 div v ds (instantaneous change
        # of variables): the integrated divergence is subtracted.
        logdet = logdet - ds * div.detach()
    base = -0.5 * (
        ((x - anchor) / sigma0).square().sum(-1)
        + d * math.log(2 * math.pi)
        + 2 * d * math.log(sigma0)
    )
    return base + logdet


# ---------------------------------------------------------------------- models


class _Conditioned(nn.Module):
    """What the two predictor kinds share: the standardisation buffers,
    the advance embedding and the residual MLP trunk. Every public method
    takes and returns raw latents; the standardisation is internal."""

    def __init__(self, dim, history, hidden, depth, in_dim, adv_dim=64):
        super().__init__()
        self.dim, self.history, self.hidden, self.depth, self.adv_dim = (
            int(dim),
            int(history),
            int(hidden),
            int(depth),
            int(adv_dim),
        )
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))
        self.register_buffer("norm_mean", torch.tensor(float(dim) ** 0.5))
        self.adv = nn.Sequential(
            nn.Linear(history, adv_dim), nn.SiLU(), nn.Linear(adv_dim, adv_dim)
        )
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(depth)
            ]
        )
        self.out = nn.Linear(hidden, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @torch.no_grad()
    def set_stats(self, z: torch.Tensor) -> None:
        """Standardisation from raw training latents ``z [N, D]`` (and the
        mean standardised norm for ``--reproject norm``)."""
        self.mu.copy_(z.mean(0))
        self.sd.copy_(z.std(0) + 1e-6)
        self.norm_mean.copy_(((z - self.mu) / self.sd).norm(dim=-1).mean())

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mu) / self.sd

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.sd + self.mu

    def cond(self, hs: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
        """``[B, H D + adv_dim]`` from standardised history ``hs [B, H, D]``
        and realised advances ``adv [B, H]`` (window units)."""
        return torch.cat([hs.flatten(1), self.adv(adv.float().log())], -1)

    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.inp(h))
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


class FlowPredictor(_Conditioned):
    """Conditional flow-matching transition anchored at the last latent (see
    the module docstring). ``loss``, ``sample``, ``predict_mean`` and
    ``log_prob`` take raw latents: ``hist [B, H, D]``, ``adv [B, H]``."""

    kind = "flow"

    def __init__(
        self,
        dim,
        history,
        hidden=512,
        depth=6,
        sigma0=0.3,
        adv_dim=64,
        s_dim=64,
        n_euler=20,
        reproject="none",
    ):
        super().__init__(
            dim, history, hidden, depth, dim + history * dim + adv_dim + s_dim, adv_dim
        )
        if reproject not in ("none", "norm"):
            raise ValueError(f"reproject must be none|norm, got {reproject!r}")
        self.sigma0, self.s_dim, self.n_euler, self.reproject = (
            float(sigma0),
            int(s_dim),
            int(n_euler),
            reproject,
        )

    @property
    def hparams(self) -> dict:
        return dict(
            kind=self.kind,
            dim=self.dim,
            history=self.history,
            hidden=self.hidden,
            depth=self.depth,
            sigma0=self.sigma0,
            adv_dim=self.adv_dim,
            s_dim=self.s_dim,
            n_euler=self.n_euler,
            reproject=self.reproject,
        )

    def v(self, x, s, hs, adv) -> torch.Tensor:
        """Velocity at standardised ``x [B, D]``, flow time ``s [B]``."""
        return self.trunk(
            torch.cat([x, self.cond(hs, adv), time_embedding(s, self.s_dim)], -1)
        )

    def loss(self, hist, target, adv, generator=None) -> torch.Tensor:
        hs, x1 = self.normalize(hist.float()), self.normalize(target.float())
        b = x1.shape[0]
        eps = torch.randn(x1.shape, device=x1.device, generator=generator)
        x0 = hs[:, -1] + self.sigma0 * eps
        s = torch.rand(b, device=x1.device, generator=generator)
        x_s = (1.0 - s[:, None]) * x0 + s[:, None] * x1
        return F.mse_loss(self.v(x_s, s, hs, adv), x1 - x0)

    def _project(self, x: torch.Tensor, reproject: str | None) -> torch.Tensor:
        mode = self.reproject if reproject is None else reproject
        if mode == "norm":
            return x * (self.norm_mean / x.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return x

    @torch.no_grad()
    def sample(
        self, hist, adv, n_euler=None, generator=None, reproject=None
    ) -> torch.Tensor:
        """One raw sample ``[B, D]`` of ``z_{H+1}``: Euler from ``z_H + sigma0
        eps`` with ``n_euler`` steps (default the model's), then the
        optional re-projection."""
        n = self.n_euler if n_euler is None else int(n_euler)
        hs = self.normalize(hist.float())
        b = hs.shape[0]
        x = hs[:, -1] + self.sigma0 * torch.randn(
            b, self.dim, device=hs.device, generator=generator
        )
        for i in range(n):
            s = torch.full((b,), i / n, device=hs.device, dtype=torch.float32)
            x = x + self.v(x, s, hs, adv) / n
        return self.denormalize(self._project(x, reproject))

    @torch.no_grad()
    def predict_mean(self, hist, adv, n_samples=8, n_euler=None, generator=None):
        """Mean of ``n_samples`` samples, raw ``[B, D]``."""
        return torch.stack(
            [self.sample(hist, adv, n_euler, generator) for _ in range(n_samples)]
        ).mean(0)

    def log_prob(self, z_next, hist, adv, n_steps=None, n_probes=1, generator=None):
        """``log p(z_next | hist, adv) [B]`` in raw latent units
        (:func:`anchored_log_density` on the standardised flow, minus the
        log Jacobian of the standardisation)."""
        hs, x1 = self.normalize(hist.float()), self.normalize(z_next.float())
        n = self.n_euler if n_steps is None else int(n_steps)
        lp = anchored_log_density(
            lambda x, s: self.v(x, s, hs, adv),
            x1,
            hs[:, -1],
            self.sigma0,
            n,
            n_probes,
            generator,
        )
        return lp - self.sd.log().sum()


class MsePredictor(_Conditioned):
    """The deterministic ablation: the same conditioning and trunk regress
    the standardised change ``z_{H+1} - z_H``; ``predict_mean`` (and
    ``sample``) is that output, ``log_prob`` is NaN."""

    kind = "mse"

    def __init__(self, dim, history, hidden=512, depth=6, adv_dim=64):
        super().__init__(dim, history, hidden, depth, history * dim + adv_dim, adv_dim)

    @property
    def hparams(self) -> dict:
        return dict(
            kind=self.kind,
            dim=self.dim,
            history=self.history,
            hidden=self.hidden,
            depth=self.depth,
            adv_dim=self.adv_dim,
        )

    def delta(self, hs, adv) -> torch.Tensor:
        return self.trunk(self.cond(hs, adv))

    def loss(self, hist, target, adv, generator=None) -> torch.Tensor:
        hs, x1 = self.normalize(hist.float()), self.normalize(target.float())
        return F.mse_loss(self.delta(hs, adv), x1 - hs[:, -1])

    @torch.no_grad()
    def predict_mean(self, hist, adv, n_samples=None, n_euler=None, generator=None):
        hs = self.normalize(hist.float())
        return self.denormalize(hs[:, -1] + self.delta(hs, adv))

    sample = predict_mean

    def log_prob(self, z_next, hist, adv, n_steps=None, n_probes=1, generator=None):
        return torch.full((z_next.shape[0],), float("nan"), device=z_next.device)


PREDICTORS = dict(flow=FlowPredictor, mse=MsePredictor)


def build_predictor(hparams: dict) -> nn.Module:
    hp = dict(hparams)
    return PREDICTORS[hp.pop("kind")](**hp)


def save_predictor(path, model, meta: dict, **extra) -> None:
    """``dict(kind, hparams, state_dict, latent_meta, args, step, metrics,
    **extra)`` written atomically; ``meta`` carries ``latent_meta`` (the
    ``latents.pt`` meta: ckpt, stride, window, dim, min_tokens), ``args``,
    ``step`` and ``metrics``."""
    state = dict(
        kind=model.kind, hparams=model.hparams, state_dict=model.state_dict()
    )
    state.update(meta)
    state.update(extra)
    save_atomic(state, path)


def load_predictor(path, device="cpu"):
    """``(model in eval mode, meta)`` with ``meta`` everything in the file
    but the weights (``kind``, ``hparams``, ``latent_meta``, ``args``,
    ``step``, ``metrics``, ...)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_predictor(ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model.to(device).eval(), meta


# ------------------------------------------------------------------- sequences


class LatentStore:
    """The cached latents of one split on the device with what the sampler
    needs: ``z`` float32 ``[N, D]`` (raw), ``valid`` (``n_tokens >=
    min_tokens``), ``end`` (the exclusive last row of every row's object)
    and ``starts`` (the valid rows, the candidate sequence starts)."""

    def __init__(self, cache: dict, split: str, device, min_tokens: int):
        off, cnt = cache["splits"][split]
        self.split, self.n = split, int(cnt)
        self.z = cache["z"][off : off + cnt].float().to(device)
        self.valid = (cache["n_tokens"][off : off + cnt].long() >= min_tokens).to(
            device
        )
        ptr = cache["ptr"][split].long() - off
        self.n_objects = len(ptr) - 1
        self.end = torch.repeat_interleave(ptr[1:], ptr[1:] - ptr[:-1]).to(device)
        self.starts = torch.nonzero(self.valid).flatten()
        if len(self.starts) == 0:
            raise ValueError(f"no valid window in split {split!r}")

    def draw(
        self, n, adv_range, history, stride, generator, extra=0, max_rounds=64
    ):
        """``n`` sequences: ``rows [n, H + 1 + extra]`` (long) and the
        realised advances ``adv [n, H + extra]`` (window units). A start is
        uniform over the valid rows, the ``H`` advances uniform in
        ``adv_range`` rounded to the grid (:func:`round_advance`), the
        ``extra`` steps after the target repeat the last advance (for the
        rollout evaluation). Sequences leaving the object or touching an
        invalid row are rejected: ``4 n`` are drawn per round and the first
        ``n`` accepted kept, more rounds fill up; the acceptance rate below
        1 % raises."""
        lo, hi = adv_range
        dev = self.z.device
        rows_out, adv_out, drawn, kept, need = [], [], 0, 0, int(n)
        for _ in range(max_rounds):
            m = 4 * need
            i = self.starts[
                torch.randint(len(self.starts), (m,), device=dev, generator=generator)
            ]
            a = lo + (hi - lo) * torch.rand(m, history, device=dev, generator=generator)
            k = round_advance(a, stride)
            if extra:
                k = torch.cat([k, k[:, -1:].expand(m, extra)], 1)
            rows = torch.cat([i[:, None], i[:, None] + torch.cumsum(k, 1)], 1)
            ok = rows[:, -1] < self.end[i]
            ok &= self.valid[rows.clamp(max=self.n - 1)].all(1)
            drawn, kept = drawn + m, kept + int(ok.sum())
            sel = torch.nonzero(ok).flatten()[:need]
            rows_out.append(rows[sel])
            adv_out.append(k[sel].float() * stride)
            need -= len(sel)
            if need <= 0:
                break
        rate = kept / max(drawn, 1)
        if need > 0 or rate < 0.01:
            raise RuntimeError(
                f"sequence acceptance {rate:.3%} in split {self.split!r} (history "
                f"{history}, advance {lo}-{hi}, {extra} extra steps): the windows "
                "are too sparse for this history and advance"
            )
        return torch.cat(rows_out), torch.cat(adv_out)


# ------------------------------------------------------------------ evaluation


def fit_gaussian_null(model, store, adv_range, stride, generator, n=50_000):
    """``(m, v)`` of the standardised change ``z_{H+1} - z_H`` over ``n``
    training sequences at ``adv_range``: the null ``N(z_H + m, diag v)``."""
    rows, _ = store.draw(n, adv_range, model.history, stride, generator)
    z = model.normalize(store.z[rows[:, -2:]])
    d = z[:, 1] - z[:, 0]
    return d.mean(0), d.var(0) + 1e-6


@torch.no_grad()
def evaluate(model, store, rows, adv, gauss, args, seed) -> dict:
    """The validation metrics of the module docstring on the fixed sequences
    ``rows [V, H + horizons]``, ``adv [V, H + horizons - 1]``, each next to
    its null; standardised latent units throughout."""
    model.eval()
    dev = model.mu.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    H, hz, D = model.history, args.eval_horizons, model.dim
    flow = model.kind == "flow"
    sums, n_tot = {}, 0

    def add(key, value, n):
        sums[key] = sums.get(key, 0.0) + float(value) * n

    for lo in range(0, rows.shape[0], args.eval_batch):
        r, a = rows[lo : lo + args.eval_batch], adv[lo : lo + args.eval_batch]
        n = r.shape[0]
        z = store.z[r]  # [n, H + hz, D] raw
        hist, target, a_h = z[:, :H], z[:, H], a[:, :H]
        hs, ts = model.normalize(hist), model.normalize(target)
        zH = hs[:, -1]
        add("val_loss", model.loss(hist, target, a_h, gen), n)
        pred = model.normalize(model.predict_mean(hist, a_h, args.eval_samples, generator=gen))
        add("val_mse", (pred - ts).square().mean(), n)
        add("val_mse_persist", (zH - ts).square().mean(), n)
        add("val_mse_histmean", (hs.mean(1) - ts).square().mean(), n)
        d = ts - zH
        add("val_true_var", d.var(), n)
        m, v = gauss
        add("val_nll_persist_gauss", 0.5 * ((d - m).square() / v + (2 * math.pi * v).log()).mean(), n)
        if flow:
            lp = model.log_prob(target, hist, a_h, generator=gen) + model.sd.log().sum()
            add("val_nll", -lp.mean() / D, n)
            samples = torch.stack(
                [model.normalize(model.sample(hist, a_h, generator=gen)) for _ in range(args.eval_samples)]
            )
            add("val_sample_std", samples.std(0).mean(), n)
        chain, a_win = hist, a_h
        for h in range(1, hz + 1):
            nxt = model.sample(chain[:, -H:], a_win, generator=gen)
            chain = torch.cat([chain, nxt[:, None]], 1)
            a_win = torch.cat([a_win[:, 1:], a[:, H + h - 1 : H + h]], 1)
            truth, roll = model.normalize(z[:, H + h - 1]), model.normalize(nxt)
            add(f"val_rollout_mse_h{h}", (roll - truth).square().mean(), n)
            add(f"val_rollout_persist_h{h}", (zH - truth).square().mean(), n)
            add(
                f"val_rollout_norm_drift_h{h}",
                roll.norm(dim=-1).mean() / truth.norm(dim=-1).mean().clamp_min(1e-6),
                n,
            )
        n_tot += n
    m = {k: v / n_tot for k, v in sums.items()}
    m["val_true_std"] = math.sqrt(m.pop("val_true_var"))
    m["val_mse_ratio"] = m["val_mse"] / max(m["val_mse_persist"], 1e-12)
    if not flow:
        m["val_nll"] = m["val_sample_std"] = float("nan")
    m["val_sequences"] = int(n_tot)
    model.train()
    return m


def describe(m: dict, horizons: int) -> str:
    roll = " ".join(
        f"h{h} {m[f'val_rollout_mse_h{h}']:.4f}/{m[f'val_rollout_persist_h{h}']:.4f}"
        for h in range(1, horizons + 1)
    )
    drift = " ".join(
        f"{m[f'val_rollout_norm_drift_h{h}']:.3f}" for h in range(1, horizons + 1)
    )
    return (
        f"loss {m['val_loss']:.4f} | mse {m['val_mse']:.4f} (persist "
        f"{m['val_mse_persist']:.4f}, hist-mean {m['val_mse_histmean']:.4f}, ratio "
        f"{m['val_mse_ratio']:.3f}) | nll/dim {m['val_nll']:.3f} (gauss "
        f"{m['val_nll_persist_gauss']:.3f}) | rollout mse/persist {roll} | norm drift "
        f"{drift} | spread {m['val_sample_std']:.3f} (true {m['val_true_std']:.3f})"
    )


# ------------------------------------------------------------------------ main


def add_args(p) -> None:
    p.add_argument("--latents", required=True, help="latents.pt of cache_latents")
    p.add_argument("--out", default="project/runs/pred")
    p.add_argument("--kind", choices=tuple(PREDICTORS), default="flow")
    p.add_argument("--history", type=int, default=3)
    p.add_argument(
        "--advance",
        type=float,
        nargs=2,
        default=(1.0, 1.5),
        metavar=("LO", "HI"),
        help="advance range in window units, rounded to the latent grid",
    )
    p.add_argument(
        "--advance-start",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="curriculum: the range at step 0, ramped to --advance over "
        "--advance-ramp of the steps (off by default)",
    )
    p.add_argument("--advance-ramp", type=float, default=0.6, help="fraction of steps")
    p.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        help="a window is valid with at least this many points (default: the "
        "cache's min_tokens)",
    )
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--sigma0", type=float, default=0.3, help="anchor noise (flow)")
    p.add_argument("--n-euler", type=int, default=20, help="sampling steps (flow)")
    p.add_argument("--reproject", choices=("none", "norm"), default="none")
    p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=500, help="warm-up steps")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--ckpt-every", type=int, default=5_000)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--time-budget", type=float, default=0.0, help="stop after s")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-sequences", type=int, default=20_000)
    p.add_argument("--eval-samples", type=int, default=8)
    p.add_argument("--eval-horizons", type=int, default=4)
    p.add_argument("--eval-batch", type=int, default=4096)
    add_device_arg(p)
    add_wandb_args(p)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    seed_all(args.seed)
    dev = get_device(args)
    t_start = time.time()

    cache = torch.load(args.latents, map_location="cpu", weights_only=False)
    lmeta = cache["meta"]
    stride, H = float(lmeta["stride"]), args.history
    if args.min_tokens is None:
        args.min_tokens = int(lmeta["min_tokens"])
    train = LatentStore(cache, "train", dev, args.min_tokens)
    val = LatentStore(cache, "validation", dev, args.min_tokens)
    del cache
    dump_json(vars(args), out / "args.json")
    print(
        f"latents {args.latents}: dim {lmeta['dim']}, stride {stride} x {lmeta['window']} d, "
        f"train {train.n} windows ({int(train.valid.sum())} valid, {train.n_objects} objects), "
        f"val {val.n} windows ({int(val.valid.sum())} valid, {val.n_objects} objects); "
        f"loaded in {time.time() - t_start:.0f}s"
    )

    if ckpt is not None:
        model = build_predictor(ckpt["hparams"])
    else:
        hp = dict(
            kind=args.kind,
            dim=int(lmeta["dim"]),
            history=H,
            hidden=args.hidden,
            depth=args.depth,
        )
        if args.kind == "flow":
            hp.update(sigma0=args.sigma0, n_euler=args.n_euler, reproject=args.reproject)
        model = build_predictor(hp)
        model.set_stats(train.z[train.valid])
    model.to(dev)
    print(f"predictor {model.kind}: {n_params(model) / 1e6:.2f}M params, {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = LambdaLR(opt, cosine_schedule(args.steps, args.warmup / max(1, args.steps)))
    step, elapsed, last_metrics = 0, 0.0, None
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        step, elapsed = ckpt["step"], ckpt["elapsed"]
        last_metrics = ckpt.get("metrics")
    latent_meta = dict(
        path=str(args.latents),
        ckpt=lmeta.get("ckpt"),
        encoder_kind=lmeta.get("encoder_kind"),
        stride=stride,
        window=lmeta.get("window"),
        dim=int(lmeta["dim"]),
        min_tokens=args.min_tokens,
    )
    tracker = Tracker(
        args,
        config=dict(
            vars(args),
            params=n_params(model),
            n_train_windows=train.n,
            n_train_objects=train.n_objects,
            **{f"latent_{k}": v for k, v in latent_meta.items()},
        ),
        run_id=ckpt.get("wandb_id") if ckpt else None,
        job_type="predictor",
    )
    log = JsonlLog(out / "log.jsonl")

    # Fixed validation sequences (seeded, so a resumed run scores the same
    # set) and the Gaussian null, both at the final advance range.
    gen_val = torch.Generator(device=dev).manual_seed(args.seed + 1)
    v_rows, v_adv = val.draw(
        args.val_sequences, args.advance, H, stride, gen_val, extra=args.eval_horizons - 1
    )
    gauss = fit_gaussian_null(model, train, args.advance, stride, gen_val)
    gen = torch.Generator(device=dev).manual_seed(args.seed + 2 + step)

    def current_advance(at_step: int) -> tuple[float, float]:
        if not args.advance_start:
            return tuple(args.advance)
        t = min(1.0, at_step / max(1.0, args.advance_ramp * args.steps))
        return tuple(
            float(a + t * (b - a)) for a, b in zip(args.advance_start, args.advance)
        )

    def save(path, metrics, with_opt=True):
        extra = dict(elapsed=elapsed, wandb_id=tracker.id)
        if with_opt:
            extra.update(opt=opt.state_dict(), sched=sched.state_dict())
        save_predictor(
            path,
            model,
            dict(latent_meta=latent_meta, args=dict(vars(args)), step=step, metrics=metrics),
            **extra,
        )

    def run_eval(train_loss):
        t0 = time.time()
        m = evaluate(model, val, v_rows, v_adv, gauss, args, args.seed + 3)
        m.update(step=step, elapsed=elapsed, train_loss=train_loss, eval_seconds=time.time() - t0)
        print(f"  eval @ {step}: {describe(m, args.eval_horizons)} | {time.time() - t0:.0f}s", flush=True)
        log.write(kind="eval", **m)
        tracker.log(
            {
                "train_mean/loss": train_loss,
                **{f"val/{k[4:]}": v for k, v in m.items() if k.startswith("val_")},
            },
            step=step,
        )
        return m

    model.train()
    run, n_acc, t_last, t_run, stop = 0.0, 0, time.time(), time.time(), False
    while step < args.steps and not stop:
        adv_range = current_advance(step)
        rows, adv = train.draw(args.batch_size, adv_range, H, stride, gen)
        z = train.z[rows]
        loss = model.loss(z[:, :H], z[:, H], adv, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        sched.step()
        step += 1
        run, n_acc = run + loss.item(), n_acc + 1
        if step % args.log_every == 0:
            now = time.time()
            rate, t_last = (now - t_last) / args.log_every, now
            loss_m, lr, gpu = run / n_acc, sched.get_last_lr()[0], gpu_stats(dev)
            lo, hi = current_advance(step)
            print(
                f"step {step:7d}  loss {loss_m:.4f}  lr {lr:.2e}  advance {lo:.2f}-{hi:.2f}"
                f"  {rate:.4f} s/step",
                flush=True,
            )
            log.write(
                kind="train", step=step, loss=loss_m, lr=lr, advance_lo=lo, advance_hi=hi,
                s_per_step=rate, **gpu,
            )
            tracker.log(
                {
                    "train/loss": loss_m,
                    "train/lr": lr,
                    "train/advance_lo": lo,
                    "train/advance_hi": hi,
                    "perf/s_per_step": rate,
                    "perf/seq_per_s": args.batch_size / max(rate, 1e-9),
                    **{f"sys/{k}": v for k, v in gpu.items()},
                },
                step=step,
            )
        if step % args.eval_every == 0 or step == args.steps:
            loss_m = run / max(n_acc, 1)
            run, n_acc = 0.0, 0
            elapsed, t_run = elapsed + time.time() - t_run, time.time()
            last_metrics = run_eval(loss_m)
            t_run = time.time()
        budget = bool(args.time_budget) and (time.time() - t_start) > args.time_budget
        if step % args.ckpt_every == 0 or step == args.steps or budget:
            elapsed, t_run = elapsed + time.time() - t_run, time.time()
            save(last, last_metrics)
        if budget and step < args.steps:
            print(f"time budget reached at step {step}; checkpoint saved", flush=True)
            stop = True
    if step >= args.steps:
        save(out / "pred.pt", last_metrics, with_opt=False)
        (out / "DONE").write_text(f"{step} steps, {elapsed / 3600:.2f} h\n")
        print(f"done: {step} steps in {elapsed / 3600:.2f} h; saved {out / 'pred.pt'}")
        if last_metrics:
            tracker.summary(
                **{k: v for k, v in last_metrics.items() if isinstance(v, (int, float))}
            )
    tracker.finish()


if __name__ == "__main__":
    main()
