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

``--arch seq`` (:class:`SeqFlowPredictor`, :class:`SeqMsePredictor`) reads
the whole light curve of a star instead of a fixed history: the valid
windows of an object in window order, any number of them, so a star with a
longer baseline gives more frames. A causal transformer (the AdaLN-zero
:class:`~romae_lc.ARPredictor`, conditioned on the gap of every window from
the one before it) summarises the past into a state per position, and the
same flow head as above, given the state and the next gap, predicts the
next latent anchored at the current one. Training draws
``--batch-size`` objects, keeps every valid window with probability
``--seq-keep`` (so gaps vary and sparse curves are seen) and crops to
``--max-len`` windows at random; validation scores the full sequence of
``--val-objects`` objects, cropped to the last ``--max-len`` windows, and
reports the one-step error per history length (``val_mse_by_history``),
which says whether more history helps. ``--arch mlp`` (the default) is the
fixed-history model above, so old runs stay reproducible.

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

from romae_lc.lewm import ARPredictor

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
    "val_objects",
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


class _Standardised(nn.Module):
    """What every predictor shares: the standardisation buffers ``mu`` and
    ``sd`` (per dimension, from the training latents) and ``norm_mean`` (the
    mean standardised norm, for ``--reproject norm``). Every public method
    of a predictor takes and returns raw latents; the standardisation is
    internal."""

    arch = "mlp"

    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)
        self.reproject = "none"
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))
        self.register_buffer("norm_mean", torch.tensor(float(dim) ** 0.5))

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

    def _project(self, x: torch.Tensor, reproject: str | None) -> torch.Tensor:
        mode = self.reproject if reproject is None else reproject
        if mode == "norm":
            return x * (self.norm_mean / x.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return x


class _Conditioned(_Standardised):
    """What the two fixed-history predictor kinds share: the advance
    embedding and the residual MLP trunk."""

    def __init__(self, dim, history, hidden, depth, in_dim, adv_dim=64):
        super().__init__(dim)
        self.history, self.hidden, self.depth, self.adv_dim = (
            int(history),
            int(hidden),
            int(depth),
            int(adv_dim),
        )
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
            arch=self.arch,
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
            arch=self.arch,
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


# ------------------------------------------------------------ sequence models


class ResidualMLP(nn.Module):
    """The residual MLP trunk of :class:`_Conditioned` as a module of its
    own (Linear, SiLU, ``depth`` blocks of LayerNorm, Linear, SiLU, Linear
    with skips, a zero-initialised output), the head of the sequence
    predictors."""

    def __init__(self, in_dim, hidden, depth, out_dim):
        super().__init__()
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
        self.out = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.inp(h))
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


def crop_last(z, gaps, mask, max_len):
    """The last ``max_len`` real positions of every left-aligned sequence
    (``z [B, T, D]``, ``gaps [B, T]``, ``mask [B, T]``); unchanged when ``T
    <= max_len``. The first gap of a cropped sequence is set to 0, as in a
    freshly drawn one."""
    t = z.shape[1]
    if t <= max_len:
        return z, gaps, mask
    n_real = mask.sum(1)
    start = (n_real - max_len).clamp(min=0)
    idx = start[:, None] + torch.arange(max_len, device=z.device)
    z = z.gather(1, idx[..., None].expand(-1, -1, z.shape[2]))
    gaps, mask = gaps.gather(1, idx), mask.gather(1, idx)
    gaps = gaps.clone()
    gaps[:, 0] = 0.0
    return z, gaps, mask


def seq_positions(mask: torch.Tensor, min_hist: int = 3):
    """``(b, t)`` of the positions with at least ``min_hist`` real windows
    before them and a real ``t + 1``. Sequences are left-aligned (the real
    windows first, then padding), so the windows before ``t`` number ``t``."""
    pair = mask[:, :-1] & mask[:, 1:]
    pair[:, :min_hist] = False
    return torch.nonzero(pair, as_tuple=True)


class _SeqBase(_Standardised):
    """What the two sequence predictors share: ``inp`` (latent to the core
    width), ``gap_mlp`` (the embedding of ``log(gap + 1e-3)``, gaps in
    window units), ``core`` (the causal AdaLN-zero transformer of
    :class:`~romae_lc.ARPredictor`, conditioned on the gap of each window
    from the one before) and ``head`` (the residual MLP). Sequences are
    left-aligned: real windows first, padding after them, ``mask`` True on
    the real ones. Causal attention keeps the padding away from the real
    positions; the pad inputs are zeroed anyway."""

    arch = "seq"

    def __init__(
        self,
        dim,
        hidden=256,
        depth=4,
        heads=4,
        dim_head=64,
        mlp_dim=1024,
        dropout=0.1,
        max_len=64,
        gap_dim=64,
        head_hidden=512,
        head_depth=6,
        head_in=0,
    ):
        super().__init__(dim)
        self.hidden, self.depth, self.heads, self.dim_head = (
            int(hidden),
            int(depth),
            int(heads),
            int(dim_head),
        )
        self.mlp_dim, self.dropout, self.max_len, self.gap_dim = (
            int(mlp_dim),
            float(dropout),
            int(max_len),
            int(gap_dim),
        )
        self.head_hidden, self.head_depth = int(head_hidden), int(head_depth)
        self.inp = nn.Linear(dim, hidden)
        self.gap_mlp = nn.Sequential(
            nn.Linear(1, gap_dim), nn.SiLU(), nn.Linear(gap_dim, hidden)
        )
        self.core = ARPredictor(
            hidden,
            n_frames=max_len,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            cond_dim=hidden,
        )
        self.head = ResidualMLP(head_in, head_hidden, head_depth, dim)

    def _hparams(self) -> dict:
        return dict(
            kind=self.kind,
            arch=self.arch,
            dim=self.dim,
            hidden=self.hidden,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            max_len=self.max_len,
            gap_dim=self.gap_dim,
            head_hidden=self.head_hidden,
            head_depth=self.head_depth,
        )

    def gap_emb(self, gap: torch.Tensor) -> torch.Tensor:
        """``[..., hidden]`` from gaps ``[...]`` in window units."""
        return self.gap_mlp(gap.float().clamp_min(0).add(1e-3).log()[..., None])

    def states(self, z, gaps, mask) -> torch.Tensor:
        """The history states ``h [B, T, hidden]`` of raw latents ``z [B, T,
        D]`` with ``gaps [B, T]`` (each window's gap from the previous one,
        window units; 0 at the first) and ``mask [B, T]`` (True on real
        windows). ``h[:, t]`` sees windows ``0..t`` only; ``T <= max_len``."""
        keep = mask[..., None].to(z.dtype if z.is_floating_point() else torch.float32)
        zs = self.normalize(z.float())
        return self.core(self.inp(zs) * keep, self.gap_emb(gaps) * keep)

    @staticmethod
    def pairs(mask: torch.Tensor):
        """``(b, t)`` of the positions where ``t`` and ``t + 1`` are real."""
        return torch.nonzero(mask[:, :-1] & mask[:, 1:], as_tuple=True)

    @torch.no_grad()
    def rollout(
        self,
        z_prefix,
        gaps_prefix,
        mask_prefix,
        gap_steps,
        generator=None,
        n_euler=None,
        reproject=None,
    ) -> torch.Tensor:
        """Raw samples ``[B, S, D]`` of the ``S = gap_steps.shape[1]`` windows
        after the prefix, one per step: the core is re-run on the prefix plus
        the samples so far (cropped to the last ``max_len``), the next latent
        is sampled from the last real state with ``gap_steps[:, s]`` as the
        gap, and appended."""
        z, g, m = z_prefix.float().clone(), gaps_prefix.float().clone(), mask_prefix.clone()
        b = torch.arange(z.shape[0], device=z.device)
        out = []
        for s in range(gap_steps.shape[1]):
            z, g, m = crop_last(z, g, m, self.max_len)
            n_real = m.sum(1)
            h = self.states(z, g, m)
            g_next = gap_steps[:, s].float()
            z_new = self.sample_next(
                h[b, n_real - 1], g_next, z[b, n_real - 1], n_euler=n_euler,
                generator=generator, reproject=reproject,
            )
            out.append(z_new)
            # append at the first pad position of every row
            z = torch.cat([z, z.new_zeros(z.shape[0], 1, z.shape[2])], 1)
            g = torch.cat([g, g.new_zeros(g.shape[0], 1)], 1)
            m = torch.cat([m, m.new_zeros(m.shape[0], 1)], 1)
            z[b, n_real], g[b, n_real], m[b, n_real] = z_new, g_next, True
        return torch.stack(out, 1)


class SeqFlowPredictor(_SeqBase):
    """The flow predictor over a whole sequence (``--arch seq --kind
    flow``). The head is the flow of :class:`FlowPredictor` with the
    condition ``concat(h_t, gap_emb(gap_{t + 1}), time_embedding(s))``: the
    flow runs from ``x_0 = z_t + sigma0 eps`` to ``x_1 = z_{t + 1}``, so
    ``sample_next``, ``predict_mean`` and ``log_prob`` take the raw current
    latent ``z_t`` as the anchor next to the state ``h_t`` and the next gap."""

    kind = "flow"

    def __init__(
        self,
        dim,
        hidden=256,
        depth=4,
        heads=4,
        dim_head=64,
        mlp_dim=1024,
        dropout=0.1,
        max_len=64,
        sigma0=0.3,
        n_euler=20,
        reproject="none",
        gap_dim=64,
        head_hidden=512,
        head_depth=6,
        s_dim=64,
    ):
        if reproject not in ("none", "norm"):
            raise ValueError(f"reproject must be none|norm, got {reproject!r}")
        super().__init__(
            dim, hidden, depth, heads, dim_head, mlp_dim, dropout, max_len, gap_dim,
            head_hidden, head_depth, head_in=int(dim) + 2 * int(hidden) + int(s_dim),
        )
        self.sigma0, self.s_dim, self.n_euler, self.reproject = (
            float(sigma0),
            int(s_dim),
            int(n_euler),
            reproject,
        )

    @property
    def hparams(self) -> dict:
        return dict(
            self._hparams(),
            sigma0=self.sigma0,
            n_euler=self.n_euler,
            reproject=self.reproject,
            s_dim=self.s_dim,
        )

    def v(self, x, s, h, g) -> torch.Tensor:
        """Velocity at standardised ``x [n, D]``, flow time ``s [n]``, state
        ``h [n, hidden]`` and next-gap embedding ``g [n, hidden]``."""
        return self.head(torch.cat([x, h, g, time_embedding(s, self.s_dim)], -1))

    def loss(self, z, gaps, mask, generator=None) -> torch.Tensor:
        """Flow-matching MSE over the positions ``t`` where ``t`` and ``t +
        1`` are real; raw ``z [B, T, D]``, ``gaps [B, T]``, ``mask [B, T]``."""
        zs = self.normalize(z.float())
        h = self.states(z, gaps, mask)
        b, t = self.pairs(mask)
        if len(b) == 0:
            raise ValueError("no position with a real next window")
        h_t, g = h[b, t], self.gap_emb(gaps[b, t + 1])
        x1 = zs[b, t + 1]
        eps = torch.randn(x1.shape, device=x1.device, generator=generator)
        x0 = zs[b, t] + self.sigma0 * eps
        s = torch.rand(len(b), device=x1.device, generator=generator)
        x_s = (1.0 - s[:, None]) * x0 + s[:, None] * x1
        return F.mse_loss(self.v(x_s, s, h_t, g), x1 - x0)

    @torch.no_grad()
    def sample_next(
        self, h_t, gap_next, z_t, n_euler=None, generator=None, reproject=None
    ) -> torch.Tensor:
        """One raw sample ``[n, D]`` of the next latent from the state ``h_t
        [n, hidden]``, the next gap ``[n]`` (window units) and the raw
        anchor ``z_t [n, D]``: Euler from ``z_t + sigma0 eps``."""
        n = self.n_euler if n_euler is None else int(n_euler)
        zt, g = self.normalize(z_t.float()), self.gap_emb(gap_next)
        x = zt + self.sigma0 * torch.randn(zt.shape, device=zt.device, generator=generator)
        for i in range(n):
            s = torch.full((zt.shape[0],), i / n, device=zt.device, dtype=torch.float32)
            x = x + self.v(x, s, h_t, g) / n
        return self.denormalize(self._project(x, reproject))

    @torch.no_grad()
    def predict_mean(self, h_t, gap_next, z_t, n_samples=8, n_euler=None, generator=None):
        """Mean of ``n_samples`` samples, raw ``[n, D]``."""
        return torch.stack(
            [self.sample_next(h_t, gap_next, z_t, n_euler, generator) for _ in range(n_samples)]
        ).mean(0)

    def log_prob(self, z_next, h_t, gap_next, z_t, n_steps=None, n_probes=1, generator=None):
        """``log p(z_next | h_t, gap_next) [n]`` in raw latent units, as
        :meth:`FlowPredictor.log_prob` (:func:`anchored_log_density` on the
        standardised flow anchored at ``z_t``, minus the log Jacobian of the
        standardisation)."""
        zt, x1 = self.normalize(z_t.float()), self.normalize(z_next.float())
        g = self.gap_emb(gap_next)
        n = self.n_euler if n_steps is None else int(n_steps)
        lp = anchored_log_density(
            lambda x, s: self.v(x, s, h_t, g), x1, zt, self.sigma0, n, n_probes, generator
        )
        return lp - self.sd.log().sum()


class SeqMsePredictor(_SeqBase):
    """The deterministic sequence ablation (``--arch seq --kind mse``): the
    head regresses the standardised change ``z_{t + 1} - z_t`` from the
    state and the next gap; ``log_prob`` is NaN."""

    kind = "mse"

    def __init__(
        self,
        dim,
        hidden=256,
        depth=4,
        heads=4,
        dim_head=64,
        mlp_dim=1024,
        dropout=0.1,
        max_len=64,
        gap_dim=64,
        head_hidden=512,
        head_depth=6,
    ):
        super().__init__(
            dim, hidden, depth, heads, dim_head, mlp_dim, dropout, max_len, gap_dim,
            head_hidden, head_depth, head_in=2 * int(hidden),
        )

    @property
    def hparams(self) -> dict:
        return self._hparams()

    def delta(self, h_t, g) -> torch.Tensor:
        return self.head(torch.cat([h_t, g], -1))

    def loss(self, z, gaps, mask, generator=None) -> torch.Tensor:
        zs = self.normalize(z.float())
        h = self.states(z, gaps, mask)
        b, t = self.pairs(mask)
        if len(b) == 0:
            raise ValueError("no position with a real next window")
        d = self.delta(h[b, t], self.gap_emb(gaps[b, t + 1]))
        return F.mse_loss(d, zs[b, t + 1] - zs[b, t])

    @torch.no_grad()
    def predict_mean(self, h_t, gap_next, z_t, n_samples=None, n_euler=None, generator=None):
        zt = self.normalize(z_t.float())
        return self.denormalize(zt + self.delta(h_t, self.gap_emb(gap_next)))

    @torch.no_grad()
    def sample_next(self, h_t, gap_next, z_t, n_euler=None, generator=None, reproject=None):
        return self.predict_mean(h_t, gap_next, z_t)

    def log_prob(self, z_next, h_t, gap_next, z_t, n_steps=None, n_probes=1, generator=None):
        return torch.full((z_next.shape[0],), float("nan"), device=z_next.device)


PREDICTORS = dict(flow=FlowPredictor, mse=MsePredictor)
SEQ_PREDICTORS = dict(flow=SeqFlowPredictor, mse=SeqMsePredictor)
ARCHS = dict(mlp=PREDICTORS, seq=SEQ_PREDICTORS)


def build_predictor(hparams: dict) -> nn.Module:
    """The model of ``hparams`` (``kind``, ``arch`` and the constructor
    arguments). A file without ``arch`` is an old fixed-history one."""
    hp = dict(hparams)
    arch = hp.pop("arch", "mlp")
    return ARCHS[arch][hp.pop("kind")](**hp)


def save_predictor(path, model, meta: dict, **extra) -> None:
    """``dict(kind, arch, hparams, state_dict, latent_meta, args, step,
    metrics, **extra)`` written atomically; ``meta`` carries ``latent_meta``
    (the ``latents.pt`` meta: ckpt, stride, window, dim, min_tokens),
    ``args``, ``step`` and ``metrics``."""
    state = dict(
        kind=model.kind, arch=model.arch, hparams=model.hparams, state_dict=model.state_dict()
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
        # noise realisations of every window (cache_latents --realisations):
        # z_all [K, N, D], realisation 0 the window itself
        alt = cache.get("z_alt")
        self.z_all = (
            None
            if alt is None
            else torch.cat([self.z[None], alt[:, off : off + cnt].float().to(device)])
        )
        self.n_real = 1 if self.z_all is None else int(self.z_all.shape[0])
        self.valid = (cache["n_tokens"][off : off + cnt].long() >= min_tokens).to(
            device
        )
        ptr = cache["ptr"][split].long() - off
        self.n_objects = len(ptr) - 1
        self.end = torch.repeat_interleave(ptr[1:], ptr[1:] - ptr[:-1]).to(device)
        self.starts = torch.nonzero(self.valid).flatten()
        if len(self.starts) == 0:
            raise ValueError(f"no valid window in split {split!r}")
        # For the sequence sampler: the window index of every row, the grid
        # stride, and a padded table ``seq_rows [n_seq_objects, L_max]`` of
        # the valid rows (window order, -1 after the last) of every object
        # with at least 3 valid windows (``seq_objects``).
        self.stride = float(cache["meta"]["stride"])
        self.win = cache["win"][off : off + cnt].long().to(device)
        valid_cpu = cache["n_tokens"][off : off + cnt].long() >= min_tokens
        counts = ptr[1:] - ptr[:-1]
        obj_of_row = torch.repeat_interleave(torch.arange(self.n_objects), counts)
        n_valid = torch.zeros(self.n_objects, dtype=torch.long).index_add_(
            0, obj_of_row, valid_cpu.long()
        )
        seq_obj = torch.nonzero(n_valid >= 3).flatten()
        self.n_seq_objects = len(seq_obj)
        if self.n_seq_objects:
            vr = torch.nonzero(valid_cpu).flatten()  # valid rows, ascending
            ov = obj_of_row[vr]
            rank = torch.arange(len(vr)) - (torch.cumsum(n_valid, 0) - n_valid)[ov]
            si = torch.full((self.n_objects,), -1, dtype=torch.long)
            si[seq_obj] = torch.arange(self.n_seq_objects)
            ok = si[ov] >= 0
            table = torch.full((self.n_seq_objects, int(n_valid[seq_obj].max())), -1, dtype=torch.long)
            table[si[ov][ok], rank[ok]] = vr[ok]
            self.seq_rows = table.to(device)
        else:
            self.seq_rows = torch.zeros((0, 0), dtype=torch.long, device=device)
        self.seq_objects = seq_obj.to(device)

    def draw_sequences(self, n_objects, keep=0.5, max_len=64, generator=None, train=True):
        """``n_objects`` whole light curves as sequences of window latents:
        ``dict(rows [B, L] long, -1 where padded; gaps [B, L], the gap of a
        kept window from the previous kept one in window units, 0 at the
        first; mask [B, L] bool, True on real windows; obj [B], the object
        index in the split)``. A sequence is an object's valid windows in
        window order, left-aligned. Objects with fewer than 3 valid windows
        are skipped.

        Training (``train=True``): objects uniform with replacement; every
        valid window kept independently with probability ``keep`` (so the
        gaps vary and sparse curves are seen), all of them when fewer than 3
        would remain; a random contiguous crop of ``max_len`` windows when
        longer. Validation (``train=False``): ``keep`` is ignored, the
        objects are a random subset without replacement (all of them when
        ``n_objects`` is larger), the full valid sequence cropped to the
        LAST ``max_len`` windows. ``L`` is the longest sequence of the batch,
        at most ``max_len``."""
        dev = self.z.device
        if self.n_seq_objects == 0:
            raise ValueError(f"no object with 3 valid windows in split {self.split!r}")
        if train:
            o = torch.randint(self.n_seq_objects, (int(n_objects),), device=dev, generator=generator)
        else:
            o = torch.randperm(self.n_seq_objects, device=dev, generator=generator)[: int(n_objects)]
        rows = self.seq_rows[o]
        b, lmax = rows.shape
        real = rows >= 0
        # keep is one probability or a (lo, hi) range: with a range every
        # sequence draws its own probability, so short AND full-length
        # sequences are seen. A fixed 0.5 never showed the model more than
        # about half a light curve, and it failed on positions past 20.
        vals = list(keep) if isinstance(keep, (tuple, list)) else [keep]
        k_lo, k_hi = float(vals[0]), float(vals[-1])
        if train and k_lo < 1.0:
            p_keep = k_lo + (k_hi - k_lo) * torch.rand(b, 1, device=dev, generator=generator)
            kept = (torch.rand(b, lmax, device=dev, generator=generator) < p_keep) & real
            few = kept.sum(1) < 3
            kept = torch.where(few[:, None], real, kept)
        else:
            kept = real
        # kept windows first, in window order; the rest after them
        order = ((~kept).long() * lmax + torch.arange(lmax, device=dev)).argsort(1)
        rows, kept = rows.gather(1, order), kept.gather(1, order)
        n_kept = kept.sum(1)
        L = min(int(max_len), int(n_kept.max()))
        over = (n_kept - L).clamp(min=0)  # how many windows the crop drops
        if train:
            start = torch.minimum(
                (torch.rand(b, device=dev, generator=generator) * (over + 1)).long(), over
            )
        else:
            start = over
        idx = start[:, None] + torch.arange(L, device=dev)
        rows, mask = rows.gather(1, idx), kept.gather(1, idx)
        rows = torch.where(mask, rows, torch.full_like(rows, -1))
        win = self.win[rows.clamp(min=0)]
        gaps = torch.zeros(b, L, device=dev)
        gaps[:, 1:] = (win[:, 1:] - win[:, :-1]).float() * self.stride
        gaps = torch.where(mask, gaps, torch.zeros_like(gaps))
        return dict(rows=rows, gaps=gaps, mask=mask, obj=self.seq_objects[o])

    def gather(self, rows: torch.Tensor, generator=None) -> torch.Tensor:
        """The latents of ``rows`` with a random realisation per entry (the
        window itself when the cache has none): the training view, so the
        predictor sees the spread that measurement noise alone produces."""
        if self.z_all is None:
            return self.z[rows]
        r = torch.randint(self.n_real, rows.shape, device=rows.device, generator=generator)
        return self.z_all[r, rows]

    def replicate_std(self, rows: torch.Tensor) -> torch.Tensor:
        """Per-dimension std over the realisations of ``rows`` (``[..., D]``;
        NaN without realisations): the replicate floor in latent units."""
        if self.z_all is None or self.n_real < 2:
            return torch.full(rows.shape + (self.z.shape[1],), float("nan"), device=self.z.device)
        return self.z_all[:, rows].std(0)

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


def _design(hs: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    """Ridge features: the flattened standardised history, the log advances
    and a constant, ``[n, H D + H + 1]``."""
    return torch.cat([hs.flatten(1), adv.float().log(), hs.new_ones(hs.shape[0], 1)], 1)


def _full_gaussian(resid: torch.Tensor, jitter: float = 1e-4):
    """``(mean, cholesky of the covariance)`` of residual rows ``[n, D]``."""
    mu = resid.mean(0)
    cov = torch.cov((resid - mu).T) + jitter * torch.eye(resid.shape[1], device=resid.device)
    return mu, torch.linalg.cholesky(cov)


def full_gaussian_nll(resid: torch.Tensor, mu: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    """Negative log density per dimension, mean over rows, under
    ``N(mu, chol chol^T)``."""
    d = resid.shape[1]
    y = torch.cholesky_solve((resid - mu).T, chol).T
    maha = ((resid - mu) * y).sum(-1)
    logdet = 2 * chol.diagonal().log().sum()
    return (0.5 * (maha + logdet + d * math.log(2 * math.pi))).mean() / d


def fit_nulls(model, store, adv_range, stride, generator, n=50_000, ridge=1e-2):
    """The nulls every predictor number is read against, fitted on ``n``
    training sequences at ``adv_range`` in standardised units. The change
    ``d = z_{H+1} - z_H``:

    - ``diag``: ``(m, v)`` of ``d``, the null ``N(z_H + m, diag v)``;
    - ``full``: ``(mu, chol)`` of ``d``, the same with the full covariance
      (a change that lives in a low-dimensional subspace makes the diagonal
      null far too weak: the latent of the 2026-09-26 run gave -0.85 nats per
      dimension under the full null against +0.84 under the diagonal one);
    - ``ridge``: ``(w, mu, chol)`` of a ridge regression of ``d`` on the
      flattened history, the log advances and a constant (``ridge`` times
      ``n`` on the diagonal), with the full covariance of its residual: the
      strongest cheap predictor, the null the flow's MSE and NLL must beat.
    """
    rows, adv = store.draw(n, adv_range, model.history, stride, generator)
    z = model.normalize(store.gather(rows, generator))
    hs, d = z[:, : model.history], z[:, model.history] - z[:, model.history - 1]
    return _nulls_from(_design(hs, adv[:, : model.history]), d, ridge)


def _nulls_from(x: torch.Tensor, d: torch.Tensor, ridge: float) -> dict:
    """The three nulls of :func:`fit_nulls` from the ridge design ``x [n,
    F]`` and the standardised changes ``d [n, D]``."""
    a = x.T @ x + ridge * len(x) * torch.eye(x.shape[1], device=x.device)
    w = torch.linalg.solve(a, x.T @ d)
    return dict(
        diag=(d.mean(0), d.var(0) + 1e-6),
        full=_full_gaussian(d),
        ridge=(w, *_full_gaussian(d - x @ w)),
    )


def _seq_history(zs, gaps, b, t):
    """The ridge history of positions ``(b, t)`` of standardised sequences
    ``zs [B, T, D]``: the previous 3 latents ``[n, 3, D]`` and the 3 gaps up
    to and including the next one ``[n, 3]``, laid out like the fixed
    history of :func:`_design` (``t >= 2`` and ``t + 1`` real)."""
    hs = torch.stack([zs[b, t - 2], zs[b, t - 1], zs[b, t]], 1)
    adv = torch.stack([gaps[b, t - 1], gaps[b, t], gaps[b, t + 1]], 1)
    return hs, adv


def fit_nulls_seq(model, store, max_len, generator, n=50_000, ridge=1e-2, chunk=1024):
    """:func:`fit_nulls` for ``--arch seq``: the same three nulls on ``n``
    training positions with 3 real windows before them and a real next
    one, from sequences drawn with every valid window kept
    (:meth:`LatentStore.draw_sequences` with ``keep=1``) and a random
    realisation per row; the ridge design is the previous 3 latents, their
    log gaps and a constant."""
    xs, ds, got = [], [], 0
    while got < n:
        seq = store.draw_sequences(chunk, 1.0, max_len, generator, train=True)
        zs = model.normalize(store.gather(seq["rows"].clamp(min=0), generator))
        b, t = seq_positions(seq["mask"], 3)
        if len(b) == 0:
            raise ValueError(
                f"no training object of split {store.split!r} has 5 valid windows"
            )
        hs, adv = _seq_history(zs, seq["gaps"], b, t)
        xs.append(_design(hs, adv))
        ds.append(zs[b, t + 1] - zs[b, t])
        got += len(b)
    return _nulls_from(torch.cat(xs)[:n], torch.cat(ds)[:n], ridge)


@torch.no_grad()
def evaluate(model, store, rows, adv, nulls, args, seed) -> dict:
    """The validation metrics of the module docstring on the fixed sequences
    ``rows [V, H + horizons]``, ``adv [V, H + horizons - 1]``, each next to
    its null (:func:`fit_nulls`); standardised latent units throughout. The
    flow's density uses ``args.nll_steps`` reverse Euler steps: the
    linearised log-determinant is an upper bound on the log density that
    tightens with the step count (20 steps overstated it by 0.13 nats per
    dimension on the 2026-09-26 run, 100 steps by 0.04)."""
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
        # the replicate floor of the target window: the spread measurement
        # noise alone gives its latent (NaN without cached realisations)
        add("val_replicate_std", (store.replicate_std(r[:, H]) / model.sd).mean(), n)
        m, v = nulls["diag"]
        add("val_nll_persist_gauss", 0.5 * ((d - m).square() / v + (2 * math.pi * v).log()).mean(), n)
        add("val_nll_persist_full", full_gaussian_nll(d, *nulls["full"]), n)
        w, r_mu, r_chol = nulls["ridge"]
        resid = d - _design(hs, a_h) @ w
        add("val_mse_ridge", resid.square().mean(), n)
        add("val_nll_ridge_full", full_gaussian_nll(resid, r_mu, r_chol), n)
        if flow:
            lp = model.log_prob(
                target, hist, a_h, n_steps=args.nll_steps, generator=gen
            ) + model.sd.log().sum()
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
    m["val_mse_ratio_ridge"] = m["val_mse"] / max(m["val_mse_ridge"], 1e-12)
    if not flow:
        m["val_nll"] = m["val_sample_std"] = float("nan")
    m["val_sequences"] = int(n_tot)
    model.train()
    return m


#: Bins of ``val_mse_by_history``: real windows before the scored position.
HISTORY_BINS = (("3-5", 3, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21+", 21, 10**9))


@torch.no_grad()
def evaluate_seq(model, store, seq, nulls, args, seed) -> dict:
    """The metrics of :func:`evaluate` for ``--arch seq`` on the fixed
    validation sequences ``seq`` (:meth:`LatentStore.draw_sequences` with
    ``train=False``; latents from realisation 0). Every position ``t`` with
    at least 3 real windows before it (so every null exists) and a real ``t
    + 1`` is scored, ``val_sequences`` counts them; the nulls use the
    previous 3 latents and their gaps. Rollouts start from one random real
    position per object with ``args.eval_horizons`` real windows after it
    (objects too short are left out) and follow the true gaps.
    ``val_mse_by_history`` holds ``mse``, ``persist`` and ``n`` per bin of
    :data:`HISTORY_BINS`: whether more history helps."""
    model.eval()
    dev = model.mu.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    rows, gaps, mask = seq["rows"], seq["gaps"], seq["mask"]
    hz, D, flow = args.eval_horizons, model.dim, model.kind == "flow"
    z = store.z[rows.clamp(min=0)]  # raw [B, T, D]
    zs = model.normalize(z)
    sums, cnts = {}, {}

    def add(key, value, n):
        sums[key] = sums.get(key, 0.0) + float(value) * n
        cnts[key] = cnts.get(key, 0) + n

    # 1. the states of every object, the loss, and the scored positions
    n_obj = max(1, args.eval_batch // 16)
    bs, ts, hs = [], [], []
    for lo in range(0, rows.shape[0], n_obj):
        sl = slice(lo, lo + n_obj)
        h = model.states(z[sl], gaps[sl], mask[sl])
        add("val_loss", model.loss(z[sl], gaps[sl], mask[sl], gen), int(model.pairs(mask[sl])[0].numel()))
        b, t = seq_positions(mask[sl], 3)
        bs.append(b + lo)
        ts.append(t)
        hs.append(h[b, t])
    b, t, h = torch.cat(bs), torch.cat(ts), torch.cat(hs)
    n_tot = len(b)
    if n_tot == 0:
        raise ValueError("no validation position with 3 real windows before it and a real next one")
    z_t, z_next, g_next = z[b, t], z[b, t + 1], gaps[b, t + 1]
    zt_s, ts_s = zs[b, t], zs[b, t + 1]
    hist, adv = _seq_history(zs, gaps, b, t)
    err_model, err_persist = torch.empty(n_tot, device=dev), torch.empty(n_tot, device=dev)
    err_ridge = torch.empty(n_tot, device=dev)

    # 2. one-step numbers per position, in chunks
    for lo in range(0, n_tot, args.eval_batch):
        s = slice(lo, lo + args.eval_batch)
        n = min(args.eval_batch, n_tot - lo)
        pred = model.normalize(model.predict_mean(h[s], g_next[s], z_t[s], args.eval_samples, generator=gen))
        e = (pred - ts_s[s]).square().mean(-1)
        ep = (zt_s[s] - ts_s[s]).square().mean(-1)
        err_model[s], err_persist[s] = e, ep
        add("val_mse", e.mean(), n)
        add("val_mse_persist", ep.mean(), n)
        add("val_mse_histmean", (hist[s].mean(1) - ts_s[s]).square().mean(), n)
        d = ts_s[s] - zt_s[s]
        add("val_true_var", d.var(), n)
        add("val_replicate_std", (store.replicate_std(rows[b[s], t[s] + 1]) / model.sd).mean(), n)
        m, v = nulls["diag"]
        add("val_nll_persist_gauss", 0.5 * ((d - m).square() / v + (2 * math.pi * v).log()).mean(), n)
        add("val_nll_persist_full", full_gaussian_nll(d, *nulls["full"]), n)
        w, r_mu, r_chol = nulls["ridge"]
        resid = d - _design(hist[s], adv[s]) @ w
        err_ridge[s] = resid.square().mean(-1)
        add("val_mse_ridge", resid.square().mean(), n)
        add("val_nll_ridge_full", full_gaussian_nll(resid, r_mu, r_chol), n)
        if flow:
            lp = model.log_prob(
                z_next[s], h[s], g_next[s], z_t[s], n_steps=args.nll_steps, generator=gen
            ) + model.sd.log().sum()
            add("val_nll", -lp.mean() / D, n)
            samples = torch.stack(
                [
                    model.normalize(model.sample_next(h[s], g_next[s], z_t[s], generator=gen))
                    for _ in range(args.eval_samples)
                ]
            )
            add("val_sample_std", samples.std(0).mean(), n)

    # 3. rollouts: one chain per object from a random real position with
    # hz real windows after it, following the true gaps
    n_real = mask.sum(1)
    pos = torch.arange(mask.shape[1], device=dev)
    for lo in range(0, rows.shape[0], n_obj):
        sl = slice(lo, lo + n_obj)
        ok = torch.nonzero(n_real[sl] >= hz + 2).flatten()
        if len(ok) == 0:
            continue
        nr = n_real[sl][ok]
        t0 = (torch.rand(len(ok), device=dev, generator=gen) * (nr - hz).float()).long()
        t0 = torch.minimum(t0, nr - hz - 1)
        zc, gc, mc = z[sl][ok], gaps[sl][ok], mask[sl][ok] & (pos[None] <= t0[:, None])
        steps = t0[:, None] + 1 + torch.arange(hz, device=dev)
        chain = model.rollout(zc, gc, mc, gc.gather(1, steps), generator=gen)
        o = torch.arange(len(ok), device=dev)
        zH = model.normalize(zc[o, t0])
        for k in range(1, hz + 1):
            truth = model.normalize(zc[o, t0 + k])
            roll = model.normalize(chain[:, k - 1])
            add(f"val_rollout_mse_h{k}", (roll - truth).square().mean(), len(ok))
            add(f"val_rollout_persist_h{k}", (zH - truth).square().mean(), len(ok))
            add(
                f"val_rollout_norm_drift_h{k}",
                roll.norm(dim=-1).mean() / truth.norm(dim=-1).mean().clamp_min(1e-6),
                len(ok),
            )
    m = {k: v / cnts[k] for k, v in sums.items()}
    for k in range(1, hz + 1):
        for key in (f"val_rollout_mse_h{k}", f"val_rollout_persist_h{k}", f"val_rollout_norm_drift_h{k}"):
            m.setdefault(key, float("nan"))
    m["val_true_std"] = math.sqrt(m.pop("val_true_var"))
    m["val_mse_ratio"] = m["val_mse"] / max(m["val_mse_persist"], 1e-12)
    m["val_mse_ratio_ridge"] = m["val_mse"] / max(m["val_mse_ridge"], 1e-12)
    if not flow:
        m["val_nll"] = m["val_sample_std"] = float("nan")
    by_hist = {}
    for name, lo_h, hi_h in HISTORY_BINS:
        sel = (t >= lo_h) & (t <= hi_h)
        n = int(sel.sum())
        by_hist[name] = dict(
            mse=float(err_model[sel].mean()) if n else float("nan"),
            persist=float(err_persist[sel].mean()) if n else float("nan"),
            ridge=float(err_ridge[sel].mean()) if n else float("nan"),
            n=n,
        )
    m["val_mse_by_history"] = by_hist
    m["val_sequences"] = int(n_tot)
    model.train()
    return m


def describe(m: dict, horizons: int) -> str:
    """One line of the eval numbers; the by-history bins of ``--arch seq``
    are appended as ``bin mse/ridge/persist (n)``."""
    by_hist = ""
    if "val_mse_by_history" in m:
        by_hist = " | by history (mse/ridge/persist) " + " ".join(
            f"{k} {v['mse']:.4f}/{v.get('ridge', float('nan')):.4f}/{v['persist']:.4f} ({v['n']})"
            for k, v in m["val_mse_by_history"].items()
        )
    roll = " ".join(
        f"h{h} {m[f'val_rollout_mse_h{h}']:.4f}/{m[f'val_rollout_persist_h{h}']:.4f}"
        for h in range(1, horizons + 1)
    )
    drift = " ".join(
        f"{m[f'val_rollout_norm_drift_h{h}']:.3f}" for h in range(1, horizons + 1)
    )
    return (
        f"loss {m['val_loss']:.4f} | mse {m['val_mse']:.4f} (persist "
        f"{m['val_mse_persist']:.4f}, hist-mean {m['val_mse_histmean']:.4f}, ridge "
        f"{m['val_mse_ridge']:.4f}; ratio {m['val_mse_ratio']:.3f} to persist, "
        f"{m['val_mse_ratio_ridge']:.3f} to ridge) | nll/dim {m['val_nll']:.3f} (gauss diag "
        f"{m['val_nll_persist_gauss']:.3f}, full {m['val_nll_persist_full']:.3f}, ridge "
        f"{m['val_nll_ridge_full']:.3f}) | rollout mse/persist {roll} | norm drift "
        f"{drift} | spread {m['val_sample_std']:.3f} (true {m['val_true_std']:.3f}, "
        f"replicate {m['val_replicate_std']:.3f}){by_hist}"
    )


# ------------------------------------------------------------------------ main


def add_args(p) -> None:
    p.add_argument("--latents", required=True, help="latents.pt of cache_latents")
    p.add_argument("--out", default="project/runs/pred")
    p.add_argument("--kind", choices=tuple(PREDICTORS), default="flow")
    p.add_argument(
        "--arch",
        choices=tuple(ARCHS),
        default="mlp",
        help="mlp: a fixed history of --history latents (the default); seq: "
        "the whole light curve as a sequence of window latents, any length, "
        "summarised by a causal transformer (--seq-* flags, --max-len, "
        "--seq-keep; --hidden and --depth then size the flow head)",
    )
    p.add_argument("--history", type=int, default=3)
    p.add_argument(
        "--seq-keep",
        type=float,
        nargs="+",
        default=[0.3, 1.0],
        metavar="P",
        help="seq: probability of keeping each valid window of a training "
        "sequence; two numbers give a range and every sequence draws its own "
        "(default 0.3 1.0, so full light curves are seen too)",
    )
    p.add_argument("--max-len", type=int, default=64, help="seq: windows per sequence at most")
    p.add_argument("--seq-hidden", type=int, default=256, help="seq: width of the core")
    p.add_argument("--seq-depth", type=int, default=4, help="seq: layers of the core")
    p.add_argument("--seq-heads", type=int, default=4)
    p.add_argument("--seq-dim-head", type=int, default=64)
    p.add_argument("--seq-mlp", type=int, default=1024, help="seq: MLP width of the core")
    p.add_argument("--seq-dropout", type=float, default=0.1)
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
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="sequences per step (default 1024); objects per step with --arch seq (default 64)",
    )
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
    p.add_argument(
        "--val-objects", type=int, default=4000, help="seq: validation objects scored"
    )
    p.add_argument("--eval-samples", type=int, default=8)
    p.add_argument("--eval-horizons", type=int, default=4)
    p.add_argument("--eval-batch", type=int, default=4096)
    p.add_argument(
        "--nll-steps",
        type=int,
        default=100,
        help="reverse Euler steps of the density evaluation (flow); more is tighter",
    )
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
    seq_arch = args.arch == "seq"
    if args.batch_size is None:
        args.batch_size = 64 if seq_arch else 1024
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
    if seq_arch:
        print(
            f"sequences: {train.n_seq_objects} train and {val.n_seq_objects} val objects "
            f"with 3+ valid windows, longest {train.seq_rows.shape[1]} / {val.seq_rows.shape[1]}, "
            f"max_len {args.max_len}, keep {args.seq_keep}"
        )

    if ckpt is not None:
        model = build_predictor(ckpt["hparams"])
    elif seq_arch:
        hp = dict(
            kind=args.kind,
            arch="seq",
            dim=int(lmeta["dim"]),
            hidden=args.seq_hidden,
            depth=args.seq_depth,
            heads=args.seq_heads,
            dim_head=args.seq_dim_head,
            mlp_dim=args.seq_mlp,
            dropout=args.seq_dropout,
            max_len=args.max_len,
            head_hidden=args.hidden,
            head_depth=args.depth,
        )
        if args.kind == "flow":
            hp.update(sigma0=args.sigma0, n_euler=args.n_euler, reproject=args.reproject)
        model = build_predictor(hp)
    else:
        hp = dict(
            kind=args.kind,
            arch="mlp",
            dim=int(lmeta["dim"]),
            history=H,
            hidden=args.hidden,
            depth=args.depth,
        )
        if args.kind == "flow":
            hp.update(sigma0=args.sigma0, n_euler=args.n_euler, reproject=args.reproject)
        model = build_predictor(hp)
    model.to(dev)  # before set_stats: the buffers must sit where the latents are
    if ckpt is None:
        model.set_stats(train.z[train.valid])
    print(f"predictor {model.kind} ({model.arch}): {n_params(model) / 1e6:.2f}M params, {dev}")
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
    # set) and the nulls, both at the final advance range (mlp) or on the
    # full sequences (seq).
    gen_val = torch.Generator(device=dev).manual_seed(args.seed + 1)
    if seq_arch:
        v_seq = val.draw_sequences(args.val_objects, 1.0, args.max_len, gen_val, train=False)
        nulls = fit_nulls_seq(model, train, args.max_len, gen_val)
    else:
        v_rows, v_adv = val.draw(
            args.val_sequences, args.advance, H, stride, gen_val, extra=args.eval_horizons - 1
        )
        nulls = fit_nulls(model, train, args.advance, stride, gen_val)
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
        if seq_arch:
            m = evaluate_seq(model, val, v_seq, nulls, args, args.seed + 3)
        else:
            m = evaluate(model, val, v_rows, v_adv, nulls, args, args.seed + 3)
        m.update(step=step, elapsed=elapsed, train_loss=train_loss, eval_seconds=time.time() - t0)
        print(f"  eval @ {step}: {describe(m, args.eval_horizons)} | {time.time() - t0:.0f}s", flush=True)
        log.write(kind="eval", **m)
        # the by-history bins go to wandb as val/mse_by_history/<bin>/<stat>
        flat = {f"val/{k[4:]}": v for k, v in m.items() if k.startswith("val_") and not isinstance(v, dict)}
        for name, stats in m.get("val_mse_by_history", {}).items():
            flat[f"val/mse_by_history/{name}"] = stats
        tracker.log({"train_mean/loss": train_loss, **flat}, step=step)
        return m

    model.train()
    run, n_acc, t_last, t_run, stop = 0.0, 0, time.time(), time.time(), False
    while step < args.steps and not stop:
        adv_range = current_advance(step)
        if seq_arch:
            seq = train.draw_sequences(args.batch_size, args.seq_keep, args.max_len, gen, train=True)
            z = train.gather(seq["rows"].clamp(min=0), gen)
            loss = model.loss(z, seq["gaps"], seq["mask"], gen)
        else:
            rows, adv = train.draw(args.batch_size, adv_range, H, stride, gen)
            z = train.gather(rows, gen)
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
