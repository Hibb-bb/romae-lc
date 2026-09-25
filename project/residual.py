"""Stage 2 (M4): the residual of the deterministic predictor.

On frozen stage-1 latents, ``r_t = z_{t+1} - P(z_{<=t}, a_{<=t})`` is
collected over the training windows with its advance ``Delta`` (window
units) and modelled at two levels:

1. :class:`GaussianResidual`: a Gaussian per ``Delta`` bin (diagonal by
   default, ``--full`` for a full covariance with shrinkage), fitted by
   maximum likelihood. Its energy is ``1/2 (r - m)^T Sigma^-1 (r - m)``.
2. :class:`FlowResidual` (``--flow``): a conditional flow ``v(r_s, s | z_t,
   Delta)`` (MLP with skips) from ``N(0, I)`` to the z-scored residual; a few
   Euler or midpoint steps sample it, the energy is the norm surrogate or
   the Hutchinson log density (:func:`project.flow.log_density`).

The script also writes the diagnostic that decides between the levels
(histograms and bimodality coefficients of the residual on its top
principal directions per superclass and ``Delta`` bin), checks that the
width grows with ``Delta``, and evaluates forecasting on validation
sequences: the Gaussian NLL of the true latent at horizons of 1..h windows
under 64 stochastic rollouts (mean step plus sampled residual), against
persistence and the deterministic rollout. Horizons beyond what the records
can host at the checkpoint's window are dropped with a note.

    python -m project.residual --ckpt project/runs/wm_w500/wm.pt --out project/runs/res_w500
    python -m project.residual --ckpt ... --flow --flow-steps 5000
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from romae_lc import FrameDataset

from project.common import (
    JsonlLog,
    add_device_arg,
    data_args_from,
    dump_json,
    frame_loader,
    get_device,
    load_data,
    load_wm,
    save_atomic,
    seed_all,
    subset,
    superclass,
    to_device,
)
from project.flow import fm_loss, interpolate, log_density, sample, time_embedding
from project.tracking import Tracker, add_wandb_args

# ------------------------------------------------------------------ collection


@torch.no_grad()
def collect(model, loader, device, max_batches=None) -> dict:
    """Residuals ``r [N, D]`` with their advance ``dt [N]``, context latent
    ``z [N, D]``, ``label [N]`` and record ``index [N]`` over a loader of
    frame sequences (model in eval mode)."""
    model.eval()
    out = dict(r=[], dt=[], z=[], label=[], index=[])
    for i, batch in enumerate(loader):
        frames, actions = to_device(batch, device)
        z = model.encode(frames)
        pred = model.predict(z[:, :-1], actions[:, :-1])
        r = (z[:, 1:] - pred).float()
        t = r.shape[1]
        out["r"].append(r.flatten(0, 1).cpu())
        out["dt"].append(actions[:, :-1, 0].flatten().float().cpu())
        out["z"].append(z[:, :-1].flatten(0, 1).float().cpu())
        out["label"].append(batch["label"].repeat_interleave(t))
        out["index"].append(batch["index"].repeat_interleave(t))
        if max_batches and i + 1 >= max_batches:
            break
    return {k: torch.cat(v) for k, v in out.items()}


def bin_edges(dt: torch.Tensor, n_bins: int) -> list[float]:
    """Interior edges of ``n_bins`` quantile bins of log ``dt``."""
    q = torch.quantile(dt.log(), torch.linspace(0, 1, n_bins + 1)[1:-1]).exp()
    return [float(v) for v in q]


# -------------------------------------------------------------------- Gaussian


class GaussianResidual(nn.Module):
    """Per-``Delta``-bin Gaussian residual model (level 1)."""

    def __init__(
        self, dim: int, edges: list[float], full: bool = False, shrink: float = 0.05
    ):
        super().__init__()
        nb = len(edges) + 1
        self.dim, self.full, self.shrink = dim, full, shrink
        self.register_buffer("edges", torch.tensor(list(edges), dtype=torch.float32))
        self.register_buffer("mean", torch.zeros(nb, dim))
        self.register_buffer("var", torch.ones(nb, dim))
        self.register_buffer("prec", torch.eye(dim).repeat(nb, 1, 1))
        self.register_buffer("chol", torch.eye(dim).repeat(nb, 1, 1))
        self.register_buffer("logdet", torch.zeros(nb))
        self.register_buffer("count", torch.zeros(nb))

    @property
    def hparams(self) -> dict:
        return dict(
            dim=self.dim, edges=self.edges.tolist(), full=self.full, shrink=self.shrink
        )

    @property
    def n_bins(self) -> int:
        return len(self.edges) + 1

    def bins(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.bucketize(dt.float().contiguous(), self.edges.to(dt.device))

    @torch.no_grad()
    def fit(self, r: torch.Tensor, dt: torch.Tensor) -> "GaussianResidual":
        b = self.bins(dt)
        for i in range(self.n_bins):
            x = r[b == i].double()
            n = x.shape[0]
            self.count[i] = n
            if n < 2:
                continue
            m = x.mean(0)
            v = x.var(0, unbiased=False) + 1e-6
            self.mean[i], self.var[i] = m.float(), v.float()
            cov = torch.diag(v)
            if self.full and n > self.dim + 1:
                xc = x - m
                cov = (1 - self.shrink) * (xc.T @ xc / n) + self.shrink * torch.diag(v)
                cov = cov + 1e-6 * torch.eye(self.dim, dtype=cov.dtype)
            self.prec[i] = torch.linalg.inv(cov).float()
            self.chol[i] = torch.linalg.cholesky(cov).float()
            self.logdet[i] = torch.logdet(cov).float()
        return self

    def energy(self, r: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """``1/2 (r - m)^T Sigma^-1 (r - m)`` per row, differentiable in ``r``."""
        b = self.bins(dt)
        d = r.float() - self.mean[b]
        if self.full:
            return 0.5 * torch.einsum("ni,nij,nj->n", d, self.prec[b], d)
        return 0.5 * (d.square() / self.var[b]).sum(-1)

    def nll(self, r: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        b = self.bins(dt)
        return (
            self.energy(r, dt)
            + 0.5 * self.logdet[b]
            + 0.5 * self.dim * math.log(2 * math.pi)
        )

    @torch.no_grad()
    def sample(self, dt: torch.Tensor, generator=None) -> torch.Tensor:
        b = self.bins(dt)
        xi = torch.randn(
            len(dt), self.dim, device=self.mean.device, generator=generator
        )
        return self.mean[b] + torch.einsum("nij,nj->ni", self.chol[b], xi)

    def width(self) -> list[float]:
        """Mean residual variance per bin (should grow with ``Delta``)."""
        return [float(v) for v in self.var.mean(-1)]


# ------------------------------------------------------------------------ flow


class FlowResidual(nn.Module):
    """Conditional flow-matching residual model (level 2): an MLP with skip
    connections predicting the velocity of the z-scored residual given the
    context latent ``z_t``, ``log Delta`` and the flow time."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        hidden: int = 512,
        depth: int = 4,
        s_dim: int = 64,
    ):
        super().__init__()
        self.dim, self.cond_dim, self.hidden, self.depth, self.s_dim = (
            dim,
            cond_dim,
            hidden,
            depth,
            s_dim,
        )
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))
        self.register_buffer("dt_stats", torch.tensor([0.0, 1.0]))
        self.inp = nn.Linear(dim + cond_dim + 1 + s_dim, hidden)
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

    @property
    def hparams(self) -> dict:
        return dict(
            dim=self.dim,
            cond_dim=self.cond_dim,
            hidden=self.hidden,
            depth=self.depth,
            s_dim=self.s_dim,
        )

    @torch.no_grad()
    def set_stats(self, r: torch.Tensor, dt: torch.Tensor) -> None:
        self.mu.copy_(r.mean(0))
        self.sd.copy_(r.std(0) + 1e-6)
        self.dt_stats.copy_(torch.stack([dt.log().mean(), dt.log().std() + 1e-6]))

    def normalize(self, r):
        return (r - self.mu) / self.sd

    def denormalize(self, r):
        return r * self.sd + self.mu

    def v(self, x, s, z, dt):
        ld = (dt.float().log() - self.dt_stats[0]) / self.dt_stats[1]
        h = F.silu(
            self.inp(
                torch.cat(
                    [x, z.float(), ld[:, None], time_embedding(s, self.s_dim)], -1
                )
            )
        )
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)

    def loss(self, r, z, dt, generator=None) -> torch.Tensor:
        x1 = self.normalize(r.float())
        s = torch.rand(x1.shape[0], device=x1.device, generator=generator)
        eps = torch.randn(x1.shape, device=x1.device, generator=generator)
        x_s = interpolate(x1, eps, s[:, None])
        return fm_loss(self.v(x_s, s, z, dt), x1, eps)

    @torch.no_grad()
    def sample(
        self, z, dt, n_steps=8, method="midpoint", generator=None
    ) -> torch.Tensor:
        eps = torch.randn(z.shape[0], self.dim, device=z.device, generator=generator)
        x = sample(lambda x, s: self.v(x, s, z, dt), eps, n_steps, method)
        return self.denormalize(x)

    def energy(self, r, z, dt, exact=False, n_steps=16, generator=None) -> torch.Tensor:
        """``-log p(r | z, Delta)`` up to a constant: the norm surrogate
        ``1/2 ||r_n||^2`` of the z-scored residual, or the Hutchinson log
        density of the flow (``exact=True``, no gradient to ``r``)."""
        x1 = self.normalize(r.float())
        if not exact:
            return 0.5 * x1.square().sum(-1)
        lp = log_density(
            lambda x, s: self.v(x, s, z, dt), x1, n_steps, generator=generator
        )
        return -(lp - self.sd.log().sum())


def train_flow(
    flow,
    data,
    device,
    steps=5000,
    batch_size=512,
    lr=1e-4,
    wd=0.01,
    seed=0,
    log=print,
    tracker=None,
):
    flow.to(device).train()
    r, z, dt = (data[k].to(device) for k in ("r", "z", "dt"))
    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=wd)
    gen = torch.Generator(device=device).manual_seed(seed)
    n, run = r.shape[0], 0.0
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch_size,), device=device, generator=gen)
        loss = flow.loss(r[idx], z[idx], dt[idx], generator=gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0 and tracker is not None:
            tracker.log({"train/flow_loss": run}, step=step)
        if step % 500 == 0 or step == steps:
            log(f"  flow step {step}: loss {run:.4f}")
    flow.eval()
    return flow


# -------------------------------------------------------------------- loading


def residual_state(gauss, flow=None, meta=None) -> dict:
    return dict(
        gaussian=dict(state_dict=gauss.state_dict(), hparams=gauss.hparams),
        flow=(
            None
            if flow is None
            else dict(state_dict=flow.state_dict(), hparams=flow.hparams)
        ),
        meta=meta or {},
    )


def load_residual(path, device="cpu"):
    """``(GaussianResidual, FlowResidual | None, meta)``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    g = GaussianResidual(**ckpt["gaussian"]["hparams"])
    g.load_state_dict(ckpt["gaussian"]["state_dict"])
    f = None
    if ckpt.get("flow"):
        f = FlowResidual(**ckpt["flow"]["hparams"])
        f.load_state_dict(ckpt["flow"]["state_dict"])
        f = f.to(device).eval()
    return g.to(device).eval(), f, ckpt.get("meta", {})


# ---------------------------------------------------------------- diagnostics


def bimodality(x: np.ndarray) -> dict:
    """Skewness, excess kurtosis and Sarle's bimodality coefficient (values
    above 5/9 hint at more than one mode)."""
    n = x.size
    if n < 4:
        return dict(n=int(n), skew=None, kurt=None, bc=None)
    xc = x - x.mean()
    m2, m3, m4 = (xc**2).mean(), (xc**3).mean(), (xc**4).mean()
    g, k = m3 / m2**1.5, m4 / m2**2 - 3
    bc = (g**2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return dict(n=int(n), skew=float(g), kurt=float(k), bc=float(bc))


def multimodality(
    r: torch.Tensor,
    dt: torch.Tensor,
    groups: list[str],
    edges: list[float],
    n_pc=3,
    n_hist=40,
    max_rows=50_000,
    seed=0,
):
    """Histograms and bimodality of the standardised residual on its top
    principal directions, per superclass and ``Delta`` bin."""
    x = ((r - r.mean(0)) / (r.std(0) + 1e-6)).double()
    rng = np.random.default_rng(seed)
    sub = x[
        torch.from_numpy(
            rng.choice(x.shape[0], min(max_rows, x.shape[0]), replace=False)
        )
    ]
    _, s, vt = torch.linalg.svd(sub, full_matrices=False)
    explained = (s[:n_pc] ** 2 / (s**2).sum()).tolist()
    proj = (x @ vt[:n_pc].T).numpy()
    bins = torch.bucketize(dt.float(), torch.tensor(edges)).numpy()
    groups = np.asarray(groups)
    stats, hists = {}, {}
    lim = np.linspace(-5, 5, n_hist + 1)
    for g in ["all"] + sorted(set(groups.tolist())):
        for b in range(len(edges) + 1):
            m = (bins == b) & ((groups == g) if g != "all" else True)
            for c in range(n_pc):
                key = f"{g}/bin{b}/pc{c}"
                stats[key] = bimodality(proj[m, c])
                hists[key] = np.histogram(np.clip(proj[m, c], -5, 5), lim)[0]
    flagged = sorted(
        k
        for k, v in stats.items()
        if v["bc"] is not None and v["bc"] > 5 / 9 and v["n"] >= 200
    )
    return (
        dict(
            explained_variance=explained,
            stats=stats,
            flagged=flagged,
            hist_edges=lim.tolist(),
        ),
        hists,
    )


@torch.no_grad()
def forecast_eval(
    model,
    gauss,
    flow,
    records,
    meta,
    device,
    horizons=(1, 2, 3),
    n_chains=64,
    batch_size=32,
    n_objects=2000,
    seed=0,
    log=print,
):
    """Latent forecast NLL per horizon under stochastic rollouts against
    persistence and the deterministic rollout; horizons the records cannot
    host are dropped."""
    history, spec = model.history, meta.spec
    h_max = max(horizons)
    recs = subset(records, n_objects, seed)
    ds = None
    while h_max >= 1:
        cfg = replace(meta.cfg, n_frames=history + h_max)
        try:
            ds = FrameDataset(recs, cfg, seed=seed, epoch_seed=False)
            break
        except ValueError:
            h_max -= 1
    if ds is None:
        log("forecast: no record can host history + 1 windows")
        return {}
    horizons = [h for h in horizons if h <= h_max]
    if h_max < max(horizons or [0]) or len(horizons) < 3:
        log(f"forecast: horizons limited to {horizons} at window {meta.cfg.window}")
    loader = DataLoader(ds, batch_size, shuffle=False, collate_fn=spec.collate())
    gen = torch.Generator(device=device).manual_seed(seed)
    acc = {
        h: dict(
            nll_rollout=[],
            nll_persistence=[],
            nll_det=[],
            mse_rollout=[],
            mse_persistence=[],
            mse_det=[],
        )
        for h in horizons
    }
    model.eval()
    for batch in loader:
        frames, actions = to_device(batch, device)
        z = model.encode(frames)  # [B, T, D]
        b, d = z.shape[0], z.shape[-1]
        c = n_chains
        zc = z[:, :history].repeat_interleave(c, 0)
        ac = actions.repeat_interleave(c, 0)
        zd = z[:, :history]
        for step in range(h_max):
            n = zc.shape[1]
            mean = model.predict(zc[:, -history:], ac[:, :n][:, -history:])[:, -1]
            dt = ac[:, n - 1, 0]
            r = (
                flow.sample(zc[:, -1], dt, generator=gen)
                if flow is not None
                else gauss.sample(dt, generator=gen)
            )
            zc = torch.cat([zc, (mean + r)[:, None]], 1)
            mean_d = model.predict(zd[:, -history:], actions[:, :n][:, -history:])[
                :, -1
            ]
            zd = torch.cat([zd, mean_d[:, None]], 1)
            h = step + 1
            if h not in acc:
                continue
            truth = z[:, history + h - 1].float()
            samples = zc[:, -1].float().view(b, c, d)
            mu, var = samples.mean(1), samples.var(1) + 1e-6
            nll = lambda m_, v_: (
                0.5 * ((truth - m_).square() / v_ + (2 * math.pi * v_).log())
            ).mean(-1)
            pers = z[:, history - 1].float()
            var_d = gauss.var[gauss.bins(actions[:, history - 1, 0])]
            acc[h]["nll_rollout"] += nll(mu, var).tolist()
            acc[h]["nll_persistence"] += nll(pers, var).tolist()
            acc[h]["nll_det"] += nll(zd[:, -1].float(), var_d).tolist()
            acc[h]["mse_rollout"] += (truth - mu).square().mean(-1).tolist()
            acc[h]["mse_persistence"] += (truth - pers).square().mean(-1).tolist()
            acc[h]["mse_det"] += (truth - zd[:, -1].float()).square().mean(-1).tolist()
    out = {}
    for h, v in acc.items():
        out[str(h)] = {k: float(np.mean(x)) for k, x in v.items() if x}
        out[str(h)]["n"] = len(v["nll_rollout"])
        log(
            f"  horizon {h}: NLL rollout {out[str(h)]['nll_rollout']:.3f}  persistence {out[str(h)]['nll_persistence']:.3f}  det {out[str(h)]['nll_det']:.3f} | MSE rollout {out[str(h)]['mse_rollout']:.4f}  persistence {out[str(h)]['mse_persistence']:.4f}  det {out[str(h)]['mse_det']:.4f}"
        )
    return out


# ------------------------------------------------------------------------ main


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--out", required=True, help="output directory (residual.pt, summary.json)"
    )
    p.add_argument(
        "--bins", type=int, default=5, help="Delta bins (quantiles of log Delta)"
    )
    p.add_argument("--full", action="store_true", help="full covariance")
    p.add_argument("--shrink", type=float, default=0.05)
    p.add_argument("--flow", action="store_true", help="also fit the flow residual")
    p.add_argument("--flow-steps", type=int, default=5000)
    p.add_argument("--flow-hidden", type=int, default=512)
    p.add_argument("--flow-depth", type=int, default=4)
    p.add_argument("--horizons", type=int, nargs="*", default=[1, 2, 3])
    p.add_argument("--n-chains", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-objects", type=int, default=None, help="cap train records")
    p.add_argument("--val-objects", type=int, default=2000)
    p.add_argument(
        "--passes", type=int, default=1, help="train passes (fresh windows each)"
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data", default=None)
    p.add_argument("--classes", nargs="*", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    add_device_arg(p)
    add_wandb_args(p)
    args = p.parse_args(argv)
    seed_all(args.seed)
    dev = get_device(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, meta = load_wm(args.ckpt, dev)
    tracker = Tracker(
        args,
        config=dict(vars(args), wm_step=meta.step, window=meta.cfg.window),
        job_type="residual",
    )
    data = load_data(data_args_from(meta.args, args))
    train, val = subset(data["train"], args.train_objects, args.seed), subset(
        data["validation"], args.val_objects, args.seed
    )
    cfg, spec = meta.cfg, meta.spec
    print(
        f"{len(train)} train / {len(val)} val records; checkpoint step {meta.step}; window {cfg.window}"
    )

    t0 = time.time()
    parts = []
    for k in range(args.passes):
        loader = frame_loader(
            train,
            cfg,
            spec,
            args.batch_size,
            train=True,
            workers=args.workers,
            seed=args.seed + k,
        )
        parts.append(collect(model, loader, dev))
    tr = {key: torch.cat([q[key] for q in parts]) for key in parts[0]}
    va = collect(
        model, frame_loader(val, cfg, spec, args.batch_size, seed=args.seed), dev
    )
    print(
        f"collected {tr['r'].shape[0]} train / {va['r'].shape[0]} val residuals in {time.time() - t0:.0f}s"
    )

    edges = bin_edges(tr["dt"], args.bins)
    gauss = GaussianResidual(
        tr["r"].shape[1], edges, full=args.full, shrink=args.shrink
    ).fit(tr["r"], tr["dt"])
    nll_tr, nll_va = gauss.nll(tr["r"], tr["dt"]), gauss.nll(va["r"], va["dt"])
    diag = (
        GaussianResidual(tr["r"].shape[1], edges, full=False).fit(tr["r"], tr["dt"])
        if args.full
        else gauss
    )
    sup_va = [superclass(data["validation"][i]) for i in va["index"].tolist()]
    sup_va = np.array([superclass(val[i]) for i in va["index"].tolist()])
    per_class = {
        g: float(nll_va[torch.from_numpy(sup_va == g)].mean())
        for g in sorted(set(sup_va.tolist()))
    }
    bins_va = gauss.bins(va["dt"])
    per_bin = [
        dict(
            edge_lo=(edges[i - 1] if i else None),
            count=int(gauss.count[i]),
            width=gauss.width()[i],
            nll_val=(
                float(nll_va[bins_va == i].mean()) if (bins_va == i).any() else None
            ),
        )
        for i in range(gauss.n_bins)
    ]
    print(
        f"gaussian ({'full' if args.full else 'diag'}): NLL train {nll_tr.mean():.2f} val {nll_va.mean():.2f}; width per bin {np.round(gauss.width(), 4).tolist()}"
    )
    for row in per_bin:
        print(f"  bin {row}")

    mm, hists = multimodality(
        tr["r"],
        tr["dt"],
        [superclass(train[i]) for i in tr["index"].tolist()],
        edges,
        seed=args.seed,
    )
    np.savez_compressed(
        out / "multimodality.npz", **{k.replace("/", "__"): v for k, v in hists.items()}
    )
    print(
        f"multimodality: top-3 PCs explain {np.round(mm['explained_variance'], 3).tolist()}; flagged groups: {len(mm['flagged'])} of {len(mm['stats'])}"
    )

    flow, flow_summary = None, None
    if args.flow:
        flow = FlowResidual(
            tr["r"].shape[1], tr["z"].shape[1], args.flow_hidden, args.flow_depth
        )
        flow.set_stats(tr["r"], tr["dt"])
        train_flow(
            flow, tr, dev, steps=args.flow_steps, seed=args.seed, tracker=tracker
        )
        n_eval = min(2000, va["r"].shape[0])
        e = flow.energy(
            va["r"][:n_eval].to(dev),
            va["z"][:n_eval].to(dev),
            va["dt"][:n_eval].to(dev),
            exact=True,
        )
        flow_summary = dict(
            nll_val_exact=float(e.mean()), n=n_eval, steps=args.flow_steps
        )
        print(
            f"flow: exact NLL val {flow_summary['nll_val_exact']:.2f} (gaussian {nll_va[:n_eval].mean():.2f} on the same rows)"
        )

    fc = forecast_eval(
        model,
        gauss.to(dev),
        flow,
        val,
        meta,
        dev,
        tuple(args.horizons),
        args.n_chains,
        seed=args.seed,
    )

    save_atomic(
        residual_state(
            gauss.cpu(),
            None if flow is None else flow.cpu(),
            meta=dict(ckpt=str(args.ckpt), step=meta.step, edges=edges),
        ),
        out / "residual.pt",
    )
    dump_json(
        dict(
            ckpt=str(args.ckpt),
            step=meta.step,
            n_train=int(tr["r"].shape[0]),
            n_val=int(va["r"].shape[0]),
            edges=edges,
            full=args.full,
            gaussian=dict(
                nll_train=float(nll_tr.mean()),
                nll_val=float(nll_va.mean()),
                per_bin=per_bin,
                per_superclass_val=per_class,
            ),
            multimodality=dict(
                explained_variance=mm["explained_variance"],
                flagged=mm["flagged"],
                stats=mm["stats"],
            ),
            flow=flow_summary,
            forecast=fc,
        ),
        out / "summary.json",
    )
    tracker.log(
        dict(
            {
                "val/gaussian_nll": float(nll_va.mean()),
                "train/gaussian_nll": float(nll_tr.mean()),
            },
            **{f"val/gaussian_nll_{g}": v for g, v in per_class.items()},
            **(
                {"val/flow_nll_exact": flow_summary["nll_val_exact"]}
                if flow_summary
                else {}
            ),
            **{
                f"forecast/h{h}_{k}": v
                for h, d in fc.items()
                for k, v in d.items()
                if k != "n"
            },
        ),
        step=meta.step,
    )
    tracker.finish()
    print(f"saved {out / 'residual.pt'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
