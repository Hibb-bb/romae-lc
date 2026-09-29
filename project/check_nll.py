"""Is the flow predictor's ``val_nll`` a real density or an Euler artefact?

The anchored log density integrates the reverse flow with ``--n-euler``
Euler steps and replaces each step's exact ``log|det(I - ds J)|`` by its
first-order term ``-ds tr J`` (the Hutchinson estimate). ``log(1 - u) <= -u``
for every eigenvalue, so the linearised value is an upper bound on the true
log density whenever ``ds J`` is not small: a flow with a stiff velocity
field then reports an NLL that is too good. This script recomputes the NLL
of a fixed set of validation sequences with more reverse steps and, on a few
sequences, with the exact per-step log-determinant (``torch.func.jacrev``),
and prints them next to the sample spread and the Gaussian persistence null.

    python -m project.check_nll --pred project/runs/pred_w250/pred.pt \\
        --latents project/runs/mae_w250/latents.pt --device cuda
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from project.train_predictor import LatentStore, anchored_log_density, load_predictor


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default="project/runs/pred_w250/pred.pt")
    ap.add_argument("--latents", default="project/runs/mae_w250/latents.pt")
    ap.add_argument("--n", type=int, default=1024, help="validation sequences (Hutchinson)")
    ap.add_argument("--n-exact", type=int, default=32, help="sequences for the exact log-det")
    ap.add_argument("--n-fit", type=int, default=50000, help="training sequences for the fitted nulls")
    ap.add_argument("--steps", type=int, nargs="+", default=[20, 100, 400, 1600])
    ap.add_argument("--exact-steps", type=int, nargs="+", default=[20, 200])
    ap.add_argument("--advance", type=float, nargs=2, default=(1.0, 1.5))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    dev = torch.device(args.device)

    model, _ = load_predictor(args.pred, dev)
    cache = torch.load(args.latents, map_location="cpu", weights_only=False)
    stride = float(cache["meta"]["stride"])
    val = LatentStore(cache, "validation", dev, int(cache["meta"]["min_tokens"]))
    val.n_tokens_min = int(cache["meta"]["min_tokens"])
    del cache
    H, D = model.history, model.dim
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    rows, adv = val.draw(args.n, tuple(args.advance), H, stride, gen)
    z = val.z[rows]
    hist, target, a_h = z[:, :H], z[:, H], adv[:, :H]
    hs, ts = model.normalize(hist), model.normalize(target)
    d = ts - hs[:, -1]
    with torch.no_grad():
        s = torch.stack([model.normalize(model.sample(hist, a_h, generator=gen)) for _ in range(8)])
    print(
        f"{len(rows)} validation sequences: true change std {d.std():.3f}, flow sample std "
        f"{s.std(0).mean():.3f}; mse of the sample mean {((s.mean(0) - ts) ** 2).mean():.3f}, "
        f"persistence {(d ** 2).mean():.3f}",
        flush=True,
    )
    m, v = d.mean(0), d.var(0) + 1e-6
    nll_g = 0.5 * ((d - m).square() / v + (2 * math.pi * v).log()).mean()
    print(f"diagonal gaussian persistence null: nll/dim {nll_g:.3f}", flush=True)

    # Stronger nulls, fitted on training sequences: a full-covariance Gaussian
    # of the change (captures a low-dimensional change subspace linearly) and
    # a ridge regression of the change on the flattened history plus the
    # advances, with the full covariance of its residual.
    train = LatentStore(torch.load(args.latents, map_location="cpu", weights_only=False), "train", dev, val.n_tokens_min)
    r_tr, a_tr = train.draw(args.n_fit, tuple(args.advance), H, stride, gen)
    z_tr = model.normalize(train.z[r_tr])
    d_tr = z_tr[:, H] - z_tr[:, H - 1]
    del train

    def full_gauss_nll(resid_tr, resid_va):
        mu = resid_tr.mean(0)
        c = torch.cov((resid_tr - mu).T) + 1e-4 * torch.eye(D, device=dev)
        chol = torch.linalg.cholesky(c)
        y = torch.cholesky_solve((resid_va - mu).T, chol).T
        maha = ((resid_va - mu) * y).sum(-1)
        logdet = 2 * chol.diagonal().log().sum()
        return (0.5 * (maha + logdet + D * math.log(2 * math.pi))).mean() / D

    print(f"full-covariance gaussian persistence null: nll/dim {full_gauss_nll(d_tr, d):.3f}", flush=True)

    def design(zz, aa):
        return torch.cat([zz[:, :H].flatten(1), aa[:, :H].log(), torch.ones(len(zz), 1, device=dev)], 1)

    x_tr, y_tr = design(z_tr, a_tr), d_tr
    x_va = design(torch.cat([hs, ts[:, None]], 1), a_h)
    lam = 1e-2 * len(x_tr)
    w = torch.linalg.solve(x_tr.T @ x_tr + lam * torch.eye(x_tr.shape[1], device=dev), x_tr.T @ y_tr)
    res_tr, res_va = y_tr - x_tr @ w, d - x_va @ w
    print(
        f"ridge-from-history null ({len(x_tr)} train seq): mse {res_va.square().mean():.3f} "
        f"(flow sample-mean mse {((s.mean(0) - ts) ** 2).mean():.3f}, persistence {(d ** 2).mean():.3f}); "
        f"nll/dim with full residual covariance {full_gauss_nll(res_tr, res_va):.3f}",
        flush=True,
    )

    def hutch(n_steps, n_probes=1, sub=None):
        hh, xx, aa = (hs, ts, a_h) if sub is None else (hs[:sub], ts[:sub], a_h[:sub])
        t0 = time.time()
        lp = anchored_log_density(
            lambda x, s_: model.v(x, s_, hh, aa), xx, hh[:, -1], model.sigma0, n_steps, n_probes, gen
        )
        return -lp.mean().item() / D, time.time() - t0

    for n_steps in args.steps:
        for probes in (1, 8) if n_steps == args.steps[0] else (1,):
            nll, dt = hutch(n_steps, probes)
            print(f"hutchinson  steps {n_steps:5d} probes {probes}: nll/dim {nll:.3f}  ({dt:.0f}s)", flush=True)

    def exact(n_steps, sub):
        """Reverse Euler with the exact log|det(I - ds J)| of every step."""
        hh, xx, aa = hs[:sub], ts[:sub], a_h[:sub]
        x = xx.clone()
        ds = 1.0 / n_steps
        logdet = torch.zeros(sub, device=dev)
        eye = torch.eye(D, device=dev)
        t0 = time.time()
        for i in range(n_steps, 0, -1):
            s_ = torch.full((sub,), i * ds, device=dev)
            vs, js = [], []
            for b in range(sub):
                f = lambda xb: model.v(xb[None], s_[b : b + 1], hh[b : b + 1], aa[b : b + 1])[0]
                js.append(torch.func.jacrev(f)(x[b]).detach())
                with torch.no_grad():
                    vs.append(f(x[b]))
            vv, jm = torch.stack(vs), torch.stack(js)  # detached: no graph across steps
            step_exact = torch.linalg.slogdet(eye - ds * jm)[1].detach()
            step_lin = -ds * torch.einsum("bii->b", jm)
            logdet = logdet + step_exact
            if i in (n_steps, max(1, n_steps // 2), 1):
                print(
                    f"    step {i:4d}: exact {step_exact.mean():.4f} vs linearised {step_lin.mean():.4f} "
                    f"per step; spectral norm of J {torch.linalg.matrix_norm(jm, 2).mean():.1f}",
                    flush=True,
                )
            x = x - ds * vv.detach()
        base = -0.5 * (((x - hh[:, -1]) / model.sigma0).square().sum(-1) + D * math.log(2 * math.pi * model.sigma0**2))
        return -(base + logdet).mean().item() / D, time.time() - t0

    for n_steps in args.exact_steps:
        nll, dt = exact(n_steps, args.n_exact)
        print(f"exact logdet steps {n_steps:5d} ({args.n_exact} seq): nll/dim {nll:.3f}  ({dt:.0f}s)", flush=True)
        nll, dt = hutch(n_steps, 1, sub=args.n_exact)
        print(f"hutchinson   steps {n_steps:5d} ({args.n_exact} seq): nll/dim {nll:.3f}  ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
