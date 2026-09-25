"""Per-band baselines for imputation: constant, linear interpolation and a
Gaussian process with an RBF kernel (length scale chosen by the marginal
likelihood on the context points), plus a quasi-periodic GP that uses the
catalogue period and is therefore an oracle, not a fair baseline."""

from __future__ import annotations

import numpy as np


def constant(t_ctx, y_ctx, e_ctx, t_q):
    mu = np.median(y_ctx) if y_ctx.size else 0.0
    var = np.var(y_ctx) if y_ctx.size > 1 else 1.0
    return np.full(len(t_q), mu), np.full(len(t_q), max(var, 1e-6))


def linear(t_ctx, y_ctx, e_ctx, t_q):
    if y_ctx.size == 0:
        return constant(t_ctx, y_ctx, e_ctx, t_q)
    order = np.argsort(t_ctx)
    mu = np.interp(t_q, t_ctx[order], y_ctx[order])
    var = np.var(y_ctx) if y_ctx.size > 1 else 1.0
    return mu, np.full(len(t_q), max(var, 1e-6))


def _gp(kernel, t_ctx, y_ctx, e_ctx, t_q, params_grid):
    if y_ctx.size < 3:
        return constant(t_ctx, y_ctx, e_ctx, t_q)
    mean, amp2 = float(y_ctx.mean()), max(float(y_ctx.var()), 1e-6)
    yc = y_ctx - mean
    best = None
    for params in params_grid:
        k = amp2 * kernel(t_ctx[:, None], t_ctx[None, :], params) + np.diag(
            e_ctx**2 + 1e-6 * amp2
        )
        try:
            chol = np.linalg.cholesky(k)
        except np.linalg.LinAlgError:
            continue
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, yc))
        lml = -0.5 * yc @ alpha - np.log(np.diag(chol)).sum()
        if best is None or lml > best[0]:
            best = (lml, params, chol, alpha)
    if best is None:
        return constant(t_ctx, y_ctx, e_ctx, t_q)
    _, params, chol, alpha = best
    ks = amp2 * kernel(t_q[:, None], t_ctx[None, :], params)
    mu = mean + ks @ alpha
    v = np.linalg.solve(chol, ks.T)
    var = np.maximum(amp2 - (v * v).sum(0), 1e-6)
    return mu, var


def _rbf(a, b, params):
    (ell,) = params
    return np.exp(-0.5 * ((a - b) / ell) ** 2)


def gp_rbf(t_ctx, y_ctx, e_ctx, t_q, lengthscales=(0.5, 2.0, 10.0, 50.0, 200.0)):
    return _gp(_rbf, t_ctx, y_ctx, e_ctx, t_q, [(ell,) for ell in lengthscales])


def _quasi_periodic(a, b, params):
    period, ell_p, ell = params
    d = a - b
    return np.exp(-2 * np.sin(np.pi * d / period) ** 2 / ell_p**2) * np.exp(
        -0.5 * (d / ell) ** 2
    )


def gp_periodic(
    t_ctx, y_ctx, e_ctx, t_q, period, ell_ps=(0.3, 0.6, 1.2), ells=(1e3, 1e5)
):
    """Oracle: quasi-periodic GP at the catalogue period."""
    grid = [(period, lp, ell) for lp in ell_ps for ell in ells]
    return _gp(_quasi_periodic, t_ctx, y_ctx, e_ctx, t_q, grid)


BASELINES = dict(constant=constant, linear=linear, gp_rbf=gp_rbf)


def per_band(fn, t_ctx, y_ctx, e_ctx, b_ctx, t_q, b_q, **kw):
    """Apply a baseline band by band; returns ``(mu, var)`` for the queries."""
    mu, var = np.zeros(len(t_q)), np.ones(len(t_q))
    for b in np.unique(b_q):
        q, c = b_q == b, b_ctx == b
        mu[q], var[q] = fn(t_ctx[c], y_ctx[c], e_ctx[c], t_q[q], **kw)
    return mu, var


def gaussian_nll(y, mu, var):
    """Per-point negative log-likelihood under ``N(mu, var)``."""
    return 0.5 * ((y - mu) ** 2 / var + np.log(2 * np.pi * var))
