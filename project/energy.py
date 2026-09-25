"""Energies over latent paths and the inference procedures of the design doc
(section 7): MAP by gradient descent, Langevin sampling, a periodicity energy.

A path is ``Z [C, K, D]``: ``C`` chains over the ``K`` windows of a grid
(``frame_grid(fill=True)`` with advance ``a`` in window units, so the action
after every window is ``a``). With the stage-1 model frozen:

    E_prior(z_k)         = 1/2 ||z_k||^2 / D
    E_dyn(z_<=k -> z_k)  = mean_D (P(z_{k-h..k-1}, a) - z_k)^2        (stage 1)
                         = 1/2 r^T Sigma(a)^-1 r / D                  (stage 2 Gaussian)
    E_obs(z_k, x_k)      = mean_j (m_j - mu_hat(t_j, b_j | z_k))^2 / (2 sigma_j^2)  (stage 3)

all per window and O(1), so the weights ``w_dyn : w_obs : w_prior`` set the
balance (start by equalising the per-window magnitudes on training data, see
``infer.py calibrate``). ``E_dyn`` uses the predictor's sliding context of
``history`` windows, exactly the teacher-forced ``surprise`` of the package,
and ``E_obs`` the decoder's mean estimate. Everything is differentiable in
``Z`` so :func:`map_path` and :func:`langevin` run on the sum.
"""

from __future__ import annotations

import numpy as np
import torch

from romae_lc import Tokens

from project.decoder import decode_mean


def repeat_tokens(tokens: Tokens, c: int) -> Tokens:
    """The same window for ``c`` chains: ``[1, N] -> [c, N]``."""
    f = lambda t: None if t is None else t.expand(c, *t.shape[1:]).contiguous()
    return Tokens(
        f(tokens.values), f(tokens.positions), f(tokens.pad_mask), f(tokens.extras)
    )


class PathEnergy:
    def __init__(
        self,
        model,
        decoder=None,
        residual=None,
        w_dyn=1.0,
        w_obs=1.0,
        w_prior=0.1,
        obs_steps=10,
    ):
        self.model, self.decoder, self.residual = model, decoder, residual
        self.w_dyn, self.w_obs, self.w_prior, self.obs_steps = (
            w_dyn,
            w_obs,
            w_prior,
            obs_steps,
        )
        self.history = model.history
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if decoder is not None:
            decoder.eval()
            for p in decoder.parameters():
                p.requires_grad_(False)

    # ---- terms

    def prior(self, Z: torch.Tensor) -> torch.Tensor:
        """``[C, K]``."""
        return 0.5 * Z.square().mean(-1)

    def dyn(self, Z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``[C, K - 1]``: ``[:, k - 1]`` is the energy of window ``k`` given its
        context; ``actions [K, 1]`` (the advance after each window)."""
        c, k_n, d = Z.shape
        h = self.history
        out = Z.new_zeros(c, k_n - 1)
        if k_n < 2:
            return out
        by_len: dict[int, list[int]] = {}
        for k in range(1, k_n):
            by_len.setdefault(min(k, h), []).append(k)
        a_all = actions.to(Z.device).float()
        for length, ks in by_len.items():
            ctx = torch.stack([Z[:, k - length : k] for k in ks], 1).flatten(
                0, 1
            )  # [C*n, L, D]
            a = torch.stack([a_all[k - length : k] for k in ks], 0)  # [n, L, 1]
            a = a[None].expand(c, *a.shape).flatten(0, 1)
            pred = self.model.predict(ctx, a)[:, -1].view(c, len(ks), d)
            target = torch.stack([Z[:, k] for k in ks], 1)
            r = target - pred
            if self.residual is not None:
                dt = (
                    torch.stack([a_all[k - 1, 0] for k in ks])[None]
                    .expand(c, -1)
                    .flatten()
                )
                e = self.residual.energy(r.flatten(0, 1), dt).view(c, len(ks)) / d
            else:
                e = r.square().mean(-1)
            for j, k in enumerate(ks):
                out[:, k - 1] = e[:, j]
        return out

    def obs(self, Z: torch.Tensor, windows: list[tuple[int, Tokens]]) -> torch.Tensor:
        """``[C, len(windows)]`` for observed windows ``(k, tokens)`` with one
        row each; ``mean_j (m_j - mu_hat_j)^2 / (2 sigma_j^2)`` over real points."""
        if self.decoder is None:
            raise ValueError("E_obs needs a decoder")
        c = Z.shape[0]
        outs = []
        for k, tok in windows:
            tok = repeat_tokens(tok, c)
            mu = decode_mean(
                self.decoder, Z[:, k], tok, n_steps=self.obs_steps, grad=True
            )
            m, sig, real = tok.values[..., 0].float(), tok.extras.float(), ~tok.pad_mask
            e = ((m - mu).square() / (2 * sig.square())) * real
            outs.append(e.sum(-1) / real.sum(-1).clamp(min=1))
        return torch.stack(outs, 1) if outs else Z.new_zeros(c, 0)

    def total(self, Z, actions, windows=None):
        """``(E [C], parts)`` with ``parts`` the per-window terms (detached)."""
        pr, dy = self.prior(Z), self.dyn(Z, actions)
        e = self.w_prior * pr.sum(1) + self.w_dyn * dy.sum(1)
        parts = dict(prior=pr.detach(), dyn=dy.detach())
        if windows and self.decoder is not None and self.w_obs:
            ob = self.obs(Z, windows)
            e = e + self.w_obs * ob.sum(1)
            parts["obs"] = ob.detach()
        return e, parts


# ---------------------------------------------------------------------- init


@torch.no_grad()
def init_path(
    model, frames: list[Tokens], actions: torch.Tensor, observed
) -> torch.Tensor:
    """``Z0 [1, K, D]``: encoded latents of the observed windows, the others
    filled by rolling the predictor forward from the last observed context
    (leading unobserved windows copy the first observed latent)."""
    model.eval()
    observed = np.asarray(observed, dtype=bool)
    Z = model.encode(frames).float()  # [1, K, D]
    h = model.history
    first = int(np.flatnonzero(observed)[0]) if observed.any() else 0
    a = actions.to(Z.device).float()
    for k in range(Z.shape[1]):
        if observed[k]:
            continue
        if k < first:
            Z[:, k] = Z[:, first]
            continue
        lo = max(0, k - h)
        Z[:, k] = model.predict(Z[:, lo:k], a[lo:k][None])[:, -1]
    return Z


# ----------------------------------------------------------------- inference


def map_path(
    energy: PathEnergy, Z0, actions, windows=None, steps=300, lr=1e-2, log=None
):
    """Gradient descent (Adam) on the path energy; returns ``(Z, trace)``."""
    Z = Z0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=lr)
    trace = []
    for i in range(steps):
        e, parts = energy.total(Z, actions, windows)
        opt.zero_grad()
        e.sum().backward()
        opt.step()
        trace.append(float(e.mean()))
        if log and (i % max(1, steps // 5) == 0 or i == steps - 1):
            log(
                f"  map step {i}: E {trace[-1]:.4f} "
                + " ".join(f"{k} {v.sum(1).mean():.4f}" for k, v in parts.items())
            )
    return Z.detach(), trace


def langevin(
    energy: PathEnergy,
    Z,
    actions,
    windows=None,
    n_chains=64,
    steps=500,
    eta=1e-3,
    temperature=1.0,
    generator=None,
    log=None,
):
    """Unadjusted Langevin dynamics from ``Z [1, K, D]`` (usually the MAP):
    ``Z <- Z - eta grad E + sqrt(2 eta T) xi``; returns ``(samples [C, K, D],
    energy trace)``."""
    Zc = Z.detach().expand(n_chains, *Z.shape[1:]).clone().requires_grad_(True)
    trace = []
    for i in range(steps):
        e, _ = energy.total(Zc, actions, windows)
        (g,) = torch.autograd.grad(e.sum(), Zc)
        with torch.no_grad():
            noise = torch.randn(Zc.shape, device=Zc.device, generator=generator)
            Zc -= eta * g
            Zc += (2 * eta * temperature) ** 0.5 * noise
        trace.append(float(e.mean()))
        if log and (i % max(1, steps // 5) == 0 or i == steps - 1):
            log(f"  langevin step {i}: E {trace[-1]:.4f}")
    return Zc.detach(), trace


# ---------------------------------------------------------------- periodicity


def interpolate_path(
    Z: torch.Tensor, t: np.ndarray, t_query: np.ndarray
) -> torch.Tensor:
    """Linear interpolation of a path ``Z [K, D]`` at times ``t [K]`` (sorted)
    to ``t_query [Q]`` inside ``[t_0, t_{K-1}]``."""
    tq = np.clip(t_query, t[0], t[-1])
    j = np.clip(np.searchsorted(t, tq, side="right") - 1, 0, len(t) - 2)
    w = (tq - t[j]) / np.maximum(t[j + 1] - t[j], 1e-12)
    w = torch.as_tensor(w, dtype=Z.dtype, device=Z.device)[:, None]
    return (1 - w) * Z[j] + w * Z[j + 1]


def periodicity_energy(Z: torch.Tensor, t: np.ndarray, period: float) -> torch.Tensor:
    """``E_per(Z, P) = mean_k ||z(t_k) - z(t_k + P)||^2 / D`` over the windows
    whose shifted time stays inside the path (``Z [K, D]``, ``t`` in days).
    Only periods comparable to or longer than the window spacing are
    resolvable on a per-window path."""
    valid = t + period <= t[-1]
    if valid.sum() < 1:
        return Z.new_tensor(float("nan"))
    zs = interpolate_path(Z, t, t[valid] + period)
    return (Z[valid] - zs).square().mean()


def periodicity_scan(Z: torch.Tensor, t: np.ndarray, periods: np.ndarray) -> np.ndarray:
    return np.array([float(periodicity_energy(Z, t, float(p))) for p in periods])
