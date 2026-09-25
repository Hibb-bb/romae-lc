"""Conditional flow matching with the linear (optimal-transport) interpolant
(Lipman et al. 2023), shared by the residual model and the decoder.

``x_s = (1 - (1 - sigma_min) s) eps + s x_1`` for ``s`` in [0, 1] with
``eps ~ N(0, I)``; the regression target of the velocity field is
``x_1 - (1 - sigma_min) eps``. :func:`sample` integrates ``dx/ds = v(x, s)``
from ``s = 0`` to ``1`` with Euler or midpoint steps; :func:`log_density`
integrates the reverse flow with a Hutchinson divergence estimate.
"""

from __future__ import annotations

import math

import torch

SIGMA_MIN = 1e-4


def interpolate(x1: torch.Tensor, eps: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``s`` broadcastable to ``x1`` (e.g. ``[B, 1, 1]``)."""
    return (1.0 - (1.0 - SIGMA_MIN) * s) * eps + s * x1


def target(x1: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return x1 - (1.0 - SIGMA_MIN) * eps


def expand(s: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """``[B] -> [B, 1, ..., 1]`` with the rank of ``like``."""
    return s.view(-1, *([1] * (like.dim() - 1)))


def fm_loss(
    v: torch.Tensor, x1: torch.Tensor, eps: torch.Tensor, mask=None
) -> torch.Tensor:
    """Mean squared error of the velocity, over ``mask`` (True = count) when given."""
    err = (v.float() - target(x1, eps).float()).square()
    if mask is None:
        return err.mean()
    m = mask.to(err.dtype)
    while m.dim() < err.dim():
        m = m[..., None]
    return (err * m).sum() / (m.expand_as(err).sum().clamp(min=1))


def time_embedding(
    s: torch.Tensor, dim: int = 64, max_period: float = 1e4
) -> torch.Tensor:
    """Sinusoidal embedding ``[B, dim]`` of flow times in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=s.device) / half
    )
    ang = s.float()[:, None] * 1000.0 * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], -1)


@torch.no_grad()
def sample(
    v_fn, eps: torch.Tensor, n_steps: int = 8, method: str = "euler"
) -> torch.Tensor:
    """Integrate from ``x_0 = eps`` to ``x_1``; ``v_fn(x, s)`` with ``s [B]``."""
    x = eps
    b = x.shape[0]
    ds = 1.0 / n_steps
    for i in range(n_steps):
        s = torch.full((b,), i * ds, device=x.device, dtype=torch.float32)
        if method == "euler":
            x = x + ds * v_fn(x, s)
        elif method == "midpoint":
            k1 = v_fn(x, s)
            x = x + ds * v_fn(x + 0.5 * ds * k1, s + 0.5 * ds)
        else:
            raise ValueError(f"method must be euler|midpoint, got {method!r}")
    return x


def sample_grad(v_fn, eps: torch.Tensor, n_steps: int = 8, method: str = "euler"):
    """Like :func:`sample` but differentiable (for energies over the input)."""
    x = eps
    b = x.shape[0]
    ds = 1.0 / n_steps
    for i in range(n_steps):
        s = torch.full((b,), i * ds, device=x.device, dtype=torch.float32)
        if method == "euler":
            x = x + ds * v_fn(x, s)
        else:
            k1 = v_fn(x, s)
            x = x + ds * v_fn(x + 0.5 * ds * k1, s + 0.5 * ds)
    return x


def log_density(
    v_fn, x1: torch.Tensor, n_steps: int = 16, n_probes: int = 1, generator=None
):
    """``log p(x1) [B]`` by integrating the reverse flow from ``s = 1`` to 0
    with a Hutchinson estimate of ``div v`` (``n_probes`` Rademacher probes);
    ``x1`` is ``[B, D]``. Runs with autograd enabled."""
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
                    torch.randint(
                        0, 2, x.shape, device=x.device, generator=generator
                    ).float()
                    * 2
                    - 1
                )
                (jvp,) = torch.autograd.grad(
                    (v * probe).sum(), x_req, retain_graph=True, allow_unused=True
                )
                if jvp is not None:  # a field that ignores x has zero divergence
                    div = div + (jvp * probe).sum(-1)
            div = div / n_probes
        x = x - ds * v.detach()
        logdet = logdet + ds * div.detach()  # d log p / ds = -div v along the flow
    base = -0.5 * (x.square().sum(-1) + d * math.log(2 * math.pi))
    return base + logdet
