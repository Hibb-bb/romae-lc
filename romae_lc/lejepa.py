"""LeJEPA for light curves (Balestriero & LeCun 2025, arXiv:2511.08544).

Multi-view invariance plus SIGReg, a sliced Epps-Pulley goodness-of-fit test
that pushes projected embeddings toward an isotropic Gaussian. Views of one
object are token batches (see :class:`~romae_lc.tokenize.Tokens`) that differ
in token count, so each view is encoded separately and the pooled ``[N, D]``
embeddings are stacked for the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

_ACTS = {"relu": lambda: nn.ReLU(inplace=True), "gelu": nn.GELU, "silu": nn.SiLU}


def mlp(
    dims: list[int], batch_norm: bool = True, activation: str = "relu"
) -> nn.Sequential:
    """``Linear -> [BN] -> activation`` blocks ending in a plain ``Linear``.

    ``activation`` is ``"relu"`` (LeJEPA projector), ``"gelu"`` (the LeWM
    projector, see :func:`~romae_lc.lewm.lewm_mlp`) or ``"silu"``.
    """
    if activation not in _ACTS:
        raise ValueError(
            f"activation must be one of {sorted(_ACTS)}, got {activation!r}"
        )
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            if batch_norm:
                layers.append(nn.BatchNorm1d(b))
            layers.append(_ACTS[activation]())
    return nn.Sequential(*layers)


def _ddp() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class EppsPulley(nn.Module):
    """Epps-Pulley test statistic for univariate normality on ``[..., N, S]``
    (N samples of S slices); returns ``[..., S]``. Means are averaged across
    DDP ranks so the statistic sees the global batch; the process group is
    looked up on every call, so the module may be built before it exists."""

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2 (trapezoid rule), got {n_points}")
        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        phi = (-0.5 * t**2).exp()
        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(-2)
        x_t = x.unsqueeze(-1) * self.t
        cos_mean, sin_mean = x_t.cos().mean(-3), x_t.sin().mean(-3)
        ddp = _ddp()
        world_size = torch.distributed.get_world_size() if ddp else 1
        if ddp:  # imported lazily: torch.distributed.nn needs USE_DISTRIBUTED
            from torch.distributed.nn import all_reduce

            cos_mean = all_reduce(cos_mean, op=torch.distributed.ReduceOp.AVG)
            sin_mean = all_reduce(sin_mean, op=torch.distributed.ReduceOp.AVG)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * n * world_size


class SlicedEppsPulley(nn.Module):
    """SIGReg: mean Epps-Pulley statistic over random 1-D projections.

    ``x`` is ``[N, D]`` or ``[V, N, D]``; with views the statistic is computed
    per view over the N samples and averaged. A step counter seeds the
    projections so all DDP ranks draw identical directions.
    """

    def __init__(self, n_slices: int = 1024, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        self.n_slices = n_slices
        self.ep = EppsPulley(t_max, n_points)
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            g = torch.Generator(device=x.device).manual_seed(int(self.step))
            a = torch.randn(x.size(-1), self.n_slices, device=x.device, generator=g)
            a = a / a.norm(dim=0)
            self.step.add_(1)
        return self.ep(x @ a).mean()


@dataclass
class LeJEPAOutput:
    """Result of a :class:`LeJEPA` forward.

    Attributes:
        loss: ``inv_loss + lamb * sigreg_loss`` (0 in eval mode).
        inv_loss: Invariance term (MSE of each view's projection to the mean
            of the global projections).
        sigreg_loss: SIGReg term.
        embedding: Detached backbone embeddings of the global views
            ``[n_global * N, D]`` (train) or ``[N, D]`` (eval).
        projection: Detached projector outputs of the same rows.
        features: Undetached backbone embeddings of every view, view-major
            ``[n_views * N, D]`` (train only), for auxiliary losses.
    """

    loss: torch.Tensor
    inv_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    embedding: torch.Tensor
    projection: torch.Tensor
    features: torch.Tensor | None = None


class LeJEPA(nn.Module):
    """Multi-view invariance + SIGReg over a pooled light-curve encoder.

    ``self.backbone`` embeds every view, ``self.proj`` maps embeddings to the
    space the loss lives in, and the optional ``self.predictor`` is applied
    *between* them to the views of one instrument (``predictor_inst``): a
    noisy survey's embedding is mapped into the shared space before the
    loss, while downstream consumers still read the raw encoder output.

    Args:
        backbone: Pooled encoder ``(values, positions, pad_mask) -> [N, D]``
            exposing ``embed_dim``, e.g. :class:`~romae_lc.model.RoMAE`.
        proj: Projector; default ``mlp([D, 2048, 2048, 512])``.
        predictor: Optional ``[N, D] -> [N, D]`` head, e.g. ``mlp([D, 1024, D])``.
        predictor_inst: Instrument id (as in ``view_inst``) routed through it.
        lamb: SIGReg weight.
        n_slices: Random projections of the sliced test.
        t_max: Epps-Pulley integration bound.
        n_points: Epps-Pulley quadrature nodes.

    Example::

        model = LeJEPA(RoMAE(encoder=dict(d_model=192, nhead=3, depth=4)))
        out = model(global_views=[view_a, view_b], local_views=[view_c])
        out.loss.backward()
        model.eval(); z = model.embed(*view_a)  # [N, D]
    """

    def __init__(
        self,
        backbone: nn.Module,
        proj: nn.Module | None = None,
        predictor: nn.Module | None = None,
        predictor_inst: int | None = None,
        lamb: float = 0.02,
        n_slices: int = 1024,
        t_max: float = 3.0,
        n_points: int = 17,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim
        self.proj = proj if proj is not None else mlp([self.embed_dim, 2048, 2048, 512])
        self.predictor, self.predictor_inst = predictor, predictor_inst
        self.sigreg = SlicedEppsPulley(n_slices, t_max, n_points)
        self.lamb = lamb

    def embed(self, values, positions, pad_mask=None) -> torch.Tensor:
        """Backbone embedding ``[N, D]`` of one token batch."""
        return self.backbone(values, positions, pad_mask)

    def _predict(self, feats: list[torch.Tensor], view_inst) -> list[torch.Tensor]:
        """Route the ``predictor_inst`` rows of each view through the predictor."""
        if self.predictor is None or self.predictor_inst is None:
            return feats
        if view_inst is None:
            raise ValueError("view_inst [N, n_views] is required with a predictor")
        out = []
        for j, f in enumerate(feats):
            rows = (view_inst[:, j] == self.predictor_inst).to(f.device)
            if rows.any():
                f = f.clone()
                single = int(rows.sum()) == 1 and self.predictor.training
                if single:  # BatchNorm cannot train on one row: use running stats
                    self.predictor.eval()
                f[rows] = self.predictor(f[rows]).to(f.dtype)
                if single:
                    self.predictor.train()
            out.append(f)
        return out

    def loss_on_views(self, global_views, local_views=None, view_inst=None):
        """``(loss, inv_loss, sigreg_loss, feats, projected [V, N, K])``."""
        views = list(global_views) + list(local_views or [])
        n_global = len(global_views)
        feats = [self.backbone(*v) for v in views]
        proj = self.proj(torch.cat(self._predict(feats, view_inst)))
        proj = proj.view(len(views), -1, proj.shape[-1])
        centers = proj[:n_global].mean(0)
        inv_loss = (centers[None] - proj).square().mean()
        sigreg_loss = self.sigreg(proj)
        return inv_loss + self.lamb * sigreg_loss, inv_loss, sigreg_loss, feats, proj

    def forward(
        self,
        global_views=None,
        local_views=None,
        view_inst: torch.Tensor | None = None,
        values=None,
        positions=None,
        pad_mask=None,
    ) -> LeJEPAOutput:
        """Training: loss over ``global_views`` (and ``local_views``), each a
        :class:`~romae_lc.tokenize.Tokens` or ``(values, positions, pad_mask)``
        of the same objects; ``view_inst [N, n_views]`` gives each view's
        instrument id for the predictor. Eval: embed ``(values, positions,
        pad_mask)``."""
        if self.training:
            if not global_views:
                raise ValueError("global_views are required in training mode")
            n_global = len(global_views)
            loss, inv, sig, feats, proj = self.loss_on_views(
                global_views, local_views, view_inst
            )
            return LeJEPAOutput(
                loss=loss,
                inv_loss=inv,
                sigreg_loss=sig,
                embedding=torch.cat(feats[:n_global]).detach(),
                projection=proj[:n_global].reshape(-1, proj.shape[-1]).detach(),
                features=torch.cat(feats),
            )
        if values is None:
            raise ValueError("values are required in eval mode")
        z = self.backbone(values, positions, pad_mask)
        zero = z.new_zeros(())
        return LeJEPAOutput(
            zero, zero, zero, embedding=z.detach(), projection=self.proj(z).detach()
        )
