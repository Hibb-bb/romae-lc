"""LeWorldModel for light curves (Maes, Le Lidec et al. 2026, arXiv:2603.19312).

A JEPA world model trained end to end: an encoder embeds every frame, a
projector maps the embedding to the latent ``z_t``, and a causal AdaLN-zero
transformer predicts ``z_{t+1}`` from ``z_{<=t}`` and the actions
``a_{<=t}``. The loss is the next-latent MSE plus SIGReg (the sliced
Epps-Pulley test of :mod:`romae_lc.lejepa`) on the latents of every frame;
there is no stop-gradient and no EMA target. A frame here is one time window
of a light curve (see :mod:`romae_lc.frames`) and the action is the advance
to the next window, but any ``[B, T, A]`` action tensor works.

The predictor, action encoder and projector are ports of the official
``lucas-maes/le-wm`` modules with their attribute names and defaults; the
official code applies no custom initialisation beyond AdaLN-zero and the
random positional embedding, so nothing here calls
:func:`romae_lc.model._init_weights`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lejepa import SlicedEppsPulley, mlp


def lewm_mlp(dim: int, hidden: int = 2048, out_dim: int | None = None) -> nn.Sequential:
    """The LeWM projector ``Linear(dim, hidden) -> BatchNorm1d(hidden) -> GELU
    -> Linear(hidden, out_dim or dim)`` (official ``module.MLP`` with
    ``norm_fn=BatchNorm1d``, ``act_fn=GELU``; paper Sec. 3.1 "1-layer MLP with
    Batch Normalization"), used for both ``projector`` and ``pred_proj``.
    BatchNorm keeps PyTorch defaults (eps 1e-5, momentum 0.1, affine)."""
    return mlp([dim, hidden, out_dim or dim], batch_norm=True, activation="gelu")


class ActionEncoder(nn.Module):
    """Action embedder (official ``module.Embedder``): ``patch_embed =
    Conv1d(action_dim, smoothed_dim, kernel_size=1)`` over the time axis (a
    per-step affine map, kept as a Conv1d for parity and state-dict
    portability) then ``embed = Linear(smoothed_dim, mlp_scale * dim) -> SiLU
    -> Linear(mlp_scale * dim, dim)``. The input is cast to float32
    (official ``x.float()``) and permuted ``[B, T, A] -> [B, A, T]`` around the
    convolution.

    Args:
        action_dim: A, channels of the action vector (official: frameskip
            times the environment action width; here 1, the time advance).
        dim: Output width, the predictor's conditioning width.
        smoothed_dim: Conv output channels (official default 10).
        mlp_scale: Hidden multiplier (official default 4).

    The official pipeline also z-scores every action column over the dataset
    (``utils.get_column_normalizer``) and zeroes NaNs at trajectory
    boundaries (``train.py`` ``nan_to_num``). Neither is ported: the
    light-curve action is O(1) by construction and windows have no NaN
    boundaries. Users passing their own actions should z-score them.
    """

    def __init__(
        self, action_dim: int, dim: int, smoothed_dim: int = 10, mlp_scale: int = 4
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(action_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * dim, dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """``[B, T, A] -> [B, T, dim]``."""
        x = actions.float().permute(0, 2, 1)
        x = self.patch_embed(x).permute(0, 2, 1)
        return self.embed(x)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


class _Attention(nn.Module):
    """Causal multi-head self-attention (official ``module.Attention``)."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5  # unused: SDPA's default scale is the same
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        q, k, v = (
            t.unflatten(-1, (self.heads, -1)).transpose(1, 2)
            for t in self.to_qkv(x).chunk(3, dim=-1)
        )  # [B, heads, T, dim_head]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        return self.to_out(out.transpose(1, 2).flatten(2))


class _FeedForward(nn.Module):
    """``LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout``
    (official ``module.FeedForward``)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning (official
    ``module.ConditionalBlock``). The modulation's last ``Linear`` is
    zero-initialised (weight and bias), so at init the block is the identity
    and the attention/FFN parameters receive exactly zero gradient. The
    non-affine ``norm1``/``norm2`` followed by the affine norm inside the
    attention and FFN is the official double normalisation, ported as is."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout=0.0):
        super().__init__()
        self.attn = _Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = _FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ARPredictor(nn.Module):
    """AdaLN-zero causal transformer predicting the next latent at every
    position (official ``module.ARPredictor`` + ``Transformer`` with
    ``ConditionalBlock``; the ``transformer.`` prefix of official state dicts
    is flattened away here, nothing else is renamed).

    Defaults are the released ``config/train/model/lewm.yaml`` predictor and
    the ``module.py`` defaults: 6 layers, 16 heads of width 64 (inner
    attention width 1024, independent of ``dim``), MLP 2048, 10 % dropout on
    attention probabilities, attention output and both FFN dropouts, no
    embedding dropout, learned ``randn`` positional embedding of length
    ``n_frames``.

    Args:
        dim: Input width D (official ``input_dim = embed_dim = 192``).
        n_frames: Length of the positional embedding, the maximum context
            (official ``num_frames = history_size = 3``).
        depth: Number of blocks.
        heads: Attention heads.
        dim_head: Width per head.
        mlp_dim: FFN hidden width.
        dropout: Attention and FFN dropout.
        emb_dropout: Dropout on ``x + pos`` before the blocks.
        hidden_dim: Block width; ``None`` -> ``dim`` (official ``hidden_dim =
            embed_dim``).
        cond_dim: Conditioning width; ``None`` -> ``dim``. An extension: the
            official ``Transformer`` assumes the action embedding is
            ``input_dim`` wide; this reduces to it when ``cond_dim == dim``.
        out_dim: Output width; ``None`` -> ``dim``.
    """

    def __init__(
        self,
        dim: int,
        n_frames: int = 3,
        depth: int = 6,
        heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
        hidden_dim: int | None = None,
        cond_dim: int | None = None,
        out_dim: int | None = None,
    ):
        super().__init__()
        if n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}")
        hidden = hidden_dim or dim
        self.dim, self.n_frames, self.depth = dim, n_frames, depth
        self.heads, self.dim_head, self.mlp_dim = heads, dim_head, mlp_dim
        self.dropout_p, self.emb_dropout = dropout, emb_dropout
        self.hidden_dim, self.cond_dim = hidden, cond_dim or dim
        self.out_dim = out_dim or dim
        self.pos_embedding = nn.Parameter(torch.randn(1, n_frames, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.input_proj = nn.Linear(dim, hidden) if dim != hidden else nn.Identity()
        self.cond_proj = (
            nn.Linear(self.cond_dim, hidden)
            if self.cond_dim != hidden
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden, self.out_dim) if hidden != self.out_dim else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                _ConditionalBlock(hidden, heads, dim_head, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden)

    @property
    def hparams(self) -> dict:
        """Constructor arguments that rebuild this predictor."""
        return dict(
            dim=self.dim,
            n_frames=self.n_frames,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout_p,
            emb_dropout=self.emb_dropout,
            hidden_dim=self.hidden_dim,
            cond_dim=self.cond_dim,
            out_dim=self.out_dim,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """``x [B, T, dim]`` with ``1 <= T <= n_frames``, ``c [B, T, cond_dim]``
        or ``None`` (zero conditioning: every AdaLN emits its learned bias
        only; the official model is always action-conditioned). Returns
        ``[B, T, out_dim]``; position ``t`` depends only on ``x[:, :t + 1]``
        and ``c[:, :t + 1]``."""
        b, t, _ = x.shape
        if not 1 <= t <= self.n_frames:
            raise ValueError(
                f"sequence length {t} must be in [1, n_frames={self.n_frames}]"
            )
        if c is None:
            c = x.new_zeros(b, t, self.cond_dim)
        elif c.shape != (b, t, self.cond_dim):
            raise ValueError(
                f"c must have shape {(b, t, self.cond_dim)}, got {tuple(c.shape)}"
            )
        x = self.dropout(x + self.pos_embedding[:, :t])
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        return self.output_proj(self.norm(x))


@dataclass
class LeWMOutput:
    """Result of a :class:`LeWorldModel` forward.

    Naming follows the paper, not :class:`~romae_lc.lejepa.LeJEPAOutput`:
    ``embedding`` is the post-projector latent ``z_t`` (what the paper
    predicts, regularises and probes) and the backbone output is
    ``features``.

    Attributes:
        loss: ``pred_loss + lamb * sigreg_loss``.
        pred_loss: Mean squared next-latent error over ``B, T - 1, D``.
        sigreg_loss: SIGReg over the latents of all ``T`` frames.
        embedding: Detached latents ``z [B, T, D]``.
        predicted: Detached predictions ``[B, T - 1, D]``; ``predicted[:, t]``
            is the prediction of frame ``t + 1``.
        features: Undetached backbone embeddings ``[B, T, D_backbone]`` for
            auxiliary losses and probes.
        straightness: Detached :func:`straightness` of ``embedding`` (NaN when
            ``T < 3``), the diagnostic the paper logs (App. H).
    """

    loss: torch.Tensor
    pred_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    embedding: torch.Tensor
    predicted: torch.Tensor
    features: torch.Tensor
    straightness: torch.Tensor


def straightness(z: torch.Tensor) -> torch.Tensor:
    """Temporal path straightness (paper Eq. 9, App. H): with velocities
    ``v_t = z_{t+1} - z_t`` the mean over batch and ``t`` of
    ``cos(v_t, v_{t+1})`` (eps 1e-8 in the norms). ``z`` is ``[B, T, D]``
    with ``T >= 3``; a NaN scalar otherwise."""
    z = z.float()
    if z.shape[1] < 3:
        return z.new_full((), float("nan"))
    v = z[:, 1:] - z[:, :-1]
    return F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1, eps=1e-8).mean()


def _hidden(m: nn.Module) -> int | None:
    """Hidden width of a :func:`lewm_mlp`, ``0`` for an ``nn.Identity`` (no
    projector at all, e.g. over a frozen encoder); ``None`` for anything
    else."""
    if isinstance(m, nn.Identity):
        return 0
    if isinstance(m, nn.Sequential) and len(m) and isinstance(m[0], nn.Linear):
        return m[0].out_features
    return None


def _projector(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    """:func:`lewm_mlp` of ``hidden`` width, or ``nn.Identity`` for 0."""
    if hidden == 0:
        if in_dim != out_dim:
            raise ValueError(
                f"an Identity projector needs equal widths, got {in_dim} -> {out_dim}"
            )
        return nn.Identity()
    return lewm_mlp(in_dim, hidden, out_dim)


class LeWorldModel(nn.Module):
    """LeWorldModel (Maes, Le Lidec et al. 2026, arXiv:2603.19312) over a
    pooled light-curve encoder.

    ``self.backbone`` embeds every frame (a time window, see
    :mod:`romae_lc.frames`); ``self.projector`` maps it to the latent
    ``z_t``; ``self.action_encoder`` embeds the action that follows each
    frame; ``self.predictor`` predicts ``z_{t+1}`` from ``z_{<=t}, a_{<=t}``
    with causal masking; ``self.pred_proj`` is applied to every prediction.
    Loss = MSE(pred, next z) + lamb * SIGReg(z). No stop-gradient, no EMA:
    gradients flow through both sides of the MSE and through all modules
    (paper Sec. 3.1; official ``train.py``).

    Args:
        backbone: Pooled encoder ``(values, positions, pad_mask) ->
            [B, D_backbone]`` exposing ``embed_dim``, e.g.
            :class:`~romae_lc.model.RoMAE` (any rope/attention/pool variant).
        projector: Default ``lewm_mlp(D_backbone, 2048, embed_dim)``.
        predictor: Default ``ARPredictor(embed_dim, n_frames=history)``
            (depth 6, heads 16, dim_head 64, mlp 2048, dropout 0.1).
        action_encoder: Default ``ActionEncoder(action_dim, embed_dim)``.
        pred_proj: Default ``lewm_mlp(embed_dim, 2048)``.
        embed_dim: Latent width D; default ``backbone.embed_dim`` (official:
            encoder width = latent width = 192).
        action_dim: A; 1 = the time advance of :mod:`romae_lc.frames`.
        history: Predictor context length (official ``history_size`` 3; paper
            App. D). Sets the default predictor's ``n_frames`` and the context
            kept during :meth:`rollout` and :meth:`surprise`.
        lamb: SIGReg weight, 0.1 (paper Sec. 3.1 and Alg. 3; the released
            config uses 0.09 and paper Fig. 16 is flat on [0.01, 0.2]).
        n_slices: Random projections of SIGReg (official ``num_proj`` 1024).
        t_max: Epps-Pulley integration bound (official ``linspace(0, 3, 17)``).
        n_points: Epps-Pulley quadrature nodes (official ``knots`` 17).

    Example::

        model = LeWorldModel(RoMAE(encoder=dict(d_model=192, nhead=3, depth=12)))
        out = model(batch["frames"], batch["actions"])  # T = 4 Tokens, [B, 4, 1]
        out.loss.backward()
        model.eval()
        z = model.encode(batch["frames"])  # [B, 4, D]
        z_hat = model.rollout(batch["frames"][:3], batch["actions"])  # [B, 5, D]
    """

    def __init__(
        self,
        backbone: nn.Module,
        projector: nn.Module | None = None,
        predictor: nn.Module | None = None,
        action_encoder: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
        embed_dim: int | None = None,
        action_dim: int = 1,
        history: int = 3,
        lamb: float = 0.1,
        n_slices: int = 1024,
        t_max: float = 3.0,
        n_points: int = 17,
    ):
        super().__init__()
        if history < 1:
            raise ValueError(f"history must be >= 1, got {history}")
        self.backbone = backbone
        self.embed_dim = embed_dim or backbone.embed_dim
        self.action_dim, self.history, self.lamb = action_dim, history, lamb
        self.n_slices, self.t_max, self.n_points = n_slices, t_max, n_points
        self.projector = (
            projector
            if projector is not None
            else lewm_mlp(backbone.embed_dim, 2048, self.embed_dim)
        )
        self.predictor = (
            predictor
            if predictor is not None
            else ARPredictor(self.embed_dim, n_frames=history)
        )
        n_frames = getattr(self.predictor, "n_frames", None)
        if n_frames is not None and n_frames < history:
            raise ValueError(
                f"predictor.n_frames={n_frames} is shorter than history={history}"
            )
        self.action_encoder = (
            action_encoder
            if action_encoder is not None
            else ActionEncoder(action_dim, self.embed_dim)
        )
        self.pred_proj = (
            pred_proj if pred_proj is not None else lewm_mlp(self.embed_dim, 2048)
        )
        self.sigreg = SlicedEppsPulley(n_slices, t_max, n_points)

    @property
    def hparams(self) -> dict:
        """Arguments for :meth:`from_hparams`; ``proj_hidden`` /
        ``pred_proj_hidden`` are ``0`` for an ``nn.Identity`` and ``None``
        for other custom projector modules."""
        return dict(
            embed_dim=self.embed_dim,
            action_dim=self.action_dim,
            history=self.history,
            lamb=self.lamb,
            n_slices=self.n_slices,
            t_max=self.t_max,
            n_points=self.n_points,
            predictor=getattr(self.predictor, "hparams", None),
            proj_hidden=_hidden(self.projector),
            pred_proj_hidden=_hidden(self.pred_proj),
        )

    @classmethod
    def from_hparams(cls, backbone: nn.Module, hparams: dict) -> "LeWorldModel":
        """Rebuild an (untrained) model from :attr:`hparams`; the caller then
        ``load_state_dict``. ``predictor`` is an :class:`ARPredictor` from its
        hparams when that entry is a dict (else the default), the projectors
        are :func:`lewm_mlp` with the stored hidden widths, or ``nn.Identity``
        for a width of 0 (a world model over a frozen encoder); ``ValueError``
        when a width is ``None`` (custom modules cannot be rebuilt)."""
        kw = dict(hparams)
        pred = kw.pop("predictor", None)
        proj_hidden = kw.pop("proj_hidden", None)
        pred_proj_hidden = kw.pop("pred_proj_hidden", None)
        if proj_hidden is None or pred_proj_hidden is None:
            raise ValueError(
                "proj_hidden and pred_proj_hidden must be ints: custom projector "
                "modules cannot be rebuilt from hparams"
            )
        embed_dim = kw.get("embed_dim") or backbone.embed_dim
        return cls(
            backbone,
            projector=_projector(backbone.embed_dim, proj_hidden, embed_dim),
            predictor=ARPredictor(**pred) if isinstance(pred, dict) else None,
            pred_proj=_projector(embed_dim, pred_proj_hidden, embed_dim),
            **kw,
        )

    def embed(self, values, positions, pad_mask=None) -> torch.Tensor:
        """Backbone embedding ``[B, D_backbone]`` of one token batch
        (pre-projector), for probes; mirrors :meth:`LeJEPA.embed`."""
        return self.backbone(values, positions, pad_mask)

    def _project(self, feats: torch.Tensor) -> torch.Tensor:
        """Projector on the ``(B * T, D_backbone)`` rows (official ``(b t)``
        flattening, so BatchNorm sees ``B * T`` rows)."""
        b, t, _ = feats.shape
        return self.projector(feats.flatten(0, 1)).unflatten(0, (b, t))

    def encode(self, frames: Sequence, project: bool = True) -> torch.Tensor:
        """Latents ``[B, T, D]`` of ``T`` frames, each a
        :class:`~romae_lc.tokenize.Tokens` or ``(values, positions,
        pad_mask)`` of the same ``B`` objects. Frames are encoded one backbone
        call each (their token counts differ) and stacked on dim 1;
        ``project=False`` returns the backbone features ``[B, T,
        D_backbone]``. Train-mode BatchNorm needs at least two rows, which
        ``T >= 2`` guarantees even for ``B = 1``."""
        feats = torch.stack([self.backbone(*f) for f in frames], 1)
        return self._project(feats) if project else feats

    def predict(
        self, emb: torch.Tensor, actions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``emb [B, T, D]``, ``actions [B, T, A]`` or ``None`` -> ``z_hat
        [B, T, D]`` (action encoder -> predictor -> ``pred_proj`` on the
        ``B * T`` rows, official ``JEPA.predict``); position ``t`` is the
        prediction of frame ``t + 1``. ``None`` runs the predictor with zero
        conditioning."""
        b, t, _ = emb.shape
        c = None
        if actions is not None:
            if actions.shape != (b, t, self.action_dim):
                raise ValueError(
                    f"actions must have shape {(b, t, self.action_dim)}, "
                    f"got {tuple(actions.shape)}"
                )
            c = self.action_encoder(actions)
        pred = self.predictor(emb, c)
        return self.pred_proj(pred.flatten(0, 1)).unflatten(0, (b, t))

    def _check_actions(self, actions, b: int, t: int) -> None:
        if actions is not None and actions.shape != (b, t, self.action_dim):
            raise ValueError(
                f"actions must have shape {(b, t, self.action_dim)} (one per "
                f"frame), got {tuple(actions.shape)}"
            )

    def forward(self, frames: Sequence, actions: torch.Tensor | None = None):
        """Training or validation loss on ``T >= 2`` frames with ``T - 1 <=
        predictor.n_frames``. ``actions`` is ``[B, T, A]`` (one per frame; the
        last is unused here, as in the official data layout where every frame
        carries the action that follows it) or ``None``.

        With ``T = history + 1`` (the default 4) this is exactly the official
        ``train.py``: context ``emb[:, :history]``, target ``emb[:, 1:]``, no
        detach. For ``T > history + 1`` every position predicts the next frame
        under causal masking (paper Alg. 3); the official ``num_preds > 1``
        k-step mode is not implemented. Both losses are evaluated in float32
        even under bf16 autocast (autocast is disabled around the SIGReg term,
        whose projection matmuls would otherwise run in the autocast dtype,
        as they do in the official code). Works in eval mode too (BatchNorm
        running statistics, no dropout).
        """
        t = len(frames)
        if t < 2:
            raise ValueError(f"need at least 2 frames, got {t}")
        n_frames = getattr(self.predictor, "n_frames", None)
        if n_frames is not None and t - 1 > n_frames:
            raise ValueError(
                f"{t} frames need a predictor with n_frames >= {t - 1}, "
                f"got n_frames={n_frames}"
            )
        feats = self.encode(frames, project=False)
        emb = self._project(feats)
        self._check_actions(actions, emb.shape[0], t)
        pred = self.predict(emb[:, :-1], None if actions is None else actions[:, :-1])
        target = emb[:, 1:]
        pred_loss = (pred.float() - target.float()).square().mean()
        with torch.autocast(device_type=emb.device.type, enabled=False):
            sigreg_loss = self.sigreg(emb.float().transpose(0, 1))
        loss = pred_loss + self.lamb * sigreg_loss
        return LeWMOutput(
            loss=loss,
            pred_loss=pred_loss,
            sigreg_loss=sigreg_loss,
            embedding=emb.detach(),
            predicted=pred.detach(),
            features=feats,
            straightness=straightness(emb.detach()),
        )

    @torch.no_grad()
    def rollout(
        self,
        frames: Sequence,
        actions: torch.Tensor | None = None,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """Open-loop latent rollout (official ``JEPA.rollout``).

        ``frames`` are ``H >= 1`` context windows; ``actions`` is ``[B, H +
        n_steps - 1, A]`` (``a_i`` follows frame ``i``: ``T`` actions yield
        ``T + 1`` latents, the official convention) or ``None`` with
        ``n_steps`` given; ``n_steps`` defaults to ``actions.shape[1] - H +
        1``. Each step conditions on the last ``history`` latents and their
        actions and appends the predicted one. Returns ``[B, H + n_steps,
        D]``, the ``H`` encoded context latents first; ``H > history`` is
        accepted (only the last ``history`` condition each step, as in the
        official ``emb[:, -HS:]``). Runs with the whole model temporarily in
        eval mode (BatchNorm running statistics, no dropout)."""
        h = len(frames)
        if h < 1:
            raise ValueError("rollout needs at least one context frame")
        if actions is None and n_steps is None:
            raise ValueError("give actions or n_steps")
        if n_steps is None:
            n_steps = actions.shape[1] - h + 1
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if actions is not None and actions.shape[1] < h + n_steps - 1:
            raise ValueError(
                f"{h} context frames and {n_steps} steps need "
                f"{h + n_steps - 1} actions, got {actions.shape[1]}"
            )
        was = self.training
        self.eval()
        try:
            z = self.encode(frames)
            for _ in range(n_steps):
                n = z.shape[1]
                a = None if actions is None else actions[:, :n][:, -self.history :]
                z_next = self.predict(z[:, -self.history :], a)[:, -1:]
                z = torch.cat([z, z_next], 1)
        finally:
            self.train(was)
        return z

    @torch.no_grad()
    def surprise(
        self, frames: Sequence, actions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-step prediction error ``[B, T - 1]`` along ``T >= 2`` frames,
        the "surprise" of paper Sec. 5.2 (the paper defines it only as the
        discrepancy between predicted and actual future; the rule here is
        teacher forcing with a sliding context): for ``t = 1..T-1`` the
        context is frames ``max(0, t - history)..t-1`` with their actions and
        the error is the mean over D of ``(z_hat_t - z_t)^2`` on the
        post-projector latents. ``actions`` is ``[B, T, A]`` or ``None`` (only
        the first ``T - 1`` are used). For ``T = history + 1`` this equals
        ``((predicted - embedding[:, 1:]) ** 2).mean(-1)`` of :meth:`forward`
        in eval mode. Same eval-mode switch as :meth:`rollout`."""
        t = len(frames)
        if t < 2:
            raise ValueError(f"need at least 2 frames, got {t}")
        was = self.training
        self.eval()
        try:
            z = self.encode(frames)
            self._check_actions(actions, z.shape[0], t)
            errs = []
            for i in range(1, t):
                lo = max(0, i - self.history)
                a = None if actions is None else actions[:, lo:i]
                z_hat = self.predict(z[:, lo:i], a)[:, -1]
                errs.append((z_hat.float() - z[:, i].float()).square().mean(-1))
        finally:
            self.train(was)
        return torch.stack(errs, 1)
