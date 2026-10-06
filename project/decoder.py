"""Stage 3 (M5): a decoder from the frozen latent back to clean magnitudes.

:class:`QueryDecoder` is a small rotary transformer over the query tokens of
one window, ``(t_j - tau, b_j, log sigma_j)``, with the encoder's own time
ladder (its rotary layout is copied from the stage-1 backbone). The latent
``z`` enters as a token at position 0, the same anchoring trick as the
encoder's CLS token, and so does the flow time ``s``. Two kinds:

- ``flow``: conditional flow matching over the observed magnitudes
  (targets are the noisy ``m_j``, the network is conditioned on
  ``log sigma_j``). The mean estimate ``mu_hat`` for the energies is the
  deterministic solve from ``eps = 0`` (or an average of samples).
- ``mse``: direct regression of ``mu_j`` and an extra log-variance ``s_j``,
  trained with the Gaussian likelihood of variance ``sigma_j^2 + s_j^2``:
  the baseline decoder, one forward per window, the workhorse of ``E_obs``.

The encoder is frozen and, during training, sees the window with a random
fraction of its points dropped (``--encoder-drop``), so the decoder learns to
fill in points the encoder did not see, which is the imputation setting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from romae_lc import Tokens
from romae_lc.model import _init_weights
from romae_lc.rope import BlockRope
from romae_lc.transformer import Transformer, attention_mask, config

from project.flow import fm_loss, interpolate, sample, sample_grad, time_embedding


class QueryDecoder(nn.Module):
    def __init__(
        self,
        z_dim: int,
        rope_layout: list[dict],
        err_stats: tuple[float, float],
        kind: str = "flow",
        d_model: int = 192,
        nhead: int = 3,
        depth: int = 3,
        s_dim: int = 64,
        sigma_input: bool = True,
    ):
        super().__init__()
        if kind not in ("flow", "mse"):
            raise ValueError(f"kind must be flow|mse, got {kind!r}")
        self.kind, self.z_dim, self.s_dim = kind, z_dim, s_dim
        #: whether the query's own log sigma is an input feature. A future
        #: point's error is not known before it is observed, and in
        #: magnitudes the error tracks the brightness (fainter points have
        #: larger errors), so with it the decoder reads the brightness off
        #: the error instead of the latent (found 2026-09-29: with the
        #: errors shuffled the decoder was worse than a constant). New
        #: decoders keep it out; the error stays in the loss variance only.
        self.sigma_input = bool(sigma_input)
        self.rope_layout = [dict(b) for b in rope_layout]
        self.cfg = config(dict(d_model=d_model, nhead=nhead, depth=depth))
        dims = sum(b["dim"] for b in rope_layout)
        if dims != self.cfg.head_dim:
            raise ValueError(
                f"decoder head_dim {self.cfg.head_dim} must equal the encoder's "
                f"rotary layout width {dims} (same d_model / nhead ratio)"
            )
        self.rope = BlockRope(self.cfg.head_dim, nhead, self.rope_layout)
        self.register_buffer(
            "err_stats", torch.tensor(list(err_stats), dtype=torch.float32)
        )
        self.in_proj = nn.Linear(2 if kind == "flow" else 1, d_model)
        self.z_proj = nn.Sequential(
            nn.Linear(z_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.s_proj = (
            nn.Sequential(
                nn.Linear(s_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
            )
            if kind == "flow"
            else None
        )
        self.transformer = Transformer(self.cfg)
        self.norm = nn.RMSNorm(d_model, eps=self.cfg.norm_eps)
        self.head = nn.Linear(d_model, 1 if kind == "flow" else 2)
        self.apply(_init_weights)

    @property
    def hparams(self) -> dict:
        return dict(
            z_dim=self.z_dim,
            rope_layout=self.rope_layout,
            err_stats=self.err_stats.tolist(),
            kind=self.kind,
            d_model=self.cfg.d_model,
            nhead=self.cfg.nhead,
            depth=self.cfg.depth,
            s_dim=self.s_dim,
            sigma_input=self.sigma_input,
        )

    def log_sigma(self, tokens: Tokens) -> torch.Tensor:
        """Standardised ``log sigma [B, N]`` from ``Tokens.extras``, 0 on padding."""
        if tokens.extras is None:
            raise ValueError("tokens carry no extras (per-point sigma)")
        mu, sd = self.err_stats
        e = (tokens.extras.clamp_min(1e-12).log() - mu) / sd
        return e.masked_fill(tokens.pad_mask, 0.0)

    def forward(self, z, positions, pad_mask, log_sigma, x_s=None, s=None):
        """``z [B, Dz]``, ``positions [B, n_axes, N]``, ``pad_mask [B, N]``,
        ``log_sigma [B, N]``; flow kind also ``x_s [B, N]`` and ``s [B]``.
        Returns the velocity ``[B, N]`` (flow) or ``(mu, logvar)`` each
        ``[B, N]`` (mse)."""
        b = z.shape[0]
        if not self.sigma_input:
            log_sigma = torch.zeros_like(log_sigma)
        feats = (
            log_sigma[..., None] if x_s is None else torch.stack([x_s, log_sigma], -1)
        )
        x = self.in_proj(feats.to(self.in_proj.weight.dtype))
        c = self.z_proj(z.to(x.dtype))
        if self.s_proj is not None:
            if s is None:
                raise ValueError("the flow decoder needs the flow time s")
            c = c + self.s_proj(time_embedding(s, self.s_dim).to(x.dtype))
        x = torch.cat([c[:, None], x], 1)
        pos = torch.cat([positions.new_zeros(b, positions.shape[1], 1), positions], 2)
        pad = torch.cat([pad_mask.new_zeros(b, 1), pad_mask], 1)
        x = self.transformer(x, self.rope.prepare(pos), attention_mask(pad))
        out = self.head(self.norm(x[:, 1:])).float()
        if self.kind == "flow":
            return out[..., 0]
        return out[..., 0], out[..., 1]


def decoder_loss(
    dec: QueryDecoder, z: torch.Tensor, tokens: Tokens, generator=None
) -> torch.Tensor:
    """Flow-matching loss (flow) or Gaussian NLL with variance
    ``sigma^2 + exp(logvar)`` (mse) over the real tokens of the windows."""
    m, pad = tokens.values[..., 0].float(), tokens.pad_mask
    mask = ~pad
    ls = dec.log_sigma(tokens)
    if dec.kind == "flow":
        s = torch.rand(m.shape[0], device=m.device, generator=generator)
        eps = torch.randn(m.shape, device=m.device, generator=generator)
        x_s = interpolate(m, eps, s[:, None])
        v = dec(z, tokens.positions, pad, ls, x_s, s)
        return fm_loss(v, m, eps, mask)
    mu, logvar = dec(z, tokens.positions, pad, ls)
    var = gaussian_var(tokens.extras, logvar)
    nll = 0.5 * ((m - mu.float()).square() / var + var.log())
    return (nll * mask).sum() / mask.sum().clamp(min=1)


LOGVAR_RANGE = (-14.0, 10.0)


def gaussian_var(sigma: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """``sigma^2 + exp(logvar)`` in float32 with ``logvar`` clamped to
    :data:`LOGVAR_RANGE` and a floor of 1e-8: an unclamped head under bf16
    autocast overflowed to NaN at step 13.8k of the first stage-3 run and
    poisoned the weights for good."""
    return (sigma.float().square() + logvar.float().clamp(*LOGVAR_RANGE).exp()).clamp_min(1e-8)


def decode_mean(
    dec: QueryDecoder,
    z: torch.Tensor,
    tokens: Tokens,
    n_steps: int = 20,
    method: str = "euler",
    n_samples: int = 1,
    generator=None,
    grad: bool = False,
) -> torch.Tensor:
    """``mu_hat [B, N]`` at the query tokens: the flow's deterministic solve
    from ``eps = 0`` (``n_samples = 1``) or the mean of samples; the mse
    decoder's ``mu``. ``grad=True`` keeps the graph to ``z`` (energies)."""
    ls, pad = dec.log_sigma(tokens), tokens.pad_mask
    if dec.kind == "mse":
        mu, _ = dec(z, tokens.positions, pad, ls)
        return mu if grad else mu.detach()
    m = tokens.values[..., 0]
    v_fn = lambda x, s: dec(z, tokens.positions, pad, ls, x, s)
    integrate = sample_grad if grad else sample
    outs = []
    for i in range(n_samples):
        eps = (
            torch.zeros_like(m, dtype=torch.float32)
            if n_samples == 1
            else torch.randn(m.shape, device=m.device, generator=generator)
        )
        outs.append(integrate(v_fn, eps, n_steps, method))
    return torch.stack(outs).mean(0)


@torch.no_grad()
def decode_samples(
    dec, z, tokens, n_samples=8, n_steps=20, method="euler", generator=None
) -> torch.Tensor:
    """``[S, B, N]`` samples (the mse decoder samples ``N(mu, exp(logvar))``)."""
    ls, pad = dec.log_sigma(tokens), tokens.pad_mask
    m = tokens.values[..., 0]
    if dec.kind == "mse":
        mu, logvar = dec(z, tokens.positions, pad, ls)
        noise = torch.randn(
            (n_samples,) + mu.shape, device=mu.device, generator=generator
        )
        return mu[None] + noise * (0.5 * logvar.float().clamp(*LOGVAR_RANGE)).exp()[None]
    v_fn = lambda x, s: dec(z, tokens.positions, pad, ls, x, s)
    return torch.stack(
        [
            sample(
                v_fn,
                torch.randn(m.shape, device=m.device, generator=generator),
                n_steps,
                method,
            )
            for _ in range(n_samples)
        ]
    )


def drop_tokens(
    tokens: Tokens, frac: float, generator=None, keep_min: int = 1
) -> Tokens:
    """Mark a random fraction of the real tokens of every row as padding
    (the encoder then does not see them); ``keep_min`` real tokens stay."""
    if frac <= 0:
        return tokens
    pad = tokens.pad_mask
    real = ~pad
    u = torch.rand(pad.shape, device=pad.device, generator=generator)
    u[pad] = 2.0
    n_real = real.sum(1)
    n_drop = torch.minimum(
        (n_real.float() * frac).floor().long(), (n_real - keep_min).clamp(min=0)
    )
    rank = u.argsort(1).argsort(1)
    drop = rank < n_drop[:, None]
    return Tokens(tokens.values, tokens.positions, pad | drop, tokens.extras)


def hide_tokens(tokens: Tokens, hide: torch.Tensor) -> Tokens:
    """Tokens with ``hide [B, N]`` (True = hidden) added to the padding."""
    return Tokens(
        tokens.values, tokens.positions, tokens.pad_mask | hide, tokens.extras
    )


def decoder_state(dec: QueryDecoder, meta: dict, step: int, metrics=None) -> dict:
    return dict(
        state_dict=dec.state_dict(),
        hparams=dec.hparams,
        meta=meta,
        step=step,
        metrics=metrics,
    )


def load_decoder(path, device="cpu") -> tuple[QueryDecoder, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    dec = QueryDecoder(**ckpt["hparams"])
    dec.load_state_dict(ckpt["state_dict"])
    meta = dict(
        ckpt.get("meta", {}), step=ckpt.get("step"), metrics=ckpt.get("metrics")
    )
    return dec.to(device).eval(), meta
