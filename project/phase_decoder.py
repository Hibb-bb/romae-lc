"""The phase decoder: brightness as a function of phase, with the period
given, so the latent only has to supply shape, amplitude and level.

The window latent holds the period roughly and the phase not at all
(``phase_probe``, 2026-09-30), and no learned decoder found the fold across
hundreds of cycles. So this decoder is handed the period and works in
phase: every query and every context point is placed at ``phi = ((t -
t_ref) / P) mod 1`` and described by sines and cosines of ``phi`` at a few
harmonics plus its band, and the attention rotates queries and keys by
the phase (a rotary block whose rungs are the harmonics of one cycle), so
an attention score carries ``cos(2 pi k (phi_q - phi_c))``: a query can
attend to the context points at its own phase from the first step. The
first version had no rotary block and learned to ignore the context
(skill 0.08 with the context shuffled or removed); the coordinate is the
phase itself. The latent of the target window enters as
a token (or the "no latent" token for a share of the training samples,
``--latent-drop``, which gives the past-only forecast for free).

The ladder of tests, from most to least help (``--phase``, ``--n-ctx``,
the period source at evaluation):

- the ceiling: no context, oracle period AND oracle phase (``t_ref`` = the
  epoch of maximum light of the star's template, catalogue period). The
  decoder must draw the window from the latent alone. If this fails, the
  latent holds no shape or the target is at fault, not the phase;
- with context (the last ``--n-ctx`` windows, folded on the same period,
  phase relative to the target's start): the past supplies the template and
  the offset, the latent the adjustment;
- the period at evaluation: ``oracle`` (catalogue), ``refined`` (the
  model's rough period sharpened by the fine search on the past only),
  ``ls`` (Lomb-Scargle on the past), ``model`` (the rough read-out alone).

Training always uses the catalogue period as the coordinate (a label in
the coordinate, never in the input). ``--target obs`` scores the observed
magnitudes under their errors; ``--target template`` regresses on the
clean curve (the error-weighted Fourier fit of the star's fold on the
catalogue period, evaluated at the query times), the denoised label.
Every evaluation reports both: NLL and RMSE against the observations, and
RMSE against the template.

    python -m project.phase_decoder --ckpt project/runs/mae_w250/mae.pt --out project/runs/mae_w250/phase_ceiling --n-ctx 0 --phase oracle
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from romae_lc.model import _init_weights
from romae_lc.rope import BlockRope
from romae_lc.transformer import Transformer, attention_mask, config

from project import fold
from project.common import (
    JsonlLog,
    add_device_arg,
    cosine_schedule,
    data_args_from,
    dump_json,
    get_device,
    load_data,
    load_encoder,
    n_params,
    save_atomic,
    seed_all,
    subset,
)
from project.decoder import gaussian_var
from project.inject import band_templates
from project.lookback import ForecastDataset, make_collate
from project.period_probe import ls_periods
from project.tracking import StepTimer, Tracker, add_wandb_args, gpu_stats
from project.train_predictor import load_predictor

KIND = "phase"
PERIOD_SOURCES = ("oracle", "refined", "ls", "ls_refined", "model", "cands")


# -------------------------------------------------------------------- model


def phase_features(phi: torch.Tensor, n_harm: int) -> torch.Tensor:
    """``[..., 2 n_harm]``: sin and cos of ``2 pi k phi`` for ``k = 1..n_harm``."""
    k = torch.arange(1, n_harm + 1, device=phi.device, dtype=torch.float32)
    ang = 2.0 * np.pi * phi.float()[..., None] * k
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


class PhaseDecoder(nn.Module):
    """``(z, query phases and bands, context) -> (mu, logvar)`` per query, a
    transformer without positional encoding over ``[latent token, context
    tokens, query tokens]`` where every token carries its phase features
    and band embedding; context tokens also carry the frozen encoder's
    token features and the value channels."""

    def __init__(self, z_dim, enc_dim, d_model=192, nhead=3, depth=3, n_harm=8, n_bands=8,
                 latent_drop=0.3, val_channels=2, ctx_features="values"):  # fmt: skip
        super().__init__()
        if ctx_features not in ("values", "encoder"):
            raise ValueError("ctx_features must be values|encoder")
        self.z_dim, self.enc_dim, self.n_harm, self.n_bands = int(z_dim), int(enc_dim), int(n_harm), int(n_bands)
        self.latent_drop, self.val_channels, self.ctx_features = float(latent_drop), int(val_channels), ctx_features
        self.cfg = config(dict(d_model=d_model, nhead=nhead, depth=depth))
        head_dim = self.cfg.head_dim
        if 2 * n_harm > head_dim:
            raise ValueError(f"2 * n_harm = {2 * n_harm} rotary dims do not fit a head of {head_dim}")
        # rotary block over the phase: rung k rotates by 2 pi k phi, so q.k holds cos(2 pi k dphi)
        self.rope = BlockRope(head_dim, nhead, [dict(kind="axial", axes=[0], dim=head_dim, base=10000.0,
                                                     p=2 * n_harm / head_dim, timescales=[1.0 / (2 * np.pi * k) for k in range(1, n_harm + 1)])])  # fmt: skip
        self.z_proj = nn.Sequential(nn.Linear(z_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.no_z = nn.Parameter(torch.zeros(1, d_model))
        self.q_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.phase_proj = nn.Linear(2 * n_harm, d_model, bias=False)
        self.band_emb = nn.Embedding(n_bands, d_model)
        self.ctx_proj = nn.Linear(self.ctx_dim, d_model)
        self.type_emb = nn.Embedding(3, d_model)
        self.transformer = Transformer(self.cfg)
        self.norm = nn.RMSNorm(d_model, eps=self.cfg.norm_eps)
        self.head = nn.Linear(d_model, 2)
        self.apply(_init_weights)
        nn.init.normal_(self.no_z, std=0.02)
        nn.init.normal_(self.q_token, std=0.02)

    @property
    def kind(self) -> str:
        return KIND

    @property
    def ctx_dim(self) -> int:
        return self.val_channels + (self.enc_dim if self.ctx_features == "encoder" else 0)

    @property
    def hparams(self) -> dict:
        return dict(z_dim=self.z_dim, enc_dim=self.enc_dim, d_model=self.cfg.d_model, nhead=self.cfg.nhead,
                    depth=self.cfg.depth, n_harm=self.n_harm, n_bands=self.n_bands, latent_drop=self.latent_drop,
                    val_channels=self.val_channels, ctx_features=self.ctx_features)  # fmt: skip

    def forward(self, z, q_phase, q_band, q_pad, ctx=None, use_z=None):
        """``z [B, Dz]``; queries: ``q_phase [B, N]`` in [0, 1), ``q_band [B,
        N]`` int, ``q_pad [B, N]``; ``ctx`` = ``(feats [B, M, enc_dim +
        val_channels], phase [B, M], band [B, M], pad [B, M])`` or None;
        ``use_z`` bool ``[B]`` or None. Returns ``(mu, logvar)`` ``[B, N]``."""
        b, n = q_pad.shape
        dt = self.q_token.dtype
        c = self.z_proj(z.to(dt))
        if use_z is not None:
            c = torch.where(use_z[:, None], c, self.no_z.to(dt).expand_as(c))
        parts = [(c + self.type_emb.weight[0])[:, None]]
        pads = [q_pad.new_zeros(b, 1)]
        phs = [q_phase.new_zeros(b, 1)]
        if ctx is not None and ctx[0].shape[1] > 0:
            c_feats, c_phase, c_band, c_pad = ctx
            x_c = (self.ctx_proj(c_feats.to(dt)) + self.phase_proj(phase_features(c_phase, self.n_harm).to(dt))
                   + self.band_emb(c_band.clamp(0, self.n_bands - 1)) + self.type_emb.weight[1])  # fmt: skip
            parts.append(x_c)
            pads.append(c_pad)
            phs.append(c_phase)
        x_q = (self.q_token.expand(b, n, -1) + self.phase_proj(phase_features(q_phase, self.n_harm).to(dt))
               + self.band_emb(q_band.clamp(0, self.n_bands - 1)) + self.type_emb.weight[2])  # fmt: skip
        parts.append(x_q)
        pads.append(q_pad)
        phs.append(q_phase)
        rot = self.rope.prepare(torch.cat(phs, 1).float()[:, None, :])
        x = self.transformer(torch.cat(parts, 1), rot, attention_mask(torch.cat(pads, 1)))
        out = self.head(self.norm(x[:, -n:])).float()
        return out[..., 0], out[..., 1]


# --------------------------------------------------------------------- data


def fourier_fit(record, period: float, harmonics: int = 3, err_floor: float = 1e-3):
    """Per band, the error-weighted Fourier fit of the record folded on
    ``period``: ``{band: beta}``; bands with too few points get the median
    as a constant."""
    t, y = record.t.astype(np.float64), record.y.astype(np.float64)
    err, band = record.err.astype(np.float64), record.band
    out = {}
    for b in np.unique(band):
        m = band == b
        n_par = 2 * harmonics + 1
        if m.sum() < n_par + 3:
            out[int(b)] = np.concatenate([[np.median(y[m])], np.zeros(n_par - 1)])
            continue
        w = 1.0 / (np.maximum(err[m], 0.0) ** 2 + err_floor**2)
        sw = np.sqrt(w)
        x = fold.design(t[m], period, harmonics)
        beta, *_ = np.linalg.lstsq(x * sw[:, None], y[m] * sw, rcond=None)
        out[int(b)] = beta
    return out


def template_at(coefs: dict, t, band, period: float, harmonics: int = 3):
    """The fitted curve at absolute times ``t`` (record time) and bands."""
    t, band = np.asarray(t, dtype=np.float64), np.asarray(band)
    y = np.zeros(len(t))
    for b in np.unique(band):
        m = band == b
        beta = coefs.get(int(b))
        if beta is None:
            beta = np.concatenate([[0.0], np.zeros(2 * harmonics)])
        y[m] = fold.design(t[m], period, harmonics) @ beta
    return y


def reference_epoch(record, period: float) -> float:
    """The epoch (record time, days) of maximum light of the star's folded
    template: the oracle phase reference. 0 without a template."""
    templates, _ = band_templates(record, period)
    if not templates:
        return 0.0
    b = max(templates, key=lambda k: int((record.band == k).sum()))
    edges, vals, _ = templates[b]
    centres = 0.5 * (edges[1:] + edges[:-1])
    return float(centres[int(np.nanargmin(vals))] * period)


class PhaseDataset(ForecastDataset):
    """:class:`ForecastDataset` plus, per sample, the catalogue period, the
    oracle phase reference and the clean template at the target's times
    (both from the catalogue period). Records without a period give None."""

    def __init__(self, *args, harmonics: int = 3, **kw):
        super().__init__(*args, **kw)
        self.harmonics = int(harmonics)
        self._fit = {}

    def star(self, i):
        if i not in self._fit:
            r = self.records[i]
            p = float(r.period or 0.0)
            self._fit[i] = (p, reference_epoch(r, p), fourier_fit(r, p, self.harmonics)) if p > 0 else None
        return self._fit[i]

    def __getitem__(self, i):
        item = super().__getitem__(i)
        star = self.star(i)
        if item is None or star is None:
            return None
        p, t_ref, coefs = star
        t_rel, _, band, _ = item["target"]
        item["period"], item["t_ref"] = p, t_ref
        item["template"] = template_at(coefs, item["start"] + np.asarray(t_rel, dtype=np.float64), band, p, self.harmonics).astype(np.float32)
        return item


def make_phase_collate(spec, n_ctx: int):
    base = make_collate(spec, n_ctx)

    def collate(items):
        items = [x for x in items if x is not None]
        out = base(items)
        if out is None:
            return None
        out["t_ref"] = torch.tensor([x["t_ref"] for x in items], dtype=torch.float64)
        tmpl = [(x["target"][0], x["template"], x["target"][2], x["target"][3]) for x in items]
        out["template"] = spec.tokens(tmpl).values[..., 0].float()  # same token order as the target
        return out

    return collate


_BAND_TABLES: dict = {}


def band_table(spec) -> torch.Tensor:
    """The wavelength-axis value of every band id of the tokenizer, so a
    token's band id can be read back from ``positions[:, 1]``."""
    key = id(spec)
    if key not in _BAND_TABLES:
        n = len(spec.tokenize["band_wavelengths"])
        tok = spec.tokens([(np.arange(n, dtype=np.float32), np.zeros(n, np.float32), np.arange(n), np.ones(n, np.float32))])
        _BAND_TABLES[key] = tok.positions[0, 1].clone()
    return _BAND_TABLES[key]


def band_ids(pos1: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``[B, N]`` band ids: the nearest table entry to each wavelength value."""
    return (pos1[..., None] - table.to(pos1.device)[None, None]).abs().argmin(-1)


def phases(positions, time_scale, start, extra, period, t_ref):
    """``[B, N]`` phases of tokens whose time axis is ``positions[:, 0]``
    (window time, origin 1.0) in a window at ``start + extra`` (days),
    folded on ``period`` from ``t_ref``; all per-sample tensors ``[B]``."""
    t_abs = start[:, None] + extra[:, None] + (positions[:, 0].double() - 1.0) * time_scale
    return torch.remainder((t_abs - t_ref[:, None]) / period[:, None], 1.0).float()


@torch.no_grad()
def predicted_latent(enc, pred, batch, device, n_samples: int = 8):
    """The predictor's draw for the target window: the context windows'
    pooled latents (oldest first, one window length apart) run through the
    sequence predictor, then the mean of ``n_samples`` draws one horizon
    ahead. ``None`` for samples without a context window."""
    ctx_tok = batch["ctx"]
    if not ctx_tok:
        return None
    real = batch["slot_real"].to(device)  # [B, K], slot 0 = the window just before the gap
    zs = torch.stack([enc.encode([t.to(device)])[:, 0].float() for t in ctx_tok], 1)  # [B, K, D], newest first
    zs, real = zs.flip(1), real.flip(1)  # oldest first
    b, k = real.shape
    # left-align the real windows so the sequence ends at the newest one
    order = torch.argsort((~real).int(), dim=1, stable=True)
    zs = torch.gather(zs, 1, order[..., None].expand_as(zs))
    real = torch.gather(real, 1, order)
    gaps = torch.ones(b, k, device=device)
    gaps[:, 0] = 0.0
    h = pred.states(zs, gaps, real)
    last = (real.sum(1) - 1).clamp_min(0)
    idx = torch.arange(b, device=device)
    h_t, z_t = h[idx, last], zs[idx, last]
    gap = batch["horizon"].to(device).float()
    z_hat = pred.predict_mean(h_t, gap, z_t, n_samples=n_samples)
    z_hat[real.sum(1) == 0] = float("nan")
    return z_hat


@torch.no_grad()
def batch_inputs(enc, spec, batch, device, period, phase_mode: str, ctx_features: str = "values"):
    """Everything the decoder needs from a collated batch: the target
    tokens, its true latent, query phases and bands, the context tuple (or
    None), the clean template and the period used. ``period [B]`` (days,
    NaN = no period for that sample) is the coordinate's period;
    ``phase_mode`` ``oracle`` folds from the star's reference epoch,
    ``window`` from the target window's start."""
    ts = spec.tokenize["time_scale"]
    tgt = batch["target"].to(device)
    start = batch["start"].to(device)
    period = period.to(device).double()
    t_ref = batch["t_ref"].to(device) if phase_mode == "oracle" else start
    zero = torch.zeros_like(start)
    table = band_table(spec)
    q_phase = phases(tgt.positions, ts, start, zero, period, t_ref)
    q_band = band_ids(tgt.positions[:, 1], table)
    z = enc.encode([tgt])[:, 0].float()
    ctx = None
    if batch["ctx"]:
        feats, phs, bands, pads = [], [], [], []
        offsets, real = batch["offsets"].to(device), batch["slot_real"].to(device)
        for i, tok in enumerate(batch["ctx"]):
            tok = tok.to(device)
            pad = tok.pad_mask | ~real[:, i][:, None]
            if ctx_features == "encoder":
                x, _ = enc.model.encode(tok.values, tok.positions, tok.pad_mask)
                feats.append(torch.cat([x[:, 1:].float(), tok.values.float()], -1))
            else:
                feats.append(tok.values.float())
            phs.append(phases(tok.positions, ts, start, offsets[:, i].double(), period, t_ref))
            bands.append(band_ids(tok.positions[:, 1], table))
            pads.append(pad)
        ctx = (torch.cat(feats, 1), torch.cat(phs, 1), torch.cat(bands, 1), torch.cat(pads, 1))
    return tgt, z, q_phase, q_band, ctx, batch["template"].to(device)


def phase_loss(dec, z, tgt, q_phase, q_band, ctx, template, target: str, use_z=None):
    m, pad = tgt.values[..., 0].float(), tgt.pad_mask
    mu, logvar = dec(z, q_phase, q_band, pad, ctx, use_z)
    if target == "template":
        var = torch.exp(logvar.float().clamp(-14.0, 10.0))
        nll = 0.5 * ((template - mu.float()).square() / var + var.log())
    else:
        var = gaussian_var(tgt.extras, logvar)
        nll = 0.5 * ((m - mu.float()).square() / var + var.log())
    mask = ~pad
    return (nll * mask).sum() / mask.sum().clamp(min=1)


# ------------------------------------------------------------- eval periods


def eval_periods(ds: PhaseDataset, source: str, model_periods: dict | None, p_min: float, device, args, log=print) -> dict:
    """``{record index: period}`` for the evaluation draws of ``ds`` from
    ``source``: the catalogue, Lomb-Scargle on the past (before the
    target's start), the model's rough period, or that rough period
    sharpened by the fine search on the past."""
    out = {}
    t0 = time.time()
    for i in range(len(ds)):
        item = ds[i]
        if item is None:
            continue
        r = ds.records[i]
        if source == "oracle":
            out[i] = item["period"]
            continue
        rough = (model_periods or {}).get(i, float("nan"))
        if source == "model":
            out[i] = rough
            continue
        past = r.t.astype(np.float64) < item["start"]
        if past.sum() < 20:
            out[i] = float("nan")
            continue
        t, y, e, b = r.t[past].astype(np.float64), r.y[past].astype(np.float64), r.err[past].astype(np.float64), r.band[past]
        if source == "cands":
            # candidates: the model's top bins, the Lomb-Scargle peak and its double and half;
            # each sharpened by the fine search on the past; the fold of the past picks the winner
            ls = float(ls_periods(t, y, e, b, p_min, args.ls_oversample, args.ls_cap, 1, device)["best"])
            cands = [c for c in list((model_periods or {}).get(("top", i), [])) + [ls, 2 * ls, 0.5 * ls] if np.isfinite(c) and c > 0]
            best_p, best_r2 = float("nan"), -np.inf
            for c in cands:
                p_c, r2_c, _ = fold.refine_period(t, y, e, b, c, args.refine_rel, args.refine_oversample, device=device)
                if np.isfinite(r2_c) and r2_c > best_r2:
                    best_p, best_r2 = float(p_c), float(r2_c)
            out[i] = best_p
        elif source in ("ls", "ls_refined"):
            best = float(ls_periods(t, y, e, b, p_min, args.ls_oversample, args.ls_cap, 1, device)["best"])
            if source == "ls_refined" and np.isfinite(best) and best > 0:  # the fine search seeded by the Lomb-Scargle peak
                best = float(fold.refine_period(t, y, e, b, best, args.refine_rel, args.refine_oversample, device=device)[0])
            out[i] = best
        else:  # refined
            out[i] = float(fold.refine_period(t, y, e, b, rough, args.refine_rel, args.refine_oversample, device=device)[0]) if np.isfinite(rough) and rough > 0 else float("nan")
        if (i + 1) % 200 == 0:
            log(f"  periods ({source}): {i + 1} / {len(ds)} stars, {time.time() - t0:.0f}s")
    return out


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate(enc, dec, spec, loader, device, periods: dict, phase_mode: str, log=print, pred=None) -> dict:
    """Scores on the validation draws with the latent and without it: NLL
    per observed point (known error + the decoder's variance), RMSE of the
    mean against the observations and against the clean template, skill
    against the constant (the context's median per band, or 0 = the star's
    normalised level without context); per horizon and superclass."""
    dec.eval()
    rows = []
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for batch in loader:
        if batch is None:
            continue
        period = torch.tensor([periods.get(int(i), float("nan")) for i in batch["index"]], dtype=torch.float64)
        ok = torch.isfinite(period) & (period > 0)
        if not ok.any():
            continue
        with amp:
            tgt, z, q_phase, q_band, ctx, template = batch_inputs(enc, spec, batch, device, period.clamp_min(1e-3), phase_mode, dec.ctx_features)
        m, pad, sig = tgt.values[..., 0].float(), tgt.pad_mask, tgt.extras.float()
        preds = {}
        for name, use in (("model", True), ("no_latent", False)):
            flag = torch.full((z.shape[0],), use, device=device)
            with amp:
                mu, logvar = dec(z, q_phase, q_band, pad, ctx, flag)
            preds[name] = (mu.float(), gaussian_var(sig, logvar))
        if pred is not None:  # the whole chain: the predictor's latent instead of the window's own
            z_hat = predicted_latent(enc, pred, batch, device)
            if z_hat is not None:
                good = torch.isfinite(z_hat).all(1)
                with amp:
                    mu, logvar = dec(torch.where(good[:, None], z_hat, z), q_phase, q_band, pad, ctx, torch.ones_like(good))
                mu = torch.where(good[:, None], mu.float(), torch.full_like(mu.float(), float("nan")))
                preds["predicted"] = (mu, gaussian_var(sig, logvar))
        for b in range(m.shape[0]):
            if not ok[b]:
                continue
            real = ~pad[b]
            if real.sum() == 0:
                continue
            y, s, tm = m[b, real], sig[b, real], template[b, real]
            const = torch.zeros_like(y)
            if ctx is not None:
                c_val, c_band, c_pad = ctx[0][b, :, -dec.val_channels], ctx[2][b], ctx[3][b]
                for bd in q_band[b, real].unique():
                    cm = (~c_pad) & (c_band == bd)
                    if cm.any():
                        const[q_band[b, real] == bd] = c_val[cm].median()
            mse_c = ((y - const) ** 2).mean().clamp_min(1e-12)
            row = dict(horizon=int(batch["horizon"][b]), superclass=batch["superclass"][b], n=int(real.sum()),
                       rmse_const=float(mse_c.sqrt()), rmse_template_const=float(((tm - const) ** 2).mean().sqrt()))  # fmt: skip
            for name, (mu, var) in preds.items():
                mu_b, var_b = mu[b, real], var[b, real]
                if not torch.isfinite(mu_b).all():
                    row[f"nll_{name}"] = row[f"rmse_{name}"] = row[f"skill_{name}"] = row[f"rmse_template_{name}"] = float("nan")
                    continue
                nll = (0.5 * ((y - mu_b) ** 2 / var_b + var_b.log() + np.log(2 * np.pi))).mean()
                mse = ((y - mu_b) ** 2).mean()
                row[f"nll_{name}"], row[f"rmse_{name}"], row[f"skill_{name}"] = float(nll), float(mse.sqrt()), float(1 - mse / mse_c)
                row[f"rmse_template_{name}"] = float(((tm - mu_b) ** 2).mean().sqrt())
            rows.append(row)
    dec.train()
    if not rows:
        return dict(n=0)
    keys = sorted({k for r in rows for k in r if k.startswith(("nll_", "rmse_", "skill_"))})

    def med(key, sel):
        v = [r[key] for r in sel if key in r and np.isfinite(r[key])]
        return float(np.median(v)) if v else float("nan")

    def block(sel):
        return dict(n=len(sel), **{k: med(k, sel) for k in keys},
                    beats_no_latent=float(np.mean([r["nll_model"] < r["nll_no_latent"] for r in sel])))  # fmt: skip

    res = block(rows)
    res["by_horizon"] = {str(h): block([r for r in rows if r["horizon"] == h]) for h in sorted({r["horizon"] for r in rows})}
    groups = sorted({r["superclass"] for r in rows}, key=lambda g: -sum(r["superclass"] == g for r in rows))
    res["by_superclass"] = {g: block([r for r in rows if r["superclass"] == g]) for g in groups if sum(r["superclass"] == g for r in rows) >= 10}
    log(f"  phase forecast ({res['n']} draws): NLL model {res['nll_model']:.3f} | no latent {res['nll_no_latent']:.3f}; "
        f"RMSE obs {res['rmse_model']:.3f} / {res['rmse_no_latent']:.3f} / const {res['rmse_const']:.3f}; "
        f"RMSE vs template {res['rmse_template_model']:.3f} / {res['rmse_template_no_latent']:.3f} / const {res['rmse_template_const']:.3f}; "
        f"skill {res['skill_model']:.3f} / {res['skill_no_latent']:.3f}"
        + (f" / predicted latent {res['skill_predicted']:.3f}" if "skill_predicted" in res else "")
        + f"; beats no-latent {res['beats_no_latent']:.0%} | "
        + " ".join(f"{g} {v['skill_model']:.2f}/{v['skill_no_latent']:.2f} ({v['n']})" for g, v in res["by_superclass"].items()))
    return res


def write_tables(path, final: dict, args):
    lines = [f"# Phase decoder: {args.out}\n",
             f"n_ctx {args.n_ctx}, phase {args.phase}, target {args.target}; training coordinate = catalogue period. "
             "Scores on validation draws (medians): NLL per observed point, RMSE against the observations, RMSE against "
             "the clean template, skill = 1 - mse / mse of the constant. 'no latent' = the same decoder with the "
             "'no latent' token (past only, or nothing but the phase coordinate when n_ctx = 0).\n"]  # fmt: skip
    cols = [("source", "period at evaluation"), ("n", "n"), ("nll_model", "NLL"), ("nll_no_latent", "NLL no latent"),
            ("rmse_model", "RMSE"), ("rmse_no_latent", "RMSE no latent"), ("rmse_const", "RMSE const"),
            ("rmse_template_model", "RMSE vs template"), ("rmse_template_no_latent", "vs template, no latent"),
            ("skill_model", "skill"), ("skill_no_latent", "skill no latent")]  # fmt: skip
    if any("skill_predicted" in v for v in final.values()):
        cols += [("skill_predicted", "skill, predicted latent"), ("rmse_predicted", "RMSE, predicted latent")]
    rows = [dict(source=src, **{k: v.get(k, float("nan")) for k, _ in cols[1:]}) for src, v in final.items() if v.get("n")]
    lines.append("| " + " | ".join(c for _, c in cols) + " |\n|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        lines.append("| " + " | ".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k, _ in cols) + " |")
    for src, v in final.items():
        if not v.get("n"):
            continue
        lines.append(f"\n## {src}: by superclass (skill with / without the latent, RMSE vs template)\n")
        lines.append("| class | n | skill | skill no latent | RMSE vs template | const |\n|---|---|---|---|---|---|")
        for g, b in v["by_superclass"].items():
            lines.append(f"| {g} | {b['n']} | {b['skill_model']:.3f} | {b['skill_no_latent']:.3f} | {b['rmse_template_model']:.3f} | {b['rmse_template_const']:.3f} |")
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------- checkpoint


def load_phase_decoder(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("kind") != KIND:
        raise ValueError(f"{path} is not a phase decoder checkpoint")
    dec = PhaseDecoder(**ckpt["hparams"])
    dec.load_state_dict(ckpt["state_dict"])
    return dec.to(device).eval(), dict(ckpt.get("meta", {}), step=ckpt.get("step"), metrics=ckpt.get("metrics"))


# ------------------------------------------------------------------ training

RUN_CONTROL = ("steps", "time_budget", "workers", "device", "eval_every", "ckpt_every", "log_every", "persistent_workers",
               "prefetch", "pin_memory", "no_resume", "wandb", "wandb_name", "wandb_project", "wandb_entity", "wandb_group",
               "wandb_tags", "val_objects", "eval_periods", "model_periods", "ls_cap", "ls_oversample", "refine_rel",
               "refine_oversample", "eval_only", "pred")  # fmt: skip


def add_args(p):
    p.add_argument("--ckpt", required=True, help="stage-1 checkpoint (the frozen encoder)")
    p.add_argument("--out", required=True)
    p.add_argument("--n-ctx", type=int, default=3, help="context windows; 0 = the latent alone")
    p.add_argument("--phase", choices=("oracle", "window"), default="window",
                   help="phase reference: the star's epoch of maximum light (oracle) or the target window's start")  # fmt: skip
    p.add_argument("--target", choices=("obs", "template"), default="obs")
    p.add_argument("--harmonics", type=int, default=3, help="of the clean template")
    p.add_argument("--n-harm", type=int, default=8, help="phase harmonics (features and rotary rungs)")
    p.add_argument("--ctx-features", choices=("values", "encoder"), default="values",
                   help="context tokens carry the point's value channels only, or also the frozen encoder's token features")  # fmt: skip
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2])
    p.add_argument("--stride", type=float, default=0.25)
    p.add_argument("--latent-drop", type=float, default=0.3)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.02)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--time-budget", type=float, default=0.0)
    p.add_argument("--val-objects", type=int, default=1000)
    p.add_argument("--train-objects", type=int, default=None)
    p.add_argument("--eval-periods", nargs="*", default=["oracle"], choices=PERIOD_SOURCES,
                   help="period sources of the final evaluation")  # fmt: skip
    p.add_argument("--eval-only", action="store_true", help="no training: load <out>/dec.pt and run the period ladder")
    p.add_argument("--pred", default=None, help="a sequence predictor (pred.pt) on this encoder: adds the predicted-latent forecast to the final evaluation")
    p.add_argument("--model-periods", default=None, help="predictions.npz of period_probe (the model's rough periods, by record index)")
    p.add_argument("--ls-oversample", type=float, default=5.0)
    p.add_argument("--ls-cap", type=int, default=100_000)
    p.add_argument("--refine-rel", type=float, default=0.1)
    p.add_argument("--refine-oversample", type=float, default=5.0)
    p.add_argument("--data", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--n-sim", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    add_device_arg(p)
    add_wandb_args(p)


def load_model_periods(path) -> dict:
    """``{record index: rough period}`` plus, when the probe saved its top
    bins, ``{("top", index): [candidate periods]}``."""
    d = np.load(path, allow_pickle=True)
    out = {int(i): float(p) for i, p in zip(d["index"], d["p_model"])}
    if "p_top" in d.files:
        out.update({("top", int(i)): [float(v) for v in row] for i, row in zip(d["index"], d["p_top"])})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    dump_json(vars(args), out / "args.json")
    seed_all(args.seed)
    dev = get_device(args)
    t_start = time.time()
    enc, meta = load_encoder(args.ckpt, dev)
    cfg, spec = meta.cfg, meta.spec
    over = argparse.Namespace(data=args.data, max_rows=args.max_rows, n_sim=args.n_sim)
    data = load_data(data_args_from(meta.args, over))
    train = subset(data["train"], args.train_objects, args.seed)
    val = subset(data["validation"], args.val_objects, args.seed)
    p_min = float(min(r.period for r in data["train"] if r.period and r.period > 0))
    collate = make_phase_collate(spec, args.n_ctx)
    kw = dict(persistent_workers=args.persistent_workers and args.workers > 0,
              prefetch_factor=args.prefetch if args.workers > 0 else None, pin_memory=args.pin_memory)  # fmt: skip
    train_ds = PhaseDataset(train, cfg, args.stride, cfg.max_tokens, args.n_ctx, args.horizons, args.seed, epoch_seed=True, harmonics=args.harmonics)
    val_ds = PhaseDataset(val, cfg, args.stride, cfg.max_tokens, args.n_ctx, args.horizons, args.seed, epoch_seed=False, harmonics=args.harmonics)
    loader = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True, collate_fn=collate, num_workers=args.workers, **kw)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate, num_workers=min(args.workers, 4),
                            **dict(kw, persistent_workers=False))  # fmt: skip
    print(f"{len(train)} train / {len(val)} val records; stage-1 step {meta.step}; window {cfg.window:g}; n_ctx {args.n_ctx}, "
          f"phase {args.phase}, target {args.target}, horizons {args.horizons}; loaded in {time.time() - t_start:.0f}s", flush=True)  # fmt: skip
    n_bands = len(spec.tokenize["band_wavelengths"])
    dec = PhaseDecoder(enc.dim, enc.backbone.embed_dim, args.width, args.heads, args.depth, args.n_harm, n_bands, args.latent_drop,
                       ctx_features=args.ctx_features).to(dev)  # fmt: skip
    print(f"phase decoder: {n_params(dec) / 1e6:.2f}M params; {len(loader)} steps per epoch; {dev}")
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = LambdaLR(opt, cosine_schedule(args.steps, args.warmup))
    step, epoch, elapsed, last_metrics = 0, 0, 0.0, None
    best_nll, n_skipped = float("inf"), 0
    if ckpt is not None:
        dec.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        step, epoch, elapsed = ckpt["step"], ckpt["epoch"], ckpt["elapsed"]
        last_metrics = ckpt.get("metrics")
        best_nll = float(ckpt.get("best_nll", float("inf")))
        n_skipped = int(ckpt.get("skipped", 0))
    tracker = Tracker(args, config=dict(vars(args), params_decoder=n_params(dec), wm_step=meta.step, window=cfg.window),
                      run_id=ckpt.get("wandb_id") if ckpt else None, job_type="phase-decoder")  # fmt: skip
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    log = JsonlLog(out / "log.jsonl")
    dmeta = dict(ckpt=str(args.ckpt), wm_step=meta.step, kind=KIND, frames=asdict(cfg), spec=spec.to_dict(), stride=args.stride,
                 n_ctx=args.n_ctx, phase=args.phase, target=args.target, horizons=list(args.horizons), harmonics=args.harmonics,
                 ctx_features=args.ctx_features)  # fmt: skip
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1)
    oracle = {i: (val_ds.star(i) or (float("nan"),))[0] for i in range(len(val_ds))}

    def save(path, metrics, with_opt=True):
        state = dict(kind=KIND, state_dict=dec.state_dict(), hparams=dec.hparams, meta=dmeta, step=step, metrics=metrics,
                     args=dict(vars(args)), epoch=epoch, elapsed=elapsed, wandb_id=tracker.id, best_nll=best_nll, skipped=n_skipped)  # fmt: skip
        if with_opt:
            state.update(opt=opt.state_dict(), sched=sched.state_dict())
        save_atomic(state, path)

    def run_eval(train_loss):
        t0 = time.time()
        m = evaluate(enc, dec, spec, val_loader, dev, oracle, args.phase)
        m.update(step=step, elapsed=elapsed, train_loss=train_loss, eval_seconds=time.time() - t0)
        log.write(kind="eval", **m)
        flat = {"train_mean/loss": train_loss}
        for k in ("nll_model", "nll_no_latent", "rmse_model", "rmse_no_latent", "rmse_const", "rmse_template_model",
                  "rmse_template_no_latent", "skill_model", "skill_no_latent", "beats_no_latent"):  # fmt: skip
            if k in m:
                flat[f"val/{k}"] = m[k]
        tracker.log(flat, step=step)
        return m

    def final_eval():
        model_periods = load_model_periods(args.model_periods) if args.model_periods else None
        predictor = None
        if args.pred:
            predictor, _ = load_predictor(args.pred, dev)
            if getattr(predictor, "arch", "mlp") != "seq":
                raise SystemExit("--pred needs a sequence predictor (--arch seq)")
        final = {}
        for src in args.eval_periods:
            if src in ("model", "refined", "cands") and model_periods is None:
                print(f"skipping {src}: no --model-periods")
                continue
            print(f"evaluating with the {src} period", flush=True)
            periods = eval_periods(val_ds, src, model_periods, p_min, dev, args)
            n_ok = sum(1 for v in periods.values() if np.isfinite(v) and v > 0)
            final[src] = evaluate(enc, dec, spec, val_loader, dev, periods, args.phase, pred=predictor)
            final[src]["n_periods"] = n_ok
        dump_json(dict(final=final, args=vars(args)), out / "final.json")
        write_tables(out / "tables.md", final, args)
        return final

    if args.eval_only:
        state = torch.load(out / "dec.pt", map_location="cpu", weights_only=False)
        dec.load_state_dict(state["state_dict"])
        dec.eval()
        final_eval()
        return

    dec.train()
    timer = StepTimer(dev)
    run, n_acc, t_last, t_run, stop = 0.0, 0, time.time(), time.time(), False
    while step < args.steps and not stop:
        torch.manual_seed(args.seed + epoch)
        train_ds.set_epoch(epoch)
        timer.reset()
        for batch in loader:
            timer.got_batch()
            if batch is None:
                continue
            with torch.no_grad(), amp:
                tgt, z, q_phase, q_band, ctx, template = batch_inputs(enc, spec, batch, dev, batch["period"], args.phase, args.ctx_features)
            use_z = torch.rand(z.shape[0], device=dev, generator=gen) >= args.latent_drop
            with amp:
                loss = phase_loss(dec, z, tgt, q_phase, q_band, ctx, template, args.target, use_z)
            opt.zero_grad(set_to_none=True)
            finite = bool(torch.isfinite(loss))
            if finite:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(dec.parameters(), args.clip)
                finite = bool(torch.isfinite(gn))
            if finite:
                opt.step()
            else:
                opt.zero_grad(set_to_none=True)
                n_skipped += 1
                if n_skipped in (1, 10) or n_skipped % 100 == 0:
                    print(f"non-finite loss or gradient at step {step}: update skipped ({n_skipped} so far)", flush=True)
                if n_skipped > 1000:
                    raise RuntimeError("more than 1000 non-finite updates: stopping")
            sched.step()
            timer.done_step()
            step += 1
            if finite:
                run += loss.item()
                n_acc += 1
            if step % args.log_every == 0:
                now = time.time()
                rate, t_last = (now - t_last) / args.log_every, now
                perf, gpu = timer.report(), gpu_stats(dev)
                timer.reset()
                lr = sched.get_last_lr()[0]
                print(f"step {step:6d}  loss {run / max(n_acc, 1):.4f}  lr {lr:.2e}  {rate:.3f} s/step"
                      f"  (data {perf['data_frac']:.0%}, gpu {gpu.get('gpu_util', float('nan')):.0f}%)", flush=True)  # fmt: skip
                log.write(kind="train", step=step, loss=run / max(n_acc, 1), lr=lr, skipped=n_skipped, s_per_step=rate, **perf, **gpu)
                tracker.log({"train/loss": run / max(n_acc, 1), "train/lr": lr, "perf/s_per_step": rate, **{f"sys/{k}": v for k, v in gpu.items()}}, step=step)
            if step % args.eval_every == 0 or step == args.steps:
                elapsed, t_run = elapsed + time.time() - t_run, time.time()
                last_metrics = run_eval(run / max(n_acc, 1))
                run, n_acc = 0.0, 0
                nll = last_metrics.get("nll_model")
                if nll is not None and nll == nll and nll < best_nll:
                    best_nll = float(nll)
                    save(out / "best.pt", last_metrics, with_opt=False)
                t_run = time.time()
                timer.reset()
            budget = bool(args.time_budget) and (time.time() - t_start) > args.time_budget
            if step % args.ckpt_every == 0 or step == args.steps or budget:
                elapsed, t_run = elapsed + time.time() - t_run, time.time()
                save(last, last_metrics)
            if budget and step < args.steps:
                print(f"time budget reached at step {step}; checkpoint saved", flush=True)
                stop = True
                break
            if step >= args.steps:
                break
        else:
            epoch += 1
    if step < args.steps:
        return
    save(out / "dec.pt", last_metrics, with_opt=False)
    (out / "DONE").write_text(f"{step} steps, {elapsed / 3600:.2f} h\n")
    print(f"done: {step} steps in {elapsed / 3600:.2f} h; saved {out / 'dec.pt'}")
    final_eval()  # the period ladder
    if hasattr(tracker, "finish"):
        tracker.finish()


if __name__ == "__main__":
    main()
