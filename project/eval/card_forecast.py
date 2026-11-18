"""Test card 4: can the masked JEPA predictor fill in hidden observations? Three modes.

Tail mode (the default, ``--tail``): every token in the last ``--tail`` days of a window is hidden; the
context encoder sees the rest. The predictor's placeholder tokens sit at the hidden times and bands, as in
training. Each hidden observation j is scored by its gap from the last visible token,
``dt = t_j - t_last`` (``--dt``, 0.5 to 10 days kept), with the z-score against its error,
``z = (m_hat - m_j) / sigma_j``.

Gap mode (``--gap-share P``): instead of the tail, hide a random share P of each window's real tokens
(filling in missing points inside the window, not a forecast). Every hidden real token is scored; dt is
then the distance to the nearest visible token, and the local density is the visible tokens within 15
days of the hidden one.

Phase-gap mode (``--phase-gap W``): hide every real token whose phase at the CATALOGUE period (evaluation
only) falls in a band W cycles wide at a random place in the cycle, the same band in every cycle of the
window. No same-phase point is left to copy: the predictor must know the shape of the curve. Scored as
gap mode; stars without a catalogue period are skipped.

Rows:
  recon head                 the predictor's own magnitude output (trained with --jepa-recon-weight)
  predicted latent [mlp]     an MLP decoder of the predicted latent (the projection), fitted on the TARGET
                             latents of the decoder stars' hidden tokens: is the predicted latent decodable?
  target latent [mlp]        the same decoder on the target encoder's own latent (it SAW m_j): the ceiling
  window mean                the mean magnitude of the visible tokens
  Fourier fit (f window)     4 harmonics at the context encoder's frequency estimate, fitted to the visible tokens
  Fourier fit (catalogue f)  the same at the catalogue frequency (a reference, evaluation only)

Also the latent error, mean squared (pred - target) over the target variance, and within 2 sigma by class.

Two protocols. Which stars fit the decoder and which are scored:
  the default        ``--n-stars 600`` stars of ``--split`` (validation), window stride 0.25; 70 % of the
                     stars fit the decoder, the other 30 % are scored. Used to compare runs.
  the test card      ``TEST_CARD`` below: every test star is scored, non-overlapping windows; the decoder is
                     fitted on 1,000 validation stars. ``project.eval.test_card`` uses it, as
                     ``--split test --fit-split validation --n-fit 1000 --n-stars 0 --stride 1.0``.

The masks of all three modes draw from torch's random state, and no script seeds it (a new process starts
from a random seed). So the numbers move a little from run to run. ``--seed`` only fixes which stars are used.

    python -m project.eval.card_forecast --ckpts project/runs/R/mae.pt --out project/results/jepa_forecast/R.json
"""
from __future__ import annotations

import argparse, time
from dataclasses import dataclass

import numpy as np
import torch

from project.checkpoint import load_encoder
from project.data import data_args_from, load_data
from project.eval.common_eval import ZTFXPC, eval_device, fourier_forecast, run_name, spectral_layers, window_dataset, window_loader, write_json, zstats
from project.masks import forecast_tail_mask
from romae_lc.model import spectral_frequency

#: The protocol of the test card (the old test_forecast.sh).
TEST_CARD = dict(split="test", fit_split="validation", n_fit=1000, n_stars=0, stride=1.0)
#: The three modes of the test card, by the name they have in the card.
CARD_MODES = {"tail10": dict(tail=10.0), "gap0.3": dict(gap_share=0.3), "pgap0.2": dict(phase_gap=0.2)}


@dataclass
class Mode:
    """What is hidden and what is scored."""

    tail: float = 10.0  # days hidden at the end of every window (tail mode)
    dt: tuple = (0.5, 10.0)  # tail mode: gaps from the last visible token that are scored, days
    gap_share: float | None = None  # gap mode: share of real tokens hidden at random
    phase_gap: float | None = None  # phase-gap mode: width of the hidden band, cycles
    min_context: int = 16  # windows with fewer visible tokens are not scored
    harmonics: int = 4  # of the Fourier baselines

    @property
    def pgap(self) -> bool:
        return self.phase_gap is not None

    @property
    def gap(self) -> bool:
        return self.gap_share is not None or self.pgap

    @property
    def dt_bins(self) -> list:
        if self.gap:
            return [(0.0, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, 60.0)]
        return [(lo, hi) for lo, hi in [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 60.0)] if lo < self.dt[1]]

    @property
    def local(self) -> str:
        return "visible tokens within 15 d" if self.gap else "context tokens in the last 30 d"

    @property
    def suffix(self) -> str:
        return f"_pgap{self.phase_gap:g}" if self.pgap else f"_gap{self.gap_share:g}" if self.gap else ""


def gap_mask(t_days, pad, share):
    """Hide a random share of each row's real tokens; rows are topped up to the same hidden count (pads first),
    as the JEPA forward needs."""
    real = ~pad
    r = torch.rand(pad.shape, device=pad.device)
    n_hide = (real.sum(1).float() * share).round().clamp(min=1)
    rank = torch.where(real, r, torch.full_like(r, 2.0)).argsort(1).argsort(1)
    sel = real & (rank < n_hide[:, None])
    k = min(int(sel.sum(1).max()), pad.shape[1] - 1)
    score = torch.rand(pad.shape, device=pad.device) + 1.0 * pad.float() + 4.0 * sel.float()
    return torch.zeros_like(pad).scatter_(1, score.topk(k, dim=1).indices, True)


def phase_gap_mask(t_days, pad, period, width):
    """Hide the real tokens whose phase (t / P + u) mod 1 < width (u random per row); rows without a period
    hide nothing real. Rows are topped up to the same hidden count with pads, then random real tokens that are
    NOT scored (returned separately)."""
    real = ~pad
    ok = torch.isfinite(period) & (period > 0)
    u = torch.rand(len(period), device=pad.device)
    phi = torch.remainder(t_days / torch.where(ok, period, torch.ones_like(period))[:, None] + u[:, None], 1.0)
    band = real & ok[:, None] & (phi < width)
    band &= (real.sum(1) - band.sum(1) >= 1)[:, None]  # keep at least one visible token
    k = max(min(int(band.sum(1).max()), pad.shape[1] - 1), 1)
    score = torch.rand(pad.shape, device=pad.device) + 1.0 * pad.float() + 4.0 * band.float()
    return torch.zeros_like(pad).scatter_(1, score.topk(k, dim=1).indices, True), band


@torch.no_grad()
def collect(model, ds, spec, window, recs, mode, dev):
    """Per hidden scored token: star, window, t (days in window), dt, m, sigma, recon, predicted and target latent;
    per window: visible (t, m, sigma) for the baselines and the context encoder's frequency."""
    mu, sd = spec.err_stats
    ts = spec.tokenize["time_scale"]
    spec_layers = spectral_layers(model.context)
    loader = window_loader(ds, spec, 128)
    H = dict(star=[], win=[], t=[], dt=[], m=[], sig=[], recon=[], pred=[], target=[], nctx=[], n30=[])
    W = dict(t=[], m=[], sig=[], f=[], star=[])
    n_win = 0
    for b in loader:
        tok = b["tokens"]
        values, positions, pad = tok.values.to(dev), tok.positions.to(dev), tok.pad_mask.to(dev)
        t_days = (positions[:, 0].float() - 1.0) * ts
        if mode.pgap:
            per = torch.tensor([float(recs[int(o)].period or float("nan")) for o in b["obj"]], device=dev)
            mask, tail = phase_gap_mask(t_days, pad, per, mode.phase_gap)  # only the band is scored
        elif mode.gap:
            mask = gap_mask(t_days, pad, mode.gap_share)
            tail = mask & ~pad  # every hidden real token is scored
        else:
            mask, _ = forecast_tail_mask(torch.zeros_like(pad), t_days, pad, window, (mode.tail, mode.tail), 1.0)
            tail = (~pad) & (t_days > window - mode.tail)
        if int(mask.sum()) == 0:
            continue
        for m_ in spec_layers:
            m_.last_logits = None
        out = model(values, positions, pad, mask)
        f = spectral_frequency(spec_layers[0], None, "local").float() if spec_layers else torch.full((len(pad),), float("nan"), device=dev)
        bsz = pad.shape[0]

        def split(x):
            return x[mask].reshape(bsz, -1, *x.shape[2:])

        vis = (~mask) & (~pad)
        t_last = torch.where(vis, t_days, torch.full_like(t_days, -1e9)).max(1).values
        h_t, h_tail = split(t_days), split(tail)
        h_m = split(values[..., 0].float())
        h_sig = torch.exp(split(values[..., 1].float()) * sd + mu)
        if mode.gap:  # distance to the nearest visible token; visible tokens within 15 d
            dist = (h_t[:, :, None] - t_days[:, None, :]).abs().masked_fill(~vis[:, None, :], 1e9)
            h_dt = dist.min(2).values
            h_loc = (dist <= 15.0).sum(2).float()
            keep = h_tail & (vis.sum(1) >= mode.min_context)[:, None]
        else:
            h_dt = h_t - t_last[:, None]
            keep = h_tail & (vis.sum(1) >= mode.min_context)[:, None] & (h_dt >= mode.dt[0]) & (h_dt <= mode.dt[1])
        obj = torch.as_tensor(b["obj"].numpy(), device=dev)
        wid = torch.arange(bsz, device=dev) + n_win
        n_win += bsz
        nctx = vis.sum(1).float()  # visible (context) tokens of the window
        n30 = (vis & (t_days > (t_last - 30.0)[:, None])).sum(1).float()  # of them, in the last 30 days of the context
        n30 = h_loc if mode.gap else n30[:, None].expand_as(h_t)
        for k_, v_ in dict(nctx=nctx[:, None].expand_as(h_t), n30=n30, star=obj[:, None].expand_as(h_t), win=wid[:, None].expand_as(h_t), t=h_t, dt=h_dt, m=h_m, sig=h_sig,
                           recon=model.last_recon[..., 0] if model.recon_head is not None else torch.full_like(out.pred[..., 0], float("nan")),
                           pred=out.pred, target=out.target).items():
            H[k_].append(v_[keep].cpu() if k_ in ("star", "win") else v_[keep].float().cpu())
        tv, mv, sv = t_days.cpu().numpy(), values[..., 0].float().cpu().numpy(), torch.exp(values[..., 1].float() * sd + mu).cpu().numpy()
        vv = vis.cpu().numpy()
        for r in range(bsz):
            W["t"].append(tv[r][vv[r]]); W["m"].append(mv[r][vv[r]]); W["sig"].append(sv[r][vv[r]])
        W["f"].append(f.cpu()); W["star"].append(obj.cpu())
    Hn = {k: torch.cat(v).numpy() for k, v in H.items()}
    W["f"], W["star"] = torch.cat(W["f"]).numpy(), torch.cat(W["star"]).numpy()
    return Hn, W


def score(Hn, W, recs, star_train, label, rng, mode, save_npz=None):
    """Fit the latent decoder on the tokens of the ``star_train`` stars and score every row on the others."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    K = mode.harmonics
    star, win, t, dt, m, sig = (Hn[k] for k in ("star", "win", "t", "dt", "m", "sig"))
    tr = star_train[star]
    te = ~tr
    print(f"[{label}] {int(te.sum())} {'hidden in-window' if mode.gap else 'forecast'} tokens on {len(np.unique(star[te]))} test stars ({int(tr.sum())} for the decoder), "
          f"{'distance to nearest visible' if mode.gap else 'gap'} median {np.median(dt[te]):.2f} d", flush=True)
    sub = rng.permutation(np.flatnonzero(tr))[:60000]
    sc = StandardScaler().fit(Hn["target"][sub])
    mlp = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=80, early_stopping=True, random_state=0).fit(sc.transform(Hn["target"][sub]), m[sub])
    idx = np.flatnonzero(te)
    rows = {"recon head": Hn["recon"][idx]} if np.isfinite(Hn["recon"][idx]).any() else {}
    rows.update({"predicted latent [mlp]": mlp.predict(sc.transform(Hn["pred"][idx])),
            "target latent [mlp] (ceiling)": mlp.predict(sc.transform(Hn["target"][idx])),
            "window mean": np.array([W["m"][w].mean() for w in win[idx]])})
    four, four_c = np.empty(len(idx)), np.empty(len(idx))
    wi = win[idx]
    for w in np.unique(wi):  # one fit per window, evaluated at all its scored tokens
        sel = np.flatnonzero(wi == w)
        tw, mw, sw = W["t"][w], W["m"][w], W["sig"][w]
        fw = W["f"][w]
        p = recs[star[idx[sel[0]]]].period
        fc = 1.0 / p if p and np.isfinite(p) and p > 0 else fw
        tq = t[idx[sel]]
        four[sel] = fourier_forecast(tw, mw, sw, fw, tq, K) if np.isfinite(fw) else mw.mean()
        four_c[sel] = fourier_forecast(tw, mw, sw, fc, tq, K) if np.isfinite(fc) else mw.mean()
    rows[f"Fourier fit ({K} harmonics, f of the window)"] = four
    rows[f"Fourier fit ({K} harmonics, catalogue f)"] = four_c
    yj, sj, dtj = m[idx], sig[idx], dt[idx]
    out = {}
    for name, yhat in rows.items():
        z = (yhat - yj) / sj
        out[name] = dict(all=zstats(z), **{f"dt {lo:g}-{hi:g} d": zstats(z[(dtj >= lo) & (dtj < hi)]) for lo, hi in mode.dt_bins})
        r = out[name]["all"]
        print(f"[{label}] {name:44s} median|z| {r['median_abs_z']:.2f}  |z|<1 {r['within_1']:.3f}  |z|<2 {r['within_2']:.3f}  mean z {r['mean_z']:+.2f} | |z|<2 by gap "
              + "  ".join(f"{k} {v['within_2']:.3f}" for k, v in out[name].items() if k != "all"), flush=True)
    v_t = float(np.mean(np.var(Hn["target"][idx], 0)))
    out["latent_err_over_var"] = float(np.mean((Hn["pred"][idx] - Hn["target"][idx]) ** 2) / v_t)
    cls = np.array([str(recs[o].meta.get("class_str", recs[o].label)) for o in star[idx]])
    out["by_class"] = {}
    MAIN = [r for r in ("recon head", "predicted latent [mlp]") if r in rows]
    for name in MAIN + [f"Fourier fit ({K} harmonics, f of the window)"]:
        az = np.abs((rows[name] - yj) / sj)
        out["by_class"][name] = {c: dict(n=int((cls == c).sum()), within_2=float(np.mean(az[cls == c] < 2)))
                                 for c in sorted(set(cls), key=lambda c: -(cls == c).sum())}
    print(f"[{label}] latent error / target variance {out['latent_err_over_var']:.3f} | within 2 sigma by class, {MAIN[0]}: "
          + ", ".join(f"{c} {v['within_2']:.2f}" for c, v in list(out["by_class"][MAIN[0]].items())[:8]), flush=True)
    out["by_density"] = {}
    for key, v in (("context tokens", Hn["nctx"][idx]), (mode.local, Hn["n30"][idx])):
        qb = np.clip(np.searchsorted(np.quantile(v, [0.2, 0.4, 0.6, 0.8]), v, side="right"), 0, 4)
        out["by_density"][key] = {}
        for name in MAIN + [f"Fourier fit ({K} harmonics, f of the window)"]:
            az = np.abs((rows[name] - yj) / sj)
            out["by_density"][key][name] = {f"quintile {b + 1}": dict(n=int((qb == b).sum()), median=float(np.median(v[qb == b])),
                                                                     within_2=float(np.mean(az[qb == b] < 2))) for b in range(5)}
        print(f"[{label}] within 2 sigma by quintile of {key} (medians "
              + ", ".join(f"{x['median']:.0f}" for x in out["by_density"][key][MAIN[0]].values()) + "): "
              + " | ".join(f"{nm[:22]} " + " ".join(f"{x['within_2']:.2f}" for x in d.values()) for nm, d in out["by_density"][key].items()), flush=True)
    out["n"], out["gap_median_days"] = int(len(idx)), float(np.median(dtj))
    if save_npz:
        path = save_npz.replace(".npz", f"_{label}.npz")
        np.savez(path, dt=dtj, m_j=yj, sig_j=sj, row_names=np.array(list(rows)), yhat=np.stack([rows[k] for k in rows]), cls=cls, nctx=Hn["nctx"][idx], n30=Hn["n30"][idx])
        print("saved", path, flush=True)
    return out


def pick_stars(data, split, fit_split, n_stars, n_fit, rng):
    """The stars of one run, in a seeded random order: only stars with at least 64 points.

    Without ``fit_split``: ``n_stars`` stars of ``split`` (0 = all); a random 70 % fit the decoder and the
    rest are scored. With ``fit_split``: ``n_fit`` stars of that split fit the decoder and come first in
    the list, then the ``n_stars`` stars of ``split``, all scored. Returns ``(recs, star_train)``."""
    recs = [r for r in data[split] if r.n >= 64]
    rng.shuffle(recs)
    recs = recs[:n_stars] if n_stars > 0 else recs
    if fit_split is None:
        return recs, rng.random(len(recs)) < 0.7
    fit = [r for r in data[fit_split] if r.n >= 64]
    rng.shuffle(fit)
    fit = fit[:n_fit]
    recs = fit + recs
    return recs, np.r_[np.ones(len(fit), bool), np.zeros(len(recs) - len(fit), bool)]


def run(ckpt, mode, data=ZTFXPC, split="validation", fit_split=None, n_fit=1000, n_stars=600, stride=0.25, seed=0, save_npz=None, dev=None):
    """One checkpoint, one mode. Returns ``(label, scores)``, or None when the checkpoint is not a JEPA."""
    dev = dev or eval_device()
    rng = np.random.default_rng(seed)
    enc, meta = load_encoder(ckpt, dev)
    model = enc.model
    if not hasattr(model, "context"):
        print(f"{ckpt}: not a JEPA, skipped", flush=True)
        return None
    model.eval()
    cfg, spec = meta.cfg, meta.spec
    dargs = data_args_from(meta.args)
    dargs.data = data
    splits = (split,) if fit_split is None else tuple(dict.fromkeys((split, fit_split)))
    recs, star_train = pick_stars(load_data(dargs, splits=splits), split, fit_split, n_stars, n_fit, rng)
    print(f"{split}: {int((~star_train).sum())} stars scored; decoder on {int(star_train.sum())} stars of {fit_split or split}", flush=True)
    ds = window_dataset(split, recs, cfg, stride)
    t0 = time.time()
    Hn, W = collect(model, ds, spec, cfg.window, recs, mode, dev)
    label = run_name(ckpt) + mode.suffix
    res = score(Hn, W, recs, star_train, label, rng, mode, save_npz)
    res["seconds"] = round(time.time() - t0)
    return label, res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data", default=ZTFXPC)
    ap.add_argument("--n-stars", type=int, default=600, help="stars scored (0 = all of the split)")
    ap.add_argument("--split", default="validation", help="the split scored")
    ap.add_argument("--fit-split", default=None, help="fit the latent decoder on this split's stars (all of --split is scored); "
                    "default: 70 %% of --split's stars fit it, 30 %% are scored")
    ap.add_argument("--n-fit", type=int, default=1000, help="--fit-split: stars for the decoder")
    ap.add_argument("--stride", type=float, default=0.25, help="window advance, share of the window (1 = no overlap)")
    ap.add_argument("--tail", type=float, default=10.0, help="days hidden at the end of every window")
    ap.add_argument("--dt", type=float, nargs=2, default=(0.5, 10.0))
    ap.add_argument("--min-context", type=int, default=16)
    ap.add_argument("--harmonics", type=int, default=4)
    ap.add_argument("--gap-share", type=float, default=None, help="gap mode: hide this share of real tokens at random")
    ap.add_argument("--phase-gap", type=float, default=None, help="phase-gap mode: hide a band this many cycles wide (catalogue period)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-npz", default=None)
    a = ap.parse_args(argv)
    mode = Mode(a.tail, tuple(a.dt), a.gap_share, a.phase_gap, a.min_context, a.harmonics)
    results = {}
    for ck in a.ckpts:
        r = run(ck, mode, a.data, a.split, a.fit_split, a.n_fit, a.n_stars, a.stride, a.seed, a.save_npz)
        if r is not None:
            results[r[0]] = r[1]
    if a.out:
        write_json(a.out, results)
        print("wrote", a.out)
    return results


if __name__ == "__main__":
    main()
