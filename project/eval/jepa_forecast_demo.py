"""Forecast examples of a masked JEPA predictor (no add-on), as light curves.

For one validation star per class (a window with many points and a catalogue period), the last ``--hide`` days
of the window are hidden. The predictor gets placeholder tokens at the hidden observations AND on a dense time
grid over the hidden stretch (the window's busiest band), so its reconstruction head draws a curve. Shown: the
seen points of that band before the cut, the hidden points, the model's curve and its values at the hidden
points, and a 4-harmonic Fourier fit at the encoder's own frequency on the seen points. Each panel's title gives
the share of hidden points within 2 sigma for the model and for Fourier.

The catalogue period only picks stars that have one (evaluation only); the model uses its own frequency.

    python -m project.eval.jepa_forecast_demo --ckpt project/runs/enc_o2_fcbig_ph/mae.pt --out project/results/figures/jepa_forecast_demo
"""
import argparse, os, sys
sys.path.insert(0, "/projects/bfrf/hibb/romae-lc")
import numpy as np
import torch

from project.common import data_args_from, load_data, load_encoder
from project.cache_latents import WindowDataset, make_collate
from romae_lc.model import spectral_frequency
from project.eval.dataset_figs_style import BLUE, ORANGE, INK, INK2, MUTED, tidy, plt

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", default="/projects/bfrf/data/PC_matches/ZTFxPC")
ap.add_argument("--hide", type=float, default=30.0, help="days hidden at the end of the window")
ap.add_argument("--show", type=float, default=60.0, help="days of seen points shown before the cut")
ap.add_argument("--grid", type=int, default=200, help="query tokens on the hidden stretch")
ap.add_argument("--classes", default="EW/EB,EA,RRAB,RRC,ROT,DSCT,ELL,LPV")
ap.add_argument("--n-stars", type=int, default=3000)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"


def fourier(t, m, sig, f, t_new, K=4):
    def X(tt):
        return np.stack([np.ones_like(tt)] + [g(2 * np.pi * k * f * tt) for k in range(1, K + 1) for g in (np.cos, np.sin)], 1)
    w = 1 / np.clip(sig, 1e-6, None)
    c, *_ = np.linalg.lstsq(X(t) * w[:, None], m * w, rcond=None)
    return X(t_new) @ c


enc, meta = load_encoder(a.ckpt, dev)
model = enc.model.eval()
cfg, spec = meta.cfg, meta.spec
mu, sd = spec.err_stats
ts = spec.tokenize["time_scale"]
dargs = data_args_from(meta.args)
dargs.data = a.data
rng = np.random.default_rng(a.seed)
val = [r for r in load_data(dargs, splits=("validation",))["validation"] if r.n >= 200 and r.period and np.isfinite(r.period)]
rng.shuffle(val)
val = val[: a.n_stars]
cls_of = lambda r: str(r.meta.get("class_str", r.label))
want = a.classes.split(",")
ds = WindowDataset([("validation", val, list(range(len(val))))], cfg, 0.25, cfg.max_tokens, 0, 1, 0.5)
collate = make_collate(spec)
spec_layers = [m for m in model.context.modules() if type(m).__name__ == "SpectralLayer"]

picked = {}
for i in range(len(ds)):
    c = cls_of(val[ds.locate(i)[1]])
    if c not in want or c in picked:
        continue
    it = ds[i]
    b = collate([it])
    tok = b["tokens"]
    t_days = ((tok.positions[0, 0].float() - 1.0) * ts).numpy()
    real = ~tok.pad_mask[0].numpy()
    seen = real & (t_days <= cfg.window - a.hide)
    hid = real & (t_days > cfg.window - a.hide)
    if seen.sum() < 80 or hid.sum() < 8:
        continue
    picked[c] = (it, b)
    if len(picked) == len(want):
        break
print("picked", list(picked), flush=True)

fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), dpi=140)
for ax, c in zip(axes.flat, [c for c in want if c in picked]):
    it, b = picked[c]
    tok = b["tokens"]
    values, positions, pad = tok.values.to(dev), tok.positions.to(dev), tok.pad_mask.to(dev)
    t_days = (positions[0, 0].float() - 1.0) * ts
    real = ~pad[0]
    bands = positions[0, 1][real]
    band = bands.mode().values  # the busiest band
    # query tokens on a dense grid over the hidden stretch, in that band
    tq = torch.linspace(cfg.window - a.hide, cfg.window, a.grid, device=dev)
    pq = positions[:, :, :1].repeat(1, 1, a.grid).clone()
    pq[0, 0] = (tq / ts + 1.0).to(pq.dtype)
    pq[0, 1] = band
    vq = values[:, real.nonzero()[0, 0]][:, None].repeat(1, a.grid, 1)
    V = torch.cat([values[:, real], vq], 1)
    P = torch.cat([positions[:, :, real], pq], 2)
    Tn = (P[0, 0].float() - 1.0) * ts
    pad2 = torch.zeros(1, V.shape[1], dtype=torch.bool, device=dev)
    mask = (Tn > cfg.window - a.hide)[None]
    with torch.no_grad():
        for m_ in spec_layers:
            m_.last_logits = None
        model(V, P, pad2, mask)
        f = float(spectral_frequency(spec_layers[0], None, "local")[0]) if spec_layers else float("nan")
    rec = model.last_recon[0, :, 0].float().cpu().numpy()
    tn = Tn.cpu().numpy()
    mk = mask[0].cpu().numpy()
    is_q = np.r_[np.zeros(int(real.sum()), bool), np.ones(a.grid, bool)]
    bnd = (P[0, 1] == band).cpu().numpy()
    m_all = V[0, :, 0].float().cpu().numpy()
    s_all = torch.exp(V[0, :, 1].float() * sd + mu).cpu().numpy()
    rec_t, rec_q = tn[mk], rec
    in_q = is_q[mk]
    obs_h = mk & ~is_q & bnd  # hidden real points of the band
    seen_b = ~mk & bnd
    # model at the hidden real points of the band: the recon rows in mask order
    order = np.flatnonzero(mk)
    rec_at = {k: rec_q[j] for j, k in enumerate(order)}
    yh = np.array([rec_at[k] for k in np.flatnonzero(obs_h)])
    four_q = fourier(tn[seen_b], m_all[seen_b], s_all[seen_b], f, rec_t[in_q]) if np.isfinite(f) else None
    four_h = fourier(tn[seen_b], m_all[seen_b], s_all[seen_b], f, tn[obs_h]) if np.isfinite(f) else None
    w2m = np.mean(np.abs((yh - m_all[obs_h]) / s_all[obs_h]) < 2)
    w2f = np.mean(np.abs((four_h - m_all[obs_h]) / s_all[obs_h]) < 2) if four_h is not None else float("nan")
    lo = cfg.window - a.hide - a.show
    sel = seen_b & (tn >= lo)
    ax.errorbar(tn[sel], m_all[sel], yerr=s_all[sel], fmt="o", ms=3, color=INK2, ecolor=MUTED, lw=0.6, label="seen")
    ax.errorbar(tn[obs_h], m_all[obs_h], yerr=s_all[obs_h], fmt="o", ms=4, mfc="white", color=INK, ecolor=MUTED, lw=0.6, label="hidden (truth)")
    oq = np.argsort(rec_t[in_q])
    ax.plot(rec_t[in_q][oq], rec_q[in_q][oq], color=BLUE, lw=1.3, label="model (phase rope)")
    ax.plot(tn[obs_h], yh, "s", ms=3.5, color=BLUE)
    if four_q is not None:
        ax.plot(rec_t[in_q][oq], four_q[oq], color=ORANGE, lw=1.1, ls="--", label="Fourier (4 harmonics, own f)")
    ax.axvline(cfg.window - a.hide, color=MUTED, lw=1, ls=":")
    ax.invert_yaxis()
    P_cat = val[it["obj"]].period
    ax.set_title(f"{c}  P={P_cat:.3g} d   within 2$\\sigma$: model {w2m:.2f}, Fourier {w2f:.2f}", fontsize=9.5, pad=6)
    ax.set_xlabel("days in the window")
    tidy(ax, "y")
axes.flat[0].set_ylabel("magnitude (model units)")
axes.flat[4].set_ylabel("magnitude (model units)")
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9)
fig.suptitle(f"{os.path.basename(os.path.dirname(a.ckpt))}: last {a.hide:g} days of the window hidden, one validation star per class", y=1.02, fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out + ".png", bbox_inches="tight")
fig.savefig(a.out + ".pdf", bbox_inches="tight")
print("wrote", a.out, flush=True)
