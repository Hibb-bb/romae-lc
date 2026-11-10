"""Can the MASKED JEPA predictor forecast? The forecast test for runs with no add-on (``--forecast-tail``).

For a validation star's window, every token in the last ``--tail`` days is hidden; the context encoder sees
the rest. The predictor's placeholder tokens sit at the hidden times and bands, as in training. Each hidden
observation j is scored by its gap from the last visible token, ``dt = t_j - t_last`` (0.5 to 10 days kept),
with the z-score against its error, ``z = (m_hat - m_j) / sigma_j``, as in project.eval.state_decode_probe.

Rows:
  recon head                 the predictor's own magnitude output (trained with --jepa-recon-weight)
  predicted latent [mlp]     an MLP decoder of the predicted latent (the 32-number projection), fitted on the
                             TARGET latents of the training stars' hidden tokens: is the predicted latent decodable?
  target latent [mlp]        the same decoder on the target encoder's own latent (it SAW m_j): the ceiling
  window mean                the mean magnitude of the visible tokens
  Fourier fit (f window)     4 harmonics at the context encoder's frequency estimate, fitted to the visible tokens
  Fourier fit (catalogue f)  the same at the catalogue frequency (a reference, evaluation only)

Also the latent error, mean squared (pred - target) over the target variance, and within 2 sigma by class.
Stars are split 70 / 30 for the decoder; every score is on the 30 %.

    python -m project.eval.jepa_forecast_probe --ckpts project/runs/enc_p1_fc/mae.pt --out project/results/jepa_forecast/enc_p1_fc.json
"""
import sys, argparse, json, os, time, numpy as np, torch
sys.path.insert(0, "/projects/bfrf/hibb/romae-lc")
from project.common import data_args_from, load_data, load_encoder
from project.cache_latents import WindowDataset, make_collate
from project.pretrain_mae import forecast_tail_mask
from romae_lc.model import spectral_frequency
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--data", default="/projects/bfrf/data/PC_matches/ZTFxPC")
ap.add_argument("--n-stars", type=int, default=600)
ap.add_argument("--tail", type=float, default=10.0, help="days hidden at the end of every window")
ap.add_argument("--dt", type=float, nargs=2, default=(0.5, 10.0))
ap.add_argument("--min-context", type=int, default=16)
ap.add_argument("--harmonics", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
ap.add_argument("--save-npz", default=None)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
DT_BINS = [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0)]


def fourier_forecast(t, m, sig, f, t_new, K):
    def design(tt):
        cols = [np.ones_like(tt)]
        for k in range(1, K + 1):
            cols += [np.cos(2 * np.pi * k * f * tt), np.sin(2 * np.pi * k * f * tt)]
        return np.stack(cols, 1)
    w = 1.0 / np.clip(sig, 1e-6, None)
    coef, *_ = np.linalg.lstsq(design(t) * w[:, None], m * w, rcond=None)
    return design(t_new) @ coef


def zstats(z):
    az = np.abs(z)
    return dict(n=int(len(z)), median_abs_z=float(np.median(az)) if len(z) else float("nan"),
                within_1=float(np.mean(az < 1)) if len(z) else float("nan"), within_2=float(np.mean(az < 2)) if len(z) else float("nan"),
                mean_z=float(np.mean(z)) if len(z) else float("nan"))


@torch.no_grad()
def collect(model, ds, spec, window):
    """Per hidden tail token: star, window, t (days in window), dt, m, sigma, recon, predicted and target latent;
    per window: visible (t, m, sigma) for the baselines and the context encoder's frequency."""
    mu, sd = spec.err_stats
    ts = spec.tokenize["time_scale"]
    spec_layers = [m for m in model.context.modules() if type(m).__name__ == "SpectralLayer"]
    loader = DataLoader(ds, 128, shuffle=False, num_workers=0, collate_fn=make_collate(spec))
    H = dict(star=[], win=[], t=[], dt=[], m=[], sig=[], recon=[], pred=[], target=[], nctx=[], n30=[])
    W = dict(t=[], m=[], sig=[], f=[], star=[])
    n_win = 0
    for b in loader:
        tok = b["tokens"]
        values, positions, pad = tok.values.to(dev), tok.positions.to(dev), tok.pad_mask.to(dev)
        t_days = (positions[:, 0].float() - 1.0) * ts
        mask0 = torch.zeros_like(pad)
        mask, _ = forecast_tail_mask(mask0, t_days, pad, window, (a.tail, a.tail), 1.0)
        tail = (~pad) & (t_days > window - a.tail)
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
        h_dt = h_t - t_last[:, None]
        keep = h_tail & (vis.sum(1) >= a.min_context)[:, None] & (h_dt >= a.dt[0]) & (h_dt <= a.dt[1])
        obj = torch.as_tensor(b["obj"].numpy(), device=dev)
        wid = torch.arange(bsz, device=dev) + n_win
        n_win += bsz
        nctx = vis.sum(1).float()  # visible (context) tokens of the window
        n30 = (vis & (t_days > (t_last - 30.0)[:, None])).sum(1).float()  # of them, in the last 30 days of the context
        for k_, v_ in dict(nctx=nctx[:, None].expand_as(h_t), n30=n30[:, None].expand_as(h_t), star=obj[:, None].expand_as(h_t), win=wid[:, None].expand_as(h_t), t=h_t, dt=h_dt, m=h_m, sig=h_sig,
                           recon=model.last_recon[..., 0], pred=out.pred, target=out.target).items():
            H[k_].append(v_[keep].cpu() if k_ in ("star", "win") else v_[keep].float().cpu())
        tv, mv, sv = t_days.cpu().numpy(), values[..., 0].float().cpu().numpy(), torch.exp(values[..., 1].float() * sd + mu).cpu().numpy()
        vv = vis.cpu().numpy()
        for r in range(bsz):
            W["t"].append(tv[r][vv[r]]); W["m"].append(mv[r][vv[r]]); W["sig"].append(sv[r][vv[r]])
        W["f"].append(f.cpu()); W["star"].append(obj.cpu())
    Hn = {k: torch.cat(v).numpy() for k, v in H.items()}
    W["f"], W["star"] = torch.cat(W["f"]).numpy(), torch.cat(W["star"]).numpy()
    return Hn, W


def score(Hn, W, recs, star_train, label, rng):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    star, win, t, dt, m, sig = (Hn[k] for k in ("star", "win", "t", "dt", "m", "sig"))
    tr = star_train[star]
    te = ~tr
    print(f"[{label}] {int(te.sum())} forecast tokens on {len(np.unique(star[te]))} test stars ({int(tr.sum())} for the decoder), "
          f"gap median {np.median(dt[te]):.1f} d", flush=True)
    sub = rng.permutation(np.flatnonzero(tr))[:60000]
    sc = StandardScaler().fit(Hn["target"][sub])
    mlp = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=80, early_stopping=True, random_state=0).fit(sc.transform(Hn["target"][sub]), m[sub])
    idx = np.flatnonzero(te)
    rows = {"recon head": Hn["recon"][idx],
            "predicted latent [mlp]": mlp.predict(sc.transform(Hn["pred"][idx])),
            "target latent [mlp] (ceiling)": mlp.predict(sc.transform(Hn["target"][idx])),
            "window mean": np.array([W["m"][w].mean() for w in win[idx]])}
    four, four_c = np.empty(len(idx)), np.empty(len(idx))
    for n_, k in enumerate(idx):
        w = win[k]
        tw, mw, sw = W["t"][w], W["m"][w], W["sig"][w]
        fw = W["f"][w]
        p = recs[star[k]].period
        fc = 1.0 / p if p and np.isfinite(p) and p > 0 else fw
        four[n_] = fourier_forecast(tw, mw, sw, fw, np.array([t[k]]), a.harmonics)[0] if np.isfinite(fw) else mw.mean()
        four_c[n_] = fourier_forecast(tw, mw, sw, fc, np.array([t[k]]), a.harmonics)[0]
    rows[f"Fourier fit ({a.harmonics} harmonics, f of the window)"] = four
    rows[f"Fourier fit ({a.harmonics} harmonics, catalogue f)"] = four_c
    yj, sj, dtj = m[idx], sig[idx], dt[idx]
    out = {}
    for name, yhat in rows.items():
        z = (yhat - yj) / sj
        out[name] = dict(all=zstats(z), **{f"dt {lo:g}-{hi:g} d": zstats(z[(dtj >= lo) & (dtj < hi)]) for lo, hi in DT_BINS})
        r = out[name]["all"]
        print(f"[{label}] {name:44s} median|z| {r['median_abs_z']:.2f}  |z|<1 {r['within_1']:.3f}  |z|<2 {r['within_2']:.3f}  mean z {r['mean_z']:+.2f} | |z|<2 by gap "
              + "  ".join(f"{k} {v['within_2']:.3f}" for k, v in out[name].items() if k != "all"), flush=True)
    v_t = float(np.mean(np.var(Hn["target"][idx], 0)))
    out["latent_err_over_var"] = float(np.mean((Hn["pred"][idx] - Hn["target"][idx]) ** 2) / v_t)
    cls = np.array([str(recs[o].meta.get("class_str", recs[o].label)) for o in star[idx]])
    out["by_class"] = {}
    for name in ("recon head", "predicted latent [mlp]", f"Fourier fit ({a.harmonics} harmonics, f of the window)"):
        az = np.abs((rows[name] - yj) / sj)
        out["by_class"][name] = {c: dict(n=int((cls == c).sum()), within_2=float(np.mean(az[cls == c] < 2)))
                                 for c in sorted(set(cls), key=lambda c: -(cls == c).sum())}
    print(f"[{label}] latent error / target variance {out['latent_err_over_var']:.3f} | within 2 sigma by class, recon head: "
          + ", ".join(f"{c} {v['within_2']:.2f}" for c, v in list(out["by_class"]["recon head"].items())[:8]), flush=True)
    out["by_density"] = {}
    for key, v in (("context tokens", Hn["nctx"][idx]), ("context tokens in the last 30 d", Hn["n30"][idx])):
        qb = np.digitize(v, np.quantile(v, [0.2, 0.4, 0.6, 0.8]))
        out["by_density"][key] = {}
        for name in ("recon head", "predicted latent [mlp]", f"Fourier fit ({a.harmonics} harmonics, f of the window)"):
            az = np.abs((rows[name] - yj) / sj)
            out["by_density"][key][name] = {f"quintile {b + 1}": dict(n=int((qb == b).sum()), median=float(np.median(v[qb == b])),
                                                                     within_2=float(np.mean(az[qb == b] < 2))) for b in range(5)}
        print(f"[{label}] within 2 sigma by quintile of {key} (medians "
              + ", ".join(f"{x['median']:.0f}" for x in out["by_density"][key]["recon head"].values()) + "): "
              + " | ".join(f"{nm[:22]} " + " ".join(f"{x['within_2']:.2f}" for x in d.values()) for nm, d in out["by_density"][key].items()), flush=True)
    out["n"], out["gap_median_days"] = int(len(idx)), float(np.median(dtj))
    if a.save_npz:
        path = a.save_npz.replace(".npz", f"_{label}.npz")
        np.savez(path, dt=dtj, m_j=yj, sig_j=sj, row_names=np.array(list(rows)), yhat=np.stack([rows[k] for k in rows]), cls=cls, nctx=Hn["nctx"][idx], n30=Hn["n30"][idx])
        print("saved", path, flush=True)
    return out


results = {}
for ck in a.ckpts:
    rng = np.random.default_rng(a.seed)
    enc, meta = load_encoder(ck, dev)
    model = enc.model
    if not hasattr(model, "context") or getattr(model, "recon_head", None) is None:
        print(f"{ck}: not a JEPA with a reconstruction head, skipped", flush=True)
        continue
    model.eval()
    cfg, spec = meta.cfg, meta.spec
    dargs = data_args_from(meta.args)
    dargs.data = a.data
    val = load_data(dargs, splits=("validation",))["validation"]
    recs = [r for r in val if r.n >= 64]
    rng.shuffle(recs)
    recs = recs[: a.n_stars]
    star_train = rng.random(len(recs)) < 0.7
    ds = WindowDataset([("validation", recs, list(range(len(recs))))], cfg, 0.25, cfg.max_tokens, 0, 1, 0.5)
    t0 = time.time()
    Hn, W = collect(model, ds, spec, cfg.window)
    label = os.path.basename(os.path.dirname(ck))
    results[label] = score(Hn, W, recs, star_train, label, rng)
    results[label]["seconds"] = round(time.time() - t0)
if a.out:
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)
    print("wrote", a.out)
