"""Is the PREDICTED latent decodable? A forecast test of the dynamic-state add-on (project.dyn).

For a held-out star, a source token i sits in window A; the target j is an observation AFTER the end
of window A (so the encoder never saw it), 0.5 to 10 days after i. The add-on's predictor advances
the source state, ``pred = predict(s_i, CLS_A, f_star, t_j - t_i)``, and a decoder maps a state (plus
the window's CLS) to the magnitude. The decoder is fitted on the states of the training stars' own
tokens. Every prediction is scored by the z-score against the observation's uncertainty,

    z = (m_hat - m_j) / sigma_j      (magnitude and sigma in the same per-band units),

which is the right yardstick for noisy targets: |z| < 1 for 68 % and < 2 for 95 % of the points when
the forecast is as good as the noise allows.

Rows:
  predicted state     decode(pred, CLS_A)                     the question: does next-latent prediction decode?
  no change           decode(s_i, CLS_A)                      the source state itself (no predictor)
  target state        decode(s_j, CLS_B)                      ceiling: the target's own state (its encoder SAW m_j)
  window mean         the mean magnitude of window A          the trivial baseline
  Fourier fit         4 harmonics at f_star fitted to window A, evaluated at t_j   the classical baseline (no labels)

``f_star`` is the median of the star's window estimates from the spectral layer, as in training.
``--context`` also scores the JEPA context encoder (the add-on trains on it; downstream uses the target copy).
Stars are split 70 / 30 for the decoder; every score is on the 30 %.

    python -m project.eval.state_decode_probe --ckpts project/runs/enc_j3/mae.pt --context --out project/results/state_decode/enc_j3.json
"""
import sys, argparse, json, math, time, os, numpy as np, torch
sys.path.insert(0, "/projects/bfrf/hibb/romae-lc")
from project.common import data_args_from, load_data, load_encoder
from project.cache_latents import WindowDataset, make_collate
from romae_lc.model import spectral_frequency
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--data", default="/projects/bfrf/data/PC_matches/ZTFxPC")
ap.add_argument("--n-stars", type=int, default=600)
ap.add_argument("--pairs-per-window", type=int, default=20)
ap.add_argument("--dt", type=float, nargs=2, default=(0.5, 10.0))
ap.add_argument("--harmonics", type=int, default=4)
ap.add_argument("--context", action="store_true", help="also score the JEPA context encoder")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(a.seed)
_, meta0 = load_encoder(a.ckpts[0], "cpu")
dargs = data_args_from(meta0.args)
dargs.data = a.data
val = load_data(dargs, splits=("validation",))["validation"]
recs = [r for r in val if r.n >= 64]
rng.shuffle(recs)
recs = recs[: a.n_stars]
star_train = rng.random(len(recs)) < 0.7
print(f"{len(recs)} validation stars ({int(star_train.sum())} for the decoder fit)", flush=True)
DT_BINS = [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0)]


@torch.no_grad()
def encode_all(enc, model, ds, spec, dyn):
    """Every real token of every window: star, window, t_abs, m, sigma, state; per window: CLS, start, f."""
    mu, sd = spec.err_stats
    ts = spec.tokenize["time_scale"]
    loader = DataLoader(ds, 128, shuffle=False, num_workers=0, collate_fn=make_collate(spec))
    T = dict(star=[], win=[], t=[], m=[], sig=[], s=[])
    W = dict(cls=[], start=[], f=[], star=[], n=[])
    n_win = 0
    for b in loader:
        for mod in enc.modules():  # forget the last run of every spectral layer, so the one that runs now is found
            if type(mod).__name__ == "SpectralLayer":
                mod.last_logits = None
        tok = b["tokens"]
        x, pad = model.encode(tok.values.to(dev), tok.positions.to(dev), tok.pad_mask.to(dev))
        ran = [m for m in enc.modules() if type(m).__name__ == "SpectralLayer" and getattr(m, "last_logits", None) is not None]
        f = spectral_frequency(ran[0], None, "local").float()
        x = x.float()
        real = ~pad[:, 1:]
        s = dyn.states(x[:, 1:])
        t_days = (tok.positions[:, 0].to(dev).double() - 1.0) * ts
        t_abs = b["start"].to(dev).double()[:, None] + t_days
        m = tok.values[..., 0].to(dev).float()
        sig = torch.exp(tok.values[..., 1].to(dev).float() * sd + mu)
        obj = torch.as_tensor(b["obj"].numpy(), device=dev)
        wid = torch.arange(x.shape[0], device=dev) + n_win
        n_win += x.shape[0]
        T["star"].append(obj[:, None].expand_as(m)[real].cpu()); T["win"].append(wid[:, None].expand_as(m)[real].cpu())
        T["t"].append(t_abs[real].cpu()); T["m"].append(m[real].cpu()); T["sig"].append(sig[real].cpu()); T["s"].append(s[real].cpu())
        W["cls"].append(x[:, 0].cpu()); W["start"].append(b["start"].double()); W["f"].append(f.cpu()); W["star"].append(obj.cpu())
        W["n"].append(real.sum(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in T.items()}, {k: torch.cat(v).numpy() for k, v in W.items()}


def fourier_forecast(t, m, sig, f, t_new, K):
    """Weighted least-squares fit of K harmonics at f to (t, m); the fit evaluated at t_new."""
    def design(tt):
        cols = [np.ones_like(tt)]
        for k in range(1, K + 1):
            cols += [np.cos(2 * np.pi * k * f * tt), np.sin(2 * np.pi * k * f * tt)]
        return np.stack(cols, 1)
    w = 1.0 / np.clip(sig, 1e-6, None)
    A = design(t) * w[:, None]
    coef, *_ = np.linalg.lstsq(A, m * w, rcond=None)
    return design(t_new) @ coef


def zstats(z):
    az = np.abs(z)
    return dict(n=int(len(z)), median_abs_z=float(np.median(az)), within_1=float(np.mean(az < 1)), within_2=float(np.mean(az < 2)),
                within_3=float(np.mean(az < 3)), mean_z=float(np.mean(z)))


def score(tokens, wins, dyn, window_days, label):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    star, win, t, m, sig, s = (tokens[k] for k in ("star", "win", "t", "m", "sig", "s"))
    cls = wins["cls"]
    # per-star frequency: the median of the windows with enough tokens
    f_star = {}
    for o in np.unique(wins["star"]):
        sel = (wins["star"] == o) & (wins["n"] >= 32)
        f_star[o] = float(np.median(wins["f"][sel])) if sel.any() else float("nan")
    # decoder on the training stars' own tokens: [state, CLS of the window] -> magnitude
    tr = star_train[star]
    sub = rng.choice(np.flatnonzero(tr), min(200000, int(tr.sum())), replace=False)
    feat = lambda st, cl: np.concatenate([st, cl], 1)
    sc = StandardScaler().fit(feat(s[sub], cls[win[sub]]))
    dec = RidgeCV(alphas=np.logspace(-2, 4, 7)).fit(sc.transform(feat(s[sub], cls[win[sub]])), m[sub])
    from sklearn.neural_network import MLPRegressor
    sub_mlp = sub[:60000]
    mlp = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=80, early_stopping=True, random_state=0).fit(
        sc.transform(feat(s[sub_mlp], cls[win[sub_mlp]])), m[sub_mlp])
    decoders = {"ridge": lambda st, cl: dec.predict(sc.transform(feat(st, cl))), "mlp": lambda st, cl: mlp.predict(sc.transform(feat(st, cl)))}
    # forecast pairs on the test stars: source in window A, target after A's end
    I, J, A_ = [], [], []
    by_star = {}
    for k in np.flatnonzero(~tr):
        by_star.setdefault(star[k], []).append(k)
    for o, idx in by_star.items():
        idx = np.array(idx)
        if not np.isfinite(f_star.get(o, np.nan)):
            continue
        for wA in np.unique(win[idx]):
            endA = wins["start"][wA] + window_days
            src = idx[(win[idx] == wA) & (t[idx] >= endA - a.dt[1])]
            tgt = idx[(t[idx] > endA) & (t[idx] <= endA + a.dt[1])]
            if len(src) == 0 or len(tgt) == 0:
                continue
            for _ in range(a.pairs_per_window):
                i = rng.choice(src)
                ok = tgt[(t[tgt] - t[i] >= a.dt[0]) & (t[tgt] - t[i] <= a.dt[1])]
                if len(ok):
                    I.append(i); J.append(rng.choice(ok)); A_.append(wA)
    I, J, A_ = np.array(I), np.array(J), np.array(A_)
    dt = t[J] - t[I]
    fs = np.array([f_star[o] for o in star[I]])
    print(f"[{label}] {len(I)} forecast pairs on {len(by_star)} test stars, gap median {np.median(dt):.1f} d", flush=True)
    with torch.no_grad():
        pred = dyn.predict(torch.as_tensor(s[I]).float().to(dev), torch.as_tensor(cls[A_]).float().to(dev),
                           torch.as_tensor(fs).float().to(dev), torch.as_tensor(dt).float().to(dev)).cpu().numpy()
    yj, sj = m[J], sig[J]
    rows = {}
    for dname, decode in decoders.items():
        rows[f"predicted state [{dname}]"] = decode(pred, cls[A_])
        rows[f"no change [{dname}]"] = decode(s[I], cls[A_])
        rows[f"target state, saw m_j [{dname}]"] = decode(s[J], cls[win[J]])
    rows |= {
        "window mean": (np.bincount(win, weights=m, minlength=len(cls)) / np.maximum(np.bincount(win, minlength=len(cls)), 1))[A_],
    }
    four = np.empty(len(I))
    cache = {}
    for n_, (w, tj) in enumerate(zip(A_, t[J])):
        if w not in cache:
            sel = win == w
            cache[w] = (t[sel], m[sel], sig[sel])
        tt, mm, ss = cache[w]
        o = star[I[n_]]
        four[n_] = fourier_forecast(tt - tt.min(), mm, ss, f_star[o], np.array([tj - tt.min()]), a.harmonics)[0]
    rows[f"Fourier fit ({a.harmonics} harmonics at f_star)"] = four
    out = {}
    for name, yhat in rows.items():
        z = (yhat - yj) / sj
        out[name] = dict(all=zstats(z), **{f"dt {lo:g}-{hi:g} d": zstats(z[(dt >= lo) & (dt < hi)]) for lo, hi in DT_BINS})
        r = out[name]["all"]
        print(f"[{label}] {name:42s} median|z| {r['median_abs_z']:.2f}  |z|<1 {r['within_1']:.3f}  |z|<2 {r['within_2']:.3f}  "
              f"|z|<3 {r['within_3']:.3f}  mean z {r['mean_z']:+.2f}  | |z|<2 by gap "
              + "  ".join(f"{k} {v['within_2']:.3f}" for k, v in out[name].items() if k != "all"), flush=True)
    out["n_pairs"], out["gap_median_days"] = int(len(I)), float(np.median(dt))
    return out


results = {}
for ck in a.ckpts:
    enc, meta = load_encoder(ck, dev)
    model = enc.backbone.model if hasattr(enc.backbone, "model") else enc.backbone
    cfg, spec = meta.cfg, meta.spec
    dyn_sd = torch.load(ck, map_location="cpu", weights_only=False).get("aux", {}).get("dyn")
    if dyn_sd is None:
        print(f"{ck}: no dynamic-state head, skipped", flush=True)
        continue
    from project.dyn import DynHead, HamHead
    D, K2 = dyn_sd["s_head.weight"].shape[1], dyn_sd["s_head.weight"].shape[0]
    dyn = (HamHead if "k_mlp.0.weight" in dyn_sd else DynHead)(D, K2 // 2).to(dev)
    dyn.load_state_dict(dyn_sd)
    dyn.eval()
    ds = WindowDataset([("validation", recs, list(range(len(recs))))], cfg, 0.25, cfg.max_tokens, 0, 1, 0.5)
    variants = [("encoder used downstream", None)]
    if a.context and hasattr(model, "use_target"):
        variants.append(("JEPA context encoder (the one the add-on trains on)", False))
    for vname, use_target in variants:
        if use_target is not None:
            model.use_target = use_target
        t0 = time.time()
        tokens, wins = encode_all(enc, model, ds, spec, dyn)
        label = f"{os.path.basename(os.path.dirname(ck))} | {vname}"
        results[label] = score(tokens, wins, dyn, cfg.window, label)
        results[label]["seconds"] = round(time.time() - t0)
if a.out:
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)
    print("wrote", a.out)
