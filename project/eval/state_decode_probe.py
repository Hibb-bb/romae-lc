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

Diagnostics of the predictor (the ICML plan, Phase 1a), on the same test pairs:
  D1  predicted state [mlp fit on predicted]   the decoder refit on PREDICTED states of the training stars'
      pairs: is the predicted state off the set of real states (refit fixes it) or has it lost the information?
  D2  rows with ", catalogue f": the catalogue frequency in place of the encoder's f_star (evaluation only;
      the period never enters the encoder): how much of the error comes from f?
  D3  "state_geometry": the predictor's error in state space (over the state variance), with the encoder's
      and the catalogue f, against no change; and per star, the share of the state variance a 4-harmonic
      function of the catalogue phase explains (1 = the states trace a loop in phase, 0 = a cloud), by class.
  D4  "by_class" and "by_amplitude": within 2 sigma per class and per amplitude tercile (amplitude = the 5-95 %
      range of window A over its median error).

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
ap.add_argument("--save-npz", default=None, help="per-pair z-scores and a few example forecast curves, for the figures")
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
    from project.phase_decoder import band_ids, band_table
    T = dict(star=[], win=[], t=[], m=[], sig=[], s=[], band=[])
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
        T["band"].append(band_ids(tok.positions[:, 1].to(dev), band_table(spec))[real].cpu())
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


def score(tokens, wins, dyn, window_days, label, spec):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    star, win, t, m, sig, s = (tokens[k] for k in ("star", "win", "t", "m", "sig", "s"))
    cls = wins["cls"]
    # per-star frequency: the median of the windows with enough tokens
    f_star = {}
    for o in np.unique(wins["star"]):
        sel = (wins["star"] == o) & (wins["n"] >= 32)
        f_star[o] = float(np.median(wins["f"][sel])) if sel.any() else float("nan")
    f_cat = {o: (1.0 / recs[o].period if recs[o].period and np.isfinite(recs[o].period) and recs[o].period > 0 else float("nan"))
             for o in np.unique(wins["star"])}
    # decoder on the training stars' own tokens: [state, CLS of the window] -> magnitude
    tr = star_train[star]
    sub = rng.choice(np.flatnonzero(tr), min(200000, int(tr.sum())), replace=False)
    feat = lambda st, cl: np.concatenate([st, cl], 1)
    sc = StandardScaler().fit(feat(s[sub], cls[win[sub]]))
    dec = RidgeCV(alphas=np.logspace(-2, 4, 7)).fit(sc.transform(feat(s[sub], cls[win[sub]])), m[sub])
    sub_mlp = sub[:60000]
    mlp = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=80, early_stopping=True, random_state=0).fit(
        sc.transform(feat(s[sub_mlp], cls[win[sub_mlp]])), m[sub_mlp])
    decoders = {"ridge": lambda st, cl: dec.predict(sc.transform(feat(st, cl))), "mlp": lambda st, cl: mlp.predict(sc.transform(feat(st, cl)))}
    # forecast pairs: source in window A, target after A's end
    def make_pairs(star_mask):
        I, J, A_ = [], [], []
        by_star = {}
        for k in np.flatnonzero(star_mask):
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
        return np.array(I), np.array(J), np.array(A_), len(by_star)

    def run_pred(I, A_, fs, dt):
        with torch.no_grad():
            return dyn.predict(torch.as_tensor(s[I]).float().to(dev), torch.as_tensor(cls[A_]).float().to(dev),
                               torch.as_tensor(fs).float().to(dev), torch.as_tensor(dt).float().to(dev)).cpu().numpy()

    I, J, A_, n_test_stars = make_pairs(~tr)
    dt = t[J] - t[I]
    fs = np.array([f_star[o] for o in star[I]])
    fcat = np.array([f_cat[o] for o in star[I]])
    print(f"[{label}] {len(I)} forecast pairs on {n_test_stars} test stars, gap median {np.median(dt):.1f} d", flush=True)
    pred = run_pred(I, A_, fs, dt)
    okc = np.isfinite(fcat)
    pred_cat = run_pred(I, A_, np.where(okc, fcat, fs), dt)
    # D1: a decoder fitted on predicted states (training stars' pairs)
    Itr, Jtr, Atr, _ = make_pairs(tr)
    pred_tr = run_pred(Itr, Atr, np.array([f_star[o] for o in star[Itr]]), t[Jtr] - t[Itr])
    ktr = rng.permutation(len(Itr))[:60000]
    sc_p = StandardScaler().fit(feat(pred_tr[ktr], cls[Atr[ktr]]))
    mlp_p = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=80, early_stopping=True, random_state=0).fit(
        sc_p.transform(feat(pred_tr[ktr], cls[Atr[ktr]])), m[Jtr[ktr]])
    print(f"[{label}] D1: decoder refit on {len(ktr)} predicted states of {len(Itr)} training-star pairs", flush=True)
    yj, sj = m[J], sig[J]
    rows = {}
    for dname, decode in decoders.items():
        rows[f"predicted state [{dname}]"] = decode(pred, cls[A_])
        rows[f"no change [{dname}]"] = decode(s[I], cls[A_])
        rows[f"target state, saw m_j [{dname}]"] = decode(s[J], cls[win[J]])
        rows[f"predicted state, catalogue f [{dname}]"] = decode(pred_cat, cls[A_])
    rows["predicted state [mlp fit on predicted]"] = mlp_p.predict(sc_p.transform(feat(pred, cls[A_])))
    rows["predicted state, catalogue f [mlp fit on predicted]"] = mlp_p.predict(sc_p.transform(feat(pred_cat, cls[A_])))
    if hasattr(dyn, "dec"):
        with torch.no_grad():
            bj = torch.as_tensor(tokens["band"][J]).to(dev)
            rows["predicted state [trained decoder]"] = dyn.decode(torch.as_tensor(pred).float().to(dev), torch.as_tensor(cls[A_]).float().to(dev), bj).cpu().numpy()
            rows["predicted state, catalogue f [trained decoder]"] = dyn.decode(torch.as_tensor(pred_cat).float().to(dev), torch.as_tensor(cls[A_]).float().to(dev), bj).cpu().numpy()
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
    four_c = np.empty(len(I))
    for n_, (w, tj) in enumerate(zip(A_, t[J])):
        tt, mm, ss = cache[w]
        four_c[n_] = fourier_forecast(tt - tt.min(), mm, ss, fcat[n_] if okc[n_] else fs[n_], np.array([tj - tt.min()]), a.harmonics)[0]
    rows[f"Fourier fit ({a.harmonics} harmonics at catalogue f)"] = four_c
    out = {}
    for name, yhat in rows.items():
        z = (yhat - yj) / sj
        out[name] = dict(all=zstats(z), **{f"dt {lo:g}-{hi:g} d": zstats(z[(dt >= lo) & (dt < hi)]) for lo, hi in DT_BINS})
        r = out[name]["all"]
        print(f"[{label}] {name:42s} median|z| {r['median_abs_z']:.2f}  |z|<1 {r['within_1']:.3f}  |z|<2 {r['within_2']:.3f}  "
              f"|z|<3 {r['within_3']:.3f}  mean z {r['mean_z']:+.2f}  | |z|<2 by gap "
              + "  ".join(f"{k} {v['within_2']:.3f}" for k, v in out[name].items() if k != "all"), flush=True)
    out["n_pairs"], out["gap_median_days"] = int(len(I)), float(np.median(dt))
    out["catalogue_f_share"] = float(okc.mean())
    # D3: the predictor in state space, and the shape of a star's states
    v_s = float(np.mean(np.var(s[J], 0)))
    rel = lambda x: float(np.mean((x - s[J]) ** 2) / v_s)
    ratio = fs / np.where(okc, fcat, np.nan)
    geo = dict(pred_err_over_var=rel(pred), pred_catf_err_over_var=rel(pred_cat), no_change_err_over_var=rel(s[I]),
               f_star_over_f_cat_median=float(np.nanmedian(ratio)),
               f_within_2pc=float(np.nanmean(np.abs(ratio - 1) < 0.02)),
               f_half_or_double=float(np.nanmean((np.abs(ratio - 0.5) < 0.02) | (np.abs(ratio - 2) < 0.04))))
    cls_name = {o: str(recs[o].meta.get("class_str", recs[o].label)) for o in np.unique(star)}
    loop = {}
    for o in np.unique(star[~tr]):
        if not np.isfinite(f_cat[o]):
            continue
        k = np.flatnonzero(star == o)
        ph = np.remainder(t[k] * f_cat[o], 1.0)
        X = np.stack([np.ones_like(ph)] + [f(2 * np.pi * h * ph) for h in range(1, 5) for f in (np.cos, np.sin)], 1)
        Y = s[k] - s[k].mean(0)
        if len(k) < 40:
            continue
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        loop[o] = 1.0 - float(((Y - X @ coef) ** 2).sum() / max((Y ** 2).sum(), 1e-12))
    by_c = {}
    for o, v in loop.items():
        by_c.setdefault(cls_name[o], []).append(v)
    geo["phase_r2_median"] = float(np.median(list(loop.values()))) if loop else float("nan")
    geo["phase_r2_by_class"] = {c: dict(n=len(v), median=float(np.median(v))) for c, v in sorted(by_c.items(), key=lambda kv: -len(kv[1]))}
    out["state_geometry"] = geo
    print(f"[{label}] D3: state error / variance: predicted {geo['pred_err_over_var']:.3f}, with catalogue f {geo['pred_catf_err_over_var']:.3f}, "
          f"no change {geo['no_change_err_over_var']:.3f} | f_star within 2 % of catalogue f {geo['f_within_2pc']:.2f}, half/double {geo['f_half_or_double']:.2f} | "
          f"phase R2 of the states (median per star) {geo['phase_r2_median']:.2f}: "
          + ", ".join(f"{c} {v['median']:.2f} (n {v['n']})" for c, v in geo["phase_r2_by_class"].items()), flush=True)
    # D4: by class and by amplitude
    amp_w = {}
    for w in np.unique(A_):
        tt, mm, ss = cache[w]
        amp_w[w] = float((np.percentile(mm, 95) - np.percentile(mm, 5)) / max(np.median(ss), 1e-6))
    amp = np.array([amp_w[w] for w in A_])
    cl_pair = np.array([cls_name[o] for o in star[I]])
    terc = np.quantile(amp, [1 / 3, 2 / 3])
    abin = np.digitize(amp, terc)
    key_rows = [r for r in (f"predicted state [mlp]", "predicted state [mlp fit on predicted]", f"predicted state, catalogue f [mlp]",
                            f"Fourier fit ({a.harmonics} harmonics at f_star)", f"target state, saw m_j [mlp]") if r in rows]
    out["by_class"], out["by_amplitude"] = {}, {}
    for r in key_rows:
        az = np.abs((rows[r] - yj) / sj)
        out["by_class"][r] = {c: dict(n=int((cl_pair == c).sum()), within_2=float(np.mean(az[cl_pair == c] < 2)))
                              for c in sorted(set(cl_pair), key=lambda c: -(cl_pair == c).sum())}
        out["by_amplitude"][r] = {f"tercile {b + 1}": dict(n=int((abin == b).sum()), amp_median=float(np.median(amp[abin == b])),
                                                          within_2=float(np.mean(az[abin == b] < 2))) for b in range(3)}
    print(f"[{label}] D4: within 2 sigma by amplitude tercile (amplitude / sigma medians "
          + ", ".join(f"{v['amp_median']:.0f}" for v in out["by_amplitude"][key_rows[0]].values()) + ")", flush=True)
    for r in key_rows:
        print(f"[{label}]   {r:48s} " + "  ".join(f"{k} {v['within_2']:.3f}" for k, v in out["by_amplitude"][r].items())
              + " | by class " + ", ".join(f"{c} {v['within_2']:.2f}" for c, v in list(out["by_class"][r].items())[:8]), flush=True)
    if a.save_npz:
        # example curves: the last token of window A advanced over a grid of gaps, for the test stars with the most targets
        ex = {}
        n_t = np.bincount(np.searchsorted(np.unique(star[I]), star[I]))
        order_st = np.unique(star[I])[np.argsort(-n_t)][:8]
        grid = np.linspace(0.0, a.dt[1], 200)
        for e, o in enumerate(order_st):
            wA = np.bincount(A_[star[I] == o]).argmax()
            sel = win == wA
            i0 = np.flatnonzero(sel)[np.argmax(t[sel])]
            endA = wins["start"][wA] + window_days
            with torch.no_grad():
                pg = dyn.predict(torch.as_tensor(np.repeat(s[i0][None], len(grid), 0)).float().to(dev),
                                 torch.as_tensor(np.repeat(cls[wA][None], len(grid), 0)).float().to(dev),
                                 torch.full((len(grid),), f_star[o]).float().to(dev), torch.as_tensor(grid).float().to(dev)).cpu().numpy()
            cl = np.repeat(cls[wA][None], len(grid), 0)
            tt, mm, ss = t[sel], m[sel], sig[sel]
            after = (star == o) & (t > endA) & (t <= t[i0] + a.dt[1])
            ex[f"ex{e}_A_t"], ex[f"ex{e}_A_m"], ex[f"ex{e}_A_s"] = tt, mm, ss
            ex[f"ex{e}_B_t"], ex[f"ex{e}_B_m"], ex[f"ex{e}_B_s"] = t[after], m[after], sig[after]
            ex[f"ex{e}_grid_t"] = t[i0] + grid
            ex[f"ex{e}_pred_mlp"] = decoders["mlp"](pg, cl)
            ex[f"ex{e}_pred_ridge"] = decoders["ridge"](pg, cl)
            ex[f"ex{e}_four"] = fourier_forecast(tt - tt.min(), mm, ss, f_star[o], t[i0] + grid - tt.min(), a.harmonics)
            ex[f"ex{e}_info"] = np.array([o, f_star[o], endA, t[i0]])
        tag = label.split(" | ")[0]
        np.savez(a.save_npz.replace(".npz", f"_{tag}.npz"), dt=dt, m_j=yj, sig_j=sj, row_names=np.array(list(rows)),
                 yhat=np.stack([rows[k] for k in rows]), cls_pair=cl_pair, amp=amp, **ex)
        print("saved", a.save_npz.replace(".npz", f"_{tag}.npz"), flush=True)
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
    if "dec.0.weight" in dyn_sd:  # trained with --dyn-decode: its own magnitude decoder
        dyn.add_decoder(D, n_bands=dyn_sd["dec_band.weight"].shape[0] if "dec_band.weight" in dyn_sd else 0,
                        learn_sig=dyn_sd["dec.4.weight"].shape[0] == 2)
        dyn.to(dev)
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
        results[label] = score(tokens, wins, dyn, cfg.window, label, spec)
        results[label]["seconds"] = round(time.time() - t0)
if a.out:
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)
    print("wrote", a.out)
