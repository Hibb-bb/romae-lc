"""Does the frozen encoder's PER-TOKEN output know the phase of each observation?

The pooled-latent phase probe (project.phase_probe) sits at chance for every
encoder. Dennis's synthetic test bed (lcjepa.py, 2026-10-23) probes the token
states instead and finds the phase there, so this is the like-for-like check on
real stars. For every real token of a window the target is the observation's
phase in the star's cycle, ``psi = (t_abs / P - phi_ref) mod 1`` with ``P`` the
catalogue period and ``phi_ref`` the phase of maximum light of the star's folded
template (project.phase_probe.reference_phase). Probes predict
``(cos 2 pi psi, sin 2 pi psi)`` and report the R2 of each component: the sine
is readable from the observation's own magnitude, the cosine needs context
(brightening or fading), so the cosine column is the honest number.

Feature sets, as in the test bed's controls:
  raw        the token's own magnitude and error plus the window's mean, std,
             min and max magnitude (what a probe gets without any encoder)
  h          the encoder's token output
  h+cls      the token output and the window's CLS vector
  cls        the CLS alone (control: the pooled latent has no token phase)
  shuffled   h+cls with the targets permuted among the tokens of the same star
             (control: must sit at zero)
Ridge on all tokens, a small MLP on a subsample. Stars are split 70 / 30, so
the test tokens come from stars the probe never saw. ``--random-init`` adds the
first checkpoint's architecture with its layers re-initialised (untrained
encoder control).

    python -m project.eval.token_phase_probe --ckpts project/runs/maew_spec/mae.pt --random-init \
        --out project/results/token_phase/maew_spec.json
"""
import sys, argparse, json, time, numpy as np, torch
sys.path.insert(0, "/projects/bfrf/hibb/romae-lc")
from project.common import data_args_from, load_data, load_encoder, superclass
from project.cache_latents import WindowDataset, make_collate
from project.phase_probe import reference_phase
from torch.utils.data import DataLoader, Subset

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--data", default=None, help="data root; default = the first checkpoint's")
ap.add_argument("--n-stars", type=int, default=600)
ap.add_argument("--wins-per-star", type=int, default=6)
ap.add_argument("--fit", type=int, default=60000, help="tokens for the MLP fit and the MLP test")
ap.add_argument("--random-init", action="store_true", help="also probe the first checkpoint's architecture with re-initialised layers")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--label", default="")
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
t_start = time.time()

_, meta0 = load_encoder(a.ckpts[0], "cpu")
dargs = data_args_from(meta0.args)
if a.data:
    dargs.data = a.data
val = load_data(dargs, splits=("validation",))["validation"]
rng = np.random.default_rng(a.seed)
cand = [r for r in val if r.period is not None and np.isfinite(r.period) and r.period > 0]
rng.shuffle(cand)
recs, refs = [], []
for r in cand:
    ref = reference_phase(r, float(r.period))
    if np.isfinite(ref):
        recs.append(r)
        refs.append(ref)
    if len(recs) >= a.n_stars:
        break
refs = np.array(refs)
periods = np.array([float(r.period) for r in recs])
classes = np.array([superclass(r) for r in recs])
print(f"[{a.label}] {len(cand)} validation stars with a period, {len(recs)} with a template reference phase kept "
      f"({time.time() - t_start:.0f}s)", flush=True)
star_train = rng.random(len(recs)) < 0.7


def pick_windows(ds):
    """Up to wins_per_star random windows per star, as flat dataset indices."""
    off = ds.offsets[0]
    idx = []
    for i in range(len(recs)):
        lo, hi = int(off[i]), int(off[i + 1])
        if hi > lo:
            k = min(a.wins_per_star, hi - lo)
            idx.extend(sorted(rng.choice(np.arange(lo, hi), k, replace=False).tolist()))
    return idx


@torch.no_grad()
def embed(model, ds, spec, cfg, min_tokens, dyn=None):
    """Per real token: h, the window's cls, psi, raw features, star and window ids,
    and the dynamic state s when the checkpoint trained the add-on (project.dyn)."""
    ts = spec.tokenize["time_scale"]
    loader = DataLoader(Subset(ds, pick_windows(ds)), 128, shuffle=False, num_workers=0, collate_fn=make_collate(spec))
    H, CLS, PSI, RAW, STAR, WIN, S, PSIW = [], [], [], [], [], [], [], []
    n_win = 0
    for b in loader:
        tok = b["tokens"]
        x, pad = model.encode(tok.values.to(dev), tok.positions.to(dev), tok.pad_mask.to(dev))
        x = x.float()
        if dyn is not None:
            S.append(dyn.states(x[:, 1:])[~pad[:, 1:] & (b["n_tokens"].to(dev) >= min_tokens)[:, None]].cpu())
        real = ~pad[:, 1:] & (b["n_tokens"].to(dev) >= min_tokens)[:, None]  # windows below the checkpoint's minimum drop out
        t_days = (tok.positions[:, 0].to(dev).float() - 1.0) * ts
        obj = b["obj"].numpy()  # position in recs (the subset list is recs itself)
        P = torch.as_tensor(periods[obj], device=dev, dtype=torch.float64)
        ref = torch.as_tensor(refs[obj], device=dev, dtype=torch.float64)
        t_abs = b["start"].to(dev).double()[:, None] + t_days.double()
        psi = torch.remainder(t_abs / P[:, None] - ref[:, None], 1.0).float()
        # the "window clock": the phase counted from the WINDOW START, not from the star's reference. A token can get this
        # from its rotary position alone (time since the window start) once the frequency is known, with no light-curve content;
        # a state that scores high here and low on psi is a clock, not a phase (the add-on runs of 10-26)
        psi_w = torch.remainder(t_days.double() / P[:, None], 1.0).float()
        mag = tok.values[..., 0].to(dev).float()
        err = tok.values[..., 1].to(dev).float() if tok.values.shape[-1] > 1 else torch.zeros_like(mag)
        rf = real.float()
        n = rf.sum(1, keepdim=True).clamp(min=1)
        mean = (mag * rf).sum(1, keepdim=True) / n
        std = (((mag - mean) ** 2 * rf).sum(1, keepdim=True) / n).sqrt()
        big = torch.where(real, mag, torch.full_like(mag, float("inf")))
        small = torch.where(real, mag, torch.full_like(mag, float("-inf")))
        raw = torch.stack([mag, err, mean.expand_as(mag), std.expand_as(mag), big.min(1, keepdim=True).values.expand_as(mag),
                           small.max(1, keepdim=True).values.expand_as(mag)], -1)
        widx = (torch.arange(x.shape[0], device=dev) + n_win)[:, None].expand_as(mag)  # row of CLS for every token
        n_win += x.shape[0]
        H.append(x[:, 1:][real].half().cpu()); CLS.append(x[:, 0].half().cpu())
        PSI.append(psi[real].cpu()); RAW.append(raw[real].cpu()); PSIW.append(psi_w[real].cpu())
        STAR.append(torch.as_tensor(obj, device=dev)[:, None].expand_as(mag)[real].cpu())
        WIN.append(widx[real].cpu())
    return (torch.cat(H).numpy(), torch.cat(CLS).numpy(), torch.cat(PSI).numpy(), torch.cat(RAW).numpy(),
            torch.cat(STAR).numpy(), torch.cat(WIN).numpy(), torch.cat(S).numpy() if S else None, torch.cat(PSIW).numpy())


def r2(y, p):
    """R2 of every column: [cos, sin]."""
    return [float(v) for v in 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)]


def probes(h, cls_w, psi, raw, star, win, s=None, psi_w=None):
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    prs = np.random.default_rng(a.seed + 1)
    Y = np.stack([np.cos(2 * np.pi * psi), np.sin(2 * np.pi * psi)], 1)
    tr, te = star_train[star], ~star_train[star]
    c_tok = cls_w[win].astype(np.float32)
    h = h.astype(np.float32)
    i = prs.choice(np.flatnonzero(tr), min(a.fit, int(tr.sum())), replace=False)
    j = prs.choice(np.flatnonzero(te), min(a.fit, int(te.sum())), replace=False)

    def ridge(X, y=Y):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 4, 7)).fit(sc.transform(X[tr]), y[tr])
        return r2(y[te], m.predict(sc.transform(X[te])))

    def mlp(X, y=Y, with_pred=False):
        sc = StandardScaler().fit(X[i])
        m = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=60, early_stopping=True, random_state=0).fit(sc.transform(X[i]), y[i])
        p = m.predict(sc.transform(X[j]))
        return (r2(y[j], p), p) if with_pred else r2(y[j], p)

    out = {"n_tokens": int(len(psi)), "n_test_tokens": int(te.sum()), "n_stars": int(len(recs))}
    out["raw | mlp"] = mlp(raw)
    out["h | ridge"] = ridge(h)
    out["h | mlp"] = mlp(h)
    hc = np.concatenate([h, c_tok], 1)
    out["h+cls | mlp"], p_hc = mlp(hc, with_pred=True)
    out["cls | mlp (control)"] = mlp(c_tok)
    if s is not None:  # the trained dynamic state of the add-on (project.dyn): 2K numbers per token
        s = s.astype(np.float32)
        out["s | ridge"] = ridge(s)
        out["s | mlp"] = mlp(s)
        out["s+cls | mlp"] = mlp(np.concatenate([s, c_tok], 1))
    if psi_w is not None:  # the window clock (see embed): high here and low on psi = a clock, not a phase
        Yw = np.stack([np.cos(2 * np.pi * psi_w), np.sin(2 * np.pi * psi_w)], 1)
        out["h | ridge -> window clock (control)"] = ridge(h, Yw)
        if s is not None:
            out["s | ridge -> window clock (control)"] = ridge(s, Yw)
    # shuffle control: targets permuted among the tokens of the same star
    Ysh = Y.copy()
    for s in np.unique(star):
        rows = np.flatnonzero(star == s)
        Ysh[rows] = Y[prs.permutation(rows)]
    out["h+cls shuffled | mlp (control)"] = mlp(hc, Ysh)
    # the wrapped phase error of the h+cls MLP read-out
    ph = np.mod(np.arctan2(p_hc[:, 1], p_hc[:, 0]) / (2 * np.pi), 1.0)
    e = np.abs(np.mod(ph - psi[j] + 0.5, 1.0) - 0.5)
    out["h+cls | median phase error (cycles)"] = float(np.median(e))
    out["h+cls | within 0.1 cycles (chance 0.2)"] = float(np.mean(e < 0.1))
    # cosine R2 by class on the MLP test tokens
    by = {}
    for c in np.unique(classes[star[j]]):
        m = classes[star[j]] == c
        if m.sum() >= 500:
            by[c] = round(r2(Y[j][m], p_hc[m])[0], 3)
    out["h+cls | cos R2 by class"] = by
    return out


results = {}
for k, ck in enumerate(a.ckpts):
    enc, meta = load_encoder(ck, dev)
    model = enc.backbone.model
    cfg, spec = meta.cfg, meta.spec
    ds = WindowDataset([("validation", recs, list(range(len(recs))))], cfg, 0.25, cfg.max_tokens, 0, 1, 0.5)
    dyn = None
    dyn_sd = torch.load(ck, map_location="cpu", weights_only=False).get("aux", {}).get("dyn")
    if dyn_sd is not None:
        from project.dyn import DynHead

        dyn = DynHead(dyn_sd["s_head.weight"].shape[1], dyn_sd["s_head.weight"].shape[0] // 2).to(dev)
        dyn_sd = {k: v for k, v in dyn_sd.items() if k in dyn.state_dict()}  # a HamHead's flow is not needed: the probe reads the states
        dyn.load_state_dict(dyn_sd)
        dyn.eval()
        print(f"[{a.label}] {ck}: dynamic-state head found (K = {dyn.K}), its state s is probed too", flush=True)
    variants = [(ck, model, dyn)]
    if k == 0 and a.random_init:
        import copy
        m2 = copy.deepcopy(model)
        n_re = 0
        for mod in m2.modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters(); n_re += 1
        variants.append((f"{ck} [random init: {n_re} layers re-initialised]", m2.to(dev).eval(), None))
    for name, m, dh in variants:
        t0 = time.time()
        feats = embed(m, ds, spec, cfg, cfg.min_tokens, dh)
        res = probes(*feats)
        res["seconds"] = round(time.time() - t0)
        results[name] = res
        fmt = lambda v: "/".join(f"{x:.3f}" for x in v) if isinstance(v, list) else v
        print(f"[{a.label}] {name}: " + "  ".join(f"{kk}={fmt(v)}" for kk, v in res.items() if kk not in ("h+cls | cos R2 by class",)), flush=True)
        print(f"[{a.label}]   cos R2 by class: {res['h+cls | cos R2 by class']}", flush=True)
if a.out:
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
