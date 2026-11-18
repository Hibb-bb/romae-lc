"""Test card 3: does the frozen encoder's PER-TOKEN output know the phase of each observation?

For every real token of a window the target is the observation's phase in the star's cycle,
``psi = (t_abs / P - phi_ref) mod 1`` with ``P`` the catalogue period and ``phi_ref`` the phase of
maximum light of the star's folded template (project.phase_probe.reference_phase). Probes predict
``(cos 2 pi psi, sin 2 pi psi)`` and report the R2 of each component: the sine is readable from the
observation's own magnitude, the cosine needs context (brightening or fading), so the cosine column is
the honest number. The catalogue period is only the target; the encoder never sees it.

Feature sets:
  raw        the token's own magnitude and error plus the window's mean, std,
             min and max magnitude (what a probe gets without any encoder)
  h          the encoder's token output
  h+cls      the token output and the window's CLS vector
  cls        the CLS alone (control: the pooled latent has no token phase)
  shuffled   h+cls with the targets permuted among the tokens of the same star
             (control: must sit at zero)
Ridge on all tokens, a small MLP on a subsample.

Split: VALIDATION stars only (``--n-stars`` of them with a period and a template reference phase),
split 70 / 30 by star, so the scored tokens come from stars the probe never saw. ``--random-init``
adds the first checkpoint's architecture with its layers re-initialised (untrained-encoder control).

    python -m project.eval.card_phase --ckpts project/runs/R/mae.pt --n-stars 600 --out project/results/token_phase/R.json
"""
from __future__ import annotations

import argparse, copy, time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from project.checkpoint import load_encoder
from project.data import data_args_from, load_data, superclass
from project.eval.common_eval import eval_device, window_dataset, write_json
from project.phase_probe import reference_phase
from project.windows import make_collate


@dataclass
class Stars:
    """The stars of the phase test and what the probes need to know about them."""

    recs: list
    refs: np.ndarray  # phase of maximum light of each star's template
    periods: np.ndarray
    classes: np.ndarray
    train: np.ndarray  # True = the star's tokens fit the probes; False = they are scored
    n_candidates: int


def pick_stars(ckpt, data, n_stars, rng, split="validation"):
    """The first ``n_stars`` stars, in a seeded random order, of ``split`` that have a catalogue period
    and a template reference phase; then the 70 / 30 split by star. Data arguments come from ``ckpt``."""
    _, meta0 = load_encoder(ckpt, "cpu")
    dargs = data_args_from(meta0.args)
    if data:
        dargs.data = data
    stars = load_data(dargs, splits=(split,))[split]
    cand = [r for r in stars if r.period is not None and np.isfinite(r.period) and r.period > 0]
    rng.shuffle(cand)
    recs, refs = [], []
    for r in cand:
        ref = reference_phase(r, float(r.period))
        if np.isfinite(ref):
            recs.append(r)
            refs.append(ref)
        if len(recs) >= n_stars:
            break
    periods = np.array([float(r.period) for r in recs])
    classes = np.array([superclass(r) for r in recs])
    return Stars(recs, np.array(refs), periods, classes, rng.random(len(recs)) < 0.7, len(cand))


def pick_windows(ds, n_stars, wins_per_star, rng):
    """Up to wins_per_star random windows per star, as flat dataset indices."""
    off = ds.offsets[0]
    idx = []
    for i in range(n_stars):
        lo, hi = int(off[i]), int(off[i + 1])
        if hi > lo:
            k = min(wins_per_star, hi - lo)
            idx.extend(sorted(rng.choice(np.arange(lo, hi), k, replace=False).tolist()))
    return idx


@torch.no_grad()
def embed(model, ds, spec, min_tokens, stars, wins_per_star, rng, dev):
    """Per real token: h, the window's cls, psi, raw features, star and window ids, and the window clock."""
    ts = spec.tokenize["time_scale"]
    loader = DataLoader(Subset(ds, pick_windows(ds, len(stars.recs), wins_per_star, rng)), 128, shuffle=False, num_workers=0, collate_fn=make_collate(spec))
    H, CLS, PSI, RAW, STAR, WIN, PSIW = [], [], [], [], [], [], []
    n_win = 0
    for b in loader:
        tok = b["tokens"]
        x, pad = model.encode(tok.values.to(dev), tok.positions.to(dev), tok.pad_mask.to(dev))
        x = x.float()
        real = ~pad[:, 1:] & (b["n_tokens"].to(dev) >= min_tokens)[:, None]  # windows below the checkpoint's minimum drop out
        t_days = (tok.positions[:, 0].to(dev).float() - 1.0) * ts
        obj = b["obj"].numpy()  # position in recs (the subset list is recs itself)
        P = torch.as_tensor(stars.periods[obj], device=dev, dtype=torch.float64)
        ref = torch.as_tensor(stars.refs[obj], device=dev, dtype=torch.float64)
        t_abs = b["start"].to(dev).double()[:, None] + t_days.double()
        psi = torch.remainder(t_abs / P[:, None] - ref[:, None], 1.0).float()
        # the "window clock": the phase counted from the WINDOW START, not from the star's reference. A token can get this
        # from its rotary position alone (time since the window start) once the frequency is known, with no light-curve content;
        # a state that scores high here and low on psi is a clock, not a phase
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
            torch.cat(STAR).numpy(), torch.cat(WIN).numpy(), torch.cat(PSIW).numpy())


def r2(y, p):
    """R2 of every column: [cos, sin]."""
    return [float(v) for v in 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)]


def probes(h, cls_w, psi, raw, star, win, psi_w, stars, fit, seed):
    """Fit and score every probe. ``fit``: tokens for the MLP fit and for the MLP test."""
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    prs = np.random.default_rng(seed + 1)
    Y = np.stack([np.cos(2 * np.pi * psi), np.sin(2 * np.pi * psi)], 1)
    tr, te = stars.train[star], ~stars.train[star]
    c_tok = cls_w[win].astype(np.float32)
    h = h.astype(np.float32)
    i = prs.choice(np.flatnonzero(tr), min(fit, int(tr.sum())), replace=False)
    j = prs.choice(np.flatnonzero(te), min(fit, int(te.sum())), replace=False)

    def ridge(X, y=Y):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 4, 7)).fit(sc.transform(X[tr]), y[tr])
        return r2(y[te], m.predict(sc.transform(X[te])))

    def mlp(X, y=Y, with_pred=False):
        sc = StandardScaler().fit(X[i])
        m = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=60, early_stopping=True, random_state=0).fit(sc.transform(X[i]), y[i])
        p = m.predict(sc.transform(X[j]))
        return (r2(y[j], p), p) if with_pred else r2(y[j], p)

    out = {"n_tokens": int(len(psi)), "n_test_tokens": int(te.sum()), "n_stars": int(len(stars.recs))}
    out["raw | mlp"] = mlp(raw)
    out["h | ridge"] = ridge(h)
    out["h | mlp"] = mlp(h)
    hc = np.concatenate([h, c_tok], 1)
    out["h+cls | mlp"], p_hc = mlp(hc, with_pred=True)
    out["cls | mlp (control)"] = mlp(c_tok)
    if psi_w is not None:  # the window clock (see embed): high here and low on psi = a clock, not a phase
        Yw = np.stack([np.cos(2 * np.pi * psi_w), np.sin(2 * np.pi * psi_w)], 1)
        out["h | ridge -> window clock (control)"] = ridge(h, Yw)
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
    for c in np.unique(stars.classes[star[j]]):
        m = stars.classes[star[j]] == c
        if m.sum() >= 500:
            by[c] = round(r2(Y[j][m], p_hc[m])[0], 3)
    out["h+cls | cos R2 by class"] = by
    return out


def run(ckpts, data=None, n_stars=600, wins_per_star=6, fit=60000, random_init=False, seed=0, label="", split="validation", dev=None):
    """The phase probes of every checkpoint on the same stars. Returns ``{name: scores}``; the name is
    the checkpoint path (plus a note for the random-init control)."""
    dev = dev or eval_device()
    t_start = time.time()
    rng = np.random.default_rng(seed)
    stars = pick_stars(ckpts[0], data, n_stars, rng, split)
    print(f"[{label}] {stars.n_candidates} {split} stars with a period, {len(stars.recs)} with a template reference phase kept "
          f"({time.time() - t_start:.0f}s)", flush=True)
    results = {}
    for k, ck in enumerate(ckpts):
        enc, meta = load_encoder(ck, dev)
        model = enc.backbone.model
        cfg, spec = meta.cfg, meta.spec
        ds = window_dataset(split, stars.recs, cfg, 0.25)
        variants = [(ck, model)]
        if k == 0 and random_init:
            m2 = copy.deepcopy(model)
            n_re = 0
            for mod in m2.modules():
                if hasattr(mod, "reset_parameters"):
                    mod.reset_parameters(); n_re += 1
            variants.append((f"{ck} [random init: {n_re} layers re-initialised]", m2.to(dev).eval()))
        for name, m in variants:
            t0 = time.time()
            feats = embed(m, ds, spec, cfg.min_tokens, stars, wins_per_star, rng, dev)
            res = probes(*feats, stars, fit, seed)
            res["seconds"] = round(time.time() - t0)
            results[name] = res
            fmt = lambda v: "/".join(f"{x:.3f}" for x in v) if isinstance(v, list) else v
            print(f"[{label}] {name}: " + "  ".join(f"{kk}={fmt(v)}" for kk, v in res.items() if kk not in ("h+cls | cos R2 by class",)), flush=True)
            print(f"[{label}]   cos R2 by class: {res['h+cls | cos R2 by class']}", flush=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data", default=None, help="data root; default = the first checkpoint's")
    ap.add_argument("--n-stars", type=int, default=600)
    ap.add_argument("--wins-per-star", type=int, default=6)
    ap.add_argument("--fit", type=int, default=60000, help="tokens for the MLP fit and the MLP test")
    ap.add_argument("--random-init", action="store_true", help="also probe the first checkpoint's architecture with re-initialised layers")
    ap.add_argument("--split", default="validation", help="the split the stars come from (the card uses validation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    results = run(a.ckpts, a.data, a.n_stars, a.wins_per_star, a.fit, a.random_init, a.seed, a.label, a.split)
    if a.out:
        write_json(a.out, results)
        print(f"wrote {a.out}")
    return results


if __name__ == "__main__":
    main()
