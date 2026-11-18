"""Test card 2: the period. Two numbers, from two different inputs.

1. Log-period probe (needs the latent caches): the star probe of ``project.eval.card_class`` with one
   output, log10 of the catalogue period. Fit on train, epoch picked on validation (R2), scored on TEST;
   three seeds. Scores: R2, share within 1 % and 10 % of the period, median error in dex.

2. Spectral-peak hit rate (needs the checkpoint): the period the encoder's own spectral layer points at,
   with no fit. Per window the local-centroid frequency; per star the median over its windows with at
   least 32 tokens (window stride 0.25). Scored on VALIDATION stars with a catalogue period, the same
   stars for every checkpoint. Strict = within the tolerance of the truth; alias-tolerant also accepts
   the double and the half. Tolerances 10 %, 1 %, 0.1 %, 0.01 %.

The catalogue period is only the target here. The encoder never sees it.

    python -m project.eval.card_period --ckpts project/runs/R/mae.pt --n 100000 --out peak.json
    python -m project.eval.card_period --latents L.pt --test-latents L_test.pt --out logp.json
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from project.checkpoint import load_encoder
from project.data import data_args_from, load_data
from project.eval import card_class
from project.eval.common_eval import eval_device, period_hits, spectral_layers, window_dataset, window_loader, write_json
from romae_lc.model import spectral_frequency


def logp_probe(args, inputs=None):
    """The star probe on log10 period. ``args`` as ``card_class.parse_args`` gives them."""
    return card_class.run(args, tasks=("logp",), inputs=inputs)


def peak_stars(ckpt, data=None, n=1000, split="validation"):
    """The stars the peak test scores: a seeded draw of ``n`` stars of ``split`` that have a catalogue
    period (``n`` above their number = all of them). Returns ``(index, idx, recs, n_all)``: each star's
    place in the full split, its place among the stars with a period, the records, and how many stars
    of the split have a period."""
    _, meta = load_encoder(ckpt, "cpu")
    dargs = data_args_from(meta.args)
    if data:
        dargs.data = data
    stars = load_data(dargs, splits=(split,))[split]
    index = np.array([i for i, r in enumerate(stars) if r.period is not None and np.isfinite(r.period)])  # place in the full split
    stars = [stars[i] for i in index]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(stars), min(n, len(stars)), replace=False))
    return index[idx], idx, [stars[i] for i in idx], len(stars)


@torch.no_grad()
def peak_periods(ckpt, recs, idx, split="validation", dev=None, label=""):
    """One period per star from the spectral layer of ``ckpt``: NaN for a star with no window of at least
    32 tokens. Tokens are built with the checkpoint's own spec and window."""
    dev = dev or eval_device()
    enc, meta = load_encoder(ckpt, dev)
    cfg, spec = meta.cfg, meta.spec
    ds = window_dataset(split, recs, cfg, 0.25, idx)
    loader = window_loader(ds, spec, 256)
    off = ds.offsets[0]
    print(f"[{label}] {ckpt}: window {cfg.window} d, max_tokens {cfg.max_tokens}, time_scale {spec.tokenize['time_scale']:.5f}, {ds.total} windows", flush=True)
    f_all = torch.empty(ds.total)
    n_tok = torch.empty(ds.total, dtype=torch.long)
    for batch in loader:
        enc.encode([batch["tokens"].to(dev)])
        f_all[batch["widx"]] = spectral_frequency(spectral_layers(enc, ran=True)[0], None, "local").float().cpu()
        n_tok[batch["widx"]] = batch["n_tokens"]
    p_hat = np.full(len(recs), np.nan)
    for i in range(len(recs)):
        lo, hi = int(off[i]), int(off[i + 1])
        ok = n_tok[lo:hi] >= 32
        if ok.sum():
            p_hat[i] = np.median((1.0 / f_all[lo:hi][ok]).numpy())
    return p_hat, meta.step


def peak_hits(ckpts, data=None, n=1000, split="validation", label="", save=None, dev=None):
    """The spectral-peak hit rates of every checkpoint on the same stars (picked with the first
    checkpoint's data arguments). Returns ``{ckpt: dict(split, step, n, strict, alias)}``."""
    dev = dev or eval_device()
    index, idx, recs, n_all = peak_stars(ckpts[0], data, n, split)
    p_true = np.array([r.period for r in recs], float)
    print(f"[{label}] {n_all} {split} stars with a period, {len(recs)} probed, device {dev}", flush=True)
    saved = dict(index=index, p_true=p_true, super=np.array([str(r.meta.get("superclass_str", "?")) for r in recs]))
    out = {}
    for ck in ckpts:
        p_hat, step = peak_periods(ck, recs, idx, split, dev, label)
        n_ok, strict, alias = period_hits(p_hat, p_true)
        fmt = lambda v: "/".join(f"{x:.3f}" for x in v)
        print(f"[{label}] {ck} (step {step}) vs table period: n={n_ok}  strict within 10/1/0.1/0.01 % = {fmt(strict)}  |  alias-tolerant {fmt(alias)}", flush=True)
        saved["p_hat:" + ck] = p_hat
        out[ck] = dict(split=split, step=step, n_stars=len(recs), n=n_ok, tolerances=[0.1, 0.01, 0.001, 1e-4], strict=strict, alias=alias)
    if save:
        np.savez(save, **saved)
        print("saved", save, flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("spectral-peak hit rate")
    g.add_argument("--ckpts", nargs="+", default=None, help="checkpoints (mae.pt); all are scored on the same stars")
    g.add_argument("--data", default=None, help="data root; default = the first checkpoint's")
    g.add_argument("--n", type=int, default=1000, help="stars probed (the jobs use 100000 = all)")
    g.add_argument("--split", default="validation", help="the split scored (the card uses validation)")
    g.add_argument("--label", default="")
    g.add_argument("--save", default=None, help="npz with every probed star's index in the split, true period and each checkpoint's estimate (NaN = none)")
    g = p.add_argument_group("log-period probe")
    g.add_argument("--latents", default=None, help="cache with the train and validation splits (project.eval.encode)")
    g.add_argument("--test-latents", default=None, help="cache with the test split")
    card_class.add_probe_args(g)
    p.add_argument("--out", default=None, help="JSON with the results")
    a = p.parse_args(argv)
    if not a.ckpts and not a.latents:
        p.error("give --ckpts (spectral peak) or --latents and --test-latents (log-period probe), or both")
    if bool(a.latents) != bool(a.test_latents):
        p.error("--latents and --test-latents go together")
    results = {}
    if a.ckpts:
        results["peak"] = peak_hits(a.ckpts, a.data, a.n, a.split, a.label, a.save)
    if a.latents:
        results["logp"] = logp_probe(a)
    if a.out:
        write_json(a.out, results)
        print("wrote", a.out)
    return results


if __name__ == "__main__":
    main()
