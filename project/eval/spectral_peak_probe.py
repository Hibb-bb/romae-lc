"""Spectral-peak period hit rates on held-out stars, the same stars for every checkpoint:
local-centroid frequency per window, median over a star's windows with >= 32 tokens.
Strict = within the tolerance of the truth; alias-tolerant also accepts the double and the half."""
import argparse, csv, numpy as np, torch
from project.checkpoint import load_encoder
from project.data import data_args_from, load_data
from project.windows import make_collate, WindowDataset
from romae_lc.model import spectral_frequency
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--data", default=None, help="data root; default = the first checkpoint's")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--chen-csv", default=None, help="periods.csv of period_chen: adds Chen's sharpened period as a second truth")
ap.add_argument("--label", default="")
ap.add_argument("--save", default=None, help="npz with every probed star's validation index, true period and each checkpoint's estimate (NaN = none)")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
_, meta = load_encoder(a.ckpts[0], "cpu")
dargs = data_args_from(meta.args)
if a.data:
    dargs.data = a.data
val = load_data(dargs, splits=("validation",))["validation"]
val_index = np.array([i for i, r in enumerate(val) if r.period is not None and np.isfinite(r.period)])  # place in the full split
val = [val[i] for i in val_index]
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(val), min(a.n, len(val)), replace=False))
recs = [val[i] for i in idx]
p_true = np.array([r.period for r in recs], float)
truths = {"table period": p_true}
if a.chen_csv:
    m = {}
    with open(a.chen_csv) as f:
        for row in csv.DictReader(f):
            try:
                m[row["sourceid"]] = float(row["p_chen_sharp"])
            except ValueError:
                pass
    truths["Chen sharpened"] = np.array([m.get(str(r.meta.get("id")), np.nan) for r in recs])
print(f"[{a.label}] {len(val)} validation stars with a period, {len(recs)} probed, device {dev}", flush=True)
TOL = (0.1, 0.01, 0.001, 1e-4)

def hits(p_hat, p):
    ok = np.isfinite(p) & np.isfinite(p_hat) & (p > 0)
    r = p_hat[ok] / p[ok]
    rel = np.abs(r - 1)
    rel_a = np.minimum(rel, np.minimum(np.abs(r / 2 - 1), np.abs(2 * r - 1)))
    return int(ok.sum()), [float(np.mean(rel < t)) for t in TOL], [float(np.mean(rel_a < t)) for t in TOL]

saved = dict(index=val_index[idx], p_true=p_true, super=np.array([str(r.meta.get("superclass_str", "?")) for r in recs]))
for ck in a.ckpts:
    step = torch.load(ck, map_location="cpu", weights_only=False).get("step", "?")
    enc, meta = load_encoder(ck, dev)
    # every checkpoint gets tokens built with its own spec (time scale, error stats) and
    # its own window; the first version used the first checkpoint's for all of them
    cfg, spec = meta.cfg, meta.spec
    ds = WindowDataset([("validation", recs, list(idx))], cfg, 0.25, cfg.max_tokens, 0)
    loader = DataLoader(ds, 256, shuffle=False, num_workers=0, collate_fn=make_collate(spec))
    off = ds.offsets[0]
    print(f"[{a.label}] {ck}: window {cfg.window} d, max_tokens {cfg.max_tokens}, time_scale {spec.tokenize['time_scale']:.5f}, {ds.total} windows", flush=True)
    model = enc.backbone.model if hasattr(enc.backbone, "model") else enc.backbone
    f_all = torch.empty(ds.total)
    n_tok = torch.empty(ds.total, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            enc.encode([batch["tokens"].to(dev)])
            # the spectral layer that ran (a JEPA checkpoint wraps a copy of the encoder, so model.spectral may be idle)
            ran = [m for m in enc.modules() if type(m).__name__ == "SpectralLayer" and getattr(m, "last_logits", None) is not None]
            f_all[batch["widx"]] = spectral_frequency(ran[0], None, "local").float().cpu()
            n_tok[batch["widx"]] = batch["n_tokens"]
    p_hat = np.full(len(recs), np.nan)
    for i in range(len(recs)):
        lo, hi = int(off[i]), int(off[i + 1])
        ok = n_tok[lo:hi] >= 32
        if ok.sum():
            p_hat[i] = np.median((1.0 / f_all[lo:hi][ok]).numpy())
    for name, p in truths.items():
        n, s, al = hits(p_hat, p)
        fmt = lambda v: "/".join(f"{x:.3f}" for x in v)
        print(f"[{a.label}] {ck} (step {step}) vs {name}: n={n}  strict within 10/1/0.1/0.01 % = {fmt(s)}  |  alias-tolerant {fmt(al)}", flush=True)
    saved["p_hat:" + ck] = p_hat
if a.save:
    np.savez(a.save, **saved)
    print("saved", a.save, flush=True)
