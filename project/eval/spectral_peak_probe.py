"""Spectral-peak period hit rates on held-out stars, the same stars for every checkpoint:
local-centroid frequency per window, median over a star's windows with >= 32 tokens.
Strict = within the tolerance of the truth; alias-tolerant also accepts the double and the half."""
import sys, argparse, csv, numpy as np, torch
sys.path.insert(0, "/projects/bfrf/hibb/romae-lc")
from project.common import data_args_from, load_data, load_encoder
from project.cache_latents import WindowDataset, make_collate
from romae_lc.model import spectral_frequency
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--data", default=None, help="data root; default = the first checkpoint's")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--chen-csv", default=None, help="periods.csv of period_chen: adds Chen's sharpened period as a second truth")
ap.add_argument("--label", default="")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
_, meta = load_encoder(a.ckpts[0], "cpu")
cfg, spec = meta.cfg, meta.spec
dargs = data_args_from(meta.args)
if a.data:
    dargs.data = a.data
val = load_data(dargs, splits=("validation",))["validation"]
val = [r for r in val if r.period is not None and np.isfinite(r.period)]
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
ds = WindowDataset([("validation", recs, list(idx))], cfg, 0.25, cfg.max_tokens, 0, 1, 0.5)
loader = DataLoader(ds, 256, shuffle=False, num_workers=0, collate_fn=make_collate(spec))
off = ds.offsets[0]
TOL = (0.1, 0.01, 0.001, 1e-4)

def hits(p_hat, p):
    ok = np.isfinite(p) & np.isfinite(p_hat) & (p > 0)
    r = p_hat[ok] / p[ok]
    rel = np.abs(r - 1)
    rel_a = np.minimum(rel, np.minimum(np.abs(r / 2 - 1), np.abs(2 * r - 1)))
    return int(ok.sum()), [float(np.mean(rel < t)) for t in TOL], [float(np.mean(rel_a < t)) for t in TOL]

for ck in a.ckpts:
    step = torch.load(ck, map_location="cpu", weights_only=False).get("step", "?")
    enc, _ = load_encoder(ck, dev)
    model = enc.backbone.model if hasattr(enc.backbone, "model") else enc.backbone
    f_all = torch.empty(ds.total)
    n_tok = torch.empty(ds.total, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            enc.encode([batch["tokens"].to(dev)])
            f_all[batch["widx"]] = spectral_frequency(model.spectral, None, "local").float().cpu()
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
