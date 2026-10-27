"""Star-level attentive probe on frozen window vectors: learned queries attend over a
star's window latents (instead of averaging them), then a small head predicts the fine
class (13 classes, the anomaly classes left out) or log10 of the catalogue period.

Control: the same head on the plain mean of the window latents ("mean"), so the table
separates what attention adds from what a trained head adds; the random forest on the
mean (project.eval.classify_rf) is the reference number. Splits as classify_rf: fit on
train, pick the epoch on validation (macro F1 / R2), score on test; three seeds.

    python -m project.eval.attn_probe --latents L.pt --test-latents L_test.pt --out DIR
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from project.common import ANOMALY_CLASSES, data_args_from, load_data
from project.period_probe import object_table


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latents", required=True)
    p.add_argument("--test-latents", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", nargs="+", default=["class", "logp"])
    p.add_argument("--pools", nargs="+", default=["attn", "mean"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 200])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--queries", type=int, default=4)
    p.add_argument("--max-windows", type=int, default=64, help="windows drawn per star and step in training (all of them at evaluation)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args(argv)


class StarProbe(nn.Module):
    """Window latents [B, W, D] (pad True = no window) -> [B, out]."""

    def __init__(self, d_in, n_out, hidden=256, queries=4, pool="attn", dropout=0.1):
        super().__init__()
        self.pool = pool
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.LayerNorm(hidden))
        if pool == "attn":
            self.q = nn.Parameter(torch.randn(queries, hidden) * 0.02)
            self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=dropout)
            d_pool = queries * hidden
        else:
            d_pool = hidden
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_pool, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, z, pad):
        h = self.inp(z)
        if self.pool == "attn":
            y, _ = self.attn(self.q.expand(h.shape[0], -1, -1), h, h, key_padding_mask=pad, need_weights=False)
            v = y.reshape(h.shape[0], -1)
        else:
            real = (~pad).float()[..., None]
            v = (h * real).sum(1) / real.sum(1).clamp(min=1)
        return self.head(v)


def batches(Z, rows, idx, max_w, rng, train):
    """Pad the window sets of stars ``idx``; in training at most ``max_w`` random windows per star."""
    sets = []
    for i in idx:
        r = rows[i]
        if train and len(r) > max_w:
            r = rng.choice(r, max_w, replace=False)
        sets.append(r)
    W = max(len(r) for r in sets)
    z = torch.zeros(len(idx), W, Z.shape[1])
    pad = torch.ones(len(idx), W, dtype=torch.bool)
    for k, r in enumerate(sets):
        z[k, : len(r)] = Z[torch.as_tensor(np.asarray(r))]
        pad[k, : len(r)] = False
    return z, pad


def scores(task, pred, y, labels=None):
    if task == "class":
        from sklearn.metrics import accuracy_score, f1_score
        p = pred.argmax(1)
        return dict(macro_f1=float(f1_score(y, p, average="macro", labels=list(range(len(labels))), zero_division=0)), acc=float(accuracy_score(y, p)))
    p = pred[:, 0]
    rel = np.abs(10 ** (p - y) - 1)
    return dict(r2=float(1 - np.mean((p - y) ** 2) / np.var(y)), rec1=float(np.mean(rel < 0.01)), rec10=float(np.mean(rel < 0.1)),
                med_dex=float(np.median(np.abs(p - y))))


def run(args):
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.latents, map_location="cpu", weights_only=False)
    tcache = torch.load(args.test_latents, map_location="cpu", weights_only=False)
    meta = cache["meta"]
    ck = torch.load(meta["ckpt"], map_location="cpu", weights_only=False)
    data = load_data(data_args_from(ck["args"]), splits=("train", "validation", "test"))
    bands = sorted({int(b) for r in data["train"][:200] for b in np.unique(r.band)})
    caches = {"train": cache, "validation": cache, "test": tcache}
    tabs = {s: object_table(caches[s], s, data[s], bands, int(meta["min_tokens"])) for s in ("train", "validation", "test")}
    labels = sorted(set(tabs["train"]["fine"].tolist()) - set(ANOMALY_CLASSES))
    lab = {l: i for i, l in enumerate(labels)}
    # window latents standardised with the train statistics
    Ztr = cache["z"].float()
    tr_rows = np.concatenate([np.asarray(r) for r in tabs["train"]["rows"]])
    mu, sd = Ztr[tr_rows].mean(0), Ztr[tr_rows].std(0) + 1e-6
    Z = {"train": (Ztr - mu) / sd, "test": (tcache["z"].float() - mu) / sd}
    Z["validation"] = Z["train"]
    print(f"loaded in {time.time() - t0:.0f}s; {len(labels)} classes; stars {[(s, len(t['fine'])) for s, t in tabs.items()]}; device {dev}", flush=True)
    results = {"latents": args.latents, "labels": labels}
    for task in args.tasks:
        sel, ys = {}, {}
        for s, t in tabs.items():
            if task == "class":
                keep = np.isin(t["fine"], labels)
                ys[s] = np.array([lab.get(c, -1) for c in t["fine"]])
            else:
                keep = np.isfinite(t["logp"])
                ys[s] = t["logp"].astype(np.float32)
            sel[s] = np.flatnonzero(keep)
        n_out = len(labels) if task == "class" else 1
        if task == "class":
            cnt = np.bincount(ys["train"][sel["train"]], minlength=n_out).astype(float)
            cw = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float32, device=dev)  # sqrt inverse frequency
            cw = cw / cw.mean()
        else:
            ymu, ysd = float(ys["train"][sel["train"]].mean()), float(ys["train"][sel["train"]].std())
        for pool in args.pools:
            per_seed = []
            for seed in args.seeds:
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                net = StarProbe(Z["train"].shape[1], n_out, args.hidden, args.queries, pool, args.dropout).to(dev)
                opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
                n_tr = len(sel["train"])
                steps = args.epochs * ((n_tr + args.batch_size - 1) // args.batch_size)
                sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)

                def predict(split):
                    net.eval()
                    outs = []
                    with torch.no_grad():
                        for k in range(0, len(sel[split]), 512):
                            idx = sel[split][k : k + 512]
                            z, pad = batches(Z[split], tabs[split]["rows"], idx, 0, rng, False)
                            outs.append(net(z.to(dev), pad.to(dev)).float().cpu())
                    net.train()
                    o = torch.cat(outs).numpy()
                    return o if task == "class" else o * ysd + ymu

                best, best_state = -np.inf, None
                for ep in range(args.epochs):
                    perm = rng.permutation(sel["train"])
                    for k in range(0, n_tr, args.batch_size):
                        idx = perm[k : k + args.batch_size]
                        z, pad = batches(Z["train"], tabs["train"]["rows"], idx, args.max_windows, rng, True)
                        o = net(z.to(dev), pad.to(dev))
                        y = torch.as_tensor(ys["train"][idx], device=dev)
                        loss = F.cross_entropy(o, y.long(), weight=cw) if task == "class" else F.smooth_l1_loss(o[:, 0], (y.float() - ymu) / ysd, beta=0.1)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        opt.step()
                        sch.step()
                    v = scores(task, predict("validation"), ys["validation"][sel["validation"]], labels)
                    key = v["macro_f1"] if task == "class" else v["r2"]
                    if key > best:
                        best, best_state = key, {k_: t_.detach().clone() for k_, t_ in net.state_dict().items()}
                net.load_state_dict(best_state)
                te = scores(task, predict("test"), ys["test"][sel["test"]], labels)
                te["val_best"] = float(best)
                per_seed.append(te)
                print(f"[{task}/{pool}] seed {seed}: test {json.dumps({k: round(v, 4) for k, v in te.items()})}  ({time.time() - t0:.0f}s)", flush=True)
            agg = {k: [float(np.mean([p[k] for p in per_seed])), float(np.std([p[k] for p in per_seed]))] for k in per_seed[0]}
            results[f"{task}/{pool}"] = dict(seeds=per_seed, mean_std=agg)
            print(f"[{task}/{pool}] TEST " + "  ".join(f"{k} {m:.4f} ± {s:.4f}" for k, (m, s) in agg.items()), flush=True)
    json.dump(results, open(out / "results.json", "w"), indent=1)
    print("wrote", out / "results.json")


if __name__ == "__main__":
    run(parse_args())
