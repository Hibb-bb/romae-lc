"""Does the frozen window latent know where in the cycle the window starts?

The question behind the flat forecasts (2026-09-30): is the phase missing
from the latent, or can the decoder just not read it? This probe answers
the first half. For every cached window the target is the phase of the
window's start in the star's cycle, ``psi = (start / P - phi_ref) mod 1``,
with ``P`` the catalogue period and ``phi_ref`` the phase of maximum light
of the star's folded template (so the reference is the star's own cycle,
not the time origin). A ridge regressor and a small MLP predict
``(cos 2 pi psi, sin 2 pi psi)`` from the latent; the read-out is
``atan2``. The score is the wrapped phase error in cycles: median, the
share within 0.05 and 0.1 cycles (chance 0.1 and 0.2), and the mean of
``cos(2 pi error)`` (1 = perfect, 0 = chance). A control with the targets
shuffled among the windows of every star must sit at chance.

If the probe reads the phase, the information is in the latent and the
decoder design is the limit. If it does not, no decoder on this latent can
place the curve in time, and the phase has to come from the data (a fold
on a period from the past).

    python -m project.phase_probe --latents project/runs/mae_w250/latents.pt --out project/results/phase_mae
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from project.common import add_device_arg, data_args_from, dump_json, get_device, load_data, load_encoder, superclass
from project.inject import band_templates

HEADLINE = ("ECL", "RR", "ROT", "CEP", "DSCT", "LPV")


def reference_phase(record, period: float):
    """The phase of maximum light (the smallest magnitude of the template) in
    the band with the most points; NaN without a usable template."""
    templates, _ = band_templates(record, period)
    if not templates:
        return float("nan")
    b = max(templates, key=lambda k: int((record.band == k).sum()))
    edges, vals, _ = templates[b]
    centres = 0.5 * (edges[1:] + edges[:-1])
    return float(centres[int(np.nanargmin(vals))])


def window_targets(cache, split, records, min_tokens, log=print):
    """``(rows, psi, superclass, obj)`` over the valid windows of ``split``:
    cache row indices, the start phase in the star's cycle, the class and
    the object number (for the shuffle control)."""
    ptr = cache["ptr"][split].numpy()
    objs = cache["objects"][split]
    index = objs["index"].numpy()
    period = objs["period"].numpy().astype(np.float64)
    n_tok = cache["n_tokens"].numpy()
    start = cache["start"].numpy().astype(np.float64)
    rows, psi, sup, obj = [], [], [], []
    t0 = time.time()
    for i in range(len(ptr) - 1):
        p = period[i]
        if not (np.isfinite(p) and p > 0):
            continue
        r = records[int(index[i])]
        ref = reference_phase(r, p)
        if not np.isfinite(ref):
            continue
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        ok = np.flatnonzero(n_tok[lo:hi] >= min_tokens) + lo
        if ok.size == 0:
            continue
        rows.append(ok)
        psi.append(np.mod(start[ok] / p - ref, 1.0))
        sup.append(np.full(ok.size, superclass(r), dtype=object))
        obj.append(np.full(ok.size, i))
        if (i + 1) % 5000 == 0:
            log(f"  {split}: {i + 1} / {len(ptr) - 1} objects, {time.time() - t0:.0f}s")
    return np.concatenate(rows), np.concatenate(psi), np.concatenate(sup), np.concatenate(obj)


def phase_error(pred, true):
    """Wrapped phase error in cycles, in [0, 0.5]."""
    d = np.mod(pred - true, 1.0)
    return np.minimum(d, 1.0 - d)


def scores(pred, true, groups, order):
    def block(m):
        e = phase_error(pred[m], true[m])
        return dict(n=int(m.sum()), median=float(np.median(e)), within_005=float((e < 0.05).mean()),
                    within_01=float((e < 0.1).mean()), cos=float(np.cos(2 * np.pi * e).mean()))  # fmt: skip

    out = block(np.ones(len(true), dtype=bool))
    out["by_superclass"] = {g: block(groups == g) for g in order if (groups == g).sum() >= 20}
    return out


def fit_ridge(x_tr, y_tr, x_va, alpha=1.0):
    xt = torch.cat([x_tr, torch.ones(len(x_tr), 1, device=x_tr.device)], 1)
    xv = torch.cat([x_va, torch.ones(len(x_va), 1, device=x_va.device)], 1)
    a = xt.T @ xt + alpha * torch.eye(xt.shape[1], device=xt.device)
    w = torch.linalg.solve(a, xt.T @ y_tr)
    return xv @ w


def fit_mlp(x_tr, y_tr, x_va, hidden, steps, batch, seed, device):
    torch.manual_seed(seed)
    d = x_tr.shape[1]
    net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device=device).manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(x_tr), (batch,), device=device, generator=g)
        loss = ((net(x_tr[idx]) - y_tr[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        return torch.cat([net(x_va[i : i + 65536]) for i in range(0, len(x_va), 65536)])


def to_phase(pred):
    p = pred.cpu().numpy().astype(np.float64)
    return np.mod(np.arctan2(p[:, 1], p[:, 0]) / (2 * np.pi), 1.0)


def md_table(rows, cols):
    out = "| " + " | ".join(c for _, c in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    for r in rows:
        out += "| " + " | ".join(f"{r.get(k):.3f}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k, _ in cols) + " |\n"
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latents", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", default=None, help="the encoder (default: the cache's), for the data arguments")
    p.add_argument("--train-windows", type=int, default=300_000)
    p.add_argument("--val-windows", type=int, default=None)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--alpha", type=float, default=1.0, help="ridge penalty")
    p.add_argument("--min-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--n-sim", type=int, default=None)
    add_device_arg(p)
    return p.parse_args(argv)


def run(args):
    t0 = time.time()
    dev = get_device(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.latents, map_location="cpu", weights_only=False)
    ckpt = args.ckpt or cache["meta"]["ckpt"]
    min_tokens = args.min_tokens or int(cache["meta"]["min_tokens"])
    _, meta = load_encoder(ckpt, "cpu")
    over = argparse.Namespace(data=args.data, max_rows=args.max_rows, n_sim=args.n_sim)
    data = load_data(data_args_from(meta.args, over), splits=("train", "validation"))
    print(f"cache {args.latents} (dim {cache['z'].shape[1]}); targets from the catalogue period and the folded template", flush=True)
    rows_tr, psi_tr, sup_tr, obj_tr = window_targets(cache, "train", data["train"], min_tokens)
    rows_va, psi_va, sup_va, obj_va = window_targets(cache, "validation", data["validation"], min_tokens)
    rng = np.random.default_rng(args.seed)
    if args.train_windows and len(rows_tr) > args.train_windows:
        sel = np.sort(rng.choice(len(rows_tr), args.train_windows, replace=False))
        rows_tr, psi_tr, sup_tr, obj_tr = rows_tr[sel], psi_tr[sel], sup_tr[sel], obj_tr[sel]
    if args.val_windows and len(rows_va) > args.val_windows:
        sel = np.sort(rng.choice(len(rows_va), args.val_windows, replace=False))
        rows_va, psi_va, sup_va, obj_va = rows_va[sel], psi_va[sel], sup_va[sel], obj_va[sel]
    print(f"{len(rows_tr)} train windows, {len(rows_va)} validation windows; {time.time() - t0:.0f}s", flush=True)
    z = cache["z"]
    x_tr = z[torch.as_tensor(rows_tr)].float()
    x_va = z[torch.as_tensor(rows_va)].float()
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    x_tr, x_va = ((x_tr - mu) / sd).to(dev), ((x_va - mu) / sd).to(dev)
    # the shuffle control: the same targets, permuted among the windows of every star
    psi_sh = psi_tr.copy()
    for o in np.unique(obj_tr):
        m = np.flatnonzero(obj_tr == o)
        psi_sh[m] = psi_tr[rng.permutation(m)]
    order = [g for g in HEADLINE if (sup_va == g).any()]
    res = dict(latents=str(args.latents), ckpt=str(ckpt), n_train=int(len(rows_tr)), n_val=int(len(rows_va)), models={})
    for name, target in (("phase", psi_tr), ("shuffled control", psi_sh)):
        y_tr = torch.as_tensor(np.stack([np.cos(2 * np.pi * target), np.sin(2 * np.pi * target)], 1), dtype=torch.float32, device=dev)
        pr = to_phase(fit_ridge(x_tr, y_tr, x_va, args.alpha))
        pm = to_phase(fit_mlp(x_tr, y_tr, x_va, args.hidden, args.steps, args.batch_size, args.seed, dev))
        res["models"][f"ridge / {name}"] = scores(pr, psi_va, sup_va, order)
        res["models"][f"mlp / {name}"] = scores(pm, psi_va, sup_va, order)
        if name == "phase":
            np.savez_compressed(out / "predictions.npz", rows=rows_va, psi=psi_va, ridge=pr, mlp=pm, superclass=sup_va.astype(str))
    for k, v in res["models"].items():
        print(f"{k:24s} median error {v['median']:.3f} cycles | within 0.05: {v['within_005']:.2f} (chance 0.10) | "
              f"within 0.1: {v['within_01']:.2f} (chance 0.20) | mean cos {v['cos']:.3f} (chance 0) | "
              + " ".join(f"{g} {b['cos']:.2f}" for g, b in v["by_superclass"].items()), flush=True)  # fmt: skip
    res["seconds"] = time.time() - t0
    dump_json(res, out / "results.json")
    lines = ["# Phase probe\n", f"Latents `{args.latents}`; {res['n_train']} training windows, {res['n_val']} validation windows. "
             "Target: the phase of the window's start in the star's cycle (catalogue period, reference = maximum light of the "
             "folded template). Error in cycles, wrapped. Chance: median 0.25, within 0.05 = 0.10, within 0.1 = 0.20, mean cos = 0.\n"]  # fmt: skip
    cols = [("name", "read-out"), ("n", "n"), ("median", "median error"), ("within_005", "within 0.05"), ("within_01", "within 0.1"), ("cos", "mean cos")]
    lines.append(md_table([dict(name=k, **{c: v[c] for c, _ in cols[1:]}) for k, v in res["models"].items()], cols))
    lines.append("\n## mean cos by superclass\n")
    lines.append(md_table([dict(name=k, **{g: v["by_superclass"].get(g, {}).get("cos", float("nan")) for g in order}) for k, v in res["models"].items()],
                          [("name", "read-out")] + [(g, g) for g in order]))  # fmt: skip
    (out / "tables.md").write_text("\n".join(lines))
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return res


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
