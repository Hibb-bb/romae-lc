"""The gate on frozen latents: run it on a stage-1 autoencoder (``mae.pt``)
or a stage-2 world model (``wm.pt``) before any predictor GPU time is spent.

A stage-2 predictor trained on frozen latents can only learn what the
latents hold, and only if a window's latent moves when the window moves.
The gate answers both questions with numbers next to a null, the way the
molecular world-model kit's ``check`` stage does (effect size against the
replicate floor, a decision, exit 1 when it fails):

1. **Probe vs baseline** (hard gate): the linear class probe and the ridge
   R2 on log period of :func:`project.common.probe` on the frozen latents,
   next to :func:`project.common.baseline_probe` (hand features, no model) on
   the same records. Passes when ``r2_within`` and ``macro_f1`` both beat the
   baseline. The time-shuffle score is reported, not gated.
2. **Advance effect vs replicate floor** (hard gate): for ``--n-objects``
   validation objects a reference window at a seeded start ``s`` is encoded,
   then the window shifted by every advance ``a`` of ``--advances`` (window
   units). The replicate floor of the object is ``||z_a - z_b||`` for two
   independent random drops of ``--drop`` of the reference window's points:
   what the latent moves by when nothing but the sampling changes. The
   shifted window gets the same drop, and ``effect(a) = ||z(s + a w) - z_a||``
   is measured from one of the two replicates, so effect and floor carry the
   same sampling noise: with no real shift the ratio ``effect(a) / median
   floor`` sits at 1, and a shift of the floor's own size takes it to about
   1.4. The ratio must be at least ``--min-effect`` at ``--train-advance``
   (the smallest advance of the stage-2 training range). The median cosine
   between ``z_a`` and ``z(s + a w)`` says how far the latent has turned; a
   suggested advance range is printed as a hint.
3. **Decoder vs GP** (only with ``--decoder-results``): the stage-3 decoder's
   validation imputation NLL from a ``train_decoder`` result (its
   ``log.jsonl``, ``dec.pt`` / ``last.pt``, or a json with the same keys)
   next to the linear and RBF-GP baselines of that file; passes when the
   decoder beats the GP.

Everything goes to ``gate.json`` (default next to the checkpoint) with a
``passed`` flag per part and overall (parts 1 and 2 are hard gates, part 3
counts when present, a skipped part does not count); the last line printed
is ``GATE PASSED`` or ``GATE FAILED: ...`` and the exit code is 1 on failure.

    python -m project.gate --ckpt project/runs/mae_w250/mae.pt
    python -m project.gate --ckpt project/runs/mae_w250/mae.pt --n-objects 128 \\
        --advances 0.25 0.5 1.0 2.0 --decoder-results project/runs/mae_w250/dec_mse/log.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from romae_lc import FrameDataset, frame_grid

from project.common import (
    HEADLINE,
    add_device_arg,
    baseline_probe,
    data_args_from,
    describe_by_class,
    describe_probe,
    dump_json,
    get_device,
    load_data,
    load_encoder,
    probe,
    seed_all,
    subsample_frame,
    subset,
    superclass,
)
from project.diagnostics import shuffle_score
from project.tracking import Tracker, add_wandb_args

DEFAULT_ADVANCES = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", required=True, help="mae.pt (stage 1) or wm.pt (stage 2)")
    p.add_argument("--out", default=None, help="default <ckpt dir>/gate.json")
    g = p.add_argument_group("data (default: the checkpoint's)")
    g.add_argument("--data", default=None)
    g.add_argument("--classes", nargs="*", default=None)
    g.add_argument("--max-rows", type=int, default=None)
    g.add_argument("--n-sim", type=int, default=None)
    g = p.add_argument_group("part 1: probe vs baseline")
    g.add_argument("--probe-train", type=int, default=4000, help="probe fit records")
    g.add_argument("--probe-val", type=int, default=2000)
    g.add_argument("--shuffle-objects", type=int, default=512)
    g.add_argument("--skip-probe", action="store_true", help="skip part 1")
    g = p.add_argument_group("part 2: advance effect vs replicate floor")
    g.add_argument("--n-objects", type=int, default=512, help="validation objects")
    g.add_argument(
        "--advances",
        type=float,
        nargs="+",
        default=list(DEFAULT_ADVANCES),
        help="window shifts tested, in window units",
    )
    g.add_argument(
        "--train-advance",
        type=float,
        default=1.0,
        help="the smallest advance of the stage-2 training range: the gated one",
    )
    g.add_argument(
        "--drop", type=float, default=0.5, help="point fraction dropped for the floor"
    )
    g.add_argument("--min-effect", type=float, default=1.5, help="ratio to pass")
    g.add_argument("--max-tries", type=int, default=20, help="window draws per object")
    g = p.add_argument_group("part 3: decoder vs GP")
    g.add_argument(
        "--decoder-results",
        default=None,
        help="a train_decoder log.jsonl, dec.pt / last.pt, or json with its eval keys",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    add_device_arg(p)
    add_wandb_args(p)
    args = p.parse_args(argv)
    if any(a <= 0 for a in args.advances) or args.train_advance <= 0:
        p.error("advances must be > 0")
    if not 0 <= args.drop < 1:
        p.error("--drop must be in [0, 1)")
    return args


def _gt(a, b) -> bool:
    """``a > b`` that is False for NaN or None on either side."""
    try:
        return bool(np.isfinite(a) and np.isfinite(b) and a > b)
    except TypeError:
        return False


def kept_records(records, cfg) -> list:
    """The records that can host a frame sequence of ``cfg`` (what the probe
    embeds), so the baseline probe sees exactly the same records."""
    return list(FrameDataset(records, cfg, epoch_seed=False).records)


# ------------------------------------------------------------- part 1: probe


def probe_part(enc, meta, data, args, device) -> dict:
    cfg, spec = meta.cfg, meta.spec
    train_kept = kept_records(data["train"], cfg)
    val_kept = kept_records(data["validation"], cfg)
    probe_tr = subset(train_kept, args.probe_train, args.seed)
    probe_va = subset(val_kept, args.probe_val, args.seed)
    n_classes = len(data.classes)
    t0 = time.time()
    # project=True: the latent the stage-2 predictor sees (ignored by an autoencoder)
    pr = probe(
        enc, probe_tr, probe_va, cfg, spec, args.batch_size, device, n_classes,
        project=True, seed=args.seed,
    )  # fmt: skip
    bl = baseline_probe(probe_tr, probe_va, data.wavelengths.keys(), n_classes)
    sh = shuffle_score(
        enc, val_kept, cfg, spec, device,
        n=args.shuffle_objects, batch_size=args.batch_size, seed=args.seed,
    )  # fmt: skip
    passed = _gt(pr["r2_within"], bl["r2_within"]) and _gt(
        pr["macro_f1"], bl["macro_f1"]
    )
    print(f"  latent   {describe_probe(pr)}")
    print(f"  baseline {describe_probe(bl)}")
    print(
        f"  shuffle score {sh:.3f} (1 = blind to time; reported, not gated) | "
        f"{len(probe_tr)} fit / {len(probe_va)} val records | {time.time() - t0:.0f}s"
    )
    return dict(
        status="run",
        passed=passed,
        criterion="r2_within and macro_f1 above the hand-feature baseline",
        probe=pr,
        baseline=bl,
        shuffle_score=sh,
        n_train=len(probe_tr),
        n_val=len(probe_va),
    )


# ------------------------------------------------------ part 2: advance effect


def cut_window(record, cfg, start: float):
    """The ``(t, y, band, err)`` frame of the window ``[start, start + window)``."""
    return frame_grid(record, cfg, start=start, n_frames=1)[0][0]


def pick_start(record, cfg, a_max: float, rng, max_tries: int):
    """A window start ``s`` with at least ``cfg.min_tokens`` points in
    ``[s, s + w)`` and the record spanning ``s + (1 + a_max) w``; None when
    no draw succeeds."""
    t = record.t.astype(np.float64)
    w = float(cfg.window)
    lo, hi = float(t.min()), float(t.max()) - (1.0 + a_max) * w
    if hi < lo:
        return None
    for _ in range(max_tries):
        s = float(rng.uniform(lo, hi))
        if int(((t >= s) & (t < s + w)).sum()) >= cfg.min_tokens:
            return s
    return None


@torch.no_grad()
def encode_windows(enc, spec, frames, batch_size: int, device) -> np.ndarray:
    """``[n, D]`` latents of single windows, ``batch_size`` at a time."""
    out = []
    for i in range(0, len(frames), batch_size):
        tok = spec.tokens(frames[i : i + batch_size]).to(device)
        out.append(enc.encode([tok])[:, 0].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, enc.dim), dtype=np.float32)


def class_order(counts: dict) -> list:
    """Headline superclasses first, then the rest by count (the order of
    :func:`project.common.describe_by_class`)."""
    order = [g for g in HEADLINE if g in counts]
    return order + sorted((g for g in counts if g not in HEADLINE), key=lambda g: -counts[g])


def _median(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.median(x)) if x.size else float("nan")


def effect_part(enc, meta, records, args, device) -> dict:
    cfg, spec = meta.cfg, meta.spec
    w = float(cfg.window)
    advances = sorted(set(float(a) for a in args.advances) | {float(args.train_advance)})
    a_max = max(advances)
    t0 = time.time()
    chosen = []  # (record, start)
    for i, r in enumerate(records):
        s = pick_start(r, cfg, a_max, np.random.default_rng([args.seed, i]), args.max_tries)
        if s is not None:
            chosen.append((r, s))
    n_no_window = len(records) - len(chosen)
    refs, drop_a, drop_b, groups = [], [], [], []
    shifted = {a: [] for a in advances}  # frames or None per object
    for i, (r, s) in enumerate(chosen):
        # the reference window capped deterministically, like grid_item
        ref = subsample_frame(
            cut_window(r, cfg, s), cfg.max_tokens, np.random.default_rng([args.seed, i, 0])
        )
        n = len(ref[0])
        n_keep = max(1, int(np.ceil(n * (1.0 - args.drop))))
        rng = np.random.default_rng([args.seed, i, 1])
        # two draws from one generator: independent subsets of the same window
        drop_a.append(subsample_frame(ref, n_keep, rng))
        drop_b.append(subsample_frame(ref, n_keep, rng))
        refs.append(ref)
        groups.append(superclass(r))
        for k, a in enumerate(advances):
            f = cut_window(r, cfg, s + a * w)
            if len(f[0]) < cfg.min_tokens:
                shifted[a].append(None)
            else:
                # the shifted window gets the same drop as the replicates, so
                # effect and floor carry the same sampling noise (like for like)
                rng_k = np.random.default_rng([args.seed, i, 2, k])
                f = subsample_frame(f, cfg.max_tokens, rng_k)
                n_keep_k = max(1, int(np.ceil(len(f[0]) * (1.0 - args.drop))))
                shifted[a].append(subsample_frame(f, n_keep_k, rng_k))
    groups = np.array(groups)
    z_full = encode_windows(enc, spec, refs, args.batch_size, device)
    z_a = encode_windows(enc, spec, drop_a, args.batch_size, device)
    z_b = encode_windows(enc, spec, drop_b, args.batch_size, device)
    z_ref = z_a  # one dropped replicate is the reference the shifts are measured from
    floor = np.linalg.norm(z_a - z_b, axis=1) if len(chosen) else np.zeros(0)
    floor_med = _median(floor)
    counts = {g: int((groups == g).sum()) for g in set(groups.tolist())}
    order = class_order(counts)
    floor_by = {g: _median(floor[groups == g]) for g in order}
    per_advance = []
    for a in advances:
        idx = np.array([i for i, f in enumerate(shifted[a]) if f is not None], dtype=int)
        row = dict(advance=a, n=int(idx.size), n_skipped=int(len(chosen) - idx.size))
        if idx.size:
            z_s = encode_windows(enc, spec, [shifted[a][i] for i in idx], args.batch_size, device)
            effect = np.linalg.norm(z_s - z_ref[idx], axis=1)
            ratio = effect / floor_med if floor_med > 0 else np.full_like(effect, np.inf)
            cos = (z_s * z_ref[idx]).sum(1) / (
                np.linalg.norm(z_s, axis=1) * np.linalg.norm(z_ref[idx], axis=1) + 1e-12
            )
            g = groups[idx]
            by, n_by = {}, {}
            for c in order:
                m = g == c
                n_by[c] = int(m.sum())
                if m.any():
                    f = floor_by[c]
                    by[c] = float(_median(effect[m]) / f) if f > 0 else float("inf")
            row.update(
                ratio_median=float(np.median(ratio)),
                ratio_mean=float(ratio.mean()),
                effect_median=float(np.median(effect)),
                cosine_median=float(np.median(cos)),
                ratio_by_superclass=by,
                n_by_superclass=n_by,
            )
        else:
            row.update(
                ratio_median=float("nan"), ratio_mean=float("nan"),
                effect_median=float("nan"), cosine_median=float("nan"),
                ratio_by_superclass={}, n_by_superclass={},
            )  # fmt: skip
        per_advance.append(row)
    gated = next(r for r in per_advance if r["advance"] == float(args.train_advance))
    passed = bool(gated["ratio_median"] >= args.min_effect)  # NaN compares False
    # suggested range: first advance clearing min-effect, up to the last whose cosine is still >= 0.5
    above = [r["advance"] for r in per_advance if r["ratio_median"] >= args.min_effect]
    turned = [r["advance"] for r in per_advance if r["cosine_median"] >= 0.5]
    suggested = (
        [min(above), max(turned) if turned else max(advances)] if above else None
    )
    if suggested is not None and suggested[1] < suggested[0]:
        suggested = [suggested[0], suggested[0]]
    print(
        f"  {len(chosen)} of {len(records)} objects with a usable reference window "
        f"({n_no_window} without); replicate floor (drop {args.drop:.0%}) median "
        f"{floor_med:.4f} mean {float(floor.mean()) if floor.size else float('nan'):.4f} "
        f"| ref latent norm median {_median(np.linalg.norm(z_full, axis=1)):.3f} | "
        f"{time.time() - t0:.0f}s"
    )
    head = [g for g in order if g in HEADLINE] or order[:3]
    cols = "  ".join(f"{g:>7}" for g in head)
    print(f"  {'advance':>8} {'ratio':>7} {'mean':>7} {'cosine':>7} {'n':>5}  {cols}")
    for r in per_advance:
        mark = " <- gated" if r["advance"] == float(args.train_advance) else ""
        by = r["ratio_by_superclass"]
        cols = "  ".join(f"{by.get(g, float('nan')):7.2f}" for g in head)
        print(
            f"  {r['advance']:8.3g} {r['ratio_median']:7.2f} {r['ratio_mean']:7.2f} "
            f"{r['cosine_median']:7.3f} {r['n']:5d}  {cols}{mark}"
        )
    print("  per-class floor: " + describe_by_class(floor_by, counts))
    if suggested:
        print(
            f"  suggested advance range (hint): {suggested[0]:g} to {suggested[1]:g} window units"
        )
    else:
        print(f"  no listed advance reaches ratio {args.min_effect:g}")
    return dict(
        status="run",
        passed=passed,
        criterion=f"median effect / median floor >= {args.min_effect:g} at advance {args.train_advance:g}",
        ratio_at_train_advance=gated["ratio_median"],
        train_advance=float(args.train_advance),
        min_effect=float(args.min_effect),
        drop=float(args.drop),
        n_candidates=int(len(records)),
        n_used=int(len(chosen)),
        n_no_window=int(n_no_window),
        floor_median=floor_med,
        floor_mean=float(floor.mean()) if floor.size else float("nan"),
        floor_by_superclass=floor_by,
        n_by_superclass={g: counts[g] for g in order},
        per_advance=per_advance,
        suggested_advance_range=suggested,
    )


# ------------------------------------------------------ part 3: decoder vs GP


def read_decoder_results(path) -> dict:
    """The validation imputation metrics of a ``train_decoder`` run: the
    last ``eval`` line of its ``log.jsonl``, the ``metrics`` of its
    ``dec.pt`` / ``last.pt``, or a json holding them (or a ``metrics`` key)."""
    path = Path(path)
    if path.suffix == ".pt":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = ckpt.get("metrics")
        if not m:
            raise ValueError(f"{path} holds no evaluation metrics yet")
        return dict(m, step=ckpt.get("step", m.get("step")))
    if path.suffix == ".jsonl":
        evals = [
            l for l in (json.loads(x) for x in open(path) if x.strip()) if l.get("kind") == "eval"
        ]
        if not evals:
            raise ValueError(f"{path} has no eval line")
        return evals[-1]
    m = json.load(open(path))
    return m.get("metrics", m) if "decoder" not in m else m


def decoder_part(path) -> dict:
    m = read_decoder_results(path)
    methods = ("decoder", "gp_rbf", "linear", "constant", "gp_periodic")
    nll = {k: (m.get(k) or {}).get("nll") for k in methods if isinstance(m.get(k), dict)}
    rmse = {k: (m.get(k) or {}).get("rmse") for k in nll}
    dec, gp = nll.get("decoder"), nll.get("gp_rbf")
    note = None
    if dec is None:
        note = "no decoder NLL in the results"
    elif gp is None:
        note = "no GP baseline in the results (train_decoder --baselines)"
    passed = _gt(gp, dec) if note is None else False
    print(
        f"  imputation NLL ({m.get('holdout')}, {m.get('frac')} hidden, "
        f"{m.get('n_windows')} windows, decoder step {m.get('step')}): "
        + "  ".join(f"{k} {v:.3f}" for k, v in nll.items() if v is not None)
        + (f" | {note}" if note else "")
    )
    return dict(
        status="run",
        passed=passed,
        criterion="decoder validation imputation NLL below the RBF GP's",
        source=str(path),
        nll=nll,
        rmse=rmse,
        step=m.get("step"),
        n_windows=m.get("n_windows"),
        holdout=m.get("holdout"),
        frac=m.get("frac"),
        note=note,
    )


# ------------------------------------------------------------------- driver


def verdict(result: dict) -> str:
    if result["passed"]:
        return "GATE PASSED"
    why = []
    p = result["probe"]
    if p["status"] == "run" and not p["passed"]:
        why.append(
            f"part 1 probe vs baseline: r2_within {p['probe']['r2_within']:.3f} vs "
            f"{p['baseline']['r2_within']:.3f}, macro F1 {p['probe']['macro_f1']:.3f} vs "
            f"{p['baseline']['macro_f1']:.3f}"
        )
    e = result["effect"]
    if not e["passed"]:
        why.append(
            f"part 2 advance effect: ratio {e['ratio_at_train_advance']:.2f} at advance "
            f"{e['train_advance']:g} < {e['min_effect']:g} (floor median {e['floor_median']:.4f}, "
            f"{e['n_used']} objects)"
        )
    d = result["decoder"]
    if d["status"] == "run" and not d["passed"]:
        dec, gp = d["nll"].get("decoder"), d["nll"].get("gp_rbf")
        why.append(
            f"part 3 decoder vs GP: NLL {dec if dec is None else round(dec, 3)} vs "
            f"GP {gp if gp is None else round(gp, 3)}" + (f" ({d['note']})" if d["note"] else "")
        )
    return "GATE FAILED: " + "; ".join(why)


def run(args: argparse.Namespace) -> dict:
    """Run the parts and write ``gate.json``; returns the result dict."""
    seed_all(args.seed)
    dev = get_device(args)
    out = Path(args.out or (Path(args.ckpt).parent / "gate.json"))
    t0 = time.time()
    enc, meta = load_encoder(args.ckpt, dev)
    overrides = argparse.Namespace(
        data=args.data, classes=args.classes, max_rows=args.max_rows, n_sim=args.n_sim
    )  # the gate's --seed is its own; the checkpoint's data seed stays
    data = load_data(data_args_from(meta.args, overrides))
    print(
        f"gate on {args.ckpt} ({enc.kind}, step {meta.step}, latent {enc.dim}, "
        f"window {meta.cfg.window} d): {len(data['train'])} train / "
        f"{len(data['validation'])} val records, loaded in {time.time() - t0:.0f}s"
    )
    result = dict(
        ckpt=str(args.ckpt),
        kind=enc.kind,
        step=meta.step,
        dim=enc.dim,
        window=meta.cfg.window,
        args=dict(vars(args)),
    )
    print("part 1: probe vs hand-feature baseline")
    if args.skip_probe:
        print("  skipped (--skip-probe)")
        result["probe"] = dict(status="skipped", passed=None)
    else:
        result["probe"] = probe_part(enc, meta, data, args, dev)
    print("part 2: advance effect vs replicate floor")
    val = subset(data["validation"], args.n_objects, args.seed)
    result["effect"] = effect_part(enc, meta, val, args, dev)
    print("part 3: decoder vs GP")
    if args.decoder_results:
        result["decoder"] = decoder_part(args.decoder_results)
    else:
        print("  not run (no --decoder-results)")
        result["decoder"] = dict(status="not run", passed=None)
    parts = [result[k] for k in ("probe", "effect", "decoder")]
    result["passed"] = all(p["passed"] for p in parts if p["status"] == "run")
    result["verdict"] = verdict(result)
    dump_json(result, out)
    tracker = Tracker(args, config=dict(vars(args), kind=enc.kind, step=meta.step), job_type="gate")
    tracker.summary(
        gate_passed=result["passed"],
        gate_effect_ratio=result["effect"]["ratio_at_train_advance"],
        gate_floor_median=result["effect"]["floor_median"],
        **{
            f"gate_probe_{k}": v
            for k, v in (result["probe"].get("probe") or {}).items()
            if isinstance(v, (int, float))
        },
    )
    tracker.finish()
    print(f"wrote {out}")
    return result


def main(argv=None):
    args = parse_args(argv)
    result = run(args)
    print(result["verdict"], flush=True)
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
