"""Energy-based inference (M6, M7) with a stage-2 ``wm.pt`` of ``train_wm.py``
(frozen encoder plus MSE predictor), its ``residual.py`` Gaussian and the
stage-3 decoder; the flow predictor of ``train_predictor.py`` is not wired
into the energies yet.

Tasks (all on the ``fill`` window grid of each object, advance 1):

- ``calibrate``: per-window magnitudes of the prior, dynamics and observation
  energies on training objects with every window observed, and the weights
  that equalise them (``weights.json``; other tasks take ``--weights``).
- ``smooth``: hide a block of ``--hide-frac`` of the windows in the middle of
  each object, initialise the path (encoded latents, predictor rollout into
  the gap), run MAP then Langevin, decode the hidden windows: NLL of the
  hidden points under the known errors for the MAP path, the rollout-only
  initialisation and the encoder oracle (the encoder sees the hidden
  window), plus the spread across chains.
- ``forecast``: hide the last ``--horizon`` windows; same machinery without
  observations on them; NLL per horizon against the persistence latent.
- ``anomaly``: every window observed; after a short MAP, ``E_dyn`` per step
  and ``E_obs`` per window; object score = mean over windows; with
  ``--inject KIND`` the clean and the injected copy are scored and the AUROC
  of ``E_dyn``, ``E_obs``, their weighted sum and the raw surprise reported;
  thresholds are the percentiles of the clean object scores.
- ``period``: on objects whose catalogue period is at least ``--min-period``
  days, the periodicity energy of the encoded path over a log grid of trial
  periods; the argmin against the catalogue value (only periods comparable
  to the window spacing are resolvable on a per-window path).

    python -m project.infer calibrate --ckpt W --decoder D --residual R --out O
    python -m project.infer smooth --ckpt W --decoder D --weights O/weights.json --n-objects 50 --out O
    python -m project.infer anomaly --ckpt W --decoder D --inject phase --n-objects 200 --out O
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from project import baselines as bl
from project.common import (
    add_device_arg,
    data_args_from,
    dump_json,
    get_device,
    grid_item,
    grid_starts,
    load_data,
    load_wm,
    seed_all,
    subset,
    superclass,
)
from project.decoder import decode_mean, load_decoder
from project.energy import PathEnergy, init_path, langevin, map_path, periodicity_scan
from project.inject import KINDS, _auroc, inject
from project.residual import load_residual


def add_args(p):
    p.add_argument(
        "task", choices=("calibrate", "smooth", "forecast", "anomaly", "period")
    )
    p.add_argument("--ckpt", required=True, help="stage-1 checkpoint")
    p.add_argument("--decoder", default=None, help="stage-3 dec.pt (needed for E_obs)")
    p.add_argument(
        "--residual", default=None, help="stage-2 residual.pt (Gaussian E_dyn)"
    )
    p.add_argument("--weights", default=None, help="weights.json from calibrate")
    p.add_argument("--w-dyn", type=float, default=1.0)
    p.add_argument("--w-obs", type=float, default=1.0)
    p.add_argument("--w-prior", type=float, default=0.1)
    p.add_argument("--split", default="validation")
    p.add_argument("--n-objects", type=int, default=50)
    p.add_argument(
        "--hide-frac", type=float, default=0.25, help="smooth: windows hidden"
    )
    p.add_argument(
        "--horizon", type=int, default=1, help="forecast: windows hidden at the end"
    )
    p.add_argument("--map-steps", type=int, default=300)
    p.add_argument("--map-lr", type=float, default=1e-2)
    p.add_argument("--langevin-steps", type=int, default=300)
    p.add_argument("--chains", type=int, default=64)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--obs-steps", type=int, default=10, help="flow decoder ODE steps inside E_obs"
    )
    p.add_argument(
        "--inject", choices=KINDS, default=None, help="anomaly: inject this kind"
    )
    p.add_argument(
        "--min-period", type=float, default=None, help="period: default window / 2"
    )
    p.add_argument("--n-periods", type=int, default=200)
    p.add_argument(
        "--plot",
        type=int,
        default=0,
        help="smooth/forecast: PNGs of the first n objects",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--classes", nargs="*", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    add_device_arg(p)


def setup(args):
    seed_all(args.seed)
    dev = get_device(args)
    model, meta = load_wm(args.ckpt, dev)
    dec = load_decoder(args.decoder, dev)[0] if args.decoder else None
    res = load_residual(args.residual, dev)[0] if args.residual else None
    w = dict(w_dyn=args.w_dyn, w_obs=args.w_obs, w_prior=args.w_prior)
    if args.weights:
        w.update({k: v for k, v in json.load(open(args.weights)).items() if k in w})
    energy = PathEnergy(model, dec, res, obs_steps=args.obs_steps, **w)
    split = "train" if args.task == "calibrate" else args.split
    data = load_data(data_args_from(meta.args, args), splits=(split,))
    recs = subset(data[split], args.n_objects, args.seed)
    print(
        f"{args.task}: {len(recs)} {split} records, window {meta.cfg.window}, weights {w}, decoder {'yes' if dec else 'no'}, residual {'yes' if res else 'no'}"
    )
    return dev, model, meta, energy, recs, w


def object_grid(record, meta, dev, index=0, seed=0):
    """Tokens of every fill-grid window (one row each) with actions and starts."""
    cfg, spec = meta.cfg, meta.spec
    item = grid_item(record, cfg, index, cap=cfg.max_tokens, seed=seed)
    batch = spec.collate()([item])
    frames = [f.to(dev) for f in batch["frames"]]
    return (
        frames,
        batch["actions"][0].to(dev),
        grid_starts(record, cfg),
        batch["n_tokens"][0].numpy(),
    )


def hidden_nll(dec, Z, frames, hidden, obs_steps):
    """Per-window mean NLL (known sigma) and RMSE of the decoded hidden
    windows from a path ``Z [1, K, D]``; empty windows give NaN."""
    nlls, rmses = [], []
    for k in hidden:
        tok = frames[k]
        real = ~tok.pad_mask[0]
        if not real.any():
            nlls.append(float("nan"))
            rmses.append(float("nan"))
            continue
        mu = decode_mean(dec, Z[:, k], tok, n_steps=obs_steps)[0]
        m, s = tok.values[0, :, 0].float(), tok.extras[0].float()
        nll = bl.gaussian_nll(
            m[real].cpu().numpy(), mu[real].cpu().numpy(), s[real].cpu().numpy() ** 2
        ).mean()
        nlls.append(float(nll))
        rmses.append(float((m[real] - mu[real]).square().mean().sqrt()))
    return np.array(nlls), np.array(rmses)


def run_calibrate(args):
    dev, model, meta, energy, recs, w = setup(args)
    if energy.decoder is None:
        raise SystemExit("calibrate needs --decoder")
    pr, dy, ob = [], [], []
    with torch.no_grad():
        for i, r in enumerate(recs):
            frames, actions, _, n_tok = object_grid(r, meta, dev, i, args.seed)
            if len(frames) < 2:
                continue
            Z = model.encode(frames).float()
            observed = [
                (k, frames[k])
                for k in range(len(frames))
                if n_tok[k] >= meta.cfg.min_tokens
            ]
            pr += energy.prior(Z)[0].tolist()
            dy += energy.dyn(Z, actions)[0].tolist()
            ob += energy.obs(Z, observed)[0].tolist()
    m_pr, m_dy, m_ob = (float(np.mean(x)) for x in (pr, dy, ob))
    weights = dict(
        w_dyn=1.0,
        w_obs=m_dy / max(m_ob, 1e-12),
        w_prior=0.1 * m_dy / max(m_pr, 1e-12),
        mean_prior=m_pr,
        mean_dyn=m_dy,
        mean_obs=m_ob,
        n_windows=len(dy),
    )
    print(
        f"per-window means: prior {m_pr:.4f} dyn {m_dy:.4f} obs {m_ob:.4f} -> weights {weights}"
    )
    dump_json(weights, Path(args.out) / "weights.json")


def run_smooth(args, forecast=False):
    dev, model, meta, energy, recs, w = setup(args)
    if energy.decoder is None:
        raise SystemExit("smooth/forecast need --decoder")
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    rows, per_obj = [], []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(recs):
        frames, actions, starts, n_tok = object_grid(r, meta, dev, i, args.seed)
        K = len(frames)
        if K < 3:
            continue
        if forecast:
            hidden = list(range(max(1, K - args.horizon), K))
        else:
            n_hide = max(1, int(round(args.hide_frac * K)))
            lo = max(1, (K - n_hide) // 2)
            hidden = list(range(lo, min(K - 1, lo + n_hide)))
        observed = np.array([k not in hidden for k in range(K)])
        windows = [
            (k, frames[k])
            for k in range(K)
            if observed[k] and n_tok[k] >= meta.cfg.min_tokens
        ]
        with torch.no_grad():
            Z_enc = model.encode(frames).float()
        Z0 = init_path(model, frames, actions, observed)
        Zmap, trace = map_path(
            energy, Z0, actions, windows, args.map_steps, args.map_lr
        )
        samples, ltrace = langevin(
            energy,
            Zmap,
            actions,
            windows,
            args.chains,
            args.langevin_steps,
            args.eta,
            args.temperature,
            gen,
        )
        with torch.no_grad():
            nll_map, rmse_map = hidden_nll(
                energy.decoder, Zmap, frames, hidden, args.obs_steps
            )
            nll_init, _ = hidden_nll(energy.decoder, Z0, frames, hidden, args.obs_steps)
            nll_enc, _ = hidden_nll(
                energy.decoder, Z_enc, frames, hidden, args.obs_steps
            )
            last_obs = (
                max(k for k in range(K) if observed[k] and k < hidden[0])
                if any(observed[: hidden[0]])
                else 0
            )
            Z_pers = Z_enc.clone()
            for k in hidden:
                Z_pers[:, k] = Z_enc[:, last_obs]
            nll_pers, _ = hidden_nll(
                energy.decoder, Z_pers, frames, hidden, args.obs_steps
            )
            spread = (
                samples.std(0).mean(-1).cpu().numpy()
            )  # [K] latent spread per window
            # decoded spread on the hidden windows from a few chains
            dec_spread = []
            for k in hidden:
                tok = frames[k]
                if not (~tok.pad_mask[0]).any():
                    dec_spread.append(float("nan"))
                    continue
                mus = torch.stack(
                    [
                        decode_mean(
                            energy.decoder,
                            samples[c : c + 1, k],
                            tok,
                            n_steps=args.obs_steps,
                        )[0]
                        for c in range(min(16, args.chains))
                    ]
                )
                dec_spread.append(float(mus.std(0)[~tok.pad_mask[0]].mean()))
        rec = dict(
            index=i,
            superclass=superclass(r),
            n_windows=K,
            hidden=hidden,
            nll_map=np.nanmean(nll_map),
            nll_init=np.nanmean(nll_init),
            nll_encoder_oracle=np.nanmean(nll_enc),
            nll_persistence=np.nanmean(nll_pers),
            rmse_map=np.nanmean(rmse_map),
            latent_spread_hidden=float(np.mean(spread[hidden])),
            latent_spread_observed=float(np.mean(spread[observed])),
            decoded_spread=float(np.nanmean(dec_spread)),
            energy_map=trace[-1],
            energy_start=trace[0],
        )
        per_obj.append(rec)
        print(
            f"  obj {i:3d} {rec['superclass']:5s} K={K} hidden={hidden}: NLL map {rec['nll_map']:.3f} init {rec['nll_init']:.3f} persistence {rec['nll_persistence']:.3f} oracle {rec['nll_encoder_oracle']:.3f} | spread hidden {rec['latent_spread_hidden']:.3f} obs {rec['latent_spread_observed']:.3f} | E {trace[0]:.3f} -> {trace[-1]:.3f}",
            flush=True,
        )
        if args.plot and i < args.plot:
            plot_object(
                out / f"{args.task}_{i}.png",
                r,
                frames,
                hidden,
                Zmap,
                samples,
                energy,
                meta,
                args.obs_steps,
            )
    keys = (
        [
            k
            for k in per_obj[0]
            if isinstance(per_obj[0][k], (float, int, np.floating)) and k != "index"
        ]
        if per_obj
        else []
    )
    summary = {k: float(np.nanmean([o[k] for o in per_obj])) for k in keys}
    summary.update(
        n_objects=len(per_obj),
        task=args.task,
        weights=w,
        per_superclass={
            g: float(
                np.nanmean([o["nll_map"] for o in per_obj if o["superclass"] == g])
            )
            for g in sorted({o["superclass"] for o in per_obj})
        },
    )
    print(
        "summary:",
        {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in summary.items()
            if k != "per_superclass"
        },
    )
    dump_json(dict(summary=summary, objects=per_obj), out / f"{args.task}.json")


def plot_object(path, record, frames, hidden, Zmap, samples, energy, meta, obs_steps):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ts = meta.spec.tokenize["time_scale"]
    starts = grid_starts(record, meta.cfg)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.errorbar(
        record.t, record.y, record.err, fmt=".", ms=2, alpha=0.3, color="gray", lw=0.5
    )
    with torch.no_grad():
        for k, tok in enumerate(frames):
            real = ~tok.pad_mask[0]
            if not real.any():
                continue
            t = ((tok.positions[0, 0] - 1) * ts).cpu().numpy() + starts[k]
            mu = (
                decode_mean(energy.decoder, Zmap[:, k], tok, n_steps=obs_steps)[0]
                .cpu()
                .numpy()
            )
            mus = (
                torch.stack(
                    [
                        decode_mean(
                            energy.decoder,
                            samples[c : c + 1, k],
                            tok,
                            n_steps=obs_steps,
                        )[0]
                        for c in range(min(16, samples.shape[0]))
                    ]
                )
                .cpu()
                .numpy()
            )
            order = np.argsort(t[real.cpu().numpy()])
            tt, mm, sd = (
                t[real.cpu().numpy()][order],
                mu[real.cpu().numpy()][order],
                mus[:, real.cpu().numpy()].std(0)[order],
            )
            color = "tab:red" if k in hidden else "tab:blue"
            ax.plot(tt, mm, ".", ms=3, color=color)
            ax.fill_between(tt, mm - 2 * sd, mm + 2 * sd, color=color, alpha=0.15)
            ax.axvline(starts[k], color="k", lw=0.3, alpha=0.5)
    ax.set_xlabel("days")
    ax.set_ylabel("standardised -mag")
    ax.set_title(
        f"{record.meta.get('class_str')} P={record.period:.3g} d; hidden windows {hidden} (red = decoded from the smoothed path)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_anomaly(args):
    dev, model, meta, energy, recs, w = setup(args)
    if energy.decoder is None:
        print("no decoder: E_obs is off, scores are E_dyn only")
    rng = np.random.default_rng([args.seed, 7])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def score(record, i):
        frames, actions, starts, n_tok = object_grid(record, meta, dev, i, args.seed)
        K = len(frames)
        if K < 3:
            return None
        valid = n_tok >= meta.cfg.min_tokens
        windows = [(k, frames[k]) for k in range(K) if valid[k]]
        with torch.no_grad():
            Z = model.encode(frames).float()
            s_raw = model.surprise(frames, actions[None])[0].float().cpu().numpy()
        Zmap, _ = map_path(energy, Z, actions, windows, args.map_steps, args.map_lr)
        with torch.no_grad():
            dyn = energy.dyn(Zmap, actions)[0].cpu().numpy()
            obs = np.full(K, np.nan)
            if energy.decoder is not None and windows:
                obs[[k for k, _ in windows]] = (
                    energy.obs(Zmap, windows)[0].cpu().numpy()
                )
        dyn[~valid[1:]] = np.nan
        s_raw[~valid[1:]] = np.nan
        comb = w["w_dyn"] * dyn + w["w_obs"] * np.nan_to_num(obs[1:], nan=0.0)
        return dict(dyn=dyn, obs=obs, raw=s_raw, comb=comb, starts=starts, K=K)

    clean, injected, groups, kstars = [], [], [], []
    for i, r in enumerate(recs):
        c = score(r, i)
        if c is None:
            continue
        clean.append(c)
        groups.append(superclass(r))
        if args.inject:
            starts = c["starts"]
            t_star = float(rng.uniform(starts[1], starts[-1] + meta.cfg.window))
            k_star = int(np.searchsorted(starts, t_star, side="right") - 1)
            inj, _ = inject(r, args.inject, t_star, rng)
            injected.append(score(inj, i))
            kstars.append(k_star)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1} objects scored", flush=True)

    def obj_score(x, key):
        v = x[key][1:] if key == "obs" else x[key]
        return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan

    def obj_max(x, key):
        v = x[key][1:] if key == "obs" else x[key]
        return float(np.nanmax(v)) if np.isfinite(v).any() else np.nan

    summary = dict(n_objects=len(clean), weights=w, inject=args.inject)
    for key in ("dyn", "obs", "comb", "raw"):
        cs = np.array([obj_score(x, key) for x in clean])
        summary[f"clean_{key}"] = dict(
            mean=float(np.nanmean(cs)),
            p95=float(np.nanpercentile(cs, 95)),
            p99=float(np.nanpercentile(cs, 99)),
        )
        if args.inject:
            pos = np.array([obj_max(x, key) for x in injected])
            neg = np.array([obj_max(x, key) for x in clean])
            ok = np.isfinite(pos) & np.isfinite(neg)
            win_pos, win_neg, hits = [], [], []
            for x, ks in zip(injected, kstars):
                v = x[key][1:] if key == "obs" else x[key]
                k = np.arange(1, len(v) + 1)
                fin = np.isfinite(v)
                if not fin.any():
                    continue
                anom = (k >= ks) & (k <= ks + 2) & fin
                win_pos += v[anom].tolist()
                win_neg += v[fin & ~anom].tolist()
                hits.append(ks - 1 <= k[fin][np.nanargmax(v[fin])] <= ks + 2)
            summary[f"auroc_object_{key}"] = _auroc(pos[ok], neg[ok])
            summary[f"auroc_window_{key}"] = _auroc(
                np.array(win_pos), np.array(win_neg)
            )
            summary[f"hit_rate_{key}"] = float(np.mean(hits)) if hits else float("nan")
    for key in ("dyn", "obs", "comb", "raw"):
        line = f"{key:4s}: clean mean {summary[f'clean_{key}']['mean']:.4f} p95 {summary[f'clean_{key}']['p95']:.4f}"
        if args.inject:
            line += f" | AUROC object {summary[f'auroc_object_{key}']:.3f} window {summary[f'auroc_window_{key}']:.3f} hit {summary[f'hit_rate_{key}']:.3f}"
        print(line)
    dump_json(summary, out / f"anomaly{'_' + args.inject if args.inject else ''}.json")
    np.savez_compressed(
        out / f"anomaly{'_' + args.inject if args.inject else ''}.npz",
        **{
            f"clean_{k}": np.array([x[k] for x in clean], dtype=object)
            for k in ("dyn", "obs", "raw")
        },
        **(
            {
                f"inj_{k}": np.array([x[k] for x in injected], dtype=object)
                for k in ("dyn", "obs", "raw")
            }
            if injected
            else {}
        ),
        groups=np.array(groups),
        k_star=np.array(kstars),
    )


def run_period(args):
    dev, model, meta, energy, recs, w = setup(args)
    window = meta.cfg.window
    min_period = args.min_period if args.min_period is not None else window / 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for i, r in enumerate(recs):
            if not r.period >= min_period:
                continue
            frames, actions, starts, n_tok = object_grid(r, meta, dev, i, args.seed)
            if len(frames) < 4:
                continue
            Z = model.encode(frames).float()[0]
            span = starts[-1] - starts[0]
            periods = np.exp(
                np.linspace(
                    np.log(max(min_period, window / 4)),
                    np.log(span / 2),
                    args.n_periods,
                )
            )
            e = periodicity_scan(Z, starts, periods)
            e_half = periodicity_scan(Z, starts, periods / 2)
            # E_per is trivially smallest at the shortest lag on any smooth
            # path; a period shows as a dip relative to half the lag.
            score = e / np.maximum(e_half, 1e-12)
            ok = np.isfinite(score)
            if not ok.any():
                continue
            best = float(periods[ok][np.argmin(score[ok])])
            raw = float(periods[np.isfinite(e)][np.argmin(e[np.isfinite(e)])])
            rows.append(
                dict(
                    index=i,
                    superclass=superclass(r),
                    period=float(r.period),
                    best=best,
                    ratio=best / r.period,
                    best_raw=raw,
                    n_windows=len(frames),
                )
            )
            print(
                f"  obj {i:3d} {rows[-1]['superclass']:5s} P {r.period:8.2f} d  best {best:8.2f} d  ratio {best / r.period:.3f}  (raw argmin {raw:.1f} d)"
            )
    if not rows:
        print(f"no object with period >= {min_period} d and at least 4 windows")
        return
    ratio = np.array([x["ratio"] for x in rows])
    within = float(np.mean(np.abs(np.log(ratio)) < np.log(1.1)))
    summary = dict(
        n=len(rows),
        min_period=min_period,
        frac_within_10pct=within,
        median_ratio=float(np.median(ratio)),
        note="score = E_per(P) / E_per(P / 2) on a per-window path: only periods comparable to the window spacing are resolvable",
    )
    print("summary:", summary)
    dump_json(dict(summary=summary, objects=rows), out / "period.json")


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_args(p)
    args = p.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.task == "calibrate":
        run_calibrate(args)
    elif args.task == "smooth":
        run_smooth(args)
    elif args.task == "forecast":
        run_smooth(args, forecast=True)
    elif args.task == "anomaly":
        run_anomaly(args)
    else:
        run_period(args)
    print(f"{args.task} finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
