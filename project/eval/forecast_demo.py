"""Forecast demo: hide the LAST part of a window and let the model fill it.

For a few validation stars (one per class, many points), one 250-day window:
the last ``--hide`` of its points are hidden as one block, and dense query
tokens are placed over the first cycles of the hidden stretch. The figure
shows, per star, (left) the time view around the start of the hidden
stretch: seen points, hidden points, the whole-record template, and every
model's prediction as a curve; (right) the whole hidden stretch folded on
the template's period: hidden points against each model's prediction at
them. The title gives the variance of the template explained on the hidden
points by each model.

    python -m project.eval.forecast_demo --ckpt plain=project/runs/maew_spec/mae.pt --ckpt cyc=project/runs/enc_cyc/mae.pt \
        --out project/results/forecast_demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from project.cache_latents import object_windows
from project.common import add_device_arg, data_args_from, get_device, load_data, load_mae
from project.mae_data import make_mask
from project.plot_period_examples import INK, INK2, SURFACE, style
from project.templates import RecordTemplates, augment_with_queries, bounded, fit_record_templates, period_lookup_best

BAND_COLOUR = {0: "#2a78d6", 1: "#d6452a", 2: "#7b3fb8"}
MODEL_COLOUR = ["#8c8c8c", "#eb6834", "#1baf7a", "#6a51a3"]
NAMED_COLOUR = {"plain": "#8c8c8c", "maew_spec": "#8c8c8c", "cyc": "#eb6834", "cycph": "#6a51a3", "cycrope": "#1baf7a", "cycrope2": "#1baf7a", "cycropetpl": "#2a78d6"}
ALPHA = 0.75


def colour_of(name: str, k: int) -> str:
    return NAMED_COLOUR.get(name, MODEL_COLOUR[k % len(MODEL_COLOUR)])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", action="append", required=True, help="NAME=path, repeatable; the first one's data arguments are used")
    p.add_argument("--table", default="project/results/period_table_spec/periods.csv")
    p.add_argument("--classes", nargs="*", default=["ECL", "RR", "ROT", "CEP", "DSCT", "LPV"])
    p.add_argument("--min-points", type=int, default=400)
    p.add_argument("--hide", type=float, default=0.4, help="share of the window's points hidden at the end")
    p.add_argument("--queries", type=int, default=240)
    p.add_argument("--zoom-cycles", type=float, default=6.0, help="the time view spans this many cycles after the hidden start")
    p.add_argument("--select", choices=["classes", "estimate"], default="classes",
                   help="estimate: pick --n-good stars whose spectral frequency (first checkpoint, local centroid) is within 0.05 %% of the table period and --n-bad stars where it is off by more than 5 %%")
    p.add_argument("--n-good", type=int, default=3)
    p.add_argument("--n-bad", type=int, default=3)
    p.add_argument("--far-cycles", type=float, default=0.0, help="if > 0, the time view starts this many cycles into the hidden stretch instead of at its start")
    p.add_argument("--exclude", nargs="*", default=(), help="star ids not to pick again")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    add_device_arg(p)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    dev = get_device(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models, meta = {}, None
    for item in a.ckpt:
        name, path = item.split("=", 1)
        m, mt = load_mae(path, dev)
        models[name] = m
        meta = meta or mt
    cfg, spec = meta.cfg, meta.spec
    val = load_data(data_args_from(meta.args), splits=("validation",))["validation"]
    lookup = period_lookup_best(a.table)
    rng = np.random.default_rng(a.seed)
    picks = []
    if a.select == "classes":
        for cls in a.classes:
            cand = [i for i, r in enumerate(val) if str(r.meta.get("superclass_str")) == cls and r.t.size >= a.min_points and str(r.meta.get("id")) in lookup]
            if cand:
                picks.append(int(rng.choice(cand)))
    else:
        from romae_lc.model import spectral_frequency
        from project.mae_data import make_mask as _mm

        first = next(iter(models.values()))
        cand = [i for i, r in enumerate(val) if r.t.size >= a.min_points and str(r.meta.get("id")) in lookup and r.period and r.period < 20 and str(r.meta.get("id")) not in set(a.exclude)]
        rng.shuffle(cand)
        good, bad = [], []
        for i in cand:
            r = val[i]
            starts_, frames_, n_tok_ = object_windows(r, cfg, 0.25, cfg.max_tokens, a.seed, i)
            tok_ = spec.tokens([frames_[int(np.argmax(n_tok_))]])
            m_ = _mm(tok_.positions[:, 0], tok_.pad_mask, a.hide, "block", (1, 1), 1.0, torch.Generator().manual_seed(a.seed), "last", 1.0)
            with torch.no_grad():
                first(tok_.values.to(dev), tok_.positions.to(dev), tok_.pad_mask.to(dev), m_.to(dev))
            f_hat = float(spectral_frequency(first.spectral, None, "local")[0])
            err = abs(f_hat * float(lookup[str(r.meta["id"])]) - 1.0)
            r.meta["f_err"] = err
            if err < 5e-4 and len(good) < a.n_good:
                good.append(i)
            elif err > 5e-2 and len(bad) < a.n_bad:
                bad.append(i)
            if len(good) == a.n_good and len(bad) == a.n_bad:
                break
        picks = good + bad
    n_bands = len(spec.tokenize["band_wavelengths"])
    ts = spec.tokenize["time_scale"]
    fig, axes = plt.subplots(len(picks), 2, figsize=(13, 2.6 * len(picks)), gridspec_kw=dict(width_ratios=[2.2, 1]))
    fig.patch.set_facecolor(SURFACE)
    for row, i in enumerate(picks):
        r = val[i]
        period = float(lookup[str(r.meta["id"])])
        fitted = fit_record_templates([r], np.array([period]), n_bands=n_bands, workers=1, log=lambda *x: None)
        T = RecordTemplates(fitted, "cpu")
        starts, frames, n_tok = object_windows(r, cfg, 0.25, cfg.max_tokens, a.seed, i)
        j = int(np.argmax(n_tok))
        start, frame = float(starts[j]), frames[j]
        tok = spec.tokens([frame])
        values, positions, pad = tok.values, tok.positions, tok.pad_mask
        g = torch.Generator().manual_seed(a.seed)
        mask = make_mask(positions[:, 0], pad, a.hide, "block", (1, 1), 1.0, g, "last", 1.0)
        t_win = (positions[0, 0].float() - 1.0) * ts
        band = torch.from_numpy(np.asarray(frame[2]))[None]
        t_abs = (start + t_win)[None].double()
        rows = torch.zeros(1, dtype=torch.long)
        tv = bounded(T.eval(rows, t_abs, band), values[..., 0].float(), ~pad)
        tv = torch.where(torch.isfinite(tv), tv, values[..., 0].float())
        hid_t = t_win[mask[0] & ~pad[0]]
        t_hide = float(hid_t.min())
        if a.far_cycles > 0:
            t_hide = min(t_hide + a.far_cycles * period, float(t_win.max()) - a.zoom_cycles * period)
        zoom_hi = min(float(t_win.max()), t_hide + a.zoom_cycles * period)
        zoom_lo = max(0.0, t_hide - 3.0 * period)
        main_band = int(torch.mode(band[0][~pad[0]]).values)
        t_q = torch.linspace(t_hide, zoom_hi, a.queries)[None]
        band_q = torch.full_like(t_q, main_band, dtype=torch.long)
        target_q = bounded(T.eval(rows, (start + t_q).double(), band_q), values[..., 0].float(), ~pad)
        v2, p2, pad2, mask2, tv2, _ = augment_with_queries(values, positions, pad, mask, tv, None, t_q, band_q, target_q, spec)
        n = values.shape[1]
        ax_t, ax_f = axes[row]
        style(ax_t); style(ax_f)
        y = values[0, :, 0].numpy()
        for b in np.unique(band[0].numpy()):
            m = (band[0].numpy() == b) & ~pad[0].numpy()
            seen = m & ~mask[0].numpy()
            ax_t.plot(t_win[seen], y[seen], "o", ms=3, color=BAND_COLOUR.get(int(b), INK2), alpha=0.9, label=f"seen, band {b}" if row == 0 else None)
            ax_t.plot(t_win[m & mask[0].numpy()], y[m & mask[0].numpy()], "o", ms=3.5, mfc="none", color=BAND_COLOUR.get(int(b), INK2), alpha=0.9, label=f"hidden, band {b}" if row == 0 else None)
        tt = torch.linspace(zoom_lo, zoom_hi, 600)
        tpl = T.eval(rows, (start + tt)[None].double(), torch.full((1, 600), main_band, dtype=torch.long))[0]
        ax_t.plot(tt, tpl, "-", color=INK2, lw=0.9, alpha=0.6, label="template" if row == 0 else None)
        ax_t.axvline(t_hide, color=INK2, lw=0.8, ls=":")
        titles = []
        phase_hid = ((start + hid_t.double()) / period) % 1.0
        y_hid_obs = values[0, :, 0][mask[0] & ~pad[0]].numpy()
        tpl_hid = tv[0][mask[0] & ~pad[0]]
        ax_f.plot(phase_hid, y_hid_obs, "o", ms=3, mfc="none", color=INK2, alpha=0.7, label="hidden points" if row == 0 else None)
        for k, (name, model) in enumerate(models.items()):
            with torch.no_grad():
                outp = model(v2.to(dev), p2.to(dev), pad2.to(dev), mask2.to(dev), None, target_values=tv2.to(dev))
            pred = outp.pred[0, :, 0].float().cpu()
            isq = (torch.arange(v2.shape[1])[None].expand(1, -1) >= n)[mask2][: pred.numel()]
            tm = ((p2[0, 0].float() - 1.0) * ts)[mask2[0]]
            real = ~pad2[0][mask2[0]]
            pq, tq_ = pred[isq & real], tm[isq & real]
            ax_t.plot(tq_, pq, "-", color=colour_of(name, k), lw=1.4, alpha=ALPHA, label=name if row == 0 else None)
            ph, th = pred[~isq & real], tm[~isq & real]
            tpl_h = tv2[0][mask2[0]][~isq & real]
            ev = 1 - float(((ph - tpl_h) ** 2).mean() / tpl_h.var()) if tpl_h.numel() > 2 else float("nan")
            titles.append(f"{name} {ev:.2f}")
            ax_f.plot(((start + th.double()) / period) % 1.0, ph, "x", ms=3.5, color=colour_of(name, k), alpha=ALPHA, label=name if row == 0 else None)
        ax_t.set_xlim(zoom_lo, zoom_hi)
        est = f"; spectral estimate off by {100 * r.meta['f_err']:.3g} %" if "f_err" in r.meta else ""
        ax_t.set_title(f"{r.meta.get('class_str')}  P = {period:.5g} d  window start day {start:.0f}{est}; hidden points explained: " + ", ".join(titles), fontsize=8, loc="left", color=INK)
        ax_t.set_xlabel("days in the window", fontsize=8); ax_f.set_xlabel("phase of the hidden points (whole hidden stretch)", fontsize=8)
        ax_t.tick_params(labelsize=7); ax_f.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6.5, ncol=4, loc="upper left"); axes[0, 1].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "forecast_demo.png", dpi=130, facecolor=SURFACE)
    print(f"wrote {out / 'forecast_demo.png'}")
    # the companion: the whole light curve of every star folded on the table period, same row order
    from project import fold as _fold
    from project.eval.ls_benchmark import star_inputs
    from project.eval.period_report import fold_panel

    ncol = 2
    nrow = (len(picks) + ncol - 1) // ncol
    f2, ax2 = plt.subplots(nrow, ncol, figsize=(13, 3.2 * nrow), dpi=130, squeeze=False)
    f2.patch.set_facecolor(SURFACE)
    for k, i in enumerate(picks):
        r = val[i]
        period = float(lookup[str(r.meta["id"])])
        t_, y_, e_, b_, _ = star_inputs(r)
        r2 = _fold.fold_r2(t_, y_, e_, b_, period, 6)
        est = f", spectral estimate off by {100 * r.meta['f_err']:.3g} %" if "f_err" in r.meta else ""
        fold_panel(ax2[k // ncol, k % ncol], r, period, f"{r.meta.get('class_str')} {r.meta.get('id')}{est}", r2, template=False)
    for k in range(len(picks), nrow * ncol):
        ax2[k // ncol, k % ncol].axis("off")
    f2.suptitle("The same stars: the whole light curve folded on the table period (two cycles)", fontsize=10, color=INK, x=0.01, ha="left")
    f2.tight_layout(rect=(0, 0, 1, 0.975))
    f2.savefig(out / "folded.png", dpi=130, facecolor=SURFACE)
    print(f"wrote {out / 'folded.png'}; stars: " + " ".join(str(val[i].meta.get("id")) for i in picks))


if __name__ == "__main__":
    main()
