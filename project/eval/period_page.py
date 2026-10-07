"""Build the period artifact page (https://claude.ai/artifact/JoubE7CVCj2KSB1QtGzwub)
from the result folders. Sections: hit rates (table, two scatter panels); clean
folds (as good as the catalogue, and cleaner than it; no fitted wave); the
cost comparison (hit rate against trials per threshold, the trials table,
runtime); analysis (bars by class, hit rate against the period per threshold,
heat maps); the encoders. One figure per tolerance, little text:

    python -m project.eval.period_page --report <period_report dir> --bench <ls_benchmark dir> \
        --runtime <runtime dir> [--sweep <budget_sweep dir>] --out project/results/period_page/index.html

Only the collaborator's astropy search is shown, named Lomb-Scargle.
"""
import argparse, base64, csv, html, json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project.eval.budget_sweep import plot_budget
from project.eval.ls_benchmark import heat
from project.eval.period_report import fold_panel, plot_hit_vs_period, rel_error
from project.plot_period_examples import CLASS_COLOUR, CLASS_NAME, INK, INK2, OTHER, SURFACE, style

HEAD = '<title>ZTF Period Figures</title>\n<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Figtree:wght@500;700&family=Public+Sans:wght@400;600&display=swap">\n<style>\n/* layout: one reading column for text, figures break out wider on a light paper panel */\n:root {\n  --bg: #f3f5f8; --fg: #14202e; --muted: #526173; --rule: #d5dbe3;\n  --accent: #2a78d6; --panel: #ffffff; --paper: #fcfcfb;\n  --display: "Figtree", "Segoe UI", system-ui, sans-serif;\n  --body: "Public Sans", "Segoe UI", system-ui, sans-serif;\n}\n@media (prefers-color-scheme: dark) {\n  :root:not([data-theme="light"]) { --bg: #0f151c; --fg: #e8edf3; --muted: #9aa8b8; --rule: #2a3542; --accent: #6aa7ee; --panel: #171f29; --paper: #fcfcfb; color-scheme: dark }\n}\n:root[data-theme="dark"] { --bg: #0f151c; --fg: #e8edf3; --muted: #9aa8b8; --rule: #2a3542; --accent: #6aa7ee; --panel: #171f29; --paper: #fcfcfb; color-scheme: dark }\nbody { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 16px; line-height: 1.6; padding-inline: 20px; padding-block: 36px 64px; }\nmain { max-width: 1080px; margin-inline: auto; display: flex; flex-direction: column; gap: 40px; }\n.text { max-width: 66ch; display: flex; flex-direction: column; gap: 12px; }\nh1 { font-family: var(--display); font-weight: 700; font-size: clamp(1.7rem, 4vw, 2.3rem); line-height: 1.15; margin: 0; text-wrap: balance; }\nh2 { font-family: var(--display); font-weight: 700; font-size: 1.3rem; line-height: 1.25; margin: 0; text-wrap: balance; }\np { margin: 0; }\n.lede { color: var(--muted); font-size: 1.05rem; }\nsection { display: flex; flex-direction: column; gap: 16px; }\nfigure { margin: 0; background: var(--paper); border: 1px solid var(--rule); border-radius: 6px; padding: 8px; overflow-x: auto; }\nfigure img { display: block; width: 100%; min-width: 0; height: auto; }\nul { margin: 0; padding-left: 1.2em; display: flex; flex-direction: column; gap: 6px; }\n.tablewrap { overflow-x: auto; }\ntable { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 0.95rem; }\nth, td { text-align: left; padding: 7px 18px 7px 0; border-bottom: 1px solid var(--rule); }\nth { font-family: var(--display); font-weight: 500; color: var(--muted); font-size: 0.8rem; letter-spacing: 0.04em; text-transform: uppercase; }\ntd.n { text-align: right; padding-right: 24px; }\ncode { font-size: 0.92em; background: var(--panel); border: 1px solid var(--rule); border-radius: 4px; padding: 1px 5px; }\n.note { border-left: 3px solid var(--accent); padding-left: 14px; color: var(--muted); }\n</style>\n'
ENC_TAB = '<h2>The encoders behind these numbers</h2>\n      <p>Period hit rate of the bin read-out on the pooled latent, 5,346 validation stars, 240 bins.</p>\n    </div>\n    <div class="tablewrap"><table>\n      <tr><th>encoder</th><th>steps</th><th>within 1%</th><th>within 10%</th><th>within 20%</th></tr>\n      <tr><td>light, plain</td><td class=n>50,000</td><td class=n>0.36</td><td class=n>0.66</td><td class=n></td></tr>\n      <tr><td>wide, plain</td><td class=n>110,000</td><td class=n>0.58</td><td class=n>0.71</td><td class=n>0.80</td></tr>\n      <tr><td>light, spectral layer</td><td class=n>50,000</td><td class=n>0.35</td><td class=n>0.76</td><td class=n>0.85</td></tr>\n      <tr><td>wide, spectral layer</td><td class=n>54,000</td><td class=n>0.69</td><td class=n>0.77</td><td class=n>0.85</td></tr>\n      <tr><td>wide, spectral layer</td><td class=n>100,000</td><td class=n>0.70</td><td class=n>0.77</td><td class=n>0.84</td></tr>\n    </table></div>\n    <div class="text">\n      <p>The spectral layer sums each window\'s learned token features with the phasor at every point\'s own time over 20,000 trial frequencies, and hands a summary of that spectrum to the latent. It is the one encoder-side change that moved the period after about fifteen others did not.</p>\n    </div>\n  </section>\n'
FIGS = Path("project/results/period_page/figs")
C = dict(model="#2a78d6", refined="#1baf7a", cands="#0b0b0b", ls="#eb6834")
TOLS = (0.2, 0.1, 0.01, 0.001, 0.0001)
TOL_LABEL = ["20 %", "10 %", "1 %", "0.1 %", "0.01 %"]
CLS = {"ECL": "eclipsing", "RR": "RR Lyrae", "ROT": "rotation", "CEP": "Cepheid", "DSCT": "δ Scuti", "LPV": "long-period"}


def img(path, alt):
    b = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{b}"></figure>'


def pct(x, d=0):
    return f"{100 * x:.{d}f} %"


def rate(p_hat, p_true, tol, alias=False):
    return float(np.mean(np.where(np.isfinite(p_hat), rel_error(p_hat, p_true, alias) < tol, False)))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, facecolor=SURFACE)
    plt.close(fig)
    return FIGS / name



# ---------------------------------------------------------------- figures

def fig_scatter_two(p_true, preds: dict, sup):
    """Predicted against catalogue period, log axes, one panel per read-out."""
    names = list(preds)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.6), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    groups = [("other", ~np.isin(sup, list(CLASS_COLOUR)), OTHER)] + [(g, sup == g, c) for g, c in CLASS_COLOUR.items()]
    lim = np.array([0.03, 1000.0])
    for ax, name in zip(axes[0], names):
        p = preds[name]
        ok = np.isfinite(p) & (p > 0)
        style(ax)
        for g, m, c in groups:
            mm = m & ok
            ax.scatter(p_true[mm], p[mm], s=2.5, color=c, alpha=0.4, linewidths=0, label=f"{CLASS_NAME[g]} ({int(m.sum())})")
        for k in (1.0, 2.0, 0.5):
            ax.plot(lim, lim * k, color=INK2, lw=0.7, ls="-" if k == 1 else (0, (4, 3)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.set_title(f"{name}\n{np.mean(np.abs(p / p_true - 1) < 0.1):.1%} within 10 %, {np.mean(np.abs(p / p_true - 1) < 0.01):.1%} within 1 %", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("catalogue period (days)", fontsize=8, color=INK2)
    axes[0][0].set_ylabel("predicted period (days)", fontsize=8, color=INK2)
    axes[0][0].legend(fontsize=6.5, frameon=False, loc="upper left", markerscale=4, labelcolor=INK2)
    return save(fig, "scatter_two.png")


def fig_bars_by_class(M, ls, groups, counts, key, label):
    """Grouped bars: the hit rate per class for every method at one tolerance."""
    methods = [("model alone", "model", C["model"]), ("model + fine search", "model_refined", C["refined"]), ("model candidates + search", "model_cands", C["cands"]),
               ("Lomb-Scargle", ls, C["ls"])]
    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    x = np.arange(len(groups))
    w = 0.8 / (len(methods) + 1)
    for i, (name, m, col) in enumerate(methods):
        ax.bar(x + (i - len(methods) / 2) * w + w / 2, [M[m]["by_superclass"][g][key] for g in groups], w, color=col, label=name)
    i = len(methods)
    ax.bar(x + (i - len(methods) / 2) * w + w / 2, [M[ls]["by_superclass"][g][key.replace("within_", "alias_")] for g in groups], w, color=C["ls"], alpha=0.45,
           hatch="///", edgecolor=SURFACE, label="Lomb-Scargle, double and half accepted")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{CLS[g]}\n({counts[g]})" for g in groups], fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of the stars", fontsize=8, color=INK2)
    ax.set_title(f"Hit rate {label}, by class", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper right", labelcolor=INK2)
    return save(fig, f"bars_{key}.png")


def fig_hit_vs_period_one(p_true, sup, series, tol, label):
    """Hit rate against the catalogue period at one tolerance."""
    ser = {}
    for name, (p, col, ls_, alias) in series.items():
        ser[name] = (np.where(np.isfinite(p), rel_error(p, p_true, alias) < tol, False), col, ls_)
    plot_hit_vs_period(FIGS / f"hit_vs_period_{tol:g}.png", p_true, ser, sup, f"Hit rate {label} against the catalogue period")
    return FIGS / f"hit_vs_period_{tol:g}.png"


def fig_cost_one(SW, B, key, label):
    """Hit rate against trial frequencies at one tolerance: Lomb-Scargle muted, the model's search in blue."""
    SURF, GRID = SURFACE, "#e4e3df"
    LS, MODEL = "#b9a99a", ["#1d6fd1", "#0b4fa0", "#062f63"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
    ax.tick_params(colors=INK2, labelsize=7.5)
    labels = []
    for tag, name, ls_ in (("mb", "Lomb-Scargle, multiband", "-"), ("1b", "Lomb-Scargle, best band", (0, (3, 2)))):
        ms = sorted((m for m in B["methods"] if m.startswith(f"astropy_{tag}@")), key=lambda m: int(m.split("@")[1]))
        x = [B["methods"][m]["trials_median"] for m in ms]
        y = [B["methods"][m][key] for m in ms]
        ax.plot(x, y, color=LS, lw=1.2, alpha=0.8, marker="o", ms=3, ls=ls_)
        labels.append((x[-1], y[-1], name, INK2))
    tops = sorted({v["top"] for v in SW["settings"].values()})
    for c, k in zip(MODEL, tops):
        items = sorted((v for v in SW["settings"].values() if v["top"] == k), key=lambda v: v["trials_median"])
        x = [v["trials_median"] for v in items]
        y = [v[key] for v in items]
        ax.plot(x, y, color=c, lw=1.4, alpha=0.85, marker="o", ms=3.5)
        labels.append((x[-1], y[-1], f"model, top {k} seed{'s' if k > 1 else ''}", c))
    for group in ([l for l in labels if l[3] == INK2], [l for l in labels if l[3] != INK2]):
        placed = []
        for xe, ye, text, col in sorted(group, key=lambda l: l[1]):
            y_lab = ye if not placed else max(ye, placed[-1] + 0.05)
            placed.append(y_lab)
            ax.annotate(text, (xe, ye), xytext=(xe * 1.12, y_lab), textcoords="data", fontsize=7, color=col, va="center",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.5, alpha=0.6) if abs(y_lab - ye) > 1e-3 else None)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    lo = min(v["trials_median"] for v in SW["settings"].values())
    hi = max(B["methods"][m]["trials_median"] for m in B["methods"] if m.startswith("astropy_"))
    ax.set_xlim(lo * 0.7, hi * 12)
    ax.set_xlabel("trial frequencies per star", fontsize=8, color=INK2)
    ax.set_ylabel("share of 1,000 stars", fontsize=8, color=INK2)
    ax.set_title(f"Hit rate {label} against the search budget (the model's points: search width 1, 3, 10, 30 % of its period)", fontsize=9, color=INK, loc="left")
    return save(fig, f"cost_{key}.png")


def fig_runtime(T, budgets):
    S = T["steps"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    for tag, name, ls_ in (("mb", "Lomb-Scargle, multiband (CPU)", "-"), ("1b", "Lomb-Scargle, best band (CPU)", (0, (4, 2)))):
        ax.plot(budgets, [1e3 * S[f"astropy_{tag}@{b}"]["median"] for b in budgets], color=C["ls"], ls=ls_, marker="o", ms=4, lw=1.5, label=name)
    one, batched = 1e3 * T["model_total"]["median"], 1e3 * T["model_total_batched_per_star"]
    ax.axhline(one, color=C["model"], lw=1.4, label=f"model, whole path, one star at a time ({one:.0f} ms)")
    ax.axhline(batched, color=C["model"], lw=1.4, ls=(0, (4, 2)), label=f"model, whole path, encoder batched over stars ({batched:.0f} ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b:,}" for b in budgets], fontsize=8)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("trial frequencies per star (Lomb-Scargle)", fontsize=8, color=INK2)
    ax.set_ylabel("milliseconds per star (median)", fontsize=8, color=INK2)
    ax.set_title("Runtime per star against the search budget", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left", labelcolor=INK2)
    return save(fig, "runtime_vs_trials.png")


def fold_rows_figure(path, rows, heading):
    """Two panels per star, catalogue period and the model's period, points only."""
    fig, axes = plt.subplots(len(rows), 2, figsize=(8.2, 2.3 * len(rows)), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax_row, (r, label, cols) in zip(axes, rows):
        for ax, (p, title, r2) in zip(ax_row, cols):
            fold_panel(ax, r, p, title, r2, template=False)
        ax_row[0].set_ylabel(f"{label}\n\nbrightness", fontsize=7.5, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("phase (shown twice)", fontsize=8, color=INK2)
    fig.suptitle(heading, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- page

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--ckpt", default="project/runs/maew_spec/mae.pt", help="for the data arguments (the fold demos read the light curves)")
    ap.add_argument("--variants", nargs="*", default=[], help="name=runtime folder pairs with a batch sweep for the batched table")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rep, ben, run = Path(a.report), Path(a.bench), Path(a.runtime)
    R, B, T, SW = (json.load(open(p / "results.json")) for p in (rep, ben, run, Path(a.sweep)))
    d = np.load(rep / "per_star.npz", allow_pickle=True)
    bd = np.load(ben / "per_star.npz", allow_pickle=True)
    rows = list(csv.DictReader(open(rep / "better_than_catalogue.csv")))
    M, BM, S = R["methods"], B["methods"], T["steps"]
    n = R["n_stars"]
    ls = [m for m in M if m.startswith("astropy_mb@")][0]
    ls_budget = int(ls.split("@")[1])
    p_true, sup, fine = d["p_true"], d["superclass"].astype(str), d["fine"].astype(str)
    pm, pr, pc, pl = d["p_model"], d["p_model_refined"], d["p_model_cands"], d[f"p_{ls}"]
    r2 = {k: d[f"r2_{k}"] for k in ("catalogue", "catalogue_sharp", "model", "model_refined", "model_cands")}
    groups = [g for g in CLS if g in M["model"]["by_superclass"]]
    counts = {g: int((sup == g).sum()) for g in groups}
    thresholds = (("within_0.1", 0.1, "within 10 %"), ("within_0.01", 0.01, "within 1 %"), ("within_0.0001", 0.0001, "within 0.01 %"))

    # the light curves, for the fold demos
    from project.common import data_args_from, load_data, load_encoder
    _, meta = load_encoder(a.ckpt, "cpu")
    val = load_data(data_args_from(meta.args), splits=("validation",))["validation"]
    idx = d["index"]

    # ---- 1. hit rates
    def hits_row(name, m):
        v = M[m]
        return f"<tr><td>{name}</td>" + "".join(f"<td class=n>{v[f'within_{t:g}']:.3f}</td>" for t in TOLS) + f"<td class=n>{v['alias_0.0001']:.3f}</td><td class=n>{v['trials_median']:,.0f}</td></tr>"
    table1 = ["<tr><th>method</th><th>within 20 %</th><th>10 %</th><th>1 %</th><th>0.1 %</th><th>0.01 %</th><th>0.01 %, double and half accepted</th><th>trials per star</th></tr>",
              hits_row("model alone", "model"), hits_row("model + fine search", "model_refined"), hits_row("model candidates + search", "model_cands"),
              hits_row(f"Lomb-Scargle, multiband, {ls_budget:,} trials", ls)]
    f_sc = fig_scatter_two(p_true, {"model alone": pm, "model + fine search": pr}, sup)

    # ---- 2. folds: as good as the catalogue (typical hit per class), and cleaner than it
    err_c = rel_error(pc, p_true)
    good = []
    for cls in ("EW/EB", "EA", "RRAB", "RRC", "ROT", "CEP", "DSCT", "LPV"):
        m = (fine == cls) & (err_c < 1e-4) & (d["n_points"] >= 300)
        if m.sum() == 0:
            m = (fine == cls) & (err_c < 1e-2)
        if m.sum() == 0:
            continue
        cand = np.flatnonzero(m)
        good.append(int(cand[np.argsort(np.abs(r2["model_cands"][cand] - np.median(r2["model_cands"][cand])))[0]]))
    def two(j):
        pb = pc[j]
        off = pb / p_true[j] - 1
        how = f"{off:+.4%} off" if abs(off) < 0.1 else f"{pb / p_true[j]:.3f} x the catalogue value"
        return [(p_true[j], "catalogue period", r2["catalogue"][j]), (pb, f"model period ({how})", r2["model_cands"][j])]
    f_good = fold_rows_figure(FIGS / "folds_good.png", [(val[int(idx[j])], f"{fine[j]}\n{d['ztf_id'][j]}", two(j)) for j in good],
                              "A typical hit of every class: the catalogue period and the model's period fold the same way")
    diff = [r for r in rows if r["verdict"] == "different period"]
    imprecise = [r for r in rows if r["verdict"] != "different period"]
    pick = [int(r["val_index"]) for r in diff[:6]] + [int(r["val_index"]) for r in imprecise[:3]]
    pos = {int(i): j for j, i in enumerate(idx)}
    f_better = fold_rows_figure(FIGS / "folds_cleaner.png",
                                [(val[i], f"{fine[pos[i]]}\n{d['ztf_id'][pos[i]]}\n{'other period' if pos[i] in [pos[int(r['val_index'])] for r in diff[:6]] else 'same period, sharper'}", two(pos[i])) for i in pick],
                                "Stars where the model's period folds the light curve more cleanly than the catalogue value")
    F = R["fold"]
    bt = R["better"]
    table3 = ["<tr><th>ZTF id</th><th>class</th><th>ratio to the catalogue</th><th>catalogue P (d)</th><th>model P (d)</th><th>fold R2 catalogue</th><th>catalogue sharpened</th><th>model</th><th>points</th></tr>"]
    for r in diff[:30]:
        table3.append(f"<tr><td><code>{r['ztf_id']}</code></td><td>{r['class']}</td><td>{r['kind'].replace('different', 'other')}</td><td class=n>{float(r['p_catalogue_d']):.6f}</td><td class=n>{float(r['p_best_d']):.6f}</td>"
                      f"<td class=n>{float(r['r2_catalogue']):.2f}</td><td class=n>{float(r['r2_catalogue_sharp']):.2f}</td><td class=n>{float(r['r2_best']):.2f}</td><td class=n>{r['n_points']}</td></tr>")

    # ---- 3. cost: hit rate against trials, one figure per threshold; the trials table; runtime
    f_cost = [fig_cost_one(SW, B, key, label) for key, _, label in thresholds]
    rows_ = ["<tr><th>search</th><th>trials per star</th><th>within 10 %</th><th>within 1 %</th><th>within 0.01 %</th><th>0.01 %, double and half accepted</th></tr>"]
    for name, v in sorted(SW["settings"].items(), key=lambda kv: (kv[1]["top"], kv[1]["rel"])):
        rows_.append(f"<tr><td>model, top {v['top']} seed{'s' if v['top'] > 1 else ''}, width {v['rel']:.0%}</td><td class=n>{v['trials_median']:,.0f}</td><td class=n>{v['within_0.1']:.2f}</td><td class=n>{v['within_0.01']:.2f}</td><td class=n>{v['within_0.0001']:.2f}</td><td class=n>{v['alias_tolerant_0.0001']:.2f}</td></tr>")
    bench_budgets = sorted({int(m.split("@")[1]) for m in BM if m.startswith("astropy_mb@")})
    for b in bench_budgets:
        if b >= 20000:
            v = BM[f"astropy_mb@{b}"]
            rows_.append(f"<tr><td>Lomb-Scargle, multiband, {b:,}</td><td class=n>{v['trials_median']:,.0f}</td><td class=n>{v['within_0.1']:.2f}</td><td class=n>{v['within_0.01']:.2f}</td><td class=n>{v['within_0.0001']:.2f}</td><td class=n>{v['alias_tolerant_0.0001']:.2f}</td></tr>")
    sweep_table = '<div class="tablewrap"><table>' + "".join(rows_) + "</table></div>"
    rt_budgets = sorted({int(k.split("@")[1]) for k in S if k.startswith("astropy_mb@") and S[k]["n"]})
    f_rt = fig_runtime(T, rt_budgets)
    def trow(name, key):
        v = S[key]
        return f"<tr><td>{name}</td><td class=n>{1e3 * v['median']:.1f}</td><td class=n>{1e3 * v['p16']:.1f} to {1e3 * v['p84']:.1f}</td></tr>"
    table5 = ["<tr><th>step</th><th>median ms per star</th><th>16 to 84 % range</th></tr>",
              trow("model: windows and tokens (CPU)", "cut"), trow("model: encoder forward, one star per batch (GPU)", "encode"), trow("model: pooling and read-out", "read"),
              trow("model: fine search within 10 % (GPU)", "search"),
              f"<tr><td><strong>model: the whole path, one star at a time</strong></td><td class=n><strong>{1e3 * T['model_total']['median']:.1f}</strong></td><td class=n>{1e3 * T['model_total']['p16']:.1f} to {1e3 * T['model_total']['p84']:.1f}</td></tr>",
              f"<tr><td><strong>model: the whole path, encoder batched over stars</strong></td><td class=n><strong>{1e3 * T['model_total_batched_per_star']:.1f}</strong></td><td class=n></td></tr>"]
    for b in rt_budgets:
        table5 += [trow(f"Lomb-Scargle, best band, {b:,} trials (CPU)", f"astropy_1b@{b}"), trow(f"Lomb-Scargle, multiband, {b:,} trials (CPU)", f"astropy_mb@{b}")]
    variants = []
    for item in a.variants:
        name, path = item.split("=", 1)
        if (Path(path) / "results.json").is_file():
            variants.append((name, json.load(open(Path(path) / "results.json"))))
    batched_html = ""
    if variants:
        vr = ["<tr><th>encoder variant</th><th>best batch (windows)</th><th>GPU ms per star</th><th>end to end ms per star</th><th>stars per second</th><th>peak GPU memory GB</th><th>latent change vs bf16</th><th>own period peak within 1 %</th></tr>"]
        for name, V in variants:
            bb = V["best_batch"]
            b = V["batch_sweep"].get(str(bb), V["batch_sweep"].get(bb))
            acc = V.get("accuracy", {})
            chg = acc.get("latent_rel_change_median")
            vr.append(f"<tr><td>{name}</td><td class=n>{bb:,}</td><td class=n>{1e3 * b['gpu_s_per_star']:.1f}</td><td class=n>{1e3 * b['e2e_s_per_star']:.1f}</td><td class=n>{b['stars_per_s']:,.0f}</td><td class=n>{b['peak_mem_gb']:.0f}</td>"
                      f"<td class=n>{'' if chg is None else f'{100 * chg:.1f} %'}</td><td class=n>{acc.get('spectral_peak_within_1pct', float('nan')):.3f}</td></tr>")
        batched_html = '<div class="text"><p>Batched over many stars, as a survey run would be, in bf16 and with the transformer\'s linear layers quantised (torchao). The last two columns compare each variant with bf16 on the same windows.</p><div class="tablewrap"><table>' + "".join(vr) + "</table></div></div>"

    # ---- 4. analysis: bars by class, hit rate against the period per threshold, heat maps
    f_bars = [fig_bars_by_class(M, ls, groups, counts, key, label) for key, _, label in thresholds]
    series = {"model alone": (pm, C["model"], "-", False), "model + fine search": (pr, C["refined"], "-", False), "model candidates + search": (pc, C["cands"], "-", False),
              f"Lomb-Scargle {ls_budget:,} trials": (pl, C["ls"], "-", False), f"Lomb-Scargle {ls_budget:,} trials, double and half accepted": (pl, C["ls"], (0, (4, 2)), True)}
    f_hvp = [fig_hit_vs_period_one(p_true, sup, series, tol, label) for _, tol, label in thresholds]
    bls = f"p_astropy_mb@{max(bench_budgets)}"
    hit = {"model + fine search": np.abs(bd["p_model_refined"] / bd["p_true"] - 1) < 1e-4, f"Lomb-Scargle, {max(bench_budgets):,} trials": np.abs(bd[bls] / bd["p_true"] - 1) < 1e-4}
    ok = np.isfinite(bd["points_per_window"]) & np.isfinite(bd["n_windows"])
    heat(FIGS / "heat_points_windows.png", {k: v[ok] for k, v in hit.items()}, bd["points_per_window"][ok], bd["n_windows"][ok],
         np.array([0, 30, 60, 120, 1000]), np.array([0, 10, 20, 40, 200]), "points per window", "valid windows", "hit rate within 0.01 %")
    heat(FIGS / "heat_cadence_span.png", hit, bd["cadence"], bd["span"], np.array([0, 1, 2, 4, 10, 1000]), np.array([0, 1000, 2000, 2500, 3000, 5000]),
         "median gap between observations (days)", "baseline (days)", "hit rate within 0.01 %")

    cross = B["crossing"]["[alias-tolerant 0.01%] astropy, multiband to match model_refined"]
    page = f"""{HEAD}<main>
  <div class="text">
    <h1>Period from the model, star by star</h1>
    <p class="lede">All {n:,} validation stars from ZTF, whole light curves. The encoder (wide, with the spectral layer) gives a rough period; a fine search near it makes it sharp. Lomb-Scargle throughout is the astropy search on a uniform frequency grid. Updated 6 October.</p>
  </div>

  <section>
    <div class="text">
      <h2>Hit rates</h2>
      <p>A hit is a period within the tolerance of the catalogue period. The fine search runs within 10 % of the model's period; the candidate search also sharpens the read-out's other top bins and the double and the half, and the fold decides.</p>
      <div class="tablewrap"><table>{''.join(table1)}</table></div>
    </div>
    {img(f_sc, "Predicted against catalogue period on log axes, for the model alone and for the model with the fine search")}
  </section>

  <section>
    <div class="text">
      <h2>Phase-folded light curves</h2>
      <p>Each star folded on the catalogue period and on the model's period, two cycles. First a typical hit of every class (the star nearest the class median of the fold quality, not the best one).</p>
    </div>
    {img(f_good, "Phase-folded light curves of one typical star per class on the catalogue period and on the model period")}
    <div class="text">
      <p>Fold quality is the adjusted R2 of a 6-harmonic wave. The catalogue period was also sharpened by the same fine search within 0.2 % of its value, which is the fair reference over a 2,700 day baseline.</p>
      <div class="tablewrap"><table>
        <tr><th>fold R2, median over the stars</th><th>catalogue</th><th>catalogue sharpened</th><th>model + fine search</th><th>model candidates + search</th></tr>
        <tr><td>whole light curve</td><td class=n>{F['catalogue']['median']:.3f}</td><td class=n>{F['catalogue_sharp']['median']:.3f}</td><td class=n>{F['model_refined']['median']:.3f}</td><td class=n>{F['model_cands']['median']:.3f}</td></tr>
      </table></div>
      <p>For {bt['n']:,} stars the model's period folds more cleanly than the catalogue value as listed (R2 higher by more than 0.1, reaching 0.5). For {bt['n_catalogue_imprecise']:,} of them the catalogue period is the same one made sharper; for {bt['n_different']:,} it is another period that also folds better than the sharpened catalogue value. The full list is in <a href="better_than_catalogue.csv">better_than_catalogue.csv</a>.</p>
    </div>
    {img(f_better, "Phase-folded light curves of stars where the model period folds more cleanly than the catalogue value: other periods first, then the same period made sharper")}
    <div class="text">
      <p class="note">A cleaner fold is evidence, not proof: on night-only sampling a period one cycle per day away folds almost as well, and the half period of a symmetric eclipsing binary folds like the full one.</p>
      <div class="tablewrap"><table>{''.join(table3)}</table></div>
    </div>
  </section>

  <section>
    <div class="text">
      <h2>Cost: hit rate against the search budget</h2>
      <p>1,000 validation stars. Lomb-Scargle gets a growing grid of trial frequencies. The model's budget grows with the width of its fine search (1, 3, 10, 30 % of its period) and the number of seed bins it sharpens (1, 3, 5). Within 0.01 % Lomb-Scargle never reaches the model; with the double and the half accepted it needs {cross:,.0f} trials to match the model's {BM['model_refined']['trials_median']:,.0f}.</p>
    </div>
    {img(f_cost[0], "Hit rate within 10 percent against trial frequencies per star for Lomb-Scargle and for the model's search")}
    {img(f_cost[1], "Hit rate within 1 percent against trial frequencies per star for Lomb-Scargle and for the model's search")}
    {img(f_cost[2], "Hit rate within 0.01 percent against trial frequencies per star for Lomb-Scargle and for the model's search")}
    <div class="text">{sweep_table}</div>
    <div class="text">
      <p>Wall-clock per star on {T['n_stars']} stars (median {T['points']['median']:.0f} points, {T['windows']['median']:.0f} windows), one {html.escape(T['gpu'] or 'GPU')} and {T['cpu_threads']} CPU threads; Lomb-Scargle runs on the CPU.</p>
      <div class="tablewrap"><table>{''.join(table5)}</table></div>
    </div>
    {img(f_rt, "Milliseconds per star against trial frequencies for Lomb-Scargle, with the model path as horizontal lines")}
    {batched_html}
  </section>

  <section>
    <div class="text">
      <h2>Analysis</h2>
      <p>Hit rate by class, at three tolerances.</p>
    </div>
    {img(f_bars[0], "Bar chart of the hit rate within 10 percent per class for every method")}
    {img(f_bars[1], "Bar chart of the hit rate within 1 percent per class for every method")}
    {img(f_bars[2], "Bar chart of the hit rate within 0.01 percent per class for every method")}
    <div class="text"><p>Hit rate against the catalogue period, in log-spaced bins; the grey bars count the stars per bin.</p></div>
    {img(f_hvp[0], "Hit rate within 10 percent against the catalogue period")}
    {img(f_hvp[1], "Hit rate within 1 percent against the catalogue period")}
    {img(f_hvp[2], "Hit rate within 0.01 percent against the catalogue period")}
    <div class="text"><p>Where the hits are: within 0.01 %, over points per window against valid windows, and over observation cadence against baseline, model with the fine search against Lomb-Scargle at {max(bench_budgets):,} trials (1,000 stars).</p></div>
    {img(FIGS / "heat_points_windows.png", "Heat maps of the hit rate over points per window and number of windows")}
    {img(FIGS / "heat_cadence_span.png", "Heat maps of the hit rate over observation cadence and baseline")}
  </section>

  <section>
    <div class="text">
      {ENC_TAB}
  </section>
</main>
"""
    Path(a.out).write_text(page)
    print(f"wrote {a.out}: {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
