"""Build the period artifact page (https://claude.ai/artifact/JoubE7CVCj2KSB1QtGzwub)
from the result folders, with its own line plots of every table:

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
from project.eval.period_report import plot_hit_vs_period, rel_error
from project.plot_period_examples import INK, INK2, SURFACE, style

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


def fig_hit_vs_tol(M, ls):
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    x = np.arange(len(TOLS))
    for name, m, col, ls_ in (("model alone", "model", C["model"], "-"), ("model + fine search", "model_refined", C["refined"], "-"),
                              ("model candidates + search", "model_cands", C["cands"], "-"), ("Lomb-Scargle", ls, C["ls"], "-"),
                              ("Lomb-Scargle, double and half accepted", ls, C["ls"], (0, (4, 2))), ("model candidates, double and half accepted", "model_cands", C["cands"], (0, (4, 2)))):
        key = "alias_" if "accepted" in name else "within_"
        ax.plot(x, [M[m][f"{key}{t:g}"] for t in TOLS], color=col, ls=ls_, marker="o", ms=4, lw=1.5, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"within {t}" for t in TOL_LABEL], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of the stars", fontsize=8, color=INK2)
    ax.set_title("Hit rate against the tolerance, every validation star", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left", labelcolor=INK2)
    return save(fig, "hit_vs_tol.png")


def fig_hit_by_class(M, ls, groups, counts):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(groups))
    for ax, (tol, lab) in zip(axes, (("within_0.1", "within 10 %"), ("within_0.01", "within 1 %"), ("within_0.0001", "within 0.01 %"))):
        style(ax)
        for name, m, col in (("model alone", "model", C["model"]), ("model + fine search", "model_refined", C["refined"]), ("Lomb-Scargle", ls, C["ls"])):
            ax.plot(x, [M[m]["by_superclass"][g][tol] for g in groups], color=col, marker="o", ms=4, lw=1.5, label=name)
        ax.plot(x, [M[ls]["by_superclass"][g][tol.replace("within_", "alias_")] for g in groups], color=C["ls"], ls=(0, (4, 2)), marker="o", ms=3, lw=1.2, label="Lomb-Scargle, double and half accepted")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{CLS[g]}\n({counts[g]})" for g in groups], fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"hit rate {lab}, by class", fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("share of the stars", fontsize=8, color=INK2)
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower left", labelcolor=INK2)
    return save(fig, "hit_by_class.png")


def fig_bench(BM, budgets):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, lab) in zip(axes, (("within_0.0001", "within 0.01 %"), ("alias_tolerant_0.0001", "within 0.01 %, double and half accepted"))):
        style(ax)
        for tag, name, ls_ in (("mb", "Lomb-Scargle, multiband", "-"), ("1b", "Lomb-Scargle, best band", (0, (4, 2)))):
            ax.plot(budgets, [BM[f"astropy_{tag}@{b}"][key] for b in budgets], color=C["ls"], ls=ls_, marker="o", ms=4, lw=1.5, label=name)
        v = BM["model_refined"]
        ax.axhline(v[key], color=C["refined"], lw=1.2, ls=(0, (2, 2)))
        ax.plot([v["trials_median"]], [v[key]], marker="*", ms=12, color=C["refined"], label=f"model + fine search ({v['trials_median']:,.0f} trials)", ls="none")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("trial frequencies per star", fontsize=8, color=INK2)
        ax.set_title(f"hit rate {lab}", fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("share of 1,000 stars", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left", labelcolor=INK2)
    return save(fig, "bench_vs_trials.png")


def fig_runtime(T, budgets):
    S = T["steps"]
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=150)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--sweep", default=None, help="budget_sweep folder: the model over its own budget on the benchmark figure")
    ap.add_argument("--variants", nargs="*", default=[], help="name=runtime folder pairs with a batch sweep (bf16, quantised, graphs) for the batched table and plot")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rep, ben, run = Path(a.report), Path(a.bench), Path(a.runtime)
    R, B, T = (json.load(open(p / "results.json")) for p in (rep, ben, run))
    d = np.load(rep / "per_star.npz", allow_pickle=True)
    bd = np.load(ben / "per_star.npz", allow_pickle=True)
    rows = list(csv.DictReader(open(rep / "better_than_catalogue.csv")))
    head = HEAD
    M, BM, S = R["methods"], B["methods"], T["steps"]
    n = R["n_stars"]
    ls = [m for m in M if m.startswith("astropy_mb@")][0]
    ls_budget = int(ls.split("@")[1])
    p_true, sup = d["p_true"], d["superclass"].astype(str)
    groups = [g for g in CLS if g in M["model"]["by_superclass"]]
    counts = {g: int((sup == g).sum()) for g in groups}
    bench_budgets = sorted({int(m.split("@")[1]) for m in BM if m.startswith("astropy_mb@")})
    rt_budgets = sorted({int(k.split("@")[1]) for k in S if k.startswith("astropy_mb@") and S[k]["n"]})

    # ---- figures drawn here
    f_tol = fig_hit_vs_tol(M, ls)
    f_cls = fig_hit_by_class(M, ls, groups, counts)
    pr, pc, pl, pm = d["p_model_refined"], d["p_model_cands"], d[f"p_{ls}"], d["p_model"]
    series = {"model alone, within 10 %": (rel_error(pm, p_true) < 0.1, C["model"], (0, (4, 2))),
              "model alone, within 1 %": (rel_error(pm, p_true) < 0.01, C["model"], "-"),
              "model + fine search, within 0.01 %": (rel_error(pr, p_true) < 1e-4, C["refined"], "-"),
              "model candidates + search, double and half accepted, 0.01 %": (rel_error(pc, p_true, True) < 1e-4, C["cands"], (0, (4, 2))),
              f"Lomb-Scargle {ls_budget:,} trials, within 0.01 %": (np.where(np.isfinite(pl), rel_error(pl, p_true) < 1e-4, False), C["ls"], "-"),
              f"Lomb-Scargle {ls_budget:,} trials, double and half accepted, 0.01 %": (np.where(np.isfinite(pl), rel_error(pl, p_true, True) < 1e-4, False), C["ls"], (0, (4, 2)))}
    plot_hit_vs_period(FIGS / "hit_vs_period.png", p_true, series, sup, "Hit rate against the catalogue period, every validation star")
    if a.sweep:
        SW = json.load(open(Path(a.sweep) / "results.json"))
        plot_budget(FIGS / "bench_vs_trials.png", SW, B)
        f_bench = FIGS / "bench_vs_trials.png"
    else:
        SW = None
        f_bench = fig_bench(BM, bench_budgets)
    f_rt = fig_runtime(T, rt_budgets)
    bls = f"p_astropy_mb@{max(bench_budgets)}"
    hit = {"model + fine search": np.abs(bd["p_model_refined"] / bd["p_true"] - 1) < 1e-4, f"Lomb-Scargle, {max(bench_budgets):,} trials": np.abs(bd[bls] / bd["p_true"] - 1) < 1e-4}
    ok = np.isfinite(bd["points_per_window"]) & np.isfinite(bd["n_windows"])
    heat(FIGS / "heat_points_windows.png", {k: v[ok] for k, v in hit.items()}, bd["points_per_window"][ok], bd["n_windows"][ok],
         np.array([0, 30, 60, 120, 1000]), np.array([0, 10, 20, 40, 200]), "points per window", "valid windows", "hit rate within 0.01 %")
    heat(FIGS / "heat_cadence_span.png", hit, bd["cadence"], bd["span"], np.array([0, 1, 2, 4, 10, 1000]), np.array([0, 1000, 2000, 2500, 3000, 5000]),
         "median gap between observations (days)", "baseline (days)", "hit rate within 0.01 %")

    # ---- tables
    def hits_row(name, m):
        v = M[m]
        cells = "".join(f"<td class=n>{v[f'within_{t:g}']:.3f}</td>" for t in TOLS)
        return f"<tr><td>{name}</td>{cells}<td class=n>{v['alias_0.0001']:.3f}</td><td class=n>{v['trials_median']:,.0f}</td></tr>"
    table1 = ["<tr><th>method</th><th>within 20 %</th><th>10 %</th><th>1 %</th><th>0.1 %</th><th>0.01 %</th><th>0.01 %, double and half accepted</th><th>trials per star</th></tr>",
              hits_row("model alone (the read-out)", "model"), hits_row("model + fine search within 10 %", "model_refined"),
              hits_row("model candidates + fine search", "model_cands"), hits_row(f"Lomb-Scargle, multiband, {ls_budget:,} trials", ls)]
    if "relu" in M:
        table1.append(hits_row("the read-out with ReLU instead of SiLU (alone)", "relu"))
    table1b = ["<tr><th>within 1 % by class</th>" + "".join(f"<th>{CLS[g]} ({counts[g]})</th>" for g in groups) + "</tr>"]
    for name, m in (("model alone", "model"), ("model + fine search", "model_refined"), ("Lomb-Scargle", ls)):
        table1b.append(f"<tr><td>{name}</td>" + "".join(f"<td class=n>{M[m]['by_superclass'][g]['within_0.01']:.2f}</td>" for g in groups) + "</tr>")
    table1b.append("<tr><th>within 0.01 % by class</th>" + "".join(f"<th>{CLS[g]}</th>" for g in groups) + "</tr>")
    for name, m in (("model + fine search", "model_refined"), ("Lomb-Scargle", ls), ("Lomb-Scargle, double and half accepted", ls)):
        key = "alias_0.0001" if "accepted" in name else "within_0.0001"
        table1b.append(f"<tr><td>{name}</td>" + "".join(f"<td class=n>{M[m]['by_superclass'][g][key]:.2f}</td>" for g in groups) + "</tr>")
    relu_note = ""
    if "relu" in M:
        relu_note = (f"<p class=\"note\">Your collaborator's suggestion, ReLU instead of SiLU in the read-out MLP, was tested on the same latents (middle column of the scatter figure). "
                     f"It changes nothing: {pct(M['relu']['within_0.1'], 1)} within 10 % against {pct(M['model']['within_0.1'], 1)}, {pct(M['relu']['within_0.01'], 1)} within 1 % against {pct(M['model']['within_0.01'], 1)}. "
                     f"The read-out is not a regression that extrapolates: it is a classifier over 240 log-period bins plus an offset inside the bin, so its output cannot bend with the activation. "
                     f"The scatter at long periods comes from the data: {counts.get('LPV', 0)} long-period and {counts.get('CEP', 0)} Cepheid validation stars, a few hundred in training, and a 250 day window that holds about one cycle of a 200 day period.</p>")

    short, long_, mid = p_true < 0.2, p_true > 10, (p_true >= 0.2) & (p_true <= 1.0)
    bul2 = [f"<li>Between 0.2 and 1 day, where {pct(mid.mean())} of the stars sit, the model with the fine search is within 0.01 % for {pct(rate(pr[mid], p_true[mid], 1e-4))} of the stars; Lomb-Scargle for {pct(rate(pl[mid], p_true[mid], 1e-4))}, or {pct(rate(pl[mid], p_true[mid], 1e-4, True))} if its half and double count.</li>",
            f"<li>Below 0.2 days ({int(short.sum())} stars, mostly delta Scuti) Lomb-Scargle wins: {pct(rate(pl[short], p_true[short], 1e-4))} against {pct(rate(pr[short], p_true[short], 1e-4))} for the model. Few training stars have such short periods.</li>",
            f"<li>Above 10 days ({int(long_.sum())} stars) both fall off. The model alone is within 20 % for {pct(rate(pm[long_], p_true[long_], 0.2))} of them and within 1 % for {pct(rate(pm[long_], p_true[long_], 0.01))}; a 250 day window holds few cycles of such a period.</li>"]

    bt, F = R["better"], R["fold"]
    diff = [r for r in rows if r["verdict"] == "different period"]
    kinds = ", ".join(f"{v} {k}" for k, v in bt["by_kind_different"].items())
    table3 = ["<tr><th>ZTF id</th><th>class</th><th>kind</th><th>catalogue P (d)</th><th>model P (d)</th><th>ratio</th><th>fold R2 catalogue</th><th>sharpened</th><th>model</th><th>points</th></tr>"]
    for r in diff[:30]:
        table3.append(f"<tr><td><code>{r['ztf_id']}</code></td><td>{r['class']}</td><td>{r['kind']}</td><td class=n>{float(r['p_catalogue_d']):.6f}</td><td class=n>{float(r['p_best_d']):.6f}</td>"
                      f"<td class=n>{float(r['best_over_catalogue']):.4f}</td><td class=n>{float(r['r2_catalogue']):.2f}</td><td class=n>{float(r['r2_catalogue_sharp']):.2f}</td><td class=n>{float(r['r2_best']):.2f}</td><td class=n>{r['n_points']}</td></tr>")

    def brow(name, m):
        v = BM[m]
        return f"<tr><td>{name}</td><td class=n>{v['trials_median']:,.0f}</td><td class=n>{v['within_0.1']:.2f}</td><td class=n>{v['within_0.01']:.2f}</td><td class=n>{v['within_0.0001']:.2f}</td><td class=n>{v['alias_tolerant_0.0001']:.2f}</td></tr>"
    table4 = ["<tr><th>method</th><th>trials per star</th><th>within 10 %</th><th>within 1 %</th><th>within 0.01 %</th><th>0.01 %, double and half accepted</th></tr>", brow("model's rough period, then the fine search", "model_refined")]
    table4 += [brow(f"Lomb-Scargle, multiband, {b:,}", f"astropy_mb@{b}") for b in bench_budgets if b >= 20000]
    table4 += [brow(f"Lomb-Scargle, best band, {b:,}", f"astropy_1b@{b}") for b in bench_budgets if b >= 200000]
    cross = B["crossing"]["[alias-tolerant 0.01%] astropy, multiband to match model_refined"]
    sweep_table = ""
    if SW:
        rows_ = ["<tr><th>model's search</th><th>trials per star</th><th>within 10 %</th><th>within 1 %</th><th>within 0.01 %</th><th>0.01 %, double and half accepted</th></tr>"]
        for name, v in sorted(SW["settings"].items(), key=lambda kv: (kv[1]["top"], kv[1]["rel"])):
            rows_.append(f"<tr><td>top {v['top']} seed{'s' if v['top'] > 1 else ''}, width {v['rel']:.0%}</td><td class=n>{v['trials_median']:,.0f}</td><td class=n>{v['within_0.1']:.2f}</td><td class=n>{v['within_0.01']:.2f}</td><td class=n>{v['within_0.0001']:.2f}</td><td class=n>{v['alias_tolerant_0.0001']:.2f}</td></tr>")
        sweep_table = '<div class="tablewrap"><table>' + "".join(rows_) + "</table></div>"

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
    bmax = int(cross) if int(cross) in rt_budgets else max(rt_budgets)

    enc_tab = ENC_TAB
    variants = []
    for item in a.variants:
        name, path = item.split("=", 1)
        if (Path(path) / "results.json").is_file():
            variants.append((name, json.load(open(Path(path) / "results.json"))))
    batched_html = ""
    if variants:
        rows_ = ["<tr><th>encoder variant</th><th>best batch (windows)</th><th>GPU ms per star</th><th>end to end ms per star</th><th>stars per second</th><th>peak GPU memory GB</th><th>latent change vs bf16</th><th>own period peak within 1 %</th></tr>"]
        for name, V in variants:
            bb = V["best_batch"]
            b = V["batch_sweep"].get(str(bb), V["batch_sweep"].get(bb))
            acc = V.get("accuracy", {})
            chg = acc.get("latent_rel_change_median")
            rows_.append(f"<tr><td>{name}</td><td class=n>{bb:,}</td><td class=n>{1e3 * b['gpu_s_per_star']:.1f}</td><td class=n>{1e3 * b['e2e_s_per_star']:.1f}</td><td class=n>{b['stars_per_s']:,.0f}</td><td class=n>{b['peak_mem_gb']:.0f}</td>"
                         f"<td class=n>{'' if chg is None else f'{100 * chg:.1f} %'}</td><td class=n>{acc.get('spectral_peak_within_1pct', float('nan')):.3f}</td></tr>")
        fig, ax = plt.subplots(figsize=(8.4, 4.0), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        style(ax)
        cols = ["#2a78d6", "#1baf7a", "#0b0b0b", "#eb6834", "#b07cd6", "#52514e"]
        for (name, V), col in zip(variants, cols):
            pts = sorted((int(k), v) for k, v in V["batch_sweep"].items() if "gpu_s_per_star" in v)
            ax.plot([k for k, _ in pts], [1e3 * v["gpu_s_per_star"] for _, v in pts], color=col, marker="o", ms=4, lw=1.5, label=name)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("windows per batch", fontsize=8, color=INK2)
        ax.set_ylabel("GPU milliseconds per star", fontsize=8, color=INK2)
        ax.set_ylim(0, None)
        ax.set_title("Batched encoder throughput against the batch size", fontsize=9.5, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
        save(fig, "batched_vs_batch.png")
        batched_html = ('<div class="text"><p>Batched over many stars, as a survey run would be: the encoder at every batch size, in bf16 and with the transformer\'s linear layers quantised (torchao). '
                        'The accuracy columns compare each variant with bf16 on the same windows: the median relative change of the window latents, and how often the spectral layer\'s own peak period is within 1 % of the catalogue (bf16 is the reference).</p>'
                        '<div class="tablewrap"><table>' + "".join(rows_) + "</table></div></div>" + img(FIGS / "batched_vs_batch.png", "Line plot of GPU milliseconds per star against the batch size for the bf16 encoder and its quantised variants"))

    page = f"""{head}<main>
  <div class="text">
    <h1>Period from the model, star by star</h1>
    <p class="lede">All {n:,} validation stars from ZTF, whole light curves. The encoder (wide, with the spectral layer, 100,000 steps) gives a rough period. A fine search near that guess makes it sharp. This page shows how often that is right, where it fails, how it compares with Lomb-Scargle in hits and in time, and where the model's period folds the data better than the catalogue's. Lomb-Scargle throughout is your collaborator's astropy search. Updated 6 October.</p>
  </div>

  <section>
    <div class="text">
      <h2>Predicted period against the catalogue period</h2>
      <p>Each dot is one star. A dot on the solid line means the model agrees with the catalogue; the dashed lines are twice and half the period. The bottom row shows the ratio of predicted to catalogue period with the median per period bin, which is where a bend would show.</p>
    </div>
    {img(rep / "scatter_bend.png", "Scatter plots of predicted against catalogue period for the SiLU read-out, the ReLU read-out and the candidate search, with the ratio against the period below each")}
    <div class="text">
      <div class="tablewrap"><table>{''.join(table1)}</table></div>
      <p class="note">A hit is <code>|P predicted / P catalogue − 1| &lt; tolerance</code>, on the period in days. The candidate search tries the read-out's top five bins and the double and the half of the refined period, with the wave's harmonics scaled so every candidate reaches the same highest frequency; ties within 0.02 in fold R2 go to the longer period.</p>
    </div>
    {img(f_tol, "Line plot of the hit rate against the tolerance for the model alone, the model with the fine search, the candidate search and Lomb-Scargle")}
    <div class="text">
      <div class="tablewrap"><table>{''.join(table1b)}</table></div>
    </div>
    {img(f_cls, "Line plots of the hit rate per class within 10, 1 and 0.01 percent for the model alone, the model with the fine search and Lomb-Scargle")}
    <div class="text">
      <ul>
        <li>The fine search costs nothing in reach and buys all the precision: within 10 % the hit rate hardly moves, within 0.01 % it goes from {pct(M['model']['within_0.0001'])} to {pct(M['model_refined']['within_0.0001'])}.</li>
        <li>Lomb-Scargle finds the half period of nearly every eclipsing binary. Strictly it is within 1 % for {pct(M[ls]['by_superclass']['ECL']['within_0.01'])} of them; with the half accepted, {pct(M[ls]['by_superclass']['ECL']['alias_0.01'])}. The model finds the true period of {pct(M['model_refined']['by_superclass']['ECL']['within_0.01'])}.</li>
        <li>The candidate search (the read-out's best bins and the double and half of the refined period, each sharpened, the fold deciding with a margin) lifts the strict hit rate within 0.01 % from {pct(M['model_refined']['within_0.0001'])} to {pct(M['model_cands']['within_0.0001'])} at {M['model_cands']['trials_median']:,.0f} trials per star.</li>
        <li>On RR Lyrae the model leads at every tolerance ({M['model_refined']['by_superclass']['RR']['within_0.0001']:.2f} against {M[ls]['by_superclass']['RR']['within_0.0001']:.2f} within 0.01 %). Lomb-Scargle is better on delta Scuti and long-period stars outright: the model has seen few of those in training.</li>
      </ul>
      {relu_note}
    </div>
  </section>

  <section>
    <div class="text">
      <h2>Hit rate against the period</h2>
      <p>The same hits in log-spaced bins of the catalogue period. The grey bars count the stars per bin; bins with fewer than 15 stars are left out.</p>
    </div>
    {img(FIGS / "hit_vs_period.png", "Hit rate against the catalogue period for the model alone, the model with the fine search, the candidate search and Lomb-Scargle")}
    <div class="text"><ul>{''.join(bul2)}</ul></div>
  </section>

  <section>
    <div class="text">
      <h2>Phase-folded light curves: a typical hit of every class</h2>
      <p>One row per class. The light curve is folded on the catalogue period, on the model's period alone, and on the model's period after the search, with the fitted wave drawn through the points. Two cycles are shown. These are typical stars, the one closest to the class median of the fold quality, not the best ones.</p>
    </div>
    {img(rep / "fold_demo.png", "Phase-folded light curves of one typical star per class, on the catalogue period, the model period and the searched period, with the fitted wave")}
    <div class="text"><ul>
      <li>The middle column is a blur in every short-period row: a period off by 0.2 to 1 % cannot fold 2,700 days of data.</li>
      <li>The right column is as clean as the left one, or cleaner. After the search the period agrees with the catalogue to a few parts in a million.</li>
      <li>The long-period star folds fine on all three periods: at 178 days a 1 % error is still only a tenth of a cycle over the baseline.</li>
    </ul></div>
  </section>

  <section>
    <div class="text">
      <h2>Where the model's period folds better than the catalogue's</h2>
      <p>For every star the fold quality (adjusted R2 of a 6-harmonic wave) was computed on the catalogue period, on the catalogue period sharpened by the same fine search within 0.2 % of it, and on the model's best period. The sharpened catalogue is the fair reference: a catalogue value that is off in the fifth digit folds badly over ZTF's baseline, and the model should not get credit for fixing that alone.</p>
      <div class="tablewrap"><table>
        <tr><th>fold R2, median over the stars</th><th>catalogue</th><th>catalogue sharpened</th><th>model + fine search</th><th>model candidates + search</th></tr>
        <tr><td>whole light curve</td><td class=n>{F['catalogue']['median']:.3f}</td><td class=n>{F['catalogue_sharp']['median']:.3f}</td><td class=n>{F['model_refined']['median']:.3f}</td><td class=n>{F['model_cands']['median']:.3f}</td></tr>
      </table></div>
      <ul>
        <li><strong>{bt['n']:,} stars</strong> fold better on the model's period than on the catalogue period as given, by more than 0.1 in R2, reaching R2 above 0.5.</li>
        <li><strong>{bt['n_catalogue_imprecise']:,} of them are catalogue periods made sharp.</strong> The model's period is the catalogue's to within 1 %, and sharpening the catalogue value itself gives the same fold. The catalogue is right but not precise enough for this baseline.</li>
        <li><strong>{bt['n_different']:,} are a different period</strong> that beats even the sharpened catalogue by more than 0.1 in R2 ({kinds}). These are the stars to report. {bt['worse_than_sharp']:,} stars fold worse than the sharpened catalogue by the same margin.</li>
      </ul>
      <p class="note">A better fold is evidence, not proof. On ZTF's night-only sampling a period that differs by one cycle per day folds almost as well as the true one, and for an eclipsing binary the half period folds a symmetric curve as well as the full one. The list keeps the alias kind next to each star so the reader can judge.</p>
    </div>
    {img(rep / "fold_better.png", "Phase-folded light curves of stars where the model period folds better than the catalogue period: different periods first, then catalogue periods made sharp")}
    <div class="text">
      <p>The {min(30, len(diff))} largest gains among the different-period stars. The full list of {bt['n']:,} stars, with both verdicts, is in <a href="better_than_catalogue.csv">better_than_catalogue.csv</a> published with this page.</p>
      <div class="tablewrap"><table>{''.join(table3)}</table></div>
    </div>
  </section>

  <section>
    <div class="text">
      <h2>The model's search against Lomb-Scargle, at equal budgets</h2>
      <p>On 1,000 validation stars, whole light curves, Lomb-Scargle gets a growing budget of trial frequencies: a single sinusoid on a uniform frequency grid, on the band with the most points or with the multiband model. The model-seeded search takes the encoder's rough period and runs the fine search near it; its budget grows with the width of that search (1, 3, 10 and 30 % of the period) and with the number of seed peaks it sharpens (the read-out's best 1, 3 or 5 bins).</p>
    </div>
    {img(f_bench, "Three line plots of hit rate within 10, 1 and 0.01 percent against trial frequencies per star: Lomb-Scargle in a muted colour, the model-seeded search in blue with one line per number of seeds")}
    <div class="text">{sweep_table}</div>
    <div class="tablewrap"><table>{''.join(table4)}</table></div>
    <div class="text">
      <p class="note">Within 0.01 % of the catalogue period, Lomb-Scargle never matches the model-seeded search at any budget up to 500,000 trials. Accepting the double and the half, it needs {cross:,.0f} trials to match what the model-seeded search does with {BM['model_refined']['trials_median']:,.0f}.</p>
    </div>
  </section>

  <section>
    <div class="text">
      <h2>Where the hits are</h2>
      <p>Hit rate within 0.01 % in cells of points per window against number of valid windows, for the model-seeded search and for Lomb-Scargle at {max(bench_budgets):,} trials, with the count of stars in each cell. Below, the same over the median gap between observations against the baseline.</p>
    </div>
    {img(FIGS / "heat_points_windows.png", "Two heat maps of the hit rate over points per window and number of windows, model-seeded search against Lomb-Scargle")}
    {img(FIGS / "heat_cadence_span.png", "Two heat maps of the hit rate over observation cadence and baseline, model-seeded search against Lomb-Scargle")}
  </section>

  <section>
    <div class="text">
      <h2>Runtime per star</h2>
      <p>Wall-clock of every step on {T['n_stars']} validation stars (median {T['points']['median']:.0f} points and {T['windows']['median']:.0f} windows per star), after warm-up, on one {html.escape(T['gpu'] or 'GPU')} and {T['cpu_threads']} CPU threads. Lomb-Scargle (astropy) runs on the CPU only, so its times are on a different device.</p>
    </div>
    <div class="tablewrap"><table>{''.join(table5)}</table></div>
    {img(f_rt, "Line plot of milliseconds per star against the number of trial frequencies for Lomb-Scargle, with the model path as horizontal lines")}
    {batched_html}
    <div class="text">
      <ul>
        <li>One star at a time, the model path takes {1e3 * T['model_total']['median']:.0f} ms, almost all of it the encoder forward over the star's windows: launch overhead on a batch of 40 windows, not arithmetic.</li>
        <li>Batched over many stars, as a survey run would be, the encoder costs {1e3 * T['encode_batched_per_star']:.1f} ms per star and the whole path about {1e3 * T['model_total_batched_per_star']:.0f} ms.</li>
        <li>Lomb-Scargle at {bmax:,} trials, the budget it needs to match the model with the double and half accepted, takes {1e3 * S[f'astropy_mb@{bmax}']['median']:.0f} ms per star with the multiband model and {1e3 * S[f'astropy_1b@{bmax}']['median']:.0f} ms on the best band, on the CPU.</li>
      </ul>
    </div>
  </section>

  <section>
    <div class="text">
      {enc_tab}
  </section>
</main>
"""
    Path(a.out).write_text(page)
    print(f"wrote {a.out}: {len(page) / 1e6:.1f} MB, {len(diff)} different-period rows")


if __name__ == "__main__":
    main()
