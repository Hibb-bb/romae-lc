"""Build the period artifact page (https://claude.ai/artifact/JoubE7CVCj2KSB1QtGzwub)
from the result folders. Sections: hit rates (table, two scatter panels); clean
folds (as good as the catalogue, and model + fine search cleaner than it; no fitted wave); the
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
ENC_TAB = '<h2>The encoders behind these numbers</h2>\n      <p>Period hit rate of the bin read-out on the pooled latent, 5,346 validation stars, 240 bins.</p>\n    </div>\n    <div class="tablewrap"><table>\n      <tr><th>encoder</th><th>steps</th><th>within 1%</th><th>within 10%</th><th>within 20%</th></tr>\n      <tr><td>light, plain</td><td class=n>50,000</td><td class=n>0.36</td><td class=n>0.66</td><td class=n></td></tr>\n      <tr><td>wide, plain</td><td class=n>110,000</td><td class=n>0.58</td><td class=n>0.71</td><td class=n>0.80</td></tr>\n      <tr><td>light, spectral layer</td><td class=n>50,000</td><td class=n>0.35</td><td class=n>0.76</td><td class=n>0.85</td></tr>\n      <tr><td>wide, spectral layer</td><td class=n>54,000</td><td class=n>0.69</td><td class=n>0.77</td><td class=n>0.85</td></tr>\n      <tr><td>wide, spectral layer</td><td class=n>100,000</td><td class=n>0.70</td><td class=n>0.77</td><td class=n>0.84</td></tr>\n    </table></div>\n    <div class="text">\n    </div>\n  </section>\n'
FIGS = Path("project/results/period_page/figs")
C = dict(model="#6baed6", refined="#2171b5", cands="#08306b", ls="#e6550d", ls_alias="#fd8d3c")
LS_DASH, LS_DOT = (0, (5, 2.5)), (0, (1.2, 1.6))
STY = {"model alone": (C["model"], "-"), "model + fine search": (C["refined"], "-"), "model candidates + search": (C["cands"], "-"),
       "Lomb-Scargle": (C["ls"], LS_DASH), "Lomb-Scargle, double and half accepted": (C["ls_alias"], LS_DOT)}
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
    methods = [("model alone", "model", C["model"], None), ("model + fine search", "model_refined", C["refined"], None), ("model candidates + search", "model_cands", C["cands"], None),
               ("Lomb-Scargle", ls, C["ls"], "//"), ("Lomb-Scargle, double and half accepted", ls, C["ls_alias"], "..")]
    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    x = np.arange(len(groups))
    w = 0.8 / len(methods)
    for i, (name, m, col, hatch) in enumerate(methods):
        k = key if "accepted" not in name else key.replace("within_", "alias_")
        ax.bar(x + (i - len(methods) / 2) * w + w / 2, [M[m]["by_superclass"][g][k] for g in groups], w, color=col, hatch=hatch, edgecolor=SURFACE if hatch else col, lw=0.5, label=name)
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
    ms = sorted((m for m in B["methods"] if m.startswith("astropy_mb@")), key=lambda m: int(m.split("@")[1]))
    x = [B["methods"][m]["trials_median"] for m in ms]
    y = [B["methods"][m][key] for m in ms]
    ax.plot(x, y, color=C["ls"], lw=1.4, alpha=0.9, marker="o", ms=3, ls=LS_DASH)
    labels.append((x[-1], y[-1], "Lomb-Scargle", C["ls"]))
    tops = sorted({v["top"] for v in SW["settings"].values()})
    widths = sorted({v["rel"] for v in SW["settings"].values()})
    markers = dict(zip(widths, ["o", "s", "^", "D", "v", "P"]))
    for name, k in (("model + fine search", tops[0]), ("model candidates + search", tops[-1])):
        c = STY[name][0]
        items = sorted((v for v in SW["settings"].values() if v["top"] == k), key=lambda v: v["trials_median"])
        x = [v["trials_median"] for v in items]
        y = [v[key] for v in items]
        ax.plot(x, y, color=c, lw=1.5, alpha=0.9)
        for v, xi, yi in zip(items, x, y):
            ax.plot([xi], [yi], marker=markers[v["rel"]], ms=6, color=c, ls="none", markeredgecolor=SURF, markeredgewidth=0.6)
        labels.append((x[-1], y[-1], name, c))
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=markers[w], ms=6, color=INK2, ls="none", label=f"search width {w:.0%} of the period") for w in widths]
    handles.append(Line2D([], [], marker="o", ms=3, color=C["ls"], ls=LS_DASH, label="Lomb-Scargle grid size"))
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="center right", labelcolor=INK2, title="what each dot is", title_fontsize=7)
    for group in ([l for l in labels if l[3] == C["ls"]], [l for l in labels if l[3] != C["ls"]]):
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
    ax.set_title(f"Hit rate {label} against the search budget", fontsize=9.5, color=INK, loc="left")
    return save(fig, f"cost_{key}.png")


def fig_runtime(T, budgets):
    S = T["steps"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.plot(budgets, [1e3 * S[f"astropy_mb@{b}"]["median"] for b in budgets], color=C["ls"], ls=LS_DASH, marker="o", ms=4, lw=1.5, label="Lomb-Scargle (CPU)")
    one = 1e3 * (S["encode"]["median"] + S["read"]["median"] + S["search"]["median"])
    batched = 1e3 * T["model_total_batched_per_star"]
    ax.axhline(one, color=C["model"], lw=1.6, label=f"model, one star at a time ({one:.0f} ms)")
    ax.axhline(batched, color=C["cands"], lw=1.6, label=f"model, batched over stars ({batched:.0f} ms)")
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

def fig_spectral_diagram():
    """Two lanes of boxes: Lomb-Scargle (fixed) above, the spectral layer
    (learned) below; the shared step, the phasor sum over the actual
    times, is drawn the same way in both."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    blue, orange, shared = "#dbe9f7", "#fde3cf", "#e8e3f3"
    edge = {"blue": C["refined"], "orange": C["ls"], "shared": "#6a51a3"}

    def box(x, y, w, h, text, kind, bold=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc={"blue": blue, "orange": orange, "shared": shared}[kind],
                                    ec=edge[kind], lw=1.4 if kind != "shared" else 2.0))
        if bold:
            ax.text(x + w / 2, y + h - 3.2, bold, ha="center", va="top", fontsize=9.4, fontweight="bold", color=INK)
            ax.text(x + w / 2, y + h - 10.5, text, ha="center", va="top", fontsize=8.3, color=INK2, linespacing=1.25)
        else:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, color=INK, linespacing=1.25)

    def arrow(x0, x1, y, text=None):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.1, shrinkA=0, shrinkB=0))
        if text:
            ax.text((x0 + x1) / 2, y + 1.8, text, ha="center", va="bottom", fontsize=7.2, color=INK2, linespacing=1.15)

    # lane labels
    ax.text(0.5, 88, "Lomb-Scargle\n(fixed)", ha="left", va="center", fontsize=10, fontweight="bold", color=edge["orange"])
    ax.text(0.5, 44, "spectral layer\n(learned, inside\nthe encoder)", ha="left", va="center", fontsize=10, fontweight="bold", color=edge["blue"])
    ax.plot([0.5, 99.5], [66, 66], color="#c9cfd8", lw=0.8, ls=(0, (3, 3)))
    x1, w1, x2, w2, x3, w3, x4, w4 = 13, 16, 35, 23, 64, 16, 85, 14

    # Lomb-Scargle lane: 4 boxes
    yl, hl = 78, 20
    box(x1, yl, w1, hl, "one number per observation:\nbrightness, centred,\nweighted by 1/σ²", "orange", bold="input")
    arrow(x1 + w1 + 0.4, x2 - 0.4, yl + hl / 2)
    box(x2, yl, w2, hl, "$Z(f)=\\sum_j y_j\\,e^{2\\pi i f t_j}$\nuniform grid of trial\nfrequencies, set per run", "shared", bold="phasor sum over the actual times")
    arrow(x2 + w2 + 0.4, x3 - 0.4, yl + hl / 2)
    box(x3, yl, w3, hl, "one power per frequency,\nnormalised to [0, 1]\nfor one sinusoid", "orange", bold="periodogram")
    arrow(x3 + w3 + 0.4, x4 - 0.4, yl + hl / 2)
    box(x4, yl, w4, hl, "the highest peak;\naliases and harmonics\nhandled by rules", "orange", bold="read-out")

    # spectral-layer lane: 7 boxes in two rows
    yt, ht = 42, 20
    box(x1, yt, w1, ht, "brightness, error, band,\ntime → encoder blocks 1–2\n→ 384-wide feature $h_j$", "blue", bold="tokens of one window")
    arrow(x1 + w1 + 0.4, x2 - 0.4, yt + ht / 2, "linear map\n8 channels")
    box(x2, yt, w2, ht, "$Z_m(f)=\\sum_j a_{j,m}\\,e^{2\\pi i f t_j}$\n20,260 log-spaced frequencies,\n0.02 to 500 d, step 0.05 %", "shared", bold="the same sum, 8 learned channels")
    arrow(x2 + w2 + 0.4, x3 - 0.4, yt + ht / 2)
    box(x3, yt, w3, ht, "$|Z|^2$ per channel at f, 2f\nand f/2, plus their logs:\n48 × 20,260, standardised", "blue", bold="power map")
    arrow(x3 + w3 + 0.4, x4 - 0.4, yt + ht / 2)
    box(x4, yt, w4, ht, "dilated 1-D convolution\nalong frequency,\n32 wide, 3 blocks", "blue", bold="reader")
    # bend down to the second row (right to left)
    xm = x4 + w4 / 2
    ax.add_patch(FancyArrowPatch((xm, yt), (xm, 30), arrowstyle="-", color=INK2, lw=1.1, shrinkA=0, shrinkB=0))
    ax.add_patch(FancyArrowPatch((xm, 30), (x3 + w3 + 0.4, 30), arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.1, shrinkA=0, shrinkB=0))
    yb, hb = 6, 24
    box(x3, yb, w3, hb, "softmax over the bins:\npeak position, spread,\npooled summary", "blue", bold="a score per frequency bin")
    ax.add_patch(FancyArrowPatch((x3 - 0.4, yb + hb / 2), (x2 + w2 + 0.4, yb + hb / 2), arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.1, shrinkA=0, shrinkB=0))
    box(x2, yb, w2, hb, "the summary is added to the\nCLS token; the remaining encoder\nblocks run; the pooled latent\nfeeds the period read-outs", "blue", bold="back into the encoder")
    ax.text(x1, yb + hb / 2, "pre-training only: a small loss\n(weight 0.5) asks the scores to\npoint at the catalogue bin.\nAt inference no catalogue\nvalue enters.", ha="left", va="center", fontsize=7.6, color=INK2, linespacing=1.3)
    return save(fig, "spectral_layer.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--ckpt", default="project/runs/maew_spec/mae.pt", help="for the data arguments (the fold demos read the light curves)")
    ap.add_argument("--ls-extra", default=None, help="ls_allstars folder: Lomb-Scargle at more budgets on every star, for the first table")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rep, ben, run = Path(a.report), Path(a.bench), Path(a.runtime)
    R, B, T, SW = (json.load(open(p / "results.json")) for p in (rep, ben, run, Path(a.sweep)))
    d = np.load(rep / "per_star.npz", allow_pickle=True)
    bd = np.load(ben / "per_star.npz", allow_pickle=True)
    csv_name = "model_fine_search.csv" if (rep / "model_fine_search.csv").is_file() else "better_than_catalogue.csv"
    rows = list(csv.DictReader(open(rep / csv_name)))
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
    if a.ls_extra and (Path(a.ls_extra) / "results.json").is_file():
        X = json.load(open(Path(a.ls_extra) / "results.json"))
        if X["n_stars"] == n:
            M.update(X["methods"])
    ls_all = sorted((m for m in M if m.startswith("astropy_mb@")), key=lambda m: int(m.split("@")[1]))
    table1 = ["<tr><th>method</th><th>within 20 %</th><th>10 %</th><th>1 %</th><th>0.1 %</th><th>0.01 %</th><th>0.01 %, double and half accepted</th><th>trials per star</th></tr>",
              hits_row("model alone", "model"), hits_row("model + fine search", "model_refined"), hits_row("model candidates + search", "model_cands")]
    table1 += [hits_row(f"Lomb-Scargle, {int(m.split('@')[1]):,} trials", m) for m in ls_all]
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
    f_better = fold_rows_figure(FIGS / "folds_model_fine_search.png",
                                [(val[i], f"{fine[pos[i]]}\n{d['ztf_id'][pos[i]]}\n{'other period' if pos[i] in [pos[int(r['val_index'])] for r in diff[:6]] else 'same period, sharper'}", two(pos[i])) for i in pick],
                                "Model + fine search: stars where its period folds the light curve more cleanly than the catalogue value")
    F = R["fold"]
    bt = R["better"]

    # ---- 3. cost: hit rate against trials, one figure per threshold; the trials table; runtime
    f_cost = [fig_cost_one(SW, B, key, label) for key, _, label in thresholds]
    bench_budgets = sorted({int(m.split("@")[1]) for m in BM if m.startswith("astropy_mb@")})
    rt_budgets = sorted({int(k.split("@")[1]) for k in S if k.startswith("astropy_mb@") and S[k]["n"]})
    f_rt = fig_runtime(T, rt_budgets)
    f_spec = fig_spectral_diagram()
    one_ms = 1e3 * (S["encode"]["median"] + S["read"]["median"] + S["search"]["median"])
    table5 = ["<tr><th>method</th><th>milliseconds per star</th></tr>",
              f"<tr><td>model, encoder + read-out + fine search, batched over stars (GPU)</td><td class=n>{1e3 * T['model_total_batched_per_star']:.0f}</td></tr>",
              f"<tr><td>model, the same one star at a time (GPU)</td><td class=n>{one_ms:.0f}</td></tr>"]
    table5 += [f"<tr><td>Lomb-Scargle, {b:,} trials (CPU)</td><td class=n>{1e3 * S[f'astropy_mb@{b}']['median']:.0f}</td></tr>" for b in rt_budgets]

    # ---- 4. analysis: bars by class, hit rate against the period per threshold, heat maps
    f_bars = [fig_bars_by_class(M, ls, groups, counts, key, label) for key, _, label in thresholds]
    series = {"model alone": (pm, *STY["model alone"], False), "model + fine search": (pr, *STY["model + fine search"], False), "model candidates + search": (pc, *STY["model candidates + search"], False),
              f"Lomb-Scargle, {ls_budget:,} trials": (pl, *STY["Lomb-Scargle"], False), "Lomb-Scargle, double and half accepted": (pl, *STY["Lomb-Scargle, double and half accepted"], True)}
    f_hvp = [fig_hit_vs_period_one(p_true, sup, series, tol, label) for _, tol, label in thresholds]
    bls = f"p_astropy_mb@{max(bench_budgets)}"
    hit = {"model + fine search": np.abs(bd["p_model_refined"] / bd["p_true"] - 1) < 1e-4, f"Lomb-Scargle, {max(bench_budgets):,} trials": np.abs(bd[bls] / bd["p_true"] - 1) < 1e-4}
    ok = np.isfinite(bd["points_per_window"]) & np.isfinite(bd["n_windows"])
    heat(FIGS / "heat_points_windows.png", {k: v[ok] for k, v in hit.items()}, bd["points_per_window"][ok], bd["n_windows"][ok],
         np.array([0, 30, 60, 120, 1000]), np.array([0, 10, 20, 40, 200]), "points per window", "valid windows", "hit rate within 0.01 %")
    heat(FIGS / "heat_cadence_span.png", hit, bd["cadence"], bd["span"], np.array([0, 1, 2, 4, 10, 1000]), np.array([0, 1000, 2000, 2500, 3000, 5000]),
         "median gap between observations (days)", "baseline (days)", "hit rate within 0.01 %")

    cross = B["crossing"]["[alias-tolerant 0.01%] astropy, multiband to match model_refined"]
    top1 = sorted((v for v in SW["settings"].values() if v["top"] == min(x["top"] for x in SW["settings"].values())), key=lambda v: v["rel"])
    top5 = sorted((v for v in SW["settings"].values() if v["top"] == max(x["top"] for x in SW["settings"].values())), key=lambda v: v["rel"])
    min_trials = min(v["trials_median"] for v in SW["settings"].values())
    width_items = "".join(f"<li>width {v1['rel']:.0%} of the period: about {v1['trials_median']:,.0f} trials per star for the fine search, {v5['trials_median']:,.0f} with the candidates</li>" for v1, v5 in zip(top1, top5))
    page = f"""{HEAD}<main>
  <div class="text">
    <h1>Period from the model, star by star</h1>
    <p class="lede">All {n:,} validation stars from ZTF, whole light curves. The encoder (wide, with the spectral layer) gives a rough period; a fine search near it makes it sharp. Lomb-Scargle throughout is the astropy search on a uniform frequency grid. Updated 9 October.</p>
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
      <p>For {bt['n']:,} stars the period from model + fine search folds more cleanly than the catalogue value as listed (R2 higher by more than 0.1, reaching 0.5). For {bt['n_catalogue_imprecise']:,} of them the catalogue period is the same one made sharper; for {bt['n_different']:,} it is another period that also folds better than the sharpened catalogue value. The full list is in <a href="model_fine_search.csv">model_fine_search.csv</a>.</p>
    </div>
    {img(f_better, "Phase-folded light curves of stars where the model + fine search period folds more cleanly than the catalogue value: other periods first, then the same period made sharper")}
    <div class="text">
      <p class="note">A cleaner fold is evidence, not proof: on night-only sampling a period one cycle per day away folds almost as well, and the half period of a symmetric eclipsing binary folds like the full one.</p>
    </div>
  </section>

  <section>
    <div class="text">
      <h2>Cost: hit rate against the search budget</h2>
      <p>1,000 validation stars. The x axis is the number of trial periods evaluated per star, the common currency of both methods. For Lomb-Scargle it is the size of its frequency grid. For the model it is set by the width of the fine search around its guess: the grid is spaced by the star's baseline, so a wider window costs more trials. Each marker shape is one width:</p>
      <ul>{width_items}</ul>
      <p>Within 0.01 % Lomb-Scargle never reaches the model; with the double and the half accepted it needs {cross:,.0f} trials to match the model's {BM['model_refined']['trials_median']:,.0f}.</p>
      <p class="note">To do: push the budget further down. The smallest setting here is still {min_trials:,.0f} trials per star. Next, run narrower windows and a coarser grid (fewer trials per unit of 1/baseline) to find the budget at which model + fine search breaks, and whether it breaks gently (losing the 0.01 % hits first) or all at once.</p>
    </div>
    {img(f_cost[0], "Hit rate within 10 percent against trial frequencies per star for Lomb-Scargle and for the model's search")}
    {img(f_cost[1], "Hit rate within 1 percent against trial frequencies per star for Lomb-Scargle and for the model's search")}
    {img(f_cost[2], "Hit rate within 0.01 percent against trial frequencies per star for Lomb-Scargle and for the model's search")}
    <div class="text">
      <p>Wall-clock per star, median over {T['n_stars']} stars (about {T['points']['median']:.0f} points and {T['windows']['median']:.0f} windows each), on one {html.escape(T['gpu'] or 'GPU')}; Lomb-Scargle runs on the CPU. Data loading and tokenising are not counted.</p>
      <div class="tablewrap"><table>{''.join(table5)}</table></div>
    </div>
    {img(f_rt, "Milliseconds per star against trial frequencies for Lomb-Scargle, with the model as horizontal lines")}
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
      <h2>The spectral layer</h2>
      <p>A period within 0.01 % over a 2,700 day baseline needs frequency resolution of about one over the baseline: tens of thousands of trial frequencies. Attention cannot give that. Rotary attention does multiply by phasors, but it sums its few hundred frequency rungs inside one dot product and then normalises the scores per row, so the power at a single frequency is never formed. Lomb-Scargle gets the resolution by summing phasors over the actual observation times, one sum per trial frequency. The spectral layer keeps that mechanism and makes what is summed, and how the spectrum is read, learned.</p>
    </div>
    {img(f_spec, "Diagram of the spectral layer inside the encoder, with the Lomb-Scargle periodogram as the fixed version of the same sum")}
    <div class="text">
      <p>What happens inside, for one window of up to 256 observations. The first two encoder blocks mix brightness, error, band and time into a 384-wide feature per observation. A linear map turns each feature into 8 numbers, centred over the window. For each of 20,260 trial frequencies, log-spaced from 500 days to half an hour with a step of 0.05 %, the layer sums the 8 numbers times the phasor at the observation's own time, and keeps the squared magnitude. The map of powers (also read at twice and half the frequency, so a harmonic sits next to its fundamental) is standardised and read by a small dilated convolution along the frequency axis. The reader gives a score per frequency bin. The softmax over the bins gives a peak position, a spread, and a pooled summary, which are added to the CLS token before the remaining encoder blocks. The period read-outs and the classifier see only the pooled latent.</p>
      <p>Lomb-Scargle is one point in the layer's parameter space: one channel equal to the weighted centred brightness, and the reader replaced by the argmax. The layer starts near there (channel 0 is initialised to the brightness) and training moves it away. During pre-training a small auxiliary loss (weight 0.5) asks the reader's scores to point at the bin of the catalogue period; at inference no catalogue value enters. The table says where the two differ.</p>
      <div class="tablewrap"><table>
        <tr><th></th><th>Lomb-Scargle</th><th>spectral layer</th></tr>
        <tr><td>what is summed</td><td>the brightness, weighted by 1/σ²</td><td>8 learned numbers per observation, from brightness, error, band and context</td></tr>
        <tr><td>frequency grid</td><td>uniform, set per run</td><td>fixed log grid of 20,260 bins, 0.02 to 500 days</td></tr>
        <tr><td>signal model</td><td>one sinusoid per frequency</td><td>none; the reader learns what a peak and its harmonics mean</td></tr>
        <tr><td>output</td><td>one power per frequency, in [0, 1]</td><td>8 powers per frequency at f, 2f and f/2, then a score per bin and a summary vector</td></tr>
        <tr><td>aliases and harmonics</td><td>handled afterwards by rules</td><td>seen by the reader, learned from data</td></tr>
        <tr><td>what is learned</td><td>nothing</td><td>the 8-channel map, the reader, and the encoder blocks before it (the loss flows through the spectrum)</td></tr>
        <tr><td>cost per window</td><td>points × frequencies</td><td>points × frequencies × 8, a few milliseconds per batch on the GPU</td></tr>
      </table></div>
      <p>So the layer is a learned periodogram, not a new way to find periodicity. Its value is that the rest of the encoder is trained to make features that fold well, and that the pooled latent carries the period at the grid's resolution. The rows below are the encoders tried; the spectral layer is the one encoder-side change that moved the period.</p>
    </div>
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
