"""Two architecture diagrams for the talk, in the style of a paper's
pipeline figure: pastel rounded blocks with the operator's name inside,
token stacks as small rounded squares (blue in, orange out), thin arrows,
and the tensor shape in bold brackets under every stage.

    python -m project.eval.diagrams --out project/results/slides
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse, Rectangle

BLUE, ORANGE, GREEN, PURPLE = "#4f8ef7", "#f5a65b", "#5cc08a", "#a58af2"
BLOCK, BLOCK_EDGE, PINK, PINK_EDGE = "#d8f1f8", "#d8f1f8", "#fde0e4", "#fde0e4"
INK, GREY = "#222222", "#666666"
FS = dict(big=15, name=13, dim=13, note=10.5)


def stack(ax, x, y, n, color, w=0.32, h=0.32, gap=0.1, dots=True):
    """A vertical stack of ``n`` rounded squares centred at (x, y)."""
    total = n * h + (n - 1) * gap + (0.55 if dots else 0)
    y0 = y + total / 2
    ys = []
    for i in range(n):
        yy = y0 - h / 2 - i * (h + gap)
        ax.add_patch(FancyBboxPatch((x - w / 2, yy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.07", fc=color, ec=color))
        ys.append(yy)
    if dots:
        yd = ys[-1] - h / 2 - 0.12
        for k in range(3):
            ax.add_patch(Rectangle((x - 0.04, yd - k * 0.14 - 0.06), 0.08, 0.08, fc=INK, ec=INK))
    return y0 + 0.1, ys[-1] - h / 2 - (0.6 if dots else 0.1)


def block(ax, x, y, w, h, name, sub=None, color=BLOCK, dshape=True, fs=None):
    """A pastel block; ``dshape`` gives the encoder-like rounded right side."""
    fs = fs or FS["name"]
    if dshape:
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w - h / 2, h, fc=color, ec=color))
        ax.add_patch(Ellipse((x + w / 2 - h / 2, y), h, h, fc=color, ec=color))
        cx = x - h / 4
    else:
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc=color, ec=color))
        cx = x
    if sub:
        ax.text(cx, y + 0.22, name, ha="center", va="center", fontsize=fs, color=INK)
        ax.text(cx, y - 0.28, sub, ha="center", va="center", fontsize=FS["note"], color=GREY)
    else:
        ax.text(cx, y, name, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, x0, x1, y, label=None, dy=0.0):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=14, lw=1.3, color=INK, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, y + 0.22 + dy, label, ha="center", va="bottom", fontsize=FS["note"], color=GREY)


def dim(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="top", fontsize=FS["dim"], color=INK, fontweight="bold")


def spectral(out: Path):
    fig, ax = plt.subplots(figsize=(20, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 20); ax.set_ylim(0, 5.8); ax.axis("off")
    y = 3.1
    # 1. token features after block 2, plus the raw value channels
    top, bot = stack(ax, 1.0, y, 5, BLUE)
    ax.text(1.0, top + 0.15, "token features\nafter encoder block 2", ha="center", va="bottom", fontsize=FS["note"], color=GREY)
    dim(ax, 1.0, bot, "[N × d]\nN ≤ 256 observations, d = 384")
    arrow(ax, 1.3, 2.3, y)
    # 2. linear map to 8 channels
    block(ax, 3.1, y, 1.5, 1.0, "linear\nW", dshape=False)
    ax.text(3.1, y - 0.75, "+ raw brightness, error\ncentred per window", ha="center", va="top", fontsize=FS["note"], color=GREY)
    arrow(ax, 3.9, 4.6, y)
    top, bot = stack(ax, 5.0, y, 5, BLUE, w=0.22, h=0.22, gap=0.08)
    dim(ax, 5.0, bot, "[N × 8]")
    arrow(ax, 5.3, 6.1, y)
    # 3. the phasor sum
    block(ax, 7.9, y, 3.5, 1.6, "phasor sum", sub="Z[k, m] = Σ_j a[j, m] · e^(2πi f_k t_j)")
    ax.text(7.5, y - 1.05, "K = 20,260 trial frequencies, log-spaced\n500 d → 0.02 d, step 0.05 %", ha="center", va="top", fontsize=FS["note"], color=GREY)
    arrow(ax, 9.7, 10.4, y)
    # 4. the power map
    for i in range(6):
        ax.add_patch(Rectangle((10.5, y - 0.75 + i * 0.25), 1.6, 0.22, fc=ORANGE, ec="white", lw=0.5, alpha=0.55 + 0.45 * (i % 2)))
    ax.text(11.3, y + 0.95, "|Z|² per channel\nat f, 2f, f/2, + logs", ha="center", va="bottom", fontsize=FS["note"], color=GREY)
    dim(ax, 11.3, y - 0.9, "[48 × K]")
    arrow(ax, 12.2, 12.9, y)
    # 5. the reader
    block(ax, 14.2, y, 2.4, 1.6, "reader", sub="dilated 1-D conv\nalong frequency")
    ax.text(13.9, y - 1.05, "32 channels, 3 blocks\n(dilations 1, 2, 4)", ha="center", va="top", fontsize=FS["note"], color=GREY)
    arrow(ax, 15.5, 16.2, y)
    # 6. scores, softmax, summary
    top, bot = stack(ax, 16.6, y, 5, ORANGE, w=0.22, h=0.22, gap=0.08)
    dim(ax, 16.6, bot, "scores [K]")
    arrow(ax, 16.9, 17.6, y, "softmax")
    block(ax, 18.7, y, 1.9, 1.6, "peak, spread,\npooled summary", color=PINK, dshape=False, fs=FS["note"] + 1)
    dim(ax, 18.7, y - 0.9, "[32 + 2] → [d]")
    ax.text(18.7, y - 1.45, "added to the CLS token,\nthen encoder blocks 3–6", ha="center", va="top", fontsize=FS["note"], color=GREY)
    # training-only note
    ax.add_patch(FancyBboxPatch((15.2, 0.25), 4.5, 0.75, boxstyle="round,pad=0,rounding_size=0.15", fc=PINK, ec=PINK))
    ax.text(17.45, 0.62, "training only: cross-entropy of the scores\nagainst the catalogue period's bin (weight 0.5)", ha="center", va="center", fontsize=FS["note"], color=INK)
    ax.text(0.2, 5.5, "Spectral layer: a learned periodogram inside the encoder", ha="left", va="top", fontsize=FS["big"], color=INK, fontweight="bold")
    ax.text(0.2, 0.62, "channel 0 of W starts as the raw brightness, so the layer begins as a Lomb-Scargle periodogram;\ntraining moves the 8 channels and the reader away from it", ha="left", va="center", fontsize=FS["note"], color=GREY)
    fig.savefig(out / "diagram_spectral.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def rope(out: Path):
    import numpy as np

    fig, ax = plt.subplots(figsize=(20, 6.6))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 20); ax.set_ylim(0, 6.6); ax.axis("off")
    y = 3.9
    # 1. tokens with positions
    top, bot = stack(ax, 1.0, y, 5, BLUE)
    ax.text(1.0, top + 0.15, "tokens: brightness, error\n+ positions (t, λ)", ha="center", va="bottom", fontsize=FS["note"], color=GREY)
    dim(ax, 1.0, bot, "[N × d], positions [N × 2]\nt in days, λ the band's wavelength")
    arrow(ax, 1.35, 2.35, y, "q, k\nper head", dy=0.05)
    # 2. one head's query/key split into blocks
    x0 = 2.5
    ax.add_patch(FancyBboxPatch((x0, y - 0.45), 4.2, 0.9, boxstyle="round,pad=0,rounding_size=0.1", fc=BLUE, ec=BLUE, alpha=0.85))
    ax.add_patch(FancyBboxPatch((x0 + 4.25, y - 0.45), 0.6, 0.9, boxstyle="round,pad=0,rounding_size=0.1", fc=GREEN, ec=GREEN, alpha=0.9))
    ax.text(x0 + 2.1, y, "time block: 56 channels = 28 pairs", ha="center", va="center", fontsize=FS["note"] + 1, color="white")
    ax.text(x0 + 4.55, y, "λ\n8", ha="center", va="center", fontsize=FS["note"], color="white")
    dim(ax, x0 + 2.4, y - 0.6, "one head of q (and of k): [64] = 56 + 8")
    for i in range(28):
        xx = x0 + 0.15 + i * 0.14
        ax.add_patch(Rectangle((xx, y + 0.55), 0.1, 0.35, fc=PURPLE if i < 21 else "#cfcfcf", ec="white", lw=0.4))
    ax.text(x0 + 2.1, y + 1.0, "21 pairs rotate, one rung each; 7 pairs do not (p = 0.75)", ha="center", va="bottom", fontsize=FS["note"], color=GREY)
    arrow(ax, x0 + 5.0, x0 + 5.8, y)
    # 3. rotation
    block(ax, 10.0, y, 3.4, 1.7, "rotate pair r by", sub="angle = t / τ_r  (time pairs)\nangle = λ / τ_r  (band pairs)")
    arrow(ax, 11.8, 12.6, y)
    # 4. the score
    block(ax, 15.3, y, 4.9, 1.7, "attention score of tokens i, j", sub="s_ij = Σ_r Re[ q_i,r · k̄_j,r · e^(i (t_i − t_j) / τ_r) ]", color=PINK, dshape=False)
    dim(ax, 15.3, y - 1.0, "depends on the time gap t_i − t_j only")
    # notes, in two columns below the row
    ax.text(8.1, 2.35, "τ_1 … τ_21 is a log ladder of timescales. Every head of every layer\ngets its own band of the ladder: 756 distinct rungs over the model,\nperiods 0.022 d → 500 d. A rung of timescale τ turns once per lag 2π τ.", ha="left", va="top", fontsize=FS["note"], color=GREY)
    ax.text(14.2, 2.35, "No absolute clock: the ladder spans lags from minutes to the\nfull window, so attention can match points any number of days\napart, and the same weights serve any window start.", ha="left", va="top", fontsize=FS["note"], color=GREY)
    # ladder sketch, bottom left
    xs = np.logspace(np.log10(0.022), np.log10(500), 36)
    for xv in xs:
        xx = 0.6 + 6.5 * (np.log10(xv) - np.log10(0.022)) / (np.log10(500) - np.log10(0.022))
        ax.plot([xx, xx], [0.95, 1.25], color=PURPLE, lw=1.2)
    for lab, xv in (("0.022 d", 0.022), ("0.1", 0.1), ("1", 1), ("10", 10), ("100", 100), ("500 d", 500)):
        xx = 0.6 + 6.5 * (np.log10(xv) - np.log10(0.022)) / (np.log10(500) - np.log10(0.022))
        ax.text(xx, 0.8, lab, ha="center", va="top", fontsize=FS["note"] - 1, color=GREY)
    ax.text(3.85, 1.4, "the 36 head-ladders tile the period range (log axis)", ha="center", va="bottom", fontsize=FS["note"], color=GREY)
    ax.text(0.2, 6.35, "Continuous rotary positions (RoPE) on irregular times and bands", ha="left", va="top", fontsize=FS["big"], color=INK, fontweight="bold")
    ax.text(8.1, 0.45, "no positional vector is added to the tokens: time enters only as a rotation of the query and key channels,\nso every attention score sees relative time at 756 fixed frequencies", ha="left", va="center", fontsize=FS["note"], color=GREY)
    fig.savefig(out / "diagram_rope.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="project/results/slides")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spectral(out); rope(out)
    print(f"wrote {out}/diagram_spectral.png and diagram_rope.png")


if __name__ == "__main__":
    main()
