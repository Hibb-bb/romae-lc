"""Plot style: the colours and small helpers shared by the figure scripts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "project/results/figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the validated reference palette
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NAME_ENC = "RoPE Transformer + Masked Autoencoder"
NAME_HAND = "Hand-crafted features"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.titleweight": "medium", "axes.titlelocation": "left", "savefig.facecolor": "white"})


def tidy(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    print("wrote", f"{FIG}/{name}.png")


