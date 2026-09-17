import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
SERIES = (BLUE, ORANGE, AQUA, YELLOW)

CLASS_COLORS = (BLUE, ORANGE)
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_orange", [BLUE, "#ececea", ORANGE])

LINE_WIDTH = 2.0
MARKER_SIZE = 28
DPI = 140

RC_PARAMS: dict[str, str | bool | float] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "text.color": INK,
    "legend.frameon": False,
    "lines.linewidth": LINE_WIDTH,
    "font.size": 10,
}


def use_style() -> None:
    plt.style.use(RC_PARAMS)


def label_line_end(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=color,
        fontsize=9,
        fontweight="bold",
    )
