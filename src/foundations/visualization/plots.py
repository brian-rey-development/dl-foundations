from pathlib import Path

import numpy as np

from foundations.data import Dataset
from foundations.nn import Network, predict_proba
from foundations.training import History
from foundations.visualization.style import (
    CLASS_COLORS,
    DIVERGING,
    DPI,
    INK,
    MARKER_SIZE,
    SERIES,
    label_line_end,
    use_style,
)

use_style()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

GRID_RESOLUTION = 200
GRID_MARGIN = 0.5
CONTOUR_LEVELS = 20
CLASS_CMAP = ListedColormap(CLASS_COLORS)


def plot_series(ax: plt.Axes, history: History, field: str, title: str) -> None:
    epochs = np.arange(1, len(history) + 1)
    for color, prefix in zip(SERIES, ("train", "val"), strict=False):
        values = [getattr(m, f"{prefix}_{field}") for m in history]
        ax.plot(epochs, values, color=color, label=prefix)
        label_line_end(ax, epochs[-1], values[-1], prefix, color)
    ax.set(title=title, xlabel="epoch", ylabel=field)
    ax.legend(loc="best")


def plot_history(history: History, path: Path) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    plot_series(ax_loss, history, "loss", "Loss (binary cross-entropy)")
    plot_series(ax_acc, history, "accuracy", "Accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def prediction_grid(network: Network, data: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, y_min = data.x.min(axis=0) - GRID_MARGIN
    x_max, y_max = data.x.max(axis=0) + GRID_MARGIN
    xs, ys = np.meshgrid(
        np.linspace(x_min, x_max, GRID_RESOLUTION), np.linspace(y_min, y_max, GRID_RESOLUTION)
    )
    grid = np.column_stack([xs.ravel(), ys.ravel()])
    return xs, ys, predict_proba(network, grid).reshape(xs.shape)


def draw_decision_boundary(ax: plt.Axes, network: Network, data: Dataset, title: str) -> None:
    xs, ys, probabilities = prediction_grid(network, data)
    ax.contourf(xs, ys, probabilities, levels=CONTOUR_LEVELS, cmap=DIVERGING, alpha=0.85)
    ax.contour(xs, ys, probabilities, levels=[0.5], colors=INK, linewidths=1.2)
    ax.scatter(
        data.x[:, 0],
        data.x[:, 1],
        c=data.y.ravel(),
        cmap=CLASS_CMAP,
        edgecolors="white",
        linewidths=0.6,
        s=MARKER_SIZE,
    )
    ax.grid(False)
    ax.set(title=title, xlabel="x1 (std)", ylabel="x2 (std)")


def plot_decision_boundary(network: Network, data: Dataset, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    draw_decision_boundary(ax, network, data, "Decision boundary (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
