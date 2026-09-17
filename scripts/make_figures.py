"""Generate every figure referenced by docs/. Run with `make figures`."""

from pathlib import Path

import numpy as np

from foundations.config import DEFAULT_CONFIG
from foundations.data import Split, make_moons, shuffle, split, standardize_split
from foundations.diagnostics import gradient_check
from foundations.experiment import run_experiment
from foundations.nn import LINEAR, RELU, SIGMOID, TANH, Activation, init_network
from foundations.training import TrainConfig, train
from foundations.visualization import draw_decision_boundary, plot_history, plot_series, use_style
from foundations.visualization.style import (
    BLUE,
    CLASS_COLORS,
    DPI,
    INK_SECONDARY,
    MARKER_SIZE,
    ORANGE,
    SERIES,
)

use_style()

import matplotlib.pyplot as plt  # noqa: E402

ASSETS = Path("docs/assets")
SEED = 42
LEARNING_RATES = (0.001, 0.01, 0.05, 0.5)
PERTURBATIONS = np.logspace(-11, -1, 21)
OVERFIT_SAMPLES = 60
OVERFIT_NOISE = 0.35
OVERFIT_EPOCHS = 3000
OVERFIT_LAYERS = (2, 64, 64, 1)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(ASSETS / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote {ASSETS / name}")


def fresh_data(n_samples: int = 1000, noise: float = 0.2) -> Split:
    rng = np.random.default_rng(SEED)
    raw = shuffle(make_moons(n_samples, noise, rng), rng)
    scaled, _ = standardize_split(split(raw, 0.7, 0.15))
    return scaled


def figure_dataset() -> None:
    rng = np.random.default_rng(SEED)
    data = make_moons(1000, 0.2, rng)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, color in enumerate(CLASS_COLORS):
        mask = data.y.ravel() == label
        ax.scatter(
            data.x[mask, 0],
            data.x[mask, 1],
            c=color,
            s=MARKER_SIZE,
            edgecolors="white",
            linewidths=0.6,
            label=f"class {label}",
        )
    ax.legend(loc="upper right")
    ax.set(title="Two moons: 1000 points, noise = 0.2", xlabel="x1", ylabel="x2")
    ax.set_aspect("equal")
    save(fig, "dataset.png")


def figure_activations() -> None:
    z = np.linspace(-5, 5, 400)
    activations: tuple[Activation, ...] = (SIGMOID, RELU, TANH, LINEAR)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharex=True)
    for ax, act in zip(axes, activations, strict=True):
        ax.plot(z, act.forward(z), color=BLUE, label="f(z)")
        ax.plot(z, act.derivative(z), color=ORANGE, linestyle="--", label="f'(z)")
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.6)
        ax.axvline(0, color=INK_SECONDARY, linewidth=0.6)
        ax.set(title=act.name, xlabel="z", ylim=(-1.5, 2.5))
    axes[0].legend(loc="upper left")
    save(fig, "activations.png")


def figure_bce() -> None:
    p = np.linspace(0.001, 0.999, 500)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(p, -np.log(p), color=BLUE)
    ax.plot(p, -np.log(1 - p), color=ORANGE)
    ax.text(0.15, 2.3, "y = 1: loss = -log(p)", color=BLUE, fontweight="bold")
    ax.text(0.45, 2.3, "y = 0: loss = -log(1 - p)", color=ORANGE, fontweight="bold")
    ax.set(title="Binary cross-entropy for a single example", xlabel="p = P(y = 1)", ylabel="loss")
    ax.set_ylim(0, 5)
    save(fig, "bce_loss.png")


def _descend(start: float, lr: float, steps: int) -> np.ndarray:
    path = [start]
    for _ in range(steps):
        w = path[-1]
        path.append(w - lr * 2 * (w - 2))
    return np.array(path)


def figure_gradient_descent_1d() -> None:
    w = np.linspace(-1, 5, 300)
    cases = (
        (0.05, "lr = 0.05: slow"),
        (0.3, "lr = 0.3: about right"),
        (1.05, "lr = 1.05: diverges"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (lr, title) in zip(axes, cases, strict=True):
        path = _descend(start=-0.5, lr=lr, steps=8)
        ax.plot(w, (w - 2) ** 2, color=INK_SECONDARY, linewidth=1.2)
        ax.plot(path, (path - 2) ** 2, color=ORANGE, marker="o", markersize=5, linewidth=1.2)
        ax.annotate(
            "start",
            (path[0], (path[0] - 2) ** 2),
            xytext=(-30, 8),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=9,
        )
        ax.set(title=title, xlabel="w", ylim=(-0.5, 9))
    axes[0].set_ylabel("loss L(w) = (w - 2)^2")
    save(fig, "gradient_descent_1d.png")


def figure_learning_rates() -> None:
    data = fresh_data()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for lr, color in zip(LEARNING_RATES, SERIES, strict=True):
        rng = np.random.default_rng(SEED)
        network = init_network((2, 16, 16, 1), RELU, SIGMOID, rng)
        _, history = train(network, data, TrainConfig(200, 32, lr), rng)
        ax.plot(range(1, 201), [m.val_loss for m in history], color=color, label=f"lr = {lr}")
    ax.legend(loc="upper right")
    ax.set(title="Validation loss by learning rate", xlabel="epoch", ylabel="val loss")
    save(fig, "learning_rate_comparison.png")


def figure_linear_vs_nonlinear() -> None:
    data = fresh_data()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    cases = (
        ((2, 1), "One neuron (2, 1): a straight line only"),
        ((2, 16, 16, 1), "Network (2, 16, 16, 1)"),
    )
    for ax, (layers, title) in zip(axes, cases, strict=True):
        rng = np.random.default_rng(SEED)
        network = init_network(layers, RELU, SIGMOID, rng)
        trained, _ = train(network, data, TrainConfig(200, 32, 0.05), rng)
        draw_decision_boundary(ax, trained, data.test, title)
    save(fig, "linear_vs_nonlinear.png")


def figure_training_run() -> None:
    result = run_experiment(DEFAULT_CONFIG)
    plot_history(result.history, ASSETS / "training_curves.png")
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    draw_decision_boundary(ax, result.network, result.data.test, "Decision boundary (test set)")
    save(fig, "decision_boundary.png")
    print(f"wrote {ASSETS / 'training_curves.png'}")


def figure_overfitting() -> None:
    rng = np.random.default_rng(SEED)
    raw = shuffle(make_moons(OVERFIT_SAMPLES, OVERFIT_NOISE, rng), rng)
    data, _ = standardize_split(split(raw, 0.5, 0.5))
    network = init_network(OVERFIT_LAYERS, RELU, SIGMOID, rng)
    trained, history = train(network, data, TrainConfig(OVERFIT_EPOCHS, 8, 0.05), rng)
    fig, (ax_loss, ax_bound) = plt.subplots(1, 2, figsize=(12, 4.4))
    title = f"Overfitting: {OVERFIT_SAMPLES} points, network {OVERFIT_LAYERS}"
    plot_series(ax_loss, history, "loss", title)
    ax_loss.set_xscale("log")
    draw_decision_boundary(ax_bound, trained, data.train, "Memorised boundary (training points)")
    save(fig, "overfitting.png")


def figure_gradient_check() -> None:
    rng = np.random.default_rng(SEED)
    data = fresh_data().train.take(5)
    network = init_network((2, 4, 1), RELU, SIGMOID, rng)
    errors = [gradient_check(network, data, float(eps)) for eps in PERTURBATIONS]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(PERTURBATIONS, errors, color=BLUE, marker="o", markersize=4)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="perturbation epsilon",
        ylabel="relative error",
        title="Gradient check: relative error vs epsilon",
    )
    ax.axvline(1e-5, color=ORANGE, linestyle="--", linewidth=1.2)
    ax.text(1.3e-5, max(errors) / 5, "epsilon in use (1e-5)", color=ORANGE, fontweight="bold")
    save(fig, "gradient_check.png")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    figure_dataset()
    figure_activations()
    figure_bce()
    figure_gradient_descent_1d()
    figure_learning_rates()
    figure_linear_vs_nonlinear()
    figure_training_run()
    figure_overfitting()
    figure_gradient_check()


if __name__ == "__main__":
    main()
