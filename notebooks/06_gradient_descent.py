import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from foundations.data import make_moons, shuffle, split, standardize_split
    from foundations.nn import RELU, SIGMOID, init_network
    from foundations.training import TrainConfig, train
    from foundations.visualization import plot_series, use_style
    from foundations.visualization.style import INK_SECONDARY, ORANGE

    use_style()
    return (
        INK_SECONDARY,
        ORANGE,
        RELU,
        SIGMOID,
        TrainConfig,
        init_network,
        make_moons,
        mo,
        np,
        plot_series,
        plt,
        shuffle,
        split,
        standardize_split,
        train,
    )


@app.cell
def _(mo):
    mo.md("""
    # 06. Gradient descent

    Companion to `docs/06-gradient-descent.md`. First on a parabola you can see, then on the real
    network.
    """)
    return


@app.cell
def _(mo):
    lr_1d = mo.ui.slider(0.01, 1.2, step=0.01, value=0.3, label="learning rate")
    steps_1d = mo.ui.slider(1, 30, step=1, value=8, label="steps")
    mo.hstack([lr_1d, steps_1d])
    return lr_1d, steps_1d


@app.cell
def _(INK_SECONDARY, ORANGE, lr_1d, np, plt, steps_1d):
    path = [-0.5]
    for _ in range(steps_1d.value):
        w = path[-1]
        path.append(w - lr_1d.value * 2 * (w - 2))
    path_arr = np.array(path)
    ws = np.linspace(-1, 5, 300)
    fig_1d, ax_1d = plt.subplots(figsize=(6.5, 4))
    ax_1d.plot(ws, (ws - 2) ** 2, color=INK_SECONDARY, linewidth=1.2)
    ax_1d.plot(path_arr, (path_arr - 2) ** 2, color=ORANGE, marker="o", markersize=5, linewidth=1.2)
    ax_1d.set(
        title=f"L(w) = (w - 2)^2, lr = {lr_1d.value}", xlabel="w", ylabel="loss", ylim=(-0.5, 9)
    )
    fig_1d
    return


@app.cell
def _(mo):
    mo.md("""
    The gradient of `(w - 2)^2` is `2(w - 2)`, so each step is `w <- w - lr * 2(w - 2)`.
    Below lr = 0.5 the steps shrink toward the minimum. At exactly 0.5 you land on it in one step.
    Between 0.5 and 1.0 you overshoot but still converge, bouncing side to side. Above 1.0 every
    step lands farther away than the last.

    ## On the real network
    """)
    return


@app.cell
def _(mo):
    lr_net = mo.ui.dropdown(
        options=["0.001", "0.01", "0.05", "0.2", "0.5", "1.0", "3.0"],
        value="0.05",
        label="learning rate",
    )
    batch = mo.ui.dropdown(options=["1", "8", "32", "128", "700"], value="32", label="batch size")
    epochs = mo.ui.slider(10, 300, step=10, value=100, label="epochs")
    mo.hstack([lr_net, batch, epochs])
    return batch, epochs, lr_net


@app.cell
def _(make_moons, np, shuffle, split, standardize_split):
    data_rng = np.random.default_rng(42)
    data, _scaler = standardize_split(
        split(shuffle(make_moons(1000, 0.2, data_rng), data_rng), 0.7, 0.15)
    )
    return (data,)


@app.cell
def _(
    RELU,
    SIGMOID,
    TrainConfig,
    batch,
    data,
    epochs,
    init_network,
    lr_net,
    np,
    train,
):
    net_rng = np.random.default_rng(0)
    network = init_network((2, 16, 16, 1), RELU, SIGMOID, net_rng)
    config = TrainConfig(
        epochs=epochs.value, batch_size=int(batch.value), learning_rate=float(lr_net.value)
    )
    _trained, history = train(network, data, config, net_rng)
    steps_per_epoch = -(-len(data.train) // config.batch_size)
    return config, history, steps_per_epoch


@app.cell
def _(config, history, mo, plot_series, plt, steps_per_epoch):
    fig_net, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(12, 4))
    plot_series(ax_l, history, "loss", f"lr = {config.learning_rate}, batch = {config.batch_size}")
    plot_series(ax_a, history, "accuracy", "Accuracy")
    mo.vstack(
        [
            mo.md(
                f"{steps_per_epoch} gradient steps per epoch, {steps_per_epoch * config.epochs} in total."
            ),
            fig_net,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    Things to try: `lr = 3.0` (watch it explode or oscillate), `batch = 700` (full batch, smooth but
    slow per epoch), `batch = 1` (pure SGD, noisy, and notice how long it takes in wall-clock time
    even though each step is tiny).
    """)
    return


if __name__ == "__main__":
    app.run()
