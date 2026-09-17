import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from foundations.data import make_moons, shuffle, split, standardize_split
    from foundations.nn import RELU, SIGMOID, count_parameters, init_network
    from foundations.training import TrainConfig, train
    from foundations.visualization import draw_decision_boundary, use_style

    use_style()
    return (
        RELU,
        SIGMOID,
        TrainConfig,
        count_parameters,
        draw_decision_boundary,
        init_network,
        make_moons,
        mo,
        np,
        plt,
        shuffle,
        split,
        standardize_split,
        train,
    )


@app.cell
def _(mo):
    mo.md("""
    # 02. Neurons and layers

    Companion to `docs/02-neurons-and-layers.md`. Pick an architecture, train it, look at the boundary.
    One neuron can only draw a line. Hidden layers with ReLU bend it, and the number of kinks is
    bounded by the number of hidden neurons.
    """)
    return


@app.cell
def _(mo):
    architecture = mo.ui.dropdown(
        options={
            "(2, 1): one neuron": (2, 1),
            "(2, 2, 1): two hidden neurons": (2, 2, 1),
            "(2, 4, 1)": (2, 4, 1),
            "(2, 16, 1)": (2, 16, 1),
            "(2, 16, 16, 1): default": (2, 16, 16, 1),
            "(2, 64, 64, 1)": (2, 64, 64, 1),
        },
        value="(2, 16, 16, 1): default",
        label="architecture",
    )
    gain = mo.ui.slider(-4, 2, step=1, value=0, label="init scale, log10 multiplier on He")
    mo.hstack([architecture, gain])
    return architecture, gain


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
    architecture,
    count_parameters,
    data,
    gain,
    init_network,
    np,
    train,
):
    from dataclasses import replace

    net_rng = np.random.default_rng(0)
    initial = init_network(architecture.value, RELU, SIGMOID, net_rng)
    scale = 10.0**gain.value
    initial = tuple(replace(layer, weights=layer.weights * scale) for layer in initial)
    trained, history = train(
        initial, data, TrainConfig(epochs=150, batch_size=32, learning_rate=0.05), net_rng
    )
    n_params = count_parameters(trained)
    return history, n_params, trained


@app.cell
def _(history, mo, n_params):
    last = history[-1]
    mo.md(
        f"""
        **{n_params} parameters.** After 150 epochs: train accuracy `{last.train_accuracy:.3f}`,
        validation accuracy `{last.val_accuracy:.3f}`, validation loss `{last.val_loss:.4f}`.
        """
    )
    return


@app.cell
def _(data, draw_decision_boundary, plt, trained):
    fig_boundary, ax_boundary = plt.subplots(figsize=(6, 5))
    draw_decision_boundary(ax_boundary, trained, data.test, "Decision boundary (test set)")
    fig_boundary
    return


@app.cell
def _(mo):
    mo.md("""
    ## Shapes

    The first layer of the default network, on a batch of 5 points. Weights are `(n_in, n_out)`,
    one column per neuron. Biases broadcast across the batch.
    """)
    return


@app.cell
def _(data, mo, trained):
    from foundations.nn import dense_forward

    batch = data.train.x[:5]
    out, cache = dense_forward(trained[0], batch)
    mo.ui.table(
        [
            {"tensor": "inputs X", "shape": str(batch.shape)},
            {"tensor": "weights W", "shape": str(trained[0].weights.shape)},
            {"tensor": "biases b", "shape": str(trained[0].biases.shape)},
            {"tensor": "pre-activation Z = XW + b", "shape": str(cache.pre_activation.shape)},
            {"tensor": "output relu(Z)", "shape": str(out.shape)},
        ]
    )
    return


if __name__ == "__main__":
    app.run()
