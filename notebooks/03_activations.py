import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from foundations.data import make_moons, shuffle, split, standardize_split
    from foundations.nn import ACTIVATIONS, SIGMOID, init_network
    from foundations.training import TrainConfig, train
    from foundations.visualization import draw_decision_boundary, plot_series, use_style
    from foundations.visualization.style import BLUE, ORANGE

    use_style()
    return (
        ACTIVATIONS,
        BLUE,
        ORANGE,
        SIGMOID,
        TrainConfig,
        draw_decision_boundary,
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
    # 03. Activations

    Companion to `docs/03-activations.md`. Pick a function, see it with its derivative, then train
    a network that uses it in the hidden layers.
    """)
    return


@app.cell
def _(ACTIVATIONS, mo):
    hidden = mo.ui.dropdown(options=list(ACTIVATIONS), value="relu", label="hidden activation")
    hidden
    return (hidden,)


@app.cell
def _(ACTIVATIONS, BLUE, ORANGE, hidden, np, plt):
    act = ACTIVATIONS[hidden.value]
    z = np.linspace(-5, 5, 400)
    fig_act, ax_act = plt.subplots(figsize=(6, 3.5))
    ax_act.plot(z, act.forward(z), color=BLUE, label="f(z)")
    ax_act.plot(z, act.derivative(z), color=ORANGE, linestyle="--", label="f'(z)")
    ax_act.axhline(0, color="#52514e", linewidth=0.6)
    ax_act.set(title=act.name, xlabel="z", ylim=(-1.5, 2.5))
    ax_act.legend()
    fig_act
    return (act,)


@app.cell
def _(act, mo):
    peak = float(act.derivative(__import__("numpy").linspace(-5, 5, 2001)).max())
    mo.md(
        f"""
        Largest derivative on [-5, 5]: **{peak:.3g}**. Every layer multiplies the gradient by at most
        this number. With sigmoid that is 0.25 per layer; ten layers deep the gradient is at most
        0.25^10, about 1e-6. That is the vanishing gradient problem.
        """
    )
    return


@app.cell
def _(make_moons, np, shuffle, split, standardize_split):
    data_rng = np.random.default_rng(42)
    data, _scaler = standardize_split(
        split(shuffle(make_moons(1000, 0.2, data_rng), data_rng), 0.7, 0.15)
    )
    return (data,)


@app.cell
def _(SIGMOID, TrainConfig, act, data, init_network, np, train):
    net_rng = np.random.default_rng(0)
    network = init_network((2, 16, 16, 1), act, SIGMOID, net_rng)
    trained, history = train(
        network, data, TrainConfig(epochs=150, batch_size=32, learning_rate=0.05), net_rng
    )
    return history, trained


@app.cell
def _(data, draw_decision_boundary, history, plot_series, plt, trained):
    fig_train, (ax_loss, ax_bound) = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_series(ax_loss, history, "loss", f"Loss with {trained[0].activation.name} hidden layers")
    draw_decision_boundary(ax_bound, trained, data.test, "Decision boundary (test set)")
    fig_train
    return


@app.cell
def _(mo):
    mo.md("""
    With `linear` the boundary is a straight line no matter how many layers there are: a stack of
    linear maps is one linear map. With `sigmoid` training is slower than with `relu` for the reason
    in the derivative plot above.
    """)
    return


if __name__ == "__main__":
    app.run()
