import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from foundations.data import make_moons, shuffle, split, standardize_split
    from foundations.diagnostics import gradient_check, relative_error
    from foundations.nn import (
        RELU,
        SIGMOID,
        backward,
        binary_cross_entropy_gradient,
        forward,
        init_network,
    )
    from foundations.visualization import use_style
    from foundations.visualization.style import BLUE, ORANGE

    use_style()
    return (
        BLUE,
        ORANGE,
        RELU,
        SIGMOID,
        backward,
        binary_cross_entropy_gradient,
        forward,
        gradient_check,
        init_network,
        make_moons,
        mo,
        np,
        plt,
        relative_error,
        shuffle,
        split,
        standardize_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # 08. Gradient check

    Companion to `docs/08-gradient-check.md`. Compare the analytical gradient from `backward` with
    a finite-difference estimate, and see how the perturbation size trades truncation error
    against floating point round-off.
    """)
    return


@app.cell
def _(
    RELU,
    SIGMOID,
    init_network,
    make_moons,
    np,
    shuffle,
    split,
    standardize_split,
):
    rng = np.random.default_rng(42)
    data, _scaler = standardize_split(split(shuffle(make_moons(1000, 0.2, rng), rng), 0.7, 0.15))
    tiny = data.train.take(5)
    network = init_network((2, 4, 1), RELU, SIGMOID, rng)
    return network, tiny


@app.cell
def _(mo):
    log_eps = mo.ui.slider(-12, -1, step=0.5, value=-5, label="log10(epsilon)")
    log_eps
    return (log_eps,)


@app.cell
def _(gradient_check, log_eps, mo, network, tiny):
    eps = 10.0**log_eps.value
    err = gradient_check(network, tiny, eps)
    verdict = "correct" if err < 1e-7 else ("suspicious" if err < 1e-4 else "bug or bad epsilon")
    mo.md(f"epsilon = `{eps:.0e}`, relative error = **{err:.2e}** ({verdict})")
    return


@app.cell
def _(BLUE, ORANGE, gradient_check, log_eps, network, np, plt, tiny):
    eps_grid = np.logspace(-12, -1, 23)
    errors = [gradient_check(network, tiny, float(e)) for e in eps_grid]
    fig_eps, ax_eps = plt.subplots(figsize=(7, 4))
    ax_eps.plot(eps_grid, errors, color=BLUE, marker="o", markersize=4)
    ax_eps.axvline(10.0**log_eps.value, color=ORANGE, linestyle="--")
    ax_eps.set(
        xscale="log",
        yscale="log",
        xlabel="epsilon",
        ylabel="relative error",
        title="Round-off on the left, truncation on the right",
    )
    fig_eps
    return


@app.cell
def _(mo):
    mo.md("""
    ## Break the backward pass

    Introduce a deliberate bug in the bias gradient and see the check catch it. The broken version
    uses the mean over the batch instead of the sum. Training with it still "works", just worse:
    that is why the check exists.
    """)
    return


@app.cell
def _(mo):
    broken = mo.ui.switch(value=False, label="use mean instead of sum for dL/db")
    broken
    return (broken,)


@app.cell
def _(
    backward,
    binary_cross_entropy_gradient,
    broken,
    forward,
    mo,
    network,
    np,
    relative_error,
    tiny,
):

    from foundations.diagnostics.gradient_check import _numerical_gradient

    y_pred, caches = forward(network, tiny.x)
    grads = backward(network, caches, binary_cross_entropy_gradient(tiny.y, y_pred))
    bias_grad = grads[-1].biases
    if broken.value:
        bias_grad = bias_grad / len(tiny)
    numerical = _numerical_gradient(network, tiny, len(network) - 1, "biases", 1e-5)
    err_bias = relative_error(bias_grad, numerical)
    mo.md(
        f"""
        Output-layer bias gradient, analytical `{np.round(bias_grad.ravel(), 6).tolist()}` vs
        numerical `{np.round(numerical.ravel(), 6).tolist()}`. Relative error **{err_bias:.2e}**.
        """
    )
    return


if __name__ == "__main__":
    app.run()
