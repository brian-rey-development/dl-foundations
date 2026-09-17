import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from foundations.nn import SIGMOID, binary_cross_entropy, binary_cross_entropy_gradient
    from foundations.visualization import use_style
    from foundations.visualization.style import BLUE, ORANGE

    use_style()
    return (
        BLUE,
        ORANGE,
        SIGMOID,
        binary_cross_entropy,
        binary_cross_entropy_gradient,
        mo,
        np,
        plt,
    )


@app.cell
def _(mo):
    mo.md("""
    # 04. Loss function

    Companion to `docs/04-loss-function.md`. Binary cross-entropy for a single example, and why it
    beats squared error when the output is a sigmoid.
    """)
    return


@app.cell
def _(mo):
    p = mo.ui.slider(0.001, 0.999, step=0.001, value=0.7, label="predicted probability p")
    y = mo.ui.radio(options={"y = 1": 1.0, "y = 0": 0.0}, value="y = 1", label="true label")
    mo.hstack([p, y])
    return p, y


@app.cell
def _(binary_cross_entropy, mo, np, p, y):
    loss_value = binary_cross_entropy(np.array([[y.value]]), np.array([[p.value]]))
    mse_value = float((y.value - p.value) ** 2)
    mo.md(
        f"""
        For `y = {int(y.value)}` and `p = {p.value:.3f}`:
        cross-entropy **{loss_value:.4f}**, squared error {mse_value:.4f}.
        Drag p toward the wrong end and watch cross-entropy explode while squared error caps at 1.
        """
    )
    return


@app.cell
def _(BLUE, ORANGE, np, p, plt, y):
    grid = np.linspace(0.001, 0.999, 500)
    fig_loss, ax_loss = plt.subplots(figsize=(6.5, 4))
    ax_loss.plot(grid, -np.log(grid), color=BLUE, label="y = 1: -log(p)")
    ax_loss.plot(grid, -np.log(1 - grid), color=ORANGE, label="y = 0: -log(1 - p)")
    current = -np.log(p.value) if y.value == 1.0 else -np.log(1 - p.value)
    ax_loss.scatter([p.value], [current], color="#0b0b0b", zorder=5, s=40)
    ax_loss.set(title="Binary cross-entropy, one example", xlabel="p", ylabel="loss", ylim=(0, 7))
    ax_loss.legend()
    fig_loss
    return


@app.cell
def _(mo):
    mo.md("""
    ## Why not squared error

    Below: the gradient of the loss with respect to the pre-activation z, when the output is
    sigmoid(z) and the true label is 1. With cross-entropy the gradient is exactly p - 1: the more
    wrong you are, the harder the push. With squared error the sigmoid derivative multiplies in,
    and when the network is confidently wrong (z very negative) the gradient dies out. Exactly when
    you need it most.
    """)
    return


@app.cell
def _(BLUE, ORANGE, SIGMOID, binary_cross_entropy_gradient, np, plt):
    zs = np.linspace(-8, 8, 400)
    ps = SIGMOID.forward(zs)
    ones = np.ones_like(ps)
    grad_bce = (
        binary_cross_entropy_gradient(ones[:, None], ps[:, None]).ravel()
        * SIGMOID.derivative(zs)
        * len(zs)
    )
    grad_mse = 2 * (ps - 1) * SIGMOID.derivative(zs)
    fig_grad, ax_grad = plt.subplots(figsize=(6.5, 4))
    ax_grad.plot(zs, grad_bce, color=BLUE, label="cross-entropy: dL/dz = p - 1")
    ax_grad.plot(zs, grad_mse, color=ORANGE, label="squared error: dL/dz = 2(p - 1) p (1 - p)")
    ax_grad.set(
        title="Gradient w.r.t. z for a positive example",
        xlabel="z (pre-activation)",
        ylabel="dL/dz",
    )
    ax_grad.legend()
    fig_grad
    return


if __name__ == "__main__":
    app.run()
