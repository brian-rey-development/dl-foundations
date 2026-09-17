import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from foundations.data import make_moons, shuffle, split, standardize_split
    from foundations.nn import (
        RELU,
        SIGMOID,
        backward,
        binary_cross_entropy_gradient,
        forward,
        init_network,
    )

    return (
        RELU,
        SIGMOID,
        backward,
        binary_cross_entropy_gradient,
        forward,
        init_network,
        make_moons,
        mo,
        np,
        shuffle,
        split,
        standardize_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # 05. Backpropagation

    Companion to `docs/05-backpropagation.md`. One forward pass and one backward pass on a small
    batch, with every intermediate shape laid out. The rule to check at each row: a parameter's
    gradient has the parameter's shape.
    """)
    return


@app.cell
def _(mo):
    batch_size = mo.ui.slider(1, 32, step=1, value=4, label="batch size")
    hidden = mo.ui.slider(1, 32, step=1, value=3, label="hidden neurons")
    mo.hstack([batch_size, hidden])
    return batch_size, hidden


@app.cell
def _(
    RELU,
    SIGMOID,
    batch_size,
    hidden,
    init_network,
    make_moons,
    np,
    shuffle,
    split,
    standardize_split,
):
    rng = np.random.default_rng(42)
    data, _scaler = standardize_split(split(shuffle(make_moons(1000, 0.2, rng), rng), 0.7, 0.15))
    batch = data.train.take(batch_size.value)
    network = init_network((2, hidden.value, 1), RELU, SIGMOID, rng)
    return batch, network


@app.cell
def _(backward, batch, binary_cross_entropy_gradient, forward, network):
    y_pred, caches = forward(network, batch.x)
    grad_output = binary_cross_entropy_gradient(batch.y, y_pred)
    grads = backward(network, caches, grad_output)
    return caches, grad_output, grads, y_pred


@app.cell
def _(batch, caches, mo, network, y_pred):
    forward_rows = [{"step": "X (inputs)", "shape": str(batch.x.shape)}]
    for fi, (layer, cache) in enumerate(zip(network, caches, strict=True)):
        forward_rows.append({"step": f"layer {fi} W", "shape": str(layer.weights.shape)})
        forward_rows.append({"step": f"layer {fi} b", "shape": str(layer.biases.shape)})
        forward_rows.append(
            {"step": f"layer {fi} z = XW + b", "shape": str(cache.pre_activation.shape)}
        )
    forward_rows.append({"step": "p (output)", "shape": str(y_pred.shape)})
    mo.vstack([mo.md("## Forward"), mo.ui.table(forward_rows)])
    return


@app.cell
def _(grad_output, grads, mo, network):
    backward_rows = [{"step": "dL/dp (from the loss)", "shape": str(grad_output.shape)}]
    for bi in reversed(range(len(network))):
        backward_rows.append({"step": f"layer {bi} dL/dW", "shape": str(grads[bi].weights.shape)})
        backward_rows.append({"step": f"layer {bi} dL/db", "shape": str(grads[bi].biases.shape)})
    mo.vstack([mo.md("## Backward"), mo.ui.table(backward_rows)])
    return


@app.cell
def _(mo):
    mo.md("""
    ## The bias gradient is a sum over the batch

    The bias was broadcast to every row in the forward pass. Whatever expands in the forward pass
    sums in the backward pass. Below, the output layer's `dL/dz` for each example, and its column
    sum, which is exactly `dL/db`.
    """)
    return


@app.cell
def _(caches, grad_output, grads, network, np):
    last = network[-1]
    delta = grad_output * last.activation.derivative(caches[-1].pre_activation)
    {
        "dL/dz per example": np.round(delta.ravel(), 5).tolist(),
        "sum over batch": float(np.round(delta.sum(), 5)),
        "dL/db from backward": float(np.round(grads[-1].biases.ravel()[0], 5)),
    }
    return


if __name__ == "__main__":
    app.run()
