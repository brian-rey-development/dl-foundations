import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from foundations.data import make_moons, shuffle, split, standardize_split
    from foundations.nn import RELU, SIGMOID, init_network, predict_proba
    from foundations.training import TrainConfig, confusion_matrix, train
    from foundations.visualization import draw_decision_boundary, plot_series, use_style

    use_style()
    return (
        RELU,
        SIGMOID,
        TrainConfig,
        confusion_matrix,
        draw_decision_boundary,
        init_network,
        make_moons,
        mo,
        np,
        plot_series,
        plt,
        predict_proba,
        shuffle,
        split,
        standardize_split,
        train,
    )


@app.cell
def _(mo):
    mo.md("""
    # 07. Training and evaluation

    Companion to `docs/07-training-and-evaluation.md`. Overfit on purpose, then trade precision
    against recall by moving the decision threshold.

    Start with few points, high noise and a wide network, and push the epochs up. Watch the
    validation loss turn around while the training loss keeps falling.
    """)
    return


@app.cell
def _(mo):
    n_points = mo.ui.slider(40, 1000, step=20, value=60, label="training + validation points")
    noise = mo.ui.slider(0.0, 0.5, step=0.05, value=0.35, label="noise")
    width = mo.ui.dropdown(options=["4", "16", "64"], value="64", label="hidden width")
    epochs = mo.ui.slider(50, 3000, step=50, value=1500, label="epochs")
    mo.vstack([mo.hstack([n_points, noise]), mo.hstack([width, epochs])])
    return epochs, n_points, noise, width


@app.cell
def _(
    RELU,
    SIGMOID,
    TrainConfig,
    epochs,
    init_network,
    make_moons,
    n_points,
    noise,
    np,
    shuffle,
    split,
    standardize_split,
    train,
    width,
):
    rng = np.random.default_rng(42)
    raw = shuffle(make_moons(n_points.value, noise.value, rng), rng)
    data, _scaler = standardize_split(split(raw, 0.5, 0.5))
    hidden = int(width.value)
    network = init_network((2, hidden, hidden, 1), RELU, SIGMOID, rng)
    trained, history = train(
        network, data, TrainConfig(epochs=epochs.value, batch_size=8, learning_rate=0.05), rng
    )
    return data, history, trained


@app.cell
def _(data, draw_decision_boundary, history, plot_series, plt, trained):
    fig_fit, (ax_loss, ax_bound) = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_series(ax_loss, history, "loss", "Train vs validation loss")
    ax_loss.set_xscale("log")
    draw_decision_boundary(ax_bound, trained, data.train, "Boundary over the training points")
    fig_fit
    return


@app.cell
def _(history, mo, np):
    val_losses = np.array([m.val_loss for m in history])
    best_epoch = int(val_losses.argmin()) + 1
    mo.md(
        f"""
        Validation loss bottomed out at epoch **{best_epoch}** ({val_losses.min():.3f}) and ended at
        {val_losses[-1]:.3f} after {len(history)} epochs. Early stopping would return the network from
        epoch {best_epoch} and throw the rest away.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Threshold, precision and recall

    The network outputs a probability. Turning it into a label needs a threshold, and 0.5 is a
    default, not a law. Lower it and you call more points positive: recall goes up, precision goes
    down. Which one you want depends on what a miss costs versus what a false alarm costs.
    """)
    return


@app.cell
def _(mo):
    threshold = mo.ui.slider(0.05, 0.95, step=0.05, value=0.5, label="decision threshold")
    threshold
    return (threshold,)


@app.cell
def _(confusion_matrix, data, mo, predict_proba, threshold, trained):
    proba = predict_proba(trained, data.val.x)
    labels = (proba >= threshold.value).astype(float)
    cm = confusion_matrix(data.val.y, labels)
    acc = float((labels == data.val.y).mean())
    mo.ui.table(
        [
            {"metric": "accuracy", "value": round(acc, 3)},
            {"metric": "precision", "value": round(cm.precision, 3)},
            {"metric": "recall", "value": round(cm.recall, 3)},
            {
                "metric": "TP / FP / FN / TN",
                "value": f"{cm.true_positive} / {cm.false_positive} / {cm.false_negative} / {cm.true_negative}",
            },
        ]
    )
    return


if __name__ == "__main__":
    app.run()
