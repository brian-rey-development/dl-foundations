import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from foundations.data import make_moons, shuffle, split, standardize_split
    from foundations.visualization import use_style
    from foundations.visualization.style import CLASS_COLORS, MARKER_SIZE

    use_style()
    return (
        CLASS_COLORS,
        MARKER_SIZE,
        make_moons,
        mo,
        np,
        plt,
        shuffle,
        split,
        standardize_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # 01. Data

    Companion to `docs/01-data.md`. Move the sliders and watch the dataset change.
    Every run is deterministic for a given seed.
    """)
    return


@app.cell
def _(mo):
    noise = mo.ui.slider(0.0, 0.6, step=0.05, value=0.2, label="noise")
    seed = mo.ui.slider(0, 100, step=1, value=42, label="seed")
    n_samples = mo.ui.slider(100, 2000, step=100, value=1000, label="samples")
    mo.hstack([noise, seed, n_samples])
    return n_samples, noise, seed


@app.cell
def _(make_moons, n_samples, noise, np, seed, shuffle):
    rng = np.random.default_rng(seed.value)
    raw = shuffle(make_moons(n_samples.value, noise.value, rng), rng)
    return (raw,)


@app.cell
def _(CLASS_COLORS, MARKER_SIZE, plt, raw):
    fig_data, ax_data = plt.subplots(figsize=(6, 4.5))
    for label, color in enumerate(CLASS_COLORS):
        mask = raw.y.ravel() == label
        ax_data.scatter(
            raw.x[mask, 0],
            raw.x[mask, 1],
            c=color,
            s=MARKER_SIZE,
            edgecolors="white",
            label=f"class {label}",
        )
    ax_data.legend()
    ax_data.set(
        title=f"{len(raw)} points, x shape {raw.x.shape}, y shape {raw.y.shape}",
        xlabel="x1",
        ylabel="x2",
    )
    ax_data.set_aspect("equal")
    fig_data
    return


@app.cell
def _(mo):
    mo.md("""
    ## Splits

    Train is what the network sees. Validation is what you look at when picking hyperparameters.
    Test is touched once, at the end. Change the fractions and check the counts.
    """)
    return


@app.cell
def _(mo):
    train_frac = mo.ui.slider(0.5, 0.9, step=0.05, value=0.7, label="train fraction")
    val_frac = mo.ui.slider(0.05, 0.3, step=0.05, value=0.15, label="val fraction")
    mo.hstack([train_frac, val_frac])
    return train_frac, val_frac


@app.cell
def _(mo, raw, split, standardize_split, train_frac, val_frac):
    parts = split(raw, train_frac.value, val_frac.value)
    scaled, scaler = standardize_split(parts)
    mo.ui.table(
        [
            {
                "split": "train",
                "size": len(parts.train),
                "class 1 share": round(float(parts.train.y.mean()), 3),
            },
            {
                "split": "val",
                "size": len(parts.val),
                "class 1 share": round(float(parts.val.y.mean()), 3),
            },
            {
                "split": "test",
                "size": len(parts.test),
                "class 1 share": round(float(parts.test.y.mean()), 3),
            },
        ]
    )
    return scaled, scaler


@app.cell
def _(mo, scaler):
    mo.md(
        f"""
        ## Standardization

        Fitted on train only: mean `{scaler.mean.round(3)}`, std `{scaler.std.round(3)}`.
        Those two vectors are applied unchanged to val and test, and later to any point the API receives.
        Train ends up at mean 0 and std 1 by construction. Val and test land close to it, not exactly on it,
        because they were transformed with train statistics. That small mismatch is the honest one.
        """
    )
    return


@app.cell
def _(np, scaled):
    rows = []
    for name, part in (("train", scaled.train), ("val", scaled.val), ("test", scaled.test)):
        rows.append(
            {
                "split": name,
                "mean": np.round(part.x.mean(axis=0), 3).tolist(),
                "std": np.round(part.x.std(axis=0), 3).tolist(),
            }
        )
    rows
    return


if __name__ == "__main__":
    app.run()
