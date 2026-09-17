import numpy as np

from foundations.data import Dataset, make_moons, shuffle, split


def test_make_moons_shapes_and_balanced_labels(rng: np.random.Generator) -> None:
    data = make_moons(100, noise=0.0, rng=rng)
    assert data.x.shape == (100, 2)
    assert data.y.shape == (100, 1)
    assert data.y.sum() == 50


def test_shuffle_keeps_pairs_aligned(rng: np.random.Generator) -> None:
    data = Dataset(x=np.arange(10).reshape(-1, 1).astype(float), y=np.arange(10).reshape(-1, 1))
    shuffled = shuffle(data, rng)
    assert not np.array_equal(shuffled.x, data.x)
    assert np.array_equal(shuffled.x.ravel(), shuffled.y.ravel())


def test_split_fractions_are_disjoint_and_complete() -> None:
    data = Dataset(x=np.arange(100).reshape(-1, 1).astype(float), y=np.zeros((100, 1)))
    parts = split(data, train_frac=0.7, val_frac=0.15)
    assert (len(parts.train), len(parts.val), len(parts.test)) == (70, 15, 15)
    all_x = np.concatenate([parts.train.x, parts.val.x, parts.test.x])
    assert np.array_equal(all_x, data.x)
