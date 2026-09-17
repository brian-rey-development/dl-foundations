import numpy as np

from foundations.data import Dataset, Split, fit_standardizer, standardize, standardize_split


def test_standardize_gives_zero_mean_unit_std(rng: np.random.Generator) -> None:
    data = Dataset(x=rng.normal(loc=5, scale=3, size=(500, 2)), y=np.zeros((500, 1)))
    scaled = standardize(data, fit_standardizer(data.x))
    np.testing.assert_allclose(scaled.x.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(scaled.x.std(axis=0), 1, atol=1e-12)


def test_standardize_split_fits_only_on_train(rng: np.random.Generator) -> None:
    train = Dataset(x=rng.normal(size=(100, 2)), y=np.zeros((100, 1)))
    test = Dataset(x=rng.normal(loc=10, size=(20, 2)), y=np.zeros((20, 1)))
    scaled, _ = standardize_split(Split(train=train, val=test, test=test))
    assert abs(scaled.test.x.mean()) > 5
