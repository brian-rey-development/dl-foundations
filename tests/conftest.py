import numpy as np
import pytest

from foundations.data import Dataset, make_moons
from foundations.nn import RELU, SIGMOID, Network, init_network

SEED = 0
TINY_SAMPLES = 8


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture
def tiny_dataset(rng: np.random.Generator) -> Dataset:
    return make_moons(TINY_SAMPLES, noise=0.1, rng=rng)


@pytest.fixture
def tiny_network(rng: np.random.Generator) -> Network:
    return init_network((2, 4, 1), RELU, SIGMOID, rng)
