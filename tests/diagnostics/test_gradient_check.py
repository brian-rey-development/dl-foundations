import numpy as np

from foundations.data import Dataset
from foundations.diagnostics import gradient_check, relative_error
from foundations.nn import Network

TOLERANCE = 1e-6


def test_analytical_matches_numerical(tiny_network: Network, tiny_dataset: Dataset) -> None:
    assert gradient_check(tiny_network, tiny_dataset) < TOLERANCE


def test_relative_error_is_zero_for_identical_and_large_for_different() -> None:
    a = np.array([1.0, 2.0])
    assert relative_error(a, a) == 0.0
    assert relative_error(a, -a) > 0.9
