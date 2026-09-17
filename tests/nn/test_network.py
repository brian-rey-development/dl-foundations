import numpy as np

from foundations.nn import RELU, SIGMOID, count_parameters, forward, init_network, predict


def test_init_network_builds_expected_shapes(rng: np.random.Generator) -> None:
    network = init_network((2, 8, 4, 1), RELU, SIGMOID, rng)
    assert [layer.weights.shape for layer in network] == [(2, 8), (8, 4), (4, 1)]
    assert network[-1].activation is SIGMOID
    assert all(layer.activation is RELU for layer in network[:-1])


def test_count_parameters(rng: np.random.Generator) -> None:
    network = init_network((2, 16, 16, 1), RELU, SIGMOID, rng)
    assert count_parameters(network) == (2 * 16 + 16) + (16 * 16 + 16) + (16 * 1 + 1)


def test_forward_returns_probabilities_and_caches(rng: np.random.Generator) -> None:
    network = init_network((2, 4, 1), RELU, SIGMOID, rng)
    out, caches = forward(network, rng.normal(size=(10, 2)))
    assert out.shape == (10, 1)
    assert len(caches) == 2
    assert out.min() >= 0 and out.max() <= 1


def test_predict_returns_binary_labels(rng: np.random.Generator) -> None:
    network = init_network((2, 4, 1), RELU, SIGMOID, rng)
    labels = predict(network, rng.normal(size=(10, 2)))
    assert set(np.unique(labels)) <= {0.0, 1.0}
