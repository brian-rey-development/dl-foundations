import numpy as np

from foundations.nn import DenseGrads, Network, sgd_step


def test_sgd_moves_against_gradient_and_is_pure(tiny_network: Network) -> None:
    grads = tuple(
        DenseGrads(weights=np.ones_like(layer.weights), biases=np.ones_like(layer.biases))
        for layer in tiny_network
    )
    before = [layer.weights.copy() for layer in tiny_network]
    updated = sgd_step(tiny_network, grads, learning_rate=0.1)
    for old, layer, new in zip(before, tiny_network, updated, strict=True):
        np.testing.assert_allclose(new.weights, old - 0.1)
        np.testing.assert_array_equal(layer.weights, old)
