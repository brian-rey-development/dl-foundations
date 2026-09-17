from dataclasses import replace

from foundations.nn.layers import Dense, DenseGrads
from foundations.nn.network import Grads, Network


def _sgd_update_layer(layer: Dense, grads: DenseGrads, learning_rate: float) -> Dense:
    return replace(
        layer,
        weights=layer.weights - learning_rate * grads.weights,
        biases=layer.biases - learning_rate * grads.biases,
    )


def sgd_step(network: Network, grads: Grads, learning_rate: float) -> Network:
    return tuple(
        _sgd_update_layer(layer, layer_grads, learning_rate)
        for layer, layer_grads in zip(network, grads, strict=True)
    )
