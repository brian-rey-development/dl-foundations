from collections.abc import Sequence

import numpy as np

from foundations.nn.activations import Activation
from foundations.nn.layers import (
    Dense,
    DenseCache,
    DenseGrads,
    dense_backward,
    dense_forward,
    init_dense,
)
from foundations.types import Array

Network = tuple[Dense, ...]
Caches = tuple[DenseCache, ...]
Grads = tuple[DenseGrads, ...]

DECISION_THRESHOLD = 0.5


def init_network(
    layer_sizes: Sequence[int],
    hidden_activation: Activation,
    output_activation: Activation,
    rng: np.random.Generator,
) -> Network:
    pairs = list(zip(layer_sizes[:-1], layer_sizes[1:], strict=True))
    last = len(pairs) - 1
    return tuple(
        init_dense(n_in, n_out, output_activation if i == last else hidden_activation, rng)
        for i, (n_in, n_out) in enumerate(pairs)
    )


def forward(network: Network, x: Array) -> tuple[Array, Caches]:
    caches: list[DenseCache] = []
    current = x
    for layer in network:
        current, cache = dense_forward(layer, current)
        caches.append(cache)
    return current, tuple(caches)


def backward(network: Network, caches: Caches, grad_output: Array) -> Grads:
    grads: list[DenseGrads] = []
    current = grad_output
    for layer, cache in zip(reversed(network), reversed(caches), strict=True):
        current, layer_grads = dense_backward(layer, cache, current)
        grads.append(layer_grads)
    return tuple(reversed(grads))


def predict_proba(network: Network, x: Array) -> Array:
    output, _ = forward(network, x)
    return output


def predict(network: Network, x: Array) -> Array:
    return (predict_proba(network, x) >= DECISION_THRESHOLD).astype(float)


def count_parameters(network: Network) -> int:
    return sum(layer.weights.size + layer.biases.size for layer in network)
