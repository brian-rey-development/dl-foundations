from dataclasses import dataclass

import numpy as np

from foundations.nn.activations import RELU, Activation
from foundations.types import Array

HE_GAIN = 2.0
XAVIER_GAIN = 1.0


@dataclass(frozen=True)
class Dense:
    weights: Array
    biases: Array
    activation: Activation


@dataclass(frozen=True)
class DenseCache:
    inputs: Array
    pre_activation: Array


@dataclass(frozen=True)
class DenseGrads:
    weights: Array
    biases: Array


def init_dense(n_in: int, n_out: int, activation: Activation, rng: np.random.Generator) -> Dense:
    gain = HE_GAIN if activation is RELU else XAVIER_GAIN
    weights = rng.normal(scale=np.sqrt(gain / n_in), size=(n_in, n_out))
    biases = np.zeros((1, n_out))
    return Dense(weights=weights, biases=biases, activation=activation)


def dense_forward(layer: Dense, inputs: Array) -> tuple[Array, DenseCache]:
    pre_activation = inputs @ layer.weights + layer.biases
    output = layer.activation.forward(pre_activation)
    return output, DenseCache(inputs=inputs, pre_activation=pre_activation)


def dense_backward(layer: Dense, cache: DenseCache, grad_output: Array) -> tuple[Array, DenseGrads]:
    grad_pre_activation = grad_output * layer.activation.derivative(cache.pre_activation)
    grad_weights = cache.inputs.T @ grad_pre_activation
    grad_biases = grad_pre_activation.sum(axis=0, keepdims=True)
    grad_inputs = grad_pre_activation @ layer.weights.T
    return grad_inputs, DenseGrads(weights=grad_weights, biases=grad_biases)
