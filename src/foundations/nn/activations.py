from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from foundations.types import Array


@dataclass(frozen=True)
class Activation:
    name: str
    forward: Callable[[Array], Array]
    derivative: Callable[[Array], Array]


def _sigmoid(z: Array) -> Array:
    return 1 / (1 + np.exp(-z))


def _sigmoid_derivative(z: Array) -> Array:
    s = _sigmoid(z)
    return s * (1 - s)


def _relu(z: Array) -> Array:
    return np.maximum(z, 0)


def _relu_derivative(z: Array) -> Array:
    return (z > 0).astype(z.dtype)


def _tanh_derivative(z: Array) -> Array:
    return 1 - np.tanh(z) ** 2


def _identity(z: Array) -> Array:
    return z


def _ones(z: Array) -> Array:
    return np.ones_like(z)


SIGMOID = Activation("sigmoid", _sigmoid, _sigmoid_derivative)
RELU = Activation("relu", _relu, _relu_derivative)
TANH = Activation("tanh", np.tanh, _tanh_derivative)
LINEAR = Activation("linear", _identity, _ones)

ACTIVATIONS: dict[str, Activation] = {a.name: a for a in (SIGMOID, RELU, TANH, LINEAR)}
