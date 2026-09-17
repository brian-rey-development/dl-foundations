from dataclasses import replace

import numpy as np

from foundations.data import Dataset
from foundations.nn import (
    Dense,
    Network,
    backward,
    binary_cross_entropy,
    binary_cross_entropy_gradient,
    forward,
)
from foundations.types import Array

PERTURBATION = 1e-5
STABILITY_EPSILON = 1e-12
PARAMETER_FIELDS = ("weights", "biases")


def _loss_for(network: Network, data: Dataset) -> float:
    y_pred, _ = forward(network, data.x)
    return binary_cross_entropy(data.y, y_pred)


def _with_layer(network: Network, index: int, layer: Dense) -> Network:
    return network[:index] + (layer,) + network[index + 1 :]


def _perturbed(
    network: Network, index: int, field: str, position: tuple[int, ...], delta: float
) -> Network:
    layer = network[index]
    values = getattr(layer, field).copy()
    values[position] += delta
    return _with_layer(network, index, replace(layer, **{field: values}))


def _numerical_partial(
    network: Network,
    data: Dataset,
    index: int,
    field: str,
    position: tuple[int, ...],
    perturbation: float,
) -> float:
    loss_plus = _loss_for(_perturbed(network, index, field, position, perturbation), data)
    loss_minus = _loss_for(_perturbed(network, index, field, position, -perturbation), data)
    return (loss_plus - loss_minus) / (2 * perturbation)


def _numerical_gradient(
    network: Network, data: Dataset, index: int, field: str, perturbation: float
) -> Array:
    shape = getattr(network[index], field).shape
    partials = [
        _numerical_partial(network, data, index, field, position, perturbation)
        for position in np.ndindex(shape)
    ]
    return np.array(partials).reshape(shape)


def relative_error(analytical: Array, numerical: Array) -> float:
    numerator = np.linalg.norm(analytical - numerical)
    denominator = np.linalg.norm(analytical) + np.linalg.norm(numerical) + STABILITY_EPSILON
    return float(numerator / denominator)


def gradient_check(network: Network, data: Dataset, perturbation: float = PERTURBATION) -> float:
    y_pred, caches = forward(network, data.x)
    analytical = backward(network, caches, binary_cross_entropy_gradient(data.y, y_pred))
    errors = [
        relative_error(
            getattr(analytical[i], field),
            _numerical_gradient(network, data, i, field, perturbation),
        )
        for i in range(len(network))
        for field in PARAMETER_FIELDS
    ]
    return max(errors)
