import numpy as np

from foundations.types import Array

CLIP_EPSILON = 1e-12


def _clip_probabilities(y_pred: Array) -> Array:
    return np.clip(y_pred, CLIP_EPSILON, 1 - CLIP_EPSILON)


def binary_cross_entropy(y_true: Array, y_pred: Array) -> float:
    p = _clip_probabilities(y_pred)
    per_sample = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return float(per_sample.mean())


def binary_cross_entropy_gradient(y_true: Array, y_pred: Array) -> Array:
    p = _clip_probabilities(y_pred)
    return (p - y_true) / (p * (1 - p)) / len(y_true)
