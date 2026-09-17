import numpy as np
import pytest

from foundations.nn import LINEAR, RELU, dense_backward, dense_forward, init_dense


def test_forward_output_shape(rng: np.random.Generator) -> None:
    layer = init_dense(3, 5, RELU, rng)
    out, cache = dense_forward(layer, rng.normal(size=(7, 3)))
    assert out.shape == (7, 5)
    assert cache.pre_activation.shape == (7, 5)


def test_linear_layer_backward_matches_closed_form(rng: np.random.Generator) -> None:
    layer = init_dense(3, 2, LINEAR, rng)
    inputs = rng.normal(size=(4, 3))
    grad_out = rng.normal(size=(4, 2))
    _, cache = dense_forward(layer, inputs)
    grad_in, grads = dense_backward(layer, cache, grad_out)
    np.testing.assert_allclose(grads.weights, inputs.T @ grad_out)
    np.testing.assert_allclose(grads.biases, grad_out.sum(axis=0, keepdims=True))
    np.testing.assert_allclose(grad_in, grad_out @ layer.weights.T)


def test_he_init_scales_with_fan_in(rng: np.random.Generator) -> None:
    layer = init_dense(1000, 200, RELU, rng)
    assert layer.weights.std() == pytest.approx(np.sqrt(2 / 1000), rel=0.05)
