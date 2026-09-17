import numpy as np
import pytest

from foundations.nn import LINEAR, RELU, SIGMOID, TANH, Activation

FINITE_DIFF_STEP = 1e-6


@pytest.mark.parametrize("activation", [SIGMOID, RELU, TANH, LINEAR], ids=lambda a: a.name)
def test_derivative_matches_finite_differences(activation: Activation) -> None:
    z = np.linspace(-3, 3, 61) + 0.013
    plus, minus = activation.forward(z + FINITE_DIFF_STEP), activation.forward(z - FINITE_DIFF_STEP)
    numerical = (plus - minus) / (2 * FINITE_DIFF_STEP)
    np.testing.assert_allclose(activation.derivative(z), numerical, atol=1e-6)


def test_sigmoid_is_bounded_and_saturates_in_float64() -> None:
    out = SIGMOID.forward(np.array([-50.0, 0.0, 50.0]))
    assert out.min() >= 0 and out.max() <= 1
    assert out[1] == pytest.approx(0.5)
    assert out[2] == 1.0


def test_relu_zeroes_negatives() -> None:
    np.testing.assert_array_equal(RELU.forward(np.array([-2.0, 0.0, 3.0])), [0.0, 0.0, 3.0])
