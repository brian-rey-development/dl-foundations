import numpy as np
import pytest

from foundations.nn import binary_cross_entropy, binary_cross_entropy_gradient


def test_perfect_predictions_give_near_zero_loss() -> None:
    y = np.array([[1.0], [0.0]])
    assert binary_cross_entropy(y, y) == pytest.approx(0.0, abs=1e-9)


def test_confident_wrong_prediction_is_penalised_more() -> None:
    y = np.array([[1.0]])
    assert binary_cross_entropy(y, np.array([[0.1]])) > binary_cross_entropy(y, np.array([[0.4]]))


def test_loss_is_finite_at_extreme_probabilities() -> None:
    y = np.array([[1.0], [0.0]])
    assert np.isfinite(binary_cross_entropy(y, np.array([[0.0], [1.0]])))


def test_gradient_sign_points_towards_target() -> None:
    y = np.array([[1.0]])
    assert binary_cross_entropy_gradient(y, np.array([[0.3]]))[0, 0] < 0
    assert binary_cross_entropy_gradient(1 - y, np.array([[0.3]]))[0, 0] > 0
