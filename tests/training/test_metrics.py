import numpy as np
import pytest

from foundations.training import accuracy, confusion_matrix

Y_TRUE = np.array([[1], [1], [0], [0], [1]], dtype=float)
Y_PRED = np.array([[1], [0], [0], [1], [1]], dtype=float)


def test_accuracy() -> None:
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(3 / 5)


def test_confusion_matrix_counts() -> None:
    cm = confusion_matrix(Y_TRUE, Y_PRED)
    counts = (cm.true_positive, cm.true_negative, cm.false_positive, cm.false_negative)
    assert counts == (2, 1, 1, 1)
    assert cm.precision == pytest.approx(2 / 3)
    assert cm.recall == pytest.approx(2 / 3)


def test_precision_and_recall_handle_empty_denominators() -> None:
    cm = confusion_matrix(np.zeros((3, 1)), np.zeros((3, 1)))
    assert cm.precision == 0.0 and cm.recall == 0.0
