from dataclasses import dataclass

from foundations.types import Array


@dataclass(frozen=True)
class ConfusionMatrix:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        predicted_positive = self.true_positive + self.false_positive
        return self.true_positive / predicted_positive if predicted_positive else 0.0

    @property
    def recall(self) -> float:
        actual_positive = self.true_positive + self.false_negative
        return self.true_positive / actual_positive if actual_positive else 0.0


def accuracy(y_true: Array, y_pred: Array) -> float:
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true: Array, y_pred: Array) -> ConfusionMatrix:
    positives, negatives = y_true == 1, y_true == 0
    return ConfusionMatrix(
        true_positive=int((positives & (y_pred == 1)).sum()),
        true_negative=int((negatives & (y_pred == 0)).sum()),
        false_positive=int((negatives & (y_pred == 1)).sum()),
        false_negative=int((positives & (y_pred == 0)).sum()),
    )
