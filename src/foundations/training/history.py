from dataclasses import dataclass


@dataclass(frozen=True)
class EpochMetrics:
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


History = tuple[EpochMetrics, ...]
