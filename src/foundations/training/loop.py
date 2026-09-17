from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from foundations.data import Dataset, Split
from foundations.nn import (
    Network,
    backward,
    binary_cross_entropy,
    binary_cross_entropy_gradient,
    forward,
    predict,
    sgd_step,
)
from foundations.training.history import EpochMetrics, History
from foundations.training.metrics import accuracy


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float


def iterate_batches(data: Dataset, batch_size: int, rng: np.random.Generator) -> Iterator[Dataset]:
    order = rng.permutation(len(data))
    for start in range(0, len(data), batch_size):
        batch = order[start : start + batch_size]
        yield Dataset(x=data.x[batch], y=data.y[batch])


def train_step(network: Network, batch: Dataset, learning_rate: float) -> Network:
    y_pred, caches = forward(network, batch.x)
    grad_output = binary_cross_entropy_gradient(batch.y, y_pred)
    grads = backward(network, caches, grad_output)
    return sgd_step(network, grads, learning_rate)


def train_epoch(
    network: Network, data: Dataset, config: TrainConfig, rng: np.random.Generator
) -> Network:
    current = network
    for batch in iterate_batches(data, config.batch_size, rng):
        current = train_step(current, batch, config.learning_rate)
    return current


def evaluate(network: Network, data: Dataset) -> tuple[float, float]:
    y_proba, _ = forward(network, data.x)
    loss = binary_cross_entropy(data.y, y_proba)
    return loss, accuracy(data.y, predict(network, data.x))


def _measure(network: Network, data: Split) -> EpochMetrics:
    train_loss, train_accuracy = evaluate(network, data.train)
    val_loss, val_accuracy = evaluate(network, data.val)
    return EpochMetrics(train_loss, train_accuracy, val_loss, val_accuracy)


def train(
    network: Network, data: Split, config: TrainConfig, rng: np.random.Generator
) -> tuple[Network, History]:
    history: list[EpochMetrics] = []
    current = network
    for _ in range(config.epochs):
        current = train_epoch(current, data.train, config, rng)
        history.append(_measure(current, data))
    return current, tuple(history)
