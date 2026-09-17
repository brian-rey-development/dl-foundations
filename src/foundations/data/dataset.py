from dataclasses import dataclass

import numpy as np

from foundations.types import Array


@dataclass(frozen=True)
class Dataset:
    x: Array
    y: Array

    def __len__(self) -> int:
        return len(self.y)

    def take(self, n: int) -> "Dataset":
        return Dataset(x=self.x[:n], y=self.y[:n])


@dataclass(frozen=True)
class Split:
    train: Dataset
    val: Dataset
    test: Dataset


def shuffle(data: Dataset, rng: np.random.Generator) -> Dataset:
    order = rng.permutation(len(data))
    return Dataset(x=data.x[order], y=data.y[order])


def split(data: Dataset, train_frac: float, val_frac: float) -> Split:
    train_end = int(len(data) * train_frac)
    val_end = train_end + int(len(data) * val_frac)
    return Split(
        train=Dataset(data.x[:train_end], data.y[:train_end]),
        val=Dataset(data.x[train_end:val_end], data.y[train_end:val_end]),
        test=Dataset(data.x[val_end:], data.y[val_end:]),
    )
