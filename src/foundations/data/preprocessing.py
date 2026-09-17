from dataclasses import dataclass

from foundations.data.dataset import Dataset, Split
from foundations.types import Array


@dataclass(frozen=True)
class Standardizer:
    mean: Array
    std: Array


def fit_standardizer(x: Array) -> Standardizer:
    return Standardizer(mean=x.mean(axis=0), std=x.std(axis=0))


def standardize(data: Dataset, scaler: Standardizer) -> Dataset:
    return Dataset(x=(data.x - scaler.mean) / scaler.std, y=data.y)


def standardize_split(data: Split) -> tuple[Split, Standardizer]:
    scaler = fit_standardizer(data.train.x)
    scaled = Split(
        train=standardize(data.train, scaler),
        val=standardize(data.val, scaler),
        test=standardize(data.test, scaler),
    )
    return scaled, scaler
