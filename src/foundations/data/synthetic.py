import numpy as np

from foundations.data.dataset import Dataset

LOWER_MOON_VERTICAL_OFFSET = 0.5


def make_moons(n_samples: int, noise: float, rng: np.random.Generator) -> Dataset:
    half = n_samples // 2
    angles = np.linspace(0, np.pi, half)
    upper = np.column_stack([np.cos(angles), np.sin(angles)])
    lower = np.column_stack([1 - np.cos(angles), 1 - np.sin(angles) - LOWER_MOON_VERTICAL_OFFSET])
    x = np.vstack([upper, lower]) + rng.normal(scale=noise, size=(2 * half, 2))
    y = np.vstack([np.zeros((half, 1)), np.ones((half, 1))])
    return Dataset(x=x, y=y)
