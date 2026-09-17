from dataclasses import dataclass, field
from pathlib import Path

from foundations.training import TrainConfig


@dataclass(frozen=True)
class DataConfig:
    n_samples: int = 1000
    noise: float = 0.2
    train_fraction: float = 0.7
    val_fraction: float = 0.15


@dataclass(frozen=True)
class ModelConfig:
    layer_sizes: tuple[int, ...] = (2, 16, 16, 1)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(
        default_factory=lambda: TrainConfig(epochs=200, batch_size=32, learning_rate=0.05)
    )
    gradient_check_samples: int = 5
    output_dir: Path = Path("outputs")


DEFAULT_CONFIG = ExperimentConfig()
