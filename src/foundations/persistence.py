import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from foundations.data import Standardizer
from foundations.nn import ACTIVATIONS, Dense, Network
from foundations.training import TrainConfig

FORMAT_VERSION = 1
METADATA_KEY = "metadata"
SCALER_MEAN_KEY = "scaler_mean"
SCALER_STD_KEY = "scaler_std"


@dataclass(frozen=True)
class ModelMetadata:
    layer_sizes: tuple[int, ...]
    seed: int
    training: TrainConfig
    test_loss: float
    test_accuracy: float
    trained_at: str
    format_version: int = FORMAT_VERSION

    @staticmethod
    def now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ModelArtifact:
    network: Network
    scaler: Standardizer
    metadata: ModelMetadata


def _weights_key(index: int) -> str:
    return f"layer_{index}_weights"


def _biases_key(index: int) -> str:
    return f"layer_{index}_biases"


def _metadata_payload(artifact: ModelArtifact) -> str:
    activations = [layer.activation.name for layer in artifact.network]
    return json.dumps({**asdict(artifact.metadata), "activations": activations})


def save_artifact(artifact: ModelArtifact, path: Path) -> None:
    arrays = {
        SCALER_MEAN_KEY: artifact.scaler.mean,
        SCALER_STD_KEY: artifact.scaler.std,
        METADATA_KEY: np.array(_metadata_payload(artifact)),
    }
    for i, layer in enumerate(artifact.network):
        arrays[_weights_key(i)] = layer.weights
        arrays[_biases_key(i)] = layer.biases
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, allow_pickle=False, **arrays)


def _parse_metadata(raw: dict) -> ModelMetadata:
    return ModelMetadata(
        layer_sizes=tuple(raw["layer_sizes"]),
        seed=raw["seed"],
        training=TrainConfig(**raw["training"]),
        test_loss=raw["test_loss"],
        test_accuracy=raw["test_accuracy"],
        trained_at=raw["trained_at"],
        format_version=raw["format_version"],
    )


def load_artifact(path: Path) -> ModelArtifact:
    with np.load(path, allow_pickle=False) as bundle:
        raw = json.loads(str(bundle[METADATA_KEY]))
        network = tuple(
            Dense(
                weights=bundle[_weights_key(i)],
                biases=bundle[_biases_key(i)],
                activation=ACTIVATIONS[name],
            )
            for i, name in enumerate(raw["activations"])
        )
        scaler = Standardizer(mean=bundle[SCALER_MEAN_KEY], std=bundle[SCALER_STD_KEY])
    return ModelArtifact(network=network, scaler=scaler, metadata=_parse_metadata(raw))
