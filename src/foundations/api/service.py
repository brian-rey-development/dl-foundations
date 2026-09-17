import threading
from dataclasses import replace
from pathlib import Path

import numpy as np

from foundations.api.schemas import (
    MAX_HISTORY_POINTS,
    EpochPoint,
    ModelInfo,
    Point,
    Prediction,
    TrainRequest,
    TrainResponse,
)
from foundations.config import DEFAULT_CONFIG, ExperimentConfig, ModelConfig
from foundations.data import Dataset, standardize
from foundations.experiment import run_experiment, to_artifact
from foundations.nn import DECISION_THRESHOLD, count_parameters, predict_proba
from foundations.persistence import ModelArtifact, load_artifact, save_artifact
from foundations.training import EpochMetrics, History, TrainConfig


def _to_config(request: TrainRequest) -> ExperimentConfig:
    training = TrainConfig(request.epochs, request.batch_size, request.learning_rate)
    model = ModelConfig(layer_sizes=tuple(request.layer_sizes))
    return replace(DEFAULT_CONFIG, seed=request.seed, training=training, model=model)


def _to_epoch_point(index: int, metrics: EpochMetrics) -> EpochPoint:
    return EpochPoint(
        epoch=index + 1,
        train_loss=metrics.train_loss,
        val_loss=metrics.val_loss,
        train_accuracy=metrics.train_accuracy,
        val_accuracy=metrics.val_accuracy,
    )


def _sample_history(history: History) -> list[EpochPoint]:
    step = max(1, len(history) // MAX_HISTORY_POINTS)
    last = len(history) - 1
    return [
        _to_epoch_point(i, m) for i, m in enumerate(history) if (i + 1) % step == 0 or i == last
    ]


def _to_prediction(probability: float) -> Prediction:
    return Prediction(probability=probability, label=int(probability >= DECISION_THRESHOLD))


def describe(artifact: ModelArtifact) -> ModelInfo:
    meta = artifact.metadata
    return ModelInfo(
        layer_sizes=list(meta.layer_sizes),
        activations=[layer.activation.name for layer in artifact.network],
        n_parameters=count_parameters(artifact.network),
        seed=meta.seed,
        epochs=meta.training.epochs,
        batch_size=meta.training.batch_size,
        learning_rate=meta.training.learning_rate,
        test_loss=meta.test_loss,
        test_accuracy=meta.test_accuracy,
        trained_at=meta.trained_at,
    )


class ModelService:
    def __init__(self, artifact_path: Path) -> None:
        self._path = artifact_path
        self._lock = threading.Lock()
        self._artifact: ModelArtifact | None = None

    @property
    def is_loaded(self) -> bool:
        return self._artifact is not None

    def load_or_train(self) -> ModelInfo:
        if self._path.exists():
            self._artifact = load_artifact(self._path)
            return describe(self._artifact)
        return self.train(TrainRequest()).model

    def train(self, request: TrainRequest) -> TrainResponse:
        config = _to_config(request)
        result = run_experiment(config)
        artifact = to_artifact(result, config)
        with self._lock:
            save_artifact(artifact, self._path)
            self._artifact = artifact
        return TrainResponse(model=describe(artifact), history=_sample_history(result.history))

    def info(self) -> ModelInfo:
        return describe(self._require_artifact())

    def predict(self, points: list[Point]) -> list[Prediction]:
        artifact = self._require_artifact()
        raw = Dataset(x=np.array([[p.x1, p.x2] for p in points]), y=np.empty((len(points), 1)))
        scaled = standardize(raw, artifact.scaler)
        probabilities = predict_proba(artifact.network, scaled.x).ravel()
        return [_to_prediction(float(p)) for p in probabilities]

    def _require_artifact(self) -> ModelArtifact:
        if self._artifact is None:
            raise RuntimeError("model is not loaded")
        return self._artifact
