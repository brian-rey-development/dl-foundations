from dataclasses import dataclass

import numpy as np

from foundations.config import ExperimentConfig
from foundations.data import Split, Standardizer, make_moons, shuffle, split, standardize_split
from foundations.diagnostics import gradient_check
from foundations.nn import RELU, SIGMOID, Network, init_network, predict
from foundations.persistence import ModelArtifact, ModelMetadata
from foundations.training import ConfusionMatrix, History, confusion_matrix, evaluate, train


@dataclass(frozen=True)
class TestReport:
    loss: float
    accuracy: float
    confusion: ConfusionMatrix


@dataclass(frozen=True)
class ExperimentResult:
    network: Network
    scaler: Standardizer
    data: Split
    history: History
    gradient_check_error: float
    test: TestReport


def _prepare_data(config: ExperimentConfig, rng: np.random.Generator) -> tuple[Split, Standardizer]:
    raw = shuffle(make_moons(config.data.n_samples, config.data.noise, rng), rng)
    return standardize_split(split(raw, config.data.train_fraction, config.data.val_fraction))


def _report_test(network: Network, data: Split) -> TestReport:
    loss, acc = evaluate(network, data.test)
    confusion = confusion_matrix(data.test.y, predict(network, data.test.x))
    return TestReport(loss=loss, accuracy=acc, confusion=confusion)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    rng = np.random.default_rng(config.seed)
    data, scaler = _prepare_data(config, rng)
    network = init_network(config.model.layer_sizes, RELU, SIGMOID, rng)
    error = gradient_check(network, data.train.take(config.gradient_check_samples))
    trained, history = train(network, data, config.training, rng)
    return ExperimentResult(
        network=trained,
        scaler=scaler,
        data=data,
        history=history,
        gradient_check_error=error,
        test=_report_test(trained, data),
    )


def to_artifact(result: ExperimentResult, config: ExperimentConfig) -> ModelArtifact:
    metadata = ModelMetadata(
        layer_sizes=config.model.layer_sizes,
        seed=config.seed,
        training=config.training,
        test_loss=result.test.loss,
        test_accuracy=result.test.accuracy,
        trained_at=ModelMetadata.now(),
    )
    return ModelArtifact(network=result.network, scaler=result.scaler, metadata=metadata)
