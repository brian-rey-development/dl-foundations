from pathlib import Path

import numpy as np

from foundations.config import DEFAULT_CONFIG, ExperimentConfig, ModelConfig
from foundations.experiment import run_experiment, to_artifact
from foundations.nn import predict_proba
from foundations.persistence import load_artifact, save_artifact
from foundations.training import TrainConfig

FAST_CONFIG = ExperimentConfig(
    seed=1,
    model=ModelConfig(layer_sizes=(2, 4, 1)),
    training=TrainConfig(epochs=3, batch_size=32, learning_rate=0.05),
    data=DEFAULT_CONFIG.data,
)


def test_round_trip_preserves_predictions_and_metadata(tmp_path: Path) -> None:
    result = run_experiment(FAST_CONFIG)
    artifact = to_artifact(result, FAST_CONFIG)
    path = tmp_path / "model.npz"
    save_artifact(artifact, path)
    loaded = load_artifact(path)

    x = result.data.test.x
    expected = predict_proba(artifact.network, x)
    np.testing.assert_array_equal(predict_proba(loaded.network, x), expected)
    np.testing.assert_array_equal(loaded.scaler.mean, artifact.scaler.mean)
    assert loaded.metadata == artifact.metadata
    assert [layer.activation.name for layer in loaded.network] == ["relu", "sigmoid"]
