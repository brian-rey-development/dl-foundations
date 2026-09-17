from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from foundations.api import create_app
from foundations.api.schemas import TrainRequest
from foundations.api.service import ModelService

FAST_TRAIN = TrainRequest(epochs=5, layer_sizes=[2, 4, 1])


@pytest.fixture
def artifact_path(tmp_path: Path) -> Path:
    path = tmp_path / "model.npz"
    ModelService(path).train(FAST_TRAIN)
    return path


@pytest.fixture
def client(artifact_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(artifact_path)) as test_client:
        yield test_client
