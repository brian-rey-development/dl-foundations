import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, Request

from foundations import __version__
from foundations.api.schemas import (
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from foundations.api.service import ModelService

DEFAULT_ARTIFACT_PATH = Path("outputs/model.npz")
ARTIFACT_PATH_ENV = "FOUNDATIONS_MODEL_PATH"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

API_DESCRIPTION = """
Serves the neural network trained in `00. foundations`.

The model is a NumPy MLP that classifies 2D points into one of two interleaved half moons.
Inputs are raw coordinates: the standardizer fitted on the training set is applied server side.
"""


def _service(request: Request) -> ModelService:
    return request.app.state.service


Service = Annotated[ModelService, Depends(_service)]


def _artifact_path() -> Path:
    return Path(os.environ.get(ARTIFACT_PATH_ENV, DEFAULT_ARTIFACT_PATH))


def create_app(artifact_path: Path | None = None) -> FastAPI:
    service = ModelService(artifact_path or _artifact_path())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service
        service.load_or_train()
        yield

    app = FastAPI(
        title="Foundations API",
        version=__version__,
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(service: Service) -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=service.is_loaded)

    @app.get("/model", response_model=ModelInfo, tags=["model"])
    def model_info(service: Service) -> ModelInfo:
        return service.info()

    @app.post("/predict", response_model=PredictResponse, tags=["model"])
    def predict(body: PredictRequest, service: Service) -> PredictResponse:
        return PredictResponse(predictions=service.predict(body.points))

    @app.post("/train", response_model=TrainResponse, tags=["model"])
    def train(body: TrainRequest, service: Service) -> TrainResponse:
        return service.train(body)


def serve() -> None:
    uvicorn.run(
        "foundations.api.app:create_app", factory=True, host=DEFAULT_HOST, port=DEFAULT_PORT
    )
