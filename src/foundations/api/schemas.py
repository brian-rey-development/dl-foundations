from pydantic import BaseModel, Field

MAX_POINTS_PER_REQUEST = 10_000
MAX_EPOCHS = 5_000
MAX_HISTORY_POINTS = 50


class Point(BaseModel):
    x1: float
    x2: float


class PredictRequest(BaseModel):
    points: list[Point] = Field(min_length=1, max_length=MAX_POINTS_PER_REQUEST)


class Prediction(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)
    label: int = Field(ge=0, le=1)


class PredictResponse(BaseModel):
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    layer_sizes: list[int]
    activations: list[str]
    n_parameters: int
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    test_loss: float
    test_accuracy: float
    trained_at: str


class TrainRequest(BaseModel):
    seed: int = 42
    epochs: int = Field(default=200, ge=1, le=MAX_EPOCHS)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.05, gt=0.0)
    layer_sizes: list[int] = Field(default=[2, 16, 16, 1], min_length=2)


class EpochPoint(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float


class TrainResponse(BaseModel):
    model: ModelInfo
    history: list[EpochPoint]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
