from fastapi.testclient import TestClient

from foundations.api.schemas import MAX_POINTS_PER_REQUEST


def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_matches_trained_artifact(client: TestClient) -> None:
    body = client.get("/model").json()
    assert body["layer_sizes"] == [2, 4, 1]
    assert body["activations"] == ["relu", "sigmoid"]
    assert body["n_parameters"] == (2 * 4 + 4) + (4 * 1 + 1)
    assert 0.0 <= body["test_accuracy"] <= 1.0


def test_predict_returns_one_prediction_per_point(client: TestClient) -> None:
    points = [{"x1": 0.0, "x2": 1.0}, {"x1": 1.0, "x2": -0.5}, {"x1": 2.0, "x2": 0.3}]
    response = client.post("/predict", json={"points": points})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 3
    for p in predictions:
        assert 0.0 <= p["probability"] <= 1.0
        assert p["label"] == int(p["probability"] >= 0.5)


def test_predict_rejects_empty_and_oversized_payloads(client: TestClient) -> None:
    assert client.post("/predict", json={"points": []}).status_code == 422
    too_many = [{"x1": 0.0, "x2": 0.0}] * (MAX_POINTS_PER_REQUEST + 1)
    assert client.post("/predict", json={"points": too_many}).status_code == 422


def test_predict_rejects_malformed_point(client: TestClient) -> None:
    response = client.post("/predict", json={"points": [{"x1": "north", "x2": 0.0}]})
    assert response.status_code == 422


def test_train_replaces_model_and_returns_history(client: TestClient) -> None:
    body = {"epochs": 3, "layer_sizes": [2, 8, 1], "seed": 7}
    response = client.post("/train", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"]["layer_sizes"] == [2, 8, 1]
    assert payload["model"]["seed"] == 7
    assert [h["epoch"] for h in payload["history"]] == [1, 2, 3]
    assert client.get("/model").json()["layer_sizes"] == [2, 8, 1]


def test_train_validates_hyperparameters(client: TestClient) -> None:
    assert client.post("/train", json={"epochs": 0}).status_code == 422
    assert client.post("/train", json={"learning_rate": -1}).status_code == 422
