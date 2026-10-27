
import pytest
from fastapi.testclient import TestClient

from app import main


VALID_READING = {
    "type": "M",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 20,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Keep test predictions separate from the actual monitoring log.
    monkeypatch.setattr(
        main,
        "PREDICTION_LOG",
        tmp_path / "test_predictions.csv",
    )

    # Load the trained model.
    main.load_model()

    with TestClient(main.app) as test_client:
        yield test_client


def test_health_reports_model_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=VALID_READING)

    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["predicted_failure"] in (0, 1)
    assert body["threshold"] == 0.5
    assert body["model"] == "xgboost"


def test_prediction_is_logged(client, tmp_path):
    response = client.post("/predict", json=VALID_READING)

    assert response.status_code == 200

    log_path = tmp_path / "test_predictions.csv"
    assert log_path.exists()

    contents = log_path.read_text(encoding="utf-8")
    assert "failure_probability" in contents
    assert "predicted_failure" in contents


def test_invalid_machine_type_is_rejected(client):
    reading = {**VALID_READING, "type": "X"}

    response = client.post("/predict", json=reading)

    assert response.status_code == 422


def test_out_of_range_temperature_is_rejected(client):
    reading = {**VALID_READING, "air_temperature": 999}

    response = client.post("/predict", json=reading)

    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    reading = VALID_READING.copy()
    del reading["torque"]

    response = client.post("/predict", json=reading)

    assert response.status_code == 422


def test_metrics_endpoint_returns_prometheus_data(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "predictops_http_requests_total" in response.text
    assert "predictops_predictions_total" in response.text


def test_prediction_respects_configured_threshold(client, monkeypatch):
    monkeypatch.setenv("PREDICTION_THRESHOLD", "0.9")

    response = client.post("/predict", json=VALID_READING)

    assert response.status_code == 200

    body = response.json()
    assert body["threshold"] == 0.9
    assert body["predicted_failure"] == int(
        body["failure_probability"] >= 0.9
    )


def test_invalid_threshold_configuration_returns_error(client, monkeypatch):
    monkeypatch.setenv("PREDICTION_THRESHOLD", "1.0")

    response = client.post("/predict", json=VALID_READING)

    assert response.status_code == 500
    assert "Invalid prediction threshold" in response.json()["detail"]
