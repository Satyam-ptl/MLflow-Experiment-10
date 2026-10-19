from fastapi.testclient import TestClient
import app

client = TestClient(app.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_name"] == "IrisRandomForest"


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "IrisRandomForest"
    assert data["model_version"] == "1"


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert response.status_code == 200
    assert response.json()["predicted_class"] == "setosa"


def test_invalid_prediction_input():
    response = client.post(
        "/predict",
        json={
            "sepal_length": -1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert response.status_code == 422


def test_prometheus_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "iris_predictions_total" in response.text
    assert "iris_prediction_errors_total" in response.text
