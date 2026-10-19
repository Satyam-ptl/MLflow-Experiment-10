import os
import time

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = "IrisRandomForest"
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
MODEL_URI = os.getenv(
    "MODEL_URI",
    f"models:/{MODEL_NAME}/{MODEL_VERSION}",
)

# Only registry-based loading needs the tracking database.
if MODEL_URI.startswith("models:/"):
    mlflow.set_tracking_uri(TRACKING_URI)

model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI(
    title="Iris ML Prediction API",
    description="Iris predictions served using an MLflow model.",
    version="1.0.0",
)

PREDICTIONS_TOTAL = Counter(
    "iris_predictions_total",
    "Total Iris prediction requests",
)
PREDICTION_ERRORS_TOTAL = Counter(
    "iris_prediction_errors_total",
    "Total failed Iris prediction requests",
)
PREDICTION_LATENCY = Histogram(
    "iris_prediction_latency_seconds",
    "Time spent generating Iris predictions",
)

app.mount("/metrics", make_asgi_app())


class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)


IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


@app.get("/")
def root():
    return {
        "message": "Iris ML Prediction API is running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_uri": MODEL_URI,
        "tracking_uri": TRACKING_URI if MODEL_URI.startswith("models:/") else None,
    }


@app.post("/predict")
def predict(data: IrisInput):
    started = time.perf_counter()
    try:
        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
        ]]
        prediction = int(model.predict(features)[0])

        if prediction not in range(len(IRIS_CLASSES)):
            raise ValueError("Unexpected model class")

        PREDICTIONS_TOTAL.inc()
        return {
            "prediction_id": prediction,
            "predicted_class": IRIS_CLASSES[prediction],
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
        }
    except Exception as exc:
        PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the application logs.",
        ) from exc
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - started)
