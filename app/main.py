
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(PROJECT_ROOT / "models" / "best_model.joblib"),
    )
)

PREDICTION_LOG = Path(
    os.getenv(
        "PREDICTION_LOG",
        str(PROJECT_ROOT / "reports" / "drift" / "predictions.csv"),
    )
)

FEATURE_COLUMNS = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

VALID_TYPES = {"L", "M", "H"}

app = FastAPI(
    title="PredictOps API",
    description="Machine failure prediction and service monitoring",
    version="1.0.0",
)

REQUESTS = Counter(
    "predictops_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

LATENCY = Histogram(
    "predictops_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

PREDICTIONS = Counter(
    "predictops_predictions_total",
    "Number of successful predictions",
    ["prediction"],
)

ERRORS = Counter(
    "predictops_prediction_errors_total",
    "Number of prediction errors",
)

_log_lock = Lock()
_model = None


class MachineReading(BaseModel):
    type: str = Field(description="Machine product type: L, M, or H")
    air_temperature: float = Field(ge=250, le=350)
    process_temperature: float = Field(ge=250, le=350)
    rotational_speed: float = Field(gt=0, le=10000)
    torque: float = Field(ge=0, le=1000)
    tool_wear: float = Field(ge=0, le=1000)


class PredictionResponse(BaseModel):
    failure_probability: float
    predicted_failure: int
    threshold: float
    model: str


def load_model():
    global _model

    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")

    _model = joblib.load(MODEL_PATH)

    if not hasattr(_model, "predict_proba"):
        raise RuntimeError("Loaded model does not support predict_proba")


def append_prediction_log(reading, probability, predicted_failure):
    """Append model inputs and predictions for later drift analysis."""
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **reading,
        "failure_probability": probability,
        "predicted_failure": predicted_failure,
    }

    with _log_lock:
        file_exists = PREDICTION_LOG.exists()

        with PREDICTION_LOG.open(
            "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)


@app.on_event("startup")
def startup():
    load_model()


@app.middleware("http")
async def collect_http_metrics(request: Request, call_next):
    start = time.perf_counter()
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        endpoint = request.url.path

        # Avoid counting Prometheus scrapes as application traffic.
        if endpoint != "/metrics":
            REQUESTS.labels(
                request.method, endpoint, str(status)
            ).inc()

            LATENCY.labels(
                request.method, endpoint
            ).observe(time.perf_counter() - start)


@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "unavailable",
        "model_loaded": _model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: MachineReading):
    if _model is None:
        raise HTTPException(
            status_code=503, detail="Model is not loaded"
        )

    if reading.type not in VALID_TYPES:
        raise HTTPException(
            status_code=422,
            detail="type must be L, M, or H",
        )

    threshold = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

    if not 0 < threshold < 1:
        raise HTTPException(
            status_code=500,
            detail="Invalid prediction threshold configuration",
        )

    features = pd.DataFrame(
        [{
            "Type": reading.type,
            "Air temperature": reading.air_temperature,
            "Process temperature": reading.process_temperature,
            "Rotational speed": reading.rotational_speed,
            "Torque": reading.torque,
            "Tool wear": reading.tool_wear,
        }],
        columns=FEATURE_COLUMNS,
    )

    try:
        probability = float(_model.predict_proba(features)[0, 1])
        predicted_failure = int(probability >= threshold)

        append_prediction_log(
            features.iloc[0].to_dict(),
            probability,
            predicted_failure,
        )

        PREDICTIONS.labels(str(predicted_failure)).inc()

        return PredictionResponse(
            failure_probability=probability,
            predicted_failure=predicted_failure,
            threshold=threshold,
            model="xgboost",
        )

    except HTTPException:
        raise
    except Exception as exc:
        ERRORS.inc()
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
