import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_entry.update(record.extra_data)
        return json.dumps(log_entry)


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("mp041_api")
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        log.addHandler(handler)
    log.propagate = False
    return log


logger = _setup_logger()

FEATURE_COLUMNS = [
    "distance_km",
    "wait_min",
    "passengers",
    "hour",
    "luggage_bags",
    "fuel_price_rs",
    "booking",
    "zone",
]

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
model: Any = None


def get_model() -> Any:
    """Load the champion model from disk if not already loaded."""
    global model
    if model is not None:
        return model

    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact not found at {model_path}",
        )
    model = joblib.load(model_path)
    return model


class RideRequest(BaseModel):
    distance_km: float = Field(..., ge=0)
    wait_min: float = Field(..., ge=0, le=50)
    passengers: int = Field(..., ge=1)
    hour: int = Field(..., ge=0, le=23)
    luggage_bags: int = Field(..., ge=0)
    fuel_price_rs: float = Field(..., ge=45)
    booking: str
    zone: str


class PredictionResponse(BaseModel):
    prediction: float


app = FastAPI(title="MP041 Auto-Rickshaw Fare API", version="1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: RideRequest) -> PredictionResponse:
    loaded_model = get_model()

    row = pd.DataFrame([request.model_dump()], columns=FEATURE_COLUMNS)
    prediction_value = float(loaded_model.predict(row)[0])

    logger.info(
        "prediction_served",
        extra={
            "extra_data": {
                "distance_km": request.distance_km,
                "wait_min": request.wait_min,
                "passengers": request.passengers,
                "hour": request.hour,
                "luggage_bags": request.luggage_bags,
                "fuel_price_rs": request.fuel_price_rs,
                "booking": request.booking,
                "zone": request.zone,
                "prediction": round(prediction_value, 2),
            }
        },
    )

    return PredictionResponse(prediction=round(prediction_value, 2))
