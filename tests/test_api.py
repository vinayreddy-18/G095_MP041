import pandas as pd
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import app.main as api_module


def _build_dummy_model() -> Pipeline:
    df = pd.DataFrame(
        [
            {
                "distance_km": 2.0,
                "wait_min": 5.0,
                "passengers": 1,
                "hour": 9,
                "luggage_bags": 0,
                "fuel_price_rs": 100.0,
                "booking": "app",
                "zone": "Market",
                "fare_rs": 45.0,
            },
            {
                "distance_km": 5.0,
                "wait_min": 10.0,
                "passengers": 2,
                "hour": 14,
                "luggage_bags": 1,
                "fuel_price_rs": 102.0,
                "booking": "street",
                "zone": "Station Area",
                "fare_rs": 90.0,
            },
            {
                "distance_km": 8.0,
                "wait_min": 15.0,
                "passengers": 3,
                "hour": 20,
                "luggage_bags": 2,
                "fuel_price_rs": 104.0,
                "booking": "stand",
                "zone": "Outskirts",
                "fare_rs": 135.0,
            },
        ]
    )
    features = api_module.FEATURE_COLUMNS
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["booking", "zone"])
        ],
        remainder="passthrough",
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=10, random_state=42)),
        ]
    )
    pipeline.fit(df[features], df["fare_rs"])
    return pipeline


def test_health_endpoint():
    client = TestClient(api_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_request(monkeypatch):
    monkeypatch.setattr(api_module, "model", _build_dummy_model())
    client = TestClient(api_module.app)
    payload = {
        "distance_km": 4.5,
        "wait_min": 8.0,
        "passengers": 2,
        "hour": 11,
        "luggage_bags": 1,
        "fuel_price_rs": 101.5,
        "booking": "app",
        "zone": "Market",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert data["prediction"] > 0


def test_predict_rejects_wait_min_above_50(monkeypatch):
    monkeypatch.setattr(api_module, "model", _build_dummy_model())
    client = TestClient(api_module.app)
    payload = {
        "distance_km": 4.5,
        "wait_min": 51.0,
        "passengers": 2,
        "hour": 11,
        "luggage_bags": 1,
        "fuel_price_rs": 101.5,
        "booking": "app",
        "zone": "Market",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_fuel_price_below_45(monkeypatch):
    monkeypatch.setattr(api_module, "model", _build_dummy_model())
    client = TestClient(api_module.app)
    payload = {
        "distance_km": 4.5,
        "wait_min": 10.0,
        "passengers": 2,
        "hour": 11,
        "luggage_bags": 1,
        "fuel_price_rs": 44.9,
        "booking": "app",
        "zone": "Market",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
