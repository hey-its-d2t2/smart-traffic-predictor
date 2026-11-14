"""Integration tests for the FastAPI endpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from traffic_predictor.api.main import create_app
from traffic_predictor.api.routes import predict as predict_router
from traffic_predictor.data.dataset import TrafficDataset
from traffic_predictor.models.traditional.traffic_classifier import TrafficPredictor


@pytest.fixture(scope="module")
def client():
    """Create a test client with a pre-loaded model."""
    ds = TrafficDataset.generate_synthetic(n_records=300, n_intersections=5, random_seed=42)
    X, y_vol, y_cong, feat_names = ds.to_feature_matrix()
    model = TrafficPredictor(model_type="random_forest", random_seed=42)
    model.fit(X, y_vol, y_cong, feature_names=feat_names)
    predict_router.set_model(model)

    app = create_app()
    return TestClient(app)


SAMPLE_REQUEST = {
    "hour": 8, "day_of_week": 1, "month": 6,
    "intersection_id": 0, "intersection_type": 1,
    "weather_code": 0, "temperature_c": 22.0,
    "precipitation_mm": 0.0, "visibility_km": 10.0,
    "speed_limit_kmh": 50.0, "num_lanes": 2,
    "is_holiday": 0,
    "volume_lag1": 800.0, "volume_lag2": 750.0, "volume_lag3": 700.0,
    "speed_lag1": 35.0, "speed_lag2": 38.0, "speed_lag3": 40.0,
    "volume_rolling_mean_3h": 750.0, "volume_rolling_mean_6h": 720.0,
    "volume_rolling_mean_12h": 600.0, "speed_rolling_mean_3h": 37.0,
}


class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_info_returns_name(self, client):
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "Smart Traffic Predictor" in data["name"]

    def test_ready_returns_200(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        resp = client.post("/api/v1/predict", json=SAMPLE_REQUEST)
        assert resp.status_code == 200

    def test_predict_response_has_volume(self, client):
        resp = client.post("/api/v1/predict", json=SAMPLE_REQUEST)
        data = resp.json()
        assert "predicted_volume" in data
        assert data["predicted_volume"] > 0

    def test_predict_response_has_congestion(self, client):
        resp = client.post("/api/v1/predict", json=SAMPLE_REQUEST)
        data = resp.json()
        assert "congestion_level" in data
        assert data["congestion_level"] in {0, 1, 2, 3}
        assert "congestion_label" in data

    def test_predict_response_has_probabilities(self, client):
        resp = client.post("/api/v1/predict", json=SAMPLE_REQUEST)
        data = resp.json()
        p = data["congestion_probabilities"]
        total = p["low"] + p["moderate"] + p["high"] + p["severe"]
        assert abs(total - 1.0) < 0.01

    def test_predict_invalid_hour_returns_422(self, client):
        bad = {**SAMPLE_REQUEST, "hour": 25}
        resp = client.post("/api/v1/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_invalid_month_returns_422(self, client):
        bad = {**SAMPLE_REQUEST, "month": 13}
        resp = client.post("/api/v1/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_night_time(self, client):
        night = {**SAMPLE_REQUEST, "hour": 2}
        resp = client.post("/api/v1/predict", json=night)
        assert resp.status_code == 200
        data = resp.json()
        assert data["congestion_level"] in {0, 1, 2, 3}


class TestBatchEndpoint:
    def test_batch_returns_200(self, client):
        payload = {"requests": [SAMPLE_REQUEST, {**SAMPLE_REQUEST, "hour": 17}]}
        resp = client.post("/api/v1/predict/batch", json=payload)
        assert resp.status_code == 200

    def test_batch_result_count(self, client):
        payload = {"requests": [SAMPLE_REQUEST, {**SAMPLE_REQUEST, "hour": 12}]}
        resp = client.post("/api/v1/predict/batch", json=payload)
        data = resp.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_batch_too_many_returns_422(self, client):
        payload = {"requests": [SAMPLE_REQUEST] * 101}
        resp = client.post("/api/v1/predict/batch", json=payload)
        assert resp.status_code == 422


class TestForecastEndpoint:
    def test_forecast_returns_200(self, client):
        payload = {
            "intersection_id": 0, "intersection_type": 1,
            "start_hour": 7, "day_of_week": 1, "month": 6, "n_hours": 4,
            "weather_code": 0, "temperature_c": 20.0,
            "speed_limit_kmh": 50.0, "num_lanes": 2, "is_holiday": 0,
        }
        resp = client.post("/api/v1/forecast", json=payload)
        assert resp.status_code == 200

    def test_forecast_correct_hours(self, client):
        payload = {
            "intersection_id": 0, "intersection_type": 1,
            "start_hour": 7, "day_of_week": 1, "month": 6, "n_hours": 3,
            "weather_code": 0, "temperature_c": 20.0,
            "speed_limit_kmh": 50.0, "num_lanes": 2, "is_holiday": 0,
        }
        resp = client.post("/api/v1/forecast", json=payload)
        data = resp.json()
        assert len(data["forecast"]) == 3

    def test_forecast_hours_are_sequential(self, client):
        payload = {
            "intersection_id": 0, "intersection_type": 1,
            "start_hour": 7, "day_of_week": 1, "month": 6, "n_hours": 4,
            "weather_code": 0, "temperature_c": 20.0,
            "speed_limit_kmh": 50.0, "num_lanes": 2, "is_holiday": 0,
        }
        resp = client.post("/api/v1/forecast", json=payload)
        hours = [h["hour"] for h in resp.json()["forecast"]]
        expected = [(7 + i) % 24 for i in range(4)]
        assert hours == expected
