"""Traffic prediction endpoints."""

from __future__ import annotations

import math

import numpy as np
from fastapi import APIRouter, HTTPException, status

from traffic_predictor.api.schemas.prediction import (
    BatchTrafficRequest,
    BatchTrafficResponse,
    CongestionProbabilities,
    ForecastHour,
    ForecastRequest,
    ForecastResponse,
    TrafficPredictionRequest,
    TrafficPredictionResponse,
)
from traffic_predictor.data.dataset import CONGESTION_LEVELS
from traffic_predictor.features.feature_engineering import build_feature_vector, FEATURE_NAMES
from traffic_predictor.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["predictions"])

# Global model registry
_model = None


def set_model(model) -> None:
    global _model
    _model = model


def get_model():
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run `python scripts/train.py` first.",
        )
    return _model


def _estimated_speed(volume: float, speed_limit: float, capacity: float = 1500.0) -> float:
    """Estimate average speed from volume using BPR (Bureau of Public Roads) function."""
    vc = volume / max(capacity, 1.0)
    free_flow = speed_limit * 0.9
    speed = free_flow / (1 + 0.15 * (vc ** 4))
    return max(5.0, round(float(speed), 1))


def _build_request_features(req: TrafficPredictionRequest) -> np.ndarray:
    vec = build_feature_vector(
        hour=req.hour,
        day_of_week=req.day_of_week,
        month=req.month,
        intersection_id=req.intersection_id,
        intersection_type=req.intersection_type,
        weather_code=req.weather_code,
        temperature_c=req.temperature_c,
        precipitation_mm=req.precipitation_mm,
        visibility_km=req.visibility_km,
        speed_limit_kmh=req.speed_limit_kmh,
        num_lanes=req.num_lanes,
        volume_lag1=req.volume_lag1,
        volume_lag2=req.volume_lag2,
        volume_lag3=req.volume_lag3,
        speed_lag1=req.speed_lag1,
        speed_lag2=req.speed_lag2,
        speed_lag3=req.speed_lag3,
        volume_rolling_mean_3h=req.volume_rolling_mean_3h,
        volume_rolling_mean_6h=req.volume_rolling_mean_6h,
        volume_rolling_mean_12h=req.volume_rolling_mean_12h,
        speed_rolling_mean_3h=req.speed_rolling_mean_3h,
        is_holiday=req.is_holiday,
    )
    return vec.reshape(1, -1)


def _make_response(req: TrafficPredictionRequest, vol: float, cong: int, proba: np.ndarray, model_name: str) -> TrafficPredictionResponse:
    capacity_map = {0: 400, 1: 1200, 2: 2500, 3: 1800, 4: 900}
    capacity = capacity_map.get(req.intersection_type, 1200) * req.num_lanes
    speed = _estimated_speed(vol, req.speed_limit_kmh, capacity)

    # Pad/clip proba to 4 classes
    p = np.zeros(4)
    p[:len(proba)] = proba
    conf = float(p[cong])

    return TrafficPredictionResponse(
        intersection_id=req.intersection_id,
        hour=req.hour,
        day_of_week=req.day_of_week,
        predicted_volume=round(float(vol), 1),
        predicted_speed_kmh=speed,
        congestion_level=int(cong),
        congestion_label=CONGESTION_LEVELS.get(int(cong), "unknown"),
        congestion_probabilities=CongestionProbabilities(
            low=round(float(p[0]), 4),
            moderate=round(float(p[1]), 4),
            high=round(float(p[2]), 4),
            severe=round(float(p[3]), 4),
        ),
        confidence=round(conf, 4),
        model_name=model_name,
    )


@router.post("/predict", response_model=TrafficPredictionResponse, summary="Predict traffic for one time slot")
async def predict_single(req: TrafficPredictionRequest):
    model = get_model()
    X = _build_request_features(req)
    volumes, congestions = model.predict(X)
    probas = model.predict_proba_congestion(X)
    return _make_response(req, float(volumes[0]), int(congestions[0]), probas[0], model.model_name)


@router.post("/predict/batch", response_model=BatchTrafficResponse, summary="Batch predict (up to 100 slots)")
async def predict_batch(req: BatchTrafficRequest):
    model = get_model()
    X = np.vstack([_build_request_features(r) for r in req.requests])
    volumes, congestions = model.predict(X)
    probas = model.predict_proba_congestion(X)
    results = [
        _make_response(r, float(v), int(c), p, model.model_name)
        for r, v, c, p in zip(req.requests, volumes, congestions, probas)
    ]
    return BatchTrafficResponse(results=results, total=len(results), model_name=model.model_name)


@router.post("/forecast", response_model=ForecastResponse, summary="Forecast traffic for the next N hours")
async def forecast(req: ForecastRequest):
    """Generate a multi-hour traffic forecast for a given intersection and day."""
    model = get_model()
    forecast_hours = []
    prev_volume = 500.0
    prev_speed = req.speed_limit_kmh * 0.8

    vol_history = [prev_volume] * 12
    spd_history = [prev_speed] * 6

    for offset in range(req.n_hours):
        h = (req.start_hour + offset) % 24
        single_req = TrafficPredictionRequest(
            hour=h,
            day_of_week=req.day_of_week,
            month=req.month,
            intersection_id=req.intersection_id,
            intersection_type=req.intersection_type,
            weather_code=req.weather_code,
            temperature_c=req.temperature_c,
            precipitation_mm=0.0,
            visibility_km=10.0,
            speed_limit_kmh=req.speed_limit_kmh,
            num_lanes=req.num_lanes,
            is_holiday=req.is_holiday,
            volume_lag1=vol_history[-1],
            volume_lag2=vol_history[-2],
            volume_lag3=vol_history[-3],
            speed_lag1=spd_history[-1],
            speed_lag2=spd_history[-2],
            speed_lag3=spd_history[-3],
            volume_rolling_mean_3h=float(np.mean(vol_history[-3:])),
            volume_rolling_mean_6h=float(np.mean(vol_history[-6:])),
            volume_rolling_mean_12h=float(np.mean(vol_history)),
            speed_rolling_mean_3h=float(np.mean(spd_history[-3:])),
        )
        X = _build_request_features(single_req)
        volumes, congestions = model.predict(X)
        probas = model.predict_proba_congestion(X)
        vol = float(volumes[0])
        cong = int(congestions[0])
        p = np.zeros(4)
        p[:len(probas[0])] = probas[0]
        conf = float(p[cong])

        capacity_map = {0: 400, 1: 1200, 2: 2500, 3: 1800, 4: 900}
        capacity = capacity_map.get(req.intersection_type, 1200) * req.num_lanes
        spd = _estimated_speed(vol, req.speed_limit_kmh, capacity)

        forecast_hours.append(ForecastHour(
            hour=h,
            predicted_volume=round(vol, 1),
            congestion_label=CONGESTION_LEVELS.get(cong, "unknown"),
            congestion_level=cong,
            confidence=round(conf, 4),
        ))

        vol_history.append(vol)
        if len(vol_history) > 12:
            vol_history.pop(0)
        spd_history.append(spd)
        if len(spd_history) > 6:
            spd_history.pop(0)

    return ForecastResponse(
        intersection_id=req.intersection_id,
        start_hour=req.start_hour,
        n_hours=req.n_hours,
        day_of_week=req.day_of_week,
        forecast=forecast_hours,
        model_name=model.model_name,
    )
