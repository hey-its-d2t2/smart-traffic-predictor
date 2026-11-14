"""Request and response schemas for the traffic prediction API."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class TrafficPredictionRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0–23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month (1–12)")
    intersection_id: int = Field(..., ge=0, description="Intersection identifier")
    intersection_type: int = Field(1, ge=0, le=4, description="0=residential,1=arterial,2=highway,3=downtown,4=commercial")
    weather_code: int = Field(0, ge=0, le=5, description="0=clear,1=cloudy,2=rain,3=heavy_rain,4=snow,5=fog")
    temperature_c: float = Field(20.0, description="Temperature in Celsius")
    precipitation_mm: float = Field(0.0, ge=0.0, description="Precipitation in mm")
    visibility_km: float = Field(10.0, gt=0.0, description="Visibility in km")
    speed_limit_kmh: float = Field(50.0, gt=0.0, description="Speed limit in km/h")
    num_lanes: int = Field(2, ge=1, le=10, description="Number of lanes")
    is_holiday: int = Field(0, ge=0, le=1, description="1 if public holiday")
    volume_lag1: float = Field(0.0, ge=0.0, description="Traffic volume 1 hour ago")
    volume_lag2: float = Field(0.0, ge=0.0, description="Traffic volume 2 hours ago")
    volume_lag3: float = Field(0.0, ge=0.0, description="Traffic volume 3 hours ago")
    speed_lag1: float = Field(0.0, ge=0.0, description="Avg speed 1 hour ago (km/h)")
    speed_lag2: float = Field(0.0, ge=0.0, description="Avg speed 2 hours ago (km/h)")
    speed_lag3: float = Field(0.0, ge=0.0, description="Avg speed 3 hours ago (km/h)")
    volume_rolling_mean_3h: float = Field(0.0, ge=0.0)
    volume_rolling_mean_6h: float = Field(0.0, ge=0.0)
    volume_rolling_mean_12h: float = Field(0.0, ge=0.0)
    speed_rolling_mean_3h: float = Field(0.0, ge=0.0)


class CongestionProbabilities(BaseModel):
    low: float
    moderate: float
    high: float
    severe: float


class TrafficPredictionResponse(BaseModel):
    intersection_id: int
    hour: int
    day_of_week: int
    predicted_volume: float = Field(..., description="Predicted traffic volume (vehicles/hour)")
    predicted_speed_kmh: float = Field(..., description="Estimated average speed (km/h)")
    congestion_level: int = Field(..., description="0=low, 1=moderate, 2=high, 3=severe")
    congestion_label: str
    congestion_probabilities: CongestionProbabilities
    confidence: float = Field(..., description="Confidence in congestion classification")
    model_name: str
    disclaimer: str = "This prediction is for research/planning purposes only. Actual conditions may vary."


class BatchTrafficRequest(BaseModel):
    requests: List[TrafficPredictionRequest] = Field(..., max_length=100)


class BatchTrafficResponse(BaseModel):
    results: List[TrafficPredictionResponse]
    total: int
    model_name: str


class ForecastRequest(BaseModel):
    intersection_id: int = Field(..., ge=0)
    intersection_type: int = Field(1, ge=0, le=4)
    start_hour: int = Field(..., ge=0, le=23, description="Starting hour of forecast")
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    n_hours: int = Field(6, ge=1, le=24, description="Number of hours to forecast")
    weather_code: int = Field(0, ge=0, le=5)
    temperature_c: float = Field(20.0)
    speed_limit_kmh: float = Field(50.0, gt=0.0)
    num_lanes: int = Field(2, ge=1, le=10)
    is_holiday: int = Field(0, ge=0, le=1)


class ForecastHour(BaseModel):
    hour: int
    predicted_volume: float
    congestion_label: str
    congestion_level: int
    confidence: float


class ForecastResponse(BaseModel):
    intersection_id: int
    start_hour: int
    n_hours: int
    day_of_week: int
    forecast: List[ForecastHour]
    model_name: str
