"""Feature engineering for traffic prediction."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from traffic_predictor.utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_NAMES = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "is_morning_rush", "is_evening_rush", "is_night",
    "intersection_id", "intersection_type",
    "weather_code", "temperature_c", "precipitation_mm", "visibility_km",
    "speed_limit_kmh", "num_lanes",
    "volume_lag1", "volume_lag2", "volume_lag3",
    "speed_lag1", "speed_lag2", "speed_lag3",
    "volume_rolling_mean_3h", "volume_rolling_mean_6h", "volume_rolling_mean_12h",
    "speed_rolling_mean_3h",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]


def build_feature_vector(
    hour: int,
    day_of_week: int,
    month: int,
    intersection_id: int,
    intersection_type: int,
    weather_code: int,
    temperature_c: float,
    precipitation_mm: float,
    visibility_km: float,
    speed_limit_kmh: float,
    num_lanes: int,
    volume_lag1: float = 0.0,
    volume_lag2: float = 0.0,
    volume_lag3: float = 0.0,
    speed_lag1: float = 0.0,
    speed_lag2: float = 0.0,
    speed_lag3: float = 0.0,
    volume_rolling_mean_3h: float = 0.0,
    volume_rolling_mean_6h: float = 0.0,
    volume_rolling_mean_12h: float = 0.0,
    speed_rolling_mean_3h: float = 0.0,
    is_holiday: int = 0,
) -> np.ndarray:
    """Build a single feature vector from raw inputs."""
    is_weekend = int(day_of_week >= 5)
    is_morning_rush = int(7 <= hour <= 9 and not is_weekend)
    is_evening_rush = int(16 <= hour <= 19 and not is_weekend)
    is_night = int(hour < 6 or hour >= 22)

    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    day_sin = math.sin(2 * math.pi * day_of_week / 7)
    day_cos = math.cos(2 * math.pi * day_of_week / 7)

    return np.array([
        hour, day_of_week, month, is_weekend, is_holiday,
        is_morning_rush, is_evening_rush, is_night,
        intersection_id, intersection_type,
        weather_code, temperature_c, precipitation_mm, visibility_km,
        speed_limit_kmh, num_lanes,
        volume_lag1, volume_lag2, volume_lag3,
        speed_lag1, speed_lag2, speed_lag3,
        volume_rolling_mean_3h, volume_rolling_mean_6h, volume_rolling_mean_12h,
        speed_rolling_mean_3h,
        hour_sin, hour_cos, day_sin, day_cos,
    ], dtype=np.float32)


def volume_to_congestion(volume: float, capacity: float) -> int:
    """Map volume/capacity ratio to congestion level (0=low..3=severe)."""
    ratio = volume / max(capacity, 1.0)
    if ratio < 0.4:
        return 0
    if ratio < 0.65:
        return 1
    if ratio < 0.85:
        return 2
    return 3


class FeatureScaler:
    """Min-max scaler that works on numpy arrays (no pandas dependency)."""

    def __init__(self) -> None:
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        self._min = X.min(axis=0)
        self._max = X.max(axis=0)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        rng = self._max - self._min
        rng[rng == 0] = 1.0  # avoid divide-by-zero for constant features
        return (X - self._min) / rng

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        rng = self._max - self._min
        rng[rng == 0] = 1.0
        return X * rng + self._min
