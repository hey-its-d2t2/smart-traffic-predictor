"""Unit tests for feature engineering."""

import math
import numpy as np
import pytest
from traffic_predictor.features.feature_engineering import (
    FEATURE_NAMES, FeatureScaler, build_feature_vector, volume_to_congestion,
)


class TestBuildFeatureVector:
    def test_returns_numpy_array(self, sample_feature_vector):
        assert isinstance(sample_feature_vector, np.ndarray)

    def test_correct_length(self, sample_feature_vector):
        assert len(sample_feature_vector) == len(FEATURE_NAMES)

    def test_hour_encoded(self):
        vec = build_feature_vector(
            hour=0, day_of_week=0, month=1,
            intersection_id=0, intersection_type=0,
            weather_code=0, temperature_c=10.0,
            precipitation_mm=0.0, visibility_km=10.0,
            speed_limit_kmh=50.0, num_lanes=2,
        )
        assert vec[FEATURE_NAMES.index("hour")] == 0
        assert abs(vec[FEATURE_NAMES.index("hour_sin")] - 0.0) < 1e-5
        assert abs(vec[FEATURE_NAMES.index("hour_cos")] - 1.0) < 1e-5

    def test_weekend_flag(self):
        vec = build_feature_vector(
            hour=10, day_of_week=6, month=6,
            intersection_id=0, intersection_type=1,
            weather_code=0, temperature_c=20.0,
            precipitation_mm=0.0, visibility_km=10.0,
            speed_limit_kmh=50.0, num_lanes=2,
        )
        assert vec[FEATURE_NAMES.index("is_weekend")] == 1

    def test_rush_hour_flag(self):
        # Morning rush: hour=8, weekday
        vec = build_feature_vector(
            hour=8, day_of_week=1, month=6,
            intersection_id=0, intersection_type=1,
            weather_code=0, temperature_c=20.0,
            precipitation_mm=0.0, visibility_km=10.0,
            speed_limit_kmh=50.0, num_lanes=2,
        )
        assert vec[FEATURE_NAMES.index("is_morning_rush")] == 1
        assert vec[FEATURE_NAMES.index("is_evening_rush")] == 0

    def test_night_flag(self):
        vec = build_feature_vector(
            hour=2, day_of_week=0, month=1,
            intersection_id=0, intersection_type=0,
            weather_code=0, temperature_c=5.0,
            precipitation_mm=0.0, visibility_km=10.0,
            speed_limit_kmh=50.0, num_lanes=2,
        )
        assert vec[FEATURE_NAMES.index("is_night")] == 1


class TestVolumeToCongestion:
    def test_low(self):
        assert volume_to_congestion(100, 500) == 0

    def test_moderate(self):
        assert volume_to_congestion(300, 500) == 1

    def test_high(self):
        assert volume_to_congestion(400, 500) == 2

    def test_severe(self):
        assert volume_to_congestion(480, 500) == 3

    def test_zero_capacity_no_crash(self):
        result = volume_to_congestion(500, 0)
        assert result in {0, 1, 2, 3}


class TestFeatureScaler:
    def test_fit_transform_range(self):
        X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]], dtype=np.float32)
        scaler = FeatureScaler()
        X_scaled = scaler.fit_transform(X)
        assert X_scaled[:, 0].min() >= 0.0
        assert X_scaled[:, 0].max() <= 1.0

    def test_transform_before_fit_raises(self):
        scaler = FeatureScaler()
        with pytest.raises(RuntimeError):
            scaler.transform(np.ones((3, 2)))

    def test_inverse_transform(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        scaler = FeatureScaler()
        X_scaled = scaler.fit_transform(X)
        X_back = scaler.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X, X_back, decimal=4)

    def test_constant_feature_no_crash(self):
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]], dtype=np.float32)
        scaler = FeatureScaler()
        X_scaled = scaler.fit_transform(X)
        assert not np.isnan(X_scaled).any()
