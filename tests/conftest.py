"""Shared pytest fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from traffic_predictor.data.dataset import TrafficDataset
from traffic_predictor.features.feature_engineering import FEATURE_NAMES, build_feature_vector
from traffic_predictor.models.traditional.traffic_classifier import TrafficPredictor


@pytest.fixture(scope="session")
def small_dataset():
    return TrafficDataset.generate_synthetic(n_records=500, n_intersections=5, random_seed=42)


@pytest.fixture(scope="session")
def trained_model(small_dataset):
    X, y_vol, y_cong, feat_names = small_dataset.to_feature_matrix()
    model = TrafficPredictor(model_type="random_forest", random_seed=42)
    model.fit(X, y_vol, y_cong, feature_names=feat_names)
    return model


@pytest.fixture
def sample_feature_vector():
    return build_feature_vector(
        hour=8, day_of_week=1, month=6,
        intersection_id=0, intersection_type=1,
        weather_code=0, temperature_c=22.0,
        precipitation_mm=0.0, visibility_km=10.0,
        speed_limit_kmh=50.0, num_lanes=2,
    )
