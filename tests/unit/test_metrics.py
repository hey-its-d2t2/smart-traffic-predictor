"""Unit tests for evaluation metrics."""

import numpy as np
import pytest
from traffic_predictor.evaluation.metrics import (
    classification_metrics,
    regression_metrics,
)


class TestRegressionMetrics:
    def test_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0])
        m = regression_metrics(y, y)
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["r2"] == 1.0

    def test_mae_correct(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        m = regression_metrics(y_true, y_pred)
        assert abs(m["mae"] - 10.0) < 0.1

    def test_r2_negative_for_bad_model(self):
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([400.0, 300.0, 200.0, 100.0])
        m = regression_metrics(y_true, y_pred)
        assert m["r2"] < 0

    def test_all_keys_present(self):
        y = np.array([100.0, 200.0])
        m = regression_metrics(y, y)
        for key in ["mae", "rmse", "mape_pct", "r2"]:
            assert key in m


class TestClassificationMetrics:
    def test_perfect_prediction(self):
        y = np.array([0, 1, 2, 3, 0, 1])
        m = classification_metrics(y, y)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0

    def test_accuracy_range(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([1, 1, 2, 2])
        m = classification_metrics(y_true, y_pred)
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["macro_f1"] <= 1.0

    def test_per_class_f1_keys(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        m = classification_metrics(y_true, y_pred)
        for key in ["low", "moderate", "high"]:
            assert key in m["per_class_f1"]
