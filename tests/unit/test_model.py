"""Unit tests for TrafficPredictor model."""

import numpy as np
import pytest
from traffic_predictor.models.traditional.traffic_classifier import TrafficPredictor


class TestTrafficPredictor:
    def test_predict_returns_correct_shapes(self, trained_model, small_dataset):
        X, y_vol, y_cong, _ = small_dataset.to_feature_matrix()
        vols, congs = trained_model.predict(X[:10])
        assert vols.shape == (10,)
        assert congs.shape == (10,)

    def test_predicted_volumes_positive(self, trained_model, small_dataset):
        X, _, _, _ = small_dataset.to_feature_matrix()
        vols, _ = trained_model.predict(X[:20])
        assert (vols > 0).all()

    def test_congestion_labels_valid(self, trained_model, small_dataset):
        X, _, _, _ = small_dataset.to_feature_matrix()
        _, congs = trained_model.predict(X[:20])
        assert set(congs.tolist()).issubset({0, 1, 2, 3})

    def test_predict_proba_sums_to_one(self, trained_model, small_dataset):
        X, _, _, _ = small_dataset.to_feature_matrix()
        probas = trained_model.predict_proba_congestion(X[:5])
        sums = probas.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(5), decimal=4)

    def test_predict_before_fit_raises(self):
        model = TrafficPredictor.__new__(TrafficPredictor)
        model._is_fitted = False
        model.model_name = "test"
        with pytest.raises(RuntimeError):
            model._check_fitted()

    def test_save_and_load(self, trained_model, small_dataset, tmp_path):
        path = tmp_path / "model.pkl"
        trained_model.save(path)
        loaded = TrafficPredictor.load(path)
        X, _, _, _ = small_dataset.to_feature_matrix()
        vols1, congs1 = trained_model.predict(X[:5])
        vols2, congs2 = loaded.predict(X[:5])
        np.testing.assert_array_almost_equal(vols1, vols2, decimal=2)
        np.testing.assert_array_equal(congs1, congs2)

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError):
            TrafficPredictor(model_type="nonexistent")

    def test_feature_importance_returns_list(self, trained_model):
        importance = trained_model.get_feature_importance(top_n=5)
        assert isinstance(importance, list)
        assert len(importance) <= 5
        for name, score in importance:
            assert isinstance(name, str)
            assert isinstance(score, float)
