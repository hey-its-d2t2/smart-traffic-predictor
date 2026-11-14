"""Unit tests for TrafficDataset."""

import pytest
from traffic_predictor.data.dataset import TrafficDataset, CONGESTION_LEVELS


class TestTrafficDataset:
    def test_generate_synthetic_count(self, small_dataset):
        assert len(small_dataset) >= 490  # allow minor rounding

    def test_congestion_distribution_keys(self, small_dataset):
        dist = small_dataset.congestion_distribution()
        for level in CONGESTION_LEVELS.values():
            assert level in dist, f"Missing congestion level: {level}"

    def test_split_total(self, small_dataset):
        train, val, test = small_dataset.split()
        total = len(train) + len(val) + len(test)
        assert abs(total - len(small_dataset)) <= 2

    def test_split_non_empty(self, small_dataset):
        train, val, test = small_dataset.split()
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_feature_matrix_shape(self, small_dataset):
        X, y_vol, y_cong, feat_names = small_dataset.to_feature_matrix()
        assert X.shape[0] == len(small_dataset)
        assert X.shape[1] == len(feat_names)
        assert len(y_vol) == len(small_dataset)
        assert len(y_cong) == len(small_dataset)

    def test_save_and_load_csv(self, small_dataset, tmp_path):
        p = tmp_path / "traffic.csv"
        small_dataset.save_csv(p)
        loaded = TrafficDataset.from_csv(p)
        assert len(loaded) == len(small_dataset)

    def test_congestion_labels_valid(self, small_dataset):
        X, y_vol, y_cong, _ = small_dataset.to_feature_matrix()
        assert set(y_cong.tolist()).issubset({0, 1, 2, 3})

    def test_volume_positive(self, small_dataset):
        X, y_vol, y_cong, _ = small_dataset.to_feature_matrix()
        assert (y_vol > 0).all()
