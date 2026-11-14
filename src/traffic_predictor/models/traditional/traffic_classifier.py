"""Traditional ML model for traffic volume regression + congestion classification."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

from traffic_predictor.models.base_model import BaseTrafficModel
from traffic_predictor.utils.logger import get_logger

logger = get_logger(__name__)

_REGRESSORS = {
    "gradient_boosting": lambda: GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    ),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        n_jobs=-1, random_state=42,
    ),
    "ridge": lambda: Ridge(alpha=1.0),
}

_CLASSIFIERS = {
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.08, max_depth=4,
        subsample=0.8, random_state=42,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=42,
    ),
    "ridge": lambda: GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42,
    ),
}


class TrafficPredictor(BaseTrafficModel):
    """Dual-head predictor: regression for volume + classification for congestion.

    Both heads share the same feature input but are trained independently,
    allowing each to optimize for its own loss function.
    """

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        scale_features: bool = True,
        random_seed: int = 42,
    ) -> None:
        super().__init__(model_name=f"traffic_{model_type}")
        if model_type not in _REGRESSORS:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(_REGRESSORS)}")
        self.model_type = model_type
        self.scale_features = scale_features
        self.random_seed = random_seed

        self._regressor = _REGRESSORS[model_type]()
        self._classifier = _CLASSIFIERS[model_type]()
        self._scaler = StandardScaler() if scale_features else None
        self._feature_names: list[str] = []

    def fit(
        self,
        X: np.ndarray,
        y_volume: np.ndarray,
        y_congestion: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "TrafficPredictor":
        logger.info(f"Training {self.model_type} on {len(X)} samples, {X.shape[1]} features")

        X_scaled = self._scaler.fit_transform(X) if self._scaler else X
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        logger.info("Fitting volume regressor...")
        self._regressor.fit(X_scaled, y_volume)

        logger.info("Fitting congestion classifier...")
        self._classifier.fit(X_scaled, y_congestion)

        self._is_fitted = True
        logger.info("Training complete")
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        X_scaled = self._scaler.transform(X) if self._scaler else X
        volumes = self._regressor.predict(X_scaled)
        congestion = self._classifier.predict(X_scaled)
        return volumes.astype(np.float32), congestion.astype(np.int32)

    def predict_proba_congestion(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for all 4 congestion levels."""
        self._check_fitted()
        X_scaled = self._scaler.transform(X) if self._scaler else X
        return self._classifier.predict_proba(X_scaled)

    def get_feature_importance(self, top_n: int = 15) -> list[tuple[str, float]]:
        """Return top-N feature importances from the regressor."""
        self._check_fitted()
        if not hasattr(self._regressor, "feature_importances_"):
            return []
        importances = self._regressor.feature_importances_
        pairs = sorted(
            zip(self._feature_names, importances.tolist()),
            key=lambda x: x[1], reverse=True,
        )
        return pairs[:top_n]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "regressor": self._regressor,
                "classifier": self._classifier,
                "scaler": self._scaler,
                "feature_names": self._feature_names,
                "config": {
                    "model_type": self.model_type,
                    "scale_features": self.scale_features,
                },
            },
            path,
            compress=3,
        )
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TrafficPredictor":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model_name = f"traffic_{data['config']['model_type']}"
        obj._regressor = data["regressor"]
        obj._classifier = data["classifier"]
        obj._scaler = data["scaler"]
        obj._feature_names = data["feature_names"]
        obj.model_type = data["config"]["model_type"]
        obj.scale_features = data["config"]["scale_features"]
        obj._is_fitted = True
        logger.info(f"Model loaded from {path}")
        return obj
