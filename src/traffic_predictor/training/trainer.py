"""Training orchestrator for traffic prediction models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import KFold

from traffic_predictor.data.dataset import TrafficDataset, TrafficSplit
from traffic_predictor.evaluation.metrics import full_evaluation, plot_confusion_matrix, plot_volume_predictions, save_report
from traffic_predictor.models.traditional.traffic_classifier import TrafficPredictor
from traffic_predictor.utils.logger import get_logger

logger = get_logger(__name__)


class TrafficTrainer:
    """End-to-end training pipeline.

    Handles: data loading → splitting → model training →
             cross-validation → evaluation → artifact saving.
    """

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        n_cv_folds: int = 5,
        random_seed: int = 42,
        model_dir: str | Path = "models/saved",
        reports_dir: str | Path = "reports",
    ) -> None:
        self.model_type = model_type
        self.n_cv_folds = n_cv_folds
        self.random_seed = random_seed
        self.model_dir = Path(model_dir)
        self.reports_dir = Path(reports_dir)

    def train(
        self,
        dataset: TrafficDataset,
        run_cv: bool = True,
    ) -> dict:
        train_split, val_split, test_split = dataset.split()

        cv_results = {}
        if run_cv:
            cv_results = self.cross_validate(train_split)

        logger.info("Fitting final model on full training set...")
        model = TrafficPredictor(model_type=self.model_type, random_seed=self.random_seed)
        model.fit(
            train_split.features,
            train_split.targets_volume,
            train_split.targets_congestion,
            feature_names=train_split.feature_names,
        )

        # Evaluate on validation and test
        val_metrics = self._evaluate_split(model, val_split, "validation")
        test_metrics = self._evaluate_split(model, test_split, "test")

        # Save artifacts
        model_path = self.model_dir / f"traffic_{self.model_type}.pkl"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        save_report(val_metrics, self.reports_dir / "validation_report.json")
        save_report(test_metrics, self.reports_dir / "test_report.json")

        # Plots
        y_vol_pred, y_cong_pred = model.predict(test_split.features)
        plot_confusion_matrix(
            test_split.targets_congestion, y_cong_pred,
            save_path=self.reports_dir / "test_confusion_matrix.png",
        )
        plot_volume_predictions(
            test_split.targets_volume, y_vol_pred,
            save_path=self.reports_dir / "test_volume_predictions.png",
        )

        # Feature importance
        importance = model.get_feature_importance(top_n=15)
        if importance:
            logger.info("Top features (by importance):")
            for name, imp in importance[:10]:
                logger.info(f"  {name:<30s} {imp:.4f}")

        result = {
            "cv": cv_results,
            "validation": val_metrics,
            "test": test_metrics,
            "model_path": str(model_path),
        }
        return result

    def cross_validate(self, train_split: TrafficSplit) -> dict:
        kf = KFold(n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_seed)
        X, y_vol, y_cong = train_split.features, train_split.targets_volume, train_split.targets_congestion

        fold_maes, fold_rmses, fold_r2s, fold_f1s = [], [], [], []

        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X)):
            m = TrafficPredictor(model_type=self.model_type, random_seed=self.random_seed)
            m.fit(X[tr_idx], y_vol[tr_idx], y_cong[tr_idx])
            vol_pred, cong_pred = m.predict(X[va_idx])

            from traffic_predictor.evaluation.metrics import regression_metrics, classification_metrics
            reg = regression_metrics(y_vol[va_idx], vol_pred)
            clf = classification_metrics(y_cong[va_idx], cong_pred)

            fold_maes.append(reg["mae"])
            fold_rmses.append(reg["rmse"])
            fold_r2s.append(reg["r2"])
            fold_f1s.append(clf["macro_f1"])

            logger.info(
                f"Fold {fold_idx+1}/{self.n_cv_folds} — "
                f"MAE={reg['mae']:.1f} | RMSE={reg['rmse']:.1f} | "
                f"R²={reg['r2']:.4f} | F1={clf['macro_f1']:.4f}"
            )

        cv = {
            "mae_mean": round(float(np.mean(fold_maes)), 2),
            "mae_std": round(float(np.std(fold_maes)), 2),
            "rmse_mean": round(float(np.mean(fold_rmses)), 2),
            "rmse_std": round(float(np.std(fold_rmses)), 2),
            "r2_mean": round(float(np.mean(fold_r2s)), 4),
            "r2_std": round(float(np.std(fold_r2s)), 4),
            "macro_f1_mean": round(float(np.mean(fold_f1s)), 4),
            "macro_f1_std": round(float(np.std(fold_f1s)), 4),
        }
        logger.info(
            f"CV Results — MAE: {cv['mae_mean']:.1f} ±{cv['mae_std']:.1f} | "
            f"R²: {cv['r2_mean']:.4f} ±{cv['r2_std']:.4f} | "
            f"F1: {cv['macro_f1_mean']:.4f} ±{cv['macro_f1_std']:.4f}"
        )
        return cv

    def _evaluate_split(self, model: TrafficPredictor, split: TrafficSplit, name: str) -> dict:
        y_vol_pred, y_cong_pred = model.predict(split.features)
        return full_evaluation(
            split.targets_volume, y_vol_pred,
            split.targets_congestion, y_cong_pred,
            split_name=name,
        )
