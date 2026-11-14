"""Evaluation metrics for traffic prediction — regression + classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from traffic_predictor.utils.logger import get_logger

logger = get_logger(__name__)

CONGESTION_LABELS = ["low", "moderate", "high", "severe"]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, MAPE, R² for volume regression."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    # MAPE: exclude near-zero actuals
    mask = np.abs(y_true) > 1
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape_pct": round(mape, 2), "r2": round(r2, 4)}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute accuracy, macro-F1, per-class F1 for congestion classification."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    present_labels = sorted(set(y_true.tolist()))
    per_class = {}
    for i, label_id in enumerate(present_labels):
        name = CONGESTION_LABELS[label_id] if label_id < len(CONGESTION_LABELS) else str(label_id)
        per_class[name] = round(float(per_class_f1[i]), 4)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": per_class,
    }


def full_evaluation(
    y_vol_true: np.ndarray,
    y_vol_pred: np.ndarray,
    y_cong_true: np.ndarray,
    y_cong_pred: np.ndarray,
    split_name: str = "test",
) -> dict:
    reg = regression_metrics(y_vol_true, y_vol_pred)
    clf = classification_metrics(y_cong_true, y_cong_pred)

    logger.info(
        f"[{split_name}] Volume — MAE={reg['mae']:.1f} | RMSE={reg['rmse']:.1f} | "
        f"MAPE={reg['mape_pct']:.1f}% | R²={reg['r2']:.4f}"
    )
    logger.info(
        f"[{split_name}] Congestion — Accuracy={clf['accuracy']:.4f} | Macro-F1={clf['macro_f1']:.4f}"
    )
    return {"split": split_name, "regression": reg, "classification": clf}


def save_report(metrics: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Report saved to {path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str | Path] = None,
    title: str = "Congestion Confusion Matrix",
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — skipping plot")
        return

    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels = [CONGESTION_LABELS[i] if i < len(CONGESTION_LABELS) else str(i) for i in present]

    cm = confusion_matrix(y_true, y_pred, labels=present)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=100)
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.close(fig)


def plot_volume_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_samples: int = 200,
    save_path: Optional[str | Path] = None,
    title: str = "Predicted vs Actual Traffic Volume",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    idx = np.random.choice(len(y_true), min(n_samples, len(y_true)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter
    axes[0].scatter(y_true[idx], y_pred[idx], alpha=0.3, s=10)
    mn, mx = y_true.min(), y_true.max()
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1)
    axes[0].set_xlabel("Actual Volume (vehicles/h)")
    axes[0].set_ylabel("Predicted Volume (vehicles/h)")
    axes[0].set_title("Actual vs Predicted")

    # Time-series line for first n_samples
    xs = np.arange(min(n_samples, len(y_true)))
    axes[1].plot(xs, y_true[xs], label="Actual", alpha=0.7)
    axes[1].plot(xs, y_pred[xs], label="Predicted", alpha=0.7)
    axes[1].legend()
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Volume (vehicles/h)")
    axes[1].set_title(title)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=100)
        logger.info(f"Volume prediction plot saved to {save_path}")
    plt.close(fig)
