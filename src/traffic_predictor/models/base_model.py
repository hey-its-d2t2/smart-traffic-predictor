"""Abstract base class for all traffic prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseTrafficModel(ABC):
    """Common interface for all traffic prediction models."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y_volume: np.ndarray, y_congestion: np.ndarray) -> "BaseTrafficModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (predicted_volumes, predicted_congestion_levels)."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseTrafficModel":
        ...

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.model_name} is not fitted. Call fit() first.")
