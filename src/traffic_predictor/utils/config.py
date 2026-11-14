"""Application configuration via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    test_size: float = 0.15
    val_size: float = 0.10
    n_synthetic_records: int = 50000
    n_intersections: int = 20
    random_seed: int = 42


class ModelConfig(BaseModel):
    model_dir: Path = Path("models/saved")
    default_classifier: str = "gradient_boosting"
    n_lag_features: int = 3
    rolling_window_sizes: List[int] = Field(default_factory=lambda: [3, 6, 12])


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    log_level: str = "info"
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP__",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    app_name: str = "Smart Traffic Predictor"
    version: str = "0.1.0"
    log_level: str = "INFO"
    log_file: str | None = None

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig()
