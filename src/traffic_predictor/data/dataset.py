"""Traffic dataset — synthetic generation and I/O (polars-backed)."""

from __future__ import annotations

import random
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from traffic_predictor.utils.logger import get_logger

logger = get_logger(__name__)

# Congestion level labels
CONGESTION_LEVELS = {0: "low", 1: "moderate", 2: "high", 3: "severe"}

# US federal holidays (month, day)
_US_HOLIDAYS = {
    (1, 1), (7, 4), (12, 25), (12, 24), (11, 11),
    (1, 15), (2, 19), (5, 27), (9, 2), (10, 14), (11, 28), (11, 29),
}

# Weather condition codes: 0=clear, 1=cloudy, 2=rain, 3=heavy_rain, 4=snow, 5=fog
WEATHER_LABELS = {0: "clear", 1: "cloudy", 2: "rain", 3: "heavy_rain", 4: "snow", 5: "fog"}

# Intersection type: 0=residential, 1=arterial, 2=highway_on_ramp, 3=downtown, 4=commercial
INTERSECTION_TYPES = {0: "residential", 1: "arterial", 2: "highway_on_ramp", 3: "downtown", 4: "commercial"}


@dataclass
class TrafficSplit:
    features: np.ndarray
    targets_volume: np.ndarray
    targets_congestion: np.ndarray
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.targets_volume)


class TrafficDataset:
    """Traffic dataset manager.

    Stores features as numpy arrays; uses polars for file I/O.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self._records: list[dict] = []
        self._feature_names: list[str] = []

    @classmethod
    def generate_synthetic(
        cls,
        n_records: int = 50000,
        n_intersections: int = 20,
        random_seed: int = 42,
    ) -> "TrafficDataset":
        ds = cls(random_seed=random_seed)
        ds._records = _generate_synthetic_traffic(n_records, n_intersections, random_seed)
        logger.info(f"Generated synthetic dataset: {len(ds._records)} records across {n_intersections} intersections")
        return ds

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs) -> "TrafficDataset":
        import polars as pl
        ds = cls(**kwargs)
        df = pl.read_csv(str(path))
        ds._records = df.to_dicts()
        logger.info(f"Loaded dataset from {path}: {len(ds._records)} records")
        return ds

    def save_csv(self, path: str | Path) -> None:
        import polars as pl
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(self._records).write_csv(str(path))
        logger.info(f"Dataset saved to {path}")

    def to_feature_matrix(self, feature_names: Optional[list[str]] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Convert records to numpy feature matrix + target arrays.

        Returns: (X, y_volume, y_congestion, feature_names)
        """
        if not self._records:
            raise ValueError("Dataset is empty")

        all_feature_names = [
            "hour", "day_of_week", "month", "is_weekend", "is_holiday",
            "is_morning_rush", "is_evening_rush", "is_night",
            "intersection_id", "intersection_type",
            "weather_code", "temperature_c", "precipitation_mm", "visibility_km",
            "speed_limit_kmh", "num_lanes",
            "volume_lag1", "volume_lag2", "volume_lag3",
            "speed_lag1", "speed_lag2", "speed_lag3",
            "volume_rolling_mean_3h", "volume_rolling_mean_6h", "volume_rolling_mean_12h",
            "speed_rolling_mean_3h",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
        ]

        if feature_names is None:
            feature_names = all_feature_names

        X = np.array([[r[f] for f in feature_names] for r in self._records], dtype=np.float32)
        y_volume = np.array([r["traffic_volume"] for r in self._records], dtype=np.float32)
        y_congestion = np.array([r["congestion_level"] for r in self._records], dtype=np.int32)

        self._feature_names = feature_names
        return X, y_volume, y_congestion, feature_names

    def split(
        self,
        test_size: float = 0.15,
        val_size: float = 0.10,
    ) -> tuple[TrafficSplit, TrafficSplit, TrafficSplit]:
        from sklearn.model_selection import train_test_split

        X, y_vol, y_cong, feat_names = self.to_feature_matrix()
        n = len(X)

        X_tv, X_test, yv_tv, yv_test, yc_tv, yc_test = train_test_split(
            X, y_vol, y_cong, test_size=test_size, random_state=self.random_seed
        )
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, yv_train, yv_val, yc_train, yc_val = train_test_split(
            X_tv, yv_tv, yc_tv, test_size=val_ratio, random_state=self.random_seed
        )

        logger.info(f"Split → train:{len(X_train)} | val:{len(X_val)} | test:{len(X_test)}")
        return (
            TrafficSplit(X_train, yv_train, yc_train, feat_names),
            TrafficSplit(X_val, yv_val, yc_val, feat_names),
            TrafficSplit(X_test, yv_test, yc_test, feat_names),
        )

    def congestion_distribution(self) -> dict[str, int]:
        from collections import Counter
        counts = Counter(r["congestion_level"] for r in self._records)
        return {CONGESTION_LEVELS[k]: v for k, v in sorted(counts.items())}

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Synthetic traffic data generation
# ---------------------------------------------------------------------------

def _is_holiday(month: int, day: int) -> bool:
    return (month, day) in _US_HOLIDAYS


def _rush_hour_factor(hour: int, is_weekend: bool) -> float:
    """Return a multiplier 0.0–1.0 representing rush-hour intensity."""
    if is_weekend:
        # Weekend: midday peak
        if 11 <= hour <= 15:
            return 0.6
        if 9 <= hour <= 18:
            return 0.4
        return 0.1
    # Weekday: morning rush 7-9, evening rush 16-19
    if 7 <= hour <= 9:
        return 1.0 - abs(hour - 8) * 0.3
    if 16 <= hour <= 19:
        return 1.0 - abs(hour - 17.5) * 0.25
    if 10 <= hour <= 15:
        return 0.4
    if 20 <= hour <= 22:
        return 0.2
    return 0.05


def _weather_factor(weather_code: int) -> float:
    return {0: 1.0, 1: 0.95, 2: 0.80, 3: 0.60, 4: 0.50, 5: 0.70}.get(weather_code, 1.0)


def _generate_synthetic_traffic(
    n_records: int, n_intersections: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Assign fixed properties to each intersection
    intersection_props = {}
    for i in range(n_intersections):
        itype = rng.randint(0, 4)
        base_caps = {0: 400, 1: 1200, 2: 2500, 3: 1800, 4: 900}
        intersection_props[i] = {
            "type": itype,
            "base_capacity": base_caps[itype],
            "speed_limit": rng.choice([30, 50, 70, 90, 110]),
            "num_lanes": rng.randint(2, 6),
        }

    # Simulate rolling lag values per intersection
    # We'll generate in time order per intersection then shuffle
    records_by_intersection: dict[int, list[dict]] = {i: [] for i in range(n_intersections)}
    records_per_intersection = n_records // n_intersections

    for inter_id in range(n_intersections):
        props = intersection_props[inter_id]
        base_cap = props["base_capacity"]
        vol_history: list[float] = [base_cap * 0.3] * 12
        spd_history: list[float] = [props["speed_limit"] * 0.8] * 6

        for _ in range(records_per_intersection):
            # Random timestamp components
            month = rng.randint(1, 12)
            day = rng.randint(1, 28)
            hour = rng.randint(0, 23)
            dow = rng.randint(0, 6)  # 0=Mon, 6=Sun
            is_weekend = dow >= 5
            is_holiday = _is_holiday(month, day)

            # Weather (seasonally influenced)
            if month in (12, 1, 2):
                weather_weights = [0.2, 0.3, 0.2, 0.1, 0.15, 0.05]
            elif month in (6, 7, 8):
                weather_weights = [0.5, 0.25, 0.15, 0.05, 0.0, 0.05]
            else:
                weather_weights = [0.35, 0.3, 0.2, 0.08, 0.02, 0.05]
            weather_code = rng.choices(range(6), weights=weather_weights)[0]
            temp_c = rng.gauss(
                {12: 2, 1: 0, 2: 3, 3: 10, 4: 15, 5: 20, 6: 26, 7: 28, 8: 27, 9: 21, 10: 14, 11: 7}[month], 4
            )
            precip = 0.0
            if weather_code == 2:
                precip = rng.uniform(0.5, 10)
            elif weather_code == 3:
                precip = rng.uniform(10, 50)
            elif weather_code == 4:
                precip = rng.uniform(1, 20)
            visibility = {0: rng.uniform(8, 15), 1: rng.uniform(5, 10), 2: rng.uniform(3, 8),
                          3: rng.uniform(1, 4), 4: rng.uniform(1, 5), 5: rng.uniform(0.5, 3)}[weather_code]

            # Traffic volume model
            rush = _rush_hour_factor(hour, is_weekend)
            weather_mult = _weather_factor(weather_code)
            holiday_mult = 0.5 if is_holiday else 1.0
            noise = np_rng.normal(0, 0.05)
            volume = base_cap * rush * weather_mult * holiday_mult * (1 + noise)
            volume = max(10.0, min(volume, base_cap * 1.1))

            # Speed: inversely related to volume/capacity ratio
            vc_ratio = volume / base_cap
            free_flow_speed = props["speed_limit"] * 0.9
            speed = free_flow_speed * max(0.2, 1 - 0.8 * vc_ratio) * weather_mult
            speed += np_rng.normal(0, 2)
            speed = max(5.0, min(speed, props["speed_limit"] * 1.0))

            # Lag features
            vol_lags = vol_history[-3:]
            spd_lags = spd_history[-3:]

            # Rolling means
            vm3 = float(np.mean(vol_history[-3:]))
            vm6 = float(np.mean(vol_history[-6:]))
            vm12 = float(np.mean(vol_history[-12:]))
            sm3 = float(np.mean(spd_history[-3:]))

            # Cyclic time encoding
            hour_sin = math.sin(2 * math.pi * hour / 24)
            hour_cos = math.cos(2 * math.pi * hour / 24)
            day_sin = math.sin(2 * math.pi * dow / 7)
            day_cos = math.cos(2 * math.pi * dow / 7)

            # Congestion level based on v/c ratio
            if vc_ratio < 0.4:
                congestion = 0  # low
            elif vc_ratio < 0.65:
                congestion = 1  # moderate
            elif vc_ratio < 0.85:
                congestion = 2  # high
            else:
                congestion = 3  # severe

            record = {
                "intersection_id": inter_id,
                "intersection_type": props["type"],
                "month": month,
                "day": day,
                "hour": hour,
                "day_of_week": dow,
                "is_weekend": int(is_weekend),
                "is_holiday": int(is_holiday),
                "is_morning_rush": int(7 <= hour <= 9 and not is_weekend),
                "is_evening_rush": int(16 <= hour <= 19 and not is_weekend),
                "is_night": int(hour < 6 or hour >= 22),
                "weather_code": weather_code,
                "temperature_c": round(temp_c, 1),
                "precipitation_mm": round(precip, 2),
                "visibility_km": round(visibility, 1),
                "speed_limit_kmh": props["speed_limit"],
                "num_lanes": props["num_lanes"],
                "volume_lag1": round(vol_lags[2], 1),
                "volume_lag2": round(vol_lags[1], 1),
                "volume_lag3": round(vol_lags[0], 1),
                "speed_lag1": round(spd_lags[2], 1),
                "speed_lag2": round(spd_lags[1], 1),
                "speed_lag3": round(spd_lags[0], 1),
                "volume_rolling_mean_3h": round(vm3, 1),
                "volume_rolling_mean_6h": round(vm6, 1),
                "volume_rolling_mean_12h": round(vm12, 1),
                "speed_rolling_mean_3h": round(sm3, 1),
                "hour_sin": round(hour_sin, 4),
                "hour_cos": round(hour_cos, 4),
                "day_sin": round(day_sin, 4),
                "day_cos": round(day_cos, 4),
                "traffic_volume": round(volume, 1),
                "avg_speed_kmh": round(speed, 1),
                "congestion_level": congestion,
            }

            vol_history.append(volume)
            if len(vol_history) > 12:
                vol_history.pop(0)
            spd_history.append(speed)
            if len(spd_history) > 6:
                spd_history.pop(0)

            records_by_intersection[inter_id].append(record)

    # Flatten and shuffle
    all_records = []
    for recs in records_by_intersection.values():
        all_records.extend(recs)
    rng.shuffle(all_records)
    return all_records
