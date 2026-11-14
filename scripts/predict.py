"""CLI prediction script for Smart Traffic Predictor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from traffic_predictor.data.dataset import CONGESTION_LEVELS
from traffic_predictor.features.feature_engineering import build_feature_vector
from traffic_predictor.models.traditional.traffic_classifier import TrafficPredictor
from traffic_predictor.utils.logger import get_logger, setup_logger

setup_logger("WARNING")
logger = get_logger(__name__)

CONGESTION_EMOJI = {0: "GREEN", 1: "YELLOW", 2: "ORANGE", 3: "RED"}


def parse_args():
    p = argparse.ArgumentParser(description="Predict traffic conditions")
    p.add_argument("--model", type=str, default=None, help="Path to model .pkl file")
    p.add_argument("--hour", type=int, required=True, help="Hour of day (0-23)")
    p.add_argument("--dow", type=int, default=0, help="Day of week (0=Mon, 6=Sun)")
    p.add_argument("--month", type=int, default=6, help="Month (1-12)")
    p.add_argument("--intersection", type=int, default=0, help="Intersection ID")
    p.add_argument("--itype", type=int, default=1, help="Intersection type (0-4)")
    p.add_argument("--weather", type=int, default=0, help="Weather code (0=clear..5=fog)")
    p.add_argument("--temp", type=float, default=20.0, help="Temperature Celsius")
    p.add_argument("--speed-limit", type=float, default=50.0, help="Speed limit km/h")
    p.add_argument("--lanes", type=int, default=2, help="Number of lanes")
    p.add_argument("--forecast", type=int, default=1, help="Forecast for N hours ahead")
    return p.parse_args()


def find_latest_model() -> Path:
    model_dir = Path("models/saved")
    models = sorted(model_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not models:
        raise FileNotFoundError("No model found. Run `python scripts/train.py` first.")
    return models[0]


def main():
    args = parse_args()

    model_path = Path(args.model) if args.model else find_latest_model()
    model = TrafficPredictor.load(model_path)

    print(f"\nSmart Traffic Predictor — {model.model_name}")
    print("=" * 55)

    weather_labels = {0: "Clear", 1: "Cloudy", 2: "Rain", 3: "Heavy Rain", 4: "Snow", 5: "Fog"}
    dow_labels = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}

    for offset in range(args.forecast):
        h = (args.hour + offset) % 24
        import numpy as np
        X = build_feature_vector(
            hour=h, day_of_week=args.dow, month=args.month,
            intersection_id=args.intersection, intersection_type=args.itype,
            weather_code=args.weather, temperature_c=args.temp,
            precipitation_mm=0.0, visibility_km=10.0,
            speed_limit_kmh=args.speed_limit, num_lanes=args.lanes,
        ).reshape(1, -1)

        volumes, congestions = model.predict(X)
        probas = model.predict_proba_congestion(X)
        vol = float(volumes[0])
        cong = int(congestions[0])
        p = probas[0]

        print(f"\nHour {h:02d}:00  |  {dow_labels[args.dow]}  |  {weather_labels.get(args.weather, '?')}")
        print(f"  Volume:     {vol:.0f} vehicles/hour")
        print(f"  Congestion: [{CONGESTION_EMOJI[cong]}] {CONGESTION_LEVELS[cong].upper()}")
        print(f"  Confidence: {p[cong]*100:.1f}%")
        print(f"  Probabilities: low={p[0]:.2f} moderate={p[1]:.2f} high={p[2]:.2f} severe={p[3]:.2f}")


if __name__ == "__main__":
    main()
