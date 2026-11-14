"""CLI training script for Smart Traffic Predictor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from traffic_predictor.data.dataset import TrafficDataset
from traffic_predictor.training.trainer import TrafficTrainer
from traffic_predictor.utils.logger import get_logger, setup_logger

setup_logger("INFO")
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Smart Traffic Predictor")
    p.add_argument("--data", type=str, default=None, help="Path to CSV data file")
    p.add_argument("--model-type", type=str, default="gradient_boosting",
                   choices=["gradient_boosting", "random_forest", "ridge"],
                   help="ML model type")
    p.add_argument("--n-samples", type=int, default=50000,
                   help="Number of synthetic samples to generate")
    p.add_argument("--n-intersections", type=int, default=20,
                   help="Number of intersections in synthetic data")
    p.add_argument("--no-cv", action="store_true", help="Skip cross-validation")
    p.add_argument("--model-dir", type=str, default="models/saved")
    p.add_argument("--reports-dir", type=str, default="reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.data:
        logger.info(f"Loading data from {args.data}")
        dataset = TrafficDataset.from_csv(args.data)
    else:
        logger.info(f"Generating {args.n_samples} synthetic traffic records across {args.n_intersections} intersections...")
        dataset = TrafficDataset.generate_synthetic(
            n_records=args.n_samples,
            n_intersections=args.n_intersections,
        )

    dist = dataset.congestion_distribution()
    logger.info(f"Congestion distribution: {dist}")

    trainer = TrafficTrainer(
        model_type=args.model_type,
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
    )

    result = trainer.train(dataset, run_cv=not args.no_cv)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    if result.get("cv"):
        cv = result["cv"]
        logger.info(f"CV MAE:       {cv['mae_mean']:.1f} ± {cv['mae_std']:.1f}")
        logger.info(f"CV R²:        {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")
        logger.info(f"CV Macro-F1:  {cv['macro_f1_mean']:.4f} ± {cv['macro_f1_std']:.4f}")
    test = result["test"]
    logger.info(f"TEST MAE:     {test['regression']['mae']:.1f}")
    logger.info(f"TEST RMSE:    {test['regression']['rmse']:.1f}")
    logger.info(f"TEST MAPE:    {test['regression']['mape_pct']:.1f}%")
    logger.info(f"TEST R²:      {test['regression']['r2']:.4f}")
    logger.info(f"TEST Accuracy:{test['classification']['accuracy']:.4f}")
    logger.info(f"TEST F1:      {test['classification']['macro_f1']:.4f}")
    logger.info("=" * 60)
    logger.info(f"Model saved → {result['model_path']}")


if __name__ == "__main__":
    main()
