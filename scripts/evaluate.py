"""CLI evaluation script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from traffic_predictor.data.dataset import TrafficDataset
from traffic_predictor.evaluation.metrics import full_evaluation, plot_confusion_matrix, plot_volume_predictions, save_report
from traffic_predictor.models.traditional.traffic_classifier import TrafficPredictor
from traffic_predictor.utils.logger import get_logger, setup_logger

setup_logger("INFO")
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained traffic predictor")
    p.add_argument("--model", type=str, required=True, help="Path to .pkl model file")
    p.add_argument("--data", type=str, default=None, help="CSV data file (default: generate synthetic)")
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--reports-dir", type=str, default="reports")
    return p.parse_args()


def main():
    args = parse_args()
    model = TrafficPredictor.load(args.model)

    if args.data:
        ds = TrafficDataset.from_csv(args.data)
    else:
        logger.info(f"Generating {args.n_samples} synthetic evaluation samples...")
        ds = TrafficDataset.generate_synthetic(n_records=args.n_samples, random_seed=99)

    X, y_vol, y_cong, feat_names = ds.to_feature_matrix()
    y_vol_pred, y_cong_pred = model.predict(X)
    metrics = full_evaluation(y_vol, y_vol_pred, y_cong, y_cong_pred, split_name="evaluation")

    rd = Path(args.reports_dir)
    save_report(metrics, rd / "evaluation_report.json")
    plot_confusion_matrix(y_cong, y_cong_pred, save_path=rd / "evaluation_confusion_matrix.png")
    plot_volume_predictions(y_vol, y_vol_pred, save_path=rd / "evaluation_volume_predictions.png")

    logger.info("Evaluation complete. Reports saved to reports/")


if __name__ == "__main__":
    main()
