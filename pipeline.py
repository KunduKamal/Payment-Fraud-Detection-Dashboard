"""
pipeline.py — end-to-end fraud detection run.

Usage:
    python src/pipeline.py --train data/fraudTrain.csv --test data/fraudTest.csv

Outputs:
    outputs/metrics.json        model performance + feature importance + PR curve
    outputs/scored_sample.json  riskiest scored transactions for the dashboard
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features          # noqa: E402
from model import (                          # noqa: E402
    train_model, evaluate, feature_importance, export,
)


def run(train_path, test_path, threshold):
    print("Loading data...")
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    print(f"  train: {len(train_raw):,} rows | test: {len(test_raw):,} rows")

    print("Engineering features...")
    train_df, feature_cols, cat_rates = build_features(train_raw)
    test_df, _, _ = build_features(test_raw, category_fraud_rate=cat_rates)

    X_train, y_train = train_df[feature_cols], train_df["is_fraud"]
    X_test, y_test = test_df[feature_cols], test_df["is_fraud"]
    print(f"  features: {feature_cols}")
    print(f"  train fraud rate: {y_train.mean():.4%}")

    print("Training XGBoost (scale_pos_weight balances the imbalance)...")
    model = train_model(X_train, y_train)

    print("Evaluating...")
    metrics = evaluate(model, X_test, y_test, threshold=threshold)
    importances = feature_importance(model, feature_cols)

    print("\n=== Results (fraud class) ===")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")
    print(f"  AUPRC:     {metrics['auprc']:.3f}   <- the headline metric")
    c = metrics["confusion"]
    print(f"  Caught {c['tp']} frauds, missed {c['fn']}, "
          f"false alarms {c['fp']}")

    # Attach scores to the test set for export.
    test_df["risk_score"] = model.predict_proba(X_test)[:, 1]
    export(metrics, importances, test_df)
    print("\nDone. Open dashboard.html and load the outputs/ files.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    run(args.train, args.test, args.threshold)
