"""
model.py — train XGBoost on the engineered features, evaluate with
fraud-appropriate metrics, and export JSON the dashboard can read.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    confusion_matrix,
)
from xgboost import XGBClassifier


def train_model(X_train, y_train):
    """
    XGBoost with scale_pos_weight to counter the ~0.5% fraud rate.
    Without this the model just learns to predict 'not fraud' every time.
    """
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",   # optimise area under precision-recall curve
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, threshold=0.5):
    """Return a dict of fraud-relevant metrics at the chosen threshold."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    auprc = average_precision_score(y_test, proba)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    # Precision/recall across thresholds — powers the dashboard slider.
    p_curve, r_curve, t_curve = precision_recall_curve(y_test, proba)
    curve = [
        {"threshold": float(t), "precision": float(p), "recall": float(r)}
        for p, r, t in zip(p_curve[::50], r_curve[::50], t_curve[::50])
    ]

    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auprc": float(auprc),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "pr_curve": curve,
        "fraud_rate": float(np.mean(y_test)),
        "n_test": int(len(y_test)),
    }


def feature_importance(model, feature_cols):
    imp = model.feature_importances_
    pairs = sorted(zip(feature_cols, imp), key=lambda x: -x[1])
    return [{"feature": f, "importance": float(i)} for f, i in pairs]


def export(metrics, importances, scored_df, out_dir="outputs", n_sample=200):
    """Write metrics.json and scored_sample.json for the dashboard."""
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    metrics_payload = {**metrics, "feature_importance": importances}
    (out / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    # A readable sample of scored transactions, riskiest first.
    cols = [
        "trans_date_trans_time", "category", "amt", "distance_km",
        "amt_zscore", "txns_last_24h", "is_night", "risk_score", "is_fraud",
    ]
    cols = [c for c in cols if c in scored_df.columns]
    sample = (
        scored_df.sort_values("risk_score", ascending=False)
        .head(n_sample)[cols]
        .copy()
    )
    sample["trans_date_trans_time"] = sample["trans_date_trans_time"].astype(str)
    (out / "scored_sample.json").write_text(
        json.dumps(sample.to_dict(orient="records"), indent=2, default=str)
    )
    print(f"Wrote {out/'metrics.json'} and {out/'scored_sample.json'}")
