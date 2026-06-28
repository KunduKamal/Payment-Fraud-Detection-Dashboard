"""
features.py — interpretable fraud-detection features.

Every feature here has a plain-language business meaning, so a flagged
transaction can be explained to a risk officer, not just a data scientist.
This is deliberate: a BNPL/payments risk team needs defensible decisions.
"""

import numpy as np
import pandas as pd


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points (vectorised)."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def build_features(df: pd.DataFrame, category_fraud_rate: dict | None = None):
    """
    Take raw Sparkov columns and return (X_features_df, category_fraud_rate).

    Raw columns used:
      trans_date_trans_time, amt, lat, long, merch_lat, merch_long,
      category, cc_num, (is_fraud for training only)
    """
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

    # --- Time-of-day risk ------------------------------------------------
    df["hour"] = df["trans_date_trans_time"].dt.hour
    # Fraud skews to the early hours when cardholders are asleep.
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)

    # --- Geographic distance ---------------------------------------------
    # Distance between the cardholder's home and the merchant location.
    # A purchase 2,000 km from home is a classic CNP-fraud signal.
    df["distance_km"] = _haversine_km(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    # --- Amount anomaly vs. the card's own history -----------------------
    # z-score of this amount against everything that card has spent before.
    grp = df.groupby("cc_num")["amt"]
    card_mean = grp.transform("mean")
    card_std = grp.transform("std").replace(0, np.nan)
    df["amt_zscore"] = ((df["amt"] - card_mean) / card_std).fillna(0)

    # --- Velocity: transactions on this card in the last 24h -------------
    df["txns_last_24h"] = (
        df.set_index("trans_date_trans_time")
        .groupby("cc_num")["amt"]
        .rolling("24h")
        .count()
        .reset_index(level=0, drop=True)
        .values
    )

    # --- Category historical fraud rate ----------------------------------
    # Learned on training data, then re-applied to test data unchanged.
    if category_fraud_rate is None:
        category_fraud_rate = (
            df.groupby("category")["is_fraud"].mean().to_dict()
            if "is_fraud" in df.columns
            else {}
        )
    overall = np.mean(list(category_fraud_rate.values())) if category_fraud_rate else 0.0
    df["category_fraud_rate"] = df["category"].map(category_fraud_rate).fillna(overall)

    feature_cols = [
        "amt",
        "amt_zscore",
        "hour",
        "is_night",
        "distance_km",
        "txns_last_24h",
        "category_fraud_rate",
    ]
    return df, feature_cols, category_fraud_rate
