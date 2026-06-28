# Payment Fraud Detection Dashboard
dashboardlink: https://kundukamal.github.io/Payment-Fraud-Detection-Dashboard/dashboard.html

A card-not-present (CNP) fraud detection pipeline and interactive dashboard, built
around the kind of checkout fraud a BNPL provider faces. Two parts:

1. **Python pipeline** (`src/`) — runs on the real Kaggle Sparkov dataset:
   engineers interpretable fraud features, trains an XGBoost model with class-imbalance
   handling, and exports metrics + scored transactions.
2. **Interactive dashboard** (`dashboard.html`) — a standalone webpage with fraud
   KPIs, charts, a live risk-threshold slider, and a scored-transaction table.
   Opens in any browser, no server needed.

---

## Why this dataset

We use the **Sparkov simulated transactions** dataset (Kartik2112, Kaggle), *not*
the classic ULB `creditcard.csv`. The ULB set is anonymized into PCA components
(`V1..V28`) — you cannot explain *why* a transaction is risky. Sparkov keeps real,
interpretable columns (merchant, category, amount, location, time), so every fraud
flag has a business reason. It is also card-not-present fraud — the exact type a
checkout/BNPL business sees. License: CC0 (public domain).

Dataset: https://www.kaggle.com/datasets/kartik2112/fraud-detection

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset from Kaggle and place the CSVs in data/
#    Expected: data/fraudTrain.csv  and  data/fraudTest.csv
mkdir -p data

# 3. Run the pipeline
python src/pipeline.py --train data/fraudTrain.csv --test data/fraudTest.csv

# 4. Open the dashboard
#    dashboard.html works standalone with built-in demo data.
#    To load YOUR pipeline output, see "Connecting real output" below.
```

## What the pipeline does

1. **Load & clean** the raw transaction logs.
2. **Feature engineering** — interpretable signals:
   - `amt_zscore`: how unusual the amount is vs. that cardholder's own history
   - `hour` / `is_night`: time-of-day risk
   - `distance_km`: haversine distance between cardholder home and merchant
   - `category_fraud_rate`: historical fraud rate of the merchant category
   - `txns_last_24h`: velocity (how many times the card was used recently)
3. **Train** an XGBoost classifier with `scale_pos_weight` for the ~0.5% fraud rate.
4. **Evaluate** with the metrics that matter for imbalanced fraud — precision,
   recall, F1, and AUPRC (area under precision-recall curve). Accuracy is ignored
   on purpose: a model that predicts "never fraud" scores 99.5% accuracy and is useless.
5. **Export** `outputs/metrics.json` and `outputs/scored_sample.json` — the dashboard
   reads these.

## Connecting real output to the dashboard

After running the pipeline, `outputs/scored_sample.json` and `outputs/metrics.json`
are written. Open `dashboard.html`, click **"Load pipeline output"**, and select
those files — the KPIs and table repopulate from your real run. Without that, the
dashboard shows representative demo data so it always works in an interview.

## Files

```
fraud-dashboard/
├── README.md
├── requirements.txt
├── dashboard.html            # standalone interactive dashboard
├── src/
│   ├── pipeline.py           # orchestrates the full run
│   ├── features.py           # interpretable feature engineering
│   └── model.py              # XGBoost train + evaluate + export
└── data/                     # place Kaggle CSVs here (gitignored)
```
