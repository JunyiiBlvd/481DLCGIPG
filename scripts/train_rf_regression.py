"""
RF baseline regression on log(price_usd) using 4C + grading features.
Trains per subset (ja_natural, be_natural), evaluates on test split,
saves results/training/regression/rf_baseline.json, prints comparison
table vs vision (EfficientNetV2) regression model.
"""

import json
import math
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "splits"
OUT_DIR = pathlib.Path(__file__).parent.parent / "results" / "training" / "regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ordinal mappings (worst → best)
CUT_ORDER = {"Fair": 0, "Good": 1, "Very Good": 2, "Ideal": 3, "Ideal+": 4}
COLOR_ORDER = {c: i for i, c in enumerate(["D", "E", "F", "G", "H", "I", "J", "K"])}
CLARITY_ORDER = {c: i for i, c in enumerate(
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF", "FL"]
)}

NUMERIC_FEATURES = ["carat", "depth_pct", "table_pct"]
CATEGORICAL_FEATURES = ["cut", "color", "clarity"]  # ordinal encoded

VISION_METRICS = {
    "ja_natural": {
        "test_log_mae": 0.33165283222016656,
        "test_log_rmse": 0.4358374432979261,
        "test_usd_mae": 2184.009378638574,
        "test_usd_rmse": 6583.4563161283095,
        "test_usd_med_ape": 25.992130236200435,
    },
    "be_natural": {
        "test_log_mae": 0.3469895426935592,
        "test_log_rmse": 0.48750646633666816,
        "test_usd_mae": 1720.8469134861577,
        "test_usd_rmse": 5227.3142753550455,
        "test_usd_med_ape": 25.483129042326752,
    },
}


def encode_features(df):
    X = df[NUMERIC_FEATURES].copy()
    X["cut_ord"] = df["cut"].map(CUT_ORDER)
    X["color_ord"] = df["color"].map(COLOR_ORDER)
    X["clarity_ord"] = df["clarity"].map(CLARITY_ORDER)
    # Drop rows where encoding failed (unknown categories)
    mask = X.notna().all(axis=1)
    if not mask.all():
        print(f"  Dropping {(~mask).sum()} rows with unknown category values")
    return X[mask].values, mask


def compute_metrics(y_true_log, y_pred_log):
    y_true_usd = np.expm1(y_true_log)
    y_pred_usd = np.expm1(y_pred_log)
    log_mae = mean_absolute_error(y_true_log, y_pred_log)
    log_rmse = math.sqrt(mean_squared_error(y_true_log, y_pred_log))
    usd_mae = mean_absolute_error(y_true_usd, y_pred_usd)
    usd_rmse = math.sqrt(mean_squared_error(y_true_usd, y_pred_usd))
    med_ape = float(np.median(np.abs(y_pred_usd - y_true_usd) / y_true_usd) * 100)
    return {
        "test_log_mae": log_mae,
        "test_log_rmse": log_rmse,
        "test_usd_mae": usd_mae,
        "test_usd_rmse": usd_rmse,
        "test_usd_med_ape": med_ape,
    }


def train_subset(subset):
    print(f"\n=== {subset} ===")
    train_df = pd.read_csv(DATA_DIR / f"{subset}_train.csv")
    test_df = pd.read_csv(DATA_DIR / f"{subset}_test.csv")

    print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")
    print(f"  Features used: {NUMERIC_FEATURES + CATEGORICAL_FEATURES}")

    X_train, mask_train = encode_features(train_df)
    y_train = np.log1p(train_df["price_usd"].values[mask_train])

    X_test, mask_test = encode_features(test_df)
    y_test = np.log1p(test_df["price_usd"].values[mask_test])

    print(f"  Fitting RandomForest (n_estimators=200, n_jobs=-1)...")
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    metrics["n_test"] = int(mask_test.sum())
    metrics["model"] = "RandomForest"
    metrics["features"] = NUMERIC_FEATURES + ["cut_ord", "color_ord", "clarity_ord"]

    return metrics


def print_comparison(results):
    subsets = ["ja_natural", "be_natural"]
    metric_keys = [
        ("test_log_mae", "log_MAE"),
        ("test_log_rmse", "log_RMSE"),
        ("test_usd_mae", "USD_MAE"),
        ("test_usd_rmse", "USD_RMSE"),
        ("test_usd_med_ape", "MedAPE%"),
    ]

    print("\n" + "=" * 80)
    print("COMPARISON: RF Baseline vs EfficientNetV2 Vision Regression")
    print("=" * 80)

    for subset in subsets:
        print(f"\n  {subset}")
        print(f"  {'Metric':<14} {'RF Baseline':>14} {'Vision (EV2)':>14}  {'Delta':>10}")
        print(f"  {'-'*14} {'-'*14} {'-'*14}  {'-'*10}")
        rf_m = results[subset]
        vis_m = VISION_METRICS[subset]
        for key, label in metric_keys:
            rf_val = rf_m[key]
            vis_val = vis_m[key]
            delta = rf_val - vis_val
            sign = "+" if delta > 0 else ""
            print(f"  {label:<14} {rf_val:>14.4f} {vis_val:>14.4f}  {sign}{delta:>9.4f}")


def main():
    results = {}
    for subset in ["ja_natural", "be_natural"]:
        results[subset] = train_subset(subset)

    print_comparison(results)

    out_path = OUT_DIR / "rf_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
