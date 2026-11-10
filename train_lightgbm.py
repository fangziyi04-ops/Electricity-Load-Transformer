
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "hourly_load.csv"
OUT = BASE / "outputs" / "lightgbm"
OUT.mkdir(parents=True, exist_ok=True)

INPUT_LEN = 168
PRED_LEN = 24

df = pd.read_csv(DATA, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

load = df["load_kw"].to_numpy(dtype=np.float32)
timestamps = pd.DatetimeIndex(df["timestamp"])

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)


def make_features(origin, horizon):
    """
    origin: index of first future hour
    horizon: 0..23
    Only information available before origin is used.
    """

    hist = load[origin - INPUT_LEN:origin]
    target_time = timestamps[origin + horizon]

    features = {
        "horizon": horizon + 1,
        "hour": target_time.hour,
        "weekday": target_time.day_of_week,
        "month": target_time.month,
        "dayofyear": target_time.day_of_year,
        "is_weekend": int(target_time.day_of_week >= 5),
    }

    # Recent observations
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        features[f"lag_{lag}"] = float(hist[-lag])

    # Same clock hour in recent days
    for days in [1, 2, 3, 7]:
        index = origin + horizon - 24 * days

        if index < origin:
            features[f"same_hour_{days}d"] = float(load[index])
        else:
            features[f"same_hour_{days}d"] = np.nan

    # History statistics
    for window in [6, 12, 24, 48, 168]:
        recent = hist[-window:]

        features[f"mean_{window}"] = float(recent.mean())
        features[f"std_{window}"] = float(recent.std())
        features[f"min_{window}"] = float(recent.min())
        features[f"max_{window}"] = float(recent.max())

    # High-load state
    for window in [24, 48, 168]:
        recent = hist[-window:]

        features[f"peak_count_{window}"] = int(
            np.sum(recent >= 15.0)
        )

        features[f"peak_ratio_{window}"] = float(
            np.mean(recent >= 15.0)
        )

    features["last24_vs_prev24"] = float(
        hist[-24:].mean() - hist[-48:-24].mean()
    )

    return features


def build_split(start, end):
    """
    Each sample predicts one future hour.
    A prediction origin must have all 24 future
    hours inside the selected target period.
    """

    rows = []
    targets = []
    origins = []
    horizons = []

    first_origin = max(start, INPUT_LEN)
    last_origin = end - PRED_LEN

    for origin in range(first_origin, last_origin + 1):
        for horizon in range(PRED_LEN):
            rows.append(make_features(origin, horizon))
            targets.append(float(load[origin + horizon]))
            origins.append(origin)
            horizons.append(horizon)

    X = pd.DataFrame(rows)
    y = np.asarray(targets, dtype=np.float32)

    return X, y, np.asarray(origins), np.asarray(horizons)


print("Building training features...")
X_train, y_train, _, _ = build_split(0, train_end)

print("Building validation features...")
X_val, y_val, val_origins, val_horizons = build_split(
    train_end, val_end
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

model = lgb.LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=40,
    colsample_bytree=0.9,
    subsample=0.9,
    subsample_freq=1,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

print("\nTraining LightGBM...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="l2",
    callbacks=[
        lgb.early_stopping(60, verbose=False),
        lgb.log_evaluation(100)
    ]
)

pred = model.predict(X_val)
pred = np.maximum(pred, 0.0)

mae = mean_absolute_error(y_val, pred)
rmse = np.sqrt(mean_squared_error(y_val, pred))
bias = np.mean(pred - y_val)

print("\n===== LightGBM Validation =====")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("Bias:", round(float(bias), 4))
print("Best iteration:", model.best_iteration_)

print("\n===== High Load Performance =====")

for threshold in [5, 10, 15]:
    mask = y_val >= threshold

    if not mask.any():
        continue

    high_mae = mean_absolute_error(
        y_val[mask], pred[mask]
    )

    high_recall = np.mean(
        pred[mask] >= threshold
    )

    print(
        f">= {threshold} kW | "
        f"Count: {mask.sum()} | "
        f"MAE: {high_mae:.4f} | "
        f"Recall: {high_recall:.4f}"
    )

# Save predictions
results = pd.DataFrame({
    "origin": val_origins,
    "horizon": val_horizons + 1,
    "actual_kw": y_val,
    "predicted_kw": pred
})

results.to_csv(
    OUT / "validation_predictions.csv",
    index=False
)

model.booster_.save_model(
    str(OUT / "lightgbm_model.txt")
)

print("\nCompleted!")
print("Saved to:", OUT)
