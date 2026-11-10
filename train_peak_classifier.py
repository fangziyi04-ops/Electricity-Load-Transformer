
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs" / "peak_classifier"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    BASE / "data" / "hourly_load.csv",
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)

load = df["load_kw"].to_numpy(dtype=np.float32)
times = pd.DatetimeIndex(df["timestamp"])

INPUT_LEN = 168
PRED_LEN = 24
PEAK_KW = 15.0

train_end = int(len(load) * 0.70)
val_end = int(len(load) * 0.85)


def make_features(origin):
    hist = load[origin - INPUT_LEN:origin]
    t = times[origin]

    f = {
        "hour": t.hour,
        "weekday": t.day_of_week,
        "month": t.month,
        "dayofyear": t.day_of_year,
        "is_weekend": int(t.day_of_week >= 5),
    }

    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        f[f"lag_{lag}"] = float(hist[-lag])

    for window in [6, 12, 24, 48, 168]:
        x = hist[-window:]

        f[f"mean_{window}"] = float(x.mean())
        f[f"std_{window}"] = float(x.std())
        f[f"min_{window}"] = float(x.min())
        f[f"max_{window}"] = float(x.max())

    for window in [24, 48, 168]:
        x = hist[-window:]

        f[f"peak_count_{window}"] = int(
            (x >= PEAK_KW).sum()
        )

        f[f"peak_ratio_{window}"] = float(
            (x >= PEAK_KW).mean()
        )

    f["last24_vs_prev24"] = float(
        hist[-24:].mean() - hist[-48:-24].mean()
    )

    f["past24_peak"] = int(
        (hist[-24:] >= PEAK_KW).any()
    )

    return f


def build_split(start, end):
    rows = []
    labels = []
    origins = []

    first = max(start, INPUT_LEN)
    last = end - PRED_LEN

    for origin in range(first, last + 1):
        future = load[origin:origin + PRED_LEN]

        label = int((future >= PEAK_KW).any())

        rows.append(make_features(origin))
        labels.append(label)
        origins.append(origin)

    return (
        pd.DataFrame(rows),
        np.asarray(labels, dtype=np.int32),
        np.asarray(origins, dtype=np.int32),
    )


print("Building train features...")
X_train, y_train, train_origins = build_split(
    0, train_end
)

print("Building validation features...")
X_val, y_val, val_origins = build_split(
    train_end, val_end
)

print("Train windows:", len(y_train))
print("Validation windows:", len(y_val))
print("Train peak ratio:", round(y_train.mean(), 4))
print("Validation peak ratio:", round(y_val.mean(), 4))


# ------------------------------------------
# Train classifier
# ------------------------------------------

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=5,
    min_child_samples=80,
    reg_lambda=5.0,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

print("\nTraining peak classifier...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="binary_logloss",
    callbacks=[
        lgb.early_stopping(40, verbose=False),
        lgb.log_evaluation(100),
    ],
)

prob = model.predict_proba(X_val)[:, 1]


# ------------------------------------------
# Threshold sweep
# ------------------------------------------

rows = []

for threshold in np.arange(0.10, 0.91, 0.05):
    predicted = prob >= threshold

    rows.append({
        "threshold": round(float(threshold), 2),
        "precision": precision_score(
            y_val, predicted, zero_division=0
        ),
        "recall": recall_score(
            y_val, predicted, zero_division=0
        ),
        "f1": f1_score(
            y_val, predicted, zero_division=0
        ),
        "predicted_peaks": int(predicted.sum()),
    })

threshold_results = pd.DataFrame(rows)

best_row = threshold_results.loc[
    threshold_results["f1"].idxmax()
]

best_threshold = float(best_row["threshold"])

print("\n===== Threshold Sweep =====")
print(threshold_results.round(4).to_string(index=False))

print("\n===== Best Classifier Result =====")
print("Best threshold:", best_threshold)
print("Precision:", round(best_row["precision"], 4))
print("Recall:", round(best_row["recall"], 4))
print("F1:", round(best_row["f1"], 4))
print("Best iteration:", model.best_iteration_)

if len(np.unique(y_val)) == 2:
    print(
        "ROC AUC:",
        round(roc_auc_score(y_val, prob), 4)
    )
    print(
        "Average Precision:",
        round(average_precision_score(y_val, prob), 4)
    )


# ------------------------------------------
# Compare historical peak baseline
# ------------------------------------------

baseline_pred = X_val["past24_peak"].to_numpy()

print("\n===== Historical Peak Baseline =====")
print(
    "Precision:",
    round(precision_score(
        y_val, baseline_pred, zero_division=0
    ), 4)
)
print(
    "Recall:",
    round(recall_score(
        y_val, baseline_pred, zero_division=0
    ), 4)
)
print(
    "F1:",
    round(f1_score(
        y_val, baseline_pred, zero_division=0
    ), 4)
)


# ------------------------------------------
# Save results
# ------------------------------------------

threshold_results.to_csv(
    OUT / "threshold_sweep.csv",
    index=False
)

pd.DataFrame({
    "origin": val_origins,
    "actual_peak": y_val,
    "peak_probability": prob,
    "baseline_prediction": baseline_pred,
}).to_csv(
    OUT / "validation_predictions.csv",
    index=False
)

model.booster_.save_model(
    str(OUT / "peak_classifier.txt")
)

print("\nCompleted!")
print("Results saved to:", OUT)
