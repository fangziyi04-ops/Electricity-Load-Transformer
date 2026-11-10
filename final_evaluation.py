
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)

from dataset_v2 import test_loader, mean, std
from model_v2 import LoadTransformerV2


# ==========================================
# 1. Configuration
# ==========================================

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs" / "final_evaluation"
OUT.mkdir(parents=True, exist_ok=True)

PEAK_KW = 15.0
PRED_LEN = 24

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 2. Load trained Transformer V2
# ==========================================

checkpoint = torch.load(
    BASE / "models" / "best_transformer_v2.pth",
    map_location=device,
    weights_only=True
)

model = LoadTransformerV2().to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ==========================================
# 3. Run test predictions
# ==========================================

histories = []
actuals = []
predictions = []

with torch.no_grad():

    for x, y in test_loader:

        pred = model(x.to(device))

        histories.append(
            x[:, :, 0].cpu().numpy()
        )

        actuals.append(
            y.cpu().numpy()
        )

        predictions.append(
            pred.cpu().numpy()
        )


history = np.concatenate(histories) * std + mean
actual = np.concatenate(actuals) * std + mean
predicted = np.concatenate(predictions) * std + mean

predicted = np.maximum(predicted, 0.0)

print("Test windows:", len(actual))
print("Prediction shape:", predicted.shape)


# ==========================================
# 4. Overall regression metrics
# ==========================================

y_true = actual.reshape(-1)
y_pred = predicted.reshape(-1)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
bias = np.mean(y_pred - y_true)

print("\n===== Final Transformer V2 Test =====")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("Bias:", round(float(bias), 4))


# ==========================================
# 5. High-load error analysis
# ==========================================

rows = []

for threshold in [5, 10, 15, 20]:

    mask = y_true >= threshold

    if not mask.any():
        continue

    rows.append({
        "threshold_kw": threshold,
        "count": int(mask.sum()),
        "ratio": float(mask.mean()),
        "mae_kw": float(
            mean_absolute_error(y_true[mask], y_pred[mask])
        ),
        "rmse_kw": float(
            np.sqrt(mean_squared_error(
                y_true[mask], y_pred[mask]
            ))
        ),
        "recall": float(
            np.mean(y_pred[mask] >= threshold)
        )
    })

high_load_df = pd.DataFrame(rows)

print("\n===== High Load Test Performance =====")
print(high_load_df.round(4).to_string(index=False))

high_load_df.to_csv(
    OUT / "high_load_metrics.csv",
    index=False
)


# ==========================================
# 6. Peak risk detection
# ==========================================

# Actual peak in future 24h
future_peak = (actual >= PEAK_KW).any(axis=1)

# Historical peak in previous 24h
past_peak = (history[:, -24:] >= PEAK_KW).any(axis=1)

# V2 predicts a peak in future 24h
model_peak = (predicted >= PEAK_KW).any(axis=1)


def peak_metrics(y_true, y_pred, name):

    return {
        "method": name,
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "predicted_peak_windows": int(y_pred.sum())
    }


peak_results = pd.DataFrame([
    peak_metrics(
        future_peak,
        past_peak,
        "Historical 24h Rule"
    ),
    peak_metrics(
        future_peak,
        model_peak,
        "Transformer V2 Direct"
    )
])

print("\n===== Peak Risk Detection Test =====")
print(peak_results.round(4).to_string(index=False))

peak_results.to_csv(
    OUT / "peak_detection_metrics.csv",
    index=False
)


# ==========================================
# 7. Save prediction results
# ==========================================

np.savez_compressed(
    OUT / "test_predictions.npz",
    history=history,
    actual=actual,
    predicted=predicted,
    future_peak=future_peak,
    past_peak=past_peak
)


# ==========================================
# 8. Plot 24h forecasts
# ==========================================

# Select representative examples automatically
errors = np.mean(
    np.abs(predicted - actual),
    axis=1
)

best_idx = int(np.argmin(errors))
worst_idx = int(np.argmax(errors))

peak_indices = np.flatnonzero(future_peak)

if len(peak_indices) > 0:
    peak_idx = int(peak_indices[len(peak_indices) // 2])
else:
    peak_idx = best_idx

examples = {
    "best_case": best_idx,
    "worst_case": worst_idx,
    "peak_case": peak_idx
}

for name, idx in examples.items():

    plt.figure(figsize=(11, 4))

    hours = np.arange(1, PRED_LEN + 1)

    plt.plot(
        hours,
        actual[idx],
        label="Actual",
        linewidth=2
    )

    plt.plot(
        hours,
        predicted[idx],
        label="Transformer V2",
        linewidth=2
    )

    plt.axhline(
        PEAK_KW,
        color="gray",
        linestyle="--",
        label="Peak threshold"
    )

    plt.xlabel("Forecast horizon (hour)")
    plt.ylabel("Load (kW)")
    plt.title(name.replace("_", " ").title())
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT / f"{name}.png",
        dpi=180
    )

    plt.close()


# ==========================================
# 9. Error distribution
# ==========================================

errors_flat = y_pred - y_true

plt.figure(figsize=(9, 5))

plt.hist(
    errors_flat,
    bins=70,
    alpha=0.85
)

plt.axvline(
    0,
    color="black",
    linestyle="--"
)

plt.xlabel("Prediction error (kW)")
plt.ylabel("Count")
plt.title("Transformer V2 Test Error Distribution")
plt.tight_layout()

plt.savefig(
    OUT / "error_distribution.png",
    dpi=180
)

plt.close()


# ==========================================
# 10. Summary
# ==========================================

summary = pd.DataFrame([{
    "model": "Transformer V2",
    "test_mae_kw": mae,
    "test_rmse_kw": rmse,
    "test_bias_kw": bias,
    "test_windows": len(actual)
}])

summary.to_csv(
    OUT / "summary.csv",
    index=False
)

print("\n===== Files Generated =====")

for file in sorted(OUT.iterdir()):
    print(file.name)

print("\nFinal evaluation completed!")
