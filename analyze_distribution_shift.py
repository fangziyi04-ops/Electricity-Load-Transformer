
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. Configuration
# ==========================================

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs" / "distribution_shift"
OUT.mkdir(parents=True, exist_ok=True)

PEAK_KW = 15.0
INPUT_LEN = 168
PRED_LEN = 24

df = pd.read_csv(
    BASE / "data" / "hourly_load.csv",
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)

load = df["load_kw"].to_numpy(dtype=np.float32)

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

splits = {
    "Train": (0, train_end),
    "Validation": (train_end, val_end),
    "Test": (val_end, len(df))
}


# ==========================================
# 2. Load distribution statistics
# ==========================================

rows = []

for name, (start, end) in splits.items():

    values = load[start:end]

    rows.append({
        "split": name,
        "hours": len(values),
        "mean_kw": values.mean(),
        "std_kw": values.std(),
        "median_kw": np.median(values),
        "p95_kw": np.percentile(values, 95),
        "max_kw": values.max(),
        "peak_hours": int((values >= PEAK_KW).sum()),
        "peak_ratio": (values >= PEAK_KW).mean()
    })

distribution_df = pd.DataFrame(rows)

print("\n===== Dataset Distribution =====")
print(distribution_df.round(4).to_string(index=False))

distribution_df.to_csv(
    OUT / "split_distribution.csv",
    index=False
)


# ==========================================
# 3. Continuous peak events
# ==========================================

def get_peak_events(values, threshold):

    peak = values >= threshold

    padded = np.r_[
        False,
        peak,
        False
    ]

    changes = np.diff(padded.astype(int))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    durations = ends - starts

    return durations


event_rows = []

for name, (start, end) in splits.items():

    durations = get_peak_events(
        load[start:end],
        PEAK_KW
    )

    event_rows.append({
        "split": name,
        "event_count": len(durations),
        "mean_duration_h": (
            durations.mean() if len(durations) else 0
        ),
        "median_duration_h": (
            np.median(durations) if len(durations) else 0
        ),
        "max_duration_h": (
            durations.max() if len(durations) else 0
        ),
        "events_ge_6h": int(
            (durations >= 6).sum()
        ),
        "events_ge_12h": int(
            (durations >= 12).sum()
        )
    })

events_df = pd.DataFrame(event_rows)

print("\n===== Peak Event Duration =====")
print(events_df.round(4).to_string(index=False))

events_df.to_csv(
    OUT / "peak_event_duration.csv",
    index=False
)


# ==========================================
# 4. Window-level peak persistence
# ==========================================

persistence_rows = []

for name, (start, end) in splits.items():

    first_origin = max(start, INPUT_LEN)
    last_origin = end - PRED_LEN

    past_peaks = []
    future_peaks = []

    for origin in range(first_origin, last_origin + 1):

        past = load[origin - 24:origin]
        future = load[origin:origin + PRED_LEN]

        past_peaks.append(
            (past >= PEAK_KW).any()
        )

        future_peaks.append(
            (future >= PEAK_KW).any()
        )

    past_peaks = np.asarray(past_peaks, dtype=bool)
    future_peaks = np.asarray(future_peaks, dtype=bool)

    tp = int(np.sum(past_peaks & future_peaks))
    fp = int(np.sum(past_peaks & ~future_peaks))
    fn = int(np.sum(~past_peaks & future_peaks))
    tn = int(np.sum(~past_peaks & ~future_peaks))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    persistence_rows.append({
        "split": name,
        "windows": len(past_peaks),
        "future_peak_ratio": future_peaks.mean(),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

persistence_df = pd.DataFrame(persistence_rows)

print("\n===== Peak Persistence =====")
print(persistence_df.round(4).to_string(index=False))

persistence_df.to_csv(
    OUT / "peak_persistence.csv",
    index=False
)


# ==========================================
# 5. Monthly load statistics
# ==========================================

monthly = (
    df.set_index("timestamp")["load_kw"]
    .resample("MS")
    .agg(["mean", "max", "std"])
    .reset_index()
)

monthly.columns = [
    "month",
    "mean_kw",
    "max_kw",
    "std_kw"
]

monthly["peak_ratio"] = (
    df.set_index("timestamp")["load_kw"]
    .ge(PEAK_KW)
    .resample("MS")
    .mean()
    .to_numpy()
)

monthly.to_csv(
    OUT / "monthly_load_statistics.csv",
    index=False
)

print("\n===== Monthly Load Statistics =====")
print(monthly.round(3).to_string(index=False))


# ==========================================
# 6. Plot monthly mean load
# ==========================================

plt.figure(figsize=(13, 5))

plt.plot(
    monthly["month"],
    monthly["mean_kw"],
    marker="o",
    linewidth=1.8
)

plt.axvline(
    df.loc[train_end, "timestamp"],
    linestyle="--",
    color="gray",
    label="Train / Validation"
)

plt.axvline(
    df.loc[val_end, "timestamp"],
    linestyle=":",
    color="black",
    label="Validation / Test"
)

plt.ylabel("Monthly Mean Load (kW)")
plt.xlabel("Month")
plt.title("Monthly Electricity Load and Dataset Splits")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUT / "monthly_mean_load.png",
    dpi=180
)

plt.close()


# ==========================================
# 7. Plot load distributions
# ==========================================

plt.figure(figsize=(10, 5))

for name, (start, end) in splits.items():

    plt.hist(
        load[start:end],
        bins=70,
        density=True,
        alpha=0.35,
        label=name
    )

plt.xlabel("Load (kW)")
plt.ylabel("Density")
plt.title("Load Distribution by Dataset Split")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "load_distribution_comparison.png",
    dpi=180
)

plt.close()


# ==========================================
# 8. Plot peak persistence F1
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    persistence_df["split"],
    persistence_df["f1"]
)

plt.ylim(0, 1)
plt.ylabel("F1 Score")
plt.title("Historical Peak Rule by Dataset Split")
plt.tight_layout()

plt.savefig(
    OUT / "peak_persistence_comparison.png",
    dpi=180
)

plt.close()


# ==========================================
# 9. Done
# ==========================================

print("\n===== Generated Files =====")

for path in sorted(OUT.iterdir()):
    print(path.name)

print("\nDistribution shift analysis completed!")
