
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# 1. Configuration
INPUT_LEN = 168
PRED_LEN = 24
BATCH_SIZE = 32

BASE_DIR = Path(__file__).resolve().parent

# 2. Load original hourly data
df = pd.read_csv(
    BASE_DIR / "data" / "hourly_load.csv",
    parse_dates=["timestamp"],
    index_col="timestamp"
)

load = df["load_kw"].to_numpy(dtype=np.float32)
dates = df.index

# 3. Same chronological split boundaries as V1
train_end = int(len(load) * 0.70)
val_end = int(len(load) * 0.85)

# 4. Normalize load using training data only
mean = float(load[:train_end].mean())
std = float(load[:train_end].std())

if std <= 0:
    raise ValueError("Invalid training standard deviation")

load_scaled = (load - mean) / std

# 5. Cyclical time features
hour = dates.hour.to_numpy()
weekday = dates.dayofweek.to_numpy()

hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

week_sin = np.sin(2 * np.pi * weekday / 7)
week_cos = np.cos(2 * np.pi * weekday / 7)

# 6. Combine five features
features = np.column_stack([
    load_scaled,
    hour_sin,
    hour_cos,
    week_sin,
    week_cos
]).astype(np.float32)


# 7. Sliding-window dataset
class LoadDatasetV2(Dataset):
    def __init__(self, features, targets,
                 start, end,
                 input_len=INPUT_LEN,
                 pred_len=PRED_LEN):

        self.features = features
        self.targets = targets
        self.input_len = input_len
        self.pred_len = pred_len

        # Inputs may use past history from previous split.
        # Targets remain strictly inside the current split.
        self.first = max(0, start - input_len)
        self.last = end - input_len - pred_len

        if self.last < self.first:
            raise ValueError("Not enough data")

    def __len__(self):
        return self.last - self.first + 1

    def __getitem__(self, idx):
        i = self.first + idx

        x = self.features[
            i:i + self.input_len
        ]

        y = self.targets[
            i + self.input_len:
            i + self.input_len + self.pred_len
        ]

        return (
            torch.from_numpy(x.copy()),
            torch.from_numpy(y.copy())
        )


# 8. Create datasets
train_dataset = LoadDatasetV2(
    features, load_scaled, 0, train_end
)

val_dataset = LoadDatasetV2(
    features, load_scaled, train_end, val_end
)

test_dataset = LoadDatasetV2(
    features, load_scaled, val_end, len(load)
)


# 9. DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# 10. Test data shapes
if __name__ == "__main__":
    x, y = next(iter(train_loader))

    print("Total hours:", len(load))
    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    print("\nInput shape:", x.shape)
    print("Target shape:", y.shape)

    print("\nTraining mean:", mean)
    print("Training std:", std)
    print("\nFirst input feature vector:")
    print(x[0, 0])
