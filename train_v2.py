
from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_v2 import train_loader, val_loader, mean, std
from model_v2 import LoadTransformerV2


# 1. Configuration
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 8

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# 2. Initialize model
model = LoadTransformerV2().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Reduce LR when validation loss stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)


# 3. Training function
def train_one_epoch():
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        predictions = model(x)
        loss = criterion(predictions, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / total_samples


# 4. Validation function
@torch.no_grad()
def validate():
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        predictions = model(x)
        loss = criterion(predictions, y)

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / total_samples


# 5. Training loop
train_losses = []
val_losses = []

best_val_loss = float("inf")
best_epoch = 0
wait = 0

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

for epoch in range(EPOCHS):

    train_loss = train_one_epoch()
    val_loss = validate()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train: {train_loss:.6f} "
        f"Val: {val_loss:.6f} "
        f"LR: {current_lr:.6f}",
        flush=True
    )

    # Save best model
    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        wait = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "mean": float(mean),
                "std": float(std),
                "epoch": best_epoch,
                "val_loss": best_val_loss
            },
            "models/best_transformer_v2.pth"
        )

        print("  Best model saved!", flush=True)

    else:
        wait += 1

    # Early stopping
    if wait >= PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch+1}"
        )
        break


# 6. Save history
history = {
    "train_loss": train_losses,
    "val_loss": val_losses,
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss
}

with open("outputs/training_history_v2.json", "w") as f:
    json.dump(history, f, indent=2)


# 7. Plot training curves
plt.figure(figsize=(10, 5))

epochs = range(1, len(train_losses) + 1)

plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")

plt.axvline(
    best_epoch,
    color="gray",
    linestyle="--",
    label=f"Best Epoch: {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Transformer V2 Training History")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/training_loss_v2.png",
    dpi=150
)

plt.close()

print("\n===== Training Completed =====")
print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)
print("Total epochs trained:", len(train_losses))
