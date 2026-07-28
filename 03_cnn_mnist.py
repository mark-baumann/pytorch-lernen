#!/usr/bin/env python3
"""
03_cnn_mnist.py – CNN mit Conv2d, MaxPool, BatchNorm für MNIST
===============================================================
Architektur: 2 Conv-Blöcke (Conv2d → BatchNorm2d → ReLU → MaxPool2d)
             → Flatten → 2 Fully-Connected Layer → Dropout → Output (10 Klassen)

Training: 5 Epochen, CrossEntropyLoss, Adam-Optimizer
Evaluation: Accuracy auf Test-Set
Visualisierung: Loss-Kurve, Accuracy-Kurve, Beispiel-Vorhersagen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Reproduzierbarkeit ──────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Gerät ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {DEVICE}")

# ── Hyperparameter ──────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

# ── Datenvorbereitung ───────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Trainingsdaten: {len(train_dataset)} Bilder")
print(f"Testdaten:      {len(test_dataset)} Bilder")


# ── CNN-Modell ──────────────────────────────────────────────────────────────
class MNIST_CNN(nn.Module):
    """CNN für MNIST mit Conv2d, BatchNorm, MaxPool, Dropout."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Block 1: 1×28×28 → 32×14×14
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 32×14×14 → 64×7×7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Fully Connected
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Flatten
        x = x.view(x.size(0), -1)
        # FC
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = MNIST_CNN(num_classes=10).to(DEVICE)
print(f"\nModell:\n{model}")
print(f"Parameter: {sum(p.numel() for p in model.parameters()):,}")

# ── Loss & Optimizer ────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ── Training ────────────────────────────────────────────────────────────────
train_losses: list[float] = []
train_accs: list[float] = []
test_accs: list[float] = []


def train_epoch() -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def evaluate() -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


print("\n── Training ──")
for epoch in range(1, EPOCHS + 1):
    loss, train_acc = train_epoch()
    test_acc = evaluate()

    train_losses.append(loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

print(f"\nFinale Test-Accuracy: {test_accs[-1]:.2f}%")

# ── Visualisierung ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Loss
axes[0].plot(range(1, EPOCHS + 1), train_losses, "b-o", linewidth=2)
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(range(1, EPOCHS + 1), train_accs, "g-o", label="Train", linewidth=2)
axes[1].plot(range(1, EPOCHS + 1), test_accs, "r-s", label="Test", linewidth=2)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Beispiel-Vorhersagen
model.eval()
data_iter = iter(test_loader)
images, labels = next(data_iter)
images, labels = images[:10].to(DEVICE), labels[:10]

with torch.no_grad():
    outputs = model(images)
    _, preds = outputs.max(1)

for i in range(10):
    ax = axes[2] if i == 0 else axes[2]  # hack: show first 10 in subplot grid
    # We'll use a separate figure for predictions

# Besser: eigenes Figure für Predictions
fig2, axes2 = plt.subplots(2, 5, figsize=(12, 5))
axes2 = axes2.flatten()
for i in range(10):
    img = images[i].cpu().squeeze()
    axes2[i].imshow(img, cmap="gray")
    color = "green" if preds[i] == labels[i] else "red"
    axes2[i].set_title(f"Pred: {preds[i].item()} (True: {labels[i].item()})",
                       color=color, fontsize=10)
    axes2[i].axis("off")
fig2.suptitle("Beispiel-Vorhersagen (grün = korrekt, rot = falsch)", fontsize=13)
fig2.tight_layout()

# Speichern
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
fig.savefig(output_dir / "03_cnn_mnist_metrics.png", dpi=150, bbox_inches="tight")
fig2.savefig(output_dir / "03_cnn_mnist_predictions.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nPlots gespeichert unter: {output_dir}/")
print("03_cnn_mnist.py – Fertig!")
