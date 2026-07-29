#!/usr/bin/env python3
"""
04_transfer_learning.py – ResNet18 Fine-Tuning auf CIFAR-10
============================================================
Ansatz: Vortrainiertes ResNet18 (ImageNet) laden, letztes FC-Layer ersetzen.
        Erst alle Layer einfrieren, nur den neuen Kopf trainieren (5 Epochen).
        Dann alle Layer auftauen und feintunen (5 Epochen).

Training: 10 Epochen gesamt, CrossEntropyLoss, Adam-Optimizer
Evaluation: Accuracy auf Test-Set
Visualisierung: Loss/Accuracy-Kurven, Beispiel-Vorhersagen mit Klassen-Namen
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
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
BATCH_SIZE = 32
HEAD_EPOCHS = 2       # Nur den neuen Kopf trainieren
FINETUNE_EPOCHS = 2   # Alle Layer feintunen
LEARNING_RATE_HEAD = 0.001
LEARNING_RATE_FINETUNE = 0.0001

# ── CIFAR-10 Klassen ────────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── Datenvorbereitung ───────────────────────────────────────────────────────
# ResNet18 erwartet 224×224 Input – CIFAR-10 ist 32×32, also hochskalieren
# Für CPU-Training nutzen wir 128×128 als Kompromiss
train_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)

# Für CPU-Training: Subset verwenden (5000 Train, 1000 Test)
from torch.utils.data import Subset
train_indices = list(range(5000))
test_indices = list(range(1000))
train_dataset = Subset(train_dataset, train_indices)
test_dataset = Subset(test_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0)

print(f"Trainingsdaten: {len(train_dataset)} Bilder")
print(f"Testdaten:      {len(test_dataset)} Bilder")

# ── Modell ──────────────────────────────────────────────────────────────────
# Vortrainiertes ResNet18 laden
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Alle Layer einfrieren
for param in model.parameters():
    param.requires_grad = False

# Letztes FC-Layer ersetzen (512 → 10)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

model = model.to(DEVICE)
print(f"\nModell: ResNet18 (pretrained)")
print(f"Trainierbare Parameter: {sum(p.requires_grad for p in model.parameters()):,}")

# ── Loss ────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

# ── Training & Evaluation ───────────────────────────────────────────────────
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


# ── Phase 1: Nur den Kopf trainieren ───────────────────────────────────────
print("\n── Phase 1: Kopf-Training (nur FC-Layer) ──")
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE_HEAD)

for epoch in range(1, HEAD_EPOCHS + 1):
    loss, train_acc = train_epoch()
    test_acc = evaluate()

    train_losses.append(loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch:2d}/{HEAD_EPOCHS} | Loss: {loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

# ── Phase 2: Alle Layer feintunen ──────────────────────────────────────────
print("\n── Phase 2: Fine-Tuning (alle Layer) ──")
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_FINETUNE)

for epoch in range(1, FINETUNE_EPOCHS + 1):
    loss, train_acc = train_epoch()
    test_acc = evaluate()

    train_losses.append(loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch:2d}/{FINETUNE_EPOCHS} | Loss: {loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

print(f"\nFinale Test-Accuracy: {test_accs[-1]:.2f}%")

# ── Visualisierung ──────────────────────────────────────────────────────────
total_epochs = HEAD_EPOCHS + FINETUNE_EPOCHS
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(range(1, total_epochs + 1), train_losses, "b-o", linewidth=2)
axes[0].axvline(x=HEAD_EPOCHS + 0.5, color="gray", linestyle="--",
                label="Fine-Tuning Start")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(range(1, total_epochs + 1), train_accs, "g-o",
             label="Train", linewidth=2)
axes[1].plot(range(1, total_epochs + 1), test_accs, "r-s",
             label="Test", linewidth=2)
axes[1].axvline(x=HEAD_EPOCHS + 0.5, color="gray", linestyle="--",
                label="Fine-Tuning Start")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()

# Beispiel-Vorhersagen
model.eval()
data_iter = iter(test_loader)
images, labels = next(data_iter)
images, labels = images[:10].to(DEVICE), labels[:10]

with torch.no_grad():
    outputs = model(images)
    _, preds = outputs.max(1)

# Denormalize für Anzeige
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

fig2, axes2 = plt.subplots(2, 5, figsize=(14, 6))
axes2 = axes2.flatten()
for i in range(10):
    img = images[i].cpu() * std + mean  # denormalize
    img = img.clamp(0, 1).permute(1, 2, 0)
    axes2[i].imshow(img)
    true_name = CIFAR10_CLASSES[labels[i].item()]
    pred_name = CIFAR10_CLASSES[preds[i].item()]
    color = "green" if preds[i] == labels[i] else "red"
    axes2[i].set_title(f"Pred: {pred_name}\nTrue: {true_name}",
                       color=color, fontsize=9)
    axes2[i].axis("off")
fig2.suptitle("Beispiel-Vorhersagen (grün = korrekt, rot = falsch)", fontsize=13)
fig2.tight_layout()

# Speichern
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
fig.savefig(output_dir / "04_transfer_learning_metrics.png", dpi=150,
            bbox_inches="tight")
fig2.savefig(output_dir / "04_transfer_learning_predictions.png", dpi=150,
             bbox_inches="tight")
plt.show()

print(f"\nPlots gespeichert unter: {output_dir}/")
print("04_transfer_learning.py – Fertig!")
