"""
PyTorch Neuronales Netz — MNIST-Klassifikation
===============================================
Vollständiges Training mit:
- torch.nn.Module
- DataLoader & Datasets
- Optimizer & Loss-Funktionen
- Train/Test-Loop
- GPU-Support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np


# ═══════════════════════════════════════════════════════════════
# 1. Einfaches MLP
# ═══════════════════════════════════════════════════════════════

class SimpleMLP(nn.Module):
    """
    Gleiche Architektur wie unser NumPy-NN:
    784 → 128 → 64 → 10
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten: (N, 28, 28) → (N, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)  # Logits (kein Softmax — in CrossEntropyLoss)


# ═══════════════════════════════════════════════════════════════
# 2. CNN (Convolutional Neural Network)
# ═══════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    """
    CNN für MNIST:
    Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → FC → FC
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Nach 2x Pooling: 28→14→7, 64 Kanäle
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x: (N, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # → (N, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # → (N, 64, 7, 7)
        x = x.view(x.size(0), -1)             # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ═══════════════════════════════════════════════════════════════
# 3. Training & Evaluation
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    """Eine Trainings-Epoche."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluation auf Testdaten."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(loader), correct / total


def count_parameters(model):
    """Zählt trainierbare Parameter."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════
# 4. Hauptprogramm
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  PyTorch Neuronales Netz — MNIST")
    print("=" * 60)

    # ── Device ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ── Daten ────────────────────────────────────────────────
    print("\n📦 Lade MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])

    train_data = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    print(f"   Train: {len(train_data):,} | Test: {len(test_data):,}")

    # ── Modelle vergleichen ─────────────────────────────────
    models = {
        "MLP (784→128→64→10)": SimpleMLP(),
        "CNN (2xConv+2xFC)": SimpleCNN(),
    }

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        model = model.to(device)
        print(f"   Parameter: {count_parameters(model):,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        epochs = 5

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device
            )
            print(f"   Epoche {epoch+1}: "
                  f"Train Loss={train_loss:.4f} Acc={train_acc:.3f} | "
                  f"Test Loss={test_loss:.4f} Acc={test_acc:.3f}")

    print("\n✅ Training abgeschlossen!")


if __name__ == "__main__":
    main()
