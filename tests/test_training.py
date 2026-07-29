"""Tests für pytorch_lernen.training."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lernen.training import train_epoch, evaluate
from pytorch_lernen.models import SimpleMLP


class TestTrainEpoch:
    def test_loss_decreases(self):
        """Training sollte den Loss reduzieren."""
        torch.manual_seed(42)
        device = torch.device("cpu")

        # Einfaches XOR-Problem
        X = torch.randn(200, 784)
        y = (X.sum(dim=1) > 0).long()

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32)

        model = SimpleMLP(num_classes=2).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        loss_before, _ = train_epoch(model, loader, optimizer, criterion, device)
        loss_after, _ = train_epoch(model, loader, optimizer, criterion, device)

        assert loss_after < loss_before, f"{loss_after} >= {loss_before}"

    def test_accuracy_improves(self):
        """Accuracy sollte sich verbessern."""
        torch.manual_seed(42)
        device = torch.device("cpu")

        X = torch.randn(200, 784)
        y = (X.sum(dim=1) > 0).long()

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32)

        model = SimpleMLP(num_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        _, acc_before = train_epoch(model, loader, optimizer, criterion, device)
        _, acc_after = train_epoch(model, loader, optimizer, criterion, device)

        assert acc_after >= acc_before, f"{acc_after} < {acc_before}"


class TestEvaluate:
    def test_evaluate_returns_tuple(self):
        torch.manual_seed(42)
        device = torch.device("cpu")

        X = torch.randn(50, 784)
        y = torch.zeros(50, dtype=torch.long)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)

        model = SimpleMLP(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()

        loss, acc = evaluate(model, loader, criterion, device)
        assert isinstance(loss, float)
        assert 0.0 <= acc <= 1.0

    def test_evaluate_no_criterion(self):
        torch.manual_seed(42)
        device = torch.device("cpu")

        X = torch.randn(50, 784)
        y = torch.zeros(50, dtype=torch.long)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)

        model = SimpleMLP(num_classes=2).to(device)

        loss, acc = evaluate(model, loader, None, device)
        assert loss == 0.0
        assert 0.0 <= acc <= 1.0
