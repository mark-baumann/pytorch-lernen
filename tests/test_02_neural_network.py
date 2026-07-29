"""Tests für 02_neural_network.py — MLP & CNN auf MNIST."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lernen.models import SimpleMLP, SimpleCNN, count_parameters
from pytorch_lernen.training import train_epoch, evaluate


class TestSimpleMLP:
    def test_forward_shape(self):
        model = SimpleMLP()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_forward_flat(self):
        model = SimpleMLP()
        x = torch.randn(4, 784)
        out = model(x)
        assert out.shape == (4, 10)

    def test_training_step(self):
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = SimpleMLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        X = torch.randn(64, 1, 28, 28)
        y = torch.randint(0, 10, (64,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)

        loss_before, _ = train_epoch(model, loader, optimizer, criterion, device)
        loss_after, _ = train_epoch(model, loader, optimizer, criterion, device)
        assert loss_after < loss_before


class TestSimpleCNN:
    def test_forward_shape(self):
        model = SimpleCNN()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_training_step(self):
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = SimpleCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        X = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)

        loss_before, _ = train_epoch(model, loader, optimizer, criterion, device)
        loss_after, _ = train_epoch(model, loader, optimizer, criterion, device)
        assert loss_after < loss_before


class TestCountParameters:
    def test_mlp(self):
        model = SimpleMLP()
        n = count_parameters(model)
        assert n > 100_000

    def test_cnn(self):
        model = SimpleCNN()
        n = count_parameters(model)
        assert n > 100_000
