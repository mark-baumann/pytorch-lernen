"""Tests für 03_cnn_mnist.py — CNN mit BatchNorm."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lernen.models import MNIST_CNN
from pytorch_lernen.training import train_epoch


class TestMNISTCNN:
    def test_forward_shape(self):
        model = MNIST_CNN()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_batchnorm_running_stats(self):
        """BatchNorm sollte running_mean/running_var während Training aktualisieren."""
        model = MNIST_CNN()
        model.train()

        # Initial running_mean should be 0
        initial_mean = model.bn1.running_mean.clone()

        X = torch.randn(32, 1, 28, 28)
        _ = model(X)

        # running_mean should have changed
        assert not torch.allclose(model.bn1.running_mean, initial_mean)

    def test_training_reduces_loss(self):
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = MNIST_CNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        X = torch.randn(64, 1, 28, 28)
        y = torch.randint(0, 10, (64,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)

        loss_before, _ = train_epoch(model, loader, optimizer, criterion, device)
        loss_after, _ = train_epoch(model, loader, optimizer, criterion, device)
        assert loss_after < loss_before

    def test_eval_mode(self):
        model = MNIST_CNN()
        model.eval()
        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 10)
