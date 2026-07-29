"""Tests für pytorch_lernen.models."""

import torch
from pytorch_lernen.models import (
    SimpleMLP, SimpleCNN, MNIST_CNN, count_parameters
)


class TestSimpleMLP:
    def test_forward_shape(self):
        model = SimpleMLP()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_forward_flat_input(self):
        model = SimpleMLP()
        x = torch.randn(4, 784)
        out = model(x)
        assert out.shape == (4, 10)

    def test_custom_params(self):
        model = SimpleMLP(input_dim=100, hidden_dims=(50, 25), num_classes=5)
        x = torch.randn(2, 100)
        out = model(x)
        assert out.shape == (2, 5)

    def test_has_parameters(self):
        model = SimpleMLP()
        n = count_parameters(model)
        assert n > 0


class TestSimpleCNN:
    def test_forward_shape(self):
        model = SimpleCNN()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_custom_classes(self):
        model = SimpleCNN(num_classes=5)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 5)

    def test_has_parameters(self):
        model = SimpleCNN()
        n = count_parameters(model)
        assert n > 0


class TestMNIST_CNN:
    def test_forward_shape(self):
        model = MNIST_CNN()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_batchnorm_training(self):
        model = MNIST_CNN()
        model.train()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_batchnorm_eval(self):
        model = MNIST_CNN()
        model.eval()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)


class TestCountParameters:
    def test_mlp_params(self):
        model = SimpleMLP()
        n = count_parameters(model)
        # 784*128 + 128 + 128*64 + 64 + 64*10 + 10
        expected = 784 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10
        assert n == expected

    def test_frozen_params(self):
        model = SimpleMLP()
        for p in model.fc1.parameters():
            p.requires_grad = False
        n = count_parameters(model)
        # fc1 params excluded
        expected = 128 * 64 + 64 + 64 * 10 + 10
        assert n == expected
