"""Tests für 01_tensor_basics.py — Tensor-Operationen und Autograd."""

import numpy as np
import torch


class TestTensorBasics:
    def test_tensor_creation(self):
        t = torch.tensor([1, 2, 3, 4])
        assert t.shape == (4,)
        assert t.dtype == torch.int64

    def test_zeros_ones(self):
        z = torch.zeros(2, 3)
        o = torch.ones(2, 3)
        assert z.shape == (2, 3)
        assert o.shape == (2, 3)
        assert z.sum() == 0.0
        assert o.sum() == 6.0

    def test_from_numpy(self):
        np_arr = np.array([1.0, 2.0, 3.0])
        t = torch.from_numpy(np_arr)
        # from_numpy erzeugt float64, torch.tensor standardmäßig float32
        assert torch.allclose(t, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    def test_operations(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        assert torch.allclose(a + b, torch.tensor([5.0, 7.0, 9.0]))
        assert torch.allclose(a * b, torch.tensor([4.0, 10.0, 18.0]))
        assert abs((a @ b).item() - 32.0) < 1e-5

    def test_reshape(self):
        x = torch.arange(12)
        r = x.reshape(3, 4)
        assert r.shape == (3, 4)
        assert r[0, 0] == 0
        assert r[2, 3] == 11

    def test_indexing(self):
        x = torch.arange(12)
        assert torch.equal(x[2:5], torch.tensor([2, 3, 4]))


class TestAutograd:
    def test_simple_gradient(self):
        x = torch.tensor(3.0, requires_grad=True)
        y = x ** 2
        y.backward()
        assert abs(x.grad.item() - 6.0) < 1e-5

    def test_chain_rule(self):
        x = torch.tensor(2.0, requires_grad=True)
        y = (2 * x + 1) ** 2
        y.backward()
        # dy/dx = 4(2x+1) = 4*5 = 20
        assert abs(x.grad.item() - 20.0) < 1e-5

    def test_multivariable(self):
        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(3.0, requires_grad=True)
        z = x**2 + y**3
        z.backward()
        assert abs(x.grad.item() - 4.0) < 1e-5
        assert abs(y.grad.item() - 27.0) < 1e-5

    def test_gradient_accumulation(self):
        w = torch.tensor(2.0, requires_grad=True)
        for _ in range(3):
            loss = w ** 2
            loss.backward()
        # After 3 steps: grad = 2*w * 3 = 12
        assert abs(w.grad.item() - 12.0) < 1e-5


class TestLinearRegression:
    def test_manual_training(self):
        torch.manual_seed(42)
        N = 100
        X = torch.randn(N, 1) * 2
        y_true = 3 * X.squeeze() + 2 + torch.randn(N) * 0.5  # 1D

        w = torch.randn(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        lr = 0.01

        for _ in range(500):
            y_pred = X.squeeze() * w + b  # beide 1D → (N,)
            loss = ((y_pred - y_true) ** 2).mean()
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
                w.grad.zero_()
                b.grad.zero_()

        # Should be close to w=3, b=2
        assert abs(w.item() - 3.0) < 0.5, f"w={w.item():.3f}"
        assert abs(b.item() - 2.0) < 0.5, f"b={b.item():.3f}"
