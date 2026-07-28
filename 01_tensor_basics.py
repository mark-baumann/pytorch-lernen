"""
PyTorch von Grund auf — Tensor-Basics
======================================
Das Fundament: Tensoren, Operationen, GPU, Autograd.
"""

import torch
import numpy as np


def tensor_basics():
    """Die wichtigsten Tensor-Operationen."""
    print("=" * 50)
    print("  1. Tensor-Basics")
    print("=" * 50)

    # ── Erstellung ───────────────────────────────────────────
    print("\n📦 Tensoren erstellen:")

    # Aus Liste
    t = torch.tensor([1, 2, 3, 4])
    print(f"  torch.tensor([1,2,3,4])     → {t}")

    # Nullen, Einsen, Zufall
    print(f"  torch.zeros(2,3)            → \n{torch.zeros(2,3)}")
    print(f"  torch.ones(2,3)             → \n{torch.ones(2,3)}")
    print(f"  torch.randn(2,3)            → \n{torch.randn(2,3)}")

    # Aus NumPy
    np_arr = np.array([1.0, 2.0, 3.0])
    t_np = torch.from_numpy(np_arr)
    print(f"  torch.from_numpy(np.array)  → {t_np}")

    # ── Attribute ────────────────────────────────────────────
    print("\n📋 Tensor-Attribute:")
    x = torch.randn(2, 3, 4)
    print(f"  shape:    {x.shape}")
    print(f"  dtype:    {x.dtype}")
    print(f"  device:   {x.device}")
    print(f"  ndim:     {x.ndim}")

    # ── Operationen ─────────────────────────────────────────
    print("\n🔧 Grundoperationen:")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    print(f"  a + b     = {a + b}")
    print(f"  a * b     = {a * b}        (elementweise!)")
    print(f"  a @ b     = {a @ b:.1f}    (Skalarprodukt)")
    print(f"  a.mean()  = {a.mean():.1f}")
    print(f"  a.sum()   = {a.sum():.1f}")

    # ── Reshape & Indexing ──────────────────────────────────
    print("\n📐 Reshape & Indexing:")
    x = torch.arange(12)
    print(f"  arange(12)           → {x}")
    print(f"  .reshape(3,4)        → \n{x.reshape(3,4)}")
    print(f"  .view(4,3)           → \n{x.view(4,3)}")
    print(f"  x[2:5]               → {x[2:5]}")

    # ── GPU ──────────────────────────────────────────────────
    print("\n🖥️  GPU-Verfügbarkeit:")
    print(f"  CUDA verfügbar:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:             {torch.cuda.get_device_name(0)}")
        gpu_t = torch.randn(1000, 1000).cuda()
        print(f"  Tensor auf GPU:  {gpu_t.device}")


def autograd_demo():
    """Autograd: Automatische Differentiation."""
    print("\n" + "=" * 50)
    print("  2. Autograd — Automatische Gradienten")
    print("=" * 50)

    # ── Einfaches Beispiel ──────────────────────────────────
    print("\n🎯 f(x) = x² bei x=3:")

    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()
    print(f"  x = {x.item()}, y = x² = {y.item()}")
    print(f"  dy/dx = 2x = {x.grad.item()}  (erwartet: 6.0) ✓")

    # ── Kettenregel ─────────────────────────────────────────
    print("\n⛓️  Kettenregel: f(x) = (2x + 1)²")

    x = torch.tensor(2.0, requires_grad=True)
    y = (2 * x + 1) ** 2
    y.backward()
    print(f"  x = {x.item()}, y = (2x+1)² = {y.item()}")
    print(f"  dy/dx = 4(2x+1) = {x.grad.item():.1f}  (erwartet: 20.0) ✓")

    # ── Mehrere Variablen ───────────────────────────────────
    print("\n📊 Mehrere Variablen: z = x² + y³")

    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = x**2 + y**3
    z.backward()
    print(f"  x={x.item()}, y={y.item()}, z = {z.item()}")
    print(f"  ∂z/∂x = 2x = {x.grad.item():.1f}  (erwartet: 4.0) ✓")
    print(f"  ∂z/∂y = 3y² = {y.grad.item():.1f}  (erwartet: 27.0) ✓")

    # ── Gradienten akkumulieren ─────────────────────────────
    print("\n⚠️  Achtung: Gradienten akkumulieren!")
    print("  optimizer.zero_grad() nicht vergessen!")

    w = torch.tensor(2.0, requires_grad=True)
    for i in range(3):
        loss = w ** 2
        loss.backward()
        print(f"  Schritt {i+1}: w.grad = {w.grad.item():.1f}  "
              f"(akkumuliert: {2*w.item()*(i+1):.1f})")


def linear_regression_torch():
    """Lineare Regression mit PyTorch — erstes Training."""
    print("\n" + "=" * 50)
    print("  3. Lineare Regression mit PyTorch")
    print("=" * 50)

    # ── Daten generieren ────────────────────────────────────
    torch.manual_seed(42)
    N = 100
    X = torch.randn(N, 1) * 2
    y = 3 * X + 2 + torch.randn(N, 1) * 0.5  # y = 3x + 2 + noise

    print(f"\n📊 Daten: {N} Punkte, y = 3x + 2 + noise")

    # ── Modell ───────────────────────────────────────────────
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    lr = 0.01
    epochs = 100

    print(f"\n🏋️  Training ({epochs} Epochen, lr={lr}):")
    for epoch in range(epochs):
        # Forward
        y_pred = X @ w + b
        loss = ((y_pred - y) ** 2).mean()

        # Backward
        loss.backward()

        # Update (manuell, ohne Optimizer)
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()

        if epoch % 20 == 0:
            print(f"  Epoche {epoch:3d}: loss={loss.item():.4f}, "
                  f"w={w.item():.3f}, b={b.item():.3f}")

    print(f"\n✅ Gefunden: w={w.item():.3f}, b={b.item():.3f}")
    print(f"   Erwartet:  w=3.0, b=2.0")


if __name__ == "__main__":
    tensor_basics()
    autograd_demo()
    linear_regression_torch()
