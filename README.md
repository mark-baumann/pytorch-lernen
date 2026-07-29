# PyTorch lernen — von Grund auf bis Reinforcement Learning

Praktische PyTorch-Beispiele, vom Tensor bis zum DQN-Agenten.

## 📁 Struktur

| Datei | Thema |
|-------|-------|
| `01_tensor_basics.py` | Tensoren, Autograd, Lineare Regression |
| `02_neural_network.py` | MLP & CNN für MNIST-Klassifikation |
| `03_cnn_mnist.py` | CNN mit BatchNorm, Dropout, Visualisierung |
| `04_transfer_learning.py` | ResNet18 Fine-Tuning auf CIFAR-10 |
| `05_rl_cartpole.py` | DQN für CartPole-v1 (Gymnasium) |
| `06_rl_gridworld.py` | DQN für GridWorld (mit Tabular-Vergleich) |

## 🚀 Quickstart

```bash
# Virtual Environment & Abhängigkeiten
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Einzelne Skripte ausführen
python 01_tensor_basics.py
python 02_neural_network.py
# ... etc.

# Tests ausführen
pytest tests/ -v
```

## 🧪 Tests

```bash
pytest tests/ -v --tb=short
```

## 📦 Abhängigkeiten

- PyTorch (CPU oder CUDA)
- torchvision
- matplotlib
- numpy
- gymnasium (für RL)
- pytest (für Tests)

## 📝 Lizenz

MIT — frei verwenden, lernen, verbessern.
