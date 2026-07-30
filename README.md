# 🔥 PyTorch Lernen — Von Grund auf bis Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Aktiv-brightgreen.svg)]()

Interaktive Lernumgebung für **PyTorch** — von den Grundlagen der Tensor-Operationen über automatische Differentiation (Autograd) und Convolutional Neural Networks bis hin zu Transfer Learning und Deep Q-Networks (DQN) für Reinforcement Learning. Alle Konzepte werden direkt in einer Streamlit-App visualisiert und sind sofort ausführbar.

## ✨ Features

- **📦 Tensor-Basics** — Erstellung, Operationen, GPU-Unterstützung und Broadcasting interaktiv erkunden
- **⚡ Autograd** — Automatische Differentiation live nachvollziehen: Forward- und Backward-Pass
- **🖼️ CNN-MNIST** — Convolutional Neural Network auf MNIST trainieren und Filter visualisieren
- **🔁 Transfer Learning** — Vortrainierte Modelle (ResNet) auf eigene Daten anpassen
- **🎮 DQN-CartPole** — Deep Q-Network für Reinforcement Learning auf CartPole
- **📊 W&B-Integration** — Experiment-Tracking mit Weights & Biases
- **✅ Vollständige Testabdeckung** — Unit-Tests für alle Module

## 🚀 Installation

```bash
# Repository klonen
git clone https://github.com/mark-baumann/pytorch-lernen.git
cd pytorch-lernen

# Virtuelle Umgebung erstellen
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Abhängigkeiten installieren
uv pip install -e ".[dev]"
```

## 🎯 Nutzung

```bash
# Streamlit-App starten
streamlit run app.py
```

Die App öffnet sich im Browser unter `http://localhost:8501`. Wähle ein Modul aus der Seitenleiste und experimentiere interaktiv mit den PyTorch-Konzepten.

## 🧪 Tests ausführen

```bash
pytest tests/ -v
```

## 🛠️ Tech-Stack

| Technologie | Einsatz |
|-------------|---------|
| **PyTorch** | Deep-Learning-Framework (Tensoren, Autograd, CNNs) |
| **Torchvision** | Vortrainierte Modelle & Datasets |
| **Gymnasium** | Reinforcement-Learning-Umgebungen |
| **Streamlit** | Interaktive Web-App |
| **Matplotlib** | Visualisierung von Trainingsverläufen |
| **Weights & Biases** | Experiment-Tracking |
| **Pytest** | Test-Framework |

## 📁 Projektstruktur

```
pytorch-lernen/
├── app.py                          # Streamlit-Hauptapp
├── pyproject.toml                  # Projekt-Konfiguration
├── pytorch_lernen/
│   ├── __init__.py
│   ├── models.py                   # CNN-Modelldefinitionen
│   ├── training.py                 # Trainings-Loop
│   └── rl_common.py                # RL-Hilfsfunktionen
├── 01_tensor_basics.py             # Tensor-Grundlagen
├── 02_neural_network.py            # Neuronales Netz mit PyTorch
├── 03_cnn_mnist.py                 # CNN auf MNIST
├── 04_transfer_learning.py         # Transfer Learning
├── 05_rl_cartpole.py               # DQN auf CartPole
├── 06_rl_gridworld.py              # DQN auf GridWorld
└── tests/                          # Unit-Tests
    ├── test_01_tensor_basics.py
    ├── test_02_neural_network.py
    ├── test_03_cnn_mnist.py
    ├── test_04_transfer_learning.py
    ├── test_05_rl_cartpole.py
    ├── test_06_rl_gridworld.py
    ├── test_models.py
    ├── test_training.py
    └── test_rl_common.py
```

## 👤 Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann)

---

*Dieses Projekt dient als Lernressource für PyTorch. Die Konzepte sind didaktisch aufbereitet und eignen sich für Einsteiger und Fortgeschrittene.*
