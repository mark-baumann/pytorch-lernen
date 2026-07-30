"""
Streamlit-App: PyTorch Lernen — Interaktive Visualisierung
===========================================================
Tensor-Grundlagen, Autograd, CNN-Demo, Transfer-Learning, DQN-CartPole.
"""

from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn

st.set_page_config(page_title="PyTorch Lernen", page_icon="🔥", layout="wide")
st.title("🔥 PyTorch von Grund auf lernen")
st.markdown("Interaktive Visualisierung: Tensoren, Autograd, CNNs, Transfer-Learning & DQN")

# ── Seitenauswahl ──────────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Modul wählen",
    ["1. Tensor-Basics", "2. Autograd", "3. CNN-MNIST", "4. Transfer-Learning", "5. DQN-CartPole"]
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. TENSOR-BASICS
# ═══════════════════════════════════════════════════════════════════════════
if page == "1. Tensor-Basics":
    st.header("📦 Tensor-Grundlagen")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tensor erstellen")
        create_type = st.selectbox("Typ", ["torch.tensor (aus Liste)", "torch.zeros", "torch.ones", "torch.randn", "torch.arange"])
        shape_str = st.text_input("Shape (z.B. 3,4)", "3,4")

        if st.button("Tensor erstellen"):
            shape = tuple(int(x.strip()) for x in shape_str.split(","))
            if create_type == "torch.tensor (aus Liste)":
                t = torch.arange(1, np.prod(shape) + 1, dtype=torch.float32).reshape(shape)
            elif create_type == "torch.zeros":
                t = torch.zeros(shape)
            elif create_type == "torch.ones":
                t = torch.ones(shape)
            elif create_type == "torch.randn":
                t = torch.randn(shape)
            else:
                t = torch.arange(np.prod(shape)).reshape(shape)

            st.code(f"Tensor:\n{t}\n\nShape: {t.shape}\nDtype: {t.dtype}\nDevice: {t.device}\nndim: {t.ndim}")

    with col2:
        st.subheader("Operationen")
        st.markdown("""
        **Grundoperationen (elementweise):**
        - `a + b` — Addition
        - `a * b` — Multiplikation (elementweise!)
        - `a @ b` — Matrixmultiplikation
        - `a.mean()`, `a.sum()` — Statistiken

        **Reshape & Indexing:**
        - `.reshape(d1, d2)` — neue Form
        - `.view(d1, d2)` — View (shared memory)
        - `t[2:5]` — Slicing

        **GPU:**
        - `t.cuda()` — auf GPU verschieben
        - `torch.cuda.is_available()` — prüfen
        """)

    st.subheader("GPU-Status")
    if torch.cuda.is_available():
        st.success(f"✅ CUDA verfügbar: {torch.cuda.get_device_name(0)}")
    else:
        st.info("ℹ️ Keine GPU verfügbar — CPU-Modus")

# ═══════════════════════════════════════════════════════════════════════════
# 2. AUTOGRAD
# ═══════════════════════════════════════════════════════════════════════════
elif page == "2. Autograd":
    st.header("🎯 Autograd — Automatische Differentiation")

    st.markdown("""
    PyTorch's **Autograd** berechnet automatisch Gradienten.
    Jeder Tensor mit `requires_grad=True` zeichnet Operationen auf.
    `backward()` berechnet dann die Gradienten via Backpropagation.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("f(x) = x²")
        x_val = st.slider("x-Wert", -5.0, 5.0, 3.0, 0.1)
        if st.button("Gradient berechnen"):
            x = torch.tensor(x_val, requires_grad=True)
            y = x ** 2
            y.backward()
            st.success(f"f({x_val}) = {y.item():.2f}")
            st.info(f"f'(x) = 2x = {x.grad.item():.2f} (erwartet: {2*x_val:.2f})")

    with col2:
        st.subheader("Kettenregel: f(x) = (2x + 1)²")
        x_val2 = st.slider("x-Wert", -5.0, 5.0, 2.0, 0.1, key="x2")
        if st.button("Gradient berechnen", key="btn2"):
            x = torch.tensor(x_val2, requires_grad=True)
            y = (2 * x + 1) ** 2
            y.backward()
            expected = 4 * (2 * x_val2 + 1)
            st.success(f"f({x_val2}) = {y.item():.2f}")
            st.info(f"f'(x) = 4(2x+1) = {x.grad.item():.2f} (erwartet: {expected:.2f})")

    st.subheader("⚠️ Gradienten-Akkumulation")
    st.markdown("""
    **Achtung:** `backward()` akkumuliert Gradienten — sie werden nicht automatisch zurückgesetzt!
    Deshalb immer `optimizer.zero_grad()` vor jedem Trainingsschritt aufrufen.
    """)

    if st.button("Akkumulation demonstrieren"):
        w = torch.tensor(2.0, requires_grad=True)
        results = []
        for i in range(3):
            loss = w ** 2
            loss.backward()
            results.append(f"Schritt {i+1}: w.grad = {w.grad.item():.1f} (akkumuliert: {2*w.item()*(i+1):.1f})")
        st.code("\n".join(results))

# ═══════════════════════════════════════════════════════════════════════════
# 3. CNN-MNIST
# ═══════════════════════════════════════════════════════════════════════════
elif page == "3. CNN-MNIST":
    st.header("🖼️ CNN für MNIST — Hands-on Demo")

    st.markdown("""
    **Architektur:** 2 Conv-Blöcke (Conv2d → BatchNorm → ReLU → MaxPool) → 2 FC-Layer
    
    Ein vortrainiertes CNN-Modell wird geladen und kann auf Testbildern Vorhersagen machen.
    """)

    # CNN-Modell definieren (gleiche Architektur wie 03_cnn_mnist.py)
    class MNIST_CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)

    @st.cache_resource
    def load_cnn_model():
        model = MNIST_CNN()
        model_path = Path("output/mnist_cnn.pt")
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model

    model = load_cnn_model()

    st.subheader("Modell-Architektur")
    st.code(str(model))

    st.subheader("Parameter-Übersicht")
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    st.metric("Gesamt-Parameter", f"{total_params:,}")
    st.metric("Trainierbar", f"{trainable:,}")

    st.info("💡 Für vollständiges Training führe `python 03_cnn_mnist.py` aus.")

# ═══════════════════════════════════════════════════════════════════════════
# 4. TRANSFER-LEARNING
# ═══════════════════════════════════════════════════════════════════════════
elif page == "4. Transfer-Learning":
    st.header("🔄 Transfer-Learning mit ResNet18")

    st.markdown("""
    **Konzept:** Ein auf ImageNet vortrainiertes ResNet18 wird für CIFAR-10 feingetuned.
    
    **Zwei Phasen:**
    1. **Kopf-Training:** Nur das letzte FC-Layer trainieren (Rest eingefroren)
    2. **Fine-Tuning:** Alle Layer mit kleiner Learning-Rate feintunen
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Phase 1: Kopf-Training")
        st.markdown("""
        - Alle vortrainierten Layer **eingefroren**
        - Nur das neue FC-Layer (512→10) wird trainiert
        - Learning-Rate: 0.001
        - 2-5 Epochen
        """)

    with col2:
        st.subheader("Phase 2: Fine-Tuning")
        st.markdown("""
        - Alle Layer **aufgetaut**
        - Geringere Learning-Rate: 0.0001
        - Feintuning der vortrainierten Features
        - 2-5 Epochen
        """)

    st.subheader("CIFAR-10 Klassen")
    classes = ["✈️ airplane", "🚗 automobile", "🐦 bird", "🐱 cat", "🦌 deer",
               "🐕 dog", "🐸 frog", "🐴 horse", "🚢 ship", "🚛 truck"]
    cols = st.columns(5)
    for i, (col, cls) in enumerate(zip(cols * 2, classes)):
        col.markdown(cls)

    st.info("💡 Für vollständiges Training führe `python 04_transfer_learning.py` aus.")

# ═══════════════════════════════════════════════════════════════════════════
# 5. DQN-CARTPOLE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "5. DQN-CartPole":
    st.header("🎮 DQN — Deep Q-Network für CartPole")

    st.markdown("""
    **Reinforcement Learning mit Deep Q-Networks:**
    
    Ein Agent lernt, einen Stab auf einem Wagen zu balancieren — nur durch Trial & Error!
    
    **Komponenten:**
    - **Q-Network:** Neuronales Netz, das Q-Werte (erwartete Belohnung) für jede Aktion schätzt
    - **Target-Network:** Stabilisiert das Training durch verzögerte Updates
    - **Experience Replay:** Speichert Erfahrungen und lernt aus zufälligen Samples
    - **Epsilon-Greedy:** Balanciert Exploration vs. Exploitation
    """)

    st.subheader("DQN-Architektur")
    st.code("""
    QNetwork(
        Linear(4 → 128) + ReLU
        Linear(128 → 128) + ReLU
        Linear(128 → 2)   # Q-Werte für [links, rechts]
    )
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("State-Dimension", "4 (Position, Geschwindigkeit, Winkel, Winkelgeschw.)")
    with col2:
        st.metric("Action-Dimension", "2 (links, rechts)")
    with col3:
        st.metric("Ziel-Reward", "195 (über 100 Episoden)")

    st.subheader("Hyperparameter")
    params = {
        "Episoden": 500,
        "Batch-Size": 64,
        "Gamma (Discount)": 0.99,
        "Epsilon-Start": 1.0,
        "Epsilon-End": 0.01,
        "Epsilon-Decay": 0.995,
        "Learning-Rate": 0.001,
        "Target-Update": "alle 10 Episoden",
        "Replay-Buffer": "10.000 Erfahrungen",
    }
    for k, v in params.items():
        st.markdown(f"- **{k}:** {v}")

    st.info("💡 Für vollständiges Training führe `python 05_rl_cartpole.py` aus.")

st.sidebar.markdown("---")
st.sidebar.markdown("📚 **PyTorch Lernen** — Interaktive Lern-App")
st.sidebar.markdown("[GitHub Repository](https://github.com/mark-baumann/pytorch-lernen)")
