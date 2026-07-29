#!/usr/bin/env python3
"""
05_rl_cartpole.py – DQN (Deep Q-Network) mit PyTorch für CartPole-v1
=====================================================================
Implementierung: DQN mit Experience Replay, Target Network, Epsilon-Greedy Exploration.
Environment:     CartPole-v1 (OpenAI Gymnasium)

Training: 500 Episoden, MSE-Loss, Adam-Optimizer
Evaluation: Gleitender Durchschnitt der letzten 100 Episoden
Visualisierung: Reward-Kurve, Epsilon-Decay, Beispiel-Trajektorie
W&B Tracking:  Experiment-Tracking mit Weights & Biases
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import os
import sys
from pathlib import Path

# ── W&B (optional) ──────────────────────────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ── OpenPipe (optional) ─────────────────────────────────────────────────────
try:
    from openpipe import OpenAI as OpenPipeClient
    OPENPIPE_AVAILABLE = True
except ImportError:
    OPENPIPE_AVAILABLE = False
    OpenPipeClient = None

# ── Reproduzierbarkeit ──────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── Gerät ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {DEVICE}")

# ── W&B Initialisierung ─────────────────────────────────────────────────────
if WANDB_AVAILABLE:
    mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    wandb.init(
        project="pytorch-learning",
        name="dqn-cartpole",
        config={
            "algorithm": "DQN",
            "env": "CartPole-v1",
            "episodes": 500,
            "batch_size": 64,
            "gamma": 0.99,
            "lr": 0.001,
            "target_update": 10,
            "memory_size": 10000,
        },
        mode=mode,
        tags=["dqn", "cartpole", "pytorch"]
    )
    print(f"📊 W&B initialisiert (mode={mode})")
else:
    print("⚠️  W&B nicht installiert — überspringe Tracking")

# ── OpenPipe Logger ──────────────────────────────────────────────────────────
op_logger = None
if OPENPIPE_AVAILABLE:
    sys.path.insert(0, "/opt/data/agent-reinforcement-learning")
    from rl_agent import OpenPipeLogger
    op_logger = OpenPipeLogger()
    print("🔧 OpenPipe Logger initialisiert")
else:
    print("⚠️  OpenPipe nicht installiert — überspringe Tracking")

# ── Hyperparameter ──────────────────────────────────────────────────────────
EPISODES = 500
BATCH_SIZE = 64
GAMMA = 0.99                # Discount-Faktor
EPSILON_START = 1.0         # Start-Exploration
EPSILON_END = 0.01          # Minimale Exploration
EPSILON_DECAY = 0.995       # Decay pro Episode
LEARNING_RATE = 0.001
TARGET_UPDATE = 10          # Target-Network Update-Frequenz (Episoden)
MEMORY_SIZE = 10_000        # Replay-Buffer-Größe
MIN_MEMORY = 1_000          # Mindest-Erfahrungen vor Training

# ── Environment ─────────────────────────────────────────────────────────────
try:
    import gymnasium as gym
except ImportError:
    print("Installiere gymnasium...")
    import subprocess
    subprocess.check_call(["uv", "pip", "install", "gymnasium"])
    import gymnasium as gym

env = gym.make("CartPole-v1")
STATE_DIM = env.observation_space.shape[0]   # 4
ACTION_DIM = env.action_space.n              # 2

print(f"Environment: CartPole-v1")
print(f"State-Dimension: {STATE_DIM}, Action-Dimension: {ACTION_DIM}")


# ── Q-Network ───────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """Einfaches Feed-Forward-Netz für Q-Wert-Approximation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay Buffer ───────────────────────────────────────────────────────────
class ReplayBuffer:
    """Experience Replay Buffer mit zufälligem Sampling."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.LongTensor(actions).to(DEVICE),
            torch.FloatTensor(rewards).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(dones).to(DEVICE),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ── DQN-Agent ───────────────────────────────────────────────────────────────
class DQNAgent:
    """DQN-Agent mit Target-Network und Epsilon-Greedy."""

    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = action_dim

        self.policy_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if evaluate or random.random() > self.epsilon:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state_t)
                return q_values.argmax(dim=1).item()
        else:
            return random.randrange(self.action_dim)

    def update(self) -> float | None:
        if len(self.memory) < MIN_MEMORY:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN: policy net wählt, target net bewertet)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping für Stabilität
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


agent = DQNAgent(STATE_DIM, ACTION_DIM)
print(f"Policy-Network Parameter: {sum(p.numel() for p in agent.policy_net.parameters()):,}")

# ── Training ────────────────────────────────────────────────────────────────
episode_rewards: list[float] = []
epsilons: list[float] = []
losses: list[float] = []
moving_avg: list[float] = []

print("\n── Training ──")
for episode in range(1, EPISODES + 1):
    state, _ = env.reset()
    episode_reward = 0.0
    episode_losses: list[float] = []

    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.memory.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        loss = agent.update()
        if loss is not None:
            episode_losses.append(loss)

        if done:
            break

    episode_rewards.append(episode_reward)
    epsilons.append(agent.epsilon)
    avg_loss = np.mean(episode_losses) if episode_losses else 0.0
    losses.append(avg_loss)

    agent.decay_epsilon()

    if episode % TARGET_UPDATE == 0:
        agent.update_target()

    # Gleitender Durchschnitt (letzte 100 Episoden)
    if len(episode_rewards) >= 100:
        ma = np.mean(episode_rewards[-100:])
    else:
        ma = np.mean(episode_rewards)
    moving_avg.append(ma)

    if episode % 50 == 0 or episode == 1:
        print(f"Episode {episode:4d}/{EPISODES} | Reward: {episode_reward:6.1f} | "
              f"Avg100: {ma:6.1f} | Epsilon: {agent.epsilon:.3f} | Loss: {avg_loss:.4f}")

    # ── W&B Logging ──────────────────────────────────────────
    if WANDB_AVAILABLE and episode % 10 == 0:
        wandb.log({
            "episode": episode,
            "reward": episode_reward,
            "avg_reward_100": ma,
            "epsilon": agent.epsilon,
            "loss": avg_loss,
        })

    # ── OpenPipe Logging ─────────────────────────────────────
    if OPENPIPE_AVAILABLE and episode % 50 == 0:
        op_logger.log_episode({
            "episode": episode,
            "algorithm": "dqn-cartpole",
            "reward": episode_reward,
            "avg_reward_100": ma,
            "epsilon": agent.epsilon,
        })

env.close()

print(f"\nFinaler Avg100-Reward: {moving_avg[-1]:.2f}")
solved = moving_avg[-1] >= 195.0
print(f"CartPole {'GELÖST! 🎉' if solved else 'noch nicht gelöst (Ziel: 195.0)'}")

# ── Visualisierung ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Reward pro Episode
axes[0, 0].plot(episode_rewards, alpha=0.4, color="blue", linewidth=0.8,
                label="Episode Reward")
axes[0, 0].plot(moving_avg, color="red", linewidth=2,
                label="Moving Avg (100)")
axes[0, 0].axhline(y=195.0, color="green", linestyle="--",
                   label="Solved (195)")
axes[0, 0].set_title("Reward pro Episode")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Reward")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Epsilon-Decay
axes[0, 1].plot(epsilons, color="purple", linewidth=2)
axes[0, 1].set_title("Epsilon-Decay")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Epsilon")
axes[0, 1].grid(True, alpha=0.3)

# Loss
axes[1, 0].plot(losses, color="orange", alpha=0.7, linewidth=1)
axes[1, 0].set_title("Training Loss")
axes[1, 0].set_xlabel("Episode")
axes[1, 0].set_ylabel("Loss")
axes[1, 0].grid(True, alpha=0.3)

# Reward-Histogramm
axes[1, 1].hist(episode_rewards, bins=30, color="steelblue", edgecolor="white",
                alpha=0.8)
axes[1, 1].axvline(x=195.0, color="green", linestyle="--", linewidth=2,
                   label="Solved (195)")
axes[1, 1].set_title("Reward-Verteilung")
axes[1, 1].set_xlabel("Reward")
axes[1, 1].set_ylabel("Häufigkeit")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

fig.tight_layout()

# ── Beispiel-Trajektorie (Evaluation) ────────────────────────────────────────
print("\n── Evaluation: Beispiel-Trajektorie ──")
eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
state, _ = eval_env.reset()
total_reward = 0.0
step_count = 0
frames: list = []

for step in range(500):
    action = agent.select_action(state, evaluate=True)
    state, reward, terminated, truncated, _ = eval_env.step(action)
    total_reward += reward
    step_count += 1
    frames.append(eval_env.render())
    if terminated or truncated:
        break

eval_env.close()
print(f"Evaluations-Reward: {total_reward:.1f} in {step_count} Schritten")

# Zeige 4 Frames der Trajektorie
fig3, axes3 = plt.subplots(1, 4, figsize=(16, 4))
indices = np.linspace(0, len(frames) - 1, 4, dtype=int)
for i, idx in enumerate(indices):
    axes3[i].imshow(frames[idx])
    axes3[i].set_title(f"Step {idx}")
    axes3[i].axis("off")
fig3.suptitle(f"Beispiel-Trajektorie (Reward: {total_reward:.1f})", fontsize=13)
fig3.tight_layout()

# Speichern
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
fig.savefig(output_dir / "05_rl_cartpole_metrics.png", dpi=150,
            bbox_inches="tight")
fig3.savefig(output_dir / "05_rl_cartpole_trajectory.png", dpi=150,
             bbox_inches="tight")
plt.show()

print(f"\nPlots gespeichert unter: {output_dir}/")
print("05_rl_cartpole.py – Fertig!")

# ── OpenPipe Export ──────────────────────────────────────────────────────────
if op_logger and len(op_logger.get_training_data()) > 0:
    op_logger.export_jsonl("output/05_cartpole_training_data.jsonl")

# ── W&B Cleanup ─────────────────────────────────────────────────────────────
if WANDB_AVAILABLE:
    wandb.finish()
