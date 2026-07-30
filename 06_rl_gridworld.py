#!/usr/bin/env python3
"""
06_rl_gridworld.py – Deep Q-Network (DQN) mit PyTorch für GridWorld
===================================================================
PyTorch-Implementierung von DQN auf der GridWorld-Umgebung aus
agent-reinforcement-learning. Verbindet die beiden Repos.

Architektur: 2-Layer MLP (state → 64 → 64 → 4 actions)
Training:    Experience Replay, Target Network, Epsilon-Greedy
W&B:         Experiment-Tracking mit Weights & Biases
"""

import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

# ── GridWorld aus agent-reinforcement-learning importieren ──────────
sys.path.insert(0, "/opt/data/agent-reinforcement-learning")
from rl_agent import GridWorld

# ── W&B (optional) ──────────────────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ── OpenPipe (optional) ──────────────────────────────────────────────
try:
    from openpipe import OpenAI as OpenPipeClient
    OPENPIPE_AVAILABLE = True
except ImportError:
    OPENPIPE_AVAILABLE = False
    OpenPipeClient = None

# ── Reproduzierbarkeit ──────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {DEVICE}")

# ── W&B Initialisierung ─────────────────────────────────────────────
if WANDB_AVAILABLE:
    mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    wandb.init(
        project="learn-pytorch",
        name="dqn-gridworld",
        config={
            "algorithm": "DQN",
            "env": "GridWorld-4x4",
            "episodes": 500,
            "batch_size": 32,
            "gamma": 0.99,
            "lr": 0.001,
            "target_update": 10,
            "memory_size": 5000,
        },
        mode=mode,
        tags=["dqn", "gridworld", "pytorch"]
    )
    print(f"📊 W&B initialisiert (mode={mode})")

# ── OpenPipe Logger ──────────────────────────────────────────────────
op_logger = None
if OPENPIPE_AVAILABLE:
    from rl_agent import OpenPipeLogger
    op_logger = OpenPipeLogger()
    print("🔧 OpenPipe Logger initialisiert")
else:
    print("⚠️  OpenPipe nicht installiert — überspringe Tracking")

# ── Hyperparameter ──────────────────────────────────────────────────
EPISODES = 500
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
TARGET_UPDATE = 10
MEMORY_SIZE = 5_000
MIN_MEMORY = 500

# ── Environment ─────────────────────────────────────────────────────
GRID_SIZE = 4
env = GridWorld(size=GRID_SIZE)
STATE_DIM = 2  # (row, col)
ACTION_DIM = 4  # up, right, down, left

print(f"Environment: GridWorld {GRID_SIZE}×{GRID_SIZE}")
print(f"Goal: {env.goal}, Start: {env.start}")


# ── Q-Network ───────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """MLP für Q-Wert-Approximation auf GridWorld."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
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


# ── Replay Buffer ───────────────────────────────────────────────────
class ReplayBuffer:
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


# ── DQN-Agent ───────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = action_dim
        self.policy_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START

    def _state_to_tensor(self, state: tuple) -> np.ndarray:
        """Konvertiert (row, col) → normalisiertes [row/size, col/size]."""
        return np.array([state[0] / GRID_SIZE, state[1] / GRID_SIZE], dtype=np.float32)

    def select_action(self, state: tuple, evaluate: bool = False) -> int:
        if evaluate or random.random() > self.epsilon:
            with torch.no_grad():
                state_t = torch.FloatTensor(self._state_to_tensor(state)).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state_t)
                return q_values.argmax(dim=1).item()
        else:
            return random.randrange(self.action_dim)

    def update(self) -> float | None:
        if len(self.memory) < MIN_MEMORY:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


agent = DQNAgent(STATE_DIM, ACTION_DIM)
print(f"Policy-Network Parameter: {sum(p.numel() for p in agent.policy_net.parameters()):,}")

# ── Training ────────────────────────────────────────────────────────
episode_rewards = []
epsilons = []
losses = []
moving_avg = []

print("\n── Training ──")
for episode in range(1, EPISODES + 1):
    state = env.reset()
    episode_reward = 0.0
    episode_losses = []
    steps = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        agent.memory.push(
            agent._state_to_tensor(state),
            action,
            reward,
            agent._state_to_tensor(next_state),
            done
        )
        state = next_state
        episode_reward += reward
        steps += 1

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

    if len(episode_rewards) >= 100:
        ma = np.mean(episode_rewards[-100:])
    else:
        ma = np.mean(episode_rewards)
    moving_avg.append(ma)

    if episode % 50 == 0 or episode == 1:
        print(f"Episode {episode:4d}/{EPISODES} | Reward: {episode_reward:6.3f} | "
              f"Avg100: {ma:6.3f} | Epsilon: {agent.epsilon:.3f} | Loss: {avg_loss:.4f} | Steps: {steps}")

    # ── W&B Logging ──────────────────────────────────────────
    if WANDB_AVAILABLE and episode % 10 == 0:
        wandb.log({
            "episode": episode,
            "reward": episode_reward,
            "avg_reward_100": ma,
            "epsilon": agent.epsilon,
            "loss": avg_loss,
            "steps": steps,
        })

    # ── OpenPipe Logging ─────────────────────────────────────
    if op_logger and episode % 50 == 0:
        op_logger.log_episode({
            "episode": episode,
            "algorithm": "dqn-gridworld",
            "reward": episode_reward,
            "avg_reward_100": ma,
            "epsilon": agent.epsilon,
            "steps": steps,
        })

# ── Evaluation ──────────────────────────────────────────────────────
print("\n── Evaluation ──")
eval_rewards = []
for _ in range(20):
    state = env.reset()
    total_reward = 0.0
    for _ in range(50):
        action = agent.select_action(state, evaluate=True)
        state, reward, done = env.step(action)
        total_reward += reward
        if done:
            break
    eval_rewards.append(total_reward)

avg_eval = np.mean(eval_rewards)
print(f"Evaluations-Reward (20 Episoden): {avg_eval:.3f}")

# ── Gelernte Policy visualisieren ───────────────────────────────────
print("\n── Gelernte Policy (DQN) ──")
arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}
for r in range(GRID_SIZE):
    row = "  "
    for c in range(GRID_SIZE):
        if (r, c) == env.goal:
            row += " 🎯 "
        else:
            state_t = torch.FloatTensor(agent._state_to_tensor((r, c))).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                q_values = agent.policy_net(state_t)
                best_action = q_values.argmax(dim=1).item()
            row += f" {arrows[best_action]} "
    print(row)

# ── Vergleich: Tabular Q-Learning vs DQN ────────────────────────────
print("\n── Vergleich: Tabular Q-Learning vs DQN ──")
from rl_agent import QLearning

tabular_agent = QLearning(env, lr=0.1, gamma=0.99, epsilon=0.3)
tabular_rewards = tabular_agent.train(episodes=500)
tabular_avg = np.mean(tabular_rewards[-100:])
print(f"  Tabular Q-Learning Avg100: {tabular_avg:.3f}")
print(f"  DQN (PyTorch) Avg100:      {moving_avg[-1]:.3f}")

# ── Speichern ───────────────────────────────────────────────────────
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Modell speichern
torch.save({
    'policy_net': agent.policy_net.state_dict(),
    'target_net': agent.target_net.state_dict(),
    'optimizer': agent.optimizer.state_dict(),
    'epsilon': agent.epsilon,
}, output_dir / "06_dqn_gridworld.pt")

# Plot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(episode_rewards, alpha=0.4, color="blue", linewidth=0.8, label="Episode Reward")
axes[0, 0].plot(moving_avg, color="red", linewidth=2, label="Moving Avg (100)")
axes[0, 0].set_title("Reward pro Episode — DQN GridWorld")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Reward")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(epsilons, color="purple", linewidth=2)
axes[0, 1].set_title("Epsilon-Decay")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Epsilon")
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(losses, color="orange", alpha=0.7, linewidth=1)
axes[1, 0].set_title("Training Loss")
axes[1, 0].set_xlabel("Episode")
axes[1, 0].set_ylabel("Loss")
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].bar(["Tabular QL", "DQN (PyTorch)"], [tabular_avg, moving_avg[-1]],
               color=["steelblue", "coral"])
axes[1, 1].set_title("Vergleich: Tabular vs DQN")
axes[1, 1].set_ylabel("Avg Reward (100 Ep.)")
axes[1, 1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(output_dir / "06_dqn_gridworld_metrics.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nPlots & Modell gespeichert unter: {output_dir}/")
print("06_rl_gridworld.py – Fertig!")

# ── OpenPipe Export ──────────────────────────────────────────────────
if op_logger and len(op_logger.get_training_data()) > 0:
    op_logger.export_jsonl("output/06_gridworld_training_data.jsonl")

# ── W&B Cleanup ─────────────────────────────────────────────────────
if WANDB_AVAILABLE:
    wandb.finish()
