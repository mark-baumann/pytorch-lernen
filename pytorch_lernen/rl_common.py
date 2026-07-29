"""Gemeinsame Reinforcement-Learning-Komponenten.

QNetwork, ReplayBuffer, DQNAgent — verwendet von CartPole und GridWorld.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Einfaches MLP für Q-Wert-Approximation."""

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


class ReplayBuffer:
    """Experience Replay Buffer mit zufälligem Sampling."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN-Agent mit Target-Network, Double DQN und Epsilon-Greedy.

    Args:
        state_dim: Dimension des Zustandsraums
        action_dim: Anzahl der Aktionen
        hidden_dim: Versteckte Neuronen pro Layer
        lr: Lernrate
        gamma: Discount-Faktor
        epsilon_start: Start-Exploration
        epsilon_end: Minimale Exploration
        epsilon_decay: Decay-Faktor pro Episode
        memory_size: Größe des Replay-Buffers
        min_memory: Mindest-Erfahrungen vor Trainingsbeginn
        target_update: Target-Network Update-Frequenz (Episoden)
        device: torch.device
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10_000,
        min_memory: int = 1_000,
        target_update: int = 10,
        device: torch.device | None = None,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.min_memory = min_memory
        self.target_update = target_update
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(memory_size)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.episode_count = 0

    def select_action(
        self, state: np.ndarray | torch.Tensor, evaluate: bool = False
    ) -> int:
        """Wählt Aktion via Epsilon-Greedy (oder greedy bei evaluate=True)."""
        if evaluate or random.random() > self.epsilon:
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                else:
                    state_t = state.unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax(dim=1).item()
        else:
            return random.randrange(self.action_dim)

    def update(self, batch_size: int) -> float | None:
        """Ein DQN-Update-Schritt (Double DQN)."""
        if len(self.memory) < self.min_memory:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            batch_size, self.device
        )

        # Current Q values
        q_values = (
            self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Double DQN: policy net wählt, target net bewertet
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q_values = (
                self.target_net(next_states)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self) -> None:
        """Synchronisiert Target-Network mit Policy-Network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """Reduziert Epsilon exponentiell."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    def step_episode(self) -> None:
        """Führt Target-Update und Epsilon-Decay nach einer Episode aus."""
        self.decay_epsilon()
        if self.episode_count % self.target_update == 0:
            self.update_target()
