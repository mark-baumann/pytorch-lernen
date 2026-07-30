"""Tests für 06_rl_gridworld.py — DQN GridWorld."""

import numpy as np
import torch

from pytorch_lernen.rl_common import DQNAgent


class TestGridWorldDQN:
    def test_agent_creation(self):
        agent = DQNAgent(
            state_dim=2, action_dim=4, hidden_dim=64,
            device=torch.device("cpu"),
        )
        assert agent.action_dim == 4
        assert agent.epsilon == 1.0

    def test_select_action_greedy(self):
        agent = DQNAgent(
            state_dim=2, action_dim=4, hidden_dim=64,
            device=torch.device("cpu"),
        )
        state = np.array([0.5, 0.5], dtype=np.float32)
        action = agent.select_action(state, evaluate=True)
        assert 0 <= action < 4

    def test_training_updates(self):
        agent = DQNAgent(
            state_dim=2, action_dim=4, hidden_dim=64,
            min_memory=32, memory_size=100,
            device=torch.device("cpu"),
        )
        for _ in range(50):
            state = np.random.randn(2).astype(np.float32)
            action = np.random.randint(0, 4)
            reward = float(np.random.randn())
            next_state = np.random.randn(2).astype(np.float32)
            done = False
            agent.memory.push(state, action, reward, next_state, done)

        loss = agent.update(batch_size=32)
        assert loss is not None
        assert loss > 0

    def test_epsilon_decay_over_episodes(self):
        agent = DQNAgent(
            state_dim=2, action_dim=4, hidden_dim=64,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9,
            device=torch.device("cpu"),
        )
        for _ in range(10):
            agent.decay_epsilon()
        assert agent.epsilon < 0.5
        assert agent.epsilon >= 0.01
