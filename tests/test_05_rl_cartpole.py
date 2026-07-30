"""Tests für 05_rl_cartpole.py — DQN CartPole."""

import numpy as np
import torch

from pytorch_lernen.rl_common import DQNAgent


class TestCartPoleDQN:
    def test_agent_creation(self):
        agent = DQNAgent(
            state_dim=4, action_dim=2, hidden_dim=64,
            device=torch.device("cpu"),
        )
        assert agent.action_dim == 2
        assert agent.epsilon == 1.0

    def test_select_action_greedy(self):
        agent = DQNAgent(
            state_dim=4, action_dim=2, hidden_dim=64,
            device=torch.device("cpu"),
        )
        state = np.zeros(4, dtype=np.float32)
        action = agent.select_action(state, evaluate=True)
        assert action in (0, 1)

    def test_select_action_exploration(self):
        agent = DQNAgent(
            state_dim=4, action_dim=2, hidden_dim=64,
            epsilon_start=1.0, epsilon_end=1.0,  # immer explorieren
            device=torch.device("cpu"),
        )
        state = np.zeros(4, dtype=np.float32)
        # Bei epsilon=1.0 sollte zufällig gewählt werden
        actions = [agent.select_action(state) for _ in range(50)]
        assert 0 in actions and 1 in actions  # beide Aktionen kommen vor

    def test_training_updates(self):
        agent = DQNAgent(
            state_dim=4, action_dim=2, hidden_dim=64,
            min_memory=32, memory_size=100,
            device=torch.device("cpu"),
        )
        # Fill memory
        for _ in range(50):
            state = np.random.randn(4).astype(np.float32)
            action = np.random.randint(0, 2)
            reward = float(np.random.randn())
            next_state = np.random.randn(4).astype(np.float32)
            done = False
            agent.memory.push(state, action, reward, next_state, done)

        loss = agent.update(batch_size=32)
        assert loss is not None
        assert loss > 0

    def test_double_dqn_target(self):
        """Double DQN: Target-Network sollte sich von Policy-Network unterscheiden."""
        agent = DQNAgent(
            state_dim=4, action_dim=2, hidden_dim=64,
            lr=0.1,
            device=torch.device("cpu"),
        )
        # Initial sind sie gleich
        for tp, pp in zip(agent.target_net.parameters(), agent.policy_net.parameters()):
            assert torch.allclose(tp, pp)

        # Direkt Policy-Network-Gewichte ändern (simuliert Training)
        with torch.no_grad():
            for p in agent.policy_net.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        # Jetzt sollten sie unterschiedlich sein
        any_diff = False
        for tp, pp in zip(agent.target_net.parameters(), agent.policy_net.parameters()):
            if not torch.allclose(tp, pp):
                any_diff = True
                break
        assert any_diff, "Target und Policy sollten nach manueller Änderung unterschiedlich sein"

        # update_target sollte sie wieder synchronisieren
        agent.update_target()
        for tp, pp in zip(agent.target_net.parameters(), agent.policy_net.parameters()):
            assert torch.allclose(tp, pp), "Nach update_target sollten sie gleich sein"
