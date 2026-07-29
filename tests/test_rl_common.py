"""Tests für pytorch_lernen.rl_common."""

import numpy as np
import torch
from pytorch_lernen.rl_common import QNetwork, ReplayBuffer, DQNAgent


class TestQNetwork:
    def test_forward_shape(self):
        net = QNetwork(state_dim=4, action_dim=2, hidden_dim=64)
        x = torch.randn(8, 4)
        out = net(x)
        assert out.shape == (8, 2)

    def test_single_input(self):
        net = QNetwork(state_dim=2, action_dim=4, hidden_dim=32)
        x = torch.randn(1, 2)
        out = net(x)
        assert out.shape == (1, 4)

    def test_different_dims(self):
        net = QNetwork(state_dim=10, action_dim=5, hidden_dim=128)
        x = torch.randn(16, 10)
        out = net(x)
        assert out.shape == (16, 5)


class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0
        buf.push(np.array([1.0, 2.0]), 0, 1.0, np.array([1.0, 3.0]), False)
        assert len(buf) == 1

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(np.array([i]), 0, 1.0, np.array([i + 1]), False)
        assert len(buf) == 5

    def test_sample_shapes(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(
                np.array([float(i), 0.0]),
                i % 4,
                float(i % 2),
                np.array([float(i + 1), 0.0]),
                i % 5 == 0,
            )
        device = torch.device("cpu")
        states, actions, rewards, next_states, dones = buf.sample(16, device)
        assert states.shape == (16, 2)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
        assert next_states.shape == (16, 2)
        assert dones.shape == (16,)


class TestDQNAgent:
    def test_init(self):
        agent = DQNAgent(state_dim=4, action_dim=2, hidden_dim=64, device=torch.device("cpu"))
        assert agent.action_dim == 2
        assert agent.epsilon == 1.0
        assert len(agent.memory) == 0

    def test_select_action_shape(self):
        agent = DQNAgent(state_dim=4, action_dim=2, hidden_dim=64, device=torch.device("cpu"))
        state = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
        action = agent.select_action(state, evaluate=True)
        assert action in (0, 1)

    def test_select_action_tensor(self):
        agent = DQNAgent(state_dim=4, action_dim=2, hidden_dim=64, device=torch.device("cpu"))
        state = torch.randn(4)
        action = agent.select_action(state, evaluate=True)
        assert action in (0, 1)

    def test_update_before_min_memory(self):
        agent = DQNAgent(state_dim=4, action_dim=2, min_memory=100, device=torch.device("cpu"))
        loss = agent.update(batch_size=32)
        assert loss is None

    def test_update_after_filling(self):
        agent = DQNAgent(
            state_dim=4, action_dim=2, hidden_dim=64,
            min_memory=32, memory_size=100, device=torch.device("cpu"),
        )
        # Fill memory
        for _ in range(50):
            state = np.random.randn(4).astype(np.float32)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(4).astype(np.float32)
            done = np.random.random() > 0.9
            agent.memory.push(state, action, reward, next_state, done)

        loss = agent.update(batch_size=32)
        assert loss is not None
        assert isinstance(loss, float)

    def test_epsilon_decay(self):
        agent = DQNAgent(
            state_dim=4, action_dim=2, epsilon_start=1.0,
            epsilon_end=0.01, epsilon_decay=0.9, device=torch.device("cpu"),
        )
        agent.decay_epsilon()
        assert agent.epsilon == 0.9
        agent.decay_epsilon()
        assert agent.epsilon == 0.81

    def test_epsilon_floor(self):
        agent = DQNAgent(
            state_dim=4, action_dim=2, epsilon_start=0.001,
            epsilon_end=0.01, epsilon_decay=0.9, device=torch.device("cpu"),
        )
        agent.decay_epsilon()
        assert agent.epsilon == 0.01  # clamped to epsilon_end

    def test_update_target(self):
        agent = DQNAgent(state_dim=4, action_dim=2, device=torch.device("cpu"))
        # Change policy net weights
        for p in agent.policy_net.parameters():
            p.data.add_(0.1)
        agent.update_target()
        # Target should now match policy
        for tp, pp in zip(agent.target_net.parameters(), agent.policy_net.parameters()):
            assert torch.allclose(tp, pp)

    def test_step_episode(self):
        agent = DQNAgent(
            state_dim=4, action_dim=2, target_update=3, device=torch.device("cpu"),
        )
        for _ in range(5):
            agent.step_episode()
        assert agent.episode_count == 5
        assert agent.epsilon < 1.0
