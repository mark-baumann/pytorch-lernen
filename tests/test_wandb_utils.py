"""
Tests for wandb_utils.py — W&B Experiment Tracking for pytorch-lernen.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wandb_utils import WandBTracker, WANDB_AVAILABLE


class TestWandBTracker:
    """Tests for WandBTracker."""

    def test_initialization_offline(self):
        """Tracker should initialize in offline mode."""
        tracker = WandBTracker(
            project="test-pytorch-lernen",
            config={"key": "value"},
            tags=["test"],
            group="test-group",
            job_type="test",
            notes="Test-Run",
            offline=True,
        )
        if WANDB_AVAILABLE:
            assert tracker.is_active
            assert tracker.run is not None
        else:
            assert not tracker.is_active
        tracker.finish()

    def test_log_metrics(self):
        """Metrics should log without errors."""
        tracker = WandBTracker(project="test-pytorch-lernen", offline=True)
        if tracker.is_active:
            tracker.log({"accuracy": 0.95})
        tracker.finish()

    def test_finish_cleans_up(self):
        """finish() should end the run and be safe for double calls."""
        tracker = WandBTracker(project="test-pytorch-lernen", offline=True)
        tracker.finish()
        tracker.finish()  # Double finish should be safe

    def test_log_episode(self):
        """log_episode should log RL metrics without errors."""
        tracker = WandBTracker(project="test-pytorch-lernen", offline=True)
        if tracker.is_active:
            tracker.log_episode(episode=1, reward=100.0, loss=0.05, steps=200, epsilon=0.1)
        tracker.finish()

    def test_log_episode_minimal(self):
        """log_episode with only required args should work."""
        tracker = WandBTracker(project="test-pytorch-lernen", offline=True)
        if tracker.is_active:
            tracker.log_episode(episode=5, reward=42.0)
        tracker.finish()

    def test_log_model(self):
        """log_model should log model results without errors."""
        tracker = WandBTracker(project="test-pytorch-lernen", offline=True)
        if tracker.is_active:
            tracker.log_model("cnn_mnist", accuracy=0.95, train_time=120.5, params=1_200_000)
        tracker.finish()

    def test_log_table(self):
        """log_table should log a W&B table without errors."""
        tracker = WandBTracker(project="test-pytorch-lernen", offline=True)
        if tracker.is_active:
            tracker.log_table(
                "predictions",
                columns=["input", "pred"],
                data=[["a", 1], ["b", 2]],
            )
        tracker.finish()
