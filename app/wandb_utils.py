"""
W&B Experiment Tracking für pytorch-lernen
===========================================
Integriert Weights & Biases in pytorch-lernen.
Loggt Episoden-Metriken, Modell-Ergebnisse und Tabellen-Daten.

Usage:
    from wandb_utils import WandBTracker
    tracker = WandBTracker(project="pytorch-lernen", config={...})
    tracker.log_episode(episode=1, reward=100, loss=0.05)
    tracker.log_model("cnn_mnist", accuracy=0.95, train_time=120.5, params=1_200_000)
    tracker.log_table("predictions", columns=["input", "pred"], data=[["a", 1], ["b", 2]])
    tracker.finish()
"""

import os
import time
from typing import Optional, List, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBTracker:
    """
    Encapsulated W&B tracker for pytorch-lernen.

    Features:
    - Automatic offline mode when no API key
    - Git commit logging in online mode
    - RL episode logging, model result logging, table logging
    """

    def __init__(
        self,
        project: str = "pytorch-lernen",
        config: Optional[dict] = None,
        tags: Optional[list] = None,
        group: Optional[str] = None,
        job_type: str = "train",
        notes: Optional[str] = None,
        offline: bool = False,
    ):
        self.project = project
        self.run = None
        self._start_time = time.time()

        if WANDB_AVAILABLE:
            try:
                mode = "offline" if offline or not os.environ.get("WANDB_API_KEY") else "online"
                self.run = wandb.init(
                    project=project,
                    config=config or {},
                    mode=mode,
                    tags=tags or ["pytorch-lernen"],
                    group=group,
                    job_type=job_type,
                    notes=notes,
                    dir="wandb_runs",
                )
                if mode == "online":
                    try:
                        import subprocess
                        git_commit = subprocess.check_output(
                            ["git", "rev-parse", "--short", "HEAD"],
                            stderr=subprocess.DEVNULL,
                        ).decode().strip()
                        self.log({"git_commit": git_commit})
                    except Exception:
                        pass
                print(f"📊 W&B initialisiert (mode={mode}, project={project})")
            except Exception as e:
                print(f"⚠️  W&B-Init fehlgeschlagen: {e}")

    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to W&B."""
        if self.run:
            self.run.log(metrics, step=step)

    # ── Domain-specific log methods ──────────────────────────

    def log_episode(
        self,
        episode: int,
        reward: float,
        loss: Optional[float] = None,
        steps: Optional[int] = None,
        epsilon: Optional[float] = None,
    ):
        """Log RL episode metrics."""
        metrics = {f"episode/reward": reward}
        if loss is not None:
            metrics["episode/loss"] = loss
        if steps is not None:
            metrics["episode/steps"] = steps
        if epsilon is not None:
            metrics["episode/epsilon"] = epsilon
        self.log(metrics, step=episode)

    def log_model(
        self,
        name: str,
        accuracy: float,
        train_time: float,
        params: int,
    ):
        """Log model training results."""
        self.log({
            f"model/{name}/accuracy": accuracy,
            f"model/{name}/train_time_seconds": train_time,
            f"model/{name}/parameters": params,
        })

    def log_table(
        self,
        name: str,
        columns: List[str],
        data: List[List[Any]],
    ):
        """Log a W&B table."""
        if self.run:
            table = wandb.Table(columns=columns, data=data)
            self.run.log({name: table})

    def finish(self):
        """End the W&B run. Safe to call multiple times."""
        elapsed = time.time() - self._start_time
        if self.run:
            self.log({"total_time_seconds": elapsed})
            self.run.finish()
            self.run = None

    @property
    def is_active(self) -> bool:
        return self.run is not None
