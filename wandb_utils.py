"""
W&B Experiment Tracking für PyTorch Lernen
==========================================
Integriert Weights & Biases in PyTorch-Trainings.
Loggt Trainingsmetriken, Modell-Ergebnisse und RL-Episoden.

Verwendung:
    from wandb_utils import WandBTracker
    tracker = WandBTracker(project="pytorch-lernen", config={...})
    tracker.log_episode(episode=1, reward=100.0, loss=0.05, steps=200, epsilon=0.1)
    tracker.finish()
"""

import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBTracker:
    """
    Gekapselter W&B-Tracker für PyTorch-Lernprojekte.

    Features:
    - Trainings-Metriken (Loss, Accuracy)
    - RL-Episoden-Tracking
    - Modell-Ergebnisse
    - Tabellen-Logging
    """

    def __init__(self, project: str = "pytorch-lernen",
                 config: dict = None, tags: list = None,
                 group: str = None, job_type: str = "train",
                 notes: str = None, offline: bool = False):
        self.project = project
        self.run = None

        if WANDB_AVAILABLE:
            try:
                mode = "offline" if offline or not os.environ.get("WANDB_API_KEY") else "online"
                self.run = wandb.init(
                    project=project,
                    config=config or {},
                    mode=mode,
                    tags=tags or ["pytorch"],
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
                            stderr=subprocess.DEVNULL
                        ).decode().strip()
                        self.log({"git_commit": git_commit})
                    except Exception:
                        pass
                print(f"📊 W&B initialisiert (mode={mode}, project={project})")
            except Exception as e:
                print(f"⚠️  W&B-Init fehlgeschlagen: {e}")

    def log(self, metrics: dict, step: int = None):
        """Loggt Metriken zu W&B."""
        if self.run:
            self.run.log(metrics, step=step)

    def log_episode(self, episode: int, reward: float,
                    loss: float = None, steps: int = None,
                    epsilon: float = None):
        """Loggt eine RL-Episode."""
        metrics = {
            "episode": episode,
            "reward": reward,
        }
        if loss is not None:
            metrics["loss"] = loss
        if steps is not None:
            metrics["steps"] = steps
        if epsilon is not None:
            metrics["epsilon"] = epsilon
        self.log(metrics)

    def log_model(self, name: str, accuracy: float,
                  train_time: float = None, params: int = None):
        """Loggt Modell-Ergebnisse."""
        metrics = {"model/accuracy": accuracy, "model/name": name}
        if train_time is not None:
            metrics["model/train_time"] = train_time
        if params is not None:
            metrics["model/params"] = params
        self.log(metrics)

    def log_table(self, name: str, columns: list, data: list):
        """Loggt eine Tabelle ins W&B Dashboard."""
        if not self.run:
            return
        table = wandb.Table(columns=columns)
        for row in data:
            table.add_data(*row)
        self.run.log({name: table})

    def finish(self):
        """Beendet den W&B-Run. Sicher bei mehrfachem Aufruf."""
        if self.run:
            self.run.finish()
            self.run = None

    @property
    def is_active(self) -> bool:
        return self.run is not None
