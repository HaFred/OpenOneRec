"""Checkpoint engine abstraction for recsys trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CheckpointEngineConfig:
    backend: str = "local_fs"
    manager_enabled: bool = True


class CheckpointEngineManager:
    """Lightweight compatibility layer aligned with upcoming VeRL checkpoint APIs."""

    def __init__(self, config: CheckpointEngineConfig):
        self.config = config

    def save(self, trainer: Any) -> None:
        if hasattr(trainer, "_save_checkpoint"):
            trainer._save_checkpoint()  # pylint: disable=protected-access

    def load(self, trainer: Any) -> None:
        if hasattr(trainer, "_load_checkpoint"):
            trainer._load_checkpoint()  # pylint: disable=protected-access
