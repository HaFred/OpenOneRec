"""Model-agnostic adapter contracts for recsys RL workflows."""

from __future__ import annotations

from typing import Any, Callable, Protocol


class DatasetAdapter(Protocol):
    """Transforms dataset outputs into recsys-compatible records."""

    def wrap_dataset(self, dataset: Any, is_train: bool) -> Any:
        """Return a dataset wrapper or original dataset."""


class ActorAdapter(Protocol):
    """Mutates runtime/training config for a model family."""

    def prepare_runtime(self, config: Any) -> None:
        """Adjust rollout/actor settings before workers start."""

    def describe(self) -> dict[str, Any]:
        """Return metadata used for diagnostics."""


class RewardAdapter(Protocol):
    """Optionally wraps reward functions with recsys shaping logic."""

    def wrap_reward_fn(self, reward_fn: Callable[..., Any], is_validation: bool = False) -> Callable[..., Any]:
        """Return a wrapped reward function."""

