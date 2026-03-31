"""Default recsys adapters used by the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from verl_recsys.adapters.base import ActorAdapter, DatasetAdapter, RewardAdapter


@dataclass
class DatasetNormalizationAdapter(DatasetAdapter):
    """Adds lightweight metadata needed by model-agnostic recsys paths."""

    model_family: str = "hstu"

    def wrap_dataset(self, dataset: Any, is_train: bool) -> Any:
        return _RecsysDatasetWrapper(dataset=dataset, model_family=self.model_family, is_train=is_train)


class _RecsysDatasetWrapper:
    def __init__(self, dataset: Any, model_family: str, is_train: bool):
        self._dataset = dataset
        self._model_family = model_family
        self._is_train = is_train

    def __len__(self) -> int:
        return len(self._dataset)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._dataset, item)

    def __getitem__(self, index: int) -> Any:
        sample = self._dataset[index]
        if isinstance(sample, dict):
            recsys_meta = dict(sample.get("recsys_meta", {}))
            recsys_meta.setdefault("model_family", self._model_family)
            recsys_meta.setdefault("is_train_split", self._is_train)
            sample["recsys_meta"] = recsys_meta
        return sample


@dataclass
class HSTUActorAdapter(ActorAdapter):
    """Actor adapter for HSTU-style sequence transducer policies."""

    rollout_backend: str = "vllm"

    def prepare_runtime(self, config: Any) -> None:
        config.recsys.rollout.backend = self.rollout_backend
        config.actor_rollout_ref.rollout.mode = config.recsys.rollout.get("mode", "sync")
        # Keep default rollout worker path unless users explicitly opt out.
        if config.recsys.get("actor_adapter", {}).get("prefer_reference_policy", True):
            config.actor_rollout_ref.actor.use_ref = config.actor_rollout_ref.actor.get("use_ref", True)

    def describe(self) -> dict[str, Any]:
        return {"model_family": "hstu", "rollout_backend": self.rollout_backend}


@dataclass
class SASRecActorAdapter(ActorAdapter):
    """Reference adapter proving portability to a second GR family."""

    rollout_backend: str = "vllm"

    def prepare_runtime(self, config: Any) -> None:
        config.recsys.rollout.backend = self.rollout_backend
        config.actor_rollout_ref.rollout.mode = config.recsys.rollout.get("mode", "sync")
        config.recsys.training.mode = config.recsys.training.get("mode", "ppo")

    def describe(self) -> dict[str, Any]:
        return {"model_family": "sasrec", "rollout_backend": self.rollout_backend}


@dataclass
class WeightedRewardAdapter(RewardAdapter):
    """Adds lightweight shaped reward components from recsys metadata."""

    alpha_click: float = 1.0
    alpha_watchtime: float = 0.0

    def wrap_reward_fn(self, reward_fn: Callable[..., Any], is_validation: bool = False) -> Callable[..., Any]:
        if not callable(reward_fn):
            return reward_fn

        def _wrapped(*args, **kwargs):
            result = reward_fn(*args, **kwargs)
            # Keep shaping intentionally conservative and opt-in by metadata.
            if isinstance(result, dict) and not is_validation:
                shaped = result.get("reward_extra", {})
                if isinstance(shaped, dict):
                    click = float(shaped.get("click", 0.0))
                    wt = float(shaped.get("watch_time", 0.0))
                    result["reward_shaped"] = self.alpha_click * click + self.alpha_watchtime * wt
            return result

        return _wrapped

