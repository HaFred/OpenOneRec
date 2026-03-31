"""Adapter registry for dataset/actor/reward recsys components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verl_recsys.adapters.base import ActorAdapter, DatasetAdapter, RewardAdapter
from verl_recsys.adapters.defaults import (
    DatasetNormalizationAdapter,
    HSTUActorAdapter,
    SASRecActorAdapter,
    WeightedRewardAdapter,
)


@dataclass
class AdapterBundle:
    dataset: DatasetAdapter
    actor: ActorAdapter
    reward: RewardAdapter


def _select_actor_adapter(config: Any) -> ActorAdapter:
    family = config.recsys.model.get("family", "hstu").lower()
    if family == "sasrec":
        return SASRecActorAdapter(rollout_backend=config.recsys.rollout.get("backend", "vllm"))
    return HSTUActorAdapter(rollout_backend=config.recsys.rollout.get("backend", "vllm"))


def build_adapter_bundle(config: Any) -> AdapterBundle:
    family = config.recsys.model.get("family", "hstu")
    reward_cfg = config.recsys.get("reward_adapter", {})
    return AdapterBundle(
        dataset=DatasetNormalizationAdapter(model_family=family),
        actor=_select_actor_adapter(config),
        reward=WeightedRewardAdapter(
            alpha_click=float(reward_cfg.get("alpha_click", 1.0)),
            alpha_watchtime=float(reward_cfg.get("alpha_watchtime", 0.0)),
        ),
    )

