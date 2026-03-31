"""Objective presets for model-agnostic GR/recsys RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObjectivePreset:
    adv_estimator: str
    policy_loss_mode: str
    use_critic: bool
    use_reference_policy: bool


_PRESETS = {
    "distill_hybrid": ObjectivePreset(
        adv_estimator="on_policy_distill",
        policy_loss_mode="vanilla",
        use_critic=True,
        use_reference_policy=True,
    ),
    "policy_only_grpo": ObjectivePreset(
        adv_estimator="grpo",
        policy_loss_mode="vanilla",
        use_critic=False,
        use_reference_policy=True,
    ),
    "actor_critic_gae": ObjectivePreset(
        adv_estimator="gae",
        policy_loss_mode="vanilla",
        use_critic=True,
        use_reference_policy=True,
    ),
}


def apply_objective_preset(config: Any) -> None:
    """Normalize algorithm knobs from a single recsys objective profile."""
    objective_name = config.recsys.objective.get("name", "distill_hybrid")
    preset = _PRESETS.get(objective_name, _PRESETS["distill_hybrid"])
    config.algorithm.adv_estimator = preset.adv_estimator
    config.actor_rollout_ref.actor.loss_mode = preset.policy_loss_mode
    config.actor_rollout_ref.actor.use_kl_loss = config.actor_rollout_ref.actor.get("use_kl_loss", False)
    config.recsys.objective.use_critic = preset.use_critic
    config.recsys.objective.use_reference_policy = preset.use_reference_policy

