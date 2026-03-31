"""Objective presets for model-agnostic GR/recsys RL."""

from __future__ import annotations

import os
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
    "gr2_dapo": ObjectivePreset(
        adv_estimator="grpo",
        policy_loss_mode="vanilla",
        use_critic=False,
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
    if objective_name != "gr2_dapo":
        return

    gr2_cfg = config.recsys.objective.get("gr2", {})

    # DAPO defaults: decoupled clipping + token-level aggregation.
    clip_low = float(gr2_cfg.get("clip_ratio_low", 0.2))
    clip_high = float(gr2_cfg.get("clip_ratio_high", 0.28))
    config.actor_rollout_ref.actor.clip_ratio_low = clip_low
    config.actor_rollout_ref.actor.clip_ratio_high = clip_high
    config.actor_rollout_ref.actor.loss_agg_mode = gr2_cfg.get("loss_agg_mode", "token-mean")

    # GR2 path uses the DAPO reward manager to support overlong shaping.
    if getattr(config, "reward_model", None) is not None:
        config.reward_model.reward_manager = "dapo"
        overlong_cfg = gr2_cfg.get("overlong_buffer", {})
        if overlong_cfg.get("enable", False):
            reward_kwargs = config.reward_model.get("reward_kwargs", {})
            reward_kwargs["overlong_buffer_cfg"] = {
                "enable": bool(overlong_cfg.get("enable", False)),
                "len": int(overlong_cfg.get("len", 0)),
                "penalty_factor": float(overlong_cfg.get("penalty_factor", 1.0)),
                "log": bool(overlong_cfg.get("log", False)),
            }
            config.reward_model.reward_kwargs = reward_kwargs

    # Register recsys-side paper-approx conditional verifiable reward as custom scorer.
    default_reward_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "reward",
        "conditional.py",
    )
    reward_module_path = gr2_cfg.get("reward_impl_path", default_reward_path)
    config.custom_reward_function = {
        "path": reward_module_path,
        "name": "compute_gr2_conditional_score",
        "reward_kwargs": {
            "order_similarity_threshold": float(gr2_cfg.get("order_similarity_threshold", 0.9)),
            "order_penalty_weight": float(gr2_cfg.get("order_penalty_weight", 0.5)),
            "min_base_reward_for_penalty": float(gr2_cfg.get("min_base_reward_for_penalty", 0.0)),
            "generated_order_key": gr2_cfg.get("generated_order_key", "generated_items"),
            "original_order_key": gr2_cfg.get("original_order_key", "original_items"),
        },
    }

    # Two-stage rollout defaults for GR2 generation.
    if gr2_cfg.get("enable_two_stage_rollout", True):
        config.actor_rollout_ref.rollout.name = "two_stage"
        config.actor_rollout_ref.rollout.enable_two_stage_rollout = True
        config.actor_rollout_ref.rollout.stage1_max_tokens = int(gr2_cfg.get("stage1_max_tokens", 1024))
        config.actor_rollout_ref.rollout.stage2_max_tokens = int(gr2_cfg.get("stage2_max_tokens", 16))
        config.actor_rollout_ref.rollout.stage2_beam_size = int(gr2_cfg.get("stage2_beam_size", 8))

