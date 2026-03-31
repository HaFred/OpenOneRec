"""Runtime acceleration toggles for recsys RL jobs."""

from __future__ import annotations

from typing import Any


def apply_acceleration_config(config: Any) -> None:
    """Apply optional acceleration flags in one place."""
    accel = config.recsys.get("acceleration", {})
    if accel.get("enable_async_rollout", False):
        config.recsys.training.enable_async = True
        config.recsys.rollout.mode = "async"
        config.actor_rollout_ref.rollout.mode = "async"

    if accel.get("enable_async_reward", False):
        config.trainer.launch_reward_fn_async = True

    if accel.get("enable_fused_kernels", False):
        config.actor_rollout_ref.actor.use_fused_kernels = True

    if accel.get("enable_sequence_balance", False):
        config.data.balance_batch = True

