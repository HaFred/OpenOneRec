"""Reward manager bootstrap for recsys training."""

from __future__ import annotations

from typing import Any


def register_recsys_reward_manager() -> None:
    """Register the multi-reward manager if available."""
    try:
        import multireward_mgr_support.reward_manager.register_multi  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive import guard
        raise RuntimeError(
            "Failed to register `multi` reward manager. Ensure multireward_mgr_support is importable."
        ) from exc


def ensure_reward_manager(config: Any) -> None:
    """Normalize config so recsys jobs default to multi-reward manager."""
    if getattr(config, "reward_model", None) is None:
        return
    manager_name = config.reward_model.get("reward_manager", None)
    if manager_name in (None, "", "multi"):
        register_recsys_reward_manager()
        config.reward_model.reward_manager = "multi"
