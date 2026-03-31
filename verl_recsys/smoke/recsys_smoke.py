"""Smoke checks for recsys config + trainer wiring."""

from __future__ import annotations

import os
import sys

from hydra import compose, initialize_config_module

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VERL_DIST_PATH = os.path.join(_REPO_ROOT, "verl_distillation")
if _VERL_DIST_PATH not in sys.path:
    sys.path.insert(0, _VERL_DIST_PATH)

from verl_recsys.reward.manager import ensure_reward_manager
from verl_recsys.trainer.recsys_trainer import RecsysTrainerOrchestrator


def run_smoke() -> None:
    with initialize_config_module(config_module="verl_recsys.config", version_base=None):
        config = compose(config_name="recsys_dryrun")
    ensure_reward_manager(config)
    orchestrator = RecsysTrainerOrchestrator(config)
    trainer = orchestrator.build_trainer(
        tokenizer=None,
        processor=None,
        role_worker_mapping={},
        resource_pool_manager={},
        ray_worker_group_cls=object,
        reward_fn=lambda *args, **kwargs: {"reward_extra": {"click": 0.0}},
        val_reward_fn=lambda *args, **kwargs: {"reward_extra": {"click": 0.0}},
    )
    trainer.init_workers()
    trainer.fit()
    if config.reward_model.reward_manager != "multi":
        raise RuntimeError("Smoke failed: reward manager not normalized to multi.")
    print("[OK] recsys smoke checks passed.")


if __name__ == "__main__":
    run_smoke()

