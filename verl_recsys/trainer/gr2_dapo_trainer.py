"""GR2 DAPO trainer wrapper with two-stage expansion handling."""

from __future__ import annotations

import importlib.util
import os
from typing import Any


def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from path: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GR2DapoTrainer:
    """Adapter around OneRec's PPO trainer using recsys/distillation runtime."""

    def __init__(
        self,
        config: Any,
        tokenizer: Any,
        role_worker_mapping: dict,
        resource_pool_manager: Any,
        ray_worker_group_cls: Any,
        processor: Any = None,
        reward_fn: Any = None,
        val_reward_fn: Any = None,
        train_dataset: Any = None,
        val_dataset: Any = None,
        collate_fn: Any = None,
        train_sampler: Any = None,
    ):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        onerec_trainer_path = os.path.join(repo_root, "verl_rl", "recipe", "onerec", "onerec_ray_trainer.py")
        module = _load_module_from_path("recsys_gr2_onerec_trainer_module", onerec_trainer_path)

        one_rec_role = module.Role
        one_rec_resource_pool_mgr_cls = module.ResourcePoolManager
        one_rec_trainer_cls = module.RayPPOTrainer

        mapped_role_workers = {}
        for role_key, worker_cls in role_worker_mapping.items():
            role_name = getattr(role_key, "name", None)
            if role_name is None or not hasattr(one_rec_role, role_name):
                continue
            mapped_role_workers[getattr(one_rec_role, role_name)] = worker_cls

        mapped_pool_mapping = {}
        for role_key, pool_name in resource_pool_manager.mapping.items():
            role_name = getattr(role_key, "name", None)
            if role_name is None or not hasattr(one_rec_role, role_name):
                continue
            mapped_pool_mapping[getattr(one_rec_role, role_name)] = pool_name

        mapped_resource_pool_mgr = one_rec_resource_pool_mgr_cls(
            resource_pool_spec=resource_pool_manager.resource_pool_spec,
            mapping=mapped_pool_mapping,
        )

        self._impl = one_rec_trainer_cls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=mapped_role_workers,
            resource_pool_manager=mapped_resource_pool_mgr,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )

    def __getattr__(self, item: str):
        return getattr(self._impl, item)
