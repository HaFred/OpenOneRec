"""Recsys trainer orchestration on top of VeRL trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verl.trainer.main_ppo import create_rl_sampler
from verl.utils.dataset.rl_dataset import collate_fn

from verl_recsys.adapters.registry import AdapterBundle, build_adapter_bundle
from verl_recsys.agent.loop_worker import AgentLoopWorker
from verl_recsys.objectives.registry import apply_objective_preset
from verl_recsys.checkpoint.engine import CheckpointEngineConfig, CheckpointEngineManager
from verl_recsys.reward.manager import ensure_reward_manager
from verl_recsys.rollout.server import RolloutServerAdapter, RolloutServerConfig


class RecsysSmokeTrainer:
    """No-op trainer used for config/runtime smoke checks."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def init_workers(self) -> None:
        return

    def fit(self) -> None:
        return


def create_recsys_dataset(
    data_paths,
    data_config,
    tokenizer,
    processor,
    adapter_bundle: AdapterBundle,
    is_train=True,
    max_samples: int = -1,
):
    """Recsys-aware dataset resolution with fallback to RLHFDataset."""
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.import_utils import load_extern_type

    dataset_cls = RLHFDataset
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' "
                "must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )
    return adapter_bundle.dataset.wrap_dataset(dataset, is_train=is_train)


@dataclass
class RecsysExecutionContext:
    rollout_server: RolloutServerAdapter
    checkpoint_engine: CheckpointEngineManager
    agent_loop_worker: AgentLoopWorker


class RecsysTrainerOrchestrator:
    """Build and execute a recsys trainer using existing VeRL implementations."""

    def __init__(self, config: Any):
        self.config = config
        apply_objective_preset(self.config)
        self.adapter_bundle = build_adapter_bundle(self.config)
        self.adapter_bundle.actor.prepare_runtime(self.config)
        rollout_cfg = config.recsys.rollout
        ckpt_cfg = config.recsys.checkpoint
        self.context = RecsysExecutionContext(
            rollout_server=RolloutServerAdapter(
                RolloutServerConfig(
                    backend=rollout_cfg.get("backend", "vllm"),
                    mode=rollout_cfg.get("mode", "sync"),
                    enable_router_replay=rollout_cfg.get("enable_router_replay", False),
                    enable_profile=rollout_cfg.get("enable_profile", False),
                )
            ),
            checkpoint_engine=CheckpointEngineManager(
                CheckpointEngineConfig(
                    backend=ckpt_cfg.get("backend", "local_fs"),
                    manager_enabled=ckpt_cfg.get("manager_enabled", True),
                )
            ),
            agent_loop_worker=AgentLoopWorker(
                allow_multi_output=config.recsys.agent_loop.get("allow_multi_output", True)
            ),
        )

    def _select_trainer_cls(self):
        if self.config.recsys.training.get("smoke_build_only", False):
            return RecsysSmokeTrainer
        mode = self.config.recsys.training.get("mode", "onpolicy_distill")
        if mode == "gr2_dapo":
            from verl_recsys.trainer.gr2_dapo_trainer import GR2DapoTrainer

            return GR2DapoTrainer
        if mode == "onpolicy_distill":
            try:
                from recipe.onpolicy_distill.onpolicy_distill_trainer import RayOnPolicyDistillTrainer

                return RayOnPolicyDistillTrainer
            except Exception:
                # Fallback keeps recsys entrypoint runnable when distill recipe
                # package path is not available in the current environment.
                from verl.trainer.ppo.ray_trainer import RayPPOTrainer

                return RayPPOTrainer
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        return RayPPOTrainer

    def build_trainer(
        self,
        tokenizer: Any,
        processor: Any,
        role_worker_mapping: dict,
        resource_pool_manager: Any,
        ray_worker_group_cls: Any,
        reward_fn: Any,
        val_reward_fn: Any,
    ) -> Any:
        ensure_reward_manager(self.config)
        trainer_cls = self._select_trainer_cls()
        if trainer_cls is RecsysSmokeTrainer:
            return trainer_cls(
                config=self.config,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
            )

        train_dataset = create_recsys_dataset(
            self.config.data.train_files,
            self.config.data,
            tokenizer,
            processor,
            adapter_bundle=self.adapter_bundle,
            is_train=True,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        val_dataset = create_recsys_dataset(
            self.config.data.val_files,
            self.config.data,
            tokenizer,
            processor,
            adapter_bundle=self.adapter_bundle,
            is_train=False,
            max_samples=self.config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(self.config.data, train_dataset)
        reward_fn = self.adapter_bundle.reward.wrap_reward_fn(reward_fn)
        val_reward_fn = self.adapter_bundle.reward.wrap_reward_fn(val_reward_fn, is_validation=True)

        trainer = trainer_cls(
            config=self.config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        return trainer
