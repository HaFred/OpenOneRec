"""Task runner that reuses VeRL PPO runtime for recsys."""

from __future__ import annotations

import os
import socket
from pprint import pprint

from omegaconf import OmegaConf

from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.config import validate_config
from verl.utils.fs import copy_to_local

from verl_recsys.trainer.recsys_trainer import RecsysTrainerOrchestrator


class RecsysTaskRunner(TaskRunner):
    """Task runner with recsys orchestration hooks."""

    def run(self, config):
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        orchestrator = RecsysTrainerOrchestrator(config)
        objective_cfg = config.recsys.objective
        if not objective_cfg.get("use_critic", True):
            config.critic.ppo_mini_batch_size = 0
            config.critic.ppo_micro_batch_size = 0

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        if objective_cfg.get("use_critic", True):
            self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        if objective_cfg.get("use_reference_policy", True):
            self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        tokenizer = None
        processor = None
        if not config.recsys.training.get("smoke_build_only", False):
            local_path = copy_to_local(
                config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
            )
            trust_remote_code = config.data.get("trust_remote_code", False)
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = self.init_resource_pool_mgr(config)

        trainer = orchestrator.build_trainer(
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

        trainer.init_workers()
        orchestrator.context.rollout_server.begin_step()
        orchestrator.context.rollout_server.mark_event("fit_start")
        trainer.fit()
        orchestrator.context.rollout_server.mark_event("fit_end")
        rollout_metrics = orchestrator.context.rollout_server.end_step()
        if rollout_metrics:
            print(f"rollout profile metrics: {rollout_metrics}")
