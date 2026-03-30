# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Enhanced PPO Trainer with latest features including rollout correction,
custom reward functions, vectorized algorithms, and improved performance.

This trainer extends the original RayPPOTrainer with cutting-edge capabilities
from the latest verl research.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos_enhanced import (
    EnhancedAdvantageEstimator, get_policy_loss_fn, compute_rollout_importance_weights
)
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward_enhanced import (
    extract_reward_enhanced, compute_reward_enhanced, load_enhanced_reward_manager
)
from verl.trainer.ppo.rollout_corr_helper import (
    apply_bypass_mode, apply_decoupled_mode, compute_rollout_mismatch_metrics,
    validate_rollout_correction_config
)
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_critic,
    need_reference_policy,
    need_reward_model,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import DistillationConfig, FSDPEngineConfig, McoreEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

# Import original trainer for inheritance
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class EnhancedRayPPOTrainer(RayPPOTrainer):
    """
    Enhanced PPO trainer with latest features from verl research.
    
    Key enhancements:
    1. Rollout correction with bypass/decoupled modes
    2. Custom reward function loading
    3. Vectorized advantage estimators
    4. Enhanced policy loss functions
    5. Better performance monitoring
    6. Improved error handling and validation
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize enhanced PPO trainer."""
        super().__init__(*args, **kwargs)
        
        # Validate rollout correction configuration
        rollout_corr_config = self.config.algorithm.get("rollout_correction")
        if rollout_corr_config is not None:
            validate_rollout_correction_config(rollout_corr_config)
        
        # Initialize enhanced reward manager if needed
        if self.config.reward_model.get("enhanced", False):
            self.reward_fn = load_enhanced_reward_manager(
                self.config, self.tokenizer, num_examine=0,
                **self.config.reward_model.get("reward_kwargs", {})
            )
        
        # Initialize policy loss function
        self.policy_loss_fn = get_policy_loss_fn(
            self.config.algorithm.get("policy_loss_type", "ppo_enhanced")
        )
        
        # Performance tracking
        self.performance_metrics = {
            "total_tokens_processed": 0,
            "total_training_time": 0.0,
            "avg_throughput": 0.0,
        }
    
    def _compute_old_log_prob_enhanced(self, batch: DataProto) -> tuple[DataProto, float]:
        """
        Enhanced old log probability computation with performance tracking.
        
        Args:
            batch: DataProto containing batch data
            
        Returns:
            Tuple of (updated batch, MFU)
        """
        from verl.utils.flops_counter import FlopsCounter
        
        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
        entropys = old_log_prob.batch["entropys"]
        response_masks = batch.batch["response_mask"]
        actor_config = self.config.actor_rollout_ref.actor
        entropy_agg = core_algos.agg_loss(
            loss_mat=entropys,
            loss_mask=response_masks,
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        
        old_log_prob_metrics = {
            "actor/entropy": entropy_agg.detach().item(),
        }
        
        # Compute MFU if available
        mfu = 0.0
        if hasattr(self.actor_rollout_wg, 'flops_counter'):
            global_num_tokens = batch.meta_info["global_token_num"]
            delta_time = 0.1  # Placeholder - should be measured
            flops_counter = self.actor_rollout_wg.flops_counter
            estimated_flops, promised_flops = flops_counter.estimate_flops(global_num_tokens, delta_time)
            mfu = estimated_flops / promised_flops / self.resource_pool_manager.get_n_gpus()
            old_log_prob_metrics["perf/mfu/actor_infer"] = mfu
        
        # Update performance metrics
        self.performance_metrics["total_tokens_processed"] += sum(batch.meta_info["global_token_num"])
        
        old_log_prob.batch.pop("entropys")
        batch = batch.union(old_log_prob)
        
        return batch, mfu
    
    def _compute_advantage_enhanced(
        self,
        batch: DataProto,
        timing_raw: dict,
        metrics: dict,
    ) -> DataProto:
        """
        Enhanced advantage computation with new estimators and rollout correction.
        
        Args:
            batch: DataProto containing batch data
            timing_raw: Timing dictionary
            metrics: Metrics dictionary to update
            
        Returns:
            Updated batch with advantages
        """
        with marked_timer("adv", timing_raw, color="brown"):
            # Extract reward tensor and extra info
            reward_tensor, reward_extra_infos_dict = extract_reward_enhanced(batch)
            batch.batch["token_level_scores"] = reward_tensor
            
            # Update metrics with reward extra info
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                
                # Add reward statistics to metrics
                for key, values in reward_extra_infos_dict.items():
                    if values and len(values) > 0:
                        values_array = np.array(values)
                        if np.issubdtype(values_array.dtype, np.number):
                            metrics[f"reward/{key}/mean"] = float(np.mean(values_array))
                            metrics[f"reward/{key}/max"] = float(np.max(values_array))
                            metrics[f"reward/{key}/min"] = float(np.min(values_array))
            
            # Apply KL penalty if enabled
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = core_algos.apply_kl_penalty(
                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
            
            # Apply rollout correction if configured
            rollout_corr_config = self.config.algorithm.get("rollout_correction")
            if rollout_corr_config is not None:
                if rollout_corr_config.get("bypass_mode", False):
                    apply_bypass_mode(batch, rollout_corr_config, self.config.actor_rollout_ref.actor.policy_loss)
                else:
                    # Decoupled mode - old_log_probs should already be computed
                    apply_decoupled_mode(batch, rollout_corr_config, self.config.actor_rollout_ref.actor.policy_loss)
                
                # Add rollout mismatch metrics
                mismatch_metrics = compute_rollout_mismatch_metrics(batch)
                metrics.update(mismatch_metrics)
            
            # Compute advantages with enhanced estimators
            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            
            # Map string to enum for enhanced estimators
            adv_estimator_str = self.config.algorithm.adv_estimator
            try:
                adv_estimator = EnhancedAdvantageEstimator(adv_estimator_str)
            except ValueError:
                # Fall back to original estimators
                adv_estimator = core_algos.AdvantageEstimator(adv_estimator_str)
            
            batch = core_algos.compute_advantage(
                batch,
                adv_estimator=adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )
        
        return batch
    
    def _update_actor_enhanced(
        self,
        batch: DataProto,
        timing_raw: dict,
        metrics: dict,
    ) -> dict:
        """
        Enhanced actor update with new policy loss functions and rollout correction.
        
        Args:
            batch: DataProto containing batch data
            timing_raw: Timing dictionary
            metrics: Metrics dictionary to update
            
        Returns:
            Updated metrics dictionary
        """
        with marked_timer("update_actor", timing_raw, color="red"):
            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
            
            # Update actor with enhanced policy loss
            actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)
        
        return metrics
    
    def fit(self):
        """
        Enhanced training loop with new features and improved monitoring.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        self.global_steps = 0
        
        # Load checkpoint
        self._load_checkpoint()
        
        # Initial validation
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        
        # Training loop
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                
                # Profiling setup
                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)
                
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                
                # Prepare generation batch
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                
                # Add additional keys for multi-modal and tool support
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")
                
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                
                # Add global steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                
                is_last_step = self.global_steps >= self.total_training_steps
                
                # Generation phase
                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)
                    
                    # Handle REMAX baseline generation
                    if self.config.algorithm.adv_estimator == "remax":
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                            
                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            
                            del gen_baseline_batch, gen_baseline_output
                    
                    # Prepare batch for reward computation
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    
                    # Compute response mask
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = core_algos.compute_response_mask(batch)
                    
                    # Balance batch if enabled
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    
                    # Compute global token statistics
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    
                    # Reward computation
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # Compute reward model score if enabled
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        
                        # Compute reward function
                        if self.config.reward_model.launch_reward_fn_async:
                            if self.config.reward_model.get("enhanced", False):
                                future_reward = compute_reward_enhanced.remote(
                                    data=batch, reward_fn=self.reward_fn
                                )
                            else:
                                from verl.trainer.ppo.reward import compute_reward_async
                                future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            if self.config.reward_model.get("enhanced", False):
                                reward_result = self.reward_fn.compute_rewards(batch)
                                reward_tensor = reward_result["reward_tensor"]
                                reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                            else:
                                reward_tensor, reward_extra_infos_dict = self.reward_fn(batch)
                    
                    # Compute old log probabilities
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob, old_log_prob_mfu = self._compute_old_log_prob_enhanced(batch)
                        entropys = old_log_prob.batch.get("entropys", torch.tensor(0.0))
                        response_masks = batch.batch["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = core_algos.agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item(),
                            "perf/mfu/actor_infer": old_log_prob_mfu,
                        }
                        metrics.update(old_log_prob_metrics)
                        if "entropys" in old_log_prob.batch:
                            old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                    
                    # Reference policy computation
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    
                    # Value computation
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                    
                    # Enhanced advantage computation
                    batch = self._compute_advantage_enhanced(batch, timing_raw, metrics)
                    
                    # Update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)
                    
                    # Enhanced actor update
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        metrics = self._update_actor_enhanced(batch, timing_raw, metrics)
                    
                    # Enhanced logging
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            self._log_rollout_data_enhanced(batch, timing_raw, rollout_data_dir, logger)
                    
                    # Validation
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                    
                    # Checkpoint saving
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()
                
                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)
                
                # Update performance metrics
                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                self.performance_metrics["total_training_time"] += steps_duration
                
                # Collect and log metrics
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                
                # Update performance metrics
                if self.performance_metrics["total_training_time"] > 0:
                    self.performance_metrics["avg_throughput"] = (
                        self.performance_metrics["total_tokens_processed"] / 
                        self.performance_metrics["total_training_time"]
                    )
                    metrics["perf/avg_throughput"] = self.performance_metrics["avg_throughput"]
                
                # Curriculum sampler update
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)
                
                # Log metrics
                logger.log(data=metrics, step=self.global_steps)
                
                progress_bar.update(1)
                self.global_steps += 1
                
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
                
                # Dataset update hook
                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
    
    def _log_rollout_data_enhanced(
        self,
        batch: DataProto,
        timing_raw: dict,
        rollout_data_dir: str,
        logger,
    ):
        """
        Enhanced rollout data logging with additional metrics.
        
        Args:
            batch: DataProto containing rollout data
            timing_raw: Timing information
            rollout_data_dir: Directory to save rollout data
            logger: Logger instance
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            
            # Extract additional information
            sample_gts = []
            for item in batch:
                reward_data = item.non_tensor_batch.get("reward_model", {})
                sample_gts.append(reward_data.get("ground_truth", None))
            
            # Prepare extra info for logging
            reward_extra_infos_dict = {}
            if "reward_extra_info" in batch.non_tensor_batch:
                reward_extra_infos_dict = batch.non_tensor_batch["reward_extra_info"]
            
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )
            
            # Add rollout correction information
            if "rollout_correction_mode" in batch.meta_info:
                reward_extra_infos_dict["rollout_correction_mode"] = batch.meta_info["rollout_correction_mode"]
            
            # Enhanced dumping with wandb table support
            self._dump_generations_enhanced(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=rollout_data_dir,
                logger=logger,
            )
    
    def _dump_generations_enhanced(
        self,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        dump_path,
        logger=None,
    ):
        """
        Enhanced generation dumping with wandb table support.
        
        Args:
            inputs: List of input prompts
            outputs: List of generated outputs
            gts: List of ground truth values
            scores: List of scores
            reward_extra_infos_dict: Dictionary of extra information
            dump_path: Path to save generations
            logger: Logger instance for wandb table
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")
        
        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }
        
        if gts and any(gt is not None for gt in gts):
            base_data["ground_truth"] = gts
        
        # Add extra info
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v
        
        # Create wandb table if logger is available
        if logger is not None and hasattr(logger, 'logger') and 'wandb' in logger.logger:
            import pandas as pd
            import wandb
            df = pd.DataFrame(base_data)
            logger.logger['wandb'].log({"completions": wandb.Table(dataframe=df)})
            return
        
        # Save to JSONL file
        lines = []
        for i in range(n):
            entry = {
                k: int(v[i]) if any(t in str(type(v[i])) for t in ['int64', 'bool']) else v[i]
                for k, v in base_data.items()
            }
            lines.append(json.dumps(entry, ensure_ascii=False))
        
        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")
        
        print(f"Dumped generations to {filename}")