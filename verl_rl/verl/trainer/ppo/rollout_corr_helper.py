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
Rollout correction helper functions for handling policy mismatch between
rollout and training phases.

This module implements the bypass and decoupled modes for rollout correction,
as described in the latest verl research.
"""

from typing import Optional
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.trainer.ppo.core_algos_enhanced import compute_rollout_importance_weights


def apply_bypass_mode(
    batch: DataProto,
    rollout_corr_config: DictConfig,
    policy_loss_config: DictConfig,
) -> None:
    """
    Apply bypass mode for rollout correction.
    
    In bypass mode, we set old_log_probs = rollout_log_probs,
    effectively using the rollout policy as the reference for PPO.
    This creates a 2-policy setup: π_rollout and π_θ.
    
    Args:
        batch: DataProto containing batch data
        rollout_corr_config: Configuration for rollout correction
        policy_loss_config: Configuration for policy loss
    """
    # Check if rollout log probabilities are available
    if "rollout_log_probs" not in batch.batch:
        raise ValueError("rollout_log_probs not found in batch for bypass mode")
    
    # Copy rollout log probs to old log probs
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"].clone()
    
    # Compute importance sampling weights if enabled
    if rollout_corr_config.get("enable_is_weights", False):
        is_weights, is_metrics = compute_rollout_importance_weights(
            rollout_log_probs=batch.batch["rollout_log_probs"],
            old_log_probs=batch.batch["old_log_probs"],  # Same in bypass mode
            response_mask=batch.batch["response_mask"],
            is_threshold=rollout_corr_config.get("is_threshold"),
            is_threshold_lower=rollout_corr_config.get("is_threshold_lower"),
            is_level=rollout_corr_config.get("is_level", "token"),
            is_mode=rollout_corr_config.get("is_mode", "truncate"),
            is_veto_threshold=rollout_corr_config.get("is_veto_threshold"),
        )
        
        # Store importance sampling weights for policy loss
        batch.batch["rollout_is_weights"] = is_weights
        
        # Store metrics for logging
        if "metrics" not in batch.meta_info:
            batch.meta_info["metrics"] = {}
        batch.meta_info["metrics"].update(is_metrics)
    
    # Store mode information
    batch.meta_info["rollout_correction_mode"] = "bypass"


def apply_decoupled_mode(
    batch: DataProto,
    rollout_corr_config: DictConfig,
    policy_loss_config: DictConfig,
) -> None:
    """
    Apply decoupled mode for rollout correction.
    
    In decoupled mode, we recompute old_log_probs as a proximal anchor,
    creating a 3-policy setup: π_rollout, π_old, π_θ.
    The old_log_probs serve as a stable reference during mini-batch updates.
    
    Args:
        batch: DataProto containing batch data
        rollout_corr_config: Configuration for rollout correction
        policy_loss_config: Configuration for policy loss
    """
    # old_log_probs should already be computed by the trainer
    if "old_log_probs" not in batch.batch:
        raise ValueError("old_log_probs not found in batch for decoupled mode")
    
    # Compute importance sampling weights if enabled and rollout log probs are available
    if (rollout_corr_config.get("enable_is_weights", False) and 
        "rollout_log_probs" in batch.batch):
        
        is_weights, is_metrics = compute_rollout_importance_weights(
            rollout_log_probs=batch.batch["rollout_log_probs"],
            old_log_probs=batch.batch["old_log_probs"],
            response_mask=batch.batch["response_mask"],
            is_threshold=rollout_corr_config.get("is_threshold"),
            is_threshold_lower=rollout_corr_config.get("is_threshold_lower"),
            is_level=rollout_corr_config.get("is_level", "token"),
            is_mode=rollout_corr_config.get("is_mode", "truncate"),
            is_veto_threshold=rollout_corr_config.get("is_veto_threshold"),
        )
        
        # Store importance sampling weights for policy loss
        batch.batch["rollout_is_weights"] = is_weights
        
        # Store metrics for logging
        if "metrics" not in batch.meta_info:
            batch.meta_info["metrics"] = {}
        batch.meta_info["metrics"].update(is_metrics)
    
    # Store mode information
    batch.meta_info["rollout_correction_mode"] = "decoupled"


def compute_rollout_mismatch_metrics(
    batch: DataProto,
) -> dict[str, float]:
    """
    Compute metrics for rollout policy mismatch analysis.
    
    Args:
        batch: DataProto containing batch data with rollout and current log probs
    
    Returns:
        Dictionary of mismatch metrics
    """
    metrics = {}
    
    if "rollout_log_probs" in batch.batch and "old_log_probs" in batch.batch:
        rollout_log_probs = batch.batch["rollout_log_probs"]
        old_log_probs = batch.batch["old_log_probs"]
        response_mask = batch.batch["response_mask"]
        
        # Compute KL divergence between rollout and current policies
        kl_div = rollout_log_probs - old_log_probs  # KL(π_rollout || π_current)
        kl_div = kl_div * response_mask
        
        # Compute statistics
        valid_kl = kl_div[response_mask > 0]
        if len(valid_kl) > 0:
            metrics.update({
                "rollout_mismatch/kl_mean": valid_kl.mean().item(),
                "rollout_mismatch/kl_std": valid_kl.std().item(),
                "rollout_mismatch/kl_max": valid_kl.max().item(),
                "rollout_mismatch/kl_min": valid_kl.min().item(),
            })
        
        # Compute ratio statistics
        ratio = torch.exp(old_log_probs - rollout_log_probs)  # π_current / π_rollout
        valid_ratio = ratio[response_mask > 0]
        if len(valid_ratio) > 0:
            metrics.update({
                "rollout_mismatch/ratio_mean": valid_ratio.mean().item(),
                "rollout_mismatch/ratio_std": valid_ratio.std().item(),
                "rollout_mismatch/ratio_max": valid_ratio.max().item(),
                "rollout_mismatch/ratio_min": valid_ratio.min().item(),
            })
    
    return metrics


def validate_rollout_correction_config(
    rollout_corr_config: Optional[DictConfig],
) -> bool:
    """
    Validate rollout correction configuration.
    
    Args:
        rollout_corr_config: Configuration to validate
    
    Returns:
        True if configuration is valid
    """
    if rollout_corr_config is None:
        return True
    
    # Check threshold consistency
    is_threshold = rollout_corr_config.get("is_threshold")
    is_threshold_lower = rollout_corr_config.get("is_threshold_lower")
    
    if is_threshold is not None and is_threshold_lower is not None:
        if is_threshold_lower > is_threshold:
            raise ValueError("is_threshold_lower must be <= is_threshold")
    
    # Check level
    is_level = rollout_corr_config.get("is_level", "token")
    if is_level not in ["token", "sequence", "geometric"]:
        raise ValueError(f"is_level must be one of 'token', 'sequence', 'geometric', got {is_level}")
    
    # Check mode
    is_mode = rollout_corr_config.get("is_mode", "truncate")
    if is_mode not in ["truncate", "mask"]:
        raise ValueError(f"is_mode must be one of 'truncate', 'mask', got {is_mode}")
    
    # Check veto threshold
    is_veto_threshold = rollout_corr_config.get("is_veto_threshold")
    if is_veto_threshold is not None and is_veto_threshold <= 0:
        raise ValueError("is_veto_threshold must be positive")
    
    return True