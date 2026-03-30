# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Enhanced core functions to implement PPO algorithms with latest features.
This file extends the original core_algos.py with new advantage estimators,
policy loss functions, and rollout correction capabilities.
"""

__all__ = [
    "register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator",
    "register_policy_loss", "get_policy_loss_fn", "compute_rollout_importance_weights"
]

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

# Import existing registry and functions from original core_algos
from verl.trainer.ppo.core_algos import (
    ADV_ESTIMATOR_REGISTRY, register_adv_est, get_adv_estimator_fn,
    AdvantageEstimator, AdaptiveKLController, FixedKLController, get_kl_controller,
    agg_loss
)

# Enhanced Advantage Estimator Enum with new algorithms
class EnhancedAdvantageEstimator(str, Enum):
    """Extended advantage estimator enumeration with latest algorithms."""
    
    # Original estimators
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    
    # New vectorized implementations
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    
    # New baseline methods
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"
    
    # New algorithms
    GDPO = "gdpo"
    ON_POLICY_DISTILL = "on_policy_distill"


# Policy Loss Registry
PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
        torch.Tensor | None,  # rollout_is_weights
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name."""
    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func
    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name."""
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


@register_adv_est(EnhancedAdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO implementation for improved performance.
    
    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: whether to normalize by std within group
        config: algorithm configuration
    
    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)  # (bs,)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon)
        
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(EnhancedAdvantageEstimator.RLOO_VECTORIZED)
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized RLOO implementation for improved performance.
    
    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: algorithm configuration
    
    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)  # (bs,)
        g = as_torch_index(index, device=scores.device)
        mean_g, _, _ = group_mean_std(scores, g, eps=epsilon)
        
        # RLOO: leave-one-out advantage
        # For each sample i: a_i = r_i - (sum(g) - r_i) / (|g| - 1)
        # Simplified: a_i = |g|/(|g|-1) * r_i - sum(g)/(|g|-1)
        group_sizes = torch.bincount(g, minlength=g.max().item() + 1)
        n_g = group_sizes[g]
        
        if n_g.min() > 1:  # Only apply RLOO if groups have >1 samples
            advantages_scalar = (n_g.float() / (n_g - 1).float()) * scores - mean_g[g] * n_g.float() / (n_g - 1).float()
        else:
            advantages_scalar = scores - mean_g[g]
        
        advantages = advantages_scalar.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(EnhancedAdvantageEstimator.OPTIMAL_TOKEN_BASELINE)
def compute_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage using optimal token-level baseline.
    
    This method computes token-level baselines that minimize the variance
    of the advantage estimates, following the optimal baseline theory.
    
    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        config: algorithm configuration
    
    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    with torch.no_grad():
        # Compute optimal baseline at each token position
        # b_t* = E[R_t | history_t]
        # For implementation, we use running average per position
        
        # Compute position-wise baselines
        position_rewards = token_level_rewards * response_mask
        position_counts = response_mask.sum(dim=0, keepdim=True)
        position_means = position_rewards.sum(dim=0, keepdim=True) / (position_counts + 1e-8)
        
        # Compute advantages
        advantages = (token_level_rewards - position_means) * response_mask
        returns = token_level_rewards  # For on-policy, returns = rewards
        
        return advantages, returns


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Group Policy Gradient (GPG) loss implementation.
    
    GPG uses log probabilities directly without ratio computation,
    which can be more stable for certain tasks.
    
    Args:
        old_log_prob: (bs, response_length) - unused in GPG
        log_prob: (bs, response_length)
        advantages: (bs, response_length)
        response_mask: (bs, response_length)
        loss_agg_mode: aggregation mode
        config: actor configuration
        rollout_log_probs: rollout log probabilities - unused in GPG
        rollout_is_weights: importance sampling weights
    
    Returns:
        pg_loss: scalar tensor
        pg_metrics: dict of metrics
    """
    pg_losses = -log_prob * advantages
    
    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    
    pg_loss = agg_loss(
        loss_mat=pg_losses, 
        loss_mask=response_mask, 
        loss_agg_mode=loss_agg_mode,
        **(config.global_batch_info if config else {})
    )
    
    # Return empty metrics for compatibility
    pg_metrics = {}
    return pg_loss, pg_metrics


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Clip-Cov policy loss with covariance-based clipping.
    
    This method clips based on the covariance between advantages
    and log probabilities, providing adaptive clipping.
    
    Args:
        old_log_prob: (bs, response_length)
        log_prob: (bs, response_length)
        advantages: (bs, response_length)
        response_mask: (bs, response_length)
        loss_agg_mode: aggregation mode
        config: actor configuration
        rollout_log_probs: rollout log probabilities
        rollout_is_weights: importance sampling weights
    
    Returns:
        pg_loss: scalar tensor
        pg_metrics: dict of metrics
    """
    assert config is not None, "Clip-Cov requires config for parameters"
    
    # Extract configuration parameters
    clip_cov_ratio = getattr(config.policy_loss, 'clip_cov_ratio', 0.0002)
    clip_ratio = getattr(config, 'clip_ratio', 0.2)
    clip_cov_ub = getattr(config.policy_loss, 'clip_cov_ub', 5.0)
    clip_cov_lb = getattr(config.policy_loss, 'clip_cov_lb', 1.0)
    
    assert clip_cov_ratio > 0, "clip_cov_ratio should be larger than 0."
    
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    
    # Standard PPO clipping
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)
    
    # Compute covariance
    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf
    
    # Select tokens to clip based on covariance
    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)
    
    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)
    
    corr = torch.ones_like(advantages)
    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0
    
    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)
    
    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    
    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    
    pg_loss = agg_loss(
        loss_mat=pg_losses, 
        loss_mask=response_mask, 
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info
    )
    
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }
    
    return pg_loss, pg_metrics


def compute_rollout_importance_weights(
    rollout_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    is_threshold: Optional[float] = None,
    is_threshold_lower: Optional[float] = None,
    is_level: str = "token",
    is_mode: str = "truncate",
    is_veto_threshold: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute rollout importance sampling weights for policy mismatch correction.
    
    Args:
        rollout_log_probs: (bs, response_length) - log probs from rollout policy
        old_log_probs: (bs, response_length) - log probs from current policy
        response_mask: (bs, response_length)
        is_threshold: upper threshold for IS weights
        is_threshold_lower: lower threshold for IS weights
        is_level: aggregation level ("token", "sequence", "geometric")
        is_mode: bounding mode ("truncate" or "mask")
        is_veto_threshold: per-token veto threshold
    
    Returns:
        is_weights: (bs, response_length) - importance sampling weights
        metrics: dict of IS weight statistics
    """
    with torch.no_grad():
        # Compute IS weights: w = π_old / π_rollout
        log_is_weights = old_log_probs - rollout_log_probs
        is_weights = torch.exp(log_is_weights)
        
        # Apply per-token veto threshold
        if is_veto_threshold is not None:
            veto_mask = (is_weights > is_veto_threshold) & (response_mask > 0)
            is_weights = torch.where(veto_mask, torch.tensor(0.0, device=is_weights.device), is_weights)
        
        # Apply thresholds based on mode
        if is_threshold is not None:
            if is_threshold_lower is None:
                is_threshold_lower = 1.0 / is_threshold
            
            if is_mode == "truncate":
                is_weights = torch.clamp(is_weights, is_threshold_lower, is_threshold)
            elif is_mode == "mask":
                mask = (is_weights >= is_threshold_lower) & (is_weights <= is_threshold)
                is_weights = torch.where(mask, is_weights, torch.tensor(0.0, device=is_weights.device))
        
        # Apply response mask
        is_weights = is_weights * response_mask
        
        # Aggregate based on level
        if is_level == "sequence":
            seq_is_weights = is_weights.sum(dim=-1) / (response_mask.sum(dim=-1) + 1e-8)
            is_weights = seq_is_weights.unsqueeze(-1) * response_mask
        elif is_level == "geometric":
            # Geometric mean across tokens
            masked_is_weights = is_weights + (1 - response_mask)  # Avoid log(0)
            geo_mean = torch.exp(torch.log(masked_is_weights + 1e-8).sum(dim=-1) / (response_mask.sum(dim=-1) + 1e-8))
            is_weights = geo_mean.unsqueeze(-1) * response_mask
        
        # Compute metrics
        valid_is_weights = is_weights[response_mask > 0]
        metrics = {}
        if len(valid_is_weights) > 0:
            metrics.update({
                "rollout_is/mean": valid_is_weights.mean().item(),
                "rollout_is/std": valid_is_weights.std().item(),
                "rollout_is/max": valid_is_weights.max().item(),
                "rollout_is/min": valid_is_weights.min().item(),
                "rollout_is/num_clipped": (valid_is_weights > (is_threshold or float('inf'))).sum().item() if is_threshold else 0,
            })
        
        return is_weights, metrics


# Enhanced PPO policy loss with rollout correction support
@register_policy_loss("ppo_enhanced")
def compute_policy_loss_ppo_enhanced(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Enhanced PPO policy loss with rollout correction support.
    
    This is the standard PPO loss extended with importance sampling weights
    for rollout correction.
    
    Args:
        old_log_prob: (bs, response_length)
        log_prob: (bs, response_length)
        advantages: (bs, response_length)
        response_mask: (bs, response_length)
        loss_agg_mode: aggregation mode
        config: actor configuration
        rollout_log_probs: rollout log probabilities
        rollout_is_weights: importance sampling weights
    
    Returns:
        pg_loss: scalar tensor
        pg_metrics: dict of metrics
    """
    assert config is not None, "PPO enhanced requires config for clip_ratio"
    
    clip_ratio = getattr(config, 'clip_ratio', 0.2)
    clip_ratio_low = getattr(config, 'clip_ratio_low', clip_ratio)
    clip_ratio_high = getattr(config, 'clip_ratio_high', clip_ratio)
    clip_ratio_c = getattr(config, 'clip_ratio_c', 3.0)
    
    assert clip_ratio_c > 1.0, "clip_ratio_c should be greater than 1.0"
    
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    
    # Standard PPO clipping
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    
    # Dual clipping
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    
    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    
    pg_loss = agg_loss(
        loss_mat=pg_losses, 
        loss_mask=response_mask, 
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info
    )
    
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    
    return pg_loss, pg_metrics