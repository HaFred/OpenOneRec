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
Enhanced reward system with custom reward function support and improved
reward processing capabilities.

This module extends the original reward.py with dynamic reward function
loading, advanced reward aggregation, and better error handling.
"""

import importlib.util
import inspect
import os
import sys
from functools import partial
from typing import Any, Callable, Optional

import ray
from omegaconf import DictConfig

from verl import DataProto
from verl.trainer.ppo.reward import AbstractRewardManager, load_reward_manager
from verl.utils.tracking import tqdm_bridge


class EnhancedRewardManager(AbstractRewardManager):
    """
    Enhanced reward manager with support for custom reward functions,
    reward aggregation, and advanced error handling.
    """
    
    def __init__(
        self,
        tokenizer,
        num_examine: int = 1,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "reward",
        reward_aggregation: str = "sum",
        custom_reward_config: Optional[DictConfig] = None,
        **kwargs
    ):
        """
        Initialize enhanced reward manager.
        
        Args:
            tokenizer: Tokenizer for text processing
            num_examine: Number of samples to examine for debugging
            compute_score: Custom reward function
            reward_fn_key: Key to store reward in data
            reward_aggregation: Method to aggregate rewards ("sum", "mean", "max")
            custom_reward_config: Configuration for custom reward functions
            **kwargs: Additional arguments
        """
        super().__init__(tokenizer=tokenizer, num_examine=num_examine)
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_aggregation = reward_aggregation
        self.custom_reward_config = custom_reward_config or {}
        
        # Initialize reward statistics tracking
        self.reward_stats = {
            "total_calls": 0,
            "total_rewards": 0.0,
            "max_reward": float("-inf"),
            "min_reward": float("inf"),
            "error_count": 0,
        }
    
    def compute_rewards(self, data: DataProto) -> dict[str, Any]:
        """
        Compute rewards for a batch of data with enhanced error handling
        and reward aggregation.
        
        Args:
            data: DataProto containing batch data
            
        Returns:
            Dictionary with reward tensor and metadata
        """
        self.reward_stats["total_calls"] += 1
        
        try:
            # Call the reward function
            if self.compute_score is None:
                raise ValueError("No reward function provided")
            
            result = self.compute_score(data, return_dict=True)
            reward_tensor = result["reward_tensor"]
            reward_extra_info = result.get("reward_extra_info", {})
            
            # Apply reward aggregation if specified
            if self.reward_aggregation != "sum" and len(reward_tensor.shape) > 1:
                if self.reward_aggregation == "mean":
                    reward_tensor = reward_tensor.mean(dim=-1, keepdim=True)
                elif self.reward_aggregation == "max":
                    reward_tensor = reward_tensor.max(dim=-1, keepdim=True).values
                else:
                    raise ValueError(f"Unknown reward aggregation: {self.reward_aggregation}")
            
            # Update statistics
            valid_rewards = reward_tensor[reward_tensor != 0]  # Exclude padding zeros
            if len(valid_rewards) > 0:
                self.reward_stats["total_rewards"] += valid_rewards.sum().item()
                self.reward_stats["max_reward"] = max(self.reward_stats["max_reward"], valid_rewards.max().item())
                self.reward_stats["min_reward"] = min(self.reward_stats["min_reward"], valid_rewards.min().item())
            
            # Add statistics to extra info
            reward_extra_info.update({
                "reward_stats": self.reward_stats.copy(),
                "reward_aggregation": self.reward_aggregation,
            })
            
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
            
        except Exception as e:
            self.reward_stats["error_count"] += 1
            print(f"Error in reward computation: {e}")
            
            # Return zero rewards as fallback
            batch_size = data.batch.batch_size[0]
            response_length = data.batch.get("responses", data.batch.get("input_ids")).shape[1]
            zero_rewards = torch.zeros(batch_size, response_length, device=data.batch.device)
            
            return {
                "reward_tensor": zero_rewards,
                "reward_extra_info": {
                    "error": str(e),
                    "reward_stats": self.reward_stats.copy(),
                },
            }


def load_custom_reward_function(config: DictConfig) -> Optional[Callable]:
    """
    Load a custom reward function from external file.
    
    Args:
        config: Configuration containing custom_reward_function settings
        
    Returns:
        Loaded reward function or None if not configured
    """
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    
    if not file_path:
        return None
    
    function_name = reward_fn_config.get("name")
    if not function_name:
        raise ValueError("function_name must be specified in custom_reward_function config")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found")
    
    # Load the module
    spec = importlib.util.spec_from_file_location("custom_reward_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from '{file_path}'")
    
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error executing module from '{file_path}': {e}")
    
    # Get the function
    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in '{file_path}'")
    
    raw_fn = getattr(module, function_name)
    
    # Validate function signature
    if not callable(raw_fn):
        raise TypeError(f"'{function_name}' is not callable")
    
    # Check if it's an async function
    is_async = inspect.iscoroutinefunction(raw_fn)
    
    # Wrap with kwargs
    reward_kwargs = reward_fn_config.get("reward_kwargs", {})
    
    if is_async:
        return partial(_call_async_with_kwargs, raw_fn, reward_kwargs)
    else:
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)


def _call_with_kwargs(fn: Callable, kwargs: dict[str, Any], *args, **fn_kwargs):
    """Helper function to call a function with merged kwargs."""
    merged_kwargs = {**kwargs, **fn_kwargs}
    return fn(*args, **merged_kwargs)


def _call_async_with_kwargs(async_fn: Callable, kwargs: dict[str, Any], *args, **fn_kwargs):
    """Helper function to call an async function with merged kwargs."""
    import asyncio
    
    merged_kwargs = {**kwargs, **fn_kwargs}
    
    # Create a new event loop if needed
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, async_fn(*args, **merged_kwargs))
                return future.result()
        else:
            return loop.run_until_complete(async_fn(*args, **merged_kwargs))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(async_fn(*args, **merged_kwargs))


def load_enhanced_reward_manager(
    config: DictConfig,
    tokenizer,
    num_examine: int = 1,
    **reward_kwargs
) -> EnhancedRewardManager:
    """
    Load and initialize an enhanced reward manager.
    
    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer for text processing
        num_examine: Number of samples to examine
        **reward_kwargs: Additional arguments for reward manager
        
    Returns:
        Enhanced reward manager instance
    """
    # Try to load custom reward function
    custom_reward_fn = load_custom_reward_function(config)
    
    # Get reward aggregation method
    reward_aggregation = config.reward_model.get("reward_aggregation", "sum")
    
    # Create enhanced reward manager
    reward_manager = EnhancedRewardManager(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=custom_reward_fn,
        reward_aggregation=reward_aggregation,
        custom_reward_config=config.get("custom_reward_function"),
        **reward_kwargs
    )
    
    return reward_manager


@ray.remote(num_cpus=1)
def compute_reward_enhanced(
    data: DataProto,
    config: Optional[DictConfig] = None,
    tokenizer=None,
    reward_fn=None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Enhanced reward computation function for Ray remote execution.
    
    Args:
        data: DataProto containing batch data
        config: Configuration dictionary
        tokenizer: Tokenizer for text processing
        reward_fn: Reward function to use
        
    Returns:
        Tuple of reward tensor and extra info dictionary
    """
    if reward_fn is None:
        if config is None or tokenizer is None:
            raise ValueError("config and tokenizer must be provided when reward_fn is None")
        
        # Load enhanced reward manager
        reward_fn = load_enhanced_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
    
    # Compute rewards
    if hasattr(reward_fn, 'compute_rewards'):
        result = reward_fn.compute_rewards(data)
        reward_tensor = result["reward_tensor"]
        reward_extra_info = result.get("reward_extra_info", {})
    else:
        # Fallback to original interface
        result = reward_fn(data, return_dict=True)
        reward_tensor = result["reward_tensor"]
        reward_extra_info = result.get("reward_extra_info", {})
    
    return reward_tensor, reward_extra_info


def extract_reward_enhanced(batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Extract reward tensor and extra information from batch with enhanced validation.
    
    Args:
        batch: DataProto containing batch data
        
    Returns:
        Tuple of reward tensor and extra info dictionary
    """
    if "token_level_scores" not in batch.batch:
        raise ValueError("token_level_scores not found in batch")
    
    reward_tensor = batch.batch["token_level_scores"]
    reward_extra_infos_dict = {}
    
    # Extract reward extra info from non_tensor_batch
    if "reward_extra_info" in batch.non_tensor_batch:
        reward_extra_infos_dict = batch.non_tensor_batch["reward_extra_info"]
        if not isinstance(reward_extra_infos_dict, dict):
            reward_extra_infos_dict = {}
    
    # Extract additional reward-related fields
    reward_fields = ["reward_model", "data_source", "reward_stats"]
    for field in reward_fields:
        if field in batch.non_tensor_batch:
            reward_extra_infos_dict[field] = batch.non_tensor_batch[field]
    
    return reward_tensor, reward_extra_infos_dict


def validate_reward_function(reward_fn: Callable, sample_data: DataProto) -> bool:
    """
    Validate a reward function with sample data.
    
    Args:
        reward_fn: Reward function to validate
        sample_data: Sample data for testing
        
    Returns:
        True if validation passes
    """
    try:
        # Test the reward function
        result = reward_fn(sample_data, return_dict=True)
        
        # Check required fields
        if "reward_tensor" not in result:
            raise ValueError("Reward function must return 'reward_tensor' in result dict")
        
        reward_tensor = result["reward_tensor"]
        
        # Check tensor properties
        if not isinstance(reward_tensor, torch.Tensor):
            raise ValueError("reward_tensor must be a torch.Tensor")
        
        if reward_tensor.dim() < 2:
            raise ValueError("reward_tensor must be at least 2D (batch_size, sequence_length)")
        
        # Check shape compatibility
        batch_size = sample_data.batch.batch_size[0]
        if reward_tensor.shape[0] != batch_size:
            raise ValueError(f"reward_tensor batch size mismatch: expected {batch_size}, got {reward_tensor.shape[0]}")
        
        return True
        
    except Exception as e:
        print(f"Reward function validation failed: {e}")
        return False