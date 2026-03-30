# Copyright 2025 OpenOneRec Contributors
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
MultiRewardManager: A reward manager that composes multiple reward functions
with configurable aggregation strategies for RL training of omni/generative
recommendation models.

This module is designed to be registered into verl's reward manager registry
and used as a drop-in replacement for single-reward managers in PPO/GRPO
training pipelines.

Key features:
  - Compose N heterogeneous reward functions (rule-based, model-based, API-based)
  - Configurable per-reward weights and aggregation (weighted_sum, product, min, max)
  - Per-reward-dimension logging for W&B / TensorBoard tracking
  - Optional per-reward normalization (running mean/std)
  - Compatible with verl's AbstractRewardManager interface and registry
  - Designed for generative recsys signals: engagement, diversity, novelty,
    coherence, format compliance, etc.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    from verl import DataProto
    from verl.workers.reward_manager.abstract import AbstractRewardManager
    HAS_VERL = True
except ImportError:
    HAS_VERL = False
    # Provide a no-op base class for standalone usage/testing
    class AbstractRewardManager:  # type: ignore[no-redef]
        pass
    DataProto = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

class AggregationStrategy(str, Enum):
    """Supported strategies for combining multiple reward signals."""
    WEIGHTED_SUM = "weighted_sum"
    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    WEIGHTED_SUM_CLIPPED = "weighted_sum_clipped"


@dataclass
class RewardComponentConfig:
    """Configuration for a single reward component.

    Attributes:
        name: Unique identifier for this reward component (e.g., "engagement", "diversity").
        weight: Scalar weight applied to this component during aggregation.
        compute_score_fn: The scoring function. Signature must be compatible with verl's
            ``compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)``.
        normalize: Whether to apply running normalization to this component's scores.
        clip_min: Optional minimum value to clip the component score.
        clip_max: Optional maximum value to clip the component score.
    """
    name: str
    weight: float = 1.0
    compute_score_fn: Optional[Callable] = None
    normalize: bool = False
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None


@dataclass
class MultiRewardConfig:
    """Top-level configuration for MultiRewardManager.

    Attributes:
        components: List of RewardComponentConfig defining each reward signal.
        aggregation: Strategy used to combine component scores into a scalar reward.
        normalize_final: Whether to normalize the aggregated reward.
        log_components: Whether to include per-component scores in reward_extra_info.
        clip_final_min: Optional clip for the final aggregated reward.
        clip_final_max: Optional clip for the final aggregated reward.
    """
    components: list[RewardComponentConfig] = field(default_factory=list)
    aggregation: AggregationStrategy = AggregationStrategy.WEIGHTED_SUM
    normalize_final: bool = False
    log_components: bool = True
    clip_final_min: Optional[float] = None
    clip_final_max: Optional[float] = None


# ---------------------------------------------------------------------------
# Running normalizer for online reward normalization
# ---------------------------------------------------------------------------

class RunningNormalizer:
    """Welford's online algorithm for running mean/std normalization."""

    def __init__(self, eps: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.eps = eps

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        # Welford's M2 update
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x: float) -> float:
        std = max(self.var ** 0.5, self.eps)
        return (x - self.mean) / std


# ---------------------------------------------------------------------------
# MultiRewardManager
# ---------------------------------------------------------------------------

class MultiRewardManager(AbstractRewardManager):
    """Reward manager that composes multiple reward functions.

    This manager evaluates each reward component independently, then aggregates
    them into a single scalar reward per sample. All per-component scores are
    tracked in ``reward_extra_info`` for logging.

    Compatible with verl's reward manager registry via ``@register("multi")``.

    Example usage in verl config:
        reward_model:
          reward_manager: multi
          reward_kwargs:
            aggregation: weighted_sum
            components:
              - name: engagement
                weight: 0.5
              - name: diversity
                weight: 0.3
              - name: format
                weight: 0.2
    """

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        multi_reward_config: Optional[MultiRewardConfig] = None,
        component_score_fns: Optional[dict[str, Callable]] = None,
        aggregation: str = "weighted_sum",
        components: Optional[list[dict]] = None,
        normalize_final: bool = False,
        log_components: bool = True,
        clip_final_min: Optional[float] = None,
        clip_final_max: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the MultiRewardManager.

        Can be initialized either with a ``MultiRewardConfig`` object or with
        individual keyword arguments (for compatibility with verl's config-based
        instantiation via ``reward_kwargs``).

        Args:
            tokenizer: Tokenizer for decoding token IDs.
            num_examine: Number of debug samples to print.
            compute_score: Fallback scoring function (used as default for components
                that don't specify their own ``compute_score_fn``).
            reward_fn_key: Key for data source lookup in non_tensor_batch.
            multi_reward_config: Full configuration object. If provided, other
                component-related kwargs are ignored.
            component_score_fns: Dict mapping component name -> scoring function.
                Used when constructing from kwargs instead of MultiRewardConfig.
            aggregation: Aggregation strategy name.
            components: List of component dicts with keys: name, weight, normalize,
                clip_min, clip_max.
            normalize_final: Whether to normalize the final aggregated score.
            log_components: Whether to log per-component scores.
            clip_final_min: Min clip for final aggregated reward.
            clip_final_max: Max clip for final aggregated reward.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.default_compute_score = compute_score
        self.reward_fn_key = reward_fn_key

        # Build config from either object or kwargs
        if multi_reward_config is not None:
            self.config = multi_reward_config
        else:
            self.config = self._build_config_from_kwargs(
                aggregation=aggregation,
                components=components or [],
                component_score_fns=component_score_fns or {},
                normalize_final=normalize_final,
                log_components=log_components,
                clip_final_min=clip_final_min,
                clip_final_max=clip_final_max,
            )

        # Assign scoring functions from component_score_fns dict
        if component_score_fns:
            for comp in self.config.components:
                if comp.compute_score_fn is None and comp.name in component_score_fns:
                    comp.compute_score_fn = component_score_fns[comp.name]

        # Validate: every component must have a scoring function
        for comp in self.config.components:
            if comp.compute_score_fn is None:
                if self.default_compute_score is not None:
                    comp.compute_score_fn = self.default_compute_score
                    logger.info(
                        f"Component '{comp.name}' has no score fn; using default compute_score."
                    )
                else:
                    raise ValueError(
                        f"Component '{comp.name}' has no compute_score_fn and no default "
                        f"compute_score was provided."
                    )

        # Running normalizers per component
        self._normalizers: dict[str, RunningNormalizer] = {}
        for comp in self.config.components:
            if comp.normalize:
                self._normalizers[comp.name] = RunningNormalizer()

        if self.config.normalize_final:
            self._final_normalizer = RunningNormalizer()
        else:
            self._final_normalizer = None

        logger.info(
            f"MultiRewardManager initialized with {len(self.config.components)} components: "
            f"{[c.name for c in self.config.components]}, "
            f"aggregation={self.config.aggregation.value}"
        )

    @staticmethod
    def _build_config_from_kwargs(
        aggregation: str,
        components: list[dict],
        component_score_fns: dict[str, Callable],
        normalize_final: bool,
        log_components: bool,
        clip_final_min: Optional[float],
        clip_final_max: Optional[float],
    ) -> MultiRewardConfig:
        """Build MultiRewardConfig from flat kwargs."""
        comp_configs = []
        for comp_dict in components:
            name = comp_dict["name"]
            comp_configs.append(
                RewardComponentConfig(
                    name=name,
                    weight=comp_dict.get("weight", 1.0),
                    compute_score_fn=component_score_fns.get(name),
                    normalize=comp_dict.get("normalize", False),
                    clip_min=comp_dict.get("clip_min"),
                    clip_max=comp_dict.get("clip_max"),
                )
            )
        return MultiRewardConfig(
            components=comp_configs,
            aggregation=AggregationStrategy(aggregation),
            normalize_final=normalize_final,
            log_components=log_components,
            clip_final_min=clip_final_min,
            clip_final_max=clip_final_max,
        )

    # ------------------------------------------------------------------
    # Core aggregation logic
    # ------------------------------------------------------------------

    def _aggregate_scores(self, component_scores: dict[str, float]) -> float:
        """Aggregate per-component scores into a single scalar.

        Args:
            component_scores: Dict mapping component name -> score.

        Returns:
            Aggregated scalar reward.
        """
        strategy = self.config.aggregation
        components = self.config.components

        if strategy == AggregationStrategy.WEIGHTED_SUM:
            total = sum(
                comp.weight * component_scores[comp.name] for comp in components
            )
            return total

        elif strategy == AggregationStrategy.WEIGHTED_SUM_CLIPPED:
            total = sum(
                comp.weight * component_scores[comp.name] for comp in components
            )
            if self.config.clip_final_min is not None:
                total = max(total, self.config.clip_final_min)
            if self.config.clip_final_max is not None:
                total = min(total, self.config.clip_final_max)
            return total

        elif strategy == AggregationStrategy.PRODUCT:
            result = 1.0
            for comp in components:
                result *= component_scores[comp.name] ** comp.weight
            return result

        elif strategy == AggregationStrategy.MIN:
            return min(
                comp.weight * component_scores[comp.name] for comp in components
            )

        elif strategy == AggregationStrategy.MAX:
            return max(
                comp.weight * component_scores[comp.name] for comp in components
            )

        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    def _compute_component_score(
        self,
        comp: RewardComponentConfig,
        data_source: str,
        solution_str: str,
        ground_truth: Any,
        extra_info: dict,
    ) -> float:
        """Compute score for a single reward component, with optional normalization and clipping."""
        try:
            raw_score = comp.compute_score_fn(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        except Exception as e:
            logger.warning(
                f"Component '{comp.name}' scoring failed: {e}. Returning 0.0."
            )
            raw_score = 0.0

        # Handle dict returns (some score fns return {"score": ..., ...})
        if isinstance(raw_score, dict):
            score = float(raw_score.get("score", 0.0))
        elif isinstance(raw_score, (int, float, bool)):
            score = float(raw_score)
        else:
            score = float(raw_score[0])

        # Per-component clipping
        if comp.clip_min is not None:
            score = max(score, comp.clip_min)
        if comp.clip_max is not None:
            score = min(score, comp.clip_max)

        # Per-component normalization
        if comp.normalize and comp.name in self._normalizers:
            self._normalizers[comp.name].update(score)
            score = self._normalizers[comp.name].normalize(score)

        return score

    # ------------------------------------------------------------------
    # verl AbstractRewardManager interface
    # ------------------------------------------------------------------

    def _extract_reward_from_rm_scores(
        self, data: DataProto, return_dict: bool = False
    ):
        """Extract reward from already-computed rm_scores if available."""
        if "rm_scores" not in data.batch.keys():
            return None
        if return_dict:
            reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
            reward_extra_info = {
                key: data.non_tensor_batch[key] for key in reward_extra_keys
            }
            return {
                "reward_tensor": data.batch["rm_scores"],
                "reward_extra_info": reward_extra_info,
            }
        else:
            return data.batch["rm_scores"]

    def __call__(
        self, data: DataProto, return_dict: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        """Compute multi-component rewards for the batch.

        For each sample, evaluates all reward components, aggregates them,
        and places the final reward at the last valid response token position.

        Args:
            data: DataProto batch from verl pipeline.
            return_dict: If True, return dict with reward_tensor and reward_extra_info.

        Returns:
            reward_tensor or dict with reward_tensor and per-component extra info.
        """
        # Short-circuit if rm_scores are pre-computed
        reward_from_rm = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm is not None:
            return reward_from_rm

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)

        already_print_data_sources: dict[str, int] = {}

        for i in range(len(data)):
            data_item = data[i]

            # Extract prompt and response
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            # Compute each component score
            component_scores: dict[str, float] = {}
            for comp in self.config.components:
                score = self._compute_component_score(
                    comp=comp,
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                component_scores[comp.name] = score

            # Aggregate
            aggregated_reward = self._aggregate_scores(component_scores)

            # Final normalization
            if self._final_normalizer is not None:
                self._final_normalizer.update(aggregated_reward)
                aggregated_reward = self._final_normalizer.normalize(aggregated_reward)

            # Final clipping
            if self.config.clip_final_min is not None:
                aggregated_reward = max(aggregated_reward, self.config.clip_final_min)
            if self.config.clip_final_max is not None:
                aggregated_reward = min(aggregated_reward, self.config.clip_final_max)

            reward_tensor[i, valid_response_length - 1] = aggregated_reward

            # Log per-component info
            if self.config.log_components:
                for comp_name, comp_score in component_scores.items():
                    reward_extra_info[f"reward_{comp_name}"].append(comp_score)
                reward_extra_info["score"].append(aggregated_reward)

            # Debug printing
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                for comp_name, comp_score in component_scores.items():
                    print(f"[reward_{comp_name}]", comp_score)
                print("[aggregated_score]", aggregated_reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        else:
            return reward_tensor
