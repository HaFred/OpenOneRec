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
Example: Using MultiRewardManager with verl for generative recsys RL training.

This script demonstrates two usage patterns:
  1. Programmatic (Python API) — construct MultiRewardManager directly
  2. Config-based (verl integration) — register and use via verl's reward_manager config

Run standalone (no verl dependency needed for pattern 1):
    python usage_example.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from multireward_mgr_support.reward_manager.multi_reward_manager import (
    AggregationStrategy,
    MultiRewardConfig,
    MultiRewardManager,
    RewardComponentConfig,
)
from multireward_mgr_support.reward_functions.recsys_rewards import (
    coherence_reward,
    diversity_reward,
    engagement_reward,
    format_compliance_reward,
    novelty_reward,
)


# ============================================================================
# Pattern 1: Programmatic construction (standalone, no verl needed)
# ============================================================================

def demo_programmatic():
    """Demonstrate MultiRewardManager with explicit Python config."""

    print("=" * 70)
    print("Pattern 1: Programmatic MultiRewardManager")
    print("=" * 70)

    # --- Option A: Build from MultiRewardConfig object ---
    config = MultiRewardConfig(
        components=[
            RewardComponentConfig(
                name="engagement",
                weight=0.5,
                compute_score_fn=engagement_reward,
            ),
            RewardComponentConfig(
                name="diversity",
                weight=0.2,
                compute_score_fn=diversity_reward,
            ),
            RewardComponentConfig(
                name="novelty",
                weight=0.15,
                compute_score_fn=novelty_reward,
                normalize=False,
            ),
            RewardComponentConfig(
                name="coherence",
                weight=0.1,
                compute_score_fn=coherence_reward,
            ),
            RewardComponentConfig(
                name="format",
                weight=0.05,
                compute_score_fn=format_compliance_reward,
                clip_min=0.0,
                clip_max=1.0,
            ),
        ],
        aggregation=AggregationStrategy.WEIGHTED_SUM,
        log_components=True,
    )

    # Simple mock tokenizer for demo
    class SimpleTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return "item1, item2, item3"

    mgr = MultiRewardManager(
        tokenizer=SimpleTokenizer(),
        num_examine=0,
        multi_reward_config=config,
    )

    # Simulate scoring a single sample
    sample_scores = {}
    for comp in mgr.config.components:
        score = mgr._compute_component_score(
            comp=comp,
            data_source="recsys_v1",
            solution_str="item1, item2, item3, item4, item5",
            ground_truth="item1, item3, item5, item7",
            extra_info={
                "item_categories": {
                    "item1": "electronics",
                    "item2": "clothing",
                    "item3": "electronics",
                    "item4": "books",
                    "item5": "clothing",
                },
                "user_history": ["item1", "item10"],
                "expected_format": "list",
            },
        )
        sample_scores[comp.name] = score
        print(f"  {comp.name:20s}: {score:.4f} (weight={comp.weight})")

    final = mgr._aggregate_scores(sample_scores)
    print(f"\n  {'AGGREGATED':20s}: {final:.4f}")
    print()

    # --- Option B: Build from kwargs (simulating verl config loading) ---
    print("--- Option B: From kwargs ---")
    mgr2 = MultiRewardManager(
        tokenizer=SimpleTokenizer(),
        num_examine=0,
        aggregation="weighted_sum",
        components=[
            {"name": "engagement", "weight": 0.6},
            {"name": "diversity", "weight": 0.4},
        ],
        component_score_fns={
            "engagement": engagement_reward,
            "diversity": diversity_reward,
        },
    )
    print(f"  Components: {[c.name for c in mgr2.config.components]}")
    print(f"  Aggregation: {mgr2.config.aggregation.value}")
    print()


# ============================================================================
# Pattern 2: verl config integration (requires verl installed)
# ============================================================================

def demo_verl_integration():
    """Show how to register and use MultiRewardManager in verl."""

    print("=" * 70)
    print("Pattern 2: verl Config Integration")
    print("=" * 70)

    try:
        # Step 1: Register the multi reward manager
        import multireward_mgr_support.reward_manager.register_multi  # noqa: F401
        from verl.workers.reward_manager import get_reward_manager_cls

        cls = get_reward_manager_cls("multi")
        print(f"  Registered class: {cls.__name__}")
        print(f"  Available via: reward_model.reward_manager = 'multi'")
        print()

        # Step 2: Show how load_reward_manager would instantiate it
        print("  In your verl config YAML:")
        print("    reward_model:")
        print("      reward_manager: multi")
        print("      reward_kwargs:")
        print("        aggregation: weighted_sum")
        print("        components:")
        print("          - name: engagement")
        print("            weight: 0.5")
        print("          - name: diversity")
        print("            weight: 0.3")
        print()
        print("  Then provide component_score_fns via custom_reward_function")
        print("  or by extending the MultiRewardManager.")

    except ImportError:
        print("  [SKIP] verl not installed — showing config template only.")
        print()
        print("  To use in verl:")
        print("  1. pip install verl")
        print("  2. In your training script entrypoint, add:")
        print("       import multireward_mgr_support.reward_manager.register_multi")
        print("  3. Set reward_model.reward_manager: multi in your YAML config")
        print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    demo_programmatic()
    demo_verl_integration()
