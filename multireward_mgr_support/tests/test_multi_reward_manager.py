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
Unit tests for MultiRewardManager.

These tests use mock DataProto objects to simulate verl's data pipeline
without requiring the full verl installation for unit testing.
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from multireward_mgr_support.reward_manager.multi_reward_manager import (
    AggregationStrategy,
    MultiRewardConfig,
    MultiRewardManager,
    RewardComponentConfig,
    RunningNormalizer,
    HAS_TORCH,
)


# ---------------------------------------------------------------------------
# Mock tokenizer for testing
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Minimal tokenizer mock for testing."""

    def decode(self, token_ids, skip_special_tokens=True):
        # Return a fake comma-separated item list
        return "item1, item2, item3"

    def batch_decode(self, token_ids_batch, skip_special_tokens=True):
        return [self.decode(ids) for ids in token_ids_batch]


# ---------------------------------------------------------------------------
# Mock DataProto for testing without full verl dependency
# ---------------------------------------------------------------------------

class MockDataItem:
    """Mock for a single DataProto item."""

    def __init__(self, batch, non_tensor_batch):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch


# ---------------------------------------------------------------------------
# Test RunningNormalizer
# ---------------------------------------------------------------------------

class TestRunningNormalizer:
    def test_single_value(self):
        norm = RunningNormalizer()
        norm.update(5.0)
        # After one sample, mean=5.0, var approaches 0
        result = norm.normalize(5.0)
        assert abs(result) < 1e-4  # Should be ~0 (centered)

    def test_multiple_values(self):
        norm = RunningNormalizer()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            norm.update(v)
        # mean should be 3.0
        assert abs(norm.mean - 3.0) < 1e-6
        # Normalizing 3.0 should give ~0
        assert abs(norm.normalize(3.0)) < 1e-4

    def test_zero_variance_protection(self):
        norm = RunningNormalizer(eps=1e-8)
        norm.update(5.0)
        # Should not crash even with near-zero variance
        result = norm.normalize(10.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Test MultiRewardManager construction
# ---------------------------------------------------------------------------

def _make_score_fn(fixed_score: float):
    """Create a simple scoring function that returns a fixed score."""
    def fn(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
        return fixed_score
    return fn


def _make_dict_score_fn(fixed_score: float):
    """Create a scoring function that returns a dict."""
    def fn(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
        return {"score": fixed_score, "detail": "test"}
    return fn


class TestMultiRewardManagerConstruction:
    def test_from_config_object(self):
        config = MultiRewardConfig(
            components=[
                RewardComponentConfig(name="r1", weight=0.5, compute_score_fn=_make_score_fn(1.0)),
                RewardComponentConfig(name="r2", weight=0.5, compute_score_fn=_make_score_fn(0.5)),
            ],
            aggregation=AggregationStrategy.WEIGHTED_SUM,
        )
        mgr = MultiRewardManager(
            tokenizer=MockTokenizer(),
            num_examine=0,
            multi_reward_config=config,
        )
        assert len(mgr.config.components) == 2

    def test_from_kwargs(self):
        mgr = MultiRewardManager(
            tokenizer=MockTokenizer(),
            num_examine=0,
            aggregation="weighted_sum",
            components=[
                {"name": "r1", "weight": 0.6},
                {"name": "r2", "weight": 0.4},
            ],
            component_score_fns={
                "r1": _make_score_fn(1.0),
                "r2": _make_score_fn(0.5),
            },
        )
        assert len(mgr.config.components) == 2
        assert mgr.config.components[0].weight == 0.6

    def test_missing_score_fn_raises(self):
        with pytest.raises(ValueError, match="no compute_score_fn"):
            MultiRewardManager(
                tokenizer=MockTokenizer(),
                num_examine=0,
                components=[{"name": "r1", "weight": 1.0}],
            )

    def test_default_score_fn_fallback(self):
        mgr = MultiRewardManager(
            tokenizer=MockTokenizer(),
            num_examine=0,
            compute_score=_make_score_fn(0.7),
            components=[{"name": "r1", "weight": 1.0}],
        )
        assert mgr.config.components[0].compute_score_fn is not None


# ---------------------------------------------------------------------------
# Test aggregation strategies
# ---------------------------------------------------------------------------

class TestAggregation:
    def _make_mgr(self, strategy: str, weights=None):
        if weights is None:
            weights = [0.5, 0.5]
        config = MultiRewardConfig(
            components=[
                RewardComponentConfig(
                    name="r1", weight=weights[0], compute_score_fn=_make_score_fn(1.0)
                ),
                RewardComponentConfig(
                    name="r2", weight=weights[1], compute_score_fn=_make_score_fn(0.5)
                ),
            ],
            aggregation=AggregationStrategy(strategy),
        )
        return MultiRewardManager(
            tokenizer=MockTokenizer(),
            num_examine=0,
            multi_reward_config=config,
        )

    def test_weighted_sum(self):
        mgr = self._make_mgr("weighted_sum", [0.6, 0.4])
        scores = {"r1": 1.0, "r2": 0.5}
        result = mgr._aggregate_scores(scores)
        expected = 0.6 * 1.0 + 0.4 * 0.5
        assert abs(result - expected) < 1e-6

    def test_product(self):
        mgr = self._make_mgr("product", [1.0, 1.0])
        scores = {"r1": 0.8, "r2": 0.5}
        result = mgr._aggregate_scores(scores)
        expected = 0.8 ** 1.0 * 0.5 ** 1.0
        assert abs(result - expected) < 1e-6

    def test_min(self):
        mgr = self._make_mgr("min", [1.0, 1.0])
        scores = {"r1": 0.8, "r2": 0.3}
        result = mgr._aggregate_scores(scores)
        assert abs(result - 0.3) < 1e-6

    def test_max(self):
        mgr = self._make_mgr("max", [1.0, 1.0])
        scores = {"r1": 0.8, "r2": 0.3}
        result = mgr._aggregate_scores(scores)
        assert abs(result - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# Test component score computation
# ---------------------------------------------------------------------------

class TestComponentScoreComputation:
    def test_dict_return(self):
        comp = RewardComponentConfig(
            name="test", compute_score_fn=_make_dict_score_fn(0.7)
        )
        mgr = MultiRewardManager(
            tokenizer=MockTokenizer(),
            num_examine=0,
            multi_reward_config=MultiRewardConfig(components=[comp]),
        )
        score = mgr._compute_component_score(
            comp, "test", "response", "gt", {}
        )
        assert abs(score - 0.7) < 1e-6

    def test_float_return(self):
        comp = RewardComponentConfig(
            name="test", compute_score_fn=_make_score_fn(0.3)
        )
        mgr = MultiRewardManager(
            tokenizer=MockTokenizer(),
            num_examine=0,
            multi_reward_config=MultiRewardConfig(components=[comp]),
        )
        score = mgr._compute_component_score(
            comp, "test", "response", "gt", {}
        )
        assert abs(score - 0.3) < 1e-6

    def test_clipping(self):
        comp = RewardComponentConfig(
            name="test",
            compute_score_fn=_make_score_fn(1.5),
            clip_min=0.0,
            clip_max=1.0,
        )
        mgr = MultiRewardManager(
            tokenizer=MockTokenizer(),
            num_examine=0,
            multi_reward_config=MultiRewardConfig(components=[comp]),
        )
        score = mgr._compute_component_score(
            comp, "test", "response", "gt", {}
        )
        assert score == 1.0

    def test_exception_handling(self):
        def failing_fn(**kwargs):
            raise RuntimeError("test error")

        comp = RewardComponentConfig(
            name="test", compute_score_fn=failing_fn
        )
        mgr = MultiRewardManager(
            tokenizer=MockTokenizer(),
            num_examine=0,
            multi_reward_config=MultiRewardConfig(components=[comp]),
        )
        score = mgr._compute_component_score(
            comp, "test", "response", "gt", {}
        )
        assert score == 0.0


# ---------------------------------------------------------------------------
# Integration test with mock DataProto
# ---------------------------------------------------------------------------

class TestMultiRewardManagerIntegration:
    """Integration tests that mock the verl DataProto structure.

    NOTE: These tests patch MultiRewardManager.__call__ to use mock data.
    Full integration tests require the verl package installed.
    """

    def test_aggregated_scores_are_computed(self):
        """Test that the aggregation logic works end-to-end on component scores."""
        config = MultiRewardConfig(
            components=[
                RewardComponentConfig(
                    name="engagement", weight=0.5, compute_score_fn=_make_score_fn(0.8)
                ),
                RewardComponentConfig(
                    name="diversity", weight=0.3, compute_score_fn=_make_score_fn(0.6)
                ),
                RewardComponentConfig(
                    name="novelty", weight=0.2, compute_score_fn=_make_score_fn(1.0)
                ),
            ],
            aggregation=AggregationStrategy.WEIGHTED_SUM,
            log_components=True,
        )
        mgr = MultiRewardManager(
            tokenizer=MockTokenizer(),
            num_examine=0,
            multi_reward_config=config,
        )

        # Test aggregation directly
        scores = {"engagement": 0.8, "diversity": 0.6, "novelty": 1.0}
        result = mgr._aggregate_scores(scores)
        expected = 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 1.0
        assert abs(result - expected) < 1e-6

    def test_normalization_updates(self):
        """Test that running normalization updates correctly."""
        config = MultiRewardConfig(
            components=[
                RewardComponentConfig(
                    name="r1",
                    weight=1.0,
                    compute_score_fn=_make_score_fn(0.5),
                    normalize=True,
                ),
            ],
        )
        mgr = MultiRewardManager(
            tokenizer=MockTokenizer(),
            num_examine=0,
            multi_reward_config=config,
        )
        assert "r1" in mgr._normalizers
        # Simulate multiple score computations
        comp = config.components[0]
        for _ in range(10):
            mgr._compute_component_score(comp, "test", "resp", "gt", {})
        assert mgr._normalizers["r1"].count == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
