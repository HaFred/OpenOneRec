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

"""Unit tests for recsys reward functions."""

import json
import pytest

import sys
import os

# Add the parent directory to sys.path so we can import the reward functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from multireward_mgr_support.reward_functions.recsys_rewards import (
    engagement_reward,
    diversity_reward,
    novelty_reward,
    coherence_reward,
    format_compliance_reward,
    _parse_item_list,
    _parse_ground_truth,
)


# ---------------------------------------------------------------------------
# Tests for _parse_item_list
# ---------------------------------------------------------------------------

class TestParseItemList:
    def test_comma_separated(self):
        assert _parse_item_list("item1, item2, item3") == ["item1", "item2", "item3"]

    def test_json_array(self):
        assert _parse_item_list('["a", "b", "c"]') == ["a", "b", "c"]

    def test_numbered_list(self):
        text = "1. Movie A\n2. Movie B\n3. Movie C"
        result = _parse_item_list(text)
        assert result == ["Movie A", "Movie B", "Movie C"]

    def test_newline_separated(self):
        text = "item_x\nitem_y\nitem_z"
        result = _parse_item_list(text)
        assert result == ["item_x", "item_y", "item_z"]

    def test_empty(self):
        assert _parse_item_list("") == []

    def test_single_item(self):
        assert _parse_item_list("only_one") == ["only_one"]


class TestParseGroundTruth:
    def test_list_input(self):
        assert _parse_ground_truth(["a", "b"]) == ["a", "b"]

    def test_string_input(self):
        assert _parse_ground_truth("a, b, c") == ["a", "b", "c"]

    def test_dict_with_items(self):
        assert _parse_ground_truth({"items": ["x", "y"]}) == ["x", "y"]

    def test_empty_dict(self):
        assert _parse_ground_truth({}) == []


# ---------------------------------------------------------------------------
# Tests for engagement_reward
# ---------------------------------------------------------------------------

class TestEngagementReward:
    def test_perfect_match(self):
        result = engagement_reward(
            data_source="test",
            solution_str="item1, item2, item3",
            ground_truth="item1, item2, item3",
        )
        assert result["score"] == 1.0
        assert result["hits"] == 3

    def test_partial_match(self):
        result = engagement_reward(
            data_source="test",
            solution_str="item1, item2, item4",
            ground_truth="item1, item2, item3",
        )
        assert abs(result["score"] - 2.0 / 3.0) < 1e-6
        assert result["hits"] == 2

    def test_no_match(self):
        result = engagement_reward(
            data_source="test",
            solution_str="item4, item5",
            ground_truth="item1, item2, item3",
        )
        assert result["score"] == 0.0
        assert result["hits"] == 0

    def test_empty_ground_truth(self):
        result = engagement_reward(
            data_source="test",
            solution_str="item1, item2",
            ground_truth="",
        )
        assert result["score"] == 0.0


# ---------------------------------------------------------------------------
# Tests for diversity_reward
# ---------------------------------------------------------------------------

class TestDiversityReward:
    def test_all_unique_no_categories(self):
        result = diversity_reward(
            data_source="test",
            solution_str="item1, item2, item3",
            ground_truth="",
        )
        assert result["score"] == 1.0

    def test_with_duplicates(self):
        result = diversity_reward(
            data_source="test",
            solution_str="item1, item1, item1",
            ground_truth="",
        )
        assert abs(result["score"] - 1.0 / 3.0) < 1e-6

    def test_with_categories(self):
        result = diversity_reward(
            data_source="test",
            solution_str="movie1, movie2, movie3",
            ground_truth="",
            extra_info={
                "item_categories": {
                    "movie1": "action",
                    "movie2": "action",
                    "movie3": "comedy",
                }
            },
        )
        # 2 unique categories out of 3 items
        assert abs(result["score"] - 2.0 / 3.0) < 1e-6

    def test_empty_list(self):
        result = diversity_reward(
            data_source="test",
            solution_str="",
            ground_truth="",
        )
        assert result["score"] == 0.0


# ---------------------------------------------------------------------------
# Tests for novelty_reward
# ---------------------------------------------------------------------------

class TestNoveltyReward:
    def test_all_novel(self):
        result = novelty_reward(
            data_source="test",
            solution_str="item4, item5, item6",
            ground_truth="",
            extra_info={"user_history": ["item1", "item2", "item3"]},
        )
        assert result["history_novelty"] == 1.0

    def test_all_seen(self):
        result = novelty_reward(
            data_source="test",
            solution_str="item1, item2, item3",
            ground_truth="",
            extra_info={"user_history": ["item1", "item2", "item3"]},
        )
        assert result["history_novelty"] == 0.0

    def test_mixed(self):
        result = novelty_reward(
            data_source="test",
            solution_str="item1, item4",
            ground_truth="",
            extra_info={"user_history": ["item1", "item2", "item3"]},
        )
        assert abs(result["history_novelty"] - 0.5) < 1e-6

    def test_with_popularity(self):
        result = novelty_reward(
            data_source="test",
            solution_str="item1, item2",
            ground_truth="",
            extra_info={
                "user_history": [],
                "popularity_scores": {"item1": 0.9, "item2": 0.1},
            },
        )
        # history_novelty = 1.0, pop_novelty = avg(0.1, 0.9) = 0.5
        # score = 0.5 * 1.0 + 0.5 * 0.5 = 0.75
        assert abs(result["score"] - 0.75) < 1e-6

    def test_no_history(self):
        result = novelty_reward(
            data_source="test",
            solution_str="item1",
            ground_truth="",
        )
        assert result["score"] == 1.0  # No history = all novel


# ---------------------------------------------------------------------------
# Tests for coherence_reward
# ---------------------------------------------------------------------------

class TestCoherenceReward:
    def test_good_response(self):
        result = coherence_reward(
            data_source="test",
            solution_str="item1, item2, item3, item4, item5",
            ground_truth="",
        )
        assert result["score"] > 0.8

    def test_empty_response(self):
        result = coherence_reward(
            data_source="test",
            solution_str="",
            ground_truth="",
        )
        assert result["score"] == 0.0

    def test_many_duplicates(self):
        result = coherence_reward(
            data_source="test",
            solution_str="item1, item1, item1, item1, item1",
            ground_truth="",
        )
        # Should have repetition penalty
        assert result["score"] < 1.0

    def test_too_few_items(self):
        result = coherence_reward(
            data_source="test",
            solution_str="item1",
            ground_truth="",
        )
        assert result["score"] < 1.0


# ---------------------------------------------------------------------------
# Tests for format_compliance_reward
# ---------------------------------------------------------------------------

class TestFormatComplianceReward:
    def test_valid_json(self):
        data = json.dumps({"items": ["a", "b"], "reason": "test"})
        result = format_compliance_reward(
            data_source="test",
            solution_str=data,
            ground_truth="",
            extra_info={"expected_format": "json", "required_fields": ["items", "reason"]},
        )
        assert result["score"] == 1.0

    def test_invalid_json(self):
        result = format_compliance_reward(
            data_source="test",
            solution_str="not json at all",
            ground_truth="",
            extra_info={"expected_format": "json"},
        )
        assert result["score"] == 0.0

    def test_json_missing_fields(self):
        data = json.dumps({"items": ["a"]})
        result = format_compliance_reward(
            data_source="test",
            solution_str=data,
            ground_truth="",
            extra_info={"expected_format": "json", "required_fields": ["items", "reason"]},
        )
        assert result["score"] < 1.0
        assert "reason" in result.get("missing_fields", [])

    def test_valid_list(self):
        result = format_compliance_reward(
            data_source="test",
            solution_str="item1, item2, item3",
            ground_truth="",
            extra_info={"expected_format": "list"},
        )
        assert result["score"] == 1.0

    def test_error_pattern_penalty(self):
        result = format_compliance_reward(
            data_source="test",
            solution_str="I'm sorry, I cannot provide recommendations",
            ground_truth="",
            extra_info={"expected_format": "list"},
        )
        assert result["score"] < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
