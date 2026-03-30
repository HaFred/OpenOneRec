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
Example reward functions for generative recommendation system RL training.

These functions follow verl's ``compute_score`` signature:
    compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float | dict

Each reward function evaluates one aspect of a generative recommendation model's
output (e.g., a ranked list of items, a textual recommendation explanation, or a
structured recommendation response).

Design assumptions for generative recsys:
  - ``solution_str``: The model's generated recommendation output (could be a
    comma-separated item list, JSON, or natural language explanation).
  - ``ground_truth``: The ground-truth items or labels the user actually interacted
    with (could be a comma-separated list, JSON, or structured dict).
  - ``extra_info``: Optional dict containing auxiliary data such as:
      - "item_catalog": set of all available items
      - "user_history": list of items the user has previously interacted with
      - "item_categories": dict mapping item -> category
      - "popularity_scores": dict mapping item -> popularity float
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _parse_item_list(text: str) -> list[str]:
    """Parse a generated recommendation into a list of item identifiers.

    Handles multiple formats:
      - Comma-separated: "item1, item2, item3"
      - Numbered list: "1. item1\n2. item2"
      - JSON array: '["item1", "item2"]'
      - Newline-separated
    """
    text = text.strip()

    # Try JSON array first
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass

    # Try numbered list
    numbered = re.findall(r"\d+\.\s*(.+)", text)
    if numbered:
        return [item.strip().rstrip(",") for item in numbered if item.strip()]

    # Try comma-separated
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]

    # Fallback: newline-separated
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines


def _parse_ground_truth(ground_truth: Any) -> list[str]:
    """Parse ground truth into a list of item identifiers."""
    if isinstance(ground_truth, list):
        return [str(x).strip() for x in ground_truth]
    if isinstance(ground_truth, str):
        return _parse_item_list(ground_truth)
    if isinstance(ground_truth, dict):
        # Could be {"items": [...], ...}
        if "items" in ground_truth:
            return [str(x).strip() for x in ground_truth["items"]]
    return []


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def engagement_reward(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict[str, Any]:
    """Engagement/accuracy reward: measures how many recommended items match ground truth.

    Computes a hit-rate style score: |recommended ∩ ground_truth| / |ground_truth|.
    This is the primary "relevance" signal for recommendation RL training.

    Returns:
        dict with "score" (float in [0,1]) and "hits" (int).
    """
    recommended = _parse_item_list(solution_str)
    gt_items = _parse_ground_truth(ground_truth)

    if not gt_items:
        return {"score": 0.0, "hits": 0, "total_gt": 0, "total_rec": len(recommended)}

    rec_set = set(recommended)
    gt_set = set(gt_items)
    hits = len(rec_set & gt_set)

    score = hits / len(gt_set)
    return {
        "score": score,
        "hits": hits,
        "total_gt": len(gt_set),
        "total_rec": len(rec_set),
    }


def diversity_reward(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict[str, Any]:
    """Diversity reward: measures intra-list diversity of recommendations.

    If ``extra_info["item_categories"]`` is provided (dict: item -> category),
    computes the ratio of unique categories to total recommended items.
    Otherwise, falls back to unique-item ratio (penalizes duplicates).

    Returns:
        dict with "score" (float in [0,1]) and "unique_categories" or "unique_items".
    """
    recommended = _parse_item_list(solution_str)

    if not recommended:
        return {"score": 0.0, "unique_count": 0, "total": 0}

    extra = extra_info or {}
    item_categories = extra.get("item_categories", {})

    if item_categories:
        categories = [item_categories.get(item, f"_unknown_{item}") for item in recommended]
        unique_cats = set(categories)
        score = len(unique_cats) / len(recommended)
        return {
            "score": score,
            "unique_categories": len(unique_cats),
            "total": len(recommended),
        }
    else:
        # Penalize duplicates
        unique_items = set(recommended)
        score = len(unique_items) / len(recommended)
        return {
            "score": score,
            "unique_items": len(unique_items),
            "total": len(recommended),
        }


def novelty_reward(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict[str, Any]:
    """Novelty reward: penalizes recommending items the user has already seen.

    If ``extra_info["user_history"]`` is provided, computes the fraction of
    recommended items NOT in the user's history.

    Optionally, if ``extra_info["popularity_scores"]`` is provided (item -> float
    in [0,1] where 1 = most popular), novelty is also measured as the average
    inverse popularity of recommended items (less popular = more novel).

    Returns:
        dict with "score" (float in [0,1]) and auxiliary info.
    """
    recommended = _parse_item_list(solution_str)
    extra = extra_info or {}

    if not recommended:
        return {"score": 0.0, "novel_count": 0, "total": 0}

    user_history = set(extra.get("user_history", []))
    popularity_scores = extra.get("popularity_scores", {})

    # History-based novelty
    if user_history:
        novel_items = [item for item in recommended if item not in user_history]
        history_novelty = len(novel_items) / len(recommended)
    else:
        history_novelty = 1.0  # No history info, assume all novel

    # Popularity-based novelty (inverse popularity)
    if popularity_scores:
        inv_popularities = [
            1.0 - popularity_scores.get(item, 0.5) for item in recommended
        ]
        pop_novelty = sum(inv_popularities) / len(inv_popularities)
    else:
        pop_novelty = None

    # Combine if both available
    if pop_novelty is not None:
        score = 0.5 * history_novelty + 0.5 * pop_novelty
    else:
        score = history_novelty

    result = {
        "score": score,
        "history_novelty": history_novelty,
        "novel_count": int(history_novelty * len(recommended)),
        "total": len(recommended),
    }
    if pop_novelty is not None:
        result["popularity_novelty"] = pop_novelty

    return result


def coherence_reward(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict[str, Any]:
    """Coherence reward: evaluates structural and semantic quality of the response.

    Heuristic-based checks:
      1. Non-empty response
      2. Contains actual item recommendations (not just filler text)
      3. Reasonable length (not too short, not excessively long)
      4. Low repetition (penalize repeated phrases/items)

    Returns:
        dict with "score" (float in [0,1]) and diagnostic fields.
    """
    if not solution_str or not solution_str.strip():
        return {"score": 0.0, "reason": "empty_response"}

    recommended = _parse_item_list(solution_str)

    penalties = 0.0
    diagnostics: dict[str, Any] = {}

    # Check 1: Has items
    if not recommended:
        penalties += 0.5
        diagnostics["has_items"] = False
    else:
        diagnostics["has_items"] = True

    # Check 2: Reasonable length (at least 3 items, at most 100 for recs)
    num_items = len(recommended)
    diagnostics["num_items"] = num_items
    if num_items < 3:
        penalties += 0.2
    elif num_items > 100:
        penalties += 0.1

    # Check 3: Repetition (fraction of duplicates)
    if recommended:
        unique_ratio = len(set(recommended)) / len(recommended)
        repetition_penalty = max(0, 1.0 - unique_ratio)
        penalties += 0.3 * repetition_penalty
        diagnostics["unique_ratio"] = unique_ratio

    score = max(0.0, 1.0 - penalties)
    diagnostics["score"] = score
    return diagnostics


def format_compliance_reward(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict[str, Any]:
    """Format compliance reward: checks if the response follows expected output format.

    Supports checking for:
      - JSON parsability (if expected_format="json" in extra_info)
      - Contains required fields
      - Proper list structure

    Returns:
        dict with "score" (float in [0,1]) and diagnostics.
    """
    extra = extra_info or {}
    expected_format = extra.get("expected_format", "list")
    required_fields = extra.get("required_fields", [])

    diagnostics: dict[str, Any] = {}
    score = 1.0

    text = solution_str.strip()

    if expected_format == "json":
        try:
            parsed = json.loads(text)
            diagnostics["json_valid"] = True

            # Check required fields
            if isinstance(parsed, dict) and required_fields:
                missing = [f for f in required_fields if f not in parsed]
                if missing:
                    score -= 0.3 * (len(missing) / len(required_fields))
                    diagnostics["missing_fields"] = missing
        except json.JSONDecodeError:
            score = 0.0
            diagnostics["json_valid"] = False

    elif expected_format == "list":
        items = _parse_item_list(text)
        diagnostics["parsed_items"] = len(items)
        if not items:
            score = 0.0

    elif expected_format == "numbered_list":
        numbered = re.findall(r"\d+\.\s*(.+)", text)
        diagnostics["numbered_items"] = len(numbered)
        if not numbered:
            score *= 0.3  # Partial credit if some text exists

    # Penalize if response contains common error patterns
    error_patterns = ["I cannot", "I'm sorry", "error", "undefined"]
    for pattern in error_patterns:
        if pattern.lower() in text.lower():
            score *= 0.5
            diagnostics["error_pattern_found"] = pattern
            break

    diagnostics["score"] = score
    return diagnostics
