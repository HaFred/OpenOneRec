"""Paper-approx conditional verifiable rewards for GR2-style reranking."""

from __future__ import annotations

import re
from typing import Any

SLOT_PATTERN = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")


def _extract_sid_tuples(text: Any) -> list[tuple[str, str, str]]:
    if not isinstance(text, str):
        return []
    return [tuple(match) for match in SLOT_PATTERN.findall(text)]


def _pass_at_1(prediction: str, ground_truth: str) -> float:
    pred = _extract_sid_tuples(prediction)
    gt = set(_extract_sid_tuples(ground_truth))
    if not pred or not gt:
        return 0.0
    return float(pred[0] in gt)


def _hit_ratio(prediction: str, ground_truth: str) -> float:
    pred = _extract_sid_tuples(prediction)
    gt = set(_extract_sid_tuples(ground_truth))
    if not pred or not gt:
        return 0.0
    pred_set = set(pred)
    return len(pred_set & gt) / max(len(pred_set), 1)


def _coerce_sid_list(items: Any) -> list[Any]:
    if isinstance(items, list):
        return items
    if isinstance(items, tuple):
        return list(items)
    return []


def _order_position_similarity(generated: list[Any], original: list[Any]) -> float:
    if not generated or not original:
        return 0.0
    topk = min(len(generated), len(original))
    if topk <= 0:
        return 0.0
    same_pos = sum(1 for i in range(topk) if generated[i] == original[i])
    return same_pos / float(topk)


def compute_gr2_conditional_score(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    order_similarity_threshold: float = 0.9,
    order_penalty_weight: float = 0.5,
    min_base_reward_for_penalty: float = 0.0,
    generated_order_key: str = "generated_items",
    original_order_key: str = "original_items",
) -> dict[str, float]:
    """Compute a paper-approx conditional verifiable reward for GR2/DAPO.

    Reward = base_verifiable_reward + conditional_order_penalty
    where base_verifiable_reward blends pass@1 and hit ratio, and penalty activates
    only if generated order is too similar to original candidate order.
    """
    pass_at_1 = _pass_at_1(solution_str, ground_truth)
    hit_reward = _hit_ratio(solution_str, ground_truth)
    base_reward = 0.7 * pass_at_1 + 0.3 * hit_reward

    generated_order = _coerce_sid_list(extra_info.get(generated_order_key))
    original_order = _coerce_sid_list(extra_info.get(original_order_key))
    if not original_order:
        # Compatibility fallback: some pipelines use candidate_order.
        original_order = _coerce_sid_list(extra_info.get("candidate_order"))

    similarity = _order_position_similarity(generated_order, original_order)
    penalty = 0.0
    if base_reward >= min_base_reward_for_penalty and similarity >= order_similarity_threshold:
        denom = max(1e-6, 1.0 - order_similarity_threshold)
        penalty = -order_penalty_weight * ((similarity - order_similarity_threshold) / denom)

    final_reward = base_reward + penalty
    return {
        "score": final_reward,
        "base_reward": base_reward,
        "pass_at_1": pass_at_1,
        "hit_reward": hit_reward,
        "order_similarity": similarity,
        "order_penalty": penalty,
    }
