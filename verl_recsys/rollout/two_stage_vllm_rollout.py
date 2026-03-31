"""Two-stage vLLM rollout registration for GR2-style generation."""

from __future__ import annotations

import os

from verl.utils.import_utils import load_extern_type


def _load_onerec_rollout_cls():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    extern_path = os.path.join(repo_root, "verl_rl", "recipe", "onerec", "onerec_vllm_rollout.py")
    return load_extern_type(extern_path, "OneRecvLLMRollout")


class TwoStagevLLMRollout(_load_onerec_rollout_cls()):
    """Re-export OneRec two-stage rollout under `verl_recsys` namespace."""


def register_two_stage_rollout() -> None:
    """Register `two_stage` rollout name into VeRL rollout registry."""
    from verl.workers.rollout.base import _ROLLOUT_REGISTRY

    _ROLLOUT_REGISTRY[("two_stage", "sync")] = "verl_recsys.rollout.two_stage_vllm_rollout.TwoStagevLLMRollout"
