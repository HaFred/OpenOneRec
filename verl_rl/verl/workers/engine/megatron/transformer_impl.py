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
"""Megatron transformer helpers adapted from upstream verl transformer engine.

This module centralizes model-parallel bootstrap logic used by actor/critic/ref/reward
workers so 5D topology checks and initialization behavior are consistent.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from megatron.core import parallel_state as mpu

logger = logging.getLogger(__file__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def get_parallelism_tuple(megatron_cfg: Any) -> tuple[int, int, int, int]:
    """Return (tp, pp, cp, ep) from megatron config."""
    tp = int(_cfg_get(megatron_cfg, "tensor_model_parallel_size", 1))
    pp = int(_cfg_get(megatron_cfg, "pipeline_model_parallel_size", 1))
    cp = int(_cfg_get(megatron_cfg, "context_parallel_size", 1))
    ep = int(_cfg_get(megatron_cfg, "expert_model_parallel_size", 1))
    return tp, pp, cp, ep


def validate_parallelism(megatron_cfg: Any, world_size: int, role_name: str = "worker") -> tuple[int, int]:
    """Validate TP/PP/CP/EP divisibility and derive DP."""
    tp, pp, cp, ep = get_parallelism_tuple(megatron_cfg)

    assert tp > 0 and pp > 0 and cp > 0 and ep > 0, (
        f"[{role_name}] TP/PP/CP/EP must be positive. Got TP={tp}, PP={pp}, CP={cp}, EP={ep}"
    )
    model_parallel_size = tp * pp * cp * ep
    assert world_size % model_parallel_size == 0, (
        f"[{role_name}] world_size ({world_size}) must be divisible by TP*PP*CP*EP ({model_parallel_size}) "
        f"where TP={tp}, PP={pp}, CP={cp}, EP={ep}"
    )
    dp = world_size // model_parallel_size
    assert dp >= 1, f"[{role_name}] derived DP must be >= 1, got {dp}"
    return model_parallel_size, dp


def summarize_parallelism_state() -> dict[str, int]:
    """Return runtime Megatron parallel state summary."""
    return {
        "tp_size": int(mpu.get_tensor_model_parallel_world_size()),
        "pp_size": int(mpu.get_pipeline_model_parallel_world_size()),
        "cp_size": int(mpu.get_context_parallel_world_size()),
        "dp_size": int(mpu.get_data_parallel_world_size()),
        "tp_rank": int(mpu.get_tensor_model_parallel_rank()),
        "pp_rank": int(mpu.get_pipeline_model_parallel_rank()),
        "cp_rank": int(mpu.get_context_parallel_rank()),
        "dp_rank": int(mpu.get_data_parallel_rank()),
    }


def initialize_megatron_model_parallel(megatron_cfg: Any) -> None:
    """Initialize Megatron model-parallel groups from config.

    Expected fields on ``megatron_cfg``:
      - tensor_model_parallel_size
      - pipeline_model_parallel_size
      - virtual_pipeline_model_parallel_size
      - context_parallel_size
      - expert_model_parallel_size
      - expert_tensor_parallel_size
    Optional field:
      - dynamic_context_parallel
    """
    if mpu.is_initialized():
        return

    tp, pp, cp, ep = get_parallelism_tuple(megatron_cfg)
    etp = _cfg_get(megatron_cfg, "expert_tensor_parallel_size", None)
    vpp = _cfg_get(megatron_cfg, "virtual_pipeline_model_parallel_size", None)

    extra_args = {}
    if getattr(megatron_cfg, "dynamic_context_parallel", False):
        sig = inspect.signature(mpu.initialize_model_parallel)
        assert "dynamic_context_parallel" in sig.parameters, (
            "dynamic_context_parallel is not supported in your installed Megatron version."
        )
        extra_args["dynamic_context_parallel"] = True

    logger.info(
        "Initializing Megatron model parallel: TP=%s PP=%s CP=%s EP=%s ETP=%s VPP=%s",
        tp,
        pp,
        cp,
        ep,
        etp,
        vpp,
    )
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        pipeline_model_parallel_split_rank=None,
        use_sharp=False,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
        nccl_communicator_config_path=None,
        **extra_args,
    )

