# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Re-export compat shims from verl.utils so imports stay stable; implementation
# lives in verl.utils.megatron_mcore_compat to avoid importing this package
# before parallel_state is patched (verl.models.mcore.__init__ loads registry first).

from verl.utils.megatron_mcore_compat import (
    ensure_megatron_mcore_runtime_compat,
    ensure_megatron_model_type_encoder_decoder_alias,
    ensure_megatron_parallel_state_optional_apis,
)

__all__ = [
    "ensure_megatron_mcore_runtime_compat",
    "ensure_megatron_model_type_encoder_decoder_alias",
    "ensure_megatron_parallel_state_optional_apis",
]
