# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Install-time shims for older vendored Megatron-LM vs newer TE / mbridge / verl.
# Kept under verl.utils (not verl.models.mcore) so this module can be imported
# before verl.models.mcore.__init__ pulls in config_converter → megatron.core.
#
# 1) ModelType.encoder_and_decoder alias.
# 2) parallel_state: optional APIs added in newer Megatron-LM / expected by TE or verl
#    (missing e.g. at cb3bb4190e9e). Each is only installed if getattr(parallel_state, name)
#    is absent. Stubs assume decoder-only causal LM (no encoder–decoder PP split).

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType


def _patch_model_type_on_module(mod: ModuleType) -> bool:
    mt = getattr(mod, "ModelType", None)
    if mt is None or hasattr(mt, "encoder_and_decoder"):
        return False
    real = mt

    class ModelType:
        encoder_or_decoder = real.encoder_or_decoder
        encoder_and_decoder = real.encoder_or_decoder

    mod.ModelType = ModelType
    return True


def ensure_megatron_model_type_encoder_decoder_alias() -> None:
    """Idempotent: add encoder_and_decoder as an alias of encoder_or_decoder where missing."""
    for name in ("megatron.core.enums", "megatron.core.transformer.enums"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        _patch_model_type_on_module(mod)

    te = sys.modules.get("megatron.core.transformer.enums")
    if te is not None:
        gpt = sys.modules.get("megatron.core.models.gpt.gpt_model")
        if gpt is not None and getattr(gpt, "ModelType", None) is not None:
            if not hasattr(gpt.ModelType, "encoder_and_decoder"):
                gpt.ModelType = te.ModelType

    mb = sys.modules.get("mbridge.core.util")
    if mb is not None:
        _patch_model_type_on_module(mb)


def _parallel_state_optional_callables() -> list[tuple[str, Callable[[], Callable]]]:
    """(attribute_name, factory) where factory() returns the no-arg function to bind."""

    def is_inside_encoder() -> bool:
        return False

    def get_pipeline_model_parallel_decoder_start() -> int:
        return 0

    def get_pipeline_model_parallel_split_rank():
        return None

    def is_pipeline_stage_before_split() -> bool:
        # No encoder-only pipeline region (decoder-only / standard GPT).
        return False

    def is_pipeline_stage_after_split() -> bool:
        # Whole transformer stack treated as decoder side for stub purposes.
        return True

    def get_pipeline_model_parallel_encoder_end() -> int:
        return 0

    return [
        ("is_inside_encoder", lambda: is_inside_encoder),
        ("get_pipeline_model_parallel_decoder_start", lambda: get_pipeline_model_parallel_decoder_start),
        ("get_pipeline_model_parallel_split_rank", lambda: get_pipeline_model_parallel_split_rank),
        ("is_pipeline_stage_before_split", lambda: is_pipeline_stage_before_split),
        ("is_pipeline_stage_after_split", lambda: is_pipeline_stage_after_split),
        ("get_pipeline_model_parallel_encoder_end", lambda: get_pipeline_model_parallel_encoder_end),
    ]


def ensure_megatron_parallel_state_optional_apis() -> None:
    """Idempotent: add parallel_state APIs expected by newer TE/mcore/verl when missing."""
    try:
        import megatron.core.parallel_state as ps
    except ImportError:
        return

    for attr, factory in _parallel_state_optional_callables():
        if not hasattr(ps, attr):
            setattr(ps, attr, factory())


def ensure_megatron_mcore_runtime_compat() -> None:
    """Apply all Megatron-Core runtime shims (ModelType + parallel_state). Safe to call repeatedly."""
    ensure_megatron_model_type_encoder_decoder_alias()
    ensure_megatron_parallel_state_optional_apis()
