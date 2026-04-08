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
from typing import Any


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


def _parallel_state_optional_callables() -> list[tuple[str, Callable[[], Callable[..., Any]]]]:
    """(attribute_name, factory) where factory() returns the function to bind on parallel_state."""

    # mbridge get_model/build_model calls e.g. is_inside_encoder(pipeline_rank); Megatron may pass
    # other optional args. Accept *args, **kwargs so stubs match both call styles.

    def is_inside_encoder(*_args: Any, **_kwargs: Any) -> bool:
        return False

    def get_pipeline_model_parallel_decoder_start(*_args: Any, **_kwargs: Any) -> int:
        return 0

    def get_pipeline_model_parallel_split_rank(*_args: Any, **_kwargs: Any):
        return None

    def is_pipeline_stage_before_split(*_args: Any, **_kwargs: Any) -> bool:
        return False

    def is_pipeline_stage_after_split(*_args: Any, **_kwargs: Any) -> bool:
        return True

    def get_pipeline_model_parallel_encoder_end(*_args: Any, **_kwargs: Any) -> int:
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
    """Add parallel_state APIs expected by mbridge / TE when missing on older Megatron-LM.

    If a previous verl run installed an outdated shim (e.g. wrong arity), replace it when the
    bound function lives in this compat module; never override real megatron.core.parallel_state.
    """
    try:
        import megatron.core.parallel_state as ps
    except ImportError:
        return

    this_pkg = __name__

    for attr, factory in _parallel_state_optional_callables():
        cur = getattr(ps, attr, None)
        cur_mod = getattr(cur, "__module__", "") or ""
        if cur is not None and cur_mod.startswith("megatron.core.parallel_state"):
            continue
        if cur is None or cur_mod == this_pkg or this_pkg in cur_mod:
            setattr(ps, attr, factory())


def ensure_megatron_mcore_runtime_compat() -> None:
    """Apply all Megatron-Core runtime shims (ModelType + parallel_state). Safe to call repeatedly."""
    ensure_megatron_model_type_encoder_decoder_alias()
    ensure_megatron_parallel_state_optional_apis()
