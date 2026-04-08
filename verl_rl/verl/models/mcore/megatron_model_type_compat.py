# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Some Megatron-Core wheels only expose ModelType.encoder_or_decoder, while mbridge and
# verl.utils.megatron_utils still reference ModelType.encoder_and_decoder (legacy name).
# Patch affected modules in-place so attribute access and comparisons work.

from __future__ import annotations

import importlib
import sys
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
