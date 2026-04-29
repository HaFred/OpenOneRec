#!/usr/bin/env python3
"""Launch vLLM OpenAI server with tokenizer compatibility patches."""

from __future__ import annotations

import runpy


def _ensure_tokenizer_compatibility() -> None:
    try:
        from transformers import PreTrainedTokenizerBase
    except ImportError:
        return

    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return

    @property
    def all_special_tokens_extended(self):
        special_map = getattr(self, "special_tokens_map_extended", None) or {}
        tokens = []
        for value in special_map.values():
            if isinstance(value, (list, tuple)):
                tokens.extend(value)
            elif value is not None:
                tokens.append(value)
        if tokens:
            return tokens
        return list(getattr(self, "all_special_tokens", []))

    PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended


if __name__ == "__main__":
    _ensure_tokenizer_compatibility()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
