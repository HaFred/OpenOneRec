"""Explicit Megatron resolver for this repository.

Importing `megatron` always goes through this file so behavior is deterministic:
- default: use vendored `third_party/Megatron-LM/megatron`
- opt-out: set `VERL_DISABLE_VENDORED_MEGATRON=1` to use external Megatron
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_module_dir = Path(__file__).resolve().parent
_repo_root = _module_dir.parents[0]
_vendored_pkg = _repo_root / "third_party" / "Megatron-LM" / "megatron"

_disable_vendored = os.getenv("VERL_DISABLE_VENDORED_MEGATRON", "").lower() in {"1", "true", "yes"}
if not _disable_vendored:
    if not _vendored_pkg.exists():
        raise ModuleNotFoundError("Vendored Megatron is missing. Expected third_party/Megatron-LM/megatron/")
    __path__ = [str(_vendored_pkg)]  # type: ignore[name-defined]
else:
    external_paths = []
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry).resolve() / "megatron"
        if candidate.exists() and candidate.resolve() != _module_dir:
            external_paths.append(str(candidate))
    if not external_paths:
        raise ModuleNotFoundError(
            "VERL_DISABLE_VENDORED_MEGATRON is set, but no external 'megatron' package was found on PYTHONPATH."
        )
    __path__ = external_paths  # type: ignore[name-defined]
