"""Rollout server adapter used by verl-recsys."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass
class RolloutServerConfig:
    backend: str = "vllm"
    mode: str = "sync"
    enable_router_replay: bool = False
    enable_profile: bool = False


class RolloutServerAdapter:
    """Backend-agnostic rollout control point.

    This acts as a thin compatibility layer so recsys orchestration can track
    rollout timing regardless of whether the underlying engine is vLLM, sglang,
    or future backends such as TensorRT-LLM.
    """

    def __init__(self, config: RolloutServerConfig):
        self.config = config
        self._last_profile: dict[str, float] = {}
        self._events: dict[str, float] = {}

    def begin_step(self) -> None:
        if self.config.enable_profile:
            self._last_profile["step_start"] = perf_counter()
            self._events = {}

    def end_step(self) -> dict[str, float]:
        if not self.config.enable_profile or "step_start" not in self._last_profile:
            return {}
        elapsed = perf_counter() - self._last_profile["step_start"]
        metrics = {"rollout_server/step_latency_s": elapsed}
        metrics.update(self._events)
        return metrics

    def mark_event(self, event_name: str) -> None:
        """Capture coarse profile points used by recsys acceleration diagnostics."""
        if not self.config.enable_profile or "step_start" not in self._last_profile:
            return
        now = perf_counter() - self._last_profile["step_start"]
        self._events[f"rollout_server/{event_name}_at_s"] = now

    def inject_runtime_flags(self, runtime_env: dict[str, Any]) -> dict[str, Any]:
        """Attach rollout-related flags for downstream workers."""
        env = dict(runtime_env)
        env["VERL_RECSYS_ROLLOUT_BACKEND"] = self.config.backend
        env["VERL_RECSYS_ROLLOUT_MODE"] = self.config.mode
        env["VERL_RECSYS_ROUTER_REPLAY"] = "1" if self.config.enable_router_replay else "0"
        return env
