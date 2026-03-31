"""Agent loop worker abstractions for recsys pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentLoopOutput:
    """Container for one agent-loop pass output."""

    prompt: str
    context: dict[str, Any]
    metadata: dict[str, Any]


class AgentLoopWorker:
    """Small wrapper to normalize agent-loop semantics for training."""

    def __init__(self, allow_multi_output: bool = True):
        self.allow_multi_output = allow_multi_output

    def process_sample(self, sample: dict[str, Any]) -> list[AgentLoopOutput]:
        """Return one or more loop outputs for a single sample."""
        prompt = sample.get("prompt", "")
        default = AgentLoopOutput(prompt=prompt, context={}, metadata={"source": "default"})
        if self.allow_multi_output and "agent_outputs" in sample:
            outputs: list[AgentLoopOutput] = []
            for idx, output in enumerate(sample.get("agent_outputs", [])):
                outputs.append(
                    AgentLoopOutput(
                        prompt=output.get("prompt", prompt),
                        context=output.get("context", {}),
                        metadata={"source": "agent_outputs", "index": idx},
                    )
                )
            return outputs if outputs else [default]
        return [default]
