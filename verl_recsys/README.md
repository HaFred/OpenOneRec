# verl-recsys

`verl-recsys` is a unification layer for OpenOneRec RL workflows based on VeRL,
with a recsys-focused runtime surface:

- rollout server adapter
- agent loop worker
- multi-reward manager bootstrap
- trainer orchestrator
- checkpoint engine abstraction

## Roadmap alignment (VeRL 26Q1)

This implementation aligns with relevant items from VeRL's 26Q1 roadmap:

- **Model engine default transition**: exposed via `recsys.model_engine.mode`.
- **Rollout server evolution**: backend/mode/router replay/profile controls in `recsys.rollout`.
- **AgentLoop evolution**: multi-output sample handling via `AgentLoopWorker`.
- **Checkpoint engine abstraction**: `CheckpointEngineManager` compatibility layer.
- **On-policy distillation + multi-teacher extension point**:
  `recsys.training.mode` and `recsys.distillation.teacher_model_paths`.
- **TransferQueue-ready sync path**:
  runtime env propagation through `VERL_RECSYS_TRANSFER_QUEUE`.
- **Async trainer compatibility hooks**:
  runtime env flag `VERL_RECSYS_ASYNC_TRAINER` and rollout mode selection.

## Entry point

```bash
python -m verl_recsys.main_recsys
```
