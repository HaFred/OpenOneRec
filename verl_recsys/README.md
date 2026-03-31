# verl-recsys

`verl-recsys` is a unification layer for OpenOneRec RL workflows based on VeRL,
with a recsys-focused runtime surface:

- rollout server adapter
- agent loop worker
- multi-reward manager bootstrap
- trainer orchestrator
- checkpoint engine abstraction
- model-family adapters (HSTU and SASRec reference)
- objective preset registry (distill hybrid / GRPO / actor-critic)
- runtime acceleration toggles (async rollout/reward, fused kernels, sequence balancing)

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

## Adapter-first model support

`verl_recsys` now exposes a model-agnostic adapter bundle:

- dataset adapter: normalizes recsys metadata for training/eval splits
- actor adapter: configures family-specific runtime defaults
- reward adapter: optional lightweight shaped reward blending

Current model families:

- `hstu` (default)
- `sasrec` (reference portability adapter)

Example:

```bash
python -m verl_recsys.main_recsys \
  recsys.model.family=hstu \
  recsys.objective.name=distill_hybrid
```

## Objective presets

- `distill_hybrid`: on-policy distill + actor-critic style defaults
- `policy_only_grpo`: policy-only GRPO setup
- `actor_critic_gae`: classic PPO actor-critic baseline

Example:

```bash
python -m verl_recsys.main_recsys recsys.objective.name=policy_only_grpo
```

## Acceleration toggles

Enable performance features directly from recsys config:

```bash
python -m verl_recsys.main_recsys \
  recsys.acceleration.enable_async_rollout=true \
  recsys.acceleration.enable_async_reward=true \
  recsys.acceleration.enable_fused_kernels=true \
  recsys.acceleration.enable_sequence_balance=true
```

## Smoke checks

Dry-run resolution + no-op trainer build:

```bash
python -m verl_recsys.main_recsys --config-name recsys_dryrun --cfg job --resolve
python -m verl_recsys.smoke.recsys_smoke
```

## Portability presets

Preset overlays are available under `verl_recsys/config/presets/`:

- `hstu_balanced.yaml`
- `sasrec_balanced.yaml`

You can mirror these presets using direct overrides:

```bash
python -m verl_recsys.main_recsys \
  recsys.model.family=sasrec \
  recsys.objective.name=policy_only_grpo \
  recsys.acceleration.enable_async_rollout=true \
  recsys.acceleration.enable_async_reward=true \
  recsys.acceleration.enable_sequence_balance=true
```
