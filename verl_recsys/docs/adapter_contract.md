# GR Adapter Contract

This document defines the minimal contract to onboard a new generative recommender
model family into `verl_recsys`.

## Required adapter components

- `DatasetAdapter.wrap_dataset(dataset, is_train)`
  - Ensure emitted samples carry a `recsys_meta` dictionary.
  - Populate at least `model_family` and `is_train_split`.
- `ActorAdapter.prepare_runtime(config)`
  - Configure rollout backend/mode and any actor defaults.
  - Avoid mutating non-recsys global behavior.
- `RewardAdapter.wrap_reward_fn(reward_fn, is_validation=False)`
  - Optionally shape reward outputs.
  - Keep validation behavior deterministic and conservative.

## Objective compatibility

Each model family should support at least one objective preset:

- `distill_hybrid`
- `policy_only_grpo`
- `actor_critic_gae`

If a family cannot support one preset, document the reason and safe fallback.

## Onboarding checklist

1. Add actor adapter implementation in `verl_recsys/adapters/defaults.py` or a new module.
2. Register family in `verl_recsys/adapters/registry.py`.
3. Validate:
   - `python -m verl_recsys.main_recsys --config-name recsys_dryrun --cfg job --resolve`
   - `python -m verl_recsys.smoke.recsys_smoke`
4. Add a preset under `verl_recsys/config/presets/`.
5. Document the family in `verl_recsys/README.md`.

