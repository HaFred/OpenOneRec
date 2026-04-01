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

## Field-level reranking contract

To avoid hidden divergence between `verl_rl` and `verl_recsys`, the following
keys are the contract for reranking reward logic:

- `reward_model.ground_truth`
  - Verifiable target used by score functions.
- `extra_info.generated_items`
  - Generated ordered item list used for conditional reranking penalties.
- `extra_info.original_items`
  - Original candidate order reference for anti-order-preservation checks.
- `extra_info.candidate_order` (fallback)
  - Backward-compatible fallback when `original_items` is missing.

Required behavior:

- Missing optional keys must degrade gracefully, never crash by default.
- Reward functions must log whether fallback keys were used.
- Changes to key names must be reflected in config and docs in the same PR.

## Ownership and single source of truth

- Contract owner: `verl_recsys/docs/adapter_contract.md`.
- Shared behavior (UID grouping, expansion semantics, reward key mapping) should
  be defined once and reused by wrappers.
- If a behavior intentionally diverges in `verl_rl`, document:
  1) why divergence is needed, 2) expected impact, 3) sunset criteria.

## Compatibility matrix rule

Every model family onboarding should declare support status for:

- `distill_hybrid`
- `policy_only_grpo`
- `actor_critic_gae`
- `gr2_dapo`

If unsupported:

- provide explicit fallback preset
- provide rationale and blocker notes
- define re-evaluation trigger (what must change to support it)

## Onboarding checklist

1. Add actor adapter implementation in `verl_recsys/adapters/defaults.py` or a new module.
2. Register family in `verl_recsys/adapters/registry.py`.
3. Validate:
   - `python -m verl_recsys.main_recsys --config-name recsys_dryrun --cfg job --resolve`
   - `python -m verl_recsys.smoke.recsys_smoke`
4. Add a preset under `verl_recsys/config/presets/`.
5. Document the family in `verl_recsys/README.md`.
6. Validate field-level reranking contract:
   - ensure `ground_truth` and generated/candidate order keys are present or
     mapped with explicit fallback behavior.

