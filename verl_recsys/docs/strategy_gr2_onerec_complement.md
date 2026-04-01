# GR2 and OneRec: Strategic Complement Plan

This document explains whether GR2-style DAPO features in `verl_recsys` should
complement or replace OneRec in `verl_rl`, and defines an execution strategy.

## Executive conclusion

`verl_recsys` GR2 features are a complement to OneRec, not a replacement.

- `verl_rl/recipe/onerec` is currently the strongest dedicated path for
  OneRec-centric training/runtime behavior (including two-stage generation).
- `verl_recsys` is the right layer to standardize reranking-oriented RL
  objectives (GR2/DAPO), reward safeguards, and model-family portability.
- The strategic goal is to avoid two diverging implementations of the same core
  behavior by moving to shared contracts and shared core modules over time.

## Terminology alignment (RecSys stack)

Using common RecSys terminology:

- Retrieval: candidate generation from large corpus.
- Scoring: estimate relevance/utility on candidate set.
- Re-ranking: reorder top candidates under richer objectives/constraints.

Current interpretation in this repo:

- OneRec RL workflow is primarily in scoring/re-ranking space.
- Two-stage generation (`think` then beam/item output) is reranking-optimized
  generation logic, not a full retrieval subsystem.
- GR2-style DAPO in `verl_recsys` improves reranking objective quality and
  reward robustness (especially anti-reward-hacking constraints).

Reference terminology:
- [Google Retrieval](https://developers.google.com/machine-learning/recommendation/dnn/retrieval)

## Why the GR2 path is complementary

1. Algorithmic complement
- Adds conditional verifiable reward logic to reduce order-preservation reward
  hacking risk.
- Keeps DAPO knobs explicit in a recsys-facing configuration surface.

2. Architecture complement
- Places reranking objective evolution in a model-agnostic layer
  (`verl_recsys`) while preserving OneRec-specific runtime value in `verl_rl`.

3. Organizational complement
- Allows experimentation velocity in `verl_recsys` without destabilizing the
  OneRec-focused baseline path in `verl_rl`.

## Risks if left as-is

- Duplicate logic drift:
  - two-stage expansion semantics
  - UID grouping behavior
  - reward key conventions
- Inconsistent online/offline behavior due to implicit field assumptions.
- Higher maintenance cost and harder debugging across two trees.

## Strategic operating model

### Role split

- `verl_rl`:
  - production-ready OneRec runtime baseline
  - OneRec-specific worker/rollout optimizations
- `verl_recsys`:
  - recsys objective layer (GR2/DAPO and future variants)
  - cross-family contracts and adapters
  - standardized reward and reranking constraints

### Single source of truth policy

For shared mechanics, there must be one owning implementation:

- Candidate/order field contract: owner `verl_recsys/docs/adapter_contract.md`
- Two-stage expansion semantics: owner `verl_recsys` rollout contract
- Reward conditional penalty semantics: owner `verl_recsys/reward` contract

`verl_rl` should consume these contracts or shared modules instead of silently
redefining behavior.

## Migration roadmap

### Phase A: Stabilize contracts (now)
- Freeze field-level contract keys and fallback order.
- Add compatibility matrix for model families and objective presets.
- Define acceptance checks for reward/rerank consistency.

### Phase B: Extract shared core
- Move shared behavior (expansion mapping, UID semantics, reward schema helpers)
  into reusable modules with clear tests.
- Keep thin wrappers in both trees.

### Phase C: Converge call paths
- Prefer `verl_recsys` entrypoint for new reranking RL work.
- Backport only validated deltas to `verl_rl` baseline where needed.

## Implementation checklist (strategic recommendations in action)

- [ ] Publish contract as mandatory review gate for recsys changes.
- [ ] Require any new objective preset to declare:
  - expected data fields
  - fallback behavior
  - validation metrics
- [ ] Add parity checks between OneRec baseline and GR2 path on a shared eval
  subset (pass@1/hit/ndcg@k as applicable).
- [ ] Track drift budget:
  - if behavior diverges, document rationale and sunset plan.

## Decision rule for future changes

When adding a feature:

1. If it is OneRec hardware/runtime specific, land in `verl_rl` first.
2. If it is objective/reward/reranking-policy generic, land in `verl_recsys`.
3. If used by both, extract shared core and keep wrappers.

