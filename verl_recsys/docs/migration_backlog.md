# verl_recsys Migration Backlog

This document turns strategy into execution tasks for building `verl_recsys`
into a high-end end-to-end retrieval-ranking-reranking generative RecSys RL
post-training framework that supports OpenOneRec and GR2 both:

- independently (model-optimized paths), and
- universally (shared contracts and reusable core).

## North-star definition

`verl_recsys` should provide one framework that supports:

1. End-to-end RecSys RL post-training lifecycle
- Retrieval-aware candidate context ingestion
- Ranking policy optimization
- Re-ranking objective optimization with verifiable and anti-hacking rewards

2. Dual support mode
- Independent mode:
  - OpenOneRec-optimized path
  - GR2-optimized path
- Universal mode:
  - Shared adapters/contracts/trainer flow
  - Pluggable model family and objective preset

3. Production-minded guarantees
- Stable field-level contracts
- Reproducible objective behavior
- Cross-path parity checks
- Controlled divergence with documented ownership

## Scope model

### In scope

- Training/runtime framework architecture and contracts in `verl_recsys`
- Shared objective/reward/rollout semantics for ranking and re-ranking RL
- Compatibility surface for OpenOneRec and GR2

### Out of scope (for this backlog phase)

- Full online serving stack implementation
- ANN retrieval service implementation itself
- Infra-specific deployment automation

## Workstreams

## WS1: Contract and schema unification

Goal: enforce one source of truth for data and reward fields.

Tasks:

- [ ] WS1-T1: Finalize canonical field schema for candidate/re-ranking flow
  - keys: `ground_truth`, `generated_items`, `original_items`, fallback keys
  - location: `verl_recsys/docs/adapter_contract.md`
  - owner: recsys architecture
- [ ] WS1-T2: Add schema validation hooks in dataset/reward adapters
  - fail-safe behavior with warnings and explicit fallback metrics
- [ ] WS1-T3: Define objective compatibility matrix for all model families
  - include `gr2_dapo` support status and fallback policy

Acceptance criteria:

- A single documented schema exists and is referenced by all objective docs.
- Missing optional keys never cause silent behavior drift.
- Every model family preset declares support and fallback.

## WS2: Shared reranking core extraction

Goal: avoid logic duplication between `verl_rl` and `verl_recsys`.

Tasks:

- [ ] WS2-T1: Extract shared helper module for expansion and UID grouping
  - beam/two-stage expansion mapping
  - UID semantics for grouped advantage/reward
- [ ] WS2-T2: Extract shared reward-schema helper
  - order similarity calculation and fallback resolution
- [ ] WS2-T3: Replace duplicated inline behavior with shared module calls
  - keep thin wrappers in model-specific trees

Acceptance criteria:

- One implementation source for expansion/UID semantics.
- Wrappers in both paths consume shared helper APIs.
- Behavior parity tests pass on fixed fixtures.

## WS3: OpenOneRec and GR2 independent paths hardening

Goal: keep specialized performance while preserving framework consistency.

Tasks:

- [ ] WS3-T1: OpenOneRec profile hardening
  - explicit profile doc and config preset
  - runtime/perf assumptions captured
- [ ] WS3-T2: GR2 profile hardening
  - DAPO config pack and conditional reward thresholds
  - anti-order-preservation penalty sensitivity guidance
- [ ] WS3-T3: Profile isolation tests
  - verify independent profiles do not leak assumptions into each other

Acceptance criteria:

- Both profiles are runnable with documented knobs and expected behavior.
- Independent tuning does not break universal path contracts.

## WS4: Universal training interface

Goal: one entrypoint, many model families/objectives.

Tasks:

- [ ] WS4-T1: Define universal trainer interface contract
  - required hooks for data, rollout, reward, objective, eval
- [ ] WS4-T2: Adapter registration protocol and validation checks
  - mandatory contract checks at startup
- [ ] WS4-T3: Universal mode reference preset
  - includes OpenOneRec and GR2 as first-class exemplars

Acceptance criteria:

- New model family can be onboarded with adapters and preset only.
- Universal mode can switch family/objective without code edits.

## WS5: Evaluation and parity governance

Goal: prove improvements and prevent regressions.

Tasks:

- [ ] WS5-T1: Shared offline eval subset and protocol
  - same split for OpenOneRec path and GR2 path comparisons
- [ ] WS5-T2: Mandatory parity dashboard
  - pass@1, hit metrics, ndcg@k, and reward-hacking indicators
- [ ] WS5-T3: Drift budget and exception process
  - define acceptable deltas and required review when exceeded

Acceptance criteria:

- Every significant framework change includes parity comparison output.
- Reward-hacking indicators are tracked, not anecdotal.

## Milestones

## M1: Contract freeze (short-term)

Includes:

- WS1 complete
- initial WS5 protocol defined

Exit gate:

- no undocumented field usage in objective/reward path

## M2: Shared core convergence (mid-term)

Includes:

- WS2 complete
- WS3 profiles documented and tested

Exit gate:

- expansion/UID/reward-schema logic has one authoritative implementation

## M3: Universal framework readiness (mid-long term)

Includes:

- WS4 complete
- WS5 parity governance fully operational

Exit gate:

- OpenOneRec and GR2 run independently and via universal interface with
  reproducible parity reports

## Suggested ownership

- Architecture owner:
  - contracts, shared core, universal interface
- Model owners:
  - OpenOneRec profile, GR2 profile
- Evaluation owner:
  - parity protocol, drift governance, metrics reporting

## KPI candidates

- Framework KPIs
  - number of duplicated core logic sites (target downward)
  - time to onboard a new model family
- Model KPIs
  - pass@1/hit/ndcg@k parity or improvement
  - reward-hacking ratio under controlled eval prompts
- Reliability KPIs
  - schema fallback rate
  - contract validation warning rate

## Immediate next 2-3 sprints

Sprint A:

- complete WS1-T1/T2/T3
- stand up WS5-T1 eval subset

Sprint B:

- complete WS2-T1/T2
- draft OpenOneRec and GR2 profile docs (WS3-T1/T2)

Sprint C:

- complete WS2-T3 and WS4-T1
- run first mandatory parity report (WS5-T2)

