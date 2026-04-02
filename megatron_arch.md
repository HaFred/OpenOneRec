# Megatron Parallel Training Inclusion Analysis

## Scope

Compare:
- `8a6f5ac` — **"megatron 5D parallelism: MCore init"**
- `1f4a99e` — **merge PR #50 (clip norm & bugfix)**

Question: how Megatron parallel training is being included, and what to do next.

---

## Executive Conclusion

Megatron parallel training is included in this branch through the **`verl_rl` RL training path**, not through the `pretrain` path touched by `1f4a99e`.

In `8a6f5ac`, inclusion happens via three concrete layers:
1. **Launch/config wiring** (`run_grpo.sh`): switches strategy to `megatron`, exposes TP/PP/CP/EP, computes DP, and passes Megatron overrides.
2. **Topology validation** (`onerec_ray_trainer.py` + new helper): validates `world_size % (TP*PP*CP*EP) == 0` and computes DP consistently.
3. **Runtime model-parallel init** (`megatron_workers.py` + `engine/megatron/*`): centralizes MCore init (`initialize_model_parallel`) and applies it for actor/ref/critic/reward roles.

By contrast, `1f4a99e` only changes `pretrain/*` scripts and gradient clipping behavior (`max_grad_norm`) and does not wire or modify Megatron 5D RL initialization.

---

## What Changed in `8a6f5ac` (Megatron 5D)

### 1) Entry/launch path now explicitly enables Megatron in OneRec GRPO

File: `verl_rl/recipe/onerec/run_grpo.sh`

Key inclusion changes:
- Adds explicit parallel factors:
  - `TP_SIZE`, `PP_SIZE`, `CP_SIZE`, `EP_SIZE`
- Computes:
  - `WORLD_SIZE = N_NODES * N_GPUS`
  - `MODEL_PARALLEL_SIZE = TP * PP * CP * EP`
  - `DP_SIZE = WORLD_SIZE / MODEL_PARALLEL_SIZE`
- Fails early if divisibility is invalid.
- Uses `--config-name ppo_megatron_trainer`.
- Forces RL components to Megatron:
  - `actor_rollout_ref.actor.strategy=megatron`
  - `actor_rollout_ref.ref.strategy=megatron`
  - `critic.strategy=megatron`
  - `reward_model.strategy=megatron`
- Injects Megatron overrides for actor/ref TP/PP/CP/EP and sequence parallel.

Interpretation: this commit makes Megatron inclusion explicit and operator-facing in the GRPO launcher.

### 2) Trainer-level topology checks upgraded to 5D formula

File: `verl_rl/recipe/onerec/onerec_ray_trainer.py`

Validation changed from partial logic to full:
- Old style: mostly TP*PP plus separate CP handling.
- New style: `model_parallel_size = TP * PP * CP * EP`.
- Enforces `n_gpus % model_parallel_size == 0`.
- Derives `megatron_dp = n_gpus // model_parallel_size`.
- Uses derived DP for minimal batch-size sanity.
- Prints a 5D topology summary line for debugging.

Interpretation: this is the control-plane guardrail preventing invalid cluster/parallel settings before heavy training starts.

### 3) MCore init centralized and reused across worker roles

New file: `verl_rl/verl/workers/engine/megatron/transformer_impl.py`

Introduced helper API:
- `get_parallelism_tuple()`
- `validate_parallelism()`
- `initialize_megatron_model_parallel()`
- `summarize_parallelism_state()`

Behavior:
- Single source of truth for TP/PP/CP/EP checks and DP derivation.
- Initializes Megatron core parallel groups once (skip if already initialized).
- Supports optional `dynamic_context_parallel` with signature check.

Integration updates:
- `verl_rl/verl/workers/megatron_workers.py` now calls helper functions instead of directly inlining `mpu.initialize_model_parallel(...)` in actor/ref/critic/reward setup.
- `verl_rl/verl/workers/engine/megatron/engine_impl.py` no longer placeholder-only:
  - initializes torch distributed if needed
  - validates topology
  - initializes Megatron model-parallel groups
  - logs TP/PP/CP/DP runtime state
- `verl_rl/verl/workers/engine/__init__.py` exports `MegatronEngine`.

Interpretation: inclusion is not only "enabled" by config; it is also structurally integrated through shared initialization primitives.

---

## What `1f4a99e` Does (and Does Not Do)

`1f4a99e` is a merge commit for PR #50 (child commit `c1f67b7`):
- Adds `--max_grad_norm` in `pretrain/examples/*.sh`.
- Adds `clip_grad_norm()` utility in `pretrain/onerec_llm/training/gradients.py`.
- Switches `train_qwen3.py` gradient clipping path from value clipping to norm clipping.
- Fixes resume behavior when `resume_training_state` is false.

It does **not**:
- touch `verl_rl/recipe/onerec/*`,
- touch `verl_rl/verl/workers/megatron_workers.py`,
- touch `verl_rl/verl/workers/engine/megatron/*`,
- introduce 5D topology logic in RL path.

Interpretation: PR #50 is a **pretrain optimization/bugfix track**, while `8a6f5ac` is a **verl_rl Megatron 5D integration track**.

---

## Inclusion Path (End-to-End)

The Megatron parallel training path now looks like:

1. Operator launches `verl_rl/recipe/onerec/run_grpo.sh`
2. Script sets TP/PP/CP/EP and derives DP (with divisibility checks)
3. Hydra config uses `ppo_megatron_trainer` and Megatron strategies
4. `RayPPOTrainer._validate_config()` re-checks 5D constraints at trainer level
5. Worker startup (`ActorRolloutRefWorker`, `CriticWorker`, `RewardModelWorker`) calls centralized helper:
   - validate topology
   - initialize Megatron model parallel
   - print runtime group/rank summary
6. Megatron-based actor/ref/critic/reward training runs with consistent parallel state

This is how Megatron parallel training is being included in practice for this branch.

---

## Current Gaps / Risks

1. **Config asymmetry risk**  
   In `run_grpo.sh`, Megatron overrides are explicit for actor/ref; critic/reward rely more on inherited config defaults. If defaults drift, topology mismatch can occur.

2. **Engine shim incomplete for train/infer**  
   `MegatronEngine` initializes parallel groups but still raises `NotImplementedError` in `train_batch`/`infer_batch`. Current production path still depends on legacy `megatron_workers`.

3. **No explicit automated topology matrix tests**  
   There is runtime assert-based protection, but no clear CI/e2e matrix covering representative TP/PP/CP/EP combinations.

4. **Potential rollout/train parallel mismatch complexity**  
   `rollout.tensor_model_parallel_size` can diverge from train TP. Logged currently, but policy/validation for acceptable mismatch is still soft.

---

## OneRec-verl Megatron Training At Scale

```mermaid
flowchart TB
    subgraph L1["Layer 1: Launch and Topology Inputs"]
        U["Operator / Scheduler"]
        S1["run_grpo.sh"]
        T1["TP_SIZE"]
        T2["PP_SIZE"]
        T3["CP_SIZE"]
        T4["EP_SIZE"]
        T5["N_NODES x N_GPUS = WORLD_SIZE"]
        T6["MODEL_PARALLEL_SIZE = TP*PP*CP*EP"]
        T7["DP_SIZE = WORLD_SIZE / MODEL_PARALLEL_SIZE"]
        C0{"WORLD_SIZE % MODEL_PARALLEL_SIZE == 0 ?"}
    end

    U --> S1
    S1 --> T1
    S1 --> T2
    S1 --> T3
    S1 --> T4
    S1 --> T5
    T1 --> T6
    T2 --> T6
    T3 --> T6
    T4 --> T6
    T5 --> C0
    T6 --> C0
    C0 -->|Yes| T7
    C0 -->|No| E0["Fail Fast: invalid 5D topology"]

    subgraph L2["Layer 2: Hydra Config Wiring"]
        H0["config-name ppo_megatron_trainer"]
        H1["actor.strategy = megatron"]
        H2["ref.strategy = megatron"]
        H3["critic.strategy = megatron"]
        H4["reward_model.strategy = megatron"]
        H5["actor/ref megatron overrides\nTP/PP/CP/EP + sequence_parallel"]
    end

    T7 --> H0
    H0 --> H1
    H0 --> H2
    H0 --> H3
    H0 --> H4
    H0 --> H5

    subgraph L3["Layer 3: Trainer Validation Control Plane"]
        R0["RayPPOTrainer._validate_config()"]
        R1["Recompute model_parallel_size = TP*PP*CP*EP"]
        R2{"n_gpus % model_parallel_size == 0 ?"}
        R3["megatron_dp = n_gpus / model_parallel_size"]
        R4["minimal_bsz = megatron_dp * ppo_micro_batch_size_per_gpu"]
        R5["Log 5D summary:\nTP/PP/CP/EP/DP + rollout TP/DP"]
    end

    H1 --> R0
    H2 --> R0
    H3 --> R0
    H4 --> R0
    H5 --> R0
    R0 --> R1 --> R2
    R2 -->|Yes| R3 --> R4 --> R5
    R2 -->|No| E1["Assertion Error: invalid trainer topology"]

    subgraph L4["Layer 4: Runtime Bootstrap per Role"]
        W0["Worker roles:\nActorRolloutRef / Critic / RewardModel"]
        W1["Init torch.distributed process group\nNCCL backend + device set"]
        W2["validate_parallelism(megatron_cfg, world_size)"]
        W3["initialize_megatron_model_parallel(megatron_cfg)"]
        W4["summarize_parallelism_state()"]
        W5["Print runtime ranks:\ntp_rank, pp_rank, cp_rank, dp_rank"]
    end

    R5 --> W0 --> W1 --> W2 --> W3 --> W4 --> W5

    subgraph L5["Layer 5: Shared Megatron Helper Module"]
        M0["transformer_impl.py"]
        M1["get_parallelism_tuple()"]
        M2["validate_parallelism()"]
        M3["initialize_megatron_model_parallel()"]
        M4["summarize_parallelism_state()"]
        M5["Guard: skip init if mpu already initialized"]
        M6["Optional: dynamic_context_parallel\nsignature-checked"]
    end

    W2 --> M2
    W3 --> M3
    W4 --> M4
    M0 --> M1
    M0 --> M2
    M0 --> M3
    M0 --> M4
    M3 --> M5
    M3 --> M6

    subgraph L6["Layer 6: Effective Parallel Runtime"]
        P0["Megatron Parallel Groups"]
        P1["Tensor Parallel TP"]
        P2["Pipeline Parallel PP"]
        P3["Context Parallel CP"]
        P4["Expert Parallel EP"]
        P5["Data Parallel DP = WORLD / TP*PP*CP*EP"]
        P6["Consistent group state across roles"]
        P7["Megatron-based actor/ref/critic/reward training loop"]
    end

    W5 --> P0
    P0 --> P1
    P0 --> P2
    P0 --> P3
    P0 --> P4
    P0 --> P5
    P1 --> P6
    P2 --> P6
    P3 --> P6
    P4 --> P6
    P5 --> P6
    P6 --> P7

    subgraph X1["Orthogonal Track: 1f4a99e not RL Megatron wiring"]
        X0["pretrain scripts + gradients"]
        X2["max_grad_norm clipping + resume fix"]
    end

    X0 --> X2
```

---

## Possible Future Plan

### Phase 1: Hardening (short-term)
- Add a small validation utility/test matrix for `(TP, PP, CP, EP, WORLD_SIZE) -> DP` expectations.
- Enforce critic/reward Megatron parallel overrides explicitly in launcher (or validate they match actor/ref defaults).
- Add one concise "effective topology" printout in all roles with identical format for easier debugging.

### Phase 2: Reliability (mid-term)
- Add e2e smoke jobs for 2-3 topologies (e.g., `2x2x1x1`, `4x1x1x1`, `2x1x2x1`) in CI/nightly.
- Add fail-fast checks for rollout/training TP mismatch when unsupported combinations are set.
- Add regression coverage for re-init behavior when workers are colocated.

### Phase 3: Architecture convergence (long-term)
- Complete `MegatronEngine.train_batch` and `infer_batch` so engine path can replace legacy worker-only logic.
- Gradually migrate role-specific startup to engine abstraction while keeping helper module as shared source of truth.
- Add checkpoint/load support parity in `MegatronEngine` for full production readiness.

### Phase 4: Performance follow-up
- Profile communication cost by topology (esp. CP/EP) and define recommended presets by model scale.
- Add optional topology auto-suggestion given `world_size` and model config constraints.

---

## Bottom Line

Relative to `1f4a99e`, Megatron is included by `8a6f5ac` through explicit RL launcher wiring + 5D topology validation + centralized MCore initialization reuse across worker roles.  
`1f4a99e` is orthogonal (pretrain clipping/resume improvements) and does not contribute to Megatron 5D RL inclusion.
