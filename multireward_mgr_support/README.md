# MultiRewardManager for verl

> A composable multi-reward manager for RL training of omni models and generative recommendation systems, designed as a drop-in extension to [verl](https://github.com/verl-project/verl).

---

## Table of Contents

- [Background & Motivation](#background--motivation)
- [Research & Analysis](#research--analysis)
- [Architecture & Design](#architecture--design)
- [Installation](#installation)
- [Usage](#usage)
- [Reward Functions for Generative RecSys](#reward-functions-for-generative-recsys)
- [Configuration Reference](#configuration-reference)
- [Testing](#testing)
- [PR Submission Guide](#pr-submission-guide)
- [File Structure](#file-structure)

---

## Background & Motivation

### The Problem

Current verl reward managers (`NaiveRewardManager`, `DAPORewardManager`, `PrimeRewardManager`, `BatchRewardManager`) accept a **single** `compute_score` function that returns one scalar reward per sample. This works well for single-objective tasks (e.g., math accuracy) but is insufficient for:

1. **Generative recommendation models** that must optimize multiple competing objectives simultaneously (engagement, diversity, novelty, coherence, format compliance).
2. **Omni models** that handle heterogeneous tasks requiring different reward signals.
3. **Multi-objective RL** where per-dimension reward tracking enables better diagnostics, reward shaping, and advantage estimation (e.g., GDPO).

### The Gap

| Feature | Existing verl managers | MultiRewardManager |
|---------|----------------------|-------------------|
| Single reward function | ✅ | ✅ |
| Multiple reward components | ❌ | ✅ |
| Configurable aggregation | ❌ | ✅ (weighted_sum, product, min, max) |
| Per-component logging | ❌ (manual via dict return) | ✅ (automatic) |
| Per-component normalization | ❌ | ✅ (running mean/std) |
| Per-component clipping | ❌ | ✅ |
| Registry-compatible | N/A | ✅ (`@register("multi")`) |

---

## Research & Analysis

### Existing verl Reward Manager Architecture

verl's reward manager system consists of:

- **`AbstractRewardManager`** (`verl/workers/reward_manager/abstract.py`): ABC defining `__init__(tokenizer, num_examine, compute_score, reward_fn_key)` and `__call__(data, return_dict)`.
- **Registry** (`verl/workers/reward_manager/registry.py`): `@register(name)` decorator + `get_reward_manager_cls(name)` lookup.
- **Built-in managers**: `naive`, `dapo`, `prime`, `batch` — all single-`compute_score`.
- **`load_reward_manager()`** (`verl/trainer/ppo/reward.py`): Instantiates a manager from config, resolving custom reward functions and sandbox URLs.
- **`compute_reward()`**: Calls `reward_fn(data, return_dict=True)` and extracts `reward_tensor` + `reward_extra_info`.

### Related Work in verl Ecosystem

1. **GDPO** ([NVlabs/GDPO](https://github.com/NVlabs/GDPO)): Handles multi-dimensional rewards at the **advantage estimation** level (decoupled per-reward normalization in GRPO). GDPO modifies `core_algos.py` but does NOT provide a multi-reward **manager** — it assumes the reward function already returns per-dimension scores.

2. **Reward Loop RFC** ([verl#4318](https://github.com/verl-project/verl/issues/4318)): Plans to deprecate legacy `RewardManager` in favor of async sample-wise processing via `RewardManagerWorker`. The RFC explicitly calls out the need for "integrating rule-based functions with feedback from reward models" and "even multiple reward models" as a motivation.

3. **Issue #2115** — Users requesting per-component reward logging to W&B.
4. **Issue #4346** — Redundant reward manager implementations for different rollout modes.
5. **Issue #4390** — Questions about registering custom reward managers.

### Key Insight

There is **no existing `MultiRewardManager`** in verl or its ecosystem. The closest is GDPO's per-dimension advantage, but it operates downstream of the reward manager. A composable multi-reward manager that sits at the reward computation level fills a clear gap.

---

## Architecture & Design

```
┌─────────────────────────────────────────────────────────┐
│                   MultiRewardManager                     │
│                                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ Component 1  │ │ Component 2  │ │ Component N  │       │
│  │ engagement   │ │ diversity    │ │ format       │       │
│  │ w=0.5        │ │ w=0.3        │ │ w=0.05       │       │
│  │ score_fn()   │ │ score_fn()   │ │ score_fn()   │       │
│  │ [normalize]  │ │ [clip]       │ │ [clip]       │       │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘       │
│         │                │                │               │
│         └────────────────┼────────────────┘               │
│                          │                                │
│                 ┌────────▼────────┐                       │
│                 │  Aggregation    │                       │
│                 │  (weighted_sum) │                       │
│                 │  [normalize]    │                       │
│                 │  [clip]         │                       │
│                 └────────┬────────┘                       │
│                          │                                │
│              reward_tensor + reward_extra_info             │
└──────────────────────────┼───────────────────────────────┘
                           │
                    verl PPO pipeline
                    (advantage estimation, policy update)
```

### Design Decisions

1. **Subclass `AbstractRewardManager`**: Full compatibility with verl's existing `load_reward_manager()` and `compute_reward()` pipeline.

2. **Registry-based registration**: `@register("multi")` enables zero-code-change integration — just set `reward_manager: multi` in config.

3. **Dataclass configuration**: `MultiRewardConfig` / `RewardComponentConfig` for type safety and serialization, while also supporting flat kwargs for verl's Hydra/OmegaConf config system.

4. **Per-component scoring functions**: Each component can have its own `compute_score_fn` with the same signature as verl's standard `compute_score`. This allows mixing rule-based rewards with model-based rewards.

5. **Running normalization**: Welford's online algorithm for components with different scales (e.g., engagement in [0,1] vs. a model-based reward in [-5,5]).

6. **Automatic per-component logging**: All component scores are placed in `reward_extra_info` with keys `reward_{component_name}`, directly compatible with verl's W&B logging (addresses Issue #2115).

---

## Installation

No pip install needed — this is designed to be placed alongside verl or submitted as a PR.

```bash
# Ensure verl is installed
pip install verl

# Copy or symlink this directory into your project
cp -r multireward_mgr_support/ /path/to/your/project/
```

---

## Usage

### Pattern 1: Programmatic (Python API)

```python
from multireward_mgr_support.reward_manager import MultiRewardManager
from multireward_mgr_support.reward_manager.multi_reward_manager import (
    MultiRewardConfig, RewardComponentConfig, AggregationStrategy,
)
from multireward_mgr_support.reward_functions.recsys_rewards import (
    engagement_reward, diversity_reward, novelty_reward,
)

config = MultiRewardConfig(
    components=[
        RewardComponentConfig(name="engagement", weight=0.5, compute_score_fn=engagement_reward),
        RewardComponentConfig(name="diversity", weight=0.3, compute_score_fn=diversity_reward),
        RewardComponentConfig(name="novelty", weight=0.2, compute_score_fn=novelty_reward),
    ],
    aggregation=AggregationStrategy.WEIGHTED_SUM,
)

mgr = MultiRewardManager(
    tokenizer=my_tokenizer,
    num_examine=2,
    multi_reward_config=config,
)

# Use in verl pipeline:
# reward_tensor = mgr(data_proto)
# or
# result = mgr(data_proto, return_dict=True)
# result["reward_tensor"], result["reward_extra_info"]
```

### Pattern 2: verl Config Integration

```python
# In your training entrypoint (e.g., custom main_ppo.py):
import multireward_mgr_support.reward_manager.register_multi  # registers "multi"

# Then in your YAML config:
# reward_model:
#   reward_manager: multi
#   reward_kwargs:
#     aggregation: weighted_sum
#     components:
#       - name: engagement
#         weight: 0.5
#       - name: diversity
#         weight: 0.3
```

See `configs/multi_reward_recsys.yaml` for a complete config example.

---

## Reward Functions for Generative RecSys

Five example reward functions are provided in `reward_functions/recsys_rewards.py`:

| Function | Signal | Description |
|----------|--------|-------------|
| `engagement_reward` | Relevance | Hit-rate: `|rec ∩ gt| / |gt|` |
| `diversity_reward` | Diversity | Unique categories / total items |
| `novelty_reward` | Novelty | Fraction of items NOT in user history |
| `coherence_reward` | Quality | Heuristic: non-empty, reasonable length, low repetition |
| `format_compliance_reward` | Format | JSON validity, required fields, error patterns |

All follow verl's `compute_score(data_source, solution_str, ground_truth, extra_info)` signature and return `dict` with a `"score"` key plus diagnostic fields.

### Extending with Custom Rewards

```python
def my_custom_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Your logic here
    return {"score": 0.8, "my_metric": 42}
```

---

## Configuration Reference

### `MultiRewardConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `components` | `list[RewardComponentConfig]` | `[]` | Reward components |
| `aggregation` | `str` | `"weighted_sum"` | `weighted_sum`, `product`, `min`, `max`, `weighted_sum_clipped` |
| `normalize_final` | `bool` | `False` | Running normalization on aggregated reward |
| `log_components` | `bool` | `True` | Log per-component scores |
| `clip_final_min` | `float?` | `None` | Min clip for final reward |
| `clip_final_max` | `float?` | `None` | Max clip for final reward |

### `RewardComponentConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Unique component identifier |
| `weight` | `float` | `1.0` | Aggregation weight |
| `compute_score_fn` | `Callable?` | `None` | Scoring function (falls back to default) |
| `normalize` | `bool` | `False` | Running normalization for this component |
| `clip_min` | `float?` | `None` | Min clip for component score |
| `clip_max` | `float?` | `None` | Max clip for component score |

### Aggregation Strategies

- **`weighted_sum`**: `Σ(weight_i × score_i)` — most common, additive combination
- **`product`**: `Π(score_i ^ weight_i)` — multiplicative, rewards balanced performance
- **`min`**: `min(weight_i × score_i)` — bottleneck, optimize worst component
- **`max`**: `max(weight_i × score_i)` — optimistic, reward best component
- **`weighted_sum_clipped`**: Same as weighted_sum with final clipping applied

---

## Testing

```bash
# Run all tests (no verl dependency needed)
cd multireward_mgr_support
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_recsys_rewards.py -v
python -m pytest tests/test_multi_reward_manager.py -v

# Run example
python examples/usage_example.py
```

---

## PR Submission Guide

### Target Repository

**https://github.com/verl-project/verl** (main branch)

### Proposed File Locations in verl

When submitting the PR, the files should be reorganized into verl's structure:

```
verl/
├── workers/
│   └── reward_manager/
│       ├── __init__.py          # Add MultiRewardManager import
│       └── multi.py             # ← multi_reward_manager.py
├── utils/
│   └── reward_score/
│       └── recsys.py            # ← recsys_rewards.py (optional)
└── tests/
    └── workers/
        └── reward_manager/
            ├── test_multi_reward_manager.py
            └── test_recsys_rewards.py
```

### Changes to Existing Files

1. **`verl/workers/reward_manager/__init__.py`**: Add import:
   ```python
   from .multi import MultiRewardManager
   ```
   And add `"MultiRewardManager"` to `__all__`.

2. **`verl/workers/reward_manager/multi.py`**: The core `MultiRewardManager` class with `@register("multi")`.

3. **`docs/preparation/reward_function.rst`**: Add section on multi-reward setup.

### PR Description Template

```markdown
## Summary

Add `MultiRewardManager` — a composable reward manager that combines multiple
reward functions with configurable aggregation for multi-objective RL training.

## Motivation

- Generative recommendation models need to optimize multiple competing objectives
  (engagement, diversity, novelty, coherence, format compliance) simultaneously.
- Omni models handling heterogeneous tasks require different reward signals.
- Current reward managers only accept a single `compute_score` function.
- Addresses needs raised in #2115 (per-component W&B logging) and aligns with
  the Reward Loop RFC (#4318) vision for flexible reward function composition.

## Changes

- Add `MultiRewardManager` registered as `"multi"` in the reward manager registry
- Support configurable aggregation strategies (weighted_sum, product, min, max)
- Per-component normalization, clipping, and logging
- Example recsys reward functions (engagement, diversity, novelty, coherence, format)
- Unit tests with >90% coverage

## Usage

```yaml
reward_model:
  reward_manager: multi
  reward_kwargs:
    aggregation: weighted_sum
    components:
      - name: engagement
        weight: 0.5
      - name: diversity
        weight: 0.3
      - name: novelty
        weight: 0.2
```

## Testing

```bash
python -m pytest tests/workers/reward_manager/test_multi_reward_manager.py -v
```
```

### Checklist Before Submitting

- [ ] All tests pass locally
- [ ] Code follows verl's style (Apache 2.0 license header, type hints, docstrings)
- [ ] No new dependencies added
- [ ] Compatible with both legacy and new reward loop architecture
- [ ] Per-component scores logged in `reward_extra_info` for W&B
- [ ] Works with existing advantage estimators (GRPO, GAE, GDPO, REINFORCE++)

---

## File Structure

```
multireward_mgr_support/
├── README.md                          # This file
├── configs/
│   └── multi_reward_recsys.yaml       # Example verl YAML config
├── examples/
│   └── usage_example.py               # Standalone usage demo
├── reward_functions/
│   ├── __init__.py
│   └── recsys_rewards.py              # 5 example recsys reward functions
├── reward_manager/
│   ├── __init__.py
│   ├── multi_reward_manager.py        # Core MultiRewardManager implementation
│   └── register_multi.py             # verl registry integration
└── tests/
    ├── __init__.py
    ├── test_multi_reward_manager.py   # Manager unit tests
    └── test_recsys_rewards.py         # Reward function unit tests
```

---

## License

Apache License 2.0 — same as verl.
