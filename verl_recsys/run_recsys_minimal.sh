#!/bin/bash
set -euo pipefail

# Minimal recsys launcher built on verl_recsys + on-policy distillation path.
# Required:
#   BASE_MODEL=/path/to/student
#   TEACHER_MODEL=/path/to/teacher
#   DATASET_PARQUET=/path/to/train_or_mixed.parquet

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export BASE_MODEL="${BASE_MODEL:-}"
export TEACHER_MODEL="${TEACHER_MODEL:-}"
export DATASET_PARQUET="${DATASET_PARQUET:-}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
export NNODES="${NNODES:-1}"
export EXP_NAME="${EXP_NAME:-verl_recsys_minimal_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "$BASE_MODEL" || -z "$TEACHER_MODEL" || -z "$DATASET_PARQUET" ]]; then
  echo "[ERROR] Set BASE_MODEL, TEACHER_MODEL, DATASET_PARQUET."
  exit 1
fi

python -m verl_recsys.main_recsys \
  data.train_files="$DATASET_PARQUET" \
  data.val_files="$DATASET_PARQUET" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  algorithm.adv_estimator=on_policy_distill \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  recsys.distillation.teacher_model_paths="[$TEACHER_MODEL]" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
  trainer.project_name=verl-recsys \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.nnodes="$NNODES" \
  trainer.total_epochs=1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False
