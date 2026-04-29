#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

DEFAULT_EVAL_PY="$PROJECT_DIR/eval/openonerec_eval.py"
FALLBACK_EVAL_PY="/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/verl-gr-fork-workingbranch/eval/openonerec_eval.py"
EVAL_PY="${OPENONEREC_EVAL_PY:-$DEFAULT_EVAL_PY}"
if [[ ! -f "$EVAL_PY" && -f "$FALLBACK_EVAL_PY" ]]; then
  EVAL_PY="$FALLBACK_EVAL_PY"
fi
if [[ ! -f "$EVAL_PY" ]]; then
  echo "Cannot find openonerec_eval.py. Set OPENONEREC_EVAL_PY or place it at $DEFAULT_EVAL_PY." >&2
  exit 2
fi

ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT:-${CHECKPOINT_PATH:-}}"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  ACTOR_CHECKPOINT="$1"
  shift
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
checkpoint_args=()
if [[ -n "$ACTOR_CHECKPOINT" ]]; then
  checkpoint_args=(--actor-checkpoint "$ACTOR_CHECKPOINT")
elif [[ -n "$CHECKPOINT_ROOT" ]]; then
  checkpoint_args=(--checkpoint-root "$CHECKPOINT_ROOT")
else
  echo "Usage: $0 /path/to/global_step_N/actor [extra openonerec_eval.py args]" >&2
  echo "Or set ACTOR_CHECKPOINT / CHECKPOINT_ROOT in the environment." >&2
  exit 2
fi

if [[ -n "${EVAL_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES"
fi

TEST_PARQUET="${TEST_PARQUET:-$REPO_DIR/data/test.parquet}"
TEST_MAX_SAMPLE="${EVAL_TEST_MAX_SAMPLE:--1}"
BACKEND="${EVAL_BACKEND:-offline}"
TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${EVAL_INTERNAL_DATA_PARALLEL_SIZE:-1}"
K_VALUES="${EVAL_K_VALUES:-1,32}"
RESULT_DIR="${RESULT_DIR:-$SCRIPT_DIR/eval_outputs/results}"
RESULT_FILENAME="${RESULT_FILENAME:-}"

if [[ -n "${MERGED_MODEL_DIR:-}" ]]; then
  merged_args=(--merged-model-dir "$MERGED_MODEL_DIR")
else
  merged_args=()
fi

result_args=(--result-dir "$RESULT_DIR")
if [[ -n "$RESULT_FILENAME" ]]; then
  result_args+=(--result-filename "$RESULT_FILENAME")
fi

start_time=$(date +%s)
echo "Eval started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Eval script: $EVAL_PY"
echo "Test parquet: $TEST_PARQUET"

python "$EVAL_PY" \
  "${checkpoint_args[@]}" \
  "${merged_args[@]}" \
  --backend "$BACKEND" \
  --test-max-sample "$TEST_MAX_SAMPLE" \
  --trust-remote-code \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --data-parallel-size "$DATA_PARALLEL_SIZE" \
  --enforce-eager \
  --test-parquet "$TEST_PARQUET" \
  --k-values "$K_VALUES" \
  "${result_args[@]}" \
  "$@"
status=$?

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf 'Eval finished at: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
printf 'Eval elapsed time: %02d:%02d:%02d (%d seconds)\n' \
  $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60)) "$elapsed"

exit "$status"
