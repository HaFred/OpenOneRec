#!/usr/bin/env bash
set -euo pipefail

cd /home/dyvm6xra/dyvm6xrauser45/fred/openonerec_fredfork/verl_rl
clear
export EVAL_BACKEND="${EVAL_BACKEND:-serving}"
export EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-1}"
export EVAL_DATA_PARALLEL_SIZE="${EVAL_DATA_PARALLEL_SIZE:-8}"
export EVAL_INTERNAL_DATA_PARALLEL_SIZE="${EVAL_INTERNAL_DATA_PARALLEL_SIZE:-1}"
export EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES"
export EVAL_SERVER_BASE_PORT="${EVAL_SERVER_BASE_PORT:-18000}"
export EVAL_SERVER_START_TIMEOUT="${EVAL_SERVER_START_TIMEOUT:-600}"

python recipe/onerec/recover_checkpoint_eval_topk.py --prune "$@"