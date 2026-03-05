#!/bin/bash

# Set common variables
MODEL_PATH=$1
VERSION="${VERSION:-v1.0}"
BASE_OUTPUT_DIR="${BENCHMARK_BASE_DIR}/results/${VERSION}/results_${2}"
BASE_LOG_NAME="${BENCHMARK_BASE_DIR}/auto_eval_logs/${VERSION}/$2"
ENABLE_THINKING=$3
NUM_PROMPTS=${4:-"full"}  # Default to "full" if not specified
BATCH_SIZE=${5:-1875}     # Default batch size for product/video tasks

# Read configuration from environment variables (set by eval_script.py)
# Fallback to hardcoded paths if not set
BENCHMARK_BASE_DIR="${BENCHMARK_BASE_DIR:-/home/user/benchmark}"
DATA_VERSION="${DATA_VERSION:-v1.0}"

BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-${BENCHMARK_BASE_DIR}/data_${DATA_VERSION}}"
DATA_DIR="$BENCHMARK_DATA_DIR"

# Create output directory and log directory
mkdir -p "$(dirname "${BASE_LOG_NAME}")"
mkdir -p "$BASE_OUTPUT_DIR"

# Write debug info to log file
{
    echo "========== Task Configuration =========="
    echo "DATA_DIR: $DATA_DIR"
    echo "Enable Thinking: $ENABLE_THINKING"
    echo "Num Prompts: $NUM_PROMPTS"
    echo "Batch Size: $BATCH_SIZE"
    echo "========================================"
} >> "${BASE_LOG_NAME}.log"

# Build thinking arguments
THINKING_ARGS=""
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_ARGS="--enable_thinking"
fi

echo "Thinking args: $THINKING_ARGS"
echo "Num prompts: $NUM_PROMPTS"
echo "Batch size: $BATCH_SIZE"

echo "Running all tasks"

# Task: product
python3 -u scripts/ray-vllm/evaluate.py \
    --task_types product \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 \
    --worker_batch_size "$BATCH_SIZE" \
    --sample_size "$NUM_PROMPTS" \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1

# Task: video
python3 -u scripts/ray-vllm/evaluate.py \
    --task_types video \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --dtype bfloat16 \
    --worker_batch_size "$BATCH_SIZE" \
    --sample_size "$NUM_PROMPTS" \
    --overwrite \
    --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
    $THINKING_ARGS >> "${BASE_LOG_NAME}.log" 2>&1

echo "All tasks completed successfully"
