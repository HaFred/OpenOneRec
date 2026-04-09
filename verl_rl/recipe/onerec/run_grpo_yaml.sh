#!/bin/bash
# GRPO Training Script with Two-Stage Rollout
# Two-Stage Rollout: first generate to </think>, then insert <sid_begin> and beam search

set -e
export HYDRA_FULL_ERROR=1

# fred
clear
export CUDA_VISIBLE_DEVICES=2,3,6,7
# export CUDA_VISIBLE_DEVICES=4,5

# If you see: ncclUnhandledCudaError / "Failed to CUDA calloc ..." during broadcast_params,
# that is almost always GPU OOM (or fragmentation): ref model loads first, then actor+DDP
# in the same colocated worker, then vLLM — peak VRAM is very high.
# Mitigations: free other jobs on those GPUs (nvidia-smi), lower MAX_TOKENS_PER_GPU /
# STAGE2_BEAM_SIZE / RESPONSE_LENGTH / rollout gpu_memory_utilization, use TP>1 for huge
# models, or enable ref CPU offload below.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export RAY_IGNORE_VERSION_MISMATCH=1
export HYDRA_FULL_ERROR=1 
export TRAIN_FILES=${TRAIN_FILES:-"$DATA_DIR/train_1k.parquet"}
export VAL_FILES=${VAL_FILES:-"$DATA_DIR/test.parquet"}
export N_NODES=1
export NUM_CPUS=${NUM_CPUS:-""}
export ENABLE_THINK=True
export OPENIF_PRODUCT_PARQUET_SOURCE=data_source
export ROLLOUT_TP_SIZE=1
export TP_SIZE=1
# MAX_TOKENS_PER_GPU=8192
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export N_GPUS=4
export ONEREC_RECIPE_PATH="${ONEREC_RECIPE_PATH:-$SCRIPT_DIR/onerec_recipe.py}"

# args
export TRAIN_BATCH_SIZE=2
export ROLLOUT_N=2

# bash init_ray.sh
# # ============================================================================
# # Cluster Configuration (auto-detect from Ray)
# # ============================================================================
# pip show ray
# RAY_INFO=$(python -c "import ray; ray.init(address='auto', ignore_reinit_error=True); nodes = [n for n in ray.nodes() if n['Alive']]; gpus=next((int(n.get('Resources',{}).get('GPU',0)) for n in nodes if n.get('Resources',{}).get('GPU',0)>0), 0); print(f'{len(nodes)} {gpus}')")

# # echo $RAY_INFO  # OVERLAPPING N NODES/GPUS

# # export N_NODES=$(echo $RAY_INFO | awk '{print $1}')
# # export N_GPUS=$(echo $RAY_INFO | awk '{print $2}')

# if [ -z "$N_NODES" ] || [ -z "$N_GPUS" ] || [ "$N_NODES" -eq 0 ]; then
#     echo "Could not detect Ray cluster. Using defaults: N_NODES=1, N_GPUS=8"
#     export N_NODES=1
#     export N_GPUS=8
# else
#     echo "Detected Ray cluster: $N_NODES nodes, $N_GPUS GPUs per node"
# fi

# ============================================================================
# Model Configuration
# ============================================================================
export TP_SIZE=${TP_SIZE:-1}
export PP_SIZE=${PP_SIZE:-1}
export CP_SIZE=${CP_SIZE:-1}
export EP_SIZE=${EP_SIZE:-1}

# ============================================================================
# Model Configuration
# ============================================================================
export BASE_MODEL=${BASE_MODEL:-"/path/to/your/model"}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
export VLLM_ATTENTION_BACKEND=XFORMERS

# ============================================================================
# Training Hyperparameters
# ============================================================================
export LEARNING_RATE=${LEARNING_RATE:-2e-6}
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
export TEMPERATURE=${TEMPERATURE:-1}

# ============================================================================
# Batch Size Configuration
# ============================================================================
export USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-40960}
# export TRAIN_BATCH_SIZE=$N_GPUS

# ============================================================================
# Rollout Configuration
# ============================================================================
export ROLLOUT_N=${ROLLOUT_N:-1}
export STAGE2_BEAM_SIZE=${STAGE2_BEAM_SIZE:-32}
export RESPONSE_LENGTH=${RESPONSE_LENGTH:-2048}
export STAGE1_MAX_TOKENS=${STAGE1_MAX_TOKENS:-1024}
export STAGE2_NUM_TOKENS=${STAGE2_NUM_TOKENS:-3}

# Think mode configuration
export ENABLE_THINK=${ENABLE_THINK:-False}
export ENABLE_NONTHINK=${ENABLE_NONTHINK:-False}
export USE_FORCE_PREFIX=${USE_FORCE_PREFIX:-False}

# Thinking Quality Reward (Trust-GRPO inspired, monitors stage-1 CoT quality)
# Set ENABLE_THINKING_REWARD=True to blend thinking-quality signal into the
# GRPO reward:  score = (1-w)*outcome + w*thinking_quality*trust_weight
export ENABLE_THINKING_REWARD=${ENABLE_THINKING_REWARD:-False}
export THINKING_REWARD_WEIGHT=${THINKING_REWARD_WEIGHT:-0.3}

# ============================================================================
# Data Configuration
# ============================================================================
export DATA_DIR=${DATA_DIR:-"$(realpath ../output/rl_data)"}
export TRAIN_FILES=${TRAIN_FILES:-"[$DATA_DIR/train.parquet]"}
export VAL_FILES=${VAL_FILES:-"[$DATA_DIR/test.parquet]"}

# ============================================================================
# Output Configuration
# ============================================================================
export PROJECT_NAME=${PROJECT_NAME:-"OneRec_RL"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_two_stage"}
export OUTPUT_DIR=${OUTPUT_DIR:-"./output"}
export WANDB_MODE=${WANDB_MODE:-offline}

# ============================================================================
# Network Configuration (for distributed training)
# ============================================================================
export TCP_NIC=$(ifconfig 2>/dev/null | grep -B1 " "$(hostname -i 2>/dev/null)" " | grep -o "^\w*" || echo "eth0")
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}

# ============================================================================
# Print Configuration
# ============================================================================
echo "==================================="
echo "GRPO Training with Two-Stage Rollout"
echo "==================================="
echo "Model: $BASE_MODEL"
echo "Cluster: $N_NODES nodes x $N_GPUS GPUs"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Rollout N: $ROLLOUT_N"
echo "Stage2 Beam Size: $STAGE2_BEAM_SIZE"
echo "Enable Think: $ENABLE_THINK"
echo "Enable NonThink: $ENABLE_NONTHINK"
echo "==================================="


# ============================================================================
# Pre-flight: validate model architecture for Megatron mcore (optional)
# Module path must be recipe.onerec.* (there is no package validate_onerec).
# Set SKIP_MCORE_PREFLIGHT=1 to skip.
# ============================================================================
if [ "${SKIP_MCORE_PREFLIGHT:-0}" != "1" ]; then
    python3 -u -m recipe.onerec.validate_megatron_mcore_support \
        --model-path "$BASE_MODEL" \
        --expected-arch "$EXPECTED_MODEL_ARCH"
fi

# # Extra Hydra overrides only when thinking-quality reward is enabled (keeps
# # config tree identical to the stable script when this feature is off).
# THINK_REWARD_HYDRA=()
# case "$(printf '%s' "$ENABLE_THINKING_REWARD" | tr '[:upper:]' '[:lower:]')" in
#     true|1|yes)
#         THINK_REWARD_HYDRA+=(++custom_reward_function.reward_kwargs.enable_thinking_reward=True)
#         THINK_REWARD_HYDRA+=(++custom_reward_function.reward_kwargs.thinking_reward_weight="$THINKING_REWARD_WEIGHT")
#         ;;
# esac

mkdir -p logs

cd "$PROJECT_DIR"

# Optional Hydra overrides (OOM during init_workers / actor_rollout_init_model):
#   REF_PARAM_OFFLOAD=1 — offload reference weights to CPU after load (less VRAM, slower KL).
HYDRA_OVERRIDES=()
if [ "${REF_PARAM_OFFLOAD:-0}" = "1" ]; then
    HYDRA_OVERRIDES+=(++actor_rollout_ref.ref.megatron.param_offload=true)
fi

# Defaults live in verl/trainer/config/onerec_grpo_megatron.yaml (${oc.env:...} + Hydra compose).
# Only pass what must come from this shell (parquet lists, think flags, optional thinking reward).
python3 -u -m recipe.onerec.main_onerec_ppo "${HYDRA_OVERRIDES[@]}"