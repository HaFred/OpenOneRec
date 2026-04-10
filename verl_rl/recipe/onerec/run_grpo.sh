#!/bin/bash
# GRPO Training Script with Two-Stage Rollout
# Two-Stage Rollout: first generate to </think>, then insert <sid_begin> and beam search

set -e

export HYDRA_FULL_ERROR=1
# Default to in-process OneRec task execution (avoids CPU-only Ray TaskRunner import path).
export ONEREC_USE_RAY_TASKRUNNER=${ONEREC_USE_RAY_TASKRUNNER:-0}

# fred
clear
export CUDA_VISIBLE_DEVICES=2,3,6,7

# If you see: ncclUnhandledCudaError / "Failed to CUDA calloc ..." during broadcast_params,
# that is almost always GPU OOM: ref loads first, then actor+DDP in the same colocated worker,
# then vLLM. Hydra must not get empty ppo_micro_batch_size_per_gpu= — PPO_MICRO_BATCH_PER_GPU
# defaults to 1 below. ~8B with TP_SIZE=1 is a full replica per GPU; ref CPU offload defaults ON
# (REF_PARAM_OFFLOAD=0 to disable). vLLM KV OOM: shrink DATA_MAX_PROMPT_LENGTH / RESPONSE_LENGTH /
# STAGE2_BEAM_SIZE / ROLLOUT_MAX_NUM_SEQS, disable prefix cache (set in Hydra below), raise TP_SIZE
# and ROLLOUT_TP_SIZE, or ACTOR_PARAM_OFFLOAD=1. Preset: ONEREC_8B_4GPU=1.
#
# vLLM CuMem allocator is incompatible with PyTorch expandable_segments.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    echo "[run_grpo] Unsetting PYTORCH_CUDA_ALLOC_CONF for vLLM compatibility: ${PYTORCH_CUDA_ALLOC_CONF}"
    unset PYTORCH_CUDA_ALLOC_CONF
fi
# BASE_MODEL=/data/models/fredhong/hf_home/OneRec-8B-pro
# DATA_DIR=/home/fredhong/vllm_fork_genrec_workspace/fred_fork_openonerec/working_branch_fredfork_openonerec/output/rl_data
TRAIN_FILES=${TRAIN_FILES:-"[$DATA_DIR/train_1k.parquet]"}
VAL_FILES=${VAL_FILES:-"[$DATA_DIR/test.parquet]"}
N_NODES=1
N_GPUS=2
ENABLE_THINK=True
ROLLOUT_TP_SIZE=4
TP_SIZE=1
ROLLOUT_GPU_MEM_UTIL=0.75

# MAX_TOKENS_PER_GPU=8192
# TRAIN_BATCH_SIZE=1

# ============================================================================
# Cluster Configuration (auto-detect from Ray)
# ============================================================================
RAY_INFO=$(python -c "import ray; ray.init(address='auto', ignore_reinit_error=True); nodes = [n for n in ray.nodes() if n['Alive']]; gpus=next((int(n.get('Resources',{}).get('GPU',0)) for n in nodes if n.get('Resources',{}).get('GPU',0)>0), 0); print(f'{len(nodes)} {gpus}')" 2>/dev/null)

# echo $RAY_INFO  # OVERLAPPING N NODES/GPUS

export N_NODES=$(echo $RAY_INFO | awk '{print $1}')
export N_GPUS=$(echo $RAY_INFO | awk '{print $2}')

if [ -z "$N_NODES" ] || [ -z "$N_GPUS" ] || [ "$N_NODES" -eq 0 ]; then
    echo "Could not detect Ray cluster. Using defaults: N_NODES=1, N_GPUS=8"
    export N_NODES=1
    export N_GPUS=8
else
    echo "Detected Ray cluster: $N_NODES nodes, $N_GPUS GPUs per node"
fi

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
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
# vLLM V1 does not support XFORMERS attention backend.
# Default to V1 (recommended). If you explicitly need XFORMERS, force V0:
#   FORCE_VLLM_V0_XFORMERS=1
export VLLM_USE_V1=${VLLM_USE_V1:-1}
if [ "${FORCE_VLLM_V0_XFORMERS:-0}" = "1" ]; then
    export VLLM_USE_V1=0
    export VLLM_ATTENTION_BACKEND=XFORMERS
else
    unset VLLM_ATTENTION_BACKEND
fi

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
# Required: empty value becomes ppo_micro_batch_size_per_gpu= in Hydra and breaks training.
export PPO_MICRO_BATCH_PER_GPU=${PPO_MICRO_BATCH_PER_GPU:-1}
# compute_log_prob uses rollout.log_prob_micro_batch_size_per_gpu (not actor PPO micro-batch)
export LOGPROB_MICRO_BATCH_PER_GPU=${LOGPROB_MICRO_BATCH_PER_GPU:-$PPO_MICRO_BATCH_PER_GPU}
# Default ON: ref CPU offload after load (colocated ref+actor). Disable: REF_PARAM_OFFLOAD=0
export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-1}
export TRAIN_BATCH_SIZE=$((N_GPUS * N_NODES))

# ============================================================================
# Rollout / vLLM KV cache (hybrid engine shares each GPU with Megatron weights)
# ============================================================================
export ROLLOUT_N=${ROLLOUT_N:-1}
export RESPONSE_LENGTH=${RESPONSE_LENGTH:-2048}
# Cap dataset + vLLM context; 10240 + response needs ~80GB-class headroom for hybrid 8B.
export DATA_MAX_PROMPT_LENGTH=${DATA_MAX_PROMPT_LENGTH:-6144}
export STAGE1_MAX_TOKENS=${STAGE1_MAX_TOKENS:-1024}
export STAGE2_NUM_TOKENS=${STAGE2_NUM_TOKENS:-3}
# Beam width multiplies memory in stage-2; increase only if nvidia-smi shows free VRAM.
export STAGE2_BEAM_SIZE=${STAGE2_BEAM_SIZE:-8}
# vLLM scheduler slot count (2048 is far beyond typical GRPO microbatches and hurts KV planning).
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
# Fraction of total GPU memory vLLM may use; too low -> "No available memory for cache blocks" on v1.
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.68}
# Chunked prefill requires max_num_batched_tokens >= max_model_len (≈ prompt + response).
_ROLLOUT_ML=$((DATA_MAX_PROMPT_LENGTH + RESPONSE_LENGTH))
export ROLLOUT_MAX_MODEL_LEN=$_ROLLOUT_ML
if [ -z "${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-}" ]; then
    if [ "$_ROLLOUT_ML" -gt 8192 ]; then
        export ROLLOUT_MAX_NUM_BATCHED_TOKENS=$_ROLLOUT_ML
    else
        export ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
    fi
fi

# Optional preset: 8B + 4 GPUs — TP=2 for actor/ref + vLLM, actor offload before rollout (must run before echo / Hydra arrays).
if [ "${ONEREC_8B_4GPU:-0}" = "1" ]; then
    export TP_SIZE="${TP_SIZE:-2}"
    export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
    export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-1}"
    export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.72}"
fi

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
export OPENIF_PRODUCT_PARQUET_SOURCE=${OPENIF_PRODUCT_PARQUET_SOURCE:-"data_source"}

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
echo "Megatron TP_SIZE=$TP_SIZE  ROLLOUT_TP_SIZE=$ROLLOUT_TP_SIZE  ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-0}"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Rollout N: $ROLLOUT_N"
echo "Stage2 Beam Size: $STAGE2_BEAM_SIZE"
echo "Enable Think: $ENABLE_THINK"
echo "Enable NonThink: $ENABLE_NONTHINK"
echo "PPO micro-batch/GPU: $PPO_MICRO_BATCH_PER_GPU  REF_PARAM_OFFLOAD: $REF_PARAM_OFFLOAD"
echo "LogProb micro-batch/GPU (rollout/ref): $LOGPROB_MICRO_BATCH_PER_GPU"
echo "Data prompt cap: $DATA_MAX_PROMPT_LENGTH  response: $RESPONSE_LENGTH  vLLM max_model_len≈$ROLLOUT_MAX_MODEL_LEN"
echo "vLLM max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS  max_num_seqs=$ROLLOUT_MAX_NUM_SEQS  gpu_mem_util=$ROLLOUT_GPU_MEM_UTIL"
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

# Extra Hydra overrides only when thinking-quality reward is enabled (keeps
# config tree identical to the stable script when this feature is off).
THINK_REWARD_HYDRA=()
case "$(printf '%s' "$ENABLE_THINKING_REWARD" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes)
        THINK_REWARD_HYDRA+=(++custom_reward_function.reward_kwargs.enable_thinking_reward=True)
        THINK_REWARD_HYDRA+=(++custom_reward_function.reward_kwargs.thinking_reward_weight="$THINKING_REWARD_WEIGHT")
        ;;
esac

# Reference CPU offload (REF_PARAM_OFFLOAD exported above; slower KL when enabled)
REF_OFFLOAD_HYDRA=()
case "$(printf '%s' "$REF_PARAM_OFFLOAD" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off) ;;
    *)
        REF_OFFLOAD_HYDRA+=(++actor_rollout_ref.ref.megatron.param_offload=true)
        ;;
esac

# Optional extra VRAM (slower): actor weights / optimizer on CPU when set to 1
ACTOR_OFFLOAD_HYDRA=()
if [ "${ACTOR_PARAM_OFFLOAD:-0}" = "1" ]; then
    ACTOR_OFFLOAD_HYDRA+=(++actor_rollout_ref.actor.megatron.param_offload=true)
fi
if [ "${ACTOR_OPTIMIZER_OFFLOAD:-0}" = "1" ]; then
    ACTOR_OFFLOAD_HYDRA+=(++actor_rollout_ref.actor.megatron.optimizer_offload=true)
fi

# ============================================================================
# Launch Training
# ============================================================================
mkdir -p logs

python3 -u -m recipe.onerec.main_onerec_ppo \
    --config-name ppo_megatron_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.max_prompt_length=$DATA_MAX_PROMPT_LENGTH \
    ++data.enable_think=$ENABLE_THINK \
    ++data.enable_nonthink=$ENABLE_NONTHINK \
    ++data.use_force_prefix=$USE_FORCE_PREFIX \
    data.prompt_key='prompt' \
    data.shuffle=True \
    data.max_response_length=$RESPONSE_LENGTH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=$SCRIPT_DIR/onerec_recipe.py \
    data.custom_cls.name=OneRecDataset \
    data.reward_fn_key=$OPENIF_PRODUCT_PARQUET_SOURCE \
    ++data.data_source_key=$OPENIF_PRODUCT_PARQUET_SOURCE \
    ++actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    ++actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    ++actor_rollout_ref.model.enable_activation_offload=True \
    ++actor_rollout_ref.model.use_remove_padding=True \
    custom_reward_function.path=$SCRIPT_DIR/onerec_recipe.py \
    custom_reward_function.name=compute_score \
    "${THINK_REWARD_HYDRA[@]}" \
    "${REF_OFFLOAD_HYDRA[@]}" \
    "${ACTOR_OFFLOAD_HYDRA[@]}" \
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_PER_GPU \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BATCH_PER_GPU \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BATCH_PER_GPU \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
    ++actor_rollout_ref.rollout.max_model_len=$((ROLLOUT_MAX_MODEL_LEN)) \
    ++actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    ++actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$((ROLLOUT_N)) \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$((ROLLOUT_TP_SIZE)) \
    actor_rollout_ref.rollout.name=two_stage \
    ++actor_rollout_ref.rollout.backend=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
    ++actor_rollout_ref.rollout.max_length=$((RESPONSE_LENGTH)) \
    ++actor_rollout_ref.rollout.stage1_max_tokens=$((STAGE1_MAX_TOKENS)) \
    ++actor_rollout_ref.rollout.stage2_num_tokens=$((STAGE2_NUM_TOKENS)) \
    ++actor_rollout_ref.rollout.stage2_beam_size=$((STAGE2_BEAM_SIZE)) \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=320 \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$((N_GPUS))\
    trainer.nnodes=$((N_NODES)) \
    trainer.save_freq=10 \
    +trainer.max_ckpt_to_keep=2 \
    trainer.test_freq=50 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR/ckpt \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    ++critic.enable=False \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    critic.strategy=megatron \
    reward_model.strategy=megatron \
    ++actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    ++actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    ++actor_rollout_ref.actor.megatron.context_parallel_size=$CP_SIZE \
    ++actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP_SIZE \
    ++actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP_SIZE \
    ++actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP_SIZE \
    ++actor_rollout_ref.ref.megatron.context_parallel_size=$CP_SIZE \
    ++actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP_SIZE \
    ++actor_rollout_ref.actor.megatron.use_mbridge=True \
    ++critic.enable=False \
    ++actor_rollout_ref.actor.megatron.sequence_parallel=True \
    ++actor_rollout_ref.ref.megatron.sequence_parallel=True \
    "$@"


# actor_rollout_ref.actor.strategy=fsdp2 \
# actor_rollout_ref.ref.strategy=fsdp2 \
# critic.strategy=fsdp2 \