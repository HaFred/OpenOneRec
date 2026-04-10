#!/bin/bash
# GRPO Training Script — FSDP Backend with Profiling
# ---------------------------------------------------
# Profiling approach (matching this codebase's capabilities):
#
# 1. NSIGHT SYSTEMS (end-to-end):  nsys wraps the process; trainer.profile_steps
#    triggers torch.cuda.profiler.start/stop on profiled steps;
#    @DistProfiler.annotate NVTX markers label generate_sequences,
#    compute_log_prob, update_actor, etc.
#    → produces .nsys-rep files viewable in Nsight Systems or nsys stats
#
# NOTE: Unlike Megatron, the FSDP actor has no built-in PyTorch Profiler
# (actor.profile). Nsight Systems is the primary profiling tool here.
#
# Usage:
#   bash run_grpo_fsdp_profile.sh

set -e

# ============================================================================
# Cluster Configuration (auto-detect from Ray)
# ============================================================================
RAY_INFO=$(python -c "import ray; ray.init(address='auto', ignore_reinit_error=True); nodes = [n for n in ray.nodes() if n['Alive']]; gpus=next((int(n.get('Resources',{}).get('GPU',0)) for n in nodes if n.get('Resources',{}).get('GPU',0)>0), 0); print(f'{len(nodes)} {gpus}')" 2>/dev/null)

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
export BASE_MODEL=${BASE_MODEL:-"/path/to/your/model"}
export TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-False}
export EXTERNAL_LIB=${EXTERNAL_LIB:-null}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}
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
export WORLD_SIZE=$((N_GPUS * N_NODES))
export DP_SIZE=$WORLD_SIZE
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$DP_SIZE}
export PPO_MICRO_BATCH_PER_GPU=${PPO_MICRO_BATCH_PER_GPU:-1}

# ============================================================================
# Rollout Configuration
# ============================================================================
export ROLLOUT_N=${ROLLOUT_N:-1}
export STAGE2_BEAM_SIZE=${STAGE2_BEAM_SIZE:-32}
export RESPONSE_LENGTH=${RESPONSE_LENGTH:-2048}
export STAGE1_MAX_TOKENS=${STAGE1_MAX_TOKENS:-1024}
export STAGE2_NUM_TOKENS=${STAGE2_NUM_TOKENS:-3}

export ENABLE_THINK=${ENABLE_THINK:-False}
export ENABLE_NONTHINK=${ENABLE_NONTHINK:-False}
export USE_FORCE_PREFIX=${USE_FORCE_PREFIX:-False}

# ============================================================================
# Data Configuration
# ============================================================================
export DATA_DIR=${DATA_DIR:-"$(realpath ../output/rl_data)"}
export TRAIN_FILES=${TRAIN_FILES:-"[$DATA_DIR/train.parquet]"}
export VAL_FILES=${VAL_FILES:-"[$DATA_DIR/test.parquet]"}

# ============================================================================
# Output & Profiling Configuration
# ============================================================================
export PROJECT_NAME=${PROJECT_NAME:-"OneRec_RL"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_fsdp_profile"}
export OUTPUT_DIR=${OUTPUT_DIR:-"./output"}
export WANDB_MODE=${WANDB_MODE:-offline}
export PROFILE_DIR=${PROFILE_DIR:-"$OUTPUT_DIR/profile/fsdp"}
export PROFILE_STEP_START=${PROFILE_STEP_START:-2}
export PROFILE_STEP_END=${PROFILE_STEP_END:-4}

mkdir -p "$PROFILE_DIR"

# ============================================================================
# Network Configuration
# ============================================================================
export TCP_NIC=$(ifconfig 2>/dev/null | grep -B1 " "$(hostname -i 2>/dev/null)" " | grep -o "^\w*" || echo "eth0")
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}

# ============================================================================
# Print Configuration
# ============================================================================
echo "============================================="
echo "GRPO FSDP Profiling Run"
echo "============================================="
echo "Backend:  FSDP"
echo "Model:    $BASE_MODEL"
echo "Cluster:  $N_NODES nodes x $N_GPUS GPUs"
echo "DP Size:  $DP_SIZE  (all GPUs for data parallelism)"
echo "Rollout TP: $ROLLOUT_TP_SIZE"
echo "Profile steps:  $PROFILE_STEP_START-$PROFILE_STEP_END"
echo "Profile dir:    $PROFILE_DIR"
echo "Nsight Systems: REQUIRED (FSDP has no built-in PyTorch Profiler)"
echo "============================================="

# ============================================================================
# Build the training command
# ============================================================================
TRAIN_CMD=(
    python3 -u -m recipe.onerec.main_onerec_ppo
    --config-name ppo_trainer
    algorithm.adv_estimator=grpo
    data.train_files=$TRAIN_FILES
    data.val_files=$VAL_FILES
    data.max_prompt_length=10240
    ++data.enable_think=$ENABLE_THINK
    ++data.enable_nonthink=$ENABLE_NONTHINK
    ++data.use_force_prefix=$USE_FORCE_PREFIX
    data.prompt_key='prompt'
    data.shuffle=True
    data.max_response_length=$RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_BATCH_SIZE
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.custom_cls.path=$SCRIPT_DIR/onerec_recipe.py
    data.custom_cls.name=OneRecDataset
    data.reward_fn_key='source'
    ++data.data_source_key='source'
    ++actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    ++actor_rollout_ref.actor.entropy_checkpointing=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.actor.clip_ratio_high=0.28
    ++actor_rollout_ref.model.enable_activation_offload=True
    ++actor_rollout_ref.model.use_remove_padding=True
    ++actor_rollout_ref.model.trust_remote_code=$TRUST_REMOTE_CODE
    ++actor_rollout_ref.model.external_lib=$EXTERNAL_LIB
    custom_reward_function.path=$SCRIPT_DIR/onerec_recipe.py
    custom_reward_function.name=compute_score
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_PER_GPU
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_TOKENS_PER_GPU
    actor_rollout_ref.rollout.max_num_seqs=2048
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.model.path=$BASE_MODEL
    ++actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.rollout.n=$ROLLOUT_N
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE
    actor_rollout_ref.rollout.name=two_stage
    ++actor_rollout_ref.rollout.backend=vllm
    actor_rollout_ref.rollout.mode=sync
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    ++actor_rollout_ref.rollout.max_length=$RESPONSE_LENGTH
    ++actor_rollout_ref.rollout.stage1_max_tokens=$STAGE1_MAX_TOKENS
    ++actor_rollout_ref.rollout.stage2_num_tokens=$STAGE2_NUM_TOKENS
    ++actor_rollout_ref.rollout.stage2_beam_size=$STAGE2_BEAM_SIZE
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=320
    actor_rollout_ref.rollout.temperature=$TEMPERATURE
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.do_sample=True
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    algorithm.norm_adv_by_std_in_grpo=True
    algorithm.use_kl_in_reward=False
    trainer.default_hdfs_dir=null
    trainer.n_gpus_per_node=$N_GPUS
    trainer.nnodes=$N_NODES
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXPERIMENT_NAME
    trainer.default_local_dir=$OUTPUT_DIR/ckpt
    trainer.total_epochs=1
    trainer.val_before_train=False
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.ref.strategy=fsdp
    ++critic.enable=False
    # --- Nsight profiling: which steps to profile (DistProfiler start/stop) ---
    "trainer.profile_steps=[$PROFILE_STEP_START,$PROFILE_STEP_END]"
    # --- Worker-level DistProfiler config (NVTX annotations per function) ---
    actor_rollout_ref.profiler.discrete=True
    actor_rollout_ref.profiler.all_ranks=False
    "actor_rollout_ref.profiler.ranks=[0]"
    "$@"
)

# ============================================================================
# Launch with Nsight Systems
# ============================================================================
mkdir -p logs

NUM_PROFILED_STEPS=$(( PROFILE_STEP_END - PROFILE_STEP_START + 1 ))

if ! command -v nsys &> /dev/null; then
    echo ""
    echo "[profile] ERROR: nsys not found."
    echo "  FSDP backend has no built-in PyTorch Profiler (unlike Megatron)."
    echo "  Nsight Systems is required for profiling. Install it from:"
    echo "    https://developer.nvidia.com/nsight-systems"
    echo "  Or use 'apt install nsight-systems' / 'conda install -c nvidia nsight-systems'"
    echo ""
    echo "  Proceeding without profiling (timing logs will still be available)..."
    echo ""
    "${TRAIN_CMD[@]}"
else
    echo "[profile] Launching with Nsight Systems"
    nsys profile \
        --output "$PROFILE_DIR/fsdp_grpo" \
        --trace "cuda,nvtx,cublas,ucx" \
        --cuda-memory-usage=true \
        --cuda-graph-trace=graph \
        --capture-range=cudaProfilerApi \
        --capture-range-end="repeat-shutdown:${NUM_PROFILED_STEPS}" \
        --kill=none \
        "${TRAIN_CMD[@]}"
fi

echo ""
echo "============================================="
echo "FSDP profiling complete."
echo "Outputs:"
echo "  Nsight traces: $PROFILE_DIR/fsdp_grpo.nsys-rep"
echo ""
echo "View:  nsys stats $PROFILE_DIR/fsdp_grpo.nsys-rep"
echo "       or open in Nsight Systems GUI"
echo "============================================="
