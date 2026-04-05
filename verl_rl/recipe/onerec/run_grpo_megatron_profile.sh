#!/bin/bash
# GRPO Training Script — Megatron Backend with Profiling
# -------------------------------------------------------
# Profiling approach (matching this codebase's capabilities):
#
# 1. NSIGHT SYSTEMS (end-to-end):  nsys wraps the process; trainer.profile_steps
#    triggers torch.cuda.profiler.start/stop on profiled steps;
#    @DistProfiler.annotate NVTX markers label generate_sequences,
#    compute_log_prob, update_actor, etc.
#    → produces .nsys-rep files viewable in Nsight Systems or nsys stats
#
# 2. PYTORCH PROFILER (Megatron update_policy only):  actor.profile.use_profile
#    wraps update_policy with torch.profiler.profile, exporting Chrome traces.
#    → produces .json files viewable in chrome://tracing or Perfetto
#
# Usage:
#   bash run_grpo_megatron_profile.sh            # both Nsight + PyTorch Profiler
#   SKIP_NSYS=1 bash run_grpo_megatron_profile.sh  # PyTorch Profiler only
# 
# 


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
export EXPECTED_MODEL_ARCH=${EXPECTED_MODEL_ARCH:-"Qwen3ForCausalLM"}
export TP_SIZE=${TP_SIZE:-2}
export PP_SIZE=${PP_SIZE:-2}
export CP_SIZE=${CP_SIZE:-1}
export EP_SIZE=${EP_SIZE:-1}
export MEGATRON_USE_MBRIDGE=${MEGATRON_USE_MBRIDGE:-True}
export TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-False}
export EXTERNAL_LIB=${EXTERNAL_LIB:-null}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-$TP_SIZE}
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
export MODEL_PARALLEL_SIZE=$((TP_SIZE * PP_SIZE * CP_SIZE * EP_SIZE))
if [ "$MODEL_PARALLEL_SIZE" -le 0 ]; then
    echo "Invalid MODEL_PARALLEL_SIZE=$MODEL_PARALLEL_SIZE. Check TP/PP/CP/EP settings."
    exit 1
fi
if [ $((WORLD_SIZE % MODEL_PARALLEL_SIZE)) -ne 0 ]; then
    echo "Invalid 5D topology: WORLD_SIZE=$WORLD_SIZE is not divisible by TP*PP*CP*EP=$MODEL_PARALLEL_SIZE"
    exit 1
fi
export DP_SIZE=$((WORLD_SIZE / MODEL_PARALLEL_SIZE))
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
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_megatron_profile"}
export OUTPUT_DIR=${OUTPUT_DIR:-"./output"}
export WANDB_MODE=${WANDB_MODE:-offline}
export PROFILE_DIR=${PROFILE_DIR:-"$OUTPUT_DIR/profile/megatron"}
export PROFILE_STEP_START=${PROFILE_STEP_START:-2}
export PROFILE_STEP_END=${PROFILE_STEP_END:-4}
export SKIP_NSYS=${SKIP_NSYS:-0}

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
echo "GRPO Megatron Profiling Run"
echo "============================================="
echo "Backend:  MEGATRON"
echo "Model:    $BASE_MODEL"
echo "Cluster:  $N_NODES nodes x $N_GPUS GPUs"
echo "5D Para:  TP=$TP_SIZE PP=$PP_SIZE CP=$CP_SIZE EP=$EP_SIZE DP=$DP_SIZE"
echo "Profile steps:  $PROFILE_STEP_START-$PROFILE_STEP_END"
echo "Profile dir:    $PROFILE_DIR"
echo "Nsight Systems: $([ "$SKIP_NSYS" = "1" ] && echo "DISABLED" || echo "ENABLED")"
echo "PyTorch Prof:   ENABLED (update_policy, rank 0)"
echo "============================================="

# ============================================================================
# Pre-flight: validate model architecture
# ============================================================================
python3 -u -m recipe.onerec.megatron_mcore_support \
    --model-path "$BASE_MODEL" \
    --expected-arch "$EXPECTED_MODEL_ARCH"

# ============================================================================
# Build the training command
# ============================================================================
TRAIN_CMD=(
    python3 -u -m recipe.onerec.main_onerec_ppo
    --config-name ppo_megatron_trainer
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
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.ref.strategy=megatron
    critic.strategy=megatron
    reward_model.strategy=megatron
    ++actor_rollout_ref.model.trust_remote_code=$TRUST_REMOTE_CODE
    ++actor_rollout_ref.model.external_lib=$EXTERNAL_LIB
    ++actor_rollout_ref.actor.megatron.use_mbridge=$MEGATRON_USE_MBRIDGE
    ++actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE
    ++actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE
    ++actor_rollout_ref.actor.megatron.context_parallel_size=$CP_SIZE
    ++actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP_SIZE
    ++actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP_SIZE
    ++actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP_SIZE
    ++actor_rollout_ref.ref.megatron.context_parallel_size=$CP_SIZE
    ++actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP_SIZE
    ++actor_rollout_ref.ref.megatron.use_mbridge=$MEGATRON_USE_MBRIDGE
    ++critic.enable=False
    ++actor_rollout_ref.actor.megatron.sequence_parallel=True
    ++actor_rollout_ref.ref.megatron.sequence_parallel=True
    # --- Nsight profiling: which steps to profile (DistProfiler start/stop) ---
    "trainer.profile_steps=[$PROFILE_STEP_START,$PROFILE_STEP_END]"
    # --- Worker-level DistProfiler config (NVTX annotations per function) ---
    actor_rollout_ref.profiler.discrete=True
    actor_rollout_ref.profiler.all_ranks=False
    "actor_rollout_ref.profiler.ranks=[0]"
    # --- Megatron actor PyTorch Profiler (update_policy Chrome traces) ---
    ++actor_rollout_ref.actor.profile.use_profile=True
    "++actor_rollout_ref.actor.profile.profile_ranks=[0]"
    ++actor_rollout_ref.actor.profile.step_start=$PROFILE_STEP_START
    ++actor_rollout_ref.actor.profile.step_end=$PROFILE_STEP_END
    ++actor_rollout_ref.actor.profile.save_path=$PROFILE_DIR
    "$@"
)

# ============================================================================
# Launch: with or without Nsight Systems
# ============================================================================
mkdir -p logs

NUM_PROFILED_STEPS=$(( PROFILE_STEP_END - PROFILE_STEP_START + 1 ))

if [ "$SKIP_NSYS" = "1" ]; then
    echo "[profile] Running WITHOUT Nsight Systems (PyTorch Profiler only)"
    echo "[profile] Only update_policy will have traces (no generate/logprob traces)"
    "${TRAIN_CMD[@]}"
else
    if ! command -v nsys &> /dev/null; then
        echo "[profile] WARNING: nsys not found. Install NVIDIA Nsight Systems or set SKIP_NSYS=1."
        echo "[profile] Falling back to PyTorch Profiler only."
        "${TRAIN_CMD[@]}"
    else
        echo "[profile] Launching with Nsight Systems"
        nsys profile \
            --output "$PROFILE_DIR/megatron_grpo" \
            --trace "cuda,nvtx,cublas,ucx" \
            --cuda-memory-usage=true \
            --cuda-graph-trace=graph \
            --capture-range=cudaProfilerApi \
            --capture-range-end="repeat-shutdown:${NUM_PROFILED_STEPS}" \
            --kill=none \
            "${TRAIN_CMD[@]}"
    fi
fi

echo ""
echo "============================================="
echo "Megatron profiling complete."
echo "Outputs:"
echo "  Nsight traces: $PROFILE_DIR/megatron_grpo.nsys-rep  (if nsys was used)"
echo "  PyTorch traces: $PROFILE_DIR/prof_start_*_rank_*.json"
echo ""
echo "View Nsight:  nsys stats $PROFILE_DIR/megatron_grpo.nsys-rep"
echo "              or open in Nsight Systems GUI"
echo "View PyTorch: open .json in chrome://tracing or https://ui.perfetto.dev/"
echo "============================================="
