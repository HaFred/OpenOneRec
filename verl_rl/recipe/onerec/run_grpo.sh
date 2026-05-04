#!/bin/bash
# GRPO Training Script with Two-Stage Rollout
# Two-Stage Rollout: first generate to </think>, then insert <sid_begin> and beam search

clear
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export HYDRA_FULL_ERROR=1
ROLLOUT_N=${ROLLOUT_N:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-12288}
BASE_MODEL=${BASE_MODEL:-/scratch/dyvm6xra/dyvm6xrauser45/fred/models--OpenOneRec--OneRec-1.7B-pretrain/snapshots/db455d0bdcf4b5e0b42f30c45d65260a49656a7f}
export CUDA_VISIBLE_DEVICES=0,1,2,3
HYDRA_FULL_ERROR=1
ROLLOUT_N=1
MAX_TOKENS_PER_GPU=20480
BASE_MODEL=/scratch/dyvm6xra/dyvm6xrauser45/fred/models--OpenOneRec--OneRec-1.7B-pretrain/snapshots/db455d0bdcf4b5e0b42f30c45d65260a49656a7f
DATA_DIR=/home/dyvm6xra/dyvm6xrauser45/fred/openonerec_fredfork/data
# DATA_DIR=/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/v-gr-ms/verl_gr/recipes/openonerec/output/rl_data
# OUTPUT_DIR="output/ckpt_best3_selection_on_valpassat32"
OUTPUT_DIR="output/nothinking"

export TENSORBOARD_RUN_ID=${TENSORBOARD_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
export TENSORBOARD_DIR=${OUTPUT_DIR}/${TENSORBOARD_RUN_ID}

BEST_CKPTS_TO_KEEP=${BEST_CKPTS_TO_KEEP:-3}
BEST_CKPT_PRUNE_ENABLE=${BEST_CKPT_PRUNE_ENABLE:-True}
RAY_TMPDIR=${RAY_TMPDIR:-$OUTPUT_DIR/ray_tmp}
RAY_SPILL_DIR=${RAY_SPILL_DIR:-$RAY_TMPDIR/spill}

# ============================================================================
# Cluster Configuration (auto-detect from Ray)
# ============================================================================
# Ray auto-discovery can fail on single-node/local runs; do not hard-exit under `set -e`.
RAY_INFO=$(python -c "import ray; ray.init(address='auto', ignore_reinit_error=True); nodes = [n for n in ray.nodes() if n['Alive']]; gpus=next((int(n.get('Resources',{}).get('GPU',0)) for n in nodes if n.get('Resources',{}).get('GPU',0)>0), 0); print(f'{len(nodes)} {gpus}')" 2>/dev/null || true)

export N_NODES=$(echo $RAY_INFO | awk '{print $1}')
# export N_GPUS=$(echo $RAY_INFO | awk '{print $2}')  
N_NODES=1
N_GPUS=4

# if [ -z "$N_NODES" ] || [ -z "$N_GPUS" ] || [ "$N_NODES" -eq 0 ]; then
#     echo "Could not detect Ray cluster. Using defaults: N_NODES=1, N_GPUS=8"
#     export N_NODES=1
#     export N_GPUS=2
# else
#     echo "Detected Ray cluster: $N_NODES nodes, $N_GPUS GPUs per node"
# fi

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============================================================================
# Model Configuration
# ============================================================================
export BASE_MODEL=${BASE_MODEL:-"/path/to/your/model"}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
export VLLM_ATTENTION_BACKEND=XFORMERS
# export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"

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
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-12288}
export TRAIN_BATCH_SIZE=$((N_GPUS * N_NODES))

# ============================================================================
# Rollout Configuration
# ============================================================================
export ROLLOUT_N=${ROLLOUT_N:-1}
export STAGE2_BEAM_SIZE=${STAGE2_BEAM_SIZE:-32}
export RESPONSE_LENGTH=${RESPONSE_LENGTH:-2048}
export STAGE1_MAX_TOKENS=${STAGE1_MAX_TOKENS:-1024}
export STAGE2_NUM_TOKENS=${STAGE2_NUM_TOKENS:-3}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-512}

# Think mode configuration
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
# Output Configuration
# ============================================================================
export PROJECT_NAME=${PROJECT_NAME:-"OneRec_RL"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"4gpu_fullymatchedarg"}
export OUTPUT_DIR=${OUTPUT_DIR:-"/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/verl-gr-fork-workingbranch/outputs/${EXPERIMENT_NAME}"}
export WANDB_MODE=${WANDB_MODE:-offline}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
export VAL_LOG_GENERATIONS=${VAL_LOG_GENERATIONS:-4}
export VAL_DUMP_GENERATIONS=${VAL_DUMP_GENERATIONS:-True}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-100}
export SAVE_FREQ=${SAVE_FREQ:-100}
export TEST_FREQ=${TEST_FREQ:-100}
export BEST_CKPT_METRIC=${BEST_CKPT_METRIC:-"val-aux/*/pass_at_32/mean"}
export VAL_THINKING_TEMPERATURE=${VAL_THINKING_TEMPERATURE:-0.6}
export VAL_THINKING_TOP_P=${VAL_THINKING_TOP_P:-0.95}
export VAL_THINKING_TOP_K=${VAL_THINKING_TOP_K:-50}
export VALIDATION_ADAPTIVE_CONCURRENCY=${VALIDATION_ADAPTIVE_CONCURRENCY:-True}
export VALIDATION_MIN_CONCURRENT_REQUESTS=${VALIDATION_MIN_CONCURRENT_REQUESTS:-32}
export VALIDATION_MAX_CONCURRENT_REQUESTS=${VALIDATION_MAX_CONCURRENT_REQUESTS:-64}
export VALIDATION_TARGET_GPU_UTILIZATION=${VALIDATION_TARGET_GPU_UTILIZATION:-85.0}
export VALIDATION_GPU_UTIL_TOLERANCE=${VALIDATION_GPU_UTIL_TOLERANCE:-7.5}
export VALIDATION_CONCURRENCY_STEP=${VALIDATION_CONCURRENCY_STEP:-32}
export ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
export ROLLOUT_MODE=${ROLLOUT_MODE:-async}
export AGENT_LOOP_NUM_WORKERS=${AGENT_LOOP_NUM_WORKERS:-${N_GPUS:-1}}

if [[ "$VAL_DUMP_GENERATIONS" == "True" ]]; then
    VAL_DATA_DIR=${VAL_DATA_DIR:-$OUTPUT_DIR/val_generations}
    mkdir -p "$VAL_DATA_DIR"
    VALIDATION_DATA_DIR_ARG=$VAL_DATA_DIR
else
    VALIDATION_DATA_DIR_ARG=null
fi

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
echo "TensorBoard Log: $TENSORBOARD_DIR"
echo "Ray temp dir: $RAY_TMPDIR"
echo "Ray spill dir: $RAY_SPILL_DIR"
echo "==================================="

# ============================================================================
# Launch Training
# ============================================================================
mkdir -p logs "$TENSORBOARD_DIR" "$RAY_TMPDIR" "$RAY_SPILL_DIR"

python3 -u -m recipe.onerec.main_onerec_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.max_prompt_length=10240 \
    data.train_max_samples=20000 \
    data.val_max_samples=$VAL_MAX_SAMPLES \
    data.val_batch_size=$VAL_BATCH_SIZE \
    ++data.filter_overlong_prompts_workers=16 \
    ++data.enable_think=$ENABLE_THINK \
    ++data.enable_nonthink=$ENABLE_NONTHINK \
    ++data.use_force_prefix=$USE_FORCE_PREFIX \
    data.prompt_key='prompt' \
    data.shuffle=False \
    data.max_response_length=$RESPONSE_LENGTH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=$SCRIPT_DIR/onerec_recipe.py \
    data.custom_cls.name=OneRecDataset \
    data.reward_fn_key='source' \
    ++data.data_source_key='source' \
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    custom_reward_function.path=$SCRIPT_DIR/onerec_recipe.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
    actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER \
    actor_rollout_ref.rollout.custom.validation_adaptive_concurrency=$VALIDATION_ADAPTIVE_CONCURRENCY \
    actor_rollout_ref.rollout.custom.validation_min_concurrent_requests=$VALIDATION_MIN_CONCURRENT_REQUESTS \
    actor_rollout_ref.rollout.custom.validation_max_concurrent_requests=$VALIDATION_MAX_CONCURRENT_REQUESTS \
    actor_rollout_ref.rollout.custom.validation_target_gpu_utilization=$VALIDATION_TARGET_GPU_UTILIZATION \
    actor_rollout_ref.rollout.custom.validation_gpu_util_tolerance=$VALIDATION_GPU_UTIL_TOLERANCE \
    actor_rollout_ref.rollout.custom.validation_concurrency_step=$VALIDATION_CONCURRENCY_STEP \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_LOOP_NUM_WORKERS \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=two_stage \
    ++actor_rollout_ref.rollout.mode=$ROLLOUT_MODE \
    ++actor_rollout_ref.rollout.backend=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    ++actor_rollout_ref.rollout.max_length=$RESPONSE_LENGTH \
    ++actor_rollout_ref.rollout.stage1_max_tokens=$STAGE1_MAX_TOKENS \
    ++actor_rollout_ref.rollout.stage2_num_tokens=$STAGE2_NUM_TOKENS \
    ++actor_rollout_ref.rollout.stage2_beam_size=$STAGE2_BEAM_SIZE \
    actor_rollout_ref.rollout.custom.beam_width=$STAGE2_BEAM_SIZE \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=320 \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_THINKING_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_THINKING_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.top_k=$VAL_THINKING_TOP_K \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    ++trainer.logger='[tensorboard, wandb]' \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.log_val_generations=$VAL_LOG_GENERATIONS \
    trainer.validation_data_dir=$VALIDATION_DATA_DIR_ARG \
    ++trainer.best_ckpt_prune_enable=$BEST_CKPT_PRUNE_ENABLE \
    ++trainer.best_ckpts_to_keep=$BEST_CKPTS_TO_KEEP \
    ++trainer.best_ckpt_metric="$BEST_CKPT_METRIC" \
    trainer.resume_mode=auto \
    trainer.remove_previous_ckpt_in_save=False \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR/ckpt \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.actor.strategy=fsdp \
    +ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR \
    +ray_kwargs.ray_init.object_spilling_directory=$RAY_SPILL_DIR \
    global_profiler.save_path=${GLOBAL_PROFILER_SAVE_PATH:-$OUTPUT_DIR/profiles} \
    ++critic.enable=False \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    "$@"


# ++trainer.remove_previous_ckpt_in_save=True \