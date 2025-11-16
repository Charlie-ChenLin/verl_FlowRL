#!/usr/bin/env bash
set -xeuo pipefail

export HYDRA_FULL_ERROR=1

project_name='FlowRL'

# AlphaGFN setting to match the training run
alphagfn_alpha=0.9

# Root directory where FlowRL checkpoints are saved (same as in training script)
CKPT_ROOT="/mnt/shared-storage-user/formalverification-shared/chenlin1/verl/ckpts/${project_name}"

# Find the latest experiment directory for this AlphaGFN setting
EXP_PREFIX="FlowRL-cispo-clip-AlphaGFN-${alphagfn_alpha}-Qwen2.5-7B-"
EXP_DIR=$(ls -dt "${CKPT_ROOT}/${EXP_PREFIX}"*/ 2>/dev/null | head -n 1 || true)

if [[ -z "${EXP_DIR}" ]]; then
  echo "No experiment directory found under ${CKPT_ROOT} with prefix ${EXP_PREFIX}" >&2
  exit 1
fi

echo "Using experiment directory: ${EXP_DIR}"

# Find the latest global_step checkpoint inside this experiment
GLOBAL_STEP_DIR=$(ls -dt "${EXP_DIR}"/global_step_*/ 2>/dev/null | head -n 1 || true)

if [[ -z "${GLOBAL_STEP_DIR}" ]]; then
  echo "No global_step_* directories found under ${EXP_DIR}" >&2
  exit 1
fi

echo "Using checkpoint: ${GLOBAL_STEP_DIR}"

ACTOR_CKPT_DIR="${GLOBAL_STEP_DIR}/actor"

# Directories for merged HuggingFace model
HF_WITH_PROJ_Z_DIR="${GLOBAL_STEP_DIR}/hf_with_proj_z"
HF_CLEAN_DIR="${GLOBAL_STEP_DIR}/hf_clean"

###########################
# Step 1: Merge FSDP ckpt #
###########################

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_CKPT_DIR}" \
  --target_dir "${HF_WITH_PROJ_Z_DIR}"

########################################
# Step 2: Remove FlowRL proj_z head    #
########################################

python recipe/flowrl/eval/remove_proj_z.py \
  "${HF_WITH_PROJ_Z_DIR}" \
  "${HF_CLEAN_DIR}"

########################################
# Step 3: Generation on math test set  #
########################################

# Math evaluation datasets (AIME24 + CNMO en/zh). Feel free to add more.
declare -A MATH_DATASETS=(
  [aime24]="/mnt/shared-storage-user/chenlin1/verl/downloads/data/aime-2024.parquet"
  [cnmo24_en]="/mnt/shared-storage-user/chenlin1/verl/downloads/data/cnmo2024_en.parquet"
  [cnmo24_zh]="/mnt/shared-storage-user/chenlin1/verl/downloads/data/cnmo2024_zh.parquet"
)

N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

for DATASET_NAME in "${!MATH_DATASETS[@]}"; do
  DATA_PATH="${MATH_DATASETS[$DATASET_NAME]}"
  OUTPUT_DIR="${GLOBAL_STEP_DIR}/math_eval_${DATASET_NAME}"
  mkdir -p "${OUTPUT_DIR}"
  OUTPUT_PATH="${OUTPUT_DIR}/${DATASET_NAME}-output-16.parquet"

  echo "==== Generation: ${DATASET_NAME} (${DATA_PATH}) ===="
  python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    data.path="${DATA_PATH}" \
    data.prompt_key=prompt \
    data.batch_size=1024 \
    data.n_samples=16 \
    data.output_path="${OUTPUT_PATH}" \
    model.path="${HF_CLEAN_DIR}" \
    rollout.temperature=0.6 \
    rollout.top_p=0.95 \
    rollout.prompt_length=2048 \
    rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8 \
    rollout.max_num_batched_tokens=65536

  echo "==== Evaluation: ${DATASET_NAME} ===="
  python3 -m recipe.r1.main_eval \
    data.path="${OUTPUT_PATH}" \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=recipe/r1/reward_score.py \
    custom_reward_function.name=reward_func

  echo "Results for ${DATASET_NAME} saved to: ${OUTPUT_PATH}"
done

echo "Done. All math evals stored under ${GLOBAL_STEP_DIR}."
