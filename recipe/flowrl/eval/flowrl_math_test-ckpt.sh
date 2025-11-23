#!/usr/bin/env bash
set -euo pipefail
export HYDRA_FULL_ERROR=1

CKPT_PATH="${1:-${CKPT_PATH:-}}"
: "${OUTPUT_ROOT:=/mnt/shared-storage-user/formalverification-shared/chenlin1/verl/output/FlowRL}"
declare -A MATH_DATASETS=(
  [math_test]="/mnt/shared-storage-user/chenlin1/FlowRL/data/math_data/test.parquet"
)

N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
mkdir -p "${OUTPUT_ROOT}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "Usage: CKPT_PATH=/path/to/ckpt ./merge_model.sh  (or pass as first arg)" >&2
  exit 1
fi
if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "Checkpoint ${CKPT_PATH} does not exist." >&2
  exit 1
fi

MODEL_NAME=$(basename "$(dirname "${CKPT_PATH}")")
CKPT_NAME=$(basename "${CKPT_PATH}")

for DATASET_NAME in "${!MATH_DATASETS[@]}"; do
  DATA_PATH="${MATH_DATASETS[$DATASET_NAME]}"
  [[ -f "${DATA_PATH}" ]] || { echo "Missing ${DATA_PATH}, skipping."; continue; }

  RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${CKPT_NAME}/${DATASET_NAME}"
  mkdir -p "${RUN_OUTPUT_DIR}"
  OUTPUT_PATH="${RUN_OUTPUT_DIR}/${DATASET_NAME}-output-16.parquet"

  echo "==== Generation: model=${MODEL_NAME}, ckpt=${CKPT_NAME}, dataset=${DATASET_NAME} ===="
  python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    data.path="${DATA_PATH}" \
    data.prompt_key=prompt \
    data.batch_size=1024 \
    data.n_samples=16 \
    data.output_path="${OUTPUT_PATH}" \
    model.path="${CKPT_PATH}" \
    rollout.temperature=0.6 \
    rollout.top_p=0.95 \
    rollout.prompt_length=2048 \
    rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8 \
    rollout.max_num_batched_tokens=65536 \
    +rollout.data_parallel_size=1

  echo "==== Evaluation: model=${MODEL_NAME}, ckpt=${CKPT_NAME}, dataset=${DATASET_NAME} ===="
  python3 -m recipe.r1.main_eval \
    data.path="${OUTPUT_PATH}" \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=recipe/r1/reward_score.py \
    custom_reward_function.name=reward_func

  echo "Results saved under ${RUN_OUTPUT_DIR}"
done

echo "Done. Outputs grouped under ${OUTPUT_ROOT}."
