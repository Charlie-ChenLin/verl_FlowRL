#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1

: "${CKPT_ROOT:=/mnt/shared-storage-user/formalverification-shared/chenlin1/verl/ckpts/FlowRL}"
: "${OUTPUT_ROOT:=/mnt/shared-storage-user/formalverification-shared/chenlin1/verl/output/FlowRL}"

declare -A MATH_DATASETS=(
  [math_test]="/mnt/shared-storage-user/chenlin1/FlowRL/data/math_data/test.parquet"
)

N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
mkdir -p "${OUTPUT_ROOT}"

if [[ ! -d "${CKPT_ROOT}" ]]; then
  echo "Checkpoint root ${CKPT_ROOT} does not exist." >&2
  exit 1
fi

mapfile -t MODEL_DIRS < <(find "${CKPT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print | LC_ALL=C sort)
for MODEL_DIR in "${MODEL_DIRS[@]}"; do
  [[ -d "${MODEL_DIR}" ]] || continue
  MODEL_NAME=$(basename "${MODEL_DIR}")

  mapfile -t CKPT_DIRS < <(find "${MODEL_DIR}" -mindepth 1 -maxdepth 1 -type d -name '*_merged' -print | LC_ALL=C sort)
  for CKPT_DIR in "${CKPT_DIRS[@]}"; do
    [[ -d "${CKPT_DIR}" ]] || continue
    CKPT_NAME=$(basename "${CKPT_DIR}")

    for DATASET_NAME in "${!MATH_DATASETS[@]}"; do
      DATA_PATH="${MATH_DATASETS[$DATASET_NAME]}"
      if [[ ! -f "${DATA_PATH}" ]]; then
        echo "Dataset ${DATASET_NAME} not found at ${DATA_PATH}, skipping." >&2
        continue
      fi

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
        model.path="${CKPT_DIR}" \
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
  done
done

echo "Done. Outputs grouped under ${OUTPUT_ROOT}."
