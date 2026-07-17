#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src/utils:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONFIG="${CONFIG:-configs/train_config.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_config.yaml}"
USE_ACCELERATE="${USE_ACCELERATE:-1}"

if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
  swanlab login -k "${SWANLAB_API_KEY}"
fi

if [[ "${USE_ACCELERATE}" != "1" ]]; then
  python train.py --config "${CONFIG}" "$@"
  exit 0
fi

if [[ -n "${PET_NNODES:-}" && -n "${PET_NPROC_PER_NODE:-}" ]]; then
  NUM_PROCESSES="${NUM_PROCESSES:-$((PET_NNODES * PET_NPROC_PER_NODE))}"
  accelerate launch \
    --config_file="${ACCELERATE_CONFIG}" \
    --main_process_ip="${PET_MASTER_ADDR}" \
    --main_process_port="${PET_MASTER_PORT}" \
    --num_machines="${PET_NNODES}" \
    --num_processes="${NUM_PROCESSES}" \
    --machine_rank="${PET_NODE_RANK}" \
    train.py --config "${CONFIG}" "$@"
else
  accelerate launch \
    --config_file="${ACCELERATE_CONFIG}" \
    train.py --config "${CONFIG}" "$@"
fi
