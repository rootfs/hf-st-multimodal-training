#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to node0 private IP or hostname}"
MASTER_PORT="${MASTER_PORT:-29500}"
MACHINE_RANK="${MACHINE_RANK:?set MACHINE_RANK to this node rank (0..N-1)}"
NUM_MACHINES="${NUM_MACHINES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

TOTAL_PROCESSES="$((NUM_MACHINES * GPUS_PER_NODE))"

TRAIN_CONFIG="${TRAIN_CONFIG:-/workspace/app/configs/train_server_datacenter_8gpu_cached.yaml}"
ACCEL_CONFIG="${ACCEL_CONFIG:-/workspace/app/configs/accelerate_2node_16gpu.yaml}"
OUTPUT_DIR_HOST="${OUTPUT_DIR_HOST:-/scratch/hf_st_mm_outputs/server_datacenter_8gpu_tri_encoder}"

export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_PROTO="${NCCL_PROTO:-Simple}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export RCCL_TIMEOUT="${RCCL_TIMEOUT:-3600}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"

if [[ "${1:-}" == "build" ]]; then
  docker compose build
  exit 0
fi

docker compose run --rm trainer bash -lc "\
set -euo pipefail; \
export PYTHONUNBUFFERED=1; \
latest_ckpt=\$(find ${OUTPUT_DIR_HOST} -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1 || true); \
if [ -n \"\$latest_ckpt\" ]; then resume_args=\"--resume \$latest_ckpt\"; else resume_args=\"\"; fi; \
accelerate launch \
  --config_file ${ACCEL_CONFIG} \
  --num_machines ${NUM_MACHINES} \
  --num_processes ${TOTAL_PROCESSES} \
  --machine_rank ${MACHINE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --rdzv_backend c10d \
  /workspace/app/scripts/train_st_multimodal.py \
  --config ${TRAIN_CONFIG} \
  \$resume_args\
"
