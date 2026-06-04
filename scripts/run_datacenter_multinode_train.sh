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
TRAIN_CACHE_DIR_HOST="${TRAIN_CACHE_DIR_HOST:-/scratch/2dmse-data/server_full_datacenter_cache/train}"
VAL_CACHE_DIR_HOST="${VAL_CACHE_DIR_HOST:-/scratch/2dmse-data/server_full_datacenter_cache/val}"
WAIT_FOR_NODE0_READY_SECONDS="${WAIT_FOR_NODE0_READY_SECONDS:-600}"
NODE0_READY_MARKER="${NODE0_READY_MARKER:-${OUTPUT_DIR_HOST}/.multinode_node0_ready}"

export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_PROTO="${NCCL_PROTO:-Simple}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export RCCL_TIMEOUT="${RCCL_TIMEOUT:-3600}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"

if [[ "${1:-}" == "build" ]]; then
  docker compose build
  exit 0
fi

require_cache_dir() {
  local cache_dir="$1"
  local label="$2"
  if [[ ! -d "$cache_dir" ]]; then
    echo "[multinode-preflight] missing ${label} cache directory: ${cache_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${cache_dir}/metadata.json" ]]; then
    echo "[multinode-preflight] missing ${label} cache metadata: ${cache_dir}/metadata.json" >&2
    exit 1
  fi
  local first_shard
  first_shard="$(find "$cache_dir" -maxdepth 1 -type f -name 'shard_*.pt' | sort | head -1 || true)"
  if [[ -z "$first_shard" ]]; then
    echo "[multinode-preflight] no ${label} shard files found under ${cache_dir}" >&2
    exit 1
  fi
}

mkdir -p "$OUTPUT_DIR_HOST"
require_cache_dir "$TRAIN_CACHE_DIR_HOST" "train"
require_cache_dir "$VAL_CACHE_DIR_HOST" "validation"

if [[ "$MACHINE_RANK" == "0" ]]; then
  {
    echo "ready=1"
    echo "master_addr=${MASTER_ADDR}"
    echo "master_port=${MASTER_PORT}"
    echo "num_machines=${NUM_MACHINES}"
    echo "gpus_per_node=${GPUS_PER_NODE}"
    echo "train_cache_dir_host=${TRAIN_CACHE_DIR_HOST}"
    echo "val_cache_dir_host=${VAL_CACHE_DIR_HOST}"
  } > "$NODE0_READY_MARKER"
  echo "[multinode-preflight] wrote node0 readiness marker: ${NODE0_READY_MARKER}" >&2
else
  echo "[multinode-preflight] waiting for node0 readiness marker: ${NODE0_READY_MARKER}" >&2
  waited=0
  while [[ ! -f "$NODE0_READY_MARKER" ]]; do
    sleep 5
    waited=$((waited + 5))
    if (( waited >= WAIT_FOR_NODE0_READY_SECONDS )); then
      echo "[multinode-preflight] timed out waiting for node0 readiness marker after ${WAIT_FOR_NODE0_READY_SECONDS}s" >&2
      exit 1
    fi
  done
fi

docker compose run --rm trainer bash -lc "\
set -euo pipefail; \
export PYTHONUNBUFFERED=1; \
latest_ckpt=\$(find '${OUTPUT_DIR_HOST}' -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1 || true); \
if [ -n \"\$latest_ckpt\" ]; then resume_args=\"--resume \$latest_ckpt\"; else resume_args=\"\"; fi; \
accelerate launch \
  --config_file '${ACCEL_CONFIG}' \
  --num_machines ${NUM_MACHINES} \
  --num_processes ${TOTAL_PROCESSES} \
  --machine_rank ${MACHINE_RANK} \
  --main_process_ip '${MASTER_ADDR}' \
  --main_process_port ${MASTER_PORT} \
  --rdzv_backend c10d \
  /workspace/app/scripts/train_st_multimodal.py \
  --config '${TRAIN_CONFIG}' \
  \$resume_args\
"
