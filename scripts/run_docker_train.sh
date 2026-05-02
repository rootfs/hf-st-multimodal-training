#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_TIMEOUT=3600
export RCCL_TIMEOUT=3600
export HSA_ENABLE_SDMA=0

if [[ "${1:-}" == "build" ]]; then
  docker compose build
  exit 0
fi

if [[ "${1:-}" == "download" ]]; then
  docker compose run --rm trainer bash -lc "\
python /workspace/app/scripts/download_real_data.py \
  --output-root /scratch/2dmse-data/server \
  --num-text 256 \
  --num-image 256 \
  --num-audio 128\
"
  exit 0
fi

docker compose run --rm trainer bash -lc "\
python /workspace/app/scripts/download_real_data.py \
  --output-root /scratch/2dmse-data/server \
  --num-text 256 \
  --num-image 256 \
  --num-audio 128 && \
python scripts/validate_dataset.py \
  --manifest /scratch/2dmse-data/server/train_manifest.jsonl \
  --image-root /scratch/2dmse-data/server/images \
  --audio-root /scratch/2dmse-data/server/audio \
  --check-files && \
accelerate launch --config_file /workspace/app/configs/accelerate_4gpu.yaml \
  /workspace/app/scripts/train_st_multimodal.py \
  --config /workspace/app/configs/train_server_native.yaml\
"
