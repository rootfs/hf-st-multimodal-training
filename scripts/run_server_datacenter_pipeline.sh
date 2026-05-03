#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/2dmse-data/server_full}"
TRAIN_CONFIG="${TRAIN_CONFIG:-/workspace/app/configs/train_server_datacenter_8gpu_cached.yaml}"
ACCEL_CONFIG="${ACCEL_CONFIG:-/workspace/app/configs/accelerate_8gpu.yaml}"
LLAVA_ROOT="${LLAVA_ROOT:-/scratch/2dmse-data/server/llava-cc3m-595k}"
OUTPUT_DIR_HOST="${OUTPUT_DIR_HOST:-/scratch/hf_st_mm_outputs/server_datacenter_8gpu_tri_encoder}"

export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_TIMEOUT=3600
export RCCL_TIMEOUT=3600
export HSA_ENABLE_SDMA=0

if [[ "${1:-}" == "build" ]]; then
  docker compose build
  exit 0
fi

REQUIRED_AUDIO=(librispeech_clean librispeech_other voxpopuli)
OPTIONAL_AUDIO=(tedlium peoples_speech common_voice gigaspeech)
RUN_OPTIONAL_AUDIO="${RUN_OPTIONAL_AUDIO:-0}"

required_csv="$(IFS=,; echo "${REQUIRED_AUDIO[*]}")"
optional_csv="$(IFS=,; echo "${OPTIONAL_AUDIO[*]}")"
final_audio_csv="${required_csv}"
if [[ "${RUN_OPTIONAL_AUDIO}" == "1" ]]; then
  final_audio_csv="${required_csv},${optional_csv}"
fi

docker compose run --rm trainer bash -lc "\
set -euo pipefail; \
export PYTHONUNBUFFERED=1; \
python /workspace/app/scripts/download_real_data.py \
  --output-root ${OUTPUT_ROOT} \
  --num-text 0 \
  --num-image 0 \
  --num-audio 0 \
  --llava-root ${LLAVA_ROOT} \
  --skip-audio \
  --skip-finalize\
"

required_pids=()
for dataset_name in "${REQUIRED_AUDIO[@]}"; do
  docker compose run --rm trainer bash -lc "\
set -euo pipefail; \
export PYTHONUNBUFFERED=1; \
python /workspace/app/scripts/download_real_data.py \
  --output-root ${OUTPUT_ROOT} \
  --num-audio 0 \
  --skip-text \
  --skip-image \
  --skip-finalize \
  --audio-datasets ${dataset_name}\
" &
  required_pids+=("$!")
done

optional_pids=()
if [[ "${RUN_OPTIONAL_AUDIO}" == "1" ]]; then
  for dataset_name in "${OPTIONAL_AUDIO[@]}"; do
    (
      docker compose run --rm trainer bash -lc "\
set -euo pipefail; \
export PYTHONUNBUFFERED=1; \
python /workspace/app/scripts/download_real_data.py \
  --output-root ${OUTPUT_ROOT} \
  --num-audio 0 \
  --skip-text \
  --skip-image \
  --skip-finalize \
  --audio-datasets ${dataset_name} \
  --optional-audio-datasets ${dataset_name}\
" || true
    ) &
    optional_pids+=("$!")
  done
else
  echo "Skipping optional audio datasets by default: ${optional_csv}" >&2
  echo "Set RUN_OPTIONAL_AUDIO=1 to include them in the blocking datacenter pipeline." >&2
fi

for pid in "${required_pids[@]}"; do
  wait "$pid"
done

for pid in "${optional_pids[@]}"; do
  wait "$pid"
done

docker compose run --rm trainer bash -lc "\
set -euo pipefail; \
export PYTHONUNBUFFERED=1; \
python /workspace/app/scripts/download_real_data.py \
  --output-root ${OUTPUT_ROOT} \
  --num-text 0 \
  --num-image 0 \
  --num-audio 0 \
  --llava-root ${LLAVA_ROOT} \
  --audio-datasets ${final_audio_csv} \
  --optional-audio-datasets ${optional_csv} && \
python /workspace/app/scripts/validate_dataset.py \
  --manifest ${OUTPUT_ROOT}/train_manifest.jsonl \
  --check-files && \
python /workspace/app/scripts/validate_dataset.py \
  --manifest ${OUTPUT_ROOT}/val_manifest.jsonl \
  --check-files && \
latest_ckpt=\$(find ${OUTPUT_DIR_HOST} -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1 || true); \
if [ -n \"\$latest_ckpt\" ]; then resume_args=\"--resume \$latest_ckpt\"; else resume_args=\"\"; fi; \
accelerate launch --config_file ${ACCEL_CONFIG} \
  /workspace/app/scripts/train_st_multimodal.py \
  --config ${TRAIN_CONFIG} \
  \$resume_args\
"