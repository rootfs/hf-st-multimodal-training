#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

REPORT_PATH="${1:-${ROOT_DIR}/workflows/operational_flow_localization_report.txt}"
mkdir -p "$(dirname "$REPORT_PATH")"

{
  echo "== Operational Flow Localization Report =="
  echo "repo_root=${ROOT_DIR}"
  echo "generated_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo

  echo "== Stage: download/cache/preprocess/validate/train/multinode/checkpoint surfaces =="
  rg -n --glob '*.py' --glob '*.sh' --glob '*.yaml' --glob '*.yml' --glob '*.md' \
    "download_real_data|preprocess_manifest_cache|validate_dataset|train_st_multimodal|run_datacenter_multinode_train|run_server_datacenter_pipeline|cache_dir|checkpoint|output_dir|resume|MACHINE_RANK|num_machines|WORLD_SIZE|RANK" \
    scripts configs src README.md MULTI_NODE_3x8GPU_PLAN.md || true
  echo

  echo "== Stage: major launch scripts =="
  ls -1 scripts/*.sh | sed 's#^#- #'
  echo

  echo "== Stage: training configs =="
  ls -1 configs/*.yaml | sed 's#^#- #'
  echo

  echo "== Stage: shell syntax checks =="
  failed=0
  while IFS= read -r script_path; do
    if bash -n "$script_path"; then
      echo "ok  ${script_path}"
    else
      echo "bad ${script_path}"
      failed=1
    fi
  done < <(find scripts workflows -maxdepth 1 -type f -name '*.sh' | sort)
  echo
  if [[ "$failed" -ne 0 ]]; then
    echo "status=failed"
    exit 1
  fi
  echo "status=ok"
} | tee "$REPORT_PATH"

