# Repository Operational Flow Map

This map covers non-AST execution flow: data bootstrap, cache/preprocess, validation, launch, multi-node behavior, and output handoffs.

| Stage | Major scripts/configs | Input handoff | Output handoff |
| --- | --- | --- | --- |
| Data download/bootstrap | `scripts/download_real_data.py`, `scripts/run_server_datacenter_pipeline.sh` | Dataset roots (`--output-root`, `LLAVA_ROOT`), selected audio sets | `train_manifest.jsonl`, `val_manifest.jsonl`, media roots, `download_summary.json` |
| Manifest/media validation | `scripts/validate_dataset.py`, `scripts/run_server_datacenter_pipeline.sh` | Manifest JSONL + media roots | Validation pass/fail; blocks train launch on missing files |
| Cache preprocess (tri-encoder path) | `scripts/preprocess_manifest_cache.py` | Manifest + encoder names + media roots | Cache dirs with `metadata.json` and `shard_*.pt` |
| Single-node train launch | `scripts/run_docker_train.sh`, `scripts/train_st_multimodal.py`, `configs/accelerate_*.yaml`, `configs/train_server_*.yaml` | Native path: raw manifest/media; tri-encoder path: `data.cache_dir` (+ optional `validation.cache_dir`) | `checkpoint-*`, `final/`, `train_status.json` under `output_dir` |
| Multi-node launch | `scripts/run_datacenter_multinode_train.sh`, `configs/accelerate_2node_16gpu.yaml` | Rendezvous (`MASTER_ADDR`, `MASTER_PORT`), rank topology, shared/replicated cache and output paths | Coordinated distributed run, shared checkpoint discovery/resume |
| Checkpoint/final evaluation | `scripts/evaluate_tri_encoder.py`, `scripts/watch_final_eval_and_upload.py`, `scripts/upload_tri_encoder_to_hf.py` | `checkpoint-*` or `final/` artifacts + config context | Eval metrics, optional upload/publish flow |

## Datacenter Cached Path: Exact Stage Ordering

1. Run dataset bootstrap (`download_real_data.py` or `run_server_datacenter_pipeline.sh`) until manifests/media are materialized.
2. Run `preprocess_manifest_cache.py` for train and validation to build shard caches.
3. Confirm cache directories contain both `metadata.json` and `shard_*.pt`.
4. Launch multi-node training with `run_datacenter_multinode_train.sh` (node0 first, then remaining ranks).
5. Resume from latest shared checkpoint automatically via `OUTPUT_DIR_HOST`.
6. Evaluate `checkpoint-*` or `final/` outputs with `evaluate_tri_encoder.py`.

## Primary Artifact Contracts

- **Manifest contract:** `train_manifest.jsonl` / `val_manifest.jsonl` produced by download bootstrap.
- **Cache contract:** `metadata.json` + shard files `shard_*.pt` for each cache split.
- **Checkpoint contract:** `output_dir/checkpoint-*` plus `trainer_state.json` for resume.
- **Final artifact contract:** `output_dir/final/model.pt` and `output_dir/final/config.json`.
- **Run status contract:** `output_dir/train_status.json`.

