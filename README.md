# HF Sentence-Transformers Multimodal Server Training

Standalone training project for server-scale multimodal embeddings on AMD MI300X using Docker + ROCm + Hugging Face tooling.

The training entrypoint now uses the native Sentence Transformers multimodal trainer stack (`SentenceTransformer`, `SentenceTransformerTrainer`, `CachedMultipleNegativesRankingLoss`, `InformationRetrievalEvaluator`) instead of the previous custom tri-encoder loop.

This project is intentionally isolated so it can be moved into a new git repo.

## Native model requirement

The native Sentence Transformers API expects a single multimodal model checkpoint exposed through `model.model_name`.

The bundled example configs use `Qwen/Qwen3-VL-Embedding-2B`, which is a text+image model. Because your manifests also contain audio, the example configs set `data.allowed_modalities: [text, image]` and skip audio records during training.

If you want to train on audio too, replace `model.model_name` with an audio-capable native multimodal checkpoint and expand `data.allowed_modalities` accordingly.

## Project layout

- `configs/train_server_native.yaml`: Native Sentence Transformers multimodal config for the smaller server dataset
- `configs/accelerate_8gpu.yaml`: Accelerate multi-GPU config
- `configs/train_server_datacenter_8gpu_native.yaml`: Native Sentence Transformers config for the datacenter-scale run
- `scripts/train_st_multimodal.py`: Main training entrypoint
- `scripts/validate_dataset.py`: Manifest/media validator
- `scripts/run_docker_train.sh`: End-to-end Docker launch helper
- `scripts/run_server_datacenter_pipeline.sh`: Full resumable server-size dataset/cache/train pipeline

## Dataset contract

Each line in the train manifest can be either explicit triplet format:

```json
{"query":{"type":"image","value":"img/a.jpg"},"positive":{"type":"text","value":"a cat on a sofa"},"negative":{"type":"text","value":"a truck in snow"}}
```

or compatible pair formats:

```json
{"texts_a":"query text","texts_b":"positive text"}
{"image_path":"img/a.jpg","caption":"a cat on a sofa"}
{"audio_path":"wav/a.wav","caption":"spoken caption"}
```

If paths are relative, they resolve against `image_root` and `audio_root`.

For native Sentence Transformers training, these manifest records are converted into a Hugging Face `Dataset` with `query`, `positive`, and optional `negative_0` columns. If some rows have negatives and others do not, the trainer drops the negative column and trains on pairs only, matching the Sentence Transformers loss contract.

### Training dataset interfaces

There are three distinct dataset interfaces in the native trainer path, and each one exists because a different library boundary expects a different representation.

1. Manifest interface

The JSONL manifest is the source-of-truth interface for the dataset generator and validator. It is explicit about semantic roles and modalities:

```json
{"query":{"type":"image","value":"img/a.jpg"},"positive":{"type":"text","value":"a cat on a sofa"},"negative":{"type":"text","value":"a truck in snow"}}
```

This is the interface consumed by `JsonlManifestDataset` in `src/hf_st_mm/data.py`.

2. Sentence Transformers row interface

After parsing, each manifest row is converted into Sentence Transformers columns:

```python
{
	"query": {"image": <PIL.Image.Image>} or {"text": "..."},
	"positive": {"text": "..."},
	"negative_0": {"text": "..."},
}
```

This is the interface expected by `SentenceTransformerTrainer`, `CachedMultipleNegativesRankingLoss`, and `InformationRetrievalEvaluator`. The column names matter: Sentence Transformers uses positional column semantics, so `query` must stay first and `positive` second.

3. Hugging Face Arrow serialization interface

Once those rows are materialized in a Hugging Face `Dataset`, image payloads are no longer kept as raw PIL objects. Arrow normalizes them into a storage-friendly structure that looks like:

```python
{"bytes": b"...", "path": "..."}
```

That representation is fine for storage, but it is not the multimodal input shape that Qwen3-VL expects when `model.preprocess(...)` runs inside Sentence Transformers.

### Why the trainer-side normalization exists

The native Sentence Transformers trainer calls `model.preprocess(...)` through its collator. For text rows, the Arrow-backed dataset representation is already usable. For image rows, it is not: the model expects a real image object or a supported multimodal dict such as `{"image": PIL_image}`, while the dataset hands back `{"image": {"bytes": ..., "path": ...}}`.

That is why `scripts/train_st_multimodal.py` installs a custom `BaseDataCollator` wrapper. It normalizes each batch element right before preprocessing by converting Arrow image payloads back into `PIL.Image.Image` objects. Without that layer, the native trainer fails before or during the forward pass with multimodal type errors.

### Why the modality filter interfaces exist

The dataset adapter supports both global and role-specific modality filters:

- `allowed_modalities`: keep only rows whose modalities are all supported by the selected model.
- `query_modalities`: constrain the query side only.
- `positive_modalities`: constrain the positive side only.
- `negative_modalities`: constrain the negative side only.

These filters are needed because the manifest may contain mixed modality combinations, while a specific native checkpoint or smoke test may only support a subset safely. In the validated image smoke path, the working slice was:

```yaml
allowed_modalities: [text, image]
query_modalities: [image]
positive_modalities: [text]
negative_modalities: [text]
```

That keeps the training batch homogeneous on the query side. Without this role-specific filter, the smoke run could mix text-query and image-query rows in the same training path, which triggered a Qwen3-VL forward failure in this container.

### Why we do not reuse the old cache interface

The previous training stack used cached encoder-specific tensor shards built by `scripts/preprocess_manifest_cache.py`. The native Sentence Transformers trainer does not accept those shards, because it needs raw examples that it can preprocess with the selected multimodal model and processor at training time.

That is why the native path consumes raw manifests plus media roots instead of the older cached tensor interface.

## Build and run

```bash
cd hf-st-multimodal-server
bash scripts/run_docker_train.sh build
bash scripts/run_docker_train.sh
```

The launcher does:
1. dataset validation
2. distributed training with `accelerate launch`

By default it uses `configs/train_server_native.yaml`.

## Full Server-Size Pipeline

The standalone repo still includes the full datacenter data recipe:

- image-text: COCO + LLaVA-CC3M-595K
- audio-text: LibriSpeech clean + LibriSpeech other + VoxPopuli + TEDLIUM
- optional audio expansion hooks: People's Speech, Common Voice, GigaSpeech
- resumable dataset make-up and training

Run the full pipeline without deleting existing downloads:

```bash
cd hf-st-multimodal-server
bash scripts/run_server_datacenter_pipeline.sh build
bash scripts/run_server_datacenter_pipeline.sh
```

Default roots:

- dataset root: `/scratch/2dmse-data/server_full`
- output dir: `/scratch/hf_st_mm_outputs/server_datacenter_8gpu`

The native trainer path no longer supports the older tensor cache built by `scripts/preprocess_manifest_cache.py`, because those shards contain encoder-specific preprocessed tensors from the previous custom tri-encoder implementation.

## Direct launch command

```bash
docker compose run --rm trainer bash -lc "accelerate launch --config_file /workspace/app/configs/accelerate_8gpu.yaml /workspace/app/scripts/train_st_multimodal.py --config /workspace/app/configs/train_server_native.yaml"
```

## Notes for MI300X

- Compose mounts `/dev/kfd` and `/dev/dri` and uses `ipc: host` with `shm_size: 64gb`.
- RCCL/NCCL defaults are set for stability in Docker and launch script.
- If 8-GPU launch is unstable in your environment, copy the accelerate config and reduce `num_processes` to 4.
- `datasets` stays pinned to `2.21.0` because `scripts/download_real_data.py` still relies on `load_dataset(..., trust_remote_code=True)` for some source datasets.
