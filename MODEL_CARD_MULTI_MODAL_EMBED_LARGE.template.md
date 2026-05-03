---
license: apache-2.0
library_name: pytorch
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- multimodal
- embeddings
- retrieval
- image-text
- audio-text
- text-image-audio
- tri-encoder
- semantic-router
- pytorch
model-index:
- name: __MODEL_NAME__
  results:
  - task:
      type: sentence-similarity
    dataset:
      name: Internal cached validation set
      type: cached_retrieval_validation
    metrics:
    - name: Eval loss
      type: eval_loss
      value: __EVAL_LOSS__
    - name: Eval top1
      type: eval_top1
      value: __EVAL_TOP1__
---

# __MODEL_NAME__

`__MODEL_NAME__` is the large production multimodal embedding model from the [llm-semantic-router](https://huggingface.co/llm-semantic-router) project.

It is designed for routing, retrieval, and cross-modal matching across text, image, and audio rather than for generative chat. The model uses a tri-encoder architecture with separate text, image, and audio towers projected into a shared embedding space.

## Model Description

This release packages the large routing-grade tri-encoder trained in PyTorch with the server training stack from this repository.

Architecture:

- text encoder: `__TEXT_ENCODER_NAME__`
- image encoder: `__IMAGE_ENCODER_NAME__`
- audio encoder: `__AUDIO_ENCODER_NAME__`
- shared embedding dimension: `__EMBEDDING_DIM__`
- max text length: `__MAX_TEXT_LENGTH__`

Training characteristics:

- objective: cached multiple negatives ranking loss
- training stack: PyTorch + Accelerate
- target hardware: AMD MI300X
- data pipeline: cached tensor shards with sequential shard loading and worker-local prefetch

## Intended Use

This model is intended for:

- semantic routing
- multimodal retrieval
- matching text to images or audio
- embedding user inputs from different modalities into one shared space

This model is not a native `AutoModel.from_pretrained(...)` multimodal checkpoint. It is a custom tri-encoder exported as:

- `model.pt`
- `config.json`
- `src/hf_st_mm/...` source package for loading and inference

## Validation Snapshot

At upload time, the final export was evaluated with the repository's tri-encoder evaluator.

- `eval_loss`: `__EVAL_LOSS__`
- `eval_top1`: `__EVAL_TOP1__`

## Installation

```bash
pip install torch sentence-transformers transformers accelerate safetensors pillow librosa soundfile huggingface_hub
```

## Python Usage

The easiest way to use this model in Python is to download the full repository snapshot for the model, add the packaged `src/` directory to `sys.path`, and load the tri-encoder from `model.pt` and `config.json`.

```python
import json
import os
import sys

import torch
from huggingface_hub import snapshot_download

repo_id = "__REPO_ID__"
local_dir = snapshot_download(repo_id=repo_id)

sys.path.insert(0, os.path.join(local_dir, "src"))

from hf_st_mm.data import PairItem
from hf_st_mm.model import MultiModalSentenceEmbedder

with open(os.path.join(local_dir, "config.json"), "r", encoding="utf-8") as handle:
    cfg = json.load(handle)

model = MultiModalSentenceEmbedder(
    text_encoder_name=cfg["model"]["text_encoder_name"],
    image_encoder_name=cfg["model"]["image_encoder_name"],
    audio_encoder_name=cfg["model"]["audio_encoder_name"],
    embedding_dim=int(cfg["model"]["embedding_dim"]),
    max_text_length=int(cfg["model"]["max_text_length"]),
)
state_dict = torch.load(os.path.join(local_dir, "model.pt"), map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
```

### Encode Text, Image, and Audio

```python
items = [
    PairItem(modality="text", value="route this request to the billing team"),
    PairItem(modality="image", value="/path/to/screenshot.png"),
    PairItem(modality="audio", value="/path/to/call.wav"),
]

with torch.no_grad():
    embeddings = model.encode_items(items)

print(embeddings.shape)  # [3, __EMBEDDING_DIM__]
```

### Compare Similarity Across Modalities

```python
import torch.nn.functional as F

query = PairItem(modality="text", value="refund request for wrong charge")
candidate = PairItem(modality="audio", value="/path/to/refund_call.wav")

with torch.no_grad():
    embs = model.encode_items([query, candidate])

similarity = F.cosine_similarity(embs[0:1], embs[1:2]).item()
print(f"similarity={similarity:.4f}")
```

## Notes

- Text inputs can be provided as raw strings or tokenized features.
- Image and audio inputs can be provided as file paths.
- Cached tensor payloads are supported by the training stack, but the simplest inference path is to use file paths or raw text.
- This release is intended for production retrieval and routing use cases rather than for instruction-following or caption generation.

## Limitations

- This is a custom tri-encoder export, not a standard Transformers auto-class package.
- Inference currently relies on the packaged `hf_st_mm` source code.
- The validation metrics reported here come from the repository's cached retrieval validation path, not from a public benchmark leaderboard.

## Training Code

Training and evaluation code live in the server training project that produced this checkpoint.

- trainer: `scripts/train_st_multimodal.py`
- evaluator: `scripts/evaluate_tri_encoder.py`
- model: `src/hf_st_mm/model.py`
