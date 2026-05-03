#!/usr/bin/env python3
import os
import sys

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _is_primary_process() -> bool:
    for env_name in ("ACCELERATE_PROCESS_INDEX", "RANK", "LOCAL_RANK"):
        env_value = os.environ.get(env_name)
        if env_value is not None:
            return env_value in {"0", "-1"}
    return True

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(line_buffering=True, write_through=True)

if _is_primary_process():
    print("[bootstrap] importing trainer dependencies", file=sys.stderr, flush=True)

import argparse
import json
import math
import random
from contextlib import nullcontext
from io import BytesIO
from importlib import metadata
from typing import Any, Dict, Optional

import torch
import yaml
from accelerate import Accelerator
from datasets import IterableDataset as HFIterableDataset
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hf_st_mm.data import CachedShardDataset, SequentialShardDataset, collate_records, manifest_to_sentence_transformers_dataset
from hf_st_mm.model import MultiModalSentenceEmbedder, multiple_negatives_ranking_loss

try:
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
except ImportError:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments

try:
    from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
except ImportError:
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

try:
    from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss, MatryoshkaLoss
except ImportError:
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, MatryoshkaLoss

try:
    from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking import RandContext, _create_minibatch
except ImportError:
    from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking import RandContext, _create_minibatch

try:
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
except ImportError:
    from sentence_transformers.training_args import BatchSamplers

try:
    from sentence_transformers.base.data_collator import BaseDataCollator
except ImportError:
    from sentence_transformers.base.data_collator import BaseDataCollator


def configure_unbuffered_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


def log_progress(message: str) -> None:
    if _is_primary_process():
        print(message, flush=True)


class NativeSentenceTransformerTrainer(SentenceTransformerTrainer):
    def add_model_card_callback(self, default_args_dict: Dict[str, Any]) -> None:
        # Sentence Transformers eagerly inspects dataset examples for model-card metadata.
        # In this container, that path trips over torchvision JPEG decoding before training starts.
        # Skipping the callback keeps training functional and does not affect optimization.
        return

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError(f"Training requires specifying a train_dataset to the {self.__class__.__name__}.")

        self.accelerator.even_batches = False
        if isinstance(self.train_dataset, HFIterableDataset):
            self.accelerator.dataloader_config.dispatch_batches = False
            self.accelerator.dataloader_config.split_batches = False

        self._train_dataloader = self.accelerator.prepare(
            self._build_dataloader(self.train_dataset, self.args.train_batch_size, dataset_kind="train")
        )
        return self._train_dataloader


class NativeCachedMultipleNegativesRankingLoss(CachedMultipleNegativesRankingLoss):
    @staticmethod
    def _prune_empty_multimodal_tensors(sentence_feature_minibatch: Dict[str, Any]) -> Dict[str, Any]:
        pruned = dict(sentence_feature_minibatch)

        for grid_key, pixel_key, count_key in (
            ("image_grid_thw", "pixel_values", "num_images_per_sample"),
            ("video_grid_thw", "pixel_values_videos", "num_videos_per_sample"),
        ):
            grid = pruned.get(grid_key)
            if getattr(grid, "numel", lambda: 0)() == 0:
                pruned.pop(grid_key, None)
                pruned.pop(pixel_key, None)
                pruned.pop(count_key, None)

        return pruned

    def embed_minibatch(
        self,
        sentence_feature: dict[str, torch.Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ):
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = _create_minibatch(sentence_feature, begin, end)
        sentence_feature_minibatch = self._prune_empty_multimodal_tensors(sentence_feature_minibatch)
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]
        return reps, random_state


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_sentence_transformers_version() -> str:
    try:
        version = metadata.version("sentence-transformers")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install the project requirements in the trainer environment."
        ) from exc

    major = int(version.split(".", 1)[0])
    if major < 5:
        raise RuntimeError(
            f"sentence-transformers>={5}.0.0 is required for the native multimodal trainer path; found {version}."
        )
    return version


def build_model(cfg: Dict[str, Any]) -> SentenceTransformer:
    model_cfg = cfg["model"]
    model_name = model_cfg.get("model_name")
    if not model_name:
        raise ValueError(
            "Native Sentence Transformers multimodal training requires model.model_name. "
            "The previous text/image/audio encoder triplet config is not compatible with this path."
        )

    model_kwargs = dict(model_cfg.get("model_kwargs", {}))
    processor_kwargs = dict(model_cfg.get("processor_kwargs", {}))
    if bool(model_cfg.get("trust_remote_code", False)):
        model_kwargs.setdefault("trust_remote_code", True)
        processor_kwargs.setdefault("trust_remote_code", True)

    model = SentenceTransformer(
        model_name,
        model_kwargs=model_kwargs or None,
        processor_kwargs=processor_kwargs or None,
        truncate_dim=model_cfg.get("truncate_dim"),
    )

    max_text_length = model_cfg.get("max_text_length")
    if max_text_length:
        model.max_seq_length = int(max_text_length)

    return model


def resolve_supported_modalities(model: SentenceTransformer) -> set[str]:
    modalities = getattr(model, "modalities", None)
    if modalities:
        return {str(item) for item in modalities}

    supported = set()
    supports = getattr(model, "supports", None)
    if callable(supports):
        for modality in ("text", "image", "audio", "video"):
            try:
                if supports(modality):
                    supported.add(modality)
            except Exception:
                continue
    return supported


def build_loss(model: SentenceTransformer, cfg: Dict[str, Any]):
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "cached_mnrl")
    if loss_type != "cached_mnrl":
        raise ValueError(f"Unsupported loss type for native trainer path: {loss_type}")

    loss = NativeCachedMultipleNegativesRankingLoss(
        model,
        scale=float(loss_cfg.get("scale", 20.0)),
        mini_batch_size=int(loss_cfg.get("mini_batch_size", 1)),
    )

    matryoshka_dims = loss_cfg.get("matryoshka_dims") or []
    if matryoshka_dims:
        loss = MatryoshkaLoss(model, loss, matryoshka_dims=[int(dim) for dim in matryoshka_dims])
    return loss


def build_evaluator(eval_dataset, cfg: Dict[str, Any]) -> Optional[InformationRetrievalEvaluator]:
    if eval_dataset is None:
        return None

    eval_rows = list(eval_dataset)
    if not eval_rows:
        return None

    queries = {idx: normalize_st_input(row["query"]) for idx, row in enumerate(eval_rows)}
    corpus = {idx: normalize_st_input(row["positive"]) for idx, row in enumerate(eval_rows)}
    relevant_docs = {idx: [idx] for idx in range(len(eval_rows))}

    if "negative_0" in eval_dataset.column_names:
        offset = len(eval_rows)
        for idx, row in enumerate(eval_rows):
            corpus[offset + idx] = normalize_st_input(row["negative_0"])

    validation_cfg = cfg.get("validation", {})
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        batch_size=int(validation_cfg.get("batch_size", 1)),
        show_progress_bar=True,
        name=validation_cfg.get("name", "multimodal-ir"),
    )


def normalize_mixed_precision(value: Any) -> str:
    if isinstance(value, bool):
        return "bf16" if value else "no"

    normalized = str(value or "bf16").strip().lower()
    if normalized in {"false", "off", "none", "no"}:
        return "no"
    if normalized in {"true", "on"}:
        return "bf16"
    if normalized not in {"no", "fp8", "fp16", "bf16"}:
        raise ValueError(f"Unsupported mixed_precision mode: {value}")
    return normalized


def build_training_args(cfg: Dict[str, Any]) -> SentenceTransformerTrainingArguments:
    training_cfg = cfg["training"]
    use_iterable_dataset = bool(cfg.get("data", {}).get("use_iterable_dataset", False))
    mixed_precision = normalize_mixed_precision(training_cfg.get("mixed_precision", "bf16"))
    eval_enabled = bool(cfg.get("validation", {}).get("manifest_path"))
    max_steps = training_cfg.get("max_steps")
    warmup_steps = training_cfg.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = float(training_cfg.get("warmup_ratio", 0.0))

    args = SentenceTransformerTrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=float(training_cfg["epochs"]),
        max_steps=int(max_steps) if max_steps is not None else -1,
        per_device_train_batch_size=int(training_cfg["batch_size"]),
        per_device_eval_batch_size=int(training_cfg.get("eval_batch_size", training_cfg["batch_size"])),
        gradient_accumulation_steps=int(training_cfg.get("grad_accum_steps", 1)),
        learning_rate=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        warmup_steps=float(warmup_steps),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        dataloader_num_workers=int(training_cfg.get("num_workers", 4)),
        dataloader_drop_last=bool(training_cfg.get("drop_last", True)),
        batch_sampler=BatchSamplers.BATCH_SAMPLER if use_iterable_dataset else BatchSamplers.NO_DUPLICATES,
        accelerator_config={"dispatch_batches": False} if use_iterable_dataset else None,
        save_strategy="steps",
        save_steps=int(training_cfg.get("save_every", 1000)),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        logging_strategy="steps",
        logging_steps=int(training_cfg.get("log_every", 10)),
        logging_first_step=True,
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=int(training_cfg.get("eval_every", training_cfg.get("save_every", 1000))) if eval_enabled else None,
        fp16=mixed_precision == "fp16",
        bf16=mixed_precision == "bf16",
        seed=int(cfg.get("seed", 42)),
        remove_unused_columns=False,
        run_name=training_cfg.get("run_name"),
        disable_tqdm=False,
    )
    return args


def resolve_training_max_steps(cfg: Dict[str, Any], train_info: Dict[str, Any]) -> Optional[int]:
    training_cfg = cfg["training"]
    explicit_max_steps = training_cfg.get("max_steps")
    if explicit_max_steps is not None:
        return int(explicit_max_steps)

    if not bool(cfg.get("data", {}).get("use_iterable_dataset", False)):
        return None

    world_size = int(os.environ.get("WORLD_SIZE") or os.environ.get("ACCELERATE_NUM_PROCESSES") or 1)
    world_size = max(world_size, 1)
    per_device_batch_size = int(training_cfg["batch_size"])
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    drop_last = bool(training_cfg.get("drop_last", True))
    global_micro_batch = max(per_device_batch_size * world_size, 1)
    train_rows = int(train_info["num_rows"])

    if drop_last:
        micro_steps_per_epoch = train_rows // global_micro_batch
    else:
        micro_steps_per_epoch = math.ceil(train_rows / global_micro_batch)

    if micro_steps_per_epoch <= 0:
        raise ValueError("Computed zero training steps per epoch; dataset is too small for the configured global batch size.")

    optimizer_steps_per_epoch = max(1, math.ceil(micro_steps_per_epoch / grad_accum_steps))
    num_epochs = float(training_cfg["epochs"])
    return max(1, math.ceil(num_epochs * optimizer_steps_per_epoch))


def deserialize_image_payload(payload: Any) -> Any:
    if payload is None or isinstance(payload, Image.Image):
        return payload
    if isinstance(payload, (list, tuple)):
        return [deserialize_image_payload(item) for item in payload]
    if isinstance(payload, str):
        with Image.open(payload) as image:
            return image.convert("RGB").copy()
    if isinstance(payload, dict):
        if payload.get("bytes") is not None:
            with Image.open(BytesIO(payload["bytes"])) as image:
                return image.convert("RGB").copy()
        if payload.get("path"):
            with Image.open(payload["path"]) as image:
                return image.convert("RGB").copy()
    return payload


def normalize_st_input(value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    normalized: Dict[str, Any] = {}
    if value.get("text") is not None:
        normalized["text"] = value["text"]
    if value.get("image") is not None:
        normalized["image"] = deserialize_image_payload(value["image"])
    if value.get("audio") is not None:
        normalized["audio"] = value["audio"]
    if value.get("video") is not None:
        normalized["video"] = value["video"]

    if normalized:
        return normalized
    return value


def build_data_collator(model: SentenceTransformer) -> BaseDataCollator:
    def preprocess_fn(inputs, prompt=None, task=None):
        normalized_inputs = [normalize_st_input(item) for item in inputs]
        preprocessed = model.preprocess(normalized_inputs, prompt=prompt, task=task)

        num_images_per_sample = []
        num_videos_per_sample = []
        has_images = False
        has_videos = False
        for item in normalized_inputs:
            if isinstance(item, dict):
                image_value = item.get("image")
                video_value = item.get("video")
            else:
                image_value = None
                video_value = None

            image_count = 0 if image_value is None else (len(image_value) if isinstance(image_value, list) else 1)
            video_count = 0 if video_value is None else (len(video_value) if isinstance(video_value, list) else 1)
            num_images_per_sample.append(image_count)
            num_videos_per_sample.append(video_count)
            has_images = has_images or image_count > 0
            has_videos = has_videos or video_count > 0

        if has_images:
            preprocessed["num_images_per_sample"] = torch.tensor(num_images_per_sample, dtype=torch.long)
        if has_videos:
            preprocessed["num_videos_per_sample"] = torch.tensor(num_videos_per_sample, dtype=torch.long)

        return preprocessed

    return BaseDataCollator(preprocess_fn=preprocess_fn)


def load_sentence_transformers_datasets(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    allowed_modalities = data_cfg.get("allowed_modalities")
    train_use_iterable_dataset = bool(data_cfg.get("use_iterable_dataset", False))
    if data_cfg.get("cache_dir"):
        raise ValueError(
            "cache_dir is not supported by the native Sentence Transformers trainer path. "
            "The existing cache shards contain encoder-specific tensors from preprocess_manifest_cache.py and cannot be reused."
        )

    train_dataset, train_info = manifest_to_sentence_transformers_dataset(
        manifest_path=data_cfg["manifest_path"],
        image_root=data_cfg.get("image_root"),
        audio_root=data_cfg.get("audio_root"),
        allow_missing_negative=bool(data_cfg.get("allow_missing_negative", True)),
        allowed_modalities=allowed_modalities,
        query_modalities=data_cfg.get("query_modalities"),
        positive_modalities=data_cfg.get("positive_modalities"),
        negative_modalities=data_cfg.get("negative_modalities"),
        as_iterable=train_use_iterable_dataset,
    )

    validation_cfg = cfg.get("validation", {})
    eval_use_iterable_dataset = bool(validation_cfg.get("use_iterable_dataset", False))
    eval_dataset = None
    eval_info = None
    if validation_cfg.get("cache_dir"):
        raise ValueError(
            "validation.cache_dir is not supported by the native Sentence Transformers trainer path. "
            "Use validation.manifest_path with raw file paths instead."
        )
    if validation_cfg.get("manifest_path"):
        eval_dataset, eval_info = manifest_to_sentence_transformers_dataset(
            manifest_path=validation_cfg["manifest_path"],
            image_root=data_cfg.get("image_root"),
            audio_root=data_cfg.get("audio_root"),
            allow_missing_negative=True,
            allowed_modalities=allowed_modalities,
            query_modalities=data_cfg.get("query_modalities"),
            positive_modalities=data_cfg.get("positive_modalities"),
            negative_modalities=data_cfg.get("negative_modalities"),
            as_iterable=eval_use_iterable_dataset,
            max_records=validation_cfg.get("max_rows"),
        )

    return train_dataset, train_info, eval_dataset, eval_info


def validate_modalities(model: SentenceTransformer, dataset_info: Dict[str, Any]) -> None:
    supported_modalities = resolve_supported_modalities(model)
    if not supported_modalities:
        return

    unsupported = [modality for modality in dataset_info["modalities"] if modality not in supported_modalities]
    if unsupported:
        supported_csv = ", ".join(sorted(supported_modalities))
        unsupported_csv = ", ".join(sorted(unsupported))
        raise ValueError(
            f"Model does not support dataset modalities: {unsupported_csv}. Supported modalities: {supported_csv}."
        )


def write_status(output_dir: str, payload: Dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_status.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def is_datacenter_tri_encoder_config(cfg: Dict[str, Any]) -> bool:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    return bool(
        data_cfg.get("cache_dir")
        and model_cfg.get("text_encoder_name")
        and model_cfg.get("image_encoder_name")
        and model_cfg.get("audio_encoder_name")
    )


def summarize_cached_dataset(dataset: CachedShardDataset, sample_size: int = 256) -> Dict[str, Any]:
    observed_modalities = set()
    negatives_present = 0
    negatives_missing = 0
    rows_to_sample = min(len(dataset), sample_size)

    for idx in range(rows_to_sample):
        record = dataset[idx]
        observed_modalities.add(record.query.modality)
        observed_modalities.add(record.positive.modality)
        if record.negative is None:
            negatives_missing += 1
        else:
            negatives_present += 1
            observed_modalities.add(record.negative.modality)

    return {
        "num_rows": len(dataset),
        "modalities": sorted(observed_modalities),
        "num_negatives_present": negatives_present,
        "num_negatives_missing": negatives_missing,
        "has_uniform_negatives": negatives_present == 0 or negatives_missing == 0,
        "sampled_rows": rows_to_sample,
    }


def load_datacenter_tri_encoder_datasets(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    training_cfg = cfg.get("training", {})
    summary_dataset = CachedShardDataset(data_cfg["cache_dir"])
    train_info = summarize_cached_dataset(summary_dataset)

    if bool(training_cfg.get("sequential_shard_loading", False)):
        rank = int(os.environ.get("ACCELERATE_PROCESS_INDEX") or os.environ.get("RANK") or 0)
        world_size = int(os.environ.get("WORLD_SIZE") or os.environ.get("ACCELERATE_NUM_PROCESSES") or 1)
        train_dataset = SequentialShardDataset(
            data_cfg["cache_dir"],
            shuffle=bool(training_cfg.get("shuffle", True)),
            rank=rank,
            world_size=world_size,
            shard_cache_limit=int(training_cfg.get("shard_cache_limit", 4)),
            prefetch_shards=int(training_cfg.get("shard_prefetch", 2)),
        )
    else:
        train_dataset = CachedShardDataset(
            data_cfg["cache_dir"],
            shard_cache_limit=int(training_cfg.get("shard_cache_limit", 4)),
            prefetch_shards=int(training_cfg.get("shard_prefetch", 2)),
        )

    validation_cfg = cfg.get("validation", {})
    eval_dataset = None
    eval_info = None
    if validation_cfg.get("cache_dir"):
        eval_dataset = CachedShardDataset(
            validation_cfg["cache_dir"],
            shard_cache_limit=int(validation_cfg.get("shard_cache_limit", 2)),
            prefetch_shards=int(validation_cfg.get("shard_prefetch", 1)),
        )
        eval_info = summarize_cached_dataset(eval_dataset)

    return train_dataset, train_info, eval_dataset, eval_info


def build_datacenter_tri_encoder_model(cfg: Dict[str, Any]) -> MultiModalSentenceEmbedder:
    model_cfg = cfg["model"]
    return MultiModalSentenceEmbedder(
        text_encoder_name=model_cfg["text_encoder_name"],
        image_encoder_name=model_cfg["image_encoder_name"],
        audio_encoder_name=model_cfg["audio_encoder_name"],
        embedding_dim=int(model_cfg.get("embedding_dim", model_cfg.get("output_dim", 768))),
        max_text_length=int(model_cfg.get("max_text_length", 32768)),
    )


def _encode_items_for_training(model: torch.nn.Module, items):
    encode_owner = model.module if hasattr(model, "module") else model
    return encode_owner.encode_items(items)


def _encode_query_positive_batch(model: torch.nn.Module, query_items, positive_items) -> tuple[torch.Tensor, torch.Tensor]:
    query_count = len(query_items)
    combined_embeddings = _encode_items_for_training(model, list(query_items) + list(positive_items))
    return combined_embeddings[:query_count], combined_embeddings[query_count:]


def _build_cached_loader_kwargs(num_workers: int, prefetch_factor: int) -> Dict[str, Any]:
    loader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return loader_kwargs


class InterleavedModalityBatchSampler(BatchSampler):
    def __init__(self, dataset: CachedShardDataset, batch_size: int, drop_last: bool, seed: int = 42) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.pattern = self._detect_query_modality_pattern()
        self.period = len(self.pattern)
        self.modality_offsets = {modality: offset for offset, modality in enumerate(self.pattern)}
        self.modality_counts = {
            modality: self._count_indices_for_offset(offset) for modality, offset in self.modality_offsets.items()
        }

    def _count_indices_for_offset(self, offset: int) -> int:
        if offset >= len(self.dataset):
            return 0
        return ((len(self.dataset) - 1 - offset) // self.period) + 1

    def _detect_query_modality_pattern(self) -> list[str]:
        sample_rows = min(len(self.dataset), 18)
        if sample_rows == 0:
            raise ValueError("Cannot build a modality batch sampler for an empty cached dataset.")

        pattern: list[str] = []
        for idx in range(sample_rows):
            modality = self.dataset[idx].query.modality
            if modality not in pattern:
                pattern.append(modality)
            if len(pattern) >= 3:
                break

        if len(pattern) < 2:
            raise ValueError("Cached dataset does not expose enough query-modality variation for modality-aware batching.")

        verification_rows = min(len(self.dataset), len(pattern) * 64)
        for idx in range(verification_rows):
            expected = pattern[idx % len(pattern)]
            observed = self.dataset[idx].query.modality
            if observed != expected:
                raise ValueError(
                    "Cached dataset query modalities are not globally interleaved; disable modality-aware batching "
                    "or regenerate the cache with stable ordering."
                )

        generator = random.Random(self.seed)
        sampled_indices = {generator.randrange(len(self.dataset)) for _ in range(min(len(self.dataset), 256))}
        for idx in sampled_indices:
            expected = pattern[idx % len(pattern)]
            observed = self.dataset[idx].query.modality
            if observed != expected:
                raise ValueError(
                    "Cached dataset query modalities are not globally interleaved; disable modality-aware batching "
                    "or regenerate the cache with stable ordering."
                )
        return pattern

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        shuffled_positions = {
            modality: torch.randperm(count, generator=generator)
            for modality, count in self.modality_counts.items()
            if count > 0
        }

        batch_descriptors: list[tuple[str, int]] = []
        for modality, count in self.modality_counts.items():
            if count <= 0:
                continue
            num_batches = count // self.batch_size if self.drop_last else math.ceil(count / self.batch_size)
            for batch_idx in range(num_batches):
                batch_descriptors.append((modality, batch_idx))

        if not batch_descriptors:
            return

        batch_order = torch.randperm(len(batch_descriptors), generator=generator).tolist()
        for order_idx in batch_order:
            modality, batch_idx = batch_descriptors[order_idx]
            positions = shuffled_positions[modality]
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            batch_positions = positions[start:end]
            if len(batch_positions) < self.batch_size and self.drop_last:
                continue
            offset = self.modality_offsets[modality]
            yield (offset + batch_positions * self.period).tolist()

    def __len__(self) -> int:
        total_batches = 0
        for count in self.modality_counts.values():
            total_batches += count // self.batch_size if self.drop_last else math.ceil(count / self.batch_size)
        return total_batches


def _tri_encoder_checkpoint_state_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "trainer_state.json")


def save_tri_encoder_checkpoint(
    accelerator: Accelerator,
    checkpoint_dir: str,
    epoch: int,
    micro_step_in_epoch: int,
    global_step: int,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    accelerator.save_state(checkpoint_dir)
    if accelerator.is_main_process:
        with open(_tri_encoder_checkpoint_state_path(checkpoint_dir), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "epoch": epoch,
                    "micro_step_in_epoch": micro_step_in_epoch,
                    "global_step": global_step,
                },
                handle,
                indent=2,
            )


def load_tri_encoder_checkpoint_state(checkpoint_dir: str) -> Dict[str, int]:
    state_path = _tri_encoder_checkpoint_state_path(checkpoint_dir)
    if not os.path.exists(state_path):
        return {"epoch": 0, "micro_step_in_epoch": 0, "global_step": 0}

    with open(state_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        "epoch": int(raw.get("epoch", 0)),
        "micro_step_in_epoch": int(raw.get("micro_step_in_epoch", 0)),
        "global_step": int(raw.get("global_step", 0)),
    }


def evaluate_tri_encoder_model(
    model: torch.nn.Module,
    eval_loader: Optional[DataLoader],
    accelerator: Accelerator,
    scale: float,
) -> Dict[str, float]:
    if eval_loader is None:
        return {}

    model.eval()
    eval_losses = []
    eval_top1 = []
    eval_progress = None
    if accelerator.is_main_process:
        eval_progress = tqdm(
            eval_loader,
            desc="eval",
            leave=False,
            dynamic_ncols=True,
        )
    eval_iterable = eval_progress if eval_progress is not None else eval_loader
    with torch.no_grad():
        for batch in eval_iterable:
            anchor, positive = _encode_query_positive_batch(model, batch["query"], batch["positive"])
            loss = multiple_negatives_ranking_loss(anchor, positive, scale=scale)
            scores = torch.matmul(anchor, positive.T)
            labels = torch.arange(scores.shape[0], device=scores.device)
            top1 = (scores.argmax(dim=1) == labels).float().mean()

            gathered_loss = accelerator.gather_for_metrics(loss.detach().reshape(1))
            gathered_top1 = accelerator.gather_for_metrics(top1.detach().reshape(1))
            eval_losses.append(gathered_loss.float().mean().item())
            eval_top1.append(gathered_top1.float().mean().item())

            if eval_progress is not None:
                eval_progress.set_postfix(
                    loss=f"{sum(eval_losses) / len(eval_losses):.4f}",
                    top1=f"{sum(eval_top1) / len(eval_top1):.4f}",
                )

    if eval_progress is not None:
        eval_progress.close()

    model.train()
    if not eval_losses:
        return {}
    return {
        "eval_loss": float(sum(eval_losses) / len(eval_losses)),
        "eval_top1": float(sum(eval_top1) / len(eval_top1)),
    }


def save_tri_encoder_final_artifacts(accelerator: Accelerator, model: torch.nn.Module, cfg: Dict[str, Any]) -> None:
    final_dir = os.path.join(cfg["output_dir"], "final")
    os.makedirs(final_dir, exist_ok=True)

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), os.path.join(final_dir, "model.pt"))
        with open(os.path.join(final_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(cfg, handle, indent=2)


def run_datacenter_tri_encoder_training(cfg: Dict[str, Any], resume_path: Optional[str], st_version: str) -> None:
    training_cfg = cfg["training"]
    loss_cfg = cfg.get("loss", {})
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training_cfg.get("grad_accum_steps", 1)),
        mixed_precision=normalize_mixed_precision(training_cfg.get("mixed_precision", "bf16")),
    )
    train_num_workers = int(training_cfg.get("num_workers", 4))
    eval_num_workers = int(cfg.get("validation", {}).get("num_workers", max(1, train_num_workers // 2)))
    prefetch_factor = int(training_cfg.get("prefetch_factor", 4))

    log_progress("[startup] loading cached datacenter tri-encoder datasets")
    train_dataset, train_info, eval_dataset, eval_info = load_datacenter_tri_encoder_datasets(cfg)
    log_progress(
        f"[startup] loaded cached train dataset with {train_info['num_rows']} rows and observed modalities {train_info['modalities']}"
    )
    if eval_info is not None:
        log_progress(
            f"[startup] loaded cached eval dataset with {eval_info['num_rows']} rows and observed modalities {eval_info['modalities']}"
        )

    log_progress("[startup] building datacenter tri-encoder model")
    model = build_datacenter_tri_encoder_model(cfg)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )

    use_modality_homogeneous_batches = bool(training_cfg.get("modality_homogeneous_batches", False))
    use_sequential_shard_loading = isinstance(train_dataset, SequentialShardDataset)
    train_loader_kwargs = _build_cached_loader_kwargs(train_num_workers, prefetch_factor)
    if use_sequential_shard_loading:
        train_loader_kwargs = {}
    elif use_modality_homogeneous_batches:
        train_batch_sampler = InterleavedModalityBatchSampler(
            train_dataset,
            batch_size=int(training_cfg["batch_size"]),
            drop_last=bool(training_cfg.get("drop_last", True)),
            seed=int(cfg.get("seed", 42)),
        )
        train_loader_kwargs["batch_sampler"] = train_batch_sampler
    else:
        train_loader_kwargs["batch_size"] = int(training_cfg["batch_size"])
        train_loader_kwargs["shuffle"] = True
        train_loader_kwargs["drop_last"] = bool(training_cfg.get("drop_last", True))

    train_loader = None
    if not use_sequential_shard_loading:
        train_loader = DataLoader(
            train_dataset,
            collate_fn=collate_records,
            **train_loader_kwargs,
        )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=int(cfg.get("validation", {}).get("batch_size", training_cfg["batch_size"])),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_records,
            **_build_cached_loader_kwargs(eval_num_workers, prefetch_factor),
        )

    if use_sequential_shard_loading:
        shard_batches_per_epoch = train_dataset.estimated_num_batches(
            batch_size=int(training_cfg["batch_size"]),
            drop_last=bool(training_cfg.get("drop_last", True)),
        )
        updates_per_epoch = max(1, math.ceil(shard_batches_per_epoch / max(int(training_cfg.get("grad_accum_steps", 1)), 1)))
    else:
        updates_per_epoch = max(
            1,
            math.ceil(len(train_loader) / max(int(training_cfg.get("grad_accum_steps", 1)), 1)),
        )
    max_train_steps = int(training_cfg.get("max_steps") or math.ceil(float(training_cfg["epochs"]) * updates_per_epoch))
    warmup_cfg = training_cfg.get("warmup_steps")
    if warmup_cfg is None:
        warmup_cfg = float(training_cfg.get("warmup_ratio", 0.0))
        warmup_steps = int(math.ceil(max_train_steps * float(warmup_cfg)))
    else:
        warmup_steps = int(warmup_cfg)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    if use_sequential_shard_loading:
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
        if eval_loader is not None:
            eval_loader = accelerator.prepare(eval_loader)
    accelerator.register_for_checkpointing(scheduler)

    start_epoch = 0
    resume_batch_in_epoch = 0
    global_step = 0
    if resume_path:
        log_progress(f"[startup] resuming datacenter tri-encoder checkpoint from {resume_path}")
        accelerator.load_state(resume_path)
        resume_state = load_tri_encoder_checkpoint_state(resume_path)
        start_epoch = resume_state["epoch"]
        resume_batch_in_epoch = resume_state["micro_step_in_epoch"]
        global_step = resume_state["global_step"]
        if resume_batch_in_epoch >= len(train_loader):
            start_epoch += 1
            resume_batch_in_epoch = 0

    loss_scale = float(loss_cfg.get("scale", 20.0))
    save_every = int(training_cfg.get("save_every", 1000))
    eval_every = int(training_cfg.get("eval_every", save_every))
    log_every = int(training_cfg.get("log_every", 10))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    progress_bar = None

    log_progress(
        f"[startup] cached dataloader workers train={train_num_workers} eval={eval_num_workers} prefetch_factor={prefetch_factor}"
    )
    log_progress(
        "[startup] cached shard prefetch "
        f"train_prefetch={getattr(train_dataset, 'prefetch_shards', 0)} train_cache_limit={getattr(train_dataset, 'shard_cache_limit', 0)} "
        f"eval_prefetch={getattr(eval_dataset, 'prefetch_shards', 0) if eval_dataset is not None else 0}"
    )
    if use_modality_homogeneous_batches:
        log_progress(
            f"[startup] modality-homogeneous batching enabled with interleave pattern {train_batch_sampler.pattern}"
        )
    if use_sequential_shard_loading:
        log_progress("[startup] sequential shard loading enabled; train dataloader workers forced to 0")
    if accelerator.is_main_process:
        progress_bar = tqdm(
            total=max_train_steps,
            initial=global_step,
            desc="train",
            unit="step",
            dynamic_ncols=True,
            smoothing=0.1,
            mininterval=5.0,
            file=sys.stdout,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, int(math.ceil(float(training_cfg["epochs"])) )):
        if not use_sequential_shard_loading and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if use_sequential_shard_loading:
            has_shard = train_dataset.reset(epoch)
            epoch_micro_step = 0
            while has_shard:
                shard_loader = DataLoader(
                    train_dataset,
                    batch_size=int(training_cfg["batch_size"]),
                    shuffle=bool(training_cfg.get("shuffle_within_shard", True)),
                    drop_last=bool(training_cfg.get("drop_last", True)),
                    collate_fn=collate_records,
                    num_workers=0,
                    pin_memory=True,
                )
                for batch in shard_loader:
                    if epoch == start_epoch and epoch_micro_step < resume_batch_in_epoch:
                        epoch_micro_step += 1
                        continue

                    with accelerator.accumulate(model):
                        anchor, positive = _encode_query_positive_batch(model, batch["query"], batch["positive"])
                        loss = multiple_negatives_ranking_loss(anchor, positive, scale=loss_scale)
                        accelerator.backward(loss)

                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            global_step += 1
                            if progress_bar is not None:
                                progress_bar.update(1)
                                progress_bar.set_postfix(epoch=epoch + 1, lr=f"{scheduler.get_last_lr()[0]:.3e}")

                    epoch_micro_step += 1

                    if accelerator.sync_gradients and global_step % log_every == 0:
                        mean_loss = accelerator.gather_for_metrics(loss.detach().reshape(1)).float().mean().item()
                        if progress_bar is not None:
                            progress_bar.set_postfix(epoch=epoch + 1, loss=f"{mean_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.3e}")
                        log_progress(
                            f"[train] epoch={epoch + 1} step={global_step}/{max_train_steps} loss={mean_loss:.5f} lr={scheduler.get_last_lr()[0]:.3e}"
                        )

                    if accelerator.sync_gradients and save_every > 0 and global_step % save_every == 0:
                        checkpoint_dir = os.path.join(cfg["output_dir"], f"checkpoint-{global_step}")
                        save_tri_encoder_checkpoint(
                            accelerator,
                            checkpoint_dir,
                            epoch=epoch,
                            micro_step_in_epoch=epoch_micro_step,
                            global_step=global_step,
                        )
                        log_progress(f"[checkpoint] saved datacenter tri-encoder checkpoint to {checkpoint_dir}")

                    if accelerator.sync_gradients and eval_loader is not None and eval_every > 0 and global_step % eval_every == 0:
                        eval_metrics = evaluate_tri_encoder_model(model, eval_loader, accelerator, loss_scale)
                        if eval_metrics:
                            log_progress(
                                f"[eval] step={global_step} eval_loss={eval_metrics['eval_loss']:.5f} eval_top1={eval_metrics['eval_top1']:.4f}"
                            )

                    if global_step >= max_train_steps:
                        break

                if global_step >= max_train_steps:
                    break
                has_shard = train_dataset.next_shard()
        else:
            for step, batch in enumerate(train_loader):
                if epoch == start_epoch and step < resume_batch_in_epoch:
                    continue

                with accelerator.accumulate(model):
                    anchor, positive = _encode_query_positive_batch(model, batch["query"], batch["positive"])
                    loss = multiple_negatives_ranking_loss(anchor, positive, scale=loss_scale)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        if progress_bar is not None:
                            progress_bar.update(1)
                            progress_bar.set_postfix(epoch=epoch + 1, lr=f"{scheduler.get_last_lr()[0]:.3e}")

                if accelerator.sync_gradients and global_step % log_every == 0:
                    mean_loss = accelerator.gather_for_metrics(loss.detach().reshape(1)).float().mean().item()
                    if progress_bar is not None:
                        progress_bar.set_postfix(epoch=epoch + 1, loss=f"{mean_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.3e}")
                    log_progress(
                        f"[train] epoch={epoch + 1} step={global_step}/{max_train_steps} loss={mean_loss:.5f} lr={scheduler.get_last_lr()[0]:.3e}"
                    )

                if accelerator.sync_gradients and save_every > 0 and global_step % save_every == 0:
                    checkpoint_dir = os.path.join(cfg["output_dir"], f"checkpoint-{global_step}")
                    save_tri_encoder_checkpoint(
                        accelerator,
                        checkpoint_dir,
                        epoch=epoch,
                        micro_step_in_epoch=step + 1,
                        global_step=global_step,
                    )
                    log_progress(f"[checkpoint] saved datacenter tri-encoder checkpoint to {checkpoint_dir}")

                if accelerator.sync_gradients and eval_loader is not None and eval_every > 0 and global_step % eval_every == 0:
                    eval_metrics = evaluate_tri_encoder_model(model, eval_loader, accelerator, loss_scale)
                    if eval_metrics:
                        log_progress(
                            f"[eval] step={global_step} eval_loss={eval_metrics['eval_loss']:.5f} eval_top1={eval_metrics['eval_top1']:.4f}"
                        )

                if global_step >= max_train_steps:
                    break

        resume_batch_in_epoch = 0
        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if progress_bar is not None:
        progress_bar.close()
    save_tri_encoder_final_artifacts(accelerator, model, cfg)

    metrics = {
        "sentence_transformers_version": st_version,
        "train_rows": train_info["num_rows"],
        "train_modalities": train_info["modalities"],
        "global_step": global_step,
        "max_train_steps": max_train_steps,
        "datacenter_tri_encoder_training": True,
    }
    if eval_info is not None:
        metrics.update({
            "eval_rows": eval_info["num_rows"],
            "eval_modalities": eval_info["modalities"],
        })
    if accelerator.is_main_process:
        write_status(cfg["output_dir"], metrics)
    log_progress(f"[done] wrote status to {os.path.join(cfg['output_dir'], 'train_status.json')}")


def main() -> None:
    configure_unbuffered_output()
    parser = argparse.ArgumentParser(description="Native Sentence Transformers multimodal training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    log_progress(f"[startup] loading config from {args.config}")
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))
    st_version = require_sentence_transformers_version()
    log_progress(f"[startup] sentence-transformers version {st_version}")

    if is_datacenter_tri_encoder_config(cfg):
        log_progress("[startup] detected datacenter tri-encoder config")
        run_datacenter_tri_encoder_training(cfg, args.resume, st_version)
        return

    log_progress("[startup] loading datasets")
    train_dataset, train_info, eval_dataset, eval_info = load_sentence_transformers_datasets(cfg)
    log_progress(
        f"[startup] loaded train dataset with {train_info['num_rows']} rows and modalities {train_info['modalities']}"
    )
    if eval_info is not None:
        log_progress(
            f"[startup] loaded eval dataset with {eval_info['num_rows']} rows and modalities {eval_info['modalities']}"
        )

    log_progress(f"[startup] loading model {cfg['model']['model_name']}")
    model = build_model(cfg)
    validate_modalities(model, train_info)
    if eval_info is not None:
        validate_modalities(model, eval_info)
    log_progress("[startup] model loaded and modalities validated")

    if train_info["num_negatives_present"] and not train_info["has_uniform_negatives"]:
        print(
            "Training manifest has mixed negative availability; dropping negative_0 and training on pairs only.",
            file=sys.stderr,
        )

    resolved_max_steps = resolve_training_max_steps(cfg, train_info)
    if resolved_max_steps is not None:
        cfg.setdefault("training", {})["max_steps"] = resolved_max_steps
        log_progress(f"[startup] resolved max_steps={resolved_max_steps} for iterable training")

    log_progress("[startup] building loss, evaluator, training arguments, and collator")
    loss = build_loss(model, cfg)
    evaluator = build_evaluator(eval_dataset, cfg)
    training_args = build_training_args(cfg)
    data_collator = build_data_collator(model)

    log_progress("[startup] constructing trainer")
    trainer = NativeSentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
        data_collator=data_collator,
    )

    log_progress("[startup] starting training")
    train_result = trainer.train(resume_from_checkpoint=args.resume or None)
    log_progress("[startup] training finished, saving final model")
    trainer.save_model(os.path.join(cfg["output_dir"], "final"))

    metrics = dict(train_result.metrics)
    metrics.update(
        {
            "sentence_transformers_version": st_version,
            "train_rows": train_info["num_rows"],
            "train_modalities": train_info["modalities"],
        }
    )
    if eval_info is not None:
        metrics.update(
            {
                "eval_rows": eval_info["num_rows"],
                "eval_modalities": eval_info["modalities"],
            }
        )
    write_status(cfg["output_dir"], metrics)
    log_progress(f"[done] wrote status to {os.path.join(cfg['output_dir'], 'train_status.json')}")


if __name__ == "__main__":
    main()
