#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from io import BytesIO
from importlib import metadata
from typing import Any, Dict, Optional

import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hf_st_mm.data import manifest_to_sentence_transformers_dataset

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
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
except ImportError:
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

try:
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, MatryoshkaLoss
except ImportError:
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, MatryoshkaLoss

try:
    from sentence_transformers.training_args import BatchSamplers
except ImportError:
    from sentence_transformers.training_args import BatchSamplers

try:
    from sentence_transformers.base.data_collator import BaseDataCollator
except ImportError:
    from sentence_transformers.base.data_collator import BaseDataCollator


class NativeSentenceTransformerTrainer(SentenceTransformerTrainer):
    def add_model_card_callback(self, default_args_dict: Dict[str, Any]) -> None:
        # Sentence Transformers eagerly inspects dataset examples for model-card metadata.
        # In this container, that path trips over torchvision JPEG decoding before training starts.
        # Skipping the callback keeps training functional and does not affect optimization.
        return


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

    loss = CachedMultipleNegativesRankingLoss(
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


def build_training_args(cfg: Dict[str, Any]) -> SentenceTransformerTrainingArguments:
    training_cfg = cfg["training"]
    mixed_precision = str(training_cfg.get("mixed_precision", "bf16")).lower()
    eval_enabled = bool(cfg.get("validation", {}).get("manifest_path"))
    max_steps = training_cfg.get("max_steps")

    args = SentenceTransformerTrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=float(training_cfg["epochs"]),
        max_steps=int(max_steps) if max_steps is not None else -1,
        per_device_train_batch_size=int(training_cfg["batch_size"]),
        per_device_eval_batch_size=int(training_cfg.get("eval_batch_size", training_cfg["batch_size"])),
        gradient_accumulation_steps=int(training_cfg.get("grad_accum_steps", 1)),
        learning_rate=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.0)),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        dataloader_num_workers=int(training_cfg.get("num_workers", 4)),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="steps",
        save_steps=int(training_cfg.get("save_every", 1000)),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        logging_strategy="steps",
        logging_steps=int(training_cfg.get("log_every", 10)),
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=int(training_cfg.get("eval_every", training_cfg.get("save_every", 1000))) if eval_enabled else None,
        fp16=mixed_precision == "fp16",
        bf16=mixed_precision == "bf16",
        seed=int(cfg.get("seed", 42)),
        remove_unused_columns=False,
        run_name=training_cfg.get("run_name"),
    )
    return args


def deserialize_image_payload(payload: Any) -> Any:
    if payload is None or isinstance(payload, Image.Image):
        return payload
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
        return model.preprocess(normalized_inputs, prompt=prompt, task=task)

    return BaseDataCollator(preprocess_fn=preprocess_fn)


def load_sentence_transformers_datasets(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    allowed_modalities = data_cfg.get("allowed_modalities")
    use_iterable_dataset = bool(data_cfg.get("use_iterable_dataset", False))
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
        as_iterable=use_iterable_dataset,
    )

    validation_cfg = cfg.get("validation", {})
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
            as_iterable=use_iterable_dataset,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Native Sentence Transformers multimodal training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))
    st_version = require_sentence_transformers_version()

    train_dataset, train_info, eval_dataset, eval_info = load_sentence_transformers_datasets(cfg)
    model = build_model(cfg)
    validate_modalities(model, train_info)
    if eval_info is not None:
        validate_modalities(model, eval_info)

    if train_info["num_negatives_present"] and not train_info["has_uniform_negatives"]:
        print(
            "Training manifest has mixed negative availability; dropping negative_0 and training on pairs only.",
            file=sys.stderr,
        )

    loss = build_loss(model, cfg)
    evaluator = build_evaluator(eval_dataset, cfg)
    training_args = build_training_args(cfg)
    data_collator = build_data_collator(model)

    trainer = NativeSentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
        data_collator=data_collator,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume or None)
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


if __name__ == "__main__":
    main()
