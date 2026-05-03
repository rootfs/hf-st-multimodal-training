#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict

import torch
import yaml
from accelerate import Accelerator
from safetensors.torch import load_file as safetensors_load_file
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

from hf_st_mm.data import CachedShardDataset, collate_records
from train_st_multimodal import (
    _build_cached_loader_kwargs,
    build_datacenter_tri_encoder_model,
    evaluate_tri_encoder_model,
    normalize_mixed_precision,
)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        if config_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(handle)
        return json.load(handle)


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.config:
        return load_config(args.config)

    if args.final_dir:
        config_path = os.path.join(args.final_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No config.json found under final dir {args.final_dir}. Pass --config explicitly for checkpoint eval."
            )
        return load_config(config_path)

    raise ValueError("--config is required when evaluating a checkpoint directory.")


def load_final_weights(model: torch.nn.Module, final_dir: str) -> None:
    model_path = os.path.join(final_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Final model weights not found: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_dir: str) -> None:
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        state_dict = safetensors_load_file(safetensors_path, device="cpu")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No model weights found under {checkpoint_dir}. Expected model.safetensors or pytorch_model.bin."
        )
    model.load_state_dict(state_dict)


def build_eval_loader(cfg: Dict[str, Any], max_samples: int | None = None) -> DataLoader:
    validation_cfg = cfg.get("validation", {})
    cache_dir = validation_cfg.get("cache_dir")
    if not cache_dir:
        raise ValueError("validation.cache_dir is required for tri-encoder evaluation.")

    eval_dataset = CachedShardDataset(
        cache_dir,
        shard_cache_limit=int(validation_cfg.get("shard_cache_limit", 2)),
        prefetch_shards=int(validation_cfg.get("shard_prefetch", 1)),
    )
    if max_samples is not None:
        eval_dataset = Subset(eval_dataset, range(min(max_samples, len(eval_dataset))))
    training_cfg = cfg.get("training", {})
    return DataLoader(
        eval_dataset,
        batch_size=int(validation_cfg.get("batch_size", training_cfg.get("batch_size", 1))),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_records,
        **_build_cached_loader_kwargs(
            int(validation_cfg.get("num_workers", max(1, int(training_cfg.get("num_workers", 4)) // 2))),
            int(training_cfg.get("prefetch_factor", 4)),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate datacenter tri-encoder checkpoint or final model")
    parser.add_argument("--config", default=None, help="Path to training config YAML or JSON")
    parser.add_argument("--checkpoint-dir", default=None, help="Accelerate checkpoint directory, e.g. checkpoint-2000")
    parser.add_argument("--final-dir", default=None, help="Final export directory containing model.pt and config.json")
    parser.add_argument("--max-samples", type=int, default=None, help="Optionally evaluate only the first N validation samples")
    args = parser.parse_args()

    if bool(args.checkpoint_dir) == bool(args.final_dir):
        raise ValueError("Provide exactly one of --checkpoint-dir or --final-dir.")

    cfg = resolve_config(args)
    model = build_datacenter_tri_encoder_model(cfg)

    if args.final_dir:
        load_final_weights(model, args.final_dir)
    else:
        load_checkpoint_weights(model, args.checkpoint_dir)

    accelerator = Accelerator(mixed_precision=normalize_mixed_precision(cfg.get("training", {}).get("mixed_precision", "bf16")))
    eval_loader = build_eval_loader(cfg, max_samples=args.max_samples)
    model, eval_loader = accelerator.prepare(model, eval_loader)

    metrics = evaluate_tri_encoder_model(
        model,
        eval_loader,
        accelerator,
        float(cfg.get("loss", {}).get("scale", 20.0)),
    )
    if accelerator.is_main_process:
        payload = {
            "mode": "final" if args.final_dir else "checkpoint",
            "path": args.final_dir or args.checkpoint_dir,
            **metrics,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()