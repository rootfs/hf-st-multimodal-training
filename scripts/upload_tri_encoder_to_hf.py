#!/usr/bin/env python3

import argparse
import configparser
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from evaluate_tri_encoder import build_eval_loader, load_config, load_final_weights
from train_st_multimodal import (
    build_datacenter_tri_encoder_model,
    evaluate_tri_encoder_model,
    normalize_mixed_precision,
)


def ensure_hf_hub_installed() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"]) 


def read_token(token: Optional[str], token_name: str) -> str:
    if token:
        return token

    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token

    token_store = Path.home() / ".cache" / "huggingface" / "stored_tokens"
    parser = configparser.ConfigParser()
    parser.read(token_store)
    if token_name in parser and "hf_token" in parser[token_name]:
        return parser[token_name]["hf_token"].strip()

    raise ValueError(
        "No Hugging Face token found. Pass --token, set HF_TOKEN, or store it in ~/.cache/huggingface/stored_tokens."
    )


def evaluate_final_export(final_dir: str, max_samples: Optional[int]) -> Dict[str, float]:
    config_path = os.path.join(final_dir, "config.json")
    cfg = load_config(config_path)
    model = build_datacenter_tri_encoder_model(cfg)
    load_final_weights(model, final_dir)

    from accelerate import Accelerator

    accelerator = Accelerator(
        mixed_precision=normalize_mixed_precision(cfg.get("training", {}).get("mixed_precision", "bf16"))
    )
    eval_loader = build_eval_loader(cfg, max_samples=max_samples)
    model, eval_loader = accelerator.prepare(model, eval_loader)

    metrics = evaluate_tri_encoder_model(
        model,
        eval_loader,
        accelerator,
        float(cfg.get("loss", {}).get("scale", 20.0)),
    )
    accelerator.wait_for_everyone()
    return metrics


def render_model_card(template_path: str, repo_id: str, cfg: Dict[str, Any], metrics: Dict[str, float]) -> str:
    with open(template_path, "r", encoding="utf-8") as handle:
        template = handle.read()

    model_cfg = cfg["model"]
    replacements = {
        "__MODEL_NAME__": repo_id.split("/")[-1],
        "__REPO_ID__": repo_id,
        "__TEXT_ENCODER_NAME__": str(model_cfg["text_encoder_name"]),
        "__IMAGE_ENCODER_NAME__": str(model_cfg["image_encoder_name"]),
        "__AUDIO_ENCODER_NAME__": str(model_cfg["audio_encoder_name"]),
        "__EMBEDDING_DIM__": str(model_cfg["embedding_dim"]),
        "__MAX_TEXT_LENGTH__": str(model_cfg["max_text_length"]),
        "__EVAL_LOSS__": f"{metrics.get('eval_loss', float('nan')):.6f}",
        "__EVAL_TOP1__": f"{metrics.get('eval_top1', float('nan')):.6f}",
    }

    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def prepare_upload_bundle(final_dir: str, repo_id: str, metrics: Dict[str, float], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(final_dir, "config.json")
    model_path = os.path.join(final_dir, "model.pt")
    cfg = load_config(config_path)

    shutil.copy2(model_path, os.path.join(output_dir, "model.pt"))
    shutil.copy2(config_path, os.path.join(output_dir, "config.json"))

    template_path = os.path.join(PROJECT_ROOT, "MODEL_CARD_MULTI_MODAL_EMBED_LARGE.template.md")
    readme = render_model_card(template_path, repo_id, cfg, metrics)
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(readme)

    dst_src = os.path.join(output_dir, "src")
    if os.path.exists(dst_src):
        shutil.rmtree(dst_src)
    shutil.copytree(SRC_ROOT, dst_src)

    return output_dir


def upload_folder(folder_path: str, repo_id: str, token: str, commit_message: str) -> None:
    ensure_hf_hub_installed()
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    HfApi().upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message=commit_message,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the final tri-encoder export to Hugging Face Hub")
    parser.add_argument(
        "--final-dir",
        default="/scratch/hf_st_mm_outputs/server_datacenter_8gpu_tri_encoder/final",
        help="Final export directory containing model.pt and config.json",
    )
    parser.add_argument(
        "--repo-id",
        default="llm-semantic-router/multi-modal-embed-large",
        help="Destination Hugging Face model repository",
    )
    parser.add_argument("--token", default=None, help="Hugging Face token")
    parser.add_argument(
        "--token-name",
        default="model training",
        help="Entry name in ~/.cache/huggingface/stored_tokens when --token is not provided",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optionally evaluate only the first N validation samples")
    parser.add_argument("--min-eval-top1", type=float, default=None, help="Optional minimum eval_top1 required before upload")
    parser.add_argument("--max-eval-loss", type=float, default=None, help="Optional maximum eval_loss allowed before upload")
    parser.add_argument("--skip-eval", action="store_true", help="Upload without running final evaluation")
    parser.add_argument("--skip-upload", action="store_true", help="Prepare the upload bundle but do not upload it")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the prepared upload bundle. Defaults to a temporary directory.",
    )
    args = parser.parse_args()

    final_dir = os.path.abspath(args.final_dir)
    if not os.path.exists(os.path.join(final_dir, "model.pt")):
        raise FileNotFoundError(f"Final model weights not found under {final_dir}")
    if not os.path.exists(os.path.join(final_dir, "config.json")):
        raise FileNotFoundError(f"Final config not found under {final_dir}")

    metrics: Dict[str, float] = {}
    if not args.skip_eval:
        metrics = evaluate_final_export(final_dir, args.max_samples)
        if args.min_eval_top1 is not None and metrics.get("eval_top1", float("-inf")) < args.min_eval_top1:
            raise ValueError(
                f"Refusing upload because eval_top1={metrics.get('eval_top1')} is below required {args.min_eval_top1}"
            )
        if args.max_eval_loss is not None and metrics.get("eval_loss", float("inf")) > args.max_eval_loss:
            raise ValueError(
                f"Refusing upload because eval_loss={metrics.get('eval_loss')} exceeds allowed {args.max_eval_loss}"
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.abspath(args.output_dir) if args.output_dir else temp_dir
        bundle_dir = prepare_upload_bundle(final_dir, args.repo_id, metrics, output_dir)
        print(json.dumps({"prepared": True, "bundle_dir": bundle_dir, **metrics}, indent=2, sort_keys=True))

        if args.skip_upload:
            return

        token = read_token(args.token, args.token_name)
        upload_folder(bundle_dir, args.repo_id, token, commit_message=f"Upload {args.repo_id.split('/')[-1]} final model")
        print(json.dumps({"uploaded": True, "repo_id": args.repo_id, **metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()