#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
from typing import Iterable, List, Optional
import zipfile

from datasets import Audio, concatenate_datasets, load_dataset
import requests
import soundfile as sf
from tqdm import tqdm


LLAVA_CHAT_URL = "https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/chat.json"
LLAVA_IMAGES_URL = "https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/images.zip"
LLAVA_READY_THRESHOLD = 500000

AUDIO_DATASET_CONFIGS = {
    "librispeech_clean": {
        "path": "openslr/librispeech_asr",
        "config": "clean",
        "splits": ["train.100", "train.360"],
        "audio_col": "audio",
        "text_col": "text",
        "prefix": "librispeech",
        "optional": False,
    },
    "librispeech_other": {
        "path": "openslr/librispeech_asr",
        "config": "other",
        "splits": ["train.500"],
        "audio_col": "audio",
        "text_col": "text",
        "prefix": "librispeech_other",
        "optional": False,
    },
    "voxpopuli": {
        "path": "facebook/voxpopuli",
        "config": "en",
        "splits": ["train"],
        "audio_col": "audio",
        "text_col": "raw_text",
        "prefix": "voxpopuli",
        "optional": False,
    },
    "tedlium": {
        "path": "LIUM/tedlium",
        "config": "release3",
        "splits": ["train"],
        "audio_col": "audio",
        "text_col": "text",
        "prefix": "tedlium",
        "optional": True,
    },
    "peoples_speech": {
        "path": "MLCommons/peoples_speech",
        "config": "clean",
        "splits": ["train"],
        "audio_col": "audio",
        "text_col": "text",
        "prefix": "peoples_speech",
        "optional": True,
    },
    "common_voice": {
        "path": "mozilla-foundation/common_voice_17_0",
        "config": "en",
        "splits": ["train"],
        "audio_col": "audio",
        "text_col": "sentence",
        "prefix": "common_voice",
        "optional": True,
    },
    "gigaspeech": {
        "path": "speechcolab/gigaspeech",
        "config": "xs",
        "splits": ["train"],
        "audio_col": "audio",
        "text_col": "text",
        "prefix": "gigaspeech_xs",
        "optional": True,
    },
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, records) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def append_jsonl(path: str, records) -> None:
    if not records:
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def count_image_files(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(1 for name in os.listdir(path) if name.lower().endswith((".jpg", ".jpeg", ".png")))


def download_with_resume(url: str, output_path: str, desc: str) -> None:
    existing_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}
    response = requests.get(url, stream=True, headers=headers, timeout=60)
    response.raise_for_status()

    mode = "ab"
    total = existing_size + int(response.headers.get("content-length", 0) or 0)
    if existing_size > 0 and response.status_code != 206:
        mode = "wb"
        existing_size = 0
        total = int(response.headers.get("content-length", 0) or 0)

    with open(output_path, mode) as handle:
        with tqdm(total=total or None, initial=existing_size, unit="B", unit_scale=True, desc=desc) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))


def pick_existing_path(candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def ensure_llava_cc3m(output_root: str, llava_root: Optional[str] = None) -> str:
    shared_root = "/scratch/2dmse-data/llava-cc3m-595k"
    dataset_root = pick_existing_path(
        [
            llava_root,
            os.path.join(output_root, "llava-cc3m-595k"),
            "/scratch/2dmse-data/server/llava-cc3m-595k",
            shared_root,
        ]
    ) or shared_root
    ensure_dir(dataset_root)

    chat_file = os.path.join(dataset_root, "chat.json")
    images_dir = os.path.join(dataset_root, "images")
    images_zip = os.path.join(dataset_root, "images.zip")

    num_images = max(count_image_files(images_dir), count_image_files(dataset_root))
    if not os.path.exists(chat_file):
        download_with_resume(LLAVA_CHAT_URL, chat_file, "llava chat.json")

    if num_images < LLAVA_READY_THRESHOLD:
        if not os.path.exists(images_zip) or os.path.getsize(images_zip) == 0:
            download_with_resume(LLAVA_IMAGES_URL, images_zip, "llava images.zip")
        elif num_images == 0:
            # Keep and reuse any existing partial zip instead of discarding it.
            download_with_resume(LLAVA_IMAGES_URL, images_zip, "llava images.zip")

        with zipfile.ZipFile(images_zip, "r") as archive:
            archive.extractall(dataset_root)
        num_images = max(count_image_files(images_dir), count_image_files(dataset_root))

    if not os.path.exists(chat_file) or num_images < LLAVA_READY_THRESHOLD:
        raise RuntimeError(f"LLaVA-CC3M is incomplete at {dataset_root}: chat={os.path.exists(chat_file)} images={num_images}")

    return dataset_root


def build_text_records(num_samples: int):
    dataset = load_dataset(
        "glue",
        "qqp",
        split="train",
    )
    records = []
    for item in dataset:
        if int(item.get("label", 0)) != 1:
            continue
        record = {
            "query": {"type": "text", "value": item["question1"]},
            "positive": {"type": "text", "value": item["question2"]},
        }
        records.append(record)
        if num_samples > 0 and len(records) >= num_samples:
            break
    for idx, record in enumerate(records):
        if len(records) > 1:
            record["negative"] = {
                "type": "text",
                "value": records[(idx + 1) % len(records)]["positive"]["value"],
            }
    return records


def build_coco_records(num_samples: int):
    split = "train" if num_samples <= 0 else f"train[:{num_samples}]"
    dataset = load_dataset(
        "HuggingFaceM4/COCO",
        split=split,
        trust_remote_code=True,
    )

    image_root = None
    if len(dataset) > 0:
        sample_image = dataset[0].get("image")
        sample_path = getattr(sample_image, "filename", None)
        if sample_path:
            image_root = os.path.dirname(sample_path)
    if "image" in dataset.column_names:
        dataset = dataset.remove_columns(["image"])

    rows = []
    for idx, item in enumerate(dataset):
        sentences = item.get("sentences") or []
        caption = ""
        if isinstance(sentences, dict):
            caption = sentences.get("raw", "")
        elif sentences:
            first = sentences[0]
            caption = first.get("raw", "") if isinstance(first, dict) else str(first)
        if not caption:
            continue
        filename = item.get("filename") or item.get("filepath")
        abs_path = os.path.join(image_root, filename) if image_root and filename else None
        if abs_path and not os.path.exists(abs_path):
            abs_path = None
        if not abs_path or not os.path.exists(abs_path):
            continue
        rows.append({"path": abs_path, "caption": caption})

    records = []
    for idx, row in enumerate(rows):
        record = {
            "query": {"type": "image", "value": row["path"]},
            "positive": {"type": "text", "value": row["caption"]},
            "source": "coco",
        }
        if len(rows) > 1:
            record["negative"] = {
                "type": "text",
                "value": rows[(idx + 1) % len(rows)]["caption"],
            }
        records.append(record)
    return records


def build_llava_records(output_root: str, num_samples: int, llava_root: Optional[str] = None):
    dataset_root = ensure_llava_cc3m(output_root, llava_root=llava_root)
    chat_file = os.path.join(dataset_root, "chat.json")
    images_dir = os.path.join(dataset_root, "images")

    with open(chat_file, "r", encoding="utf-8") as handle:
        items = json.load(handle)

    rows = []
    for item in items:
        image_name = item.get("image", "")
        if not image_name:
            continue
        image_path = pick_existing_path(
            [
                os.path.join(images_dir, image_name),
                os.path.join(dataset_root, image_name),
            ]
        )
        if not image_path:
            continue
        caption = ""
        for conv in item.get("conversations", []):
            if conv.get("from") == "gpt" and conv.get("value"):
                caption = conv["value"]
                break
        if not caption:
            continue
        rows.append({"path": image_path, "caption": caption})
        if num_samples > 0 and len(rows) >= num_samples:
            break

    records = []
    for idx, row in enumerate(rows):
        record = {
            "query": {"type": "image", "value": row["path"]},
            "positive": {"type": "text", "value": row["caption"]},
            "source": "llava_cc3m",
        }
        if len(rows) > 1:
            record["negative"] = {
                "type": "text",
                "value": rows[(idx + 1) % len(rows)]["caption"],
            }
        records.append(record)
    return records


def _load_audio_dataset(dataset_name: str):
    if dataset_name not in AUDIO_DATASET_CONFIGS:
        raise ValueError(f"Unsupported audio dataset '{dataset_name}'")
    cfg = AUDIO_DATASET_CONFIGS[dataset_name]
    datasets = [
        load_dataset(
            cfg["path"],
            cfg["config"],
            split=split,
            trust_remote_code=True,
        )
        for split in cfg["splits"]
    ]
    dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    return dataset.cast_column(cfg["audio_col"], Audio(sampling_rate=16000)), cfg


def build_audio_records(output_root: str, dataset_name: str, num_samples: int, source_cache: Optional[str] = None):
    dataset, cfg = _load_audio_dataset(dataset_name)
    if num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    audio_root = os.path.join(output_root, "audio")
    ensure_dir(audio_root)

    prefix = cfg["prefix"]
    existing_files = sorted(glob(os.path.join(audio_root, f"{prefix}_*.wav")))
    resume_count = min(len(existing_files), len(dataset))
    checkpoint_every = 2000

    print(f"Building audio source {dataset_name}: dataset_size={len(dataset)} resume_count={resume_count}", flush=True)

    rows = []
    for idx in range(resume_count):
        item = dataset[idx]
        text = item.get(cfg["text_col"], "")
        if not text:
            continue
        abs_path = os.path.join(audio_root, f"{prefix}_{idx:06d}.wav")
        if os.path.exists(abs_path):
            rows.append({"path": abs_path, "text": text})

    if source_cache and rows and not os.path.exists(source_cache):
        write_jsonl(source_cache, [
            {
                "query": {"type": "audio", "value": row["path"]},
                "positive": {"type": "text", "value": row["text"]},
                "source": dataset_name,
            }
            for row in rows
        ])

    pending_records = []
    for idx in range(resume_count, len(dataset)):
        item = dataset[idx]
        audio = item[cfg["audio_col"]]
        text = item.get(cfg["text_col"], "")
        if not text:
            continue
        rel_path = f"{prefix}_{idx:06d}.wav"
        abs_path = os.path.join(audio_root, rel_path)
        sf.write(abs_path, audio["array"], audio["sampling_rate"])
        rows.append({"path": abs_path, "text": text})
        pending_records.append(
            {
                "query": {"type": "audio", "value": abs_path},
                "positive": {"type": "text", "value": text},
                "source": dataset_name,
            }
        )
        if source_cache and len(pending_records) >= checkpoint_every:
            append_jsonl(source_cache, pending_records)
            print(f"audio source {dataset_name}: materialized {idx + 1}/{len(dataset)}", flush=True)
            pending_records = []

    records = []
    for idx, row in enumerate(rows):
        record = {
            "query": {"type": "audio", "value": row["path"]},
            "positive": {"type": "text", "value": row["text"]},
            "source": dataset_name,
        }
        if len(rows) > 1:
            record["negative"] = {
                "type": "text",
                "value": rows[(idx + 1) % len(rows)]["text"],
            }
        records.append(record)

    if source_cache and pending_records:
        append_jsonl(source_cache, pending_records)
    print(f"Finished audio source {dataset_name}: records={len(records)}", flush=True)
    return records


def interleave_records(*groups):
    merged = []
    max_len = max(len(group) for group in groups if group)
    for idx in range(max_len):
        for group in groups:
            if idx < len(group):
                merged.append(group[idx])
    return merged


def main():
    parser = argparse.ArgumentParser(description="Download real multimodal data for training")
    parser.add_argument("--output-root", default="/scratch/2dmse-data/server")
    parser.add_argument("--num-text", type=int, default=0)
    parser.add_argument("--num-image", type=int, default=0)
    parser.add_argument("--num-image-coco", type=int, default=None)
    parser.add_argument("--num-image-llava", type=int, default=None)
    parser.add_argument("--num-audio", type=int, default=0)
    parser.add_argument(
        "--audio-datasets",
        default="librispeech_clean,librispeech_other,voxpopuli,tedlium",
        help="Comma-separated audio datasets to include",
    )
    parser.add_argument(
        "--optional-audio-datasets",
        default="peoples_speech,common_voice,gigaspeech",
        help="Comma-separated optional audio datasets to try and skip on access/download errors",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument("--skip-llava", action="store_true")
    parser.add_argument("--llava-root", default=None)
    args = parser.parse_args()

    output_root = args.output_root
    ensure_dir(output_root)

    text_cache = os.path.join(output_root, "text_records.jsonl")
    image_cache = os.path.join(output_root, "image_records.jsonl")
    image_coco_cache = os.path.join(output_root, "image_coco_records.jsonl")
    image_llava_cache = os.path.join(output_root, "image_llava_records.jsonl")
    audio_cache = os.path.join(output_root, "audio_records.jsonl")
    audio_counts: Dict[str, int] = {}
    legacy_image_cache = image_cache

    if args.skip_text:
        text_records = load_jsonl(text_cache) if os.path.exists(text_cache) else []
    else:
        text_records = load_jsonl(text_cache) if os.path.exists(text_cache) else build_text_records(args.num_text)
        if not os.path.exists(text_cache):
            write_jsonl(text_cache, text_records)

    num_image_coco = args.num_image if args.num_image_coco is None else args.num_image_coco
    num_image_llava = args.num_image if args.num_image_llava is None else args.num_image_llava

    if args.skip_image:
        coco_records = load_jsonl(image_coco_cache) if os.path.exists(image_coco_cache) else []
        llava_records = load_jsonl(image_llava_cache) if os.path.exists(image_llava_cache) else []
    else:
        if os.path.exists(image_coco_cache):
            coco_records = load_jsonl(image_coco_cache)
        elif os.path.exists(legacy_image_cache):
            coco_records = load_jsonl(legacy_image_cache)
            write_jsonl(image_coco_cache, coco_records)
        else:
            coco_records = build_coco_records(num_image_coco)
            write_jsonl(image_coco_cache, coco_records)

        llava_records = []
        if not args.skip_llava:
            if os.path.exists(image_llava_cache):
                llava_records = load_jsonl(image_llava_cache)
            else:
                llava_records = build_llava_records(output_root, num_image_llava, llava_root=args.llava_root)
                write_jsonl(image_llava_cache, llava_records)

    image_records = coco_records + llava_records
    if not args.skip_image:
        write_jsonl(image_cache, image_records)

    requested_audio = [name.strip() for name in args.audio_datasets.split(",") if name.strip()]
    optional_audio = {name.strip() for name in args.optional_audio_datasets.split(",") if name.strip()}
    legacy_audio_cache = os.path.join(output_root, "audio_records.jsonl")
    if args.skip_audio:
        # Bootstrap phases may skip both audio processing and finalization.
        # In that case, the merged audio cache is irrelevant and may be stale.
        if args.skip_finalize:
            audio_records = []
        else:
            audio_records = load_jsonl(audio_cache) if os.path.exists(audio_cache) else []
    else:
        audio_records = []
        for dataset_name in requested_audio:
            source_cache = os.path.join(output_root, f"audio_{dataset_name}_records.jsonl")
            if os.path.exists(source_cache):
                source_records = load_jsonl(source_cache)
            elif dataset_name == "librispeech_clean" and os.path.exists(legacy_audio_cache):
                source_records = load_jsonl(legacy_audio_cache)
                write_jsonl(source_cache, source_records)
            else:
                try:
                    source_records = build_audio_records(output_root, dataset_name, args.num_audio, source_cache=source_cache)
                except Exception as exc:
                    if dataset_name in optional_audio or AUDIO_DATASET_CONFIGS.get(dataset_name, {}).get("optional"):
                        print(f"Skipping optional audio dataset {dataset_name}: {exc}")
                        continue
                    raise
                write_jsonl(source_cache, source_records)
            audio_counts[dataset_name] = len(source_records)
            audio_records.extend(source_records)

        write_jsonl(audio_cache, audio_records)

    if args.skip_finalize:
        return

    records = interleave_records(text_records, image_records, audio_records)

    split_idx = max(1, int(len(records) * (1.0 - args.val_fraction)))
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    write_jsonl(os.path.join(output_root, "train_manifest.jsonl"), train_records)
    write_jsonl(os.path.join(output_root, "val_manifest.jsonl"), val_records)

    summary = {
        "output_root": output_root,
        "num_train": len(train_records),
        "num_val": len(val_records),
        "num_text": len(text_records),
        "num_image": len(image_records),
        "num_image_coco": len(coco_records),
        "num_image_llava": len(llava_records),
        "num_audio": len(audio_records),
        "audio_sources": audio_counts,
    }
    with open(os.path.join(output_root, "download_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
