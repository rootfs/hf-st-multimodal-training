import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, Features, Value
from PIL import Image


SUPPORTED_MODALITIES = {"text", "image", "audio"}


@dataclass
class PairItem:
    modality: str
    value: Any


@dataclass
class TrainRecord:
    query: PairItem
    positive: PairItem
    negative: Optional[PairItem] = None


def _parse_item(obj: Any, prefix: str) -> PairItem:
    if isinstance(obj, dict):
        modality = obj.get("type")
        value = obj.get("value")
    else:
        modality = None
        value = None

    if not modality or not value:
        raise ValueError(f"{prefix} must include type/value")
    if modality not in SUPPORTED_MODALITIES:
        raise ValueError(f"Unsupported modality '{modality}' in {prefix}")
    return PairItem(modality=modality, value=value)


def parse_record(raw: Dict[str, Any]) -> TrainRecord:
    if "query" in raw and "positive" in raw:
        query = _parse_item(raw["query"], "query")
        positive = _parse_item(raw["positive"], "positive")
        negative = _parse_item(raw["negative"], "negative") if raw.get("negative") else None
        return TrainRecord(query=query, positive=positive, negative=negative)

    # Compatibility with common pair formats in existing repos
    if "texts_a" in raw and "texts_b" in raw:
        query = PairItem("text", raw["texts_a"])
        positive = PairItem("text", raw["texts_b"])
        return TrainRecord(query=query, positive=positive)

    if "image_path" in raw and "caption" in raw:
        query = PairItem("image", raw["image_path"])
        positive = PairItem("text", raw["caption"])
        return TrainRecord(query=query, positive=positive)

    if "audio_path" in raw and "caption" in raw:
        query = PairItem("audio", raw["audio_path"])
        positive = PairItem("text", raw["caption"])
        return TrainRecord(query=query, positive=positive)

    raise ValueError("Record does not match supported schemas")


class JsonlManifestDataset:
    def __init__(
        self,
        manifest_path: str,
        image_root: Optional[str] = None,
        audio_root: Optional[str] = None,
        allow_missing_negative: bool = True,
    ) -> None:
        self.manifest_path = manifest_path
        self.image_root = image_root
        self.audio_root = audio_root
        self.allow_missing_negative = allow_missing_negative
        self.records = self._load_records()

    def _resolve_media(self, item: PairItem) -> PairItem:
        if item.modality == "image" and self.image_root and not os.path.isabs(item.value):
            return PairItem(item.modality, os.path.join(self.image_root, item.value))
        if item.modality == "audio" and self.audio_root and not os.path.isabs(item.value):
            return PairItem(item.modality, os.path.join(self.audio_root, item.value))
        return item

    def _load_records(self) -> List[TrainRecord]:
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        records: List[TrainRecord] = []
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                rec = parse_record(raw)
                rec = TrainRecord(
                    query=self._resolve_media(rec.query),
                    positive=self._resolve_media(rec.positive),
                    negative=self._resolve_media(rec.negative) if rec.negative else None,
                )
                if rec.negative is None and not self.allow_missing_negative:
                    raise ValueError(f"Missing negative at line {line_no}")
                records.append(rec)
        if not records:
            raise ValueError(f"No records loaded from {self.manifest_path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> TrainRecord:
        return self.records[idx]


class CachedShardDataset:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        self.shard_files = self._discover_shards()
        self.index = self._build_index()
        self._shard_cache: Dict[int, List[Dict[str, Any]]] = {}

    def _discover_shards(self) -> List[str]:
        if not os.path.isdir(self.cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")
        shards: List[str] = []
        for name in sorted(os.listdir(self.cache_dir)):
            if not (name.startswith("shard_") and name.endswith(".pt")):
                continue
            shard_path = os.path.join(self.cache_dir, name)
            try:
                torch.load(shard_path, map_location="cpu", weights_only=False)
            except Exception:
                continue
            shards.append(shard_path)
        if not shards:
            raise ValueError(f"No cache shards found under {self.cache_dir}")
        return shards

    def _build_index(self) -> List[tuple[int, int]]:
        index: List[tuple[int, int]] = []
        for shard_idx, shard_path in enumerate(self.shard_files):
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            records = payload.get("records")
            if not isinstance(records, list):
                raise ValueError(f"Invalid shard format in {shard_path}")
            index.extend((shard_idx, local_idx) for local_idx in range(len(records)))
        return index

    def _load_shard(self, shard_idx: int) -> List[Dict[str, Any]]:
        if shard_idx not in self._shard_cache:
            payload = torch.load(self.shard_files[shard_idx], map_location="cpu", weights_only=False)
            self._shard_cache[shard_idx] = payload["records"]
            if len(self._shard_cache) > 2:
                oldest = min(key for key in self._shard_cache.keys() if key != shard_idx)
                del self._shard_cache[oldest]
        return self._shard_cache[shard_idx]

    @staticmethod
    def _deserialize_item(raw: Optional[Dict[str, Any]]) -> Optional[PairItem]:
        if raw is None:
            return None
        modality = raw["type"]
        if modality == "text" and "tokens" in raw:
            value = raw["tokens"]
        elif modality == "text":
            value = raw["value"]
        elif "tensor" in raw:
            value = raw["tensor"]
        else:
            value = raw.get("value")
        return PairItem(modality=modality, value=value)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> TrainRecord:
        shard_idx, local_idx = self.index[idx]
        raw = self._load_shard(shard_idx)[local_idx]
        return TrainRecord(
            query=self._deserialize_item(raw["query"]),
            positive=self._deserialize_item(raw["positive"]),
            negative=self._deserialize_item(raw.get("negative")),
        )


class RawSentenceTransformersIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        super().__init__()
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []
        self.features = Features({key: Value("null") for key in self.column_names})

    def __iter__(self):
        yield from self.rows

    def __len__(self) -> int:
        return len(self.rows)


def collate_records(batch: List[TrainRecord]) -> Dict[str, List[PairItem]]:
    return {
        "query": [r.query for r in batch],
        "positive": [r.positive for r in batch],
        "negative": [r.negative for r in batch],
    }


def sentence_transformers_input(item: PairItem) -> Any:
    payload: Dict[str, Any] = {}
    if item.modality == "text":
        payload["text"] = item.value
        return payload
    if item.modality == "image":
        with Image.open(item.value) as image:
            payload["image"] = image.convert("RGB").copy()
            return payload
    if item.modality == "audio":
        payload["audio"] = item.value
        return payload
    return item.value


def manifest_to_sentence_transformers_dataset(
    manifest_path: str,
    image_root: Optional[str] = None,
    audio_root: Optional[str] = None,
    allow_missing_negative: bool = True,
    allowed_modalities: Optional[List[str]] = None,
    query_modalities: Optional[List[str]] = None,
    positive_modalities: Optional[List[str]] = None,
    negative_modalities: Optional[List[str]] = None,
    as_iterable: bool = False,
) -> tuple[Dataset | RawSentenceTransformersIterableDataset, Dict[str, Any]]:
    dataset = JsonlManifestDataset(
        manifest_path=manifest_path,
        image_root=image_root,
        audio_root=audio_root,
        allow_missing_negative=allow_missing_negative,
    )

    rows: List[Dict[str, Any]] = []
    modalities = set()
    negatives_present = 0
    negatives_missing = 0
    skipped_rows = 0
    allowed = set(allowed_modalities or [])
    allowed_query = set(query_modalities or [])
    allowed_positive = set(positive_modalities or [])
    allowed_negative = set(negative_modalities or [])

    for record in dataset.records:
        record_modalities = {record.query.modality, record.positive.modality}
        if record.negative is not None:
            record_modalities.add(record.negative.modality)
        if allowed and not record_modalities.issubset(allowed):
            skipped_rows += 1
            continue
        if allowed_query and record.query.modality not in allowed_query:
            skipped_rows += 1
            continue
        if allowed_positive and record.positive.modality not in allowed_positive:
            skipped_rows += 1
            continue
        if record.negative is not None and allowed_negative and record.negative.modality not in allowed_negative:
            skipped_rows += 1
            continue

        row = {
            "query": sentence_transformers_input(record.query),
            "positive": sentence_transformers_input(record.positive),
        }
        modalities.add(record.query.modality)
        modalities.add(record.positive.modality)

        if record.negative is not None:
            row["negative_0"] = sentence_transformers_input(record.negative)
            modalities.add(record.negative.modality)
            negatives_present += 1
        else:
            negatives_missing += 1

        rows.append(row)

    use_negative_column = negatives_present > 0 and negatives_missing == 0
    if negatives_present > 0 and negatives_missing > 0:
        for row in rows:
            row.pop("negative_0", None)

    dataset_out: Dataset | RawSentenceTransformersIterableDataset
    if as_iterable:
        dataset_out = RawSentenceTransformersIterableDataset(rows)
    else:
        dataset_out = Dataset.from_list(rows)

    return dataset_out, {
        "modalities": sorted(modalities),
        "num_rows": len(rows),
        "has_uniform_negatives": use_negative_column,
        "num_negatives_present": negatives_present,
        "num_negatives_missing": negatives_missing,
        "skipped_rows": skipped_rows,
    }
