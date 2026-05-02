from collections import defaultdict
from typing import Any, Dict, List

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor, WhisperFeatureExtractor, WhisperModel

from .data import PairItem


class MultiModalSentenceEmbedder(nn.Module):
    def __init__(
        self,
        text_encoder_name: str,
        image_encoder_name: str,
        audio_encoder_name: str,
        embedding_dim: int,
        max_text_length: int,
    ) -> None:
        super().__init__()
        self.text_model = SentenceTransformer(text_encoder_name)
        self.text_model.max_seq_length = max_text_length

        self.image_model = AutoModel.from_pretrained(image_encoder_name, trust_remote_code=True)
        self.image_processor = AutoProcessor.from_pretrained(image_encoder_name, trust_remote_code=True)

        whisper = WhisperModel.from_pretrained(audio_encoder_name)
        self.audio_model = whisper.encoder
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(audio_encoder_name)

        text_dim = self.text_model.get_sentence_embedding_dimension()
        image_dim = self._get_vision_dim(self.image_model)
        audio_dim = whisper.config.d_model

        self.text_proj = nn.Linear(text_dim, embedding_dim) if text_dim != embedding_dim else nn.Identity()
        self.image_proj = nn.Linear(image_dim, embedding_dim) if image_dim != embedding_dim else nn.Identity()
        self.audio_proj = nn.Linear(audio_dim, embedding_dim) if audio_dim != embedding_dim else nn.Identity()

    @staticmethod
    def _get_vision_dim(model: nn.Module) -> int:
        if hasattr(model, "vision_model") and hasattr(model.config, "vision_config"):
            return int(model.config.vision_config.hidden_size)
        if hasattr(model.config, "hidden_size"):
            return int(model.config.hidden_size)
        raise ValueError("Could not infer image hidden size")

    def _encode_text(self, texts: List[Any]) -> torch.Tensor:
        device = next(self.parameters()).device
        normalized: List[torch.Tensor | None] = [None] * len(texts)

        dict_positions = [idx for idx, item in enumerate(texts) if isinstance(item, dict)]
        if dict_positions:
            pad_values = {
                "input_ids": 0,
                "attention_mask": 0,
                "token_type_ids": 0,
            }
            dict_items = [texts[idx] for idx in dict_positions]
            features = {
                key: pad_sequence(
                    [item[key].detach().cpu() for item in dict_items],
                    batch_first=True,
                    padding_value=pad_values.get(key, 0),
                ).to(device)
                for key in dict_items[0].keys()
            }
            out = self.text_model(features)
            emb = F.normalize(self.text_proj(out["sentence_embedding"]), p=2, dim=-1)
            for loc, row in zip(dict_positions, emb):
                normalized[loc] = row

        raw_positions = [idx for idx, item in enumerate(texts) if not isinstance(item, dict)]
        if raw_positions:
            raw_texts = [texts[idx] for idx in raw_positions]
            features = self.text_model.tokenize(raw_texts)
            features = {
                k: (v.to(device) if hasattr(v, "to") else v)
                for k, v in features.items()
            }
            out = self.text_model(features)
            emb = F.normalize(self.text_proj(out["sentence_embedding"]), p=2, dim=-1)
            for loc, row in zip(raw_positions, emb):
                normalized[loc] = row

        return torch.stack([row for row in normalized if row is not None], dim=0)

    def _encode_image_paths(self, paths: List[str]) -> torch.Tensor:
        images = [Image.open(path).convert("RGB") for path in paths]
        proc = self.image_processor(images=images, return_tensors="pt")
        device = next(self.parameters()).device
        proc = {k: v.to(device) for k, v in proc.items()}
        return self._encode_image_pixel_values(proc["pixel_values"])

    def _encode_image_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        proc = {"pixel_values": pixel_values.to(device)}
        if hasattr(self.image_model, "vision_model"):
            out = self.image_model.vision_model(**proc, output_hidden_states=False)
            hidden = out.last_hidden_state
        else:
            out = self.image_model(**proc, output_hidden_states=False)
            hidden = out.last_hidden_state
        pooled = hidden[:, 1:].mean(dim=1) if hidden.shape[1] > 1 else hidden.mean(dim=1)
        emb = self.image_proj(pooled)
        return F.normalize(emb, p=2, dim=-1)

    def _encode_audio_paths(self, paths: List[str]) -> torch.Tensor:
        waves = [librosa.load(path, sr=16000, mono=True)[0] for path in paths]
        proc = self.audio_processor(waves, sampling_rate=16000, return_tensors="pt")
        return self._encode_audio_features(proc["input_features"])

    def _encode_audio_features(self, input_features: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        input_features = input_features.to(device)
        input_features = input_features.to(self.audio_model.conv1.weight.dtype)
        out = self.audio_model(input_features=input_features, output_hidden_states=False)
        pooled = out.last_hidden_state.mean(dim=1)
        emb = self.audio_proj(pooled)
        return F.normalize(emb, p=2, dim=-1)

    @staticmethod
    def _stack_tensor_values(values: List[Any]) -> torch.Tensor:
        tensors = []
        for value in values:
            if not torch.is_tensor(value):
                raise TypeError("Expected tensor payload in cached item")
            tensor = value.detach().cpu()
            if tensor.dim() > 0 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)

    def encode_items(self, items: List[PairItem]) -> torch.Tensor:
        grouped = defaultdict(list)
        for idx, item in enumerate(items):
            grouped[item.modality].append((idx, item.value))

        device = next(self.parameters()).device
        out = [None] * len(items)

        if grouped["text"]:
            idxs, vals = zip(*grouped["text"])
            embs = self._encode_text(list(vals))
            for loc, emb in zip(idxs, embs):
                out[loc] = emb

        if grouped["image"]:
            idxs, vals = zip(*grouped["image"])
            tensor_pairs = [(idx, val) for idx, val in zip(idxs, vals) if torch.is_tensor(val)]
            path_pairs = [(idx, val) for idx, val in zip(idxs, vals) if not torch.is_tensor(val)]
            if path_pairs:
                p_idxs, p_vals = zip(*path_pairs)
                embs = self._encode_image_paths(list(p_vals))
                for loc, emb in zip(p_idxs, embs):
                    out[loc] = emb
            if tensor_pairs:
                t_idxs, t_vals = zip(*tensor_pairs)
                embs = self._encode_image_pixel_values(self._stack_tensor_values(list(t_vals)))
                for loc, emb in zip(t_idxs, embs):
                    out[loc] = emb

        if grouped["audio"]:
            idxs, vals = zip(*grouped["audio"])
            tensor_pairs = [(idx, val) for idx, val in zip(idxs, vals) if torch.is_tensor(val)]
            path_pairs = [(idx, val) for idx, val in zip(idxs, vals) if not torch.is_tensor(val)]
            if path_pairs:
                p_idxs, p_vals = zip(*path_pairs)
                embs = self._encode_audio_paths(list(p_vals))
                for loc, emb in zip(p_idxs, embs):
                    out[loc] = emb
            if tensor_pairs:
                t_idxs, t_vals = zip(*tensor_pairs)
                embs = self._encode_audio_features(self._stack_tensor_values(list(t_vals)))
                for loc, emb in zip(t_idxs, embs):
                    out[loc] = emb

        stacked = torch.stack(out, dim=0).to(device=device, dtype=torch.float32)
        return F.normalize(stacked, p=2, dim=-1)


def multiple_negatives_ranking_loss(anchor: torch.Tensor, positive: torch.Tensor, scale: float = 20.0) -> torch.Tensor:
    scores = torch.matmul(anchor, positive.T) * scale
    labels = torch.arange(scores.shape[0], device=scores.device)
    loss_a = torch.nn.functional.cross_entropy(scores, labels)
    loss_b = torch.nn.functional.cross_entropy(scores.T, labels)
    return (loss_a + loss_b) * 0.5
