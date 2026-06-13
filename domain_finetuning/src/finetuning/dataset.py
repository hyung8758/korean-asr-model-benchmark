from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from core.io import read_jsonl
from data.audio import load_audio_array


class WhisperManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        processor,
        project_root: Path,
        sampling_rate: int,
    ) -> None:
        self.rows = read_jsonl(manifest_path)
        if not self.rows:
            raise ValueError(f"Manifest is empty or missing: {manifest_path}")
        self.processor = processor
        self.project_root = project_root
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio = load_audio_array(row, target_sample_rate=self.sampling_rate, project_root=self.project_root)
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
        ).input_features[0]
        labels = self.processor.tokenizer(row["text"]).input_ids
        return {
            "input_features": input_features,
            "labels": labels,
            "id": row["id"],
        }


class WhisperDataCollator:
    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": item["input_features"]} for item in features]
        label_features = [{"input_ids": item["labels"]} for item in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        decoder_start_token_id = self.processor.tokenizer.bos_token_id
        if decoder_start_token_id is not None and (labels[:, 0] == decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
