import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from data.corpora import Candidate


TARGET_SAMPLE_RATE = 16000


def prepare_output_dir(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)


def display_path(path: Path, project_root: Path | None = None) -> str:
    return str(path.resolve())


def write_jsonl_line(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def record_drop(handle, counter: Counter, base: dict[str, Any], reason: str, **extra: object) -> None:
    counter[reason] += 1
    write_jsonl_line(handle, {**base, "reason": reason, **extra})


def write_drop_rows(handle, counter: Counter, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        reason = str(row.get("reason") or "unknown")
        counter[reason] += 1
        write_jsonl_line(handle, row)


def bucket_for_duration(duration: float) -> str:
    if duration < 15.0:
        return "short"
    if duration < 300.0:
        return "mid"
    return "long"


def safe_id(dataset: str, utt_id: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", utt_id).strip("_")
    return f"{dataset}_{cleaned}"


def mono_waveform(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def load_audio(audio_path: Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = mono_waveform(waveform)
    return waveform, sample_rate


def check_audio_runtime(candidates: list[Candidate]) -> None:
    for candidate in candidates:
        try:
            load_audio(candidate.audio_path)
            return
        except Exception as exc:
            message = str(exc)
            if "TorchCodec is required" in message or "torchcodec" in message.lower():
                raise RuntimeError(
                    "torchaudio.load requires torchcodec in this environment. "
                    "Install dependencies with: pip install -r requirements.txt"
                ) from exc

    raise RuntimeError("Could not load any parsed audio sample with torchaudio.load.")


def resample_audio(waveform: torch.Tensor, original_sr: int, target_sr: int) -> torch.Tensor:
    if original_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(waveform, original_sr, target_sr)


def has_audio_segment(item: dict[str, Any]) -> bool:
    return item.get("audio_start") is not None and item.get("audio_end") is not None


def cut_audio_segment(waveform: torch.Tensor, sample_rate: int, item: dict[str, Any]) -> torch.Tensor:
    if not has_audio_segment(item):
        return waveform
    start = max(0, int(round(float(item["audio_start"]) * sample_rate)))
    end = min(waveform.size(1), int(round(float(item["audio_end"]) * sample_rate)))
    return waveform[:, start:end]


def resolve_audio_path(item: dict[str, Any], project_root: Path | None = None) -> Path:
    audio_path = Path(str(item["audio"]))
    if audio_path.is_absolute() or project_root is None:
        return audio_path
    return project_root / audio_path


def load_audio_array(
    item: dict[str, Any],
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    project_root: Path | None = None,
) -> np.ndarray:
    audio_path = resolve_audio_path(item, project_root)
    waveform, sample_rate = load_audio(audio_path)
    waveform = cut_audio_segment(waveform, sample_rate, item)
    waveform = resample_audio(waveform, sample_rate, target_sample_rate)
    return waveform.squeeze(0).detach().cpu().numpy()
