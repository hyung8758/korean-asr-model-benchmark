from dataclasses import replace
from pathlib import Path
from typing import Any

import torchaudio
from tqdm import tqdm

from data.audio import display_path, load_audio, resample_audio, safe_id
from data.corpora import Candidate, archive_root_for


DEFAULT_SAMPLE_RATE = 16000


def split_audio_root_for(data_root: Path) -> Path:
    return archive_root_for(data_root) / "splits"


def needs_split_audio(candidate: Candidate) -> bool:
    return candidate.start_sec is not None and candidate.end_sec is not None


def split_audio_path(split_root: Path, candidate: Candidate) -> Path:
    return split_root / candidate.dataset / f"{safe_id(candidate.dataset, candidate.utt_id)}.wav"


def cut_segment(candidate: Candidate, waveform, sample_rate: int):
    start_frame = max(0, int(round(float(candidate.start_sec) * sample_rate)))
    end_frame = min(waveform.size(1), int(round(float(candidate.end_sec) * sample_rate)))
    return waveform[:, start_frame:end_frame]


def split_candidate(candidate: Candidate, split_path: Path) -> Candidate:
    return replace(
        candidate,
        audio_path=split_path,
        start_sec=None,
        end_sec=None,
        original_audio_path=candidate.audio_path,
        original_start_sec=candidate.start_sec,
        original_end_sec=candidate.end_sec,
    )


def make_drop_row(candidate: Candidate, project_root: Path, reason: str, error: str) -> dict[str, Any]:
    return {
        "id": candidate.utt_id,
        "dataset": candidate.dataset,
        "source_audio": display_path(candidate.audio_path, project_root),
        "source_text": display_path(candidate.source_text, project_root),
        "reason": reason,
        "error": error,
    }


def materialize_split_audio(
    waveform,
    original_sample_rate: int,
    candidate: Candidate,
    target_path: Path,
    sample_rate: int,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    segment = cut_segment(candidate, waveform, original_sample_rate)
    segment = resample_audio(segment, original_sample_rate, sample_rate)
    torchaudio.save(target_path, segment, sample_rate, encoding="PCM_S", bits_per_sample=16)


def prepare_split_audio_cache(
    candidates: list[Candidate],
    data_root: Path,
    project_root: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    overwrite: bool = False,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    split_root = split_audio_root_for(data_root)
    prepared = []
    dropped = []
    cached_audio_path: Path | None = None
    cached_waveform = None
    cached_sample_rate: int | None = None

    for candidate in tqdm(candidates, desc="Preparing split audio cache"):
        if not needs_split_audio(candidate):
            prepared.append(candidate)
            continue

        target_path = split_audio_path(split_root, candidate)
        try:
            if not (target_path.exists() and target_path.stat().st_size > 0 and not overwrite):
                if cached_audio_path != candidate.audio_path:
                    cached_waveform, cached_sample_rate = load_audio(candidate.audio_path)
                    cached_audio_path = candidate.audio_path
                materialize_split_audio(
                    waveform=cached_waveform,
                    original_sample_rate=cached_sample_rate,
                    candidate=candidate,
                    target_path=target_path,
                    sample_rate=sample_rate,
                )
            prepared.append(split_candidate(candidate, target_path))
        except Exception as exc:
            if cached_audio_path == candidate.audio_path:
                cached_audio_path = None
                cached_waveform = None
                cached_sample_rate = None
            dropped.append(
                make_drop_row(
                    candidate=candidate,
                    project_root=project_root,
                    reason="split_audio_prepare_failed",
                    error=str(exc),
                )
            )

    return prepared, dropped
