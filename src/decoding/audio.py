import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torchaudio

from data.audio import (
    TARGET_SAMPLE_RATE,
    cut_audio_segment,
    has_audio_segment,
    load_audio,
    resample_audio,
)


def needs_temporary_wav(item: dict[str, Any]) -> bool:
    audio_path = Path(item["audio"])
    sample_rate = item.get("audio_sample_rate", item.get("source_sample_rate"))
    return (
        has_audio_segment(item)
        or audio_path.suffix.lower() != ".wav"
        or (sample_rate is not None and int(sample_rate) != TARGET_SAMPLE_RATE)
    )


@contextmanager
def prepared_audio_path(
    item: dict[str, Any],
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> Iterator[Path]:
    audio_path = Path(item["audio"])
    if not needs_temporary_wav(item):
        yield audio_path
        return

    waveform, sample_rate = load_audio(audio_path)
    waveform = cut_audio_segment(waveform, sample_rate, item)
    waveform = resample_audio(waveform, sample_rate, target_sample_rate)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / f"{item['id']}.wav"
        torchaudio.save(temp_path, waveform, target_sample_rate, encoding="PCM_S", bits_per_sample=16)
        yield temp_path
