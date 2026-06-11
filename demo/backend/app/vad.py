import os
import threading
from dataclasses import dataclass

import torch


TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class VadSegment:
    index: int
    start: float
    end: float
    pcm: bytes


@dataclass(frozen=True)
class VadConfig:
    padding_ms: int = 300
    min_speech_ms: int = 250
    min_silence_ms: int = 500
    threshold: float = 0.5


class SileroVad:
    _model = None
    _model_lock = threading.Lock()

    def __init__(self, config: VadConfig | None = None):
        self.config = config or VadConfig()
        self.pcm_parts: list[bytes] = []
        self.settings: tuple[int, int, int] | None = None
        self.total_bytes = 0
        self.processed_until_sample = 0
        self.segment_index = 0

    @classmethod
    def load_model(cls):
        with cls._model_lock:
            if cls._model is None:
                from silero_vad import load_silero_vad

                cls._model = load_silero_vad()
            return cls._model

    def append(self, frames: bytes, settings: tuple[int, int, int]) -> None:
        if self.settings is None:
            self.settings = settings
        elif self.settings != settings:
            raise ValueError("VAD 입력 오디오 형식이 중간에 변경되었습니다.")
        self.pcm_parts.append(frames)
        self.total_bytes += len(frames)

    def pop_final_segments(self, force: bool = False) -> list[VadSegment]:
        timestamps = self._speech_timestamps()
        total_samples = self._total_samples()
        finalize_before = total_samples if force else total_samples - self._samples_from_ms(self.config.min_silence_ms)
        segments = []
        for timestamp in timestamps:
            start = int(timestamp["start"])
            end = int(timestamp["end"])
            if end <= self.processed_until_sample or end > finalize_before:
                continue
            segments.append(self._make_segment(start, end))
            self.processed_until_sample = max(self.processed_until_sample, end)
        return segments

    def current_speech_segment(self) -> VadSegment | None:
        timestamps = self._speech_timestamps()
        for timestamp in reversed(timestamps):
            start = int(timestamp["start"])
            end = int(timestamp["end"])
            if end <= self.processed_until_sample:
                continue
            return self._make_segment(start, self._total_samples(), preview=True)
        return None

    def _speech_timestamps(self) -> list[dict[str, int]]:
        if not self.settings or not self.total_bytes:
            return []
        channels, sample_width, sample_rate = self.settings
        if channels != 1 or sample_width != 2 or sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError("Silero VAD는 16kHz mono 16-bit PCM 입력만 지원합니다.")

        from silero_vad import get_speech_timestamps

        return get_speech_timestamps(
            self._audio_tensor(),
            self.load_model(),
            sampling_rate=TARGET_SAMPLE_RATE,
            threshold=self.config.threshold,
            min_speech_duration_ms=self.config.min_speech_ms,
            min_silence_duration_ms=self.config.min_silence_ms,
            speech_pad_ms=self.config.padding_ms,
            return_seconds=False,
        )

    def _audio_tensor(self) -> torch.Tensor:
        data = torch.frombuffer(bytearray(self._pcm_bytes()), dtype=torch.int16)
        return data.to(torch.float32) / 32768.0

    def _make_segment(self, start_sample: int, end_sample: int, preview: bool = False) -> VadSegment:
        start_sample = max(0, start_sample)
        end_sample = min(self._total_samples(), max(start_sample, end_sample))
        pcm = self._slice_pcm(start_sample, end_sample)
        index = self.segment_index
        if not preview:
            self.segment_index += 1
        return VadSegment(
            index=index,
            start=start_sample / TARGET_SAMPLE_RATE,
            end=end_sample / TARGET_SAMPLE_RATE,
            pcm=pcm,
        )

    def _slice_pcm(self, start_sample: int, end_sample: int) -> bytes:
        if not self.settings:
            return b""
        channels, sample_width, _sample_rate = self.settings
        frame_size = channels * sample_width
        start_byte = start_sample * frame_size
        end_byte = end_sample * frame_size
        return self._pcm_bytes()[start_byte:end_byte]

    def _pcm_bytes(self) -> bytes:
        return b"".join(self.pcm_parts)

    def _total_samples(self) -> int:
        if not self.settings:
            return 0
        channels, sample_width, _sample_rate = self.settings
        frame_size = channels * sample_width
        return self.total_bytes // frame_size

    @staticmethod
    def _samples_from_ms(milliseconds: int) -> int:
        return int(TARGET_SAMPLE_RATE * milliseconds / 1000)


def create_vad(vad_id: str, settings: dict | None = None) -> SileroVad:
    if vad_id != "silero":
        raise ValueError(f"지원하지 않는 VAD: {vad_id}")
    values = settings or {}
    return SileroVad(
        VadConfig(
            padding_ms=int(os.getenv("DEMO_VAD_PADDING_MS", values.get("padding_ms", 300))),
            min_speech_ms=int(os.getenv("DEMO_VAD_MIN_SPEECH_MS", values.get("min_speech_ms", 250))),
            min_silence_ms=int(os.getenv("DEMO_VAD_MIN_SILENCE_MS", values.get("min_silence_ms", 500))),
            threshold=float(os.getenv("DEMO_VAD_THRESHOLD", values.get("threshold", 0.5))),
        )
    )
