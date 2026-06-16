import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from app.audio import TARGET_SAMPLE_RATE
from app.engine_models import EngineSpec, TranscriptionResult
from app.engine_transcribers import load_whisper_streaming_classes


ModelGetter = Callable[[EngineSpec], tuple[Any, float]]
NATIVE_STREAMING_KINDS = {"whisper_streaming"}


class StreamingSession(Protocol):
    engine_id: str

    def insert_pcm(self, pcm: bytes, settings: tuple[int, int, int]) -> TranscriptionResult:
        ...

    def finish(self) -> TranscriptionResult:
        ...


@dataclass
class WhisperStreamingSession:
    spec: EngineSpec
    processor: Any
    asr: Any
    model_load_time: float = 0.0
    samples_since_process: int = 0
    committed_segments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def engine_id(self) -> str:
        return self.spec.id

    def insert_pcm(self, pcm: bytes, settings: tuple[int, int, int]) -> TranscriptionResult:
        samples = pcm_bytes_to_float32(pcm, settings)
        start = time.perf_counter()
        self.processor.insert_audio_chunk(samples)
        self.samples_since_process += len(samples)
        if self.samples_since_process < self.min_process_samples:
            return self.result_from_outputs([], start)
        self.samples_since_process = 0
        return self.result_from_outputs([self.processor.process_iter()], start)

    def finish(self) -> TranscriptionResult:
        start = time.perf_counter()
        self.samples_since_process = 0
        return self.result_from_outputs([self.processor.finish()], start)

    @property
    def min_process_samples(self) -> int:
        return max(1, int(self.spec.streaming_min_chunk_seconds * TARGET_SAMPLE_RATE))

    def result_from_outputs(self, outputs: list[tuple[Any, Any, str]], start: float) -> TranscriptionResult:
        self.committed_segments.extend(format_streaming_outputs(outputs, start_id=len(self.committed_segments)))
        segments = list(self.committed_segments)
        text = " ".join(segment["text"] for segment in segments).strip()
        return TranscriptionResult(
            text=text,
            segments=segments,
            decode_time=time.perf_counter() - start,
            model_load_time=self.model_load_time,
            timing_source="whisper_streaming_online_processor",
        )


def supports_native_streaming(spec: EngineSpec) -> bool:
    return spec.kind in NATIVE_STREAMING_KINDS


def create_streaming_session(
    spec: EngineSpec,
    get_model: ModelGetter,
    language: str,
    beam_size: int,
    temperature: float,
) -> StreamingSession:
    if spec.kind == "whisper_streaming":
        return create_whisper_streaming_session(spec, get_model, language, beam_size, temperature)
    raise ValueError(f"native streaming을 지원하지 않는 엔진 타입: {spec.kind}")


def create_whisper_streaming_session(
    spec: EngineSpec,
    get_model: ModelGetter,
    language: str,
    beam_size: int,
    temperature: float,
) -> WhisperStreamingSession:
    asr, model_load_time = get_model(spec)
    _faster_whisper_asr, online_processor_cls = load_whisper_streaming_classes()
    asr.original_language = None if language == "auto" else language
    asr.beam_size = beam_size
    asr.temperature = temperature
    processor = online_processor_cls(
        asr,
        tokenizer=None,
        buffer_trimming=("segment", spec.streaming_buffer_trimming_seconds),
    )
    return WhisperStreamingSession(
        spec=spec,
        processor=processor,
        asr=asr,
        model_load_time=model_load_time,
    )


def pcm_bytes_to_float32(pcm: bytes, settings: tuple[int, int, int]) -> np.ndarray:
    channels, sample_width, sample_rate = settings
    if channels != 1 or sample_width != 2 or sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError("Native streaming 엔진은 16kHz mono 16-bit PCM 입력만 지원합니다.")
    if not pcm:
        return np.array([], dtype=np.float32)
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def format_streaming_outputs(outputs: list[tuple[Any, Any, str]], start_id: int = 0) -> list[dict[str, Any]]:
    rows = []
    for output in outputs:
        if not output or len(output) != 3:
            continue
        start, end, text = output
        text = str(text).strip()
        if not text:
            continue
        rows.append({"id": start_id + len(rows), "start": start, "end": end, "text": text})
    return rows
