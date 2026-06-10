import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import requests

from app.audio import TARGET_SAMPLE_RATE
from app.schemas import EngineInfo, EngineStatus


@dataclass(frozen=True)
class EngineSpec:
    id: str
    name: str
    provider: str
    kind: str
    model: str
    device: str = "cuda:0"
    compute_type: str = "float16"
    precision: str = "float16"
    return_timestamps: bool = True
    server_url: str = "http://127.0.0.1:8100/inference"
    note: str = ""


@dataclass
class TranscriptionResult:
    text: str
    segments: list[dict[str, Any]]
    decode_time: float
    model_load_time: float = 0.0
    timing_source: str = "python_timer"


DEFAULT_ENGINES = [
    EngineSpec(
        id="openai_whisper_large_v3",
        name="OpenAI Whisper large-v3",
        provider="OpenAI Whisper",
        kind="openai_whisper",
        model=os.getenv("DEMO_OPENAI_WHISPER_MODEL", "large-v3"),
        device=os.getenv("DEMO_OPENAI_WHISPER_DEVICE", "cuda:0"),
        precision=os.getenv("DEMO_OPENAI_WHISPER_PRECISION", "fp16"),
    ),
    EngineSpec(
        id="faster_whisper_large_v3",
        name="faster-whisper large-v3",
        provider="faster-whisper",
        kind="faster_whisper",
        model=os.getenv("DEMO_FASTER_WHISPER_MODEL", "large-v3"),
        device=os.getenv("DEMO_FASTER_WHISPER_DEVICE", "cuda:1"),
        compute_type=os.getenv("DEMO_FASTER_WHISPER_COMPUTE_TYPE", "float16"),
    ),
    EngineSpec(
        id="whisper_cpp_server_large_v3",
        name="whisper.cpp server large-v3",
        provider="whisper.cpp",
        kind="whisper_cpp_server",
        model=os.getenv("DEMO_WHISPER_CPP_MODEL", "large-v3-q5_0"),
        server_url=os.getenv("DEMO_WHISPER_CPP_SERVER_URL", "http://127.0.0.1:8100/inference"),
        device=os.getenv("DEMO_WHISPER_CPP_DEVICE_LABEL", "cuda:2/server"),
        note="whisper.cpp server가 먼저 실행되어 있어야 한다.",
    ),
    EngineSpec(
        id="qwen3_speech_recognition",
        name="Qwen3-ASR-1.7B",
        provider="Qwen",
        kind="qwen_speech_recognition",
        model=os.getenv("DEMO_QWEN_MODEL", "Qwen/Qwen3-ASR-1.7B"),
        device=os.getenv("DEMO_QWEN_DEVICE", "cuda:3"),
        precision=os.getenv("DEMO_QWEN_PRECISION", "bfloat16"),
        return_timestamps=False,
    ),
    EngineSpec(
        id="huggingface_crisperwhisper",
        name="CrisperWhisper",
        provider="Hugging Face Transformers",
        kind="huggingface_transformers",
        model=os.getenv("DEMO_CRISPERWHISPER_MODEL", "nyrahealth/CrisperWhisper"),
        device=os.getenv("DEMO_CRISPERWHISPER_DEVICE", os.getenv("DEMO_HUGGINGFACE_DEVICE", "cuda:4")),
        precision=os.getenv("DEMO_HUGGINGFACE_PRECISION", "float16"),
    ),
    EngineSpec(
        id="huggingface_ghost613_korean",
        name="ghost613 Korean Whisper",
        provider="Hugging Face Transformers",
        kind="huggingface_transformers",
        model=os.getenv("DEMO_GHOST613_MODEL", "ghost613/whisper-large-v3-turbo-korean"),
        device=os.getenv("DEMO_GHOST613_DEVICE", os.getenv("DEMO_HUGGINGFACE_DEVICE", "cuda:5")),
        precision=os.getenv("DEMO_HUGGINGFACE_PRECISION", "float16"),
        return_timestamps=False,
    ),
]


def engine_info(spec: EngineSpec) -> EngineInfo:
    return EngineInfo(
        id=spec.id,
        name=spec.name,
        provider=spec.provider,
        model=spec.model,
        device=spec.device if spec.kind != "whisper_cpp_server" else spec.server_url,
        note=spec.note,
    )


def torch_dtype(precision: str):
    import torch

    values = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if precision not in values:
        raise ValueError(f"지원하지 않는 precision: {precision}")
    return values[precision]


class EngineManager:
    def __init__(self, specs: list[EngineSpec]):
        self.specs = {spec.id: spec for spec in specs}
        self.cache: dict[str, Any] = {}
        self.cache_locks = {spec.id: threading.RLock() for spec in specs}
        self.decode_locks = {spec.id: threading.RLock() for spec in specs}
        self.status_lock = threading.RLock()
        self.statuses = {
            spec.id: EngineStatus(
                id=spec.id,
                state="not_loaded",
                label="server 확인 대기" if spec.kind == "whisper_cpp_server" else "모델 로딩 대기",
            )
            for spec in specs
        }

    def list_engines(self) -> list[EngineInfo]:
        return [engine_info(spec) for spec in self.specs.values()]

    def list_statuses(self) -> list[EngineStatus]:
        for spec in self.specs.values():
            if spec.kind == "whisper_cpp_server":
                self.refresh_whisper_cpp_status(spec)
        with self.status_lock:
            return [self.statuses[engine_id] for engine_id in self.specs]

    def set_status(
        self,
        engine_id: str,
        state: str,
        label: str,
        load_time: float | None = None,
        error: str = "",
    ) -> None:
        with self.status_lock:
            previous = self.statuses.get(engine_id)
            self.statuses[engine_id] = EngineStatus(
                id=engine_id,
                state=state,
                label=label,
                load_time=load_time if load_time is not None else previous.load_time if previous else None,
                error=error,
            )

    def preload_all(self, event_callback: Callable[[dict[str, Any]], None] | None = None) -> None:
        thread = threading.Thread(
            target=self.preload_all_sequentially,
            args=(event_callback,),
            daemon=True,
        )
        thread.start()

    def preload_all_sequentially(self, event_callback: Callable[[dict[str, Any]], None] | None = None) -> None:
        for spec in self.specs.values():
            self.preload_one(spec.id, event_callback)

    def preload_one(self, engine_id: str, event_callback: Callable[[dict[str, Any]], None] | None = None) -> None:
        spec = self.get_spec(engine_id)
        if spec.kind == "whisper_cpp_server":
            if self.is_whisper_cpp_server_available(spec):
                self.set_status(engine_id, "ready", "준비 완료")
            else:
                self.set_status(
                    engine_id,
                    "error",
                    "server 미실행",
                    error=f"whisper.cpp server 연결 실패: {spec.server_url}",
                )
            return

        started_at = time.perf_counter()
        try:
            self.set_status(engine_id, "loading", "모델 로딩 중")
            _model, load_time = self.get_or_load_model(spec)
            elapsed = load_time if load_time > 0 else time.perf_counter() - started_at
            self.set_status(engine_id, "ready", "준비 완료", load_time=elapsed)
            if event_callback:
                event_callback(
                    {
                        "engine_id": spec.id,
                        "engine": spec.name,
                        "model": spec.model,
                        "device": spec.device,
                        "model_load_time": round(elapsed, 6),
                        "status": "ok",
                    }
                )
        except Exception as exc:
            self.set_status(engine_id, "error", "로딩 실패", error=str(exc))
            if event_callback:
                event_callback(
                    {
                        "engine_id": spec.id,
                        "engine": spec.name,
                        "model": spec.model,
                        "device": spec.device,
                        "status": "error",
                        "error": str(exc),
                    }
                )

    def get_spec(self, engine_id: str) -> EngineSpec:
        if engine_id not in self.specs:
            names = ", ".join(self.specs)
            raise ValueError(f"알 수 없는 엔진: {engine_id}. 사용 가능: {names}")
        return self.specs[engine_id]

    def get_or_load_model(self, spec: EngineSpec) -> tuple[Any, float]:
        cached = self.cache.get(spec.id)
        if cached is not None:
            return cached, 0.0

        with self.cache_locks[spec.id]:
            cached = self.cache.get(spec.id)
            if cached is not None:
                return cached, 0.0

            self.set_status(spec.id, "loading", "모델 로딩 중")
            load_start = time.perf_counter()
            model = self.load_model(spec)
            load_time = time.perf_counter() - load_start
            self.cache[spec.id] = model
            self.set_status(spec.id, "ready", "준비 완료", load_time=load_time)
            return model, load_time

    def load_model(self, spec: EngineSpec) -> Any:
        if spec.kind == "openai_whisper":
            import whisper

            return whisper.load_model(spec.model, device=spec.device)
        if spec.kind == "faster_whisper":
            from faster_whisper import WhisperModel

            device, device_index = parse_faster_whisper_device(spec.device)
            return WhisperModel(
                spec.model,
                device=device,
                device_index=device_index,
                compute_type=spec.compute_type,
            )
        if spec.kind == "qwen_speech_recognition":
            from qwen_asr import Qwen3ASRModel

            return Qwen3ASRModel.from_pretrained(
                spec.model,
                dtype=torch_dtype(spec.precision),
                device_map=spec.device,
                max_inference_batch_size=1,
                max_new_tokens=256,
            )
        if spec.kind == "huggingface_transformers":
            return load_huggingface_pipeline(spec)
        raise ValueError(f"preload를 지원하지 않는 엔진 타입: {spec.kind}")

    def is_whisper_cpp_server_available(self, spec: EngineSpec) -> bool:
        try:
            response = requests.get(whisper_cpp_health_url(spec.server_url), timeout=0.5)
        except requests.RequestException:
            return False
        if response.status_code != 200:
            return False
        try:
            return response.json().get("status") == "ok"
        except ValueError:
            return False

    def refresh_whisper_cpp_status(self, spec: EngineSpec) -> None:
        current = self.statuses[spec.id]
        if current.state == "decoding":
            return
        if self.is_whisper_cpp_server_available(spec):
            if current.state == "error" and current.label == "server 미실행":
                self.set_status(spec.id, "ready", "준비 완료")
            elif current.state == "not_loaded":
                self.set_status(spec.id, "ready", "준비 완료")
            return
        self.set_status(
            spec.id,
            "error",
            "server 미실행",
            error=f"whisper.cpp server 연결 실패: {spec.server_url}",
        )

    def transcribe(
        self,
        engine_id: str,
        audio_path: Path,
        language: str,
        beam_size: int,
        temperature: float,
    ) -> TranscriptionResult:
        spec = self.get_spec(engine_id)
        if spec.kind == "whisper_cpp_server" and not self.is_whisper_cpp_server_available(spec):
            self.set_status(
                engine_id,
                "error",
                "server 미실행",
                error=f"whisper.cpp server 연결 실패: {spec.server_url}",
            )
            raise RuntimeError(
                "whisper.cpp server가 실행 중이 아닙니다. "
                "별도 터미널에서 `python scripts/run_whisper_cpp_server.py --experiment large-v3_f16_beam1_server`를 실행하거나 "
                f"DEMO_WHISPER_CPP_SERVER_URL을 확인하세요: {spec.server_url}"
            )
        with self.decode_locks[engine_id]:
            self.set_status(engine_id, "decoding", "인식 중")
            try:
                if spec.kind == "openai_whisper":
                    return self._transcribe_openai_whisper(spec, audio_path, language, beam_size, temperature)
                if spec.kind == "faster_whisper":
                    return self._transcribe_faster_whisper(spec, audio_path, language, beam_size, temperature)
                if spec.kind == "whisper_cpp_server":
                    return self._transcribe_whisper_cpp_server(spec, audio_path, language, beam_size, temperature)
                if spec.kind == "qwen_speech_recognition":
                    return self._transcribe_qwen(spec, audio_path, language)
                if spec.kind == "huggingface_transformers":
                    return self._transcribe_huggingface_transformers(spec, audio_path, language, beam_size)
                raise ValueError(f"지원하지 않는 엔진 타입: {spec.kind}")
            except Exception as exc:
                self.set_status(engine_id, "error", "인식 실패", error=str(exc))
                raise
            finally:
                if self.statuses[engine_id].state != "error":
                    self.set_status(engine_id, "ready", "준비 완료")

    def _transcribe_openai_whisper(
        self,
        spec: EngineSpec,
        audio_path: Path,
        language: str,
        beam_size: int,
        temperature: float,
    ) -> TranscriptionResult:
        model, model_load_time = self.get_or_load_model(spec)

        start = time.perf_counter()
        result = model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            beam_size=beam_size,
            temperature=temperature,
            condition_on_previous_text=False,
            fp16=spec.precision in {"fp16", "float16"} and spec.device.startswith("cuda"),
            verbose=False,
        )
        decode_time = time.perf_counter() - start
        return TranscriptionResult(
            text=str(result.get("text", "")).strip(),
            segments=format_openai_segments(result),
            decode_time=decode_time,
            model_load_time=model_load_time,
        )

    def _transcribe_faster_whisper(
        self,
        spec: EngineSpec,
        audio_path: Path,
        language: str,
        beam_size: int,
        temperature: float,
    ) -> TranscriptionResult:
        model, model_load_time = self.get_or_load_model(spec)

        start = time.perf_counter()
        segments, _info = model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            beam_size=beam_size,
            temperature=temperature,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=False,
        )
        text, rows = format_faster_segments(segments)
        return TranscriptionResult(text=text, segments=rows, decode_time=time.perf_counter() - start, model_load_time=model_load_time)

    def _transcribe_whisper_cpp_server(
        self,
        spec: EngineSpec,
        audio_path: Path,
        language: str,
        beam_size: int,
        temperature: float,
    ) -> TranscriptionResult:
        start = time.perf_counter()
        fields = {
            "response_format": "verbose_json",
            "language": language,
            "beam_size": str(beam_size),
            "temperature": str(temperature),
            "temperature_inc": "0.0",
            "no_language_probabilities": "true",
        }
        with audio_path.open("rb") as audio_file:
            files = {"file": (audio_path.name, audio_file, "audio/wav")}
            try:
                response = requests.post(spec.server_url, data=fields, files=files, timeout=None)
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"whisper.cpp server에 연결할 수 없습니다. 먼저 server를 실행하고 DEMO_WHISPER_CPP_SERVER_URL을 확인하세요: {spec.server_url}"
                ) from exc
        request_time = time.perf_counter() - start
        response.raise_for_status()
        data = response.json()
        text, segments, backend_time = parse_whisper_cpp_response(data)
        return TranscriptionResult(
            text=text,
            segments=segments,
            decode_time=backend_time if backend_time is not None else request_time,
            model_load_time=0.0,
            timing_source="server_timings" if backend_time is not None else "http_request",
        )

    def _transcribe_qwen(self, spec: EngineSpec, audio_path: Path, language: str) -> TranscriptionResult:
        model, model_load_time = self.get_or_load_model(spec)

        start = time.perf_counter()
        results = model.transcribe(
            audio=str(audio_path),
            language=qwen_language(language),
            return_time_stamps=spec.return_timestamps,
        )
        result = results[0] if isinstance(results, list) else results
        return TranscriptionResult(
            text=str(getattr(result, "text", "")).strip(),
            segments=format_qwen_segments(result) if spec.return_timestamps else [],
            decode_time=time.perf_counter() - start,
            model_load_time=model_load_time,
        )

    def _transcribe_huggingface_transformers(
        self,
        spec: EngineSpec,
        audio_path: Path,
        language: str,
        beam_size: int,
    ) -> TranscriptionResult:
        pipeline_model, model_load_time = self.get_or_load_model(spec)

        start = time.perf_counter()
        audio = load_audio_for_pipeline(audio_path)
        result = pipeline_model(
            audio,
            generate_kwargs=huggingface_generate_kwargs(spec, language, beam_size),
        )
        return TranscriptionResult(
            text=str(result.get("text", "")).strip(),
            segments=format_pipeline_segments(result),
            decode_time=time.perf_counter() - start,
            model_load_time=model_load_time,
        )


def parse_faster_whisper_device(device: str) -> tuple[str, int]:
    if device.startswith("cuda:"):
        return "cuda", int(device.split(":", 1)[1])
    return device, 0


def whisper_cpp_health_url(server_url: str) -> str:
    parts = urlsplit(server_url)
    return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))


def format_openai_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": segment.get("id", index),
            "start": segment.get("start"),
            "end": segment.get("end"),
            "text": str(segment.get("text", "")).strip(),
        }
        for index, segment in enumerate(result.get("segments", []) or [])
    ]


def format_faster_segments(segments) -> tuple[str, list[dict[str, Any]]]:
    texts = []
    rows = []
    for segment in segments:
        text = str(segment.text).strip()
        texts.append(text)
        rows.append({"id": segment.id, "start": segment.start, "end": segment.end, "text": text})
    return " ".join(texts).strip(), rows


def parse_whisper_cpp_response(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]], float | None]:
    segments = []
    for index, segment in enumerate(data.get("segments", []) or []):
        segments.append(
            {
                "id": segment.get("id", index),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": str(segment.get("text", "")).strip(),
            }
        )
    text = str(data.get("text", "")).strip()
    if not text:
        text = " ".join(segment["text"] for segment in segments).strip()
    inference_sec = (data.get("timings") or {}).get("inference_sec")
    return text, segments, float(inference_sec) if inference_sec is not None else None


def qwen_language(language: str) -> str:
    if language in {"ko", "korean", "Korean"}:
        return "Korean"
    return language


def huggingface_generate_kwargs(spec: EngineSpec, language: str, beam_size: int) -> dict[str, Any]:
    if spec.model.startswith("ghost613/"):
        return {"num_beams": beam_size}
    return {"language": language, "task": "transcribe", "num_beams": beam_size}


def format_qwen_segments(result: Any) -> list[dict[str, Any]]:
    segments = []
    for index, stamp in enumerate(getattr(result, "time_stamps", None) or []):
        text = getattr(stamp, "text", "")
        start = getattr(stamp, "start_time", None)
        end = getattr(stamp, "end_time", None)
        if isinstance(stamp, dict):
            text = stamp.get("text", text)
            start = stamp.get("start_time", stamp.get("start", start))
            end = stamp.get("end_time", stamp.get("end", end))
        segments.append({"id": index, "start": start, "end": end, "text": str(text).strip()})
    return segments


def load_audio_for_pipeline(audio_path: Path) -> dict[str, Any]:
    import torchaudio
    from torchaudio.functional import resample

    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
    return {"array": waveform.squeeze(0).detach().cpu().numpy(), "sampling_rate": TARGET_SAMPLE_RATE}


def load_huggingface_pipeline(spec: EngineSpec):
    import transformers.pipelines.automatic_speech_recognition as speech_recognition_pipeline
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    speech_recognition_pipeline.is_torchcodec_available = lambda: False
    dtype = torch_dtype(spec.precision)
    if spec.model.startswith("ghost613/"):
        return pipeline(
            "automatic-speech-recognition",
            model=spec.model,
            dtype=dtype,
            device=spec.device,
            chunk_length_s=30,
            batch_size=1,
            return_timestamps=spec.return_timestamps,
        )

    device_map = {"": spec.device} if spec.device.startswith("cuda") else None
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        spec.model,
        dtype=dtype,
        low_cpu_mem_usage=False,
        use_safetensors=True,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(spec.model)
    update_whisper_generation_config(model, processor)
    pipeline_kwargs = {
        "task": "automatic-speech-recognition",
        "model": model,
        "tokenizer": processor.tokenizer,
        "feature_extractor": processor.feature_extractor,
        "dtype": dtype,
        "chunk_length_s": 30,
        "batch_size": 1,
        "return_timestamps": spec.return_timestamps,
    }
    return pipeline(**pipeline_kwargs)


def format_pipeline_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, chunk in enumerate(result.get("chunks", []) or []):
        timestamp = chunk.get("timestamp") or (None, None)
        start, end = timestamp if len(timestamp) == 2 else (None, None)
        rows.append({"id": index, "start": start, "end": end, "text": str(chunk.get("text", "")).strip()})
    return rows


def update_whisper_generation_config(model: Any, processor: Any) -> None:
    tokenizer = processor.tokenizer
    if not hasattr(tokenizer, "get_vocab"):
        return

    vocab = tokenizer.get_vocab()
    generation_config = model.generation_config
    if not hasattr(generation_config, "is_multilingual"):
        generation_config.is_multilingual = True
    if not hasattr(generation_config, "lang_to_id"):
        generation_config.lang_to_id = {
            token: token_id
            for token, token_id in vocab.items()
            if token.startswith("<|") and token.endswith("|>") and len(token) == 6
        }
    if not hasattr(generation_config, "task_to_id"):
        generation_config.task_to_id = {
            "translate": tokenizer.convert_tokens_to_ids("<|translate|>"),
            "transcribe": tokenizer.convert_tokens_to_ids("<|transcribe|>"),
        }


engine_manager = EngineManager(DEFAULT_ENGINES)
