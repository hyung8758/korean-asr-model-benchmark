import sys
import time
from pathlib import Path
from typing import Any, Callable

import requests

from app.audio import TARGET_SAMPLE_RATE
from app.ctranslate2_runtime import CTRANSLATE2_CUDA_LOCK
from app.engine_models import EngineSpec, TranscriptionResult, project_path, torch_dtype


ModelGetter = Callable[[EngineSpec], tuple[Any, float]]


def load_engine_model(spec: EngineSpec) -> Any:
    if spec.kind == "openai_whisper":
        import whisper

        return whisper.load_model(spec.model, device=spec.device)
    if spec.kind == "faster_whisper":
        from faster_whisper import WhisperModel

        device, device_index = parse_faster_whisper_device(spec.device)
        with CTRANSLATE2_CUDA_LOCK:
            return WhisperModel(
                spec.model,
                device=device,
                device_index=device_index,
                compute_type=spec.compute_type,
            )
    if spec.kind == "whisper_streaming":
        return load_whisper_streaming_model(spec)
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


def transcribe_with_engine(
    spec: EngineSpec,
    get_model: ModelGetter,
    audio_path: Path,
    language: str,
    beam_size: int,
    temperature: float,
) -> TranscriptionResult:
    if spec.kind == "openai_whisper":
        return transcribe_openai_whisper(spec, get_model, audio_path, language, beam_size, temperature)
    if spec.kind == "faster_whisper":
        return transcribe_faster_whisper(spec, get_model, audio_path, language, beam_size, temperature)
    if spec.kind == "whisper_streaming":
        return transcribe_whisper_streaming(spec, get_model, audio_path, language, beam_size, temperature)
    if spec.kind == "whisper_cpp_server":
        return transcribe_whisper_cpp_server(spec, audio_path, language, beam_size, temperature)
    if spec.kind == "qwen_speech_recognition":
        return transcribe_qwen(spec, get_model, audio_path, language)
    if spec.kind == "huggingface_transformers":
        return transcribe_huggingface_transformers(spec, get_model, audio_path, language, beam_size)
    raise ValueError(f"지원하지 않는 엔진 타입: {spec.kind}")


def transcribe_openai_whisper(
    spec: EngineSpec,
    get_model: ModelGetter,
    audio_path: Path,
    language: str,
    beam_size: int,
    temperature: float,
) -> TranscriptionResult:
    model, model_load_time = get_model(spec)
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
    return TranscriptionResult(
        text=str(result.get("text", "")).strip(),
        segments=format_openai_segments(result),
        decode_time=time.perf_counter() - start,
        model_load_time=model_load_time,
    )


def transcribe_faster_whisper(
    spec: EngineSpec,
    get_model: ModelGetter,
    audio_path: Path,
    language: str,
    beam_size: int,
    temperature: float,
) -> TranscriptionResult:
    model, model_load_time = get_model(spec)
    start = time.perf_counter()
    text, rows = transcribe_faster_whisper_audio(
        model=model,
        audio=str(audio_path),
        language=language,
        beam_size=beam_size,
        temperature=temperature,
    )
    return TranscriptionResult(text=text, segments=rows, decode_time=time.perf_counter() - start, model_load_time=model_load_time)


def transcribe_whisper_streaming(
    spec: EngineSpec,
    get_model: ModelGetter,
    audio_path: Path,
    language: str,
    beam_size: int,
    temperature: float,
) -> TranscriptionResult:
    asr, model_load_time = get_model(spec)
    _faster_whisper_asr, online_processor_cls = load_whisper_streaming_classes()
    audio = load_audio_for_pipeline(audio_path)
    samples = audio["array"]
    sample_rate = int(audio["sampling_rate"])
    chunk_samples = max(1, int(spec.streaming_min_chunk_seconds * sample_rate))
    processor = online_processor_cls(
        asr,
        tokenizer=None,
        buffer_trimming=("segment", spec.streaming_buffer_trimming_seconds),
    )

    asr.original_language = None if language == "auto" else language
    asr.beam_size = beam_size
    asr.temperature = temperature

    start = time.perf_counter()
    outputs = []
    previous = 0
    with CTRANSLATE2_CUDA_LOCK:
        for end in chunk_end_points(len(samples), chunk_samples):
            processor.insert_audio_chunk(samples[previous:end])
            outputs.append(processor.process_iter())
            previous = end
        outputs.append(processor.finish())

    segments = format_whisper_streaming_outputs(outputs)
    text = " ".join(segment["text"] for segment in segments).strip()
    return TranscriptionResult(
        text=text,
        segments=segments,
        decode_time=time.perf_counter() - start,
        model_load_time=model_load_time,
        timing_source="whisper_streaming_online_processor",
    )


def transcribe_whisper_cpp_server(
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
            raise RuntimeError(f"whisper.cpp server에 연결할 수 없습니다: {spec.server_url}") from exc
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


def transcribe_qwen(
    spec: EngineSpec,
    get_model: ModelGetter,
    audio_path: Path,
    language: str,
) -> TranscriptionResult:
    model, model_load_time = get_model(spec)
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


def transcribe_huggingface_transformers(
    spec: EngineSpec,
    get_model: ModelGetter,
    audio_path: Path,
    language: str,
    beam_size: int,
) -> TranscriptionResult:
    pipeline_model, model_load_time = get_model(spec)
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


def load_whisper_streaming_classes() -> tuple[type, type]:
    source_dir = project_path("third_party/whisper_streaming")
    source_file = source_dir / "whisper_online.py"
    if not source_file.is_file():
        raise FileNotFoundError(
            "whisper-streaming submodule is missing. Run: git submodule update --init --recursive"
        )
    source_path = str(source_dir)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from whisper_online import FasterWhisperASR, OnlineASRProcessor

    return FasterWhisperASR, OnlineASRProcessor


def load_whisper_streaming_model(spec: EngineSpec) -> Any:
    faster_whisper_asr_cls, _online_processor_cls = load_whisper_streaming_classes()

    class DemoFasterWhisperASR(faster_whisper_asr_cls):
        def __init__(self, model: str, device: str, compute_type: str):
            self.demo_device = device
            self.demo_compute_type = compute_type
            self.beam_size = 1
            self.temperature = 0.0
            super().__init__("ko", modelsize=model)

        def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
            from faster_whisper import WhisperModel

            if model_dir is not None:
                model_size_or_path = model_dir
            elif modelsize is not None:
                model_size_or_path = modelsize
            else:
                raise ValueError("modelsize or model_dir parameter must be set")

            device, device_index = parse_faster_whisper_device(self.demo_device)
            return WhisperModel(
                model_size_or_path,
                device=device,
                device_index=device_index,
                compute_type=self.demo_compute_type,
                download_root=cache_dir,
            )

        def transcribe(self, audio, init_prompt=""):
            with CTRANSLATE2_CUDA_LOCK:
                segments, _info = self.model.transcribe(
                    audio,
                    language=self.original_language,
                    initial_prompt=init_prompt,
                    beam_size=self.beam_size,
                    temperature=self.temperature,
                    word_timestamps=True,
                    condition_on_previous_text=True,
                )
                return list(segments)

    with CTRANSLATE2_CUDA_LOCK:
        return DemoFasterWhisperASR(
            model=spec.model,
            device=spec.device,
            compute_type=spec.compute_type,
        )


def transcribe_faster_whisper_audio(
    model: Any,
    audio: Any,
    language: str,
    beam_size: int,
    temperature: float,
) -> tuple[str, list[dict[str, Any]]]:
    with CTRANSLATE2_CUDA_LOCK:
        segments, _info = model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=beam_size,
            temperature=temperature,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=False,
        )
        return format_faster_segments(segments)


def chunk_end_points(total_samples: int, chunk_samples: int) -> list[int]:
    if total_samples <= 0:
        return []
    points = list(range(chunk_samples, total_samples, chunk_samples))
    points.append(total_samples)
    return points


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


def format_whisper_streaming_outputs(outputs: list[tuple[Any, Any, str]]) -> list[dict[str, Any]]:
    rows = []
    for output in outputs:
        if not output or len(output) != 3:
            continue
        start, end, text = output
        text = str(text).strip()
        if not text:
            continue
        rows.append({"id": len(rows), "start": start, "end": end, "text": text})
    return rows


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
