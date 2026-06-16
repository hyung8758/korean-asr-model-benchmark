import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.concurrency import run_in_threadpool

from app.audio import read_wav_chunk, save_upload, wav_duration, write_wav
from app.config import DEMO_CONFIG
from app.engine_models import EngineSpec, TranscriptionResult
from app.engines import engine_manager
from app.schemas import EngineInfo, EngineStatus, TranscriptionResponse
from app.vad import VadSegment, create_vad


class AccessLogPathFilter(logging.Filter):
    def __init__(self, hidden_paths: set[str]):
        super().__init__()
        self.hidden_paths = hidden_paths

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(f" {path} " in message for path in self.hidden_paths)


app = FastAPI(title="한국어 STT 데모 서버")
LOGGER = logging.getLogger(__name__)
DEFAULT_RUN_DIR = Path("logs") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_log"
RUN_DIR = Path(os.getenv("DEMO_RUN_DIR", str(DEFAULT_RUN_DIR)))
LOG_DIR = Path(os.getenv("DEMO_LOG_DIR", str(RUN_DIR)))
SAVE_DIR = Path(os.getenv("DEMO_SAVE_DIR", str(RUN_DIR / "saved_audio")))
EVENT_LOG_PATH = LOG_DIR / "decoding_events.jsonl"
MODEL_LOG_PATH = LOG_DIR / "model_events.jsonl"
LOG_LOCK = threading.Lock()
DEFAULTS = DEMO_CONFIG["defaults"]
SERVER_CONFIG = DEMO_CONFIG["server"]
SECURITY_CONFIG = SERVER_CONFIG.get("security", {})
STREAMING_CONFIG = DEMO_CONFIG["streaming"]
VAD_CONFIG = DEMO_CONFIG["vad"]
DEFAULT_LANGUAGE = str(DEFAULTS["language"])
DEFAULT_BEAM_SIZE = int(DEFAULTS["beam_size"])
DEFAULT_TEMPERATURE = float(DEFAULTS["temperature"])
MAX_UPLOAD_BYTES = int(SECURITY_CONFIG.get("max_upload_mb", 100)) * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS = float(SECURITY_CONFIG.get("max_audio_duration_seconds", 1200))
MAX_ACTIVE_SESSIONS = max(1, int(SECURITY_CONFIG.get("max_active_sessions", 1)))
STREAM_PARTIAL_INTERVAL_SECONDS = float(
    os.getenv("DEMO_STREAM_PARTIAL_INTERVAL_SECONDS", STREAMING_CONFIG["partial_interval_seconds"])
)
STREAM_MIN_PARTIAL_AUDIO_SECONDS = float(
    os.getenv("DEMO_STREAM_MIN_PARTIAL_AUDIO_SECONDS", STREAMING_CONFIG.get("min_partial_audio_seconds", 1.0))
)
SESSION_SEMAPHORE = asyncio.Semaphore(MAX_ACTIVE_SESSIONS)
SESSION_LIMIT_LOCK = asyncio.Lock()

LOG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

hidden_access_log_paths = set(SECURITY_CONFIG.get("hide_access_log_paths", []))
if hidden_access_log_paths:
    access_log_filter = AccessLogPathFilter(hidden_access_log_paths)
    logging.getLogger("uvicorn.access").addFilter(access_log_filter)
    logging.getLogger("gunicorn.access").addFilter(access_log_filter)

cors_origins = SECURITY_CONFIG.get("cors_origins") or []
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root(request: Request) -> RedirectResponse:
    frontend_url = os.getenv("DEMO_FRONTEND_PUBLIC_URL")
    if frontend_url:
        return RedirectResponse(frontend_url)
    frontend_port = os.getenv("DEMO_FRONTEND_PORT", str(SERVER_CONFIG["frontend_port"]))
    return RedirectResponse(f"{request.url.scheme}://{request.url.hostname}:{frontend_port}")


def response_payload(response: TranscriptionResponse) -> dict:
    return response.model_dump()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(text: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)


def write_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def save_audio_copy(source_path: Path, request_id: str, engine_id: str, mode: str) -> Path:
    suffix = source_path.suffix or ".wav"
    output_path = SAVE_DIR / f"{request_id}_{safe_name(engine_id)}_{mode}{suffix}"
    shutil.copy2(source_path, output_path)
    return output_path


async def send_websocket_json(websocket: WebSocket, payload: dict) -> bool:
    try:
        await websocket.send_json(payload)
        return True
    except (RuntimeError, WebSocketDisconnect):
        return False


def parse_engine_ids(engine_ids: str) -> list[str]:
    values = [engine_id.strip() for engine_id in engine_ids.split(",") if engine_id.strip()]
    if not values:
        raise ValueError("최소 하나 이상의 engine_id가 필요합니다.")
    return values


def audio_seconds_from_frames(frames: bytes, settings: tuple[int, int, int]) -> float:
    channels, sample_width, sample_rate = settings
    frame_size = channels * sample_width
    if frame_size <= 0 or sample_rate <= 0:
        return 0.0
    return (len(frames) / frame_size) / sample_rate


def vad_segment_duration(segment: VadSegment) -> float:
    return max(0.0, segment.end - segment.start)


def validate_upload_size(data: bytes) -> None:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"업로드 파일이 너무 큽니다. 최대 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB까지 허용합니다.")


def validate_audio_duration(duration: float | None) -> None:
    if duration is not None and duration > MAX_AUDIO_DURATION_SECONDS:
        raise HTTPException(status_code=413, detail=f"음성이 너무 깁니다. 최대 {int(MAX_AUDIO_DURATION_SECONDS)}초까지 허용합니다.")


def build_transcription_response(
    result: TranscriptionResult,
    request_id: str,
    spec: EngineSpec,
    mode: str,
    audio_duration: float | None,
    total_time: float,
    saved_audio_path: Path | None = None,
) -> TranscriptionResponse:
    rtf = result.decode_time / audio_duration if audio_duration and audio_duration > 0 else None
    chars_per_second = len(result.text) / result.decode_time if result.decode_time > 0 else None
    audio_seconds_per_second = audio_duration / result.decode_time if audio_duration and result.decode_time > 0 else None
    return TranscriptionResponse(
        request_id=request_id,
        engine=spec.name,
        model=spec.model,
        mode=mode,
        text=result.text,
        segments=result.segments,
        saved_audio_path=str(saved_audio_path) if saved_audio_path else None,
        audio_duration=audio_duration,
        model_load_time=round(result.model_load_time, 6),
        decode_time=round(result.decode_time, 6),
        total_time=round(total_time, 6),
        rtf=round(rtf, 6) if rtf is not None else None,
        chars_per_second=round(chars_per_second, 3) if chars_per_second is not None else None,
        audio_seconds_per_second=round(audio_seconds_per_second, 3) if audio_seconds_per_second is not None else None,
        timing_source=result.timing_source,
    )


async def try_acquire_session() -> bool:
    async with SESSION_LIMIT_LOCK:
        if SESSION_SEMAPHORE.locked():
            return False
        await SESSION_SEMAPHORE.acquire()
        return True


async def send_status_event(
    websocket: WebSocket,
    engine_ids: list[str],
    status: str,
    vad_id: str,
    utterance_index: int | None = None,
    utterance_total: int | None = None,
) -> bool:
    payload = {
        "type": "status",
        "engine_ids": engine_ids,
        "status": status,
        "vad": vad_id,
    }
    if utterance_index is not None:
        payload["utterance_index"] = utterance_index
    if utterance_total is not None:
        payload["utterance_total"] = utterance_total
    return await send_websocket_json(websocket, payload)


async def decode_vad_segment(
    segment: VadSegment,
    settings: tuple[int, int, int],
    temp_dir: Path,
    request_id: str,
    engine_id: str,
    language: str,
    beam_size: int,
    temperature: float,
    mode: str,
    save_audio: bool,
) -> TranscriptionResponse:
    audio_path = temp_dir / f"utterance_{segment.index}_{safe_name(engine_id)}_{mode}.wav"
    write_wav(audio_path, segment.pcm, *settings)
    saved_audio_path = save_audio_copy(audio_path, request_id, engine_id, mode) if save_audio else None
    return await decode_file(
        audio_path=audio_path,
        engine_id=engine_id,
        language=language,
        beam_size=beam_size,
        temperature=temperature,
        mode=mode,
        request_id=request_id,
        saved_audio_path=saved_audio_path,
    )


async def send_segment_results(
    websocket: WebSocket,
    segment: VadSegment,
    settings: tuple[int, int, int],
    temp_dir: Path,
    request_id: str,
    engine_ids: list[str],
    language: str,
    beam_size: int,
    temperature: float,
    vad_id: str,
    message_type: str,
    mode: str,
    save_audio: bool,
    utterance_total: int | None = None,
) -> bool:
    if not engine_ids:
        return True

    async def decode_one(engine_id: str) -> tuple[str, TranscriptionResponse | None, Exception | None]:
        try:
            response = await decode_vad_segment(
                segment=segment,
                settings=settings,
                temp_dir=temp_dir,
                request_id=request_id,
                engine_id=engine_id,
                language=language,
                beam_size=beam_size,
                temperature=temperature,
                mode=mode,
                save_audio=save_audio,
            )
            return engine_id, response, None
        except Exception as exc:
            return engine_id, None, exc

    tasks = [asyncio.create_task(decode_one(engine_id)) for engine_id in engine_ids]
    for task in asyncio.as_completed(tasks):
        engine_id, response, error = await task
        if error is None and response is not None:
            payload = {
                "type": message_type,
                "engine_id": engine_id,
                "utterance_index": segment.index,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "vad": vad_id,
                "result": response_payload(response),
            }
            if utterance_total is not None:
                payload["utterance_total"] = utterance_total
        else:
            payload = {
                "type": "error",
                "engine_id": engine_id,
                "utterance_index": segment.index,
                "message": str(error),
            }
        if not await send_websocket_json(websocket, payload):
            return False
    return True


async def decode_and_send_segments(
    websocket: WebSocket,
    segments: list[VadSegment],
    settings: tuple[int, int, int] | None,
    temp_dir: Path,
    request_id: str,
    engine_ids: list[str],
    language: str,
    beam_size: int,
    temperature: float,
    vad_id: str,
    utterance_total: int | None = None,
) -> bool:
    if settings is None or not engine_ids:
        return True
    for segment in segments:
        if not segment.pcm:
            continue
        if not await send_status_event(
            websocket,
            engine_ids,
            "인식 중",
            vad_id,
            segment.index,
            utterance_total,
        ):
            return False
        ok = await send_segment_results(
            websocket=websocket,
            segment=segment,
            settings=settings,
            temp_dir=temp_dir,
            request_id=request_id,
            engine_ids=engine_ids,
            language=language,
            beam_size=beam_size,
            temperature=temperature,
            vad_id=vad_id,
            message_type="utterance_final",
            mode="vad_utterance",
            save_audio=True,
            utterance_total=utterance_total,
        )
        if not ok:
            return False
    return True


async def send_streaming_result(
    websocket: WebSocket,
    engine_id: str,
    result,
    request_id: str,
    mode: str,
    audio_duration: float | None,
    utterance_index: int,
    message_type: str,
) -> bool:
    if not result.text.strip():
        return True
    spec = engine_manager.get_spec(engine_id)
    response = build_transcription_response(
        result=result,
        request_id=request_id,
        spec=spec,
        mode=mode,
        audio_duration=audio_duration,
        total_time=result.decode_time,
    )
    return await send_websocket_json(
        websocket,
        {
            "type": message_type,
            "engine_id": engine_id,
            "utterance_index": utterance_index,
            "start": None,
            "end": None,
            "vad": "whisper-streaming",
            "result": response_payload(response),
        },
    )


async def create_native_streaming_sessions(
    engine_ids: list[str],
    language: str,
    beam_size: int,
    temperature: float,
) -> dict[str, str]:
    sessions = {}
    try:
        for engine_id in engine_ids:
            spec = engine_manager.get_spec(engine_id)
            if spec.kind != "whisper_streaming":
                continue
            sessions[engine_id] = await run_in_threadpool(
                engine_manager.start_stream,
                engine_id,
                language,
                beam_size,
                temperature,
            )
        return sessions
    except Exception:
        for engine_id, session_id in sessions.items():
            await run_in_threadpool(engine_manager.stream_cancel, engine_id, session_id)
        raise


@app.on_event("startup")
def preload_models() -> None:
    def log_preload_event(event: dict) -> None:
        write_event(MODEL_LOG_PATH, {"time": utc_now(), **event})

    engine_manager.preload_all(log_preload_event)


@app.on_event("shutdown")
def stop_workers() -> None:
    engine_manager.stop_all()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "log_dir": str(LOG_DIR), "save_dir": str(SAVE_DIR)}


@app.get("/api/demo-config")
def demo_config() -> dict:
    return DEMO_CONFIG


@app.get("/api/engines", response_model=list[EngineInfo])
def engines() -> list[EngineInfo]:
    return engine_manager.list_engines()


@app.get("/api/engine-status", response_model=list[EngineStatus])
def engine_status() -> list[EngineStatus]:
    return engine_manager.list_statuses()


@app.post("/api/engines/{engine_id}/activate", response_model=EngineStatus)
def activate_engine(engine_id: str) -> EngineStatus:
    try:
        return engine_manager.activate_engine(engine_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/engines/{engine_id}/deactivate", response_model=EngineStatus)
def deactivate_engine(engine_id: str) -> EngineStatus:
    try:
        return engine_manager.deactivate_engine(engine_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    engine_id: str = Form(...),
    language: str = Form(DEFAULT_LANGUAGE),
    beam_size: int = Form(DEFAULT_BEAM_SIZE),
    temperature: float = Form(DEFAULT_TEMPERATURE),
) -> TranscriptionResponse:
    if not await try_acquire_session():
        raise HTTPException(status_code=429, detail="다른 인식 작업이 진행 중입니다. 잠시 후 다시 시도하세요.")
    request_id = uuid.uuid4().hex
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    try:
        data = await file.read()
        validate_upload_size(data)
        audio_path = save_upload(data, suffix)
        validate_audio_duration(wav_duration(audio_path))
        saved_audio_path = save_audio_copy(audio_path, request_id, engine_id, "offline")
        return await decode_file(
            audio_path=audio_path,
            engine_id=engine_id,
            language=language,
            beam_size=beam_size,
            temperature=temperature,
            mode="offline",
            request_id=request_id,
            saved_audio_path=saved_audio_path,
        )
    except HTTPException:
        raise
    except Exception as exc:
        write_event(
            EVENT_LOG_PATH,
            {
                "time": utc_now(),
                "request_id": request_id,
                "engine_id": engine_id,
                "mode": "offline",
                "saved_audio_path": str(saved_audio_path) if "saved_audio_path" in locals() else None,
                "status": "error",
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if "audio_path" in locals():
            audio_path.unlink(missing_ok=True)
        SESSION_SEMAPHORE.release()


async def decode_file(
    audio_path: Path,
    engine_id: str,
    language: str,
    beam_size: int,
    temperature: float,
    mode: str,
    request_id: str,
    saved_audio_path: Path | None,
) -> TranscriptionResponse:
    spec = engine_manager.get_spec(engine_id)
    duration = wav_duration(audio_path)
    start = time.perf_counter()
    result = await run_in_threadpool(
        engine_manager.transcribe,
        engine_id,
        audio_path,
        language,
        beam_size,
        temperature,
    )
    total_time = time.perf_counter() - start
    response = build_transcription_response(
        result=result,
        request_id=request_id,
        spec=spec,
        mode=mode,
        audio_duration=duration,
        total_time=total_time,
        saved_audio_path=saved_audio_path,
    )
    write_event(
        EVENT_LOG_PATH,
        {
            "time": utc_now(),
            "request_id": request_id,
            "engine_id": engine_id,
            "engine": spec.name,
            "model": spec.model,
            "mode": mode,
            "saved_audio_path": str(saved_audio_path) if saved_audio_path else None,
            "audio_duration": duration,
            "model_load_time": response.model_load_time,
            "decode_time": response.decode_time,
            "total_time": response.total_time,
            "rtf": response.rtf,
            "chars_per_second": response.chars_per_second,
            "audio_seconds_per_second": response.audio_seconds_per_second,
            "timing_source": response.timing_source,
            "text_length": len(result.text),
            "segment_count": len(result.segments),
            "status": "ok",
        },
    )
    if result.model_load_time > 0:
        write_event(
            MODEL_LOG_PATH,
            {
                "time": utc_now(),
                "request_id": request_id,
                "engine_id": engine_id,
                "engine": spec.name,
                "model": spec.model,
                "model_load_time": response.model_load_time,
            },
        )
    return response


@app.websocket("/api/vad-stream")
async def vad_stream(
    websocket: WebSocket,
    engine_ids: str,
    vad_id: str = str(DEFAULTS["vad"]),
    stream_mode: str = str(DEFAULTS["mode"]),
    language: str = DEFAULT_LANGUAGE,
    beam_size: int = DEFAULT_BEAM_SIZE,
    temperature: float = DEFAULT_TEMPERATURE,
    input_source: str = "recording",
) -> None:
    await websocket.accept()
    if not await try_acquire_session():
        await send_websocket_json(websocket, {"type": "error", "message": "다른 인식 작업이 진행 중입니다. 잠시 후 다시 시도하세요."})
        await websocket.close(code=1013)
        return
    request_id = uuid.uuid4().hex
    wav_settings: tuple[int, int, int] | None = None
    started_at = time.perf_counter()
    last_partial_at = 0.0
    received_bytes = 0
    received_audio_seconds = 0.0

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        try:
            target_engine_ids = parse_engine_ids(engine_ids)
            native_streaming_sessions = (
                await create_native_streaming_sessions(
                    target_engine_ids,
                    language,
                    beam_size,
                    temperature,
                )
                if stream_mode == "streaming"
                else {}
            )
            vad_engine_ids = [
                engine_id
                for engine_id in target_engine_ids
                if engine_id not in native_streaming_sessions
            ]
            native_streaming_counts = {engine_id: 0 for engine_id in native_streaming_sessions}
            vad = create_vad(vad_id, VAD_CONFIG.get(vad_id, {}))
            while True:
                message = await websocket.receive()
                if "text" in message and message["text"] == "stop":
                    if not await send_websocket_json(websocket, {"type": "finalizing", "vad": vad_id}):
                        return
                    final_segments = [
                        segment
                        for segment in vad.pop_final_segments(force=True)
                        if segment.pcm and wav_settings is not None
                    ]
                    utterance_total = len(final_segments) if input_source == "file" else None
                    ok = await decode_and_send_segments(
                        websocket=websocket,
                        segments=final_segments,
                        settings=wav_settings,
                        temp_dir=temp_path,
                        request_id=request_id,
                        engine_ids=vad_engine_ids,
                        language=language,
                        beam_size=beam_size,
                        temperature=temperature,
                        vad_id=vad_id,
                        utterance_total=utterance_total,
                    )
                    if not ok:
                        return
                    for engine_id, session in list(native_streaming_sessions.items()):
                        try:
                            result = await run_in_threadpool(engine_manager.stream_finish, engine_id, session)
                        except Exception as exc:
                            LOGGER.exception("Native streaming finish failed for engine_id=%s", engine_id)
                            native_streaming_sessions.pop(engine_id, None)
                            await send_websocket_json(
                                websocket,
                                {"type": "error", "engine_id": engine_id, "message": str(exc)},
                            )
                            continue
                        if result.text.strip():
                            ok = await send_streaming_result(
                                websocket=websocket,
                                engine_id=engine_id,
                                result=result,
                                request_id=request_id,
                                mode="whisper_streaming_final",
                                audio_duration=received_audio_seconds,
                                utterance_index=native_streaming_counts[engine_id],
                                message_type="utterance_final",
                            )
                            native_streaming_counts[engine_id] += 1
                            if not ok:
                                return
                    await send_websocket_json(
                        websocket,
                        {
                            "type": "session_final",
                            "engine_ids": target_engine_ids,
                            "vad": vad_id,
                            "elapsed": round(time.perf_counter() - started_at, 3),
                        },
                    )
                    break
                if "bytes" not in message:
                    continue

                chunk_bytes = message["bytes"]
                received_bytes += len(chunk_bytes)
                if received_bytes > MAX_UPLOAD_BYTES:
                    await send_websocket_json(
                        websocket,
                        {"type": "error", "message": f"업로드 파일이 너무 큽니다. 최대 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB까지 허용합니다."},
                    )
                    await websocket.close(code=1009)
                    return

                frames, channels, sample_width, sample_rate = read_wav_chunk(chunk_bytes)
                wav_settings = (channels, sample_width, sample_rate)
                received_audio_seconds += audio_seconds_from_frames(frames, wav_settings)
                if received_audio_seconds > MAX_AUDIO_DURATION_SECONDS:
                    await send_websocket_json(
                        websocket,
                        {"type": "error", "message": f"음성이 너무 깁니다. 최대 {int(MAX_AUDIO_DURATION_SECONDS)}초까지 허용합니다."},
                    )
                    await websocket.close(code=1009)
                    return
                vad.append(frames, wav_settings)
                if not await send_status_event(websocket, target_engine_ids, "VAD 진행 중", vad_id):
                    return
                if not await send_websocket_json(websocket, {"type": "chunk_ack", "vad": vad_id}):
                    return

                for engine_id, session in list(native_streaming_sessions.items()):
                    if not await send_status_event(websocket, [engine_id], "인식 중", vad_id):
                        return
                    try:
                        result = await run_in_threadpool(engine_manager.stream_chunk, engine_id, session, frames, wav_settings)
                    except Exception as exc:
                        LOGGER.exception("Native streaming chunk failed for engine_id=%s", engine_id)
                        native_streaming_sessions.pop(engine_id, None)
                        await send_websocket_json(
                            websocket,
                            {"type": "error", "engine_id": engine_id, "message": str(exc)},
                        )
                        continue
                    if not result.text.strip():
                        continue
                    ok = await send_streaming_result(
                        websocket=websocket,
                        engine_id=engine_id,
                        result=result,
                        request_id=request_id,
                        mode="whisper_streaming_partial",
                        audio_duration=received_audio_seconds,
                        utterance_index=native_streaming_counts[engine_id],
                        message_type="partial",
                    )
                    native_streaming_counts[engine_id] += 1
                    if not ok:
                        return

                if input_source == "file":
                    continue

                ok = await decode_and_send_segments(
                    websocket=websocket,
                    segments=vad.pop_final_segments(force=False),
                    settings=wav_settings,
                    temp_dir=temp_path,
                    request_id=request_id,
                    engine_ids=vad_engine_ids,
                    language=language,
                    beam_size=beam_size,
                    temperature=temperature,
                    vad_id=vad_id,
                )
                if not ok:
                    return

                now = time.perf_counter()
                if stream_mode != "streaming":
                    continue
                if now - last_partial_at < STREAM_PARTIAL_INTERVAL_SECONDS:
                    continue
                segment = vad.current_speech_segment()
                if segment is None or not segment.pcm:
                    continue
                if vad_segment_duration(segment) < STREAM_MIN_PARTIAL_AUDIO_SECONDS:
                    continue
                last_partial_at = now
                if not await send_status_event(websocket, vad_engine_ids, "인식 중", vad_id, segment.index):
                    return
                ok = await send_segment_results(
                    websocket=websocket,
                    segment=segment,
                    settings=wav_settings,
                    temp_dir=temp_path,
                    request_id=request_id,
                    engine_ids=vad_engine_ids,
                    language=language,
                    beam_size=beam_size,
                    temperature=temperature,
                    vad_id=vad_id,
                    message_type="partial",
                    mode="vad_streaming_partial",
                    save_audio=False,
                )
                if not ok:
                    return
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await send_websocket_json(websocket, {"type": "error", "message": str(exc)})
        finally:
            for engine_id, session in locals().get("native_streaming_sessions", {}).items():
                try:
                    await run_in_threadpool(engine_manager.stream_cancel, engine_id, session)
                except Exception:
                    LOGGER.exception("Failed to cancel native streaming session for engine_id=%s", engine_id)
            SESSION_SEMAPHORE.release()
            await asyncio.sleep(0)
