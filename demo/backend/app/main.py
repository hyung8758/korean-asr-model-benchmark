import asyncio
import json
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
from app.engines import engine_manager
from app.schemas import EngineInfo, EngineStatus, TranscriptionResponse


app = FastAPI(title="한국어 STT 데모 서버")
RUNTIME_DIR = Path(os.getenv("DEMO_RUNTIME_DIR", "demo/.runtime"))
LOG_DIR = Path(os.getenv("DEMO_LOG_DIR", str(RUNTIME_DIR / "logs")))
SAVE_DIR = Path(os.getenv("DEMO_SAVE_DIR", str(RUNTIME_DIR / "saved_audio")))
EVENT_LOG_PATH = LOG_DIR / "decoding_events.jsonl"
MODEL_LOG_PATH = LOG_DIR / "model_events.jsonl"
LOG_LOCK = threading.Lock()
STREAM_PARTIAL_MIN_SECONDS = float(os.getenv("DEMO_STREAM_PARTIAL_MIN_SECONDS", "1.0"))
STREAM_PARTIAL_INTERVAL_SECONDS = float(os.getenv("DEMO_STREAM_PARTIAL_INTERVAL_SECONDS", "1.0"))
STREAM_PARTIAL_WINDOW_SECONDS = float(os.getenv("DEMO_STREAM_PARTIAL_WINDOW_SECONDS", "20.0"))

LOG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root(request: Request) -> RedirectResponse:
    frontend_url = os.getenv("DEMO_FRONTEND_PUBLIC_URL")
    if frontend_url:
        return RedirectResponse(frontend_url)
    frontend_port = os.getenv("DEMO_FRONTEND_PORT", "16010")
    return RedirectResponse(f"{request.url.scheme}://{request.url.hostname}:{frontend_port}")


def response_payload(response: TranscriptionResponse) -> dict:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response.dict()


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


def pcm_byte_duration(byte_count: int, wav_settings: tuple[int, int, int]) -> float:
    channels, sample_width, sample_rate = wav_settings
    bytes_per_second = channels * sample_width * sample_rate
    if bytes_per_second <= 0:
        return 0.0
    return byte_count / bytes_per_second


def recent_pcm(pcm_parts: list[bytes], wav_settings: tuple[int, int, int], seconds: float) -> bytes:
    channels, sample_width, sample_rate = wav_settings
    frame_size = channels * sample_width
    max_bytes = int(seconds * sample_rate * frame_size)
    max_bytes -= max_bytes % frame_size
    if max_bytes <= 0:
        return b"".join(pcm_parts)

    remaining = max_bytes
    selected = []
    for part in reversed(pcm_parts):
        if remaining <= 0:
            break
        if len(part) <= remaining:
            selected.append(part)
            remaining -= len(part)
        else:
            selected.append(part[-remaining:])
            remaining = 0
    return b"".join(reversed(selected))


async def send_websocket_json(websocket: WebSocket, payload: dict) -> bool:
    try:
        await websocket.send_json(payload)
        return True
    except (RuntimeError, WebSocketDisconnect):
        return False


@app.on_event("startup")
def preload_models() -> None:
    def log_preload_event(event: dict) -> None:
        write_event(MODEL_LOG_PATH, {"time": utc_now(), **event})

    engine_manager.preload_all(log_preload_event)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "log_dir": str(LOG_DIR), "save_dir": str(SAVE_DIR)}


@app.get("/api/engines", response_model=list[EngineInfo])
def engines() -> list[EngineInfo]:
    return engine_manager.list_engines()


@app.get("/api/engine-status", response_model=list[EngineStatus])
def engine_status() -> list[EngineStatus]:
    return engine_manager.list_statuses()


@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    engine_id: str = Form(...),
    language: str = Form("ko"),
    beam_size: int = Form(1),
    temperature: float = Form(0.0),
) -> TranscriptionResponse:
    request_id = uuid.uuid4().hex
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    audio_path = save_upload(await file.read(), suffix)
    saved_audio_path = save_audio_copy(audio_path, request_id, engine_id, "offline")
    try:
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
    except Exception as exc:
        write_event(
            EVENT_LOG_PATH,
            {
                "time": utc_now(),
                "request_id": request_id,
                "engine_id": engine_id,
                "mode": "offline",
                "saved_audio_path": str(saved_audio_path),
                "status": "error",
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        audio_path.unlink(missing_ok=True)


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
    rtf = result.decode_time / duration if duration and duration > 0 else None
    chars_per_second = len(result.text) / result.decode_time if result.decode_time > 0 else None
    audio_seconds_per_second = duration / result.decode_time if duration and result.decode_time > 0 else None
    response = TranscriptionResponse(
        request_id=request_id,
        engine=spec.name,
        model=spec.model,
        mode=mode,
        text=result.text,
        segments=result.segments,
        saved_audio_path=str(saved_audio_path) if saved_audio_path else None,
        audio_duration=duration,
        model_load_time=round(result.model_load_time, 6),
        decode_time=round(result.decode_time, 6),
        total_time=round(total_time, 6),
        rtf=round(rtf, 6) if rtf is not None else None,
        chars_per_second=round(chars_per_second, 3) if chars_per_second is not None else None,
        audio_seconds_per_second=round(audio_seconds_per_second, 3) if audio_seconds_per_second is not None else None,
        timing_source=result.timing_source,
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


@app.websocket("/api/stream")
async def stream(
    websocket: WebSocket,
    engine_id: str,
    language: str = "ko",
    beam_size: int = 1,
    temperature: float = 0.0,
) -> None:
    await websocket.accept()
    request_id = uuid.uuid4().hex
    pcm_parts: list[bytes] = []
    wav_settings: tuple[int, int, int] | None = None
    chunk_index = 0
    total_pcm_bytes = 0
    started_at = time.perf_counter()
    last_partial_at = 0.0

    with tempfile.TemporaryDirectory() as temp_dir:
        full_audio_path = Path(temp_dir) / "stream_full.wav"
        partial_audio_path = Path(temp_dir) / "stream_partial.wav"
        try:
            while True:
                message = await websocket.receive()
                if "text" in message and message["text"] == "stop":
                    if not await send_websocket_json(websocket, {"type": "finalizing"}):
                        return
                    break
                if "bytes" not in message:
                    continue

                frames, channels, sample_width, sample_rate = read_wav_chunk(message["bytes"])
                if wav_settings is None:
                    wav_settings = (channels, sample_width, sample_rate)
                elif wav_settings != (channels, sample_width, sample_rate):
                    raise ValueError("streaming chunk audio format이 중간에 변경되었습니다.")
                pcm_parts.append(frames)
                chunk_index += 1
                total_pcm_bytes += len(frames)

                duration = pcm_byte_duration(total_pcm_bytes, wav_settings)
                now = time.perf_counter()
                if duration < STREAM_PARTIAL_MIN_SECONDS:
                    continue
                if now - last_partial_at < STREAM_PARTIAL_INTERVAL_SECONDS:
                    continue

                last_partial_at = now
                write_wav(
                    partial_audio_path,
                    recent_pcm(pcm_parts, wav_settings, STREAM_PARTIAL_WINDOW_SECONDS),
                    *wav_settings,
                )
                response = await decode_file(
                    audio_path=partial_audio_path,
                    engine_id=engine_id,
                    language=language,
                    beam_size=beam_size,
                    temperature=temperature,
                    mode="streaming_partial",
                    request_id=request_id,
                    saved_audio_path=None,
                )
                if not await send_websocket_json(
                    websocket,
                    {
                        "type": "partial",
                        "chunk_index": chunk_index,
                        "elapsed": round(time.perf_counter() - started_at, 3),
                        "result": response_payload(response),
                    },
                ):
                    return

            if pcm_parts and wav_settings is not None:
                write_wav(full_audio_path, b"".join(pcm_parts), *wav_settings)
                saved_audio_path = save_audio_copy(full_audio_path, request_id, engine_id, "streaming")
                response = await decode_file(
                    audio_path=full_audio_path,
                    engine_id=engine_id,
                    language=language,
                    beam_size=beam_size,
                    temperature=temperature,
                    mode="streaming",
                    request_id=request_id,
                    saved_audio_path=saved_audio_path,
                )
                await send_websocket_json(websocket, {"type": "final", "result": response_payload(response)})
            else:
                await send_websocket_json(websocket, {"type": "final", "result": None})
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await send_websocket_json(websocket, {"type": "error", "message": str(exc)})
        finally:
            await asyncio.sleep(0)
