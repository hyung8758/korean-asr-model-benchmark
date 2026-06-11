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
from app.config import DEMO_CONFIG
from app.engines import engine_manager
from app.schemas import EngineInfo, EngineStatus, TranscriptionResponse
from app.vad import VadSegment, create_vad


app = FastAPI(title="한국어 STT 데모 서버")
DEFAULT_RUN_DIR = Path("logs") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_log"
RUN_DIR = Path(os.getenv("DEMO_RUN_DIR", str(DEFAULT_RUN_DIR)))
LOG_DIR = Path(os.getenv("DEMO_LOG_DIR", str(RUN_DIR)))
SAVE_DIR = Path(os.getenv("DEMO_SAVE_DIR", str(RUN_DIR / "saved_audio")))
EVENT_LOG_PATH = LOG_DIR / "decoding_events.jsonl"
MODEL_LOG_PATH = LOG_DIR / "model_events.jsonl"
LOG_LOCK = threading.Lock()
DEFAULTS = DEMO_CONFIG["defaults"]
SERVER_CONFIG = DEMO_CONFIG["server"]
STREAMING_CONFIG = DEMO_CONFIG["streaming"]
VAD_CONFIG = DEMO_CONFIG["vad"]
DEFAULT_LANGUAGE = str(DEFAULTS["language"])
DEFAULT_BEAM_SIZE = int(DEFAULTS["beam_size"])
DEFAULT_TEMPERATURE = float(DEFAULTS["temperature"])
STREAM_PARTIAL_INTERVAL_SECONDS = float(
    os.getenv("DEMO_STREAM_PARTIAL_INTERVAL_SECONDS", STREAMING_CONFIG["partial_interval_seconds"])
)

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
    frontend_port = os.getenv("DEMO_FRONTEND_PORT", str(SERVER_CONFIG["frontend_port"]))
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


@app.on_event("startup")
def preload_models() -> None:
    def log_preload_event(event: dict) -> None:
        write_event(MODEL_LOG_PATH, {"time": utc_now(), **event})

    engine_manager.preload_all(log_preload_event)


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


@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    engine_id: str = Form(...),
    language: str = Form(DEFAULT_LANGUAGE),
    beam_size: int = Form(DEFAULT_BEAM_SIZE),
    temperature: float = Form(DEFAULT_TEMPERATURE),
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
    request_id = uuid.uuid4().hex
    wav_settings: tuple[int, int, int] | None = None
    started_at = time.perf_counter()
    last_partial_at = 0.0

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        try:
            target_engine_ids = parse_engine_ids(engine_ids)
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
                    for segment in final_segments:
                        if segment.pcm and wav_settings is not None:
                            if not await send_status_event(
                                websocket,
                                target_engine_ids,
                                "인식 중",
                                vad_id,
                                segment.index,
                                utterance_total,
                            ):
                                return
                            ok = await send_segment_results(
                                websocket=websocket,
                                segment=segment,
                                settings=wav_settings,
                                temp_dir=temp_path,
                                request_id=request_id,
                                engine_ids=target_engine_ids,
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

                frames, channels, sample_width, sample_rate = read_wav_chunk(message["bytes"])
                wav_settings = (channels, sample_width, sample_rate)
                vad.append(frames, wav_settings)
                if not await send_status_event(websocket, target_engine_ids, "VAD 진행 중", vad_id):
                    return
                if not await send_websocket_json(websocket, {"type": "chunk_ack", "vad": vad_id}):
                    return
                if input_source == "file":
                    continue

                for segment in vad.pop_final_segments(force=False):
                    if not segment.pcm:
                        continue
                    if not await send_status_event(websocket, target_engine_ids, "인식 중", vad_id, segment.index):
                        return
                    ok = await send_segment_results(
                        websocket=websocket,
                        segment=segment,
                        settings=wav_settings,
                        temp_dir=temp_path,
                        request_id=request_id,
                        engine_ids=target_engine_ids,
                        language=language,
                        beam_size=beam_size,
                        temperature=temperature,
                        vad_id=vad_id,
                        message_type="utterance_final",
                        mode="vad_utterance",
                        save_audio=True,
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
                last_partial_at = now
                if not await send_status_event(websocket, target_engine_ids, "인식 중", vad_id, segment.index):
                    return
                ok = await send_segment_results(
                    websocket=websocket,
                    segment=segment,
                    settings=wav_settings,
                    temp_dir=temp_path,
                    request_id=request_id,
                    engine_ids=target_engine_ids,
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
            await asyncio.sleep(0)
