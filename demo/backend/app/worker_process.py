import argparse
import os
import tempfile
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.config import DEMO_CONFIG
from app.engine_models import EngineSpec, TranscriptionResult, engine_specs_from_config
from app.engine_transcribers import load_engine_model, transcribe_with_engine
from app.streaming_sessions import StreamingSession, create_streaming_session, supports_native_streaming
from app.whisper_cpp_server import WhisperCppServerController


class WorkerState:
    def __init__(self, spec: EngineSpec, gpu_index: int):
        self.spec = runtime_spec(spec, gpu_index)
        self.gpu_index = gpu_index
        self.model: Any = None
        self.model_load_time: float | None = None
        self.state = "not_loaded"
        self.error = ""
        self.sessions: dict[str, StreamingSession] = {}
        self.whisper_cpp_server = WhisperCppServerController()
        self.lock = threading.RLock()

    def health(self) -> dict[str, Any]:
        return {
            "engine_id": self.spec.id,
            "pid": os.getpid(),
            "state": self.state,
            "error": self.error,
            "model_load_time": self.model_load_time,
            "gpu_index": self.gpu_index,
        }

    def load(self) -> dict[str, Any]:
        with self.lock:
            if self.state == "ready":
                return self.health()
            self.state = "loading"
            self.error = ""
            start = time.perf_counter()
            try:
                if self.spec.kind == "whisper_cpp_server":
                    self.whisper_cpp_server.ensure_running(self.spec, self.gpu_index)
                    self.model = "whisper_cpp_server"
                else:
                    self.model = load_engine_model(self.spec)
                self.model_load_time = time.perf_counter() - start
                self.state = "ready"
                return self.health()
            except Exception as exc:
                self.state = "error"
                self.error = str(exc)
                raise

    def unload(self) -> dict[str, Any]:
        with self.lock:
            self.sessions.clear()
            if self.spec.kind == "whisper_cpp_server":
                self.whisper_cpp_server.stop(self.spec.id)
            self.model = None
            self.model_load_time = None
            self.state = "not_loaded"
            self.error = ""
            return self.health()

    def get_model(self, spec: EngineSpec) -> tuple[Any, float]:
        if spec.id != self.spec.id:
            raise ValueError(f"worker engine mismatch: {spec.id} != {self.spec.id}")
        if self.model is None:
            self.load()
        return self.model, 0.0

    def transcribe(self, audio_path: Path, language: str, beam_size: int, temperature: float) -> TranscriptionResult:
        with self.lock:
            if self.state != "ready":
                self.load()
            self.state = "decoding"
            try:
                return transcribe_with_engine(
                    spec=self.spec,
                    get_model=self.get_model,
                    audio_path=audio_path,
                    language=language,
                    beam_size=beam_size,
                    temperature=temperature,
                )
            except Exception as exc:
                self.state = "error"
                self.error = str(exc)
                raise
            finally:
                if self.state != "error":
                    self.state = "ready"

    def start_stream(self, language: str, beam_size: int, temperature: float) -> str:
        with self.lock:
            if not supports_native_streaming(self.spec):
                raise ValueError(f"{self.spec.name}은 native streaming을 지원하지 않습니다.")
            if self.state != "ready":
                self.load()
            session_id = uuid.uuid4().hex
            self.sessions[session_id] = create_streaming_session(
                self.spec,
                self.get_model,
                language,
                beam_size,
                temperature,
            )
            return session_id

    def stream_chunk(self, session_id: str, pcm: bytes, settings: tuple[int, int, int]) -> TranscriptionResult:
        with self.lock:
            session = self.get_session(session_id)
            self.state = "decoding"
            try:
                return session.insert_pcm(pcm, settings)
            except Exception as exc:
                self.state = "error"
                self.error = str(exc)
                raise
            finally:
                if self.state != "error":
                    self.state = "ready"

    def finish_stream(self, session_id: str) -> TranscriptionResult:
        with self.lock:
            session = self.get_session(session_id)
            self.state = "decoding"
            try:
                return session.finish()
            except Exception as exc:
                self.state = "error"
                self.error = str(exc)
                raise
            finally:
                self.sessions.pop(session_id, None)
                if self.state != "error":
                    self.state = "ready"

    def cancel_stream(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            self.sessions.pop(session_id, None)
            return {"session_id": session_id, "cancelled": True}

    def get_session(self, session_id: str) -> StreamingSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown streaming session: {session_id}")
        return session


def runtime_spec(spec: EngineSpec, gpu_index: int) -> EngineSpec:
    if spec.kind == "whisper_cpp_server":
        server_url = f"http://{spec.server_host}:{spec.server_port}/inference"
        return replace(spec, device=f"cuda:{gpu_index}/server", server_url=server_url)
    return replace(spec, device=f"cuda:{gpu_index}")


def result_payload(result: TranscriptionResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "segments": result.segments,
        "decode_time": result.decode_time,
        "model_load_time": result.model_load_time,
        "timing_source": result.timing_source,
    }


def find_spec(engine_id: str) -> EngineSpec:
    specs = {spec.id: spec for spec in engine_specs_from_config(DEMO_CONFIG)}
    if engine_id not in specs:
        names = ", ".join(specs)
        raise ValueError(f"unknown engine_id: {engine_id}. available: {names}")
    return specs[engine_id]


def create_app(state: WorkerState) -> FastAPI:
    app = FastAPI(title=f"STT worker: {state.spec.id}")

    @app.on_event("shutdown")
    def shutdown() -> None:
        state.unload()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return state.health()

    @app.post("/load")
    def load() -> dict[str, Any]:
        try:
            return state.load()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/unload")
    def unload() -> dict[str, Any]:
        return state.unload()

    @app.post("/transcribe")
    async def transcribe(
        file: UploadFile = File(...),
        language: str = Form("ko"),
        beam_size: int = Form(1),
        temperature: float = Form(0.0),
    ) -> dict[str, Any]:
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            audio_path = Path(handle.name)
            handle.write(await file.read())
        try:
            return result_payload(state.transcribe(audio_path, language, beam_size, temperature))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            audio_path.unlink(missing_ok=True)

    @app.post("/stream/start")
    def stream_start(
        language: str = Form("ko"),
        beam_size: int = Form(1),
        temperature: float = Form(0.0),
    ) -> dict[str, Any]:
        try:
            return {"session_id": state.start_stream(language, beam_size, temperature)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/stream/chunk")
    async def stream_chunk(
        file: UploadFile = File(...),
        session_id: str = Form(...),
        channels: int = Form(...),
        sample_width: int = Form(...),
        sample_rate: int = Form(...),
    ) -> dict[str, Any]:
        try:
            result = state.stream_chunk(session_id, await file.read(), (channels, sample_width, sample_rate))
            return result_payload(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/stream/finish")
    def stream_finish(session_id: str = Form(...)) -> dict[str, Any]:
        try:
            return result_payload(state.finish_stream(session_id))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/stream/cancel")
    def stream_cancel(session_id: str = Form(...)) -> dict[str, Any]:
        return state.cancel_stream(session_id)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-id", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(WorkerState(find_spec(args.engine_id), args.gpu_index))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
