from pathlib import Path
from typing import Any

import requests

from app.engine_models import TranscriptionResult


class WorkerClient:
    def __init__(self, base_url: str, timeout: float | None = None, health_timeout: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.health_timeout = health_timeout

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=self.health_timeout)
        return self.checked_json(response)

    def load(self) -> dict[str, Any]:
        return self.request_json("POST", "/load")

    def unload(self) -> dict[str, Any]:
        return self.request_json("POST", "/unload")

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        beam_size: int,
        temperature: float,
    ) -> TranscriptionResult:
        with audio_path.open("rb") as handle:
            response = requests.post(
                f"{self.base_url}/transcribe",
                data={
                    "language": language,
                    "beam_size": str(beam_size),
                    "temperature": str(temperature),
                },
                files={"file": (audio_path.name, handle, "audio/wav")},
                timeout=self.timeout,
            )
        return parse_result(self.checked_json(response))

    def stream_start(self, language: str, beam_size: int, temperature: float) -> str:
        data = self.request_json(
            "POST",
            "/stream/start",
            data={
                "language": language,
                "beam_size": str(beam_size),
                "temperature": str(temperature),
            },
        )
        return str(data["session_id"])

    def stream_chunk(
        self,
        session_id: str,
        pcm: bytes,
        settings: tuple[int, int, int],
    ) -> TranscriptionResult:
        channels, sample_width, sample_rate = settings
        response = requests.post(
            f"{self.base_url}/stream/chunk",
            data={
                "session_id": session_id,
                "channels": str(channels),
                "sample_width": str(sample_width),
                "sample_rate": str(sample_rate),
            },
            files={"file": ("chunk.pcm", pcm, "application/octet-stream")},
            timeout=self.timeout,
        )
        return parse_result(self.checked_json(response))

    def stream_finish(self, session_id: str) -> TranscriptionResult:
        data = self.request_json("POST", "/stream/finish", data={"session_id": session_id})
        return parse_result(data)

    def stream_cancel(self, session_id: str) -> None:
        self.request_json("POST", "/stream/cancel", data={"session_id": session_id})

    def request_json(self, method: str, path: str, data: dict[str, str] | None = None) -> dict[str, Any]:
        response = requests.request(method, f"{self.base_url}{path}", data=data, timeout=self.timeout)
        return self.checked_json(response)

    @staticmethod
    def checked_json(response: requests.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            try:
                message = response.json().get("detail", response.text)
            except ValueError:
                message = response.text
            raise RuntimeError(str(message))
        return response.json()


def parse_result(data: dict[str, Any]) -> TranscriptionResult:
    return TranscriptionResult(
        text=str(data.get("text", "")).strip(),
        segments=list(data.get("segments", []) or []),
        decode_time=float(data.get("decode_time", 0.0)),
        model_load_time=float(data.get("model_load_time", 0.0)),
        timing_source=str(data.get("timing_source", "worker_timer")),
    )
