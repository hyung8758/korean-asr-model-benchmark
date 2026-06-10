from pydantic import BaseModel


class EngineInfo(BaseModel):
    id: str
    name: str
    provider: str
    model: str
    device: str
    supports_offline: bool = True
    supports_streaming: bool = True
    note: str = ""


class EngineStatus(BaseModel):
    id: str
    state: str
    label: str
    load_time: float | None = None
    error: str = ""


class TranscriptionResponse(BaseModel):
    request_id: str
    engine: str
    model: str
    mode: str
    text: str
    segments: list[dict]
    saved_audio_path: str | None = None
    audio_duration: float | None
    model_load_time: float = 0.0
    decode_time: float
    total_time: float
    rtf: float | None
    chars_per_second: float | None = None
    audio_seconds_per_second: float | None = None
    timing_source: str
