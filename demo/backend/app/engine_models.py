from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class EngineSpec:
    id: str
    name: str
    provider: str
    kind: str
    model: str
    device: str = ""
    compute_type: str = "float16"
    precision: str = "float16"
    return_timestamps: bool = True
    server_url: str = ""
    server_host: str = "127.0.0.1"
    server_port: int = 8100
    server_binary: str = "third_party/whisper.cpp/build/bin/whisper-server"
    server_model_path: str = "third_party/whisper.cpp/models/ggml-large-v3-q5_0.bin"
    server_threads: int = 4
    server_processors: int = 1
    server_flash_attention: bool = False
    model_options: tuple[str, ...] = ()
    language_options: tuple[str, ...] = ("ko",)
    theme: str = "default"
    streaming_min_chunk_seconds: float = 1.0
    streaming_buffer_trimming_seconds: float = 15.0
    note: str = ""


@dataclass
class TranscriptionResult:
    text: str
    segments: list[dict[str, Any]]
    decode_time: float
    model_load_time: float = 0.0
    timing_source: str = "python_timer"


def project_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def parse_gpu_indices(config: dict[str, Any]) -> list[int]:
    values = config.get("resources", {}).get("gpu_indices", [])
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",") if part.strip()]
    return [int(value) for value in values]


def engine_specs_from_config(config: dict[str, Any]) -> list[EngineSpec]:
    language_options = tuple(row["id"] for row in config.get("languages", [{"id": "ko"}]))
    specs = []
    for row in config.get("engines", []):
        server = row.get("server", {})
        streaming = row.get("streaming", {})
        model = str(row["model"])
        specs.append(
            EngineSpec(
                id=str(row["id"]),
                name=str(row["name"]),
                provider=str(row.get("provider", row["name"])),
                kind=str(row["kind"]),
                model=model,
                compute_type=str(row.get("compute_type", "float16")),
                precision=str(row.get("precision", "float16")),
                return_timestamps=bool_value(row.get("return_timestamps"), True),
                server_host=str(server.get("host", "127.0.0.1")),
                server_port=int(server.get("port", 8100)),
                server_binary=str(server.get("binary", "third_party/whisper.cpp/build/bin/whisper-server")),
                server_model_path=str(server.get("model_path", "third_party/whisper.cpp/models/ggml-large-v3-q5_0.bin")),
                server_threads=int(server.get("threads", 4)),
                server_processors=int(server.get("processors", 1)),
                server_flash_attention=bool_value(server.get("flash_attention"), False),
                model_options=tuple(row.get("model_options") or [model]),
                language_options=tuple(row.get("languages") or language_options),
                theme=str(row.get("theme", "default")),
                streaming_min_chunk_seconds=float(streaming.get("min_chunk_seconds", 1.0)),
                streaming_buffer_trimming_seconds=float(streaming.get("buffer_trimming_seconds", 15.0)),
                note=str(row.get("note", "")),
            )
        )
    if not specs:
        raise RuntimeError("demo/config.yaml must define at least one engine.")
    return specs


def torch_dtype(precision: str):
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
