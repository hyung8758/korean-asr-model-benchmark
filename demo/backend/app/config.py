import copy
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

DEFAULT_DEMO_CONFIG: dict[str, Any] = {
    "server": {
        "backend_host": "0.0.0.0",
        "backend_port": 16000,
        "frontend_host": "0.0.0.0",
        "frontend_port": 16010,
        "gunicorn_workers": 1,
    },
    "whisper_cpp": {
        "start_server": 1,
        "host": "127.0.0.1",
        "port": 8100,
        "device_index": 2,
        "threads": 4,
        "processors": 1,
        "flash_attention": 0,
        "binary": "third_party/whisper.cpp/build/bin/whisper-server",
        "model_path": "third_party/whisper.cpp/models/ggml-large-v3-q5_0.bin",
    },
    "defaults": {
        "language": "ko",
        "beam_size": 1,
        "temperature": 0.0,
        "mode": "offline",
        "vad": "silero",
    },
    "languages": [{"id": "ko", "label": "한국어"}],
    "vad_options": [{"id": "silero", "label": "Silero"}],
    "recording": {
        "chunk_ms": 1000,
        "sample_rate": 16000,
    },
    "streaming": {
        "partial_interval_seconds": 1.0,
        "status_poll_interval_ms": 1000,
    },
    "vad": {
        "silero": {
            "padding_ms": 300,
            "min_speech_ms": 250,
            "min_silence_ms": 500,
            "threshold": 0.5,
        }
    },
    "ui": {"mode_change_feedback_ms": 1000},
}


def merge_config(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_demo_config() -> dict[str, Any]:
    config_path = Path(os.getenv("DEMO_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
    if not config_path.exists():
        return copy.deepcopy(DEFAULT_DEMO_CONFIG)
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    return merge_config(DEFAULT_DEMO_CONFIG, user_config)


DEMO_CONFIG = load_demo_config()
