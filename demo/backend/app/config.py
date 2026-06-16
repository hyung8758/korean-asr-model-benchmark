import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_demo_config() -> dict[str, Any]:
    config_path = Path(os.getenv("DEMO_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
    if not config_path.is_file():
        raise FileNotFoundError(f"Demo config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Demo config must be a YAML mapping: {config_path}")
    return config


DEMO_CONFIG = load_demo_config()
