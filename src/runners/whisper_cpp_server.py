import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from core.config import experiment_name, result_dir_for
from core.io import append_jsonl, write_json
from decoding.audio import prepared_audio_path
from decoding.run_utils import (
    fail_if_all_samples_failed,
    finish_run,
    make_error_row,
    make_prediction_row,
    prepare_decode_run,
)


LOGGER = logging.getLogger(__name__)


def apply_overrides(config: dict[str, Any], experiment: dict[str, Any], args) -> None:
    for key in ("manifest_path", "result_root", "language", "server_binary_path", "host", "device"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = str(value)
    for key in ("port", "base_port", "device_index"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = int(value)
    if getattr(args, "model", None) is not None:
        experiment["model"] = args.model
    if getattr(args, "model_path", None) is not None:
        experiment["model_path"] = str(args.model_path)
    if getattr(args, "beam_size", None) is not None:
        experiment["beam_size"] = args.beam_size
    if getattr(args, "quantization", None) is not None:
        experiment["quantization"] = args.quantization
    decode_defaults = config.setdefault("decode_defaults", {})
    for key in ("threads", "processors"):
        value = getattr(args, key, None)
        if value is not None:
            if value < 1:
                raise ValueError(f"--{key} must be >= 1, got {value}")
            decode_defaults[key] = int(value)


def build_run_config(config: dict[str, Any], experiment: dict[str, Any], result_dir: Path | None) -> dict[str, Any]:
    quantization = experiment.get("quantization", "unknown")
    port = int(config.get("port") or config.get("base_port", 8100) + int(config.get("device_index", 0)))
    return {
        "engine": config["engine"],
        "experiment": experiment_name(experiment),
        "model": experiment["model"],
        "model_path": experiment["model_path"],
        "beam_size": int(experiment.get("beam_size", 5)),
        "quantization": quantization,
        "precision": quantization,
        "manifest_path": config["manifest_path"],
        "result_root": config["result_root"],
        "result_dir": str(result_dir) if result_dir is not None else result_dir_for(config, experiment),
        "server_binary_path": config["server_binary_path"],
        "host": config.get("host", "127.0.0.1"),
        "port": port,
        "device": config.get("device", "cuda"),
        "device_index": int(config.get("device_index", 0)),
        "language": config["language"],
        "decode_defaults": dict(config.get("decode_defaults", {})),
        "server_start_timeout_seconds": config.get("server_start_timeout_seconds", 120),
        "request_timeout_seconds": config.get("request_timeout_seconds"),
        "warmup": bool(config.get("warmup", True)),
        "require_server_timings": bool(config.get("require_server_timings", True)),
    }


def validate_runtime(config: dict[str, Any]) -> None:
    server_binary_path = Path(config["server_binary_path"])
    model_path = Path(config["model_path"])
    if not server_binary_path.is_file():
        raise FileNotFoundError(f"whisper.cpp server binary not found: {server_binary_path}")
    if not os.access(server_binary_path, os.X_OK):
        raise PermissionError(f"whisper.cpp server binary is not executable: {server_binary_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"whisper.cpp model not found: {model_path}")


def add_bool_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def server_command(config: dict[str, Any]) -> list[str]:
    defaults = config.get("decode_defaults", {})
    command = [
        config["server_binary_path"],
        "--model",
        config["model_path"],
        "--host",
        config["host"],
        "--port",
        str(config["port"]),
        "--language",
        config["language"],
        "--beam-size",
        str(config["beam_size"]),
        "--threads",
        str(defaults.get("threads", 4)),
        "--processors",
        str(defaults.get("processors", 1)),
    ]
    if config["device"] == "cpu":
        command.append("--no-gpu")
    else:
        command.extend(["--device", str(config["device_index"])])

    flash_attn = defaults.get("flash_attn")
    if flash_attn is True:
        command.append("--flash-attn")
    elif flash_attn is False:
        command.append("--no-flash-attn")
    add_bool_flag(command, bool(defaults.get("no_fallback", False)), "--no-fallback")
    add_bool_flag(command, bool(defaults.get("no_language_probabilities", True)), "--no-language-probabilities")
    command.extend(str(arg) for arg in defaults.get("server_extra_args", []))
    return command


def request_fields(config: dict[str, Any]) -> dict[str, str]:
    defaults = config.get("decode_defaults", {})
    fields = {
        "response_format": "verbose_json",
        "temperature": str(defaults.get("temperature", 0.0)),
        "temperature_inc": str(defaults.get("temperature_inc", 0.0)),
        "beam_size": str(config["beam_size"]),
        "language": config["language"],
        "no_language_probabilities": "true",
    }
    if defaults.get("no_fallback", False):
        fields["no_fallback"] = "true"
    for key, value in defaults.get("request_fields", {}).items():
        fields[str(key)] = str(value)
    return fields


def wait_for_server(config: dict[str, Any], process: subprocess.Popen) -> None:
    deadline = time.monotonic() + float(config.get("server_start_timeout_seconds") or 120)
    url = f"http://{config['host']}:{config['port']}/health"
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"whisper-server exited early with code {process.returncode}")
        try:
            response = requests.get(url, timeout=2.0)
            if response.status_code == 200 and response.json().get("status") == "ok":
                return
            last_error = response.text
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise TimeoutError(f"whisper-server did not become ready at {url}. Last error: {last_error}")


def start_server(config: dict[str, Any], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = server_command(config)
    LOGGER.info("Starting whisper-server: %s", " ".join(command))
    log_file = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=whisper_cpp_env(config))
    process._stt_log_file = log_file
    wait_for_server(config, process)
    return process


def whisper_cpp_env(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    build_dir = Path(config["server_binary_path"]).resolve().parents[1]
    library_dirs = [
        build_dir / "src",
        build_dir / "ggml" / "src",
        build_dir / "ggml" / "src" / "ggml-cuda",
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    values = [str(path) for path in library_dirs if path.exists()]
    if existing:
        values.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(values)
    return env


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    log_file = getattr(process, "_stt_log_file", None)
    if log_file is not None:
        log_file.close()


def parse_response(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]], float | None]:
    prediction_raw = str(data.get("text", "")).strip()
    segments = []
    for index, segment in enumerate(data.get("segments", [])):
        segments.append(
            {
                "id": segment.get("id", index),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": str(segment.get("text", "")).strip(),
            }
        )
    if not prediction_raw:
        prediction_raw = " ".join(segment["text"] for segment in segments).strip()
    timings = data.get("timings", {})
    inference_sec = timings.get("inference_sec")
    return prediction_raw, segments, float(inference_sec) if inference_sec is not None else None


def transcribe(server_url: str, item: dict[str, Any], fields: dict[str, str], timeout: float | None) -> tuple[dict[str, Any], float]:
    with prepared_audio_path(item) as audio_path:
        start = time.perf_counter()
        with audio_path.open("rb") as audio_file:
            files = {"file": (audio_path.name, audio_file, "audio/wav")}
            response = requests.post(server_url, data=fields, files=files, timeout=timeout)
    request_time = time.perf_counter() - start
    response.raise_for_status()
    return response.json(), request_time


def warmup_server(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if not config.get("warmup") or not rows:
        return
    server_url = f"http://{config['host']}:{config['port']}/inference"
    try:
        transcribe(server_url, rows[0], request_fields(config), config.get("request_timeout_seconds"))
        LOGGER.info("Warmup finished with id=%s", rows[0]["id"])
    except Exception:
        LOGGER.exception("Warmup failed; 벤치마크 디코딩을 계속합니다")


def decode_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    prediction_path: Path,
    error_path: Path,
    done_ids: set[str],
    limit: int | None,
) -> tuple[int, int]:
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    server_url = f"http://{config['host']}:{config['port']}/inference"
    fields = request_fields(config)
    decoded_count = 0
    error_count = 0

    with prediction_path.open("a", encoding="utf-8") as prediction_file, error_path.open(
        "a", encoding="utf-8"
    ) as error_file:
        for item in tqdm(rows, desc="Decoding"):
            if item["id"] in done_ids:
                continue
            if limit is not None and decoded_count + error_count >= limit:
                break

            start = time.perf_counter()
            try:
                data, request_time = transcribe(
                    server_url,
                    item,
                    fields,
                    config.get("request_timeout_seconds"),
                )
                prediction_raw, segments, inference_sec = parse_response(data)
                if inference_sec is None and config.get("require_server_timings", True):
                    raise RuntimeError(
                        "whisper-server response has no timings.inference_sec. "
                        "Rebuild whisper.cpp after applying the server timing patch, "
                        "or set require_server_timings=false to use HTTP request timing."
                    )
                backend_time = inference_sec if inference_sec is not None else request_time
                row = make_prediction_row(item, prediction_raw, segments, backend_time, config)
                row["request_time"] = round(request_time, 6)
                row["backend_inference_time"] = round(backend_time, 6)
                row["timing_source"] = "server_timings" if inference_sec is not None else "http_request"
                append_jsonl(prediction_file, row)
                decoded_count += 1
            except Exception as exc:
                request_time = time.perf_counter() - start
                append_jsonl(error_file, make_error_row(item, "decode_failed", str(exc), request_time, config))
                LOGGER.exception("Decode failed for id=%s", item["id"])
                error_count += 1

    return decoded_count, error_count


def run_whisper_cpp_server(config: dict[str, Any], args) -> None:
    validate_runtime(config)
    decode_run = prepare_decode_run(config, args)
    write_json(decode_run.run_config_path, decode_run.run_config)

    LOGGER.info(
        "Running whisper.cpp server model=%s device=%s device_index=%s port=%s",
        config["model_path"],
        config["device"],
        config["device_index"],
        config["port"],
    )
    LOGGER.info("Experiment=%s beam_size=%s quantization=%s", config["experiment"], config["beam_size"], config["quantization"])
    LOGGER.info("Shard %s/%s has %s samples", args.shard_index, args.num_shards, len(decode_run.rows))

    server = None
    try:
        server_log_path = decode_run.log_path.parent / f"server.shard_{args.shard_index:03d}.log"
        server = start_server(config, server_log_path)
        warmup_server(config, decode_run.rows)
        decoded_count, error_count = decode_rows(
            rows=decode_run.rows,
            config=config,
            prediction_path=decode_run.prediction_path,
            error_path=decode_run.error_path,
            done_ids=decode_run.done_ids,
            limit=args.limit,
        )
    finally:
        stop_server(server)

    finish_run(decode_run.run_config, decoded_count, error_count)
    write_json(decode_run.run_config_path, decode_run.run_config)
    fail_if_all_samples_failed(decode_run.run_config)
    LOGGER.info("Wrote predictions to %s", decode_run.prediction_path)
    LOGGER.info("Wrote errors to %s", decode_run.error_path)
