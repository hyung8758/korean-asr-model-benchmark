import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from core.config import experiment_name, result_dir_for, safe_path_part
from core.io import append_jsonl, write_json
from decoding.run_utils import (
    fail_if_all_samples_failed,
    finish_run,
    make_error_row,
    make_prediction_row,
    prepare_decode_run,
)


LOGGER = logging.getLogger(__name__)


def apply_overrides(config: dict[str, Any], experiment: dict[str, Any], args) -> None:
    for key in ("manifest_path", "result_root", "language", "binary_path", "device"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = str(value)
    if getattr(args, "device_index", None) is not None:
        config["device_index"] = args.device_index
    if getattr(args, "model", None) is not None:
        experiment["model"] = args.model
    if getattr(args, "model_path", None) is not None:
        experiment["model_path"] = str(args.model_path)
    if getattr(args, "beam_size", None) is not None:
        experiment["beam_size"] = args.beam_size
    if getattr(args, "quantization", None) is not None:
        experiment["quantization"] = args.quantization


def build_run_config(config: dict[str, Any], experiment: dict[str, Any], result_dir: Path | None) -> dict[str, Any]:
    quantization = experiment.get("quantization", "unknown")
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
        "binary_path": config["binary_path"],
        "device": config.get("device", "cuda"),
        "device_index": int(config.get("device_index", 0)),
        "language": config["language"],
        "decode_defaults": dict(config.get("decode_defaults", {})),
        "timeout_seconds": config.get("timeout_seconds"),
    }


def validate_runtime(config: dict[str, Any]) -> None:
    binary_path = Path(config["binary_path"])
    model_path = Path(config["model_path"])
    if not binary_path.is_file():
        raise FileNotFoundError(f"whisper.cpp binary not found: {binary_path}")
    if not os.access(binary_path, os.X_OK):
        raise PermissionError(f"whisper.cpp binary is not executable: {binary_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"whisper.cpp model not found: {model_path}")


def add_bool_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def build_command(config: dict[str, Any], item: dict[str, Any], output_base: Path) -> list[str]:
    defaults = config.get("decode_defaults", {})
    command = [
        config["binary_path"],
        "--model",
        config["model_path"],
        "--file",
        item["audio"],
        "--language",
        config["language"],
        "--beam-size",
        str(config["beam_size"]),
        "--temperature",
        str(defaults.get("temperature", 0.0)),
        "--output-json",
        "--output-file",
        str(output_base),
        "--no-prints",
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
    command.extend(str(arg) for arg in defaults.get("extra_args", []))
    return command


def segment_time(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return parse_timestamp(value)
    return None


def parse_timestamp(value: str) -> float | None:
    text = value.strip().replace(",", ".")
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return None


def parse_segments(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = data.get("transcription") or data.get("segments") or []
    segments = []
    for index, segment in enumerate(raw_segments):
        timestamps = segment.get("timestamps", {})
        offsets = segment.get("offsets", {})
        start = segment_time(segment.get("start", timestamps.get("from")))
        end = segment_time(segment.get("end", timestamps.get("to")))
        if start is None:
            start = offset_to_seconds(offsets.get("from"))
        if end is None:
            end = offset_to_seconds(offsets.get("to"))
        text = str(segment.get("text", "")).strip()
        segments.append(
            {
                "id": segment.get("id", index),
                "start": start,
                "end": end,
                "text": text,
            }
        )
    return segments


def offset_to_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value) / 1000.0
    return parse_timestamp(str(value))


def parse_prediction(output_json: Path) -> tuple[str, list[dict[str, Any]]]:
    data = json.loads(output_json.read_bytes().decode("utf-8", errors="replace"))
    text = str(data.get("text", "")).strip()
    segments = parse_segments(data)
    if not text:
        text = " ".join(segment["text"] for segment in segments).strip()
    return text, segments


def decode_process_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_command(command: list[str], timeout_seconds: float | None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        timeout=timeout_seconds,
    )


def decode_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    prediction_path: Path,
    error_path: Path,
    done_ids: set[str],
    temp_dir: Path,
    limit: int | None,
) -> tuple[int, int]:
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
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

            output_base = temp_dir / safe_path_part(item["id"])
            output_json = output_base.with_suffix(".json")
            if output_json.exists():
                output_json.unlink()

            start = time.perf_counter()
            command = build_command(config, item, output_base)
            try:
                completed = run_command(command, config.get("timeout_seconds"))
                decode_time = time.perf_counter() - start
                if completed.returncode != 0:
                    stderr = decode_process_output(completed.stderr)
                    stdout = decode_process_output(completed.stdout)
                    error = "\n".join(part for part in (stderr, stdout) if part).strip()
                    append_jsonl(error_file, make_error_row(item, "decode_failed", error, decode_time, config))
                    error_count += 1
                    continue
                if not output_json.exists():
                    append_jsonl(
                        error_file,
                        make_error_row(item, "missing_output_json", str(output_json), decode_time, config),
                    )
                    error_count += 1
                    continue
                prediction_raw, segments = parse_prediction(output_json)
                append_jsonl(
                    prediction_file,
                    make_prediction_row(item, prediction_raw, segments, decode_time, config),
                )
                decoded_count += 1
            except subprocess.TimeoutExpired as exc:
                decode_time = time.perf_counter() - start
                append_jsonl(error_file, make_error_row(item, "decode_timeout", str(exc), decode_time, config))
                error_count += 1
            except Exception as exc:
                decode_time = time.perf_counter() - start
                append_jsonl(error_file, make_error_row(item, "decode_failed", str(exc), decode_time, config))
                LOGGER.exception("Decode failed for id=%s", item["id"])
                error_count += 1
            finally:
                if output_json.exists():
                    output_json.unlink()

    return decoded_count, error_count


def run_whisper_cpp(config: dict[str, Any], args) -> None:
    validate_runtime(config)

    decode_run = prepare_decode_run(config, args)
    temp_dir = decode_run.result_dir / "tmp" / f"shard_{args.shard_index:03d}"
    write_json(decode_run.run_config_path, decode_run.run_config)

    LOGGER.info(
        "Running whisper.cpp binary=%s model=%s device=%s device_index=%s",
        config["binary_path"],
        config["model_path"],
        config["device"],
        config["device_index"],
    )
    LOGGER.info("Experiment=%s beam_size=%s quantization=%s", config["experiment"], config["beam_size"], config["quantization"])
    LOGGER.info("Shard %s/%s has %s samples", args.shard_index, args.num_shards, len(decode_run.rows))

    decoded_count, error_count = decode_rows(
        rows=decode_run.rows,
        config=config,
        prediction_path=decode_run.prediction_path,
        error_path=decode_run.error_path,
        done_ids=decode_run.done_ids,
        temp_dir=temp_dir,
        limit=args.limit,
    )

    finish_run(decode_run.run_config, decoded_count, error_count)
    write_json(decode_run.run_config_path, decode_run.run_config)
    fail_if_all_samples_failed(decode_run.run_config)
    LOGGER.info("Wrote predictions to %s", decode_run.prediction_path)
    LOGGER.info("Wrote errors to %s", decode_run.error_path)
