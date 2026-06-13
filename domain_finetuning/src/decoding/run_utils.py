from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import output_paths
from core.io import read_jsonl
from core.sharding import select_shard
from core.text import normalize_text


@dataclass
class DecodeRun:
    result_dir: Path
    prediction_path: Path
    error_path: Path
    run_config_path: Path
    log_path: Path
    rows: list[dict[str, Any]]
    done_ids: set[str]
    run_config: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_finished_ids(prediction_path: Path, error_path: Path, resume: bool, retry_errors: bool) -> set[str]:
    if not resume:
        return set()

    done = set()
    for row in read_jsonl(prediction_path):
        done.add(row["id"])
    if not retry_errors:
        for row in read_jsonl(error_path):
            done.add(row["id"])
    return done


def reset_outputs_if_needed(prediction_path: Path, error_path: Path, resume: bool) -> None:
    if resume:
        return
    for path in (prediction_path, error_path):
        if path.exists():
            path.unlink()


def prepare_decode_run(config: dict[str, Any], args) -> DecodeRun:
    result_dir = Path(config["result_dir"])
    prediction_path, error_path, run_config_path, log_path = output_paths(
        result_dir,
        args.num_shards,
        args.shard_index,
    )
    resume = not args.no_resume
    retry_errors = bool(getattr(args, "retry_errors", False))
    reset_outputs_if_needed(prediction_path, error_path, resume=resume)

    manifest_rows = read_jsonl(Path(config["manifest_path"]))
    rows = select_shard(manifest_rows, args.num_shards, args.shard_index)
    done_ids = load_finished_ids(prediction_path, error_path, resume=resume, retry_errors=retry_errors)

    run_config = {
        **config,
        "prediction_path": str(prediction_path),
        "error_path": str(error_path),
        "log_path": str(log_path),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "limit": args.limit,
        "resume": resume,
        "retry_errors": retry_errors,
        "manifest_samples": len(manifest_rows),
        "shard_samples": len(rows),
        "already_done": len(done_ids),
        "started_at": utc_now(),
    }

    return DecodeRun(
        result_dir=result_dir,
        prediction_path=prediction_path,
        error_path=error_path,
        run_config_path=run_config_path,
        log_path=log_path,
        rows=rows,
        done_ids=done_ids,
        run_config=run_config,
    )


def result_metadata(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "engine",
        "runner",
        "experiment",
        "model",
        "model_path",
        "beam_size",
        "precision",
        "compute_type",
        "quantization",
        "language",
    )
    return {key: config[key] for key in keys if key in config}


def make_prediction_row(
    item: dict[str, Any],
    prediction_raw: str,
    segments: list[dict[str, Any]],
    decode_time: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    duration = float(item["duration"])
    row = {
        "id": item["id"],
        "audio": item["audio"],
        "reference": item["text"],
        "reference_raw": item.get("text_raw", ""),
        "prediction": normalize_text(prediction_raw),
        "prediction_raw": prediction_raw,
        "dataset": item["dataset"],
        "bucket": item["bucket"],
        "speaker": item.get("speaker", ""),
        "split": item.get("split", "unknown"),
        "duration": duration,
        "decode_time": round(decode_time, 6),
        "rtf": round(decode_time / duration, 6) if duration > 0 else None,
        "segments": segments,
        **result_metadata(config),
    }
    copy_optional_item_fields(row, item)
    return row


def make_error_row(
    item: dict[str, Any],
    reason: str,
    error: str,
    decode_time: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "id": item["id"],
        "audio": item["audio"],
        "dataset": item["dataset"],
        "bucket": item["bucket"],
        "duration": item.get("duration"),
        "decode_time": round(decode_time, 6),
        "reason": reason,
        "error": error,
        **result_metadata(config),
    }
    copy_optional_item_fields(row, item)
    return row


def copy_optional_item_fields(row: dict[str, Any], item: dict[str, Any]) -> None:
    optional_fields = (
        "source_audio",
        "source_text",
        "audio_sample_rate",
        "source_sample_rate",
        "audio_start",
        "audio_end",
        "source_audio_start",
        "source_audio_end",
        "finetuning_split",
    )
    for field in optional_fields:
        if field in item:
            row[field] = item[field]


def finish_run(run_config: dict[str, Any], decoded_count: int, error_count: int) -> None:
    run_config["completed_at"] = utc_now()
    run_config["decoded_count"] = decoded_count
    run_config["error_count"] = error_count


def fail_if_all_samples_failed(run_config: dict[str, Any]) -> None:
    decoded_count = int(run_config.get("decoded_count") or 0)
    error_count = int(run_config.get("error_count") or 0)
    if decoded_count > 0 or error_count == 0:
        return

    error_path = run_config.get("error_path")
    first_error = ""
    if error_path:
        for row in read_jsonl(Path(error_path)):
            first_error = str(row.get("error", "")).strip()
            break

    message = (
        "All samples failed to decode. "
        f"See error_path={error_path} and log_path={run_config.get('log_path')}."
    )
    if first_error:
        message += f" First error: {first_error[:1200]}"
    raise RuntimeError(message)
