import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from core.cuda import validate_cuda_device
from core.config import experiment_name, result_dir_for
from core.io import append_jsonl, write_json
from decoding.run_utils import (
    fail_if_all_samples_failed,
    finish_run,
    make_error_row,
    make_prediction_row,
    prepare_decode_run,
)


LOGGER = logging.getLogger(__name__)
SUPPORTED_PRECISIONS = {"fp16", "fp32"}


def apply_overrides(config: dict[str, Any], experiment: dict[str, Any], args) -> None:
    for key in ("manifest_path", "result_root", "device", "language"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = str(value)
    if getattr(args, "model", None) is not None:
        experiment["model"] = args.model
    if getattr(args, "beam_size", None) is not None:
        experiment["beam_size"] = args.beam_size
    if getattr(args, "precision", None) is not None:
        experiment["precision"] = args.precision


def build_decode_options(config: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    precision = experiment.get("precision", "fp16")
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"openai_whisper supports {sorted(SUPPORTED_PRECISIONS)}, got precision={precision}. "
            "Keep int8 experiments in the shared config for engines that support it, "
            "such as faster-whisper or whisper.cpp."
        )

    decode_options = dict(config["decode_defaults"])
    decode_options["language"] = config["language"]
    decode_options["beam_size"] = int(experiment.get("beam_size", 5))
    decode_options["fp16"] = precision == "fp16" and str(config["device"]).startswith("cuda")
    return decode_options


def build_run_config(config: dict[str, Any], experiment: dict[str, Any], result_dir: Path | None) -> dict[str, Any]:
    return {
        "engine": config["engine"],
        "experiment": experiment_name(experiment),
        "model": experiment["model"],
        "beam_size": int(experiment.get("beam_size", 5)),
        "precision": experiment.get("precision", "fp16"),
        "manifest_path": config["manifest_path"],
        "result_root": config["result_root"],
        "result_dir": str(result_dir) if result_dir is not None else result_dir_for(config, experiment),
        "device": config["device"],
        "language": config["language"],
        "decode_options": build_decode_options(config, experiment),
    }


def format_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    segments = []
    for segment in result.get("segments", []):
        segments.append(
            {
                "id": segment.get("id"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text", ""),
            }
        )
    return segments


def decode_rows(
    rows: list[dict[str, Any]],
    model,
    config: dict[str, Any],
    prediction_path: Path,
    error_path: Path,
    done_ids: set[str],
    limit: int | None,
) -> tuple[int, int]:
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
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
                result = model.transcribe(item["audio"], **config["decode_options"])
                decode_time = time.perf_counter() - start
                append_jsonl(
                    prediction_file,
                    make_prediction_row(
                        item=item,
                        prediction_raw=result.get("text", "").strip(),
                        segments=format_segments(result),
                        decode_time=decode_time,
                        config=config,
                    ),
                )
                decoded_count += 1
            except Exception as exc:
                decode_time = time.perf_counter() - start
                append_jsonl(error_file, make_error_row(item, "decode_failed", str(exc), decode_time, config))
                LOGGER.exception("Decode failed for id=%s", item["id"])
                error_count += 1

    return decoded_count, error_count


def run_openai_whisper(config: dict[str, Any], args) -> None:
    import torch
    import whisper

    validate_cuda_device(str(config["device"]))

    decode_run = prepare_decode_run(config, args)
    write_json(decode_run.run_config_path, decode_run.run_config)

    LOGGER.info("Loading OpenAI Whisper model=%s device=%s", config["model"], config["device"])
    LOGGER.info("Experiment=%s beam_size=%s precision=%s", config["experiment"], config["beam_size"], config["precision"])
    LOGGER.info("Shard %s/%s has %s samples", args.shard_index, args.num_shards, len(decode_run.rows))

    model = whisper.load_model(config["model"], device=config["device"])
    if str(config["device"]).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(config["device"])

    decoded_count, error_count = decode_rows(
        rows=decode_run.rows,
        model=model,
        config=config,
        prediction_path=decode_run.prediction_path,
        error_path=decode_run.error_path,
        done_ids=decode_run.done_ids,
        limit=args.limit,
    )

    finish_run(decode_run.run_config, decoded_count, error_count)
    if str(config["device"]).startswith("cuda"):
        decode_run.run_config["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(config["device"])
    write_json(decode_run.run_config_path, decode_run.run_config)
    fail_if_all_samples_failed(decode_run.run_config)
    LOGGER.info("Wrote predictions to %s", decode_run.prediction_path)
    LOGGER.info("Wrote errors to %s", decode_run.error_path)
