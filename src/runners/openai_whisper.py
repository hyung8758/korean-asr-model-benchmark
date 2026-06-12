import logging
from pathlib import Path
from typing import Any

from core.cuda import validate_cuda_device
from core.config import experiment_name, result_dir_for
from core.io import write_json
from decoding.audio import load_audio_array
from decoding.decode_loop import DecodeOutput, decode_rows
from decoding.run_utils import (
    fail_if_all_samples_failed,
    finish_run,
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

    def decode_one(item: dict[str, Any]) -> DecodeOutput:
        audio_input = load_audio_array(item)
        result = model.transcribe(audio_input, **config["decode_options"])
        return DecodeOutput(
            prediction_raw=result.get("text", "").strip(),
            segments=format_segments(result),
        )

    decoded_count, error_count = decode_rows(
        rows=decode_run.rows,
        config=config,
        prediction_path=decode_run.prediction_path,
        error_path=decode_run.error_path,
        done_ids=decode_run.done_ids,
        limit=args.limit,
        decode_one=decode_one,
        logger=LOGGER,
    )

    finish_run(decode_run.run_config, decoded_count, error_count)
    if str(config["device"]).startswith("cuda"):
        decode_run.run_config["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(config["device"])
    write_json(decode_run.run_config_path, decode_run.run_config)
    fail_if_all_samples_failed(decode_run.run_config)
    LOGGER.info("Wrote predictions to %s", decode_run.prediction_path)
    LOGGER.info("Wrote errors to %s", decode_run.error_path)
