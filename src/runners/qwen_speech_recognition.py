import logging
from pathlib import Path
from typing import Any

from core.cuda import validate_cuda_device
from core.config import experiment_name, result_dir_for
from core.io import write_json
from core.precision import SUPPORTED_TORCH_PRECISIONS, torch_dtype_from_precision
from decoding.decode_loop import DecodeOutput, decode_rows
from decoding.run_utils import (
    fail_if_all_samples_failed,
    finish_run,
    prepare_decode_run,
)


LOGGER = logging.getLogger(__name__)


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


def build_qwen_options(config: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(config.get("decode_defaults", {}))
    return {
        "language": experiment.get("qwen_language", defaults.get("qwen_language", "Korean")),
        "return_time_stamps": bool(experiment.get("return_timestamps", defaults.get("return_timestamps", False))),
        "max_inference_batch_size": int(defaults.get("max_inference_batch_size", 1)),
        "max_new_tokens": int(experiment.get("max_new_tokens", defaults.get("max_new_tokens", 256))),
        "attn_implementation": defaults.get("attn_implementation"),
    }


def build_run_config(config: dict[str, Any], experiment: dict[str, Any], result_dir: Path | None) -> dict[str, Any]:
    precision = experiment.get("precision", "bfloat16")
    if precision not in SUPPORTED_TORCH_PRECISIONS:
        raise ValueError(f"Unsupported precision={precision}. Supported values: {sorted(SUPPORTED_TORCH_PRECISIONS)}")

    return {
        "engine": config["engine"],
        "runner": "qwen_speech_recognition",
        "experiment": experiment_name(experiment),
        "model": experiment["model"],
        "beam_size": int(experiment.get("beam_size", config.get("decode_defaults", {}).get("beam_size", 1))),
        "precision": precision,
        "manifest_path": config["manifest_path"],
        "result_root": config["result_root"],
        "result_dir": str(result_dir) if result_dir is not None else result_dir_for(config, experiment),
        "device": config["device"],
        "language": config["language"],
        "qwen_options": build_qwen_options(config, experiment),
    }


def format_qwen_segments(result: Any) -> list[dict[str, Any]]:
    time_stamps = getattr(result, "time_stamps", None) or []
    segments = []
    for index, stamp in enumerate(time_stamps):
        text = getattr(stamp, "text", "")
        start = getattr(stamp, "start_time", None)
        end = getattr(stamp, "end_time", None)
        if isinstance(stamp, dict):
            text = stamp.get("text", text)
            start = stamp.get("start_time", stamp.get("start", start))
            end = stamp.get("end_time", stamp.get("end", end))
        segments.append({"id": index, "start": start, "end": end, "text": str(text).strip()})
    return segments


def load_qwen_model(config: dict[str, Any]):
    from qwen_asr import Qwen3ASRModel

    dtype = torch_dtype_from_precision(config["precision"])
    options = config["qwen_options"]
    init_kwargs = {
        "dtype": dtype,
        "device_map": config["device"],
        "max_inference_batch_size": options["max_inference_batch_size"],
        "max_new_tokens": options["max_new_tokens"],
    }
    if options.get("attn_implementation"):
        init_kwargs["attn_implementation"] = options["attn_implementation"]
    return Qwen3ASRModel.from_pretrained(config["model"], **init_kwargs)


def run_qwen_speech_recognition(config: dict[str, Any], args) -> None:
    import torch

    validate_cuda_device(str(config["device"]))

    decode_run = prepare_decode_run(config, args)
    write_json(decode_run.run_config_path, decode_run.run_config)

    LOGGER.info("Loading Qwen speech recognition model=%s device=%s", config["model"], config["device"])
    LOGGER.info("Experiment=%s precision=%s", config["experiment"], config["precision"])
    LOGGER.info("Shard %s/%s has %s samples", args.shard_index, args.num_shards, len(decode_run.rows))

    model = load_qwen_model(config)
    if str(config["device"]).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(config["device"])

    def decode_one(item: dict[str, Any]) -> DecodeOutput:
        results = model.transcribe(
            audio=item["audio"],
            language=config["qwen_options"]["language"],
            return_time_stamps=config["qwen_options"]["return_time_stamps"],
        )
        result = results[0]
        return DecodeOutput(
            prediction_raw=str(getattr(result, "text", "")).strip(),
            segments=format_qwen_segments(result),
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
