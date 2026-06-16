import logging
from pathlib import Path
from typing import Any

from core.cuda import validate_cuda_device
from core.config import experiment_name, result_dir_for
from core.io import write_json
from core.precision import SUPPORTED_TORCH_PRECISIONS, torch_dtype_from_precision
from data.audio import TARGET_SAMPLE_RATE, load_audio_array
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


def build_transformers_options(config: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(config.get("decode_defaults", {}))
    return {
        "chunk_length_s": defaults.get("chunk_length_s", 30),
        "batch_size": int(defaults.get("batch_size", 1)),
        "return_timestamps": experiment.get("return_timestamps", defaults.get("return_timestamps", False)),
        "generate_kwargs": {
            "language": config.get("language", "ko"),
            "task": defaults.get("task", "transcribe"),
            "num_beams": int(experiment.get("beam_size", defaults.get("beam_size", 1))),
        },
    }


def build_run_config(config: dict[str, Any], experiment: dict[str, Any], result_dir: Path | None) -> dict[str, Any]:
    precision = experiment.get("precision", "float16")
    if precision not in SUPPORTED_TORCH_PRECISIONS:
        raise ValueError(f"Unsupported precision={precision}. Supported values: {sorted(SUPPORTED_TORCH_PRECISIONS)}")

    return {
        "engine": config["engine"],
        "runner": "huggingface_transformers",
        "experiment": experiment_name(experiment),
        "model": experiment["model"],
        "beam_size": int(experiment.get("beam_size", config.get("decode_defaults", {}).get("beam_size", 1))),
        "precision": precision,
        "manifest_path": config["manifest_path"],
        "result_root": config["result_root"],
        "result_dir": str(result_dir) if result_dir is not None else result_dir_for(config, experiment),
        "device": config["device"],
        "language": config["language"],
        "transformers_options": build_transformers_options(config, experiment),
    }


def format_pipeline_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    segments = []
    for index, chunk in enumerate(result.get("chunks", []) or []):
        timestamp = chunk.get("timestamp") or (None, None)
        start, end = timestamp if len(timestamp) == 2 else (None, None)
        segments.append(
            {
                "id": index,
                "start": start,
                "end": end,
                "text": str(chunk.get("text", "")).strip(),
            }
        )
    return segments


def load_audio_for_transformers_pipeline(item: dict[str, Any]) -> dict[str, Any]:
    return {"array": load_audio_array(item), "sampling_rate": TARGET_SAMPLE_RATE}


def load_transformers_model(config: dict[str, Any]):
    import transformers.pipelines.automatic_speech_recognition as speech_recognition_pipeline
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    speech_recognition_pipeline.is_torchcodec_available = lambda: False

    dtype = torch_dtype_from_precision(config["precision"])
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        config["model"],
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(config["device"])
    processor = AutoProcessor.from_pretrained(config["model"])
    update_whisper_generation_config(model, processor)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=config["device"],
        chunk_length_s=config["transformers_options"]["chunk_length_s"],
        batch_size=config["transformers_options"]["batch_size"],
        return_timestamps=config["transformers_options"]["return_timestamps"],
    )


def update_whisper_generation_config(model: Any, processor: Any) -> None:
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    generation_config = model.generation_config

    if not hasattr(generation_config, "is_multilingual"):
        generation_config.is_multilingual = True
    if not hasattr(generation_config, "lang_to_id"):
        generation_config.lang_to_id = {
            token: token_id
            for token, token_id in vocab.items()
            if token.startswith("<|") and token.endswith("|>") and len(token) == 6
        }
    if not hasattr(generation_config, "task_to_id"):
        generation_config.task_to_id = {
            "translate": tokenizer.convert_tokens_to_ids("<|translate|>"),
            "transcribe": tokenizer.convert_tokens_to_ids("<|transcribe|>"),
        }


def run_huggingface_transformers(config: dict[str, Any], args) -> None:
    import torch

    validate_cuda_device(str(config["device"]))

    decode_run = prepare_decode_run(config, args)
    write_json(decode_run.run_config_path, decode_run.run_config)

    LOGGER.info("Loading Hugging Face transformers model=%s device=%s", config["model"], config["device"])
    LOGGER.info("Experiment=%s precision=%s beam_size=%s", config["experiment"], config["precision"], config["beam_size"])
    LOGGER.info("Shard %s/%s has %s samples", args.shard_index, args.num_shards, len(decode_run.rows))

    model = load_transformers_model(config)
    if str(config["device"]).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(config["device"])

    def decode_one(item: dict[str, Any]) -> DecodeOutput:
        audio_input = load_audio_for_transformers_pipeline(item)
        result = model(
            audio_input,
            generate_kwargs=config["transformers_options"]["generate_kwargs"],
        )
        return DecodeOutput(
            prediction_raw=str(result.get("text", "")).strip(),
            segments=format_pipeline_segments(result),
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
