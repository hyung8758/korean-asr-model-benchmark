import logging
from pathlib import Path
from typing import Any

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
TRANSCRIBE_OPTIONS = {
    "task",
    "language",
    "beam_size",
    "best_of",
    "patience",
    "length_penalty",
    "temperature",
    "compression_ratio_threshold",
    "log_prob_threshold",
    "no_speech_threshold",
    "condition_on_previous_text",
    "prompt_reset_on_temperature",
    "initial_prompt",
    "prefix",
    "suppress_blank",
    "suppress_tokens",
    "without_timestamps",
    "max_initial_timestamp",
    "word_timestamps",
    "prepend_punctuations",
    "append_punctuations",
    "vad_filter",
    "vad_parameters",
    "max_new_tokens",
    "chunk_length",
    "clip_timestamps",
    "hallucination_silence_threshold",
}


def apply_overrides(config: dict[str, Any], experiment: dict[str, Any], args) -> None:
    for key in ("manifest_path", "result_root", "language"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = str(value)
    if getattr(args, "device", None) is not None:
        config["device"] = args.device
    if getattr(args, "device_index", None) is not None:
        config["device_index"] = args.device_index
    if getattr(args, "model", None) is not None:
        experiment["model"] = args.model
    if getattr(args, "beam_size", None) is not None:
        experiment["beam_size"] = args.beam_size
    if getattr(args, "compute_type", None) is not None:
        experiment["compute_type"] = args.compute_type


def build_transcribe_options(config: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    options = {
        key: value
        for key, value in config.get("decode_defaults", {}).items()
        if key in TRANSCRIBE_OPTIONS
    }
    options["language"] = config["language"]
    options["beam_size"] = int(experiment.get("beam_size", 5))
    return options


def build_run_config(config: dict[str, Any], experiment: dict[str, Any], result_dir: Path | None) -> dict[str, Any]:
    compute_type = experiment.get("compute_type", "float16")
    return {
        "engine": config["engine"],
        "experiment": experiment_name(experiment),
        "model": experiment["model"],
        "beam_size": int(experiment.get("beam_size", 5)),
        "compute_type": compute_type,
        "precision": compute_type,
        "manifest_path": config["manifest_path"],
        "result_root": config["result_root"],
        "result_dir": str(result_dir) if result_dir is not None else result_dir_for(config, experiment),
        "device": config.get("device", "cuda"),
        "device_index": int(config.get("device_index", 0)),
        "language": config["language"],
        "transcribe_options": build_transcribe_options(config, experiment),
    }


def format_segments(segments) -> tuple[str, list[dict[str, Any]]]:
    texts = []
    rows = []
    for segment in segments:
        text = segment.text.strip()
        texts.append(text)
        rows.append(
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": text,
            }
        )
    return " ".join(texts).strip(), rows


def run_faster_whisper(config: dict[str, Any], args) -> None:
    from faster_whisper import WhisperModel

    decode_run = prepare_decode_run(config, args)
    write_json(decode_run.run_config_path, decode_run.run_config)

    LOGGER.info(
        "Loading faster-whisper model=%s device=%s device_index=%s compute_type=%s",
        config["model"],
        config["device"],
        config["device_index"],
        config["compute_type"],
    )
    LOGGER.info("Experiment=%s beam_size=%s", config["experiment"], config["beam_size"])
    LOGGER.info("Shard %s/%s has %s samples", args.shard_index, args.num_shards, len(decode_run.rows))

    model = WhisperModel(
        config["model"],
        device=config["device"],
        device_index=config["device_index"],
        compute_type=config["compute_type"],
    )

    def decode_one(item: dict[str, Any]) -> DecodeOutput:
        audio_input = load_audio_array(item)
        segment_generator, _info = model.transcribe(audio_input, **config["transcribe_options"])
        prediction_raw, segments = format_segments(segment_generator)
        return DecodeOutput(prediction_raw=prediction_raw, segments=segments)

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
    write_json(decode_run.run_config_path, decode_run.run_config)
    fail_if_all_samples_failed(decode_run.run_config)
    LOGGER.info("Wrote predictions to %s", decode_run.prediction_path)
    LOGGER.info("Wrote errors to %s", decode_run.error_path)
