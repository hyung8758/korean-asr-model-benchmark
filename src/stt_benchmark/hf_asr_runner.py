import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from stt_benchmark.config import experiment_name, result_dir_for
from stt_benchmark.io import append_jsonl, write_json
from stt_benchmark.openai_whisper_runner import validate_cuda_device
from stt_benchmark.runner_utils import (
    fail_if_all_samples_failed,
    finish_run,
    make_error_row,
    make_prediction_row,
    prepare_decode_run,
)


LOGGER = logging.getLogger(__name__)
SUPPORTED_RUNNERS = {"transformers_whisper", "qwen_asr"}
SUPPORTED_PRECISIONS = {"float16", "float32", "bfloat16"}
TARGET_SAMPLE_RATE = 16000


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


def dtype_from_precision(precision: str):
    import torch

    if precision == "float16":
        return torch.float16
    if precision == "bfloat16":
        return torch.bfloat16
    if precision == "float32":
        return torch.float32
    raise ValueError(f"Unsupported precision={precision}. Supported values: {sorted(SUPPORTED_PRECISIONS)}")


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
    runner = experiment.get("runner", "transformers_whisper")
    precision = experiment.get("precision", "float16")
    if runner not in SUPPORTED_RUNNERS:
        raise ValueError(f"Unsupported runner={runner}. Supported values: {sorted(SUPPORTED_RUNNERS)}")
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"Unsupported precision={precision}. Supported values: {sorted(SUPPORTED_PRECISIONS)}")

    run_config = {
        "engine": config["engine"],
        "runner": runner,
        "experiment": experiment_name(experiment),
        "model": experiment["model"],
        "beam_size": int(experiment.get("beam_size", config.get("decode_defaults", {}).get("beam_size", 1))),
        "precision": precision,
        "manifest_path": config["manifest_path"],
        "result_root": config["result_root"],
        "result_dir": str(result_dir) if result_dir is not None else result_dir_for(config, experiment),
        "device": config["device"],
        "language": config["language"],
    }
    if runner == "qwen_asr":
        run_config["qwen_options"] = build_qwen_options(config, experiment)
    else:
        run_config["transformers_options"] = build_transformers_options(config, experiment)
    return run_config


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


def load_audio_for_transformers_pipeline(audio_path: str) -> dict[str, Any]:
    import torchaudio
    from torchaudio.functional import resample

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
    audio = waveform.squeeze(0).detach().cpu().numpy()
    return {"array": audio, "sampling_rate": TARGET_SAMPLE_RATE}


def load_transformers_whisper_model(config: dict[str, Any]):
    import transformers.pipelines.automatic_speech_recognition as asr_pipeline
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    asr_pipeline.is_torchcodec_available = lambda: False

    dtype = dtype_from_precision(config["precision"])
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


def load_qwen_model(config: dict[str, Any]):
    from qwen_asr import Qwen3ASRModel

    dtype = dtype_from_precision(config["precision"])
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


def decode_rows(
    rows: list[dict[str, Any]],
    model: Any,
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
                if config["runner"] == "qwen_asr":
                    results = model.transcribe(
                        audio=item["audio"],
                        language=config["qwen_options"]["language"],
                        return_time_stamps=config["qwen_options"]["return_time_stamps"],
                    )
                    result = results[0]
                    prediction_raw = str(getattr(result, "text", "")).strip()
                    segments = format_qwen_segments(result)
                else:
                    audio_input = load_audio_for_transformers_pipeline(item["audio"])
                    result = model(
                        audio_input,
                        generate_kwargs=config["transformers_options"]["generate_kwargs"],
                    )
                    prediction_raw = str(result.get("text", "")).strip()
                    segments = format_pipeline_segments(result)

                decode_time = time.perf_counter() - start
                append_jsonl(prediction_file, make_prediction_row(item, prediction_raw, segments, decode_time, config))
                decoded_count += 1
            except Exception as exc:
                decode_time = time.perf_counter() - start
                append_jsonl(error_file, make_error_row(item, "decode_failed", str(exc), decode_time, config))
                LOGGER.exception("Decode failed for id=%s", item["id"])
                error_count += 1

    return decoded_count, error_count


def run_hf_asr(config: dict[str, Any], args) -> None:
    import torch

    validate_cuda_device(str(config["device"]))

    decode_run = prepare_decode_run(config, args)
    write_json(decode_run.run_config_path, decode_run.run_config)

    LOGGER.info("Loading HF ASR model=%s runner=%s device=%s", config["model"], config["runner"], config["device"])
    LOGGER.info("Experiment=%s precision=%s beam_size=%s", config["experiment"], config["precision"], config["beam_size"])
    LOGGER.info("Shard %s/%s has %s samples", args.shard_index, args.num_shards, len(decode_run.rows))

    if config["runner"] == "qwen_asr":
        model = load_qwen_model(config)
    else:
        model = load_transformers_whisper_model(config)

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
