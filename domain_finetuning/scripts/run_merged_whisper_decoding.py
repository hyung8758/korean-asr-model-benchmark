"""Merge된 Whisper checkpoint를 fine-tuning eval manifest로 디코딩하고 평가한다."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any


DOMAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOMAIN_ROOT / "src"))


LOGGER = logging.getLogger("run_merged_whisper_decoding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge된 Whisper 모델 fine-tuning eval decoding 스크립트")
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=Path("data/whisper_small_lora/eval.jsonl"),
    )
    parser.add_argument("--result_root", type=Path, default=Path("exp/results"))
    parser.add_argument("--result_dir", "--output_dir", dest="result_dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("float16", "float32", "bfloat16"), default="float16")
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--language", default="ko")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--retry_errors", action="store_true")
    parser.add_argument("--no_evaluate", action="store_true")
    return parser.parse_args()


def default_result_dir(result_root: Path, experiment_name: str) -> Path:
    return result_root / experiment_name


def resolve_required_path(path: Path) -> Path:
    from finetuning.config import resolve_project_path

    resolved = resolve_project_path(DOMAIN_ROOT, path)
    if resolved is None:
        raise ValueError("Path must not be None")
    return resolved


def build_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    from runners.huggingface_transformers import build_run_config

    model_dir = resolve_required_path(args.model_dir)
    manifest_path = resolve_required_path(args.manifest_path)
    result_root = resolve_required_path(args.result_root)
    experiment_name = f"{model_dir.name}_beam{args.beam_size}_{args.precision}"
    result_dir = (
        resolve_required_path(args.result_dir)
        if args.result_dir is not None
        else default_result_dir(result_root, experiment_name)
    )

    config = {
        "engine": "whisper_lora_merged",
        "manifest_path": str(manifest_path),
        "result_root": str(result_root),
        "device": args.device,
        "language": args.language,
        "decode_defaults": {
            "task": "transcribe",
            "beam_size": args.beam_size,
            "batch_size": 1,
            "chunk_length_s": 30,
            "return_timestamps": False,
        },
    }
    experiment = {
        "name": experiment_name,
        "model": str(model_dir),
        "beam_size": args.beam_size,
        "precision": args.precision,
        "return_timestamps": False,
    }
    return build_run_config(config, experiment, result_dir), manifest_path


def evaluate_if_needed(args: argparse.Namespace, manifest_path: Path, result_dir: Path) -> None:
    if args.no_evaluate:
        LOGGER.info("Evaluation skipped by --no_evaluate")
        return
    if args.num_shards != 1:
        LOGGER.info("Shard mode detected. Evaluate after all shards finish:")
        LOGGER.info("python scripts/evaluate_predictions.py --manifest_path %s --result_dir %s", manifest_path, result_dir)
        return

    from core.metrics import evaluate_result_dir

    evaluate_result_dir(manifest_path, result_dir)
    LOGGER.info("Wrote metrics: %s", result_dir / "metrics.json")


def main() -> None:
    from core.logging_utils import setup_logging
    from runners.huggingface_transformers import run_huggingface_transformers

    args = parse_args()
    run_config, manifest_path = build_config(args)
    result_dir = Path(run_config["result_dir"])
    setup_logging(result_dir / "logs" / "run.log")

    LOGGER.info("=== Merged Whisper decoding started ===")
    LOGGER.info("Merged model decoding: model=%s manifest=%s", run_config["model"], run_config["manifest_path"])
    run_huggingface_transformers(run_config, args)
    evaluate_if_needed(args, manifest_path, result_dir)
    LOGGER.info("Result dir: %s", result_dir)
    LOGGER.info("=== DONE Merged Whisper decoding ===")


if __name__ == "__main__":
    main()
