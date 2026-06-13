"""Whisper LoRA fine-tuning용 train/dev/eval jsonl을 raw corpus에서 만든다."""

import argparse
import logging
import sys
from pathlib import Path


DOMAIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DOMAIN_ROOT.parent
sys.path.insert(0, str(DOMAIN_ROOT / "src"))

from finetuning.data import DEFAULT_BUCKETS, DEFAULT_DATASETS, prepare_finetuning_data
from finetuning.validation import validate_finetuning_data
from data.segments import DEFAULT_SPLIT_SAMPLE_RATE


LOGGER = logging.getLogger("prepare_whisper_finetuning_data")
DEFAULT_DEV_RATIO = 0.1
DEFAULT_EVAL_RATIO = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="raw corpus에서 Whisper fine-tuning용 train/dev/eval jsonl을 생성한다."
    )
    parser.add_argument("--data_root", type=Path, default=REPO_ROOT / "data" / "download")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DOMAIN_ROOT / "data" / "whisper_small_lora",
    )
    parser.add_argument("--project_root", type=Path, default=DOMAIN_ROOT)
    parser.add_argument("--split_sample_rate", type=int, default=DEFAULT_SPLIT_SAMPLE_RATE)
    parser.add_argument("--dev_ratio", type=float, default=DEFAULT_DEV_RATIO)
    parser.add_argument("--eval_ratio", type=float, default=DEFAULT_EVAL_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--buckets", nargs="+", default=list(DEFAULT_BUCKETS))
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--max_train_hours_per_dataset", type=float, default=None)
    parser.add_argument("--max_dev_hours_per_dataset", type=float, default=None)
    parser.add_argument("--max_eval_hours_per_dataset", type=float, default=None)
    parser.add_argument("--overwrite_split_audio", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    setup_logging()
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not 0.0 <= args.dev_ratio < 1.0:
        raise ValueError("--dev_ratio must be >= 0.0 and < 1.0")
    if not 0.0 <= args.eval_ratio < 1.0:
        raise ValueError("--eval_ratio must be >= 0.0 and < 1.0")
    if args.dev_ratio + args.eval_ratio >= 1.0:
        raise ValueError("--dev_ratio + --eval_ratio must be < 1.0")

    summary = prepare_finetuning_data(
        data_root=args.data_root,
        output_root=args.output_root,
        project_root=args.project_root,
        dev_ratio=args.dev_ratio,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        datasets=set(args.datasets),
        buckets=set(args.buckets),
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_train_hours_per_dataset=args.max_train_hours_per_dataset,
        max_dev_hours_per_dataset=args.max_dev_hours_per_dataset,
        max_eval_hours_per_dataset=args.max_eval_hours_per_dataset,
        split_sample_rate=args.split_sample_rate,
        overwrite_split_audio=args.overwrite_split_audio,
        overwrite=args.overwrite,
    )
    validation = validate_finetuning_data(
        output_root=args.output_root,
        project_root=args.project_root,
    )
    if validation["errors"]:
        for error in validation["errors"]:
            LOGGER.error(error)
        raise RuntimeError("Fine-tuning data validation failed.")

    LOGGER.info("Wrote train data: %s", summary["train_path"])
    LOGGER.info("Wrote dev data: %s", summary["dev_path"])
    LOGGER.info("Wrote eval data: %s", summary["eval_path"])
    LOGGER.info("Wrote metadata: %s", summary["metadata_dir"])
    LOGGER.info("Validation passed.")
    LOGGER.info("Total samples: %s", summary["total_samples"])


if __name__ == "__main__":
    main()
