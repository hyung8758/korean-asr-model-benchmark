"""Whisper fine-tuning 데이터 구조를 빠르게 확인한다."""

import argparse
import logging
import sys
from pathlib import Path


DOMAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOMAIN_ROOT / "src"))

from finetuning.validation import validate_finetuning_data


LOGGER = logging.getLogger("validate_whisper_finetuning_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whisper fine-tuning 데이터 검증")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DOMAIN_ROOT / "data" / "whisper_small_lora",
    )
    parser.add_argument("--project_root", type=Path, default=DOMAIN_ROOT)
    parser.add_argument("--max_errors", type=int, default=20)
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
    summary = validate_finetuning_data(
        output_root=args.output_root,
        project_root=args.project_root,
        max_errors=args.max_errors,
    )
    for split, split_summary in summary["splits"].items():
        LOGGER.info("%s samples: %s", split, split_summary["samples"])
        LOGGER.info("%s first row: %s", split, split_summary["first_row"])

    if summary["errors"]:
        for error in summary["errors"]:
            LOGGER.error(error)
        raise SystemExit(1)

    LOGGER.info("Fine-tuning manifests look good: %s", args.output_root)


if __name__ == "__main__":
    main()
