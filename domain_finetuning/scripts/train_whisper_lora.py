"""Whisper small 모델을 PEFT LoRA로 fine-tuning한다."""

import argparse
import logging
import sys
from pathlib import Path


DOMAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOMAIN_ROOT / "src"))


LOGGER = logging.getLogger("train_whisper_lora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whisper LoRA 파인튜닝 실행 스크립트")
    parser.add_argument(
        "--config",
        type=Path,
        default=DOMAIN_ROOT / "config" / "whisper_small_lora.yaml",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    setup_logging()

    from finetuning.config import load_finetuning_config
    from finetuning.trainer import run_whisper_lora_training

    config = load_finetuning_config(args.config)
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    if args.resume_from_checkpoint is not None:
        config["training"]["resume_from_checkpoint"] = str(args.resume_from_checkpoint)

    LOGGER.info("Using config: %s", args.config)
    summary = run_whisper_lora_training(config, project_root=DOMAIN_ROOT)
    LOGGER.info("Wrote training summary: %s", summary["training_summary_path"])


if __name__ == "__main__":
    main()
