"""Whisper LoRA adapter를 base model에 merge해서 full checkpoint로 저장한다."""

import argparse
import logging
import sys
from pathlib import Path


DOMAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOMAIN_ROOT / "src"))


LOGGER = logging.getLogger("merge_whisper_lora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whisper LoRA adapter merge 스크립트")
    parser.add_argument("--train_dir", type=Path, required=True)
    parser.add_argument("--adapter_path", type=Path, default=None)
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--base_model_name_or_path", default=None)
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    from core.logging_utils import setup_logging
    from finetuning.merge import merge_whisper_lora

    args = parse_args()
    setup_logging()
    LOGGER.info("=== Whisper LoRA merge started ===")
    LOGGER.info("Train dir: %s", args.train_dir)
    summary = merge_whisper_lora(
        project_root=DOMAIN_ROOT,
        train_dir=args.train_dir,
        adapter_path=args.adapter_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        base_model_name_or_path=args.base_model_name_or_path,
        precision=args.precision,
        device=args.device,
        overwrite=args.overwrite,
    )
    LOGGER.info("Merged model path: %s", summary["output_dir"])
    LOGGER.info("Merge summary: %s", Path(summary["output_dir"]) / "merge_summary.json")
    LOGGER.info("=== DONE Whisper LoRA merge ===")


if __name__ == "__main__":
    main()
