import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.logging_utils import setup_logging
from core.metrics import evaluate_result_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge predictions and write metrics.json.")
    parser.add_argument("--manifest_path", type=Path, default=Path("data/benchmark/manifest.jsonl"))
    parser.add_argument("--result_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.result_dir / "logs" / "evaluate.log")
    evaluate_result_dir(args.manifest_path, args.result_dir)


if __name__ == "__main__":
    main()
