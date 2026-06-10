import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analysis.prediction_quality import analyze_rows, collect_examples
from core.io import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze common quality issues in predictions.jsonl.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--examples_output", type=Path)
    parser.add_argument("--max_examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    output_path = args.output or args.predictions.with_name("quality_analysis.json")
    examples_path = args.examples_output or args.predictions.with_name("quality_examples.json")

    write_json(output_path, analyze_rows(rows))
    write_json(examples_path, collect_examples(rows, args.max_examples))
    print(f"Wrote quality analysis: {output_path}")
    print(f"Wrote quality examples: {examples_path}")


if __name__ == "__main__":
    main()

