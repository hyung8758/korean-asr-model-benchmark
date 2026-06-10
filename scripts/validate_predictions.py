import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analysis.prediction_validation import validate_prediction_rows
from core.io import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate predictions.jsonl schema and manifest id coverage.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    manifest_rows = read_jsonl(args.manifest_path) if args.manifest_path else None
    report = validate_prediction_rows(rows, manifest_rows)

    output_path = args.output or args.predictions.with_name("validation.json")
    write_json(output_path, report)
    print(f"Wrote validation report: {output_path}")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

