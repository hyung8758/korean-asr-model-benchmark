"""한국어 STT corpus에서 Whisper benchmark manifest를 만든다."""

import argparse
import json
import logging
import shlex
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.audio import check_audio_runtime, display_path, prepare_output_dir, write_drop_rows, write_jsonl_line
from data.corpora import BUCKETS, DATASETS, Candidate, discover_corpus_dirs, parse_candidates
from data.manifest import ordered_benchmark_candidates, prepare_manifest_rows
from data.segments import DEFAULT_SAMPLE_RATE, prepare_split_audio_cache


LOGGER = logging.getLogger("prepare_whisper_benchmark_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whisper benchmark manifest.jsonl, summary.json을 생성한다."
    )
    parser.add_argument("--data_root", type=Path, default=PROJECT_ROOT / "data" / "download")
    parser.add_argument("--output_root", type=Path, default=PROJECT_ROOT / "data" / "benchmark")
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--max_hours_per_corpus", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write_kaldi", action="store_true")
    parser.add_argument("--overwrite_split_audio", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def process_candidates(
    candidates: list[Candidate],
    output_root: Path,
    max_hours_per_corpus: float | None,
    seed: int,
    initial_drops: list[dict] | None = None,
) -> tuple[list[dict], Counter]:
    dropped_path = output_root / "dropped_samples.jsonl"
    dropped_reasons: Counter = Counter()

    with dropped_path.open("w", encoding="utf-8") as dropped_handle:
        write_drop_rows(dropped_handle, dropped_reasons, initial_drops or [])
        manifest_rows = prepare_manifest_rows(
            candidates=ordered_benchmark_candidates(candidates, seed),
            project_root=PROJECT_ROOT,
            buckets=set(BUCKETS),
            min_duration=1.0,
            max_duration=None,
            max_hours_per_dataset=max_hours_per_corpus,
            dropped_handle=dropped_handle,
            dropped_reasons=dropped_reasons,
            desc="Preparing benchmark samples",
            max_hours_drop_reason="exceeds_max_hours_per_corpus",
        )

    return manifest_rows, dropped_reasons


def write_manifest(output_root: Path, rows: list[dict]) -> Path:
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            write_jsonl_line(handle, row)
    return manifest_path


def write_summary(output_root: Path, rows: list[dict], dropped_reasons: Counter, manifest_path: Path) -> None:
    per_dataset = {
        dataset: {"samples": 0, "duration_hours": 0.0}
        for dataset in DATASETS
    }
    per_bucket = {
        bucket: {"samples": 0, "duration_hours": 0.0}
        for bucket in BUCKETS
    }
    for row in rows:
        duration_hours = row["duration"] / 3600.0
        per_dataset[row["dataset"]]["samples"] += 1
        per_dataset[row["dataset"]]["duration_hours"] += duration_hours
        per_bucket[row["bucket"]]["samples"] += 1
        per_bucket[row["bucket"]]["duration_hours"] += duration_hours

    for group in (per_dataset, per_bucket):
        for stats in group.values():
            stats["duration_hours"] = round(stats["duration_hours"], 6)

    summary = {
        "total_samples": len(rows),
        "total_duration_hours": round(sum(row["duration"] for row in rows) / 3600.0, 6),
        "per_dataset": per_dataset,
        "per_bucket": per_bucket,
        "dropped_count_by_reason": dict(sorted(dropped_reasons.items())),
        "manifest_path": display_path(manifest_path, PROJECT_ROOT),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_kaldi(output_root: Path, rows: list[dict]) -> None:
    kaldi_dir = output_root / "kaldi"
    kaldi_dir.mkdir(parents=True, exist_ok=True)
    with (kaldi_dir / "wav.scp").open("w", encoding="utf-8") as wav_scp, (
        kaldi_dir / "text"
    ).open("w", encoding="utf-8") as text_file, (kaldi_dir / "utt2spk").open(
        "w", encoding="utf-8"
    ) as utt2spk:
        for row in rows:
            wav_scp.write(f"{row['id']} {kaldi_audio_value(row)}\n")
            text_file.write(f"{row['id']} {row['text']}\n")
            utt2spk.write(f"{row['id']} {row['speaker'] or row['id']}\n")


def kaldi_audio_value(row: dict) -> str:
    if row.get("audio_start") is None or row.get("audio_end") is None:
        return row["audio"]
    start = float(row["audio_start"])
    duration = max(0.0, float(row["audio_end"]) - start)
    return f"sox {shlex.quote(row['audio'])} -t wav - trim {start:.3f} {duration:.3f} |"


def main() -> None:
    args = parse_args()
    setup_logging()

    corpus_dirs = discover_corpus_dirs(args.data_root)
    for dataset, dirs in corpus_dirs.items():
        LOGGER.info("Discovered %s dirs: %s", dataset, ", ".join(str(path) for path in dirs) or "none")

    candidates = parse_candidates(corpus_dirs)
    if not candidates:
        raise RuntimeError(f"No parseable corpus samples found under {args.data_root}")

    prepare_output_dir(args.output_root, args.overwrite)
    candidates, split_drops = prepare_split_audio_cache(
        candidates=candidates,
        data_root=args.data_root,
        project_root=PROJECT_ROOT,
        sample_rate=args.sample_rate,
        overwrite=args.overwrite_split_audio,
    )
    check_audio_runtime(candidates)

    rows, dropped_reasons = process_candidates(
        candidates=candidates,
        output_root=args.output_root,
        max_hours_per_corpus=args.max_hours_per_corpus,
        seed=args.seed,
        initial_drops=split_drops,
    )
    manifest_path = write_manifest(args.output_root, rows)
    write_summary(args.output_root, rows, dropped_reasons, manifest_path)
    if args.write_kaldi:
        write_kaldi(args.output_root, rows)

    LOGGER.info("Wrote %d manifest rows to %s", len(rows), manifest_path)
    LOGGER.info("Wrote summary to %s", args.output_root / "summary.json")


if __name__ == "__main__":
    main()
