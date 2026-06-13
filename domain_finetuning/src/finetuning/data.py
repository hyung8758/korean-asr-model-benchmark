from collections import Counter
from pathlib import Path
from typing import Any

from core.io import write_json, write_jsonl
from data.audio import check_audio_runtime, prepare_output_dir, write_drop_rows
from data.corpora import parse_candidates_from_data_root
from data.manifest import (
    DEFAULT_BUCKETS,
    DEFAULT_DATASETS,
    cap_rows_by_dataset,
    duration_hours,
    filter_candidates_by_dataset_and_bucket,
    per_group_summary,
    prepare_manifest_rows,
    split_rows,
)
from data.segments import prepare_split_audio_cache
from finetuning.manifest import FINETUNING_SPLITS, finetuning_manifest_row


def build_summary(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    dropped_reasons: Counter,
    output_root: Path,
) -> dict[str, Any]:
    all_rows = train_rows + dev_rows + eval_rows
    metadata_dir = output_root / "metadata"
    return {
        "total_samples": len(all_rows),
        "total_duration_hours": duration_hours(all_rows),
        "splits": {
            "train": {"samples": len(train_rows), "duration_hours": duration_hours(train_rows)},
            "dev": {"samples": len(dev_rows), "duration_hours": duration_hours(dev_rows)},
            "eval": {"samples": len(eval_rows), "duration_hours": duration_hours(eval_rows)},
        },
        "per_dataset": per_group_summary(all_rows, "dataset"),
        "per_bucket": per_group_summary(all_rows, "bucket"),
        "dropped_count_by_reason": dict(sorted(dropped_reasons.items())),
        "train_path": str(output_root / "train.jsonl"),
        "dev_path": str(output_root / "dev.jsonl"),
        "eval_path": str(output_root / "eval.jsonl"),
        "metadata_dir": str(metadata_dir),
        "summary_path": str(metadata_dir / "summary.json"),
        "dropped_samples_path": str(metadata_dir / "dropped_samples.jsonl"),
    }


def write_split_manifests(output_root: Path, split_rows_by_name: dict[str, list[dict[str, Any]]]) -> None:
    for split in FINETUNING_SPLITS:
        rows = split_rows_by_name[split]
        write_jsonl(output_root / f"{split}.jsonl", [finetuning_manifest_row(row, split) for row in rows])


def prepare_finetuning_data(
    data_root: Path,
    output_root: Path,
    project_root: Path,
    dev_ratio: float,
    eval_ratio: float,
    seed: int,
    datasets: set[str],
    buckets: set[str],
    min_duration: float,
    max_duration: float | None,
    max_train_hours_per_dataset: float | None,
    max_dev_hours_per_dataset: float | None,
    max_eval_hours_per_dataset: float | None,
    split_sample_rate: int,
    overwrite_split_audio: bool,
    overwrite: bool,
) -> dict[str, Any]:
    candidates = parse_candidates_from_data_root(data_root)
    if not candidates:
        raise RuntimeError(f"No parseable corpus samples found under {data_root}")

    prepare_output_dir(output_root, overwrite)
    candidates, split_drops = prepare_split_audio_cache(
        candidates=candidates,
        data_root=data_root,
        project_root=project_root,
        sample_rate=split_sample_rate,
        overwrite=overwrite_split_audio,
    )
    check_audio_runtime(candidates)

    candidates = filter_candidates_by_dataset_and_bucket(candidates, datasets=datasets, buckets=buckets)
    dropped_reasons: Counter = Counter()
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    dropped_path = metadata_dir / "dropped_samples.jsonl"

    with dropped_path.open("w", encoding="utf-8") as dropped_handle:
        write_drop_rows(dropped_handle, dropped_reasons, split_drops)
        rows = prepare_manifest_rows(
            candidates=candidates,
            project_root=project_root,
            buckets=buckets,
            min_duration=min_duration,
            max_duration=max_duration,
            max_hours_per_dataset=None,
            dropped_handle=dropped_handle,
            dropped_reasons=dropped_reasons,
            desc="Preparing fine-tuning samples",
        )
        train_rows, dev_rows, eval_rows = split_rows(
            rows,
            dev_ratio=dev_ratio,
            eval_ratio=eval_ratio,
            seed=seed,
        )
        train_rows = cap_rows_by_dataset(
            rows=train_rows,
            max_hours_per_dataset=max_train_hours_per_dataset,
            dropped_handle=dropped_handle,
            dropped_reasons=dropped_reasons,
            reason="exceeds_max_train_hours_per_dataset",
        )
        dev_rows = cap_rows_by_dataset(
            rows=dev_rows,
            max_hours_per_dataset=max_dev_hours_per_dataset,
            dropped_handle=dropped_handle,
            dropped_reasons=dropped_reasons,
            reason="exceeds_max_dev_hours_per_dataset",
        )
        eval_rows = cap_rows_by_dataset(
            rows=eval_rows,
            max_hours_per_dataset=max_eval_hours_per_dataset,
            dropped_handle=dropped_handle,
            dropped_reasons=dropped_reasons,
            reason="exceeds_max_eval_hours_per_dataset",
        )

    summary = build_summary(train_rows, dev_rows, eval_rows, dropped_reasons, output_root)
    write_split_manifests(output_root, {"train": train_rows, "dev": dev_rows, "eval": eval_rows})
    write_json(metadata_dir / "summary.json", summary)
    return summary
