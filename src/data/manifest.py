import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from core.text import normalize_text
from data.audio import bucket_for_duration, display_path, load_audio, record_drop, safe_id
from data.corpora import BUCKETS, DATASETS, Candidate


DEFAULT_DATASETS = DATASETS
DEFAULT_BUCKETS = BUCKETS


def duration_hours(rows: list[dict[str, Any]]) -> float:
    return round(sum(float(row.get("duration") or 0.0) for row in rows) / 3600.0, 6)


def candidate_segment_duration(candidate: Candidate) -> float | None:
    if candidate.start_sec is None or candidate.end_sec is None:
        return None
    return max(0.0, candidate.end_sec - candidate.start_sec)


def candidate_bucket_hint(candidate: Candidate) -> str:
    duration = candidate_segment_duration(candidate)
    if duration is None:
        return "unknown"
    return bucket_for_duration(duration)


def candidate_priority(candidate: Candidate) -> tuple[int, str]:
    bucket = candidate_bucket_hint(candidate)
    priority = {"long": 0, "mid": 1, "short": 2, "unknown": 1}.get(bucket, 1)
    return priority, candidate.utt_id


def ordered_benchmark_candidates(candidates: list[Candidate], seed: int) -> list[Candidate]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_dataset[candidate.dataset].append(candidate)

    ordered: list[Candidate] = []
    for dataset in DATASETS:
        items = by_dataset.get(dataset, [])
        rng.shuffle(items)
        items.sort(key=candidate_priority)
        ordered.extend(items)
    return ordered


def split_rows(
    rows: list[dict[str, Any]],
    dev_ratio: float,
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("dataset") or "unknown"), str(row.get("bucket") or "unknown"))].append(row)

    rng = random.Random(seed)
    train_rows = []
    dev_rows = []
    eval_rows = []

    for key in sorted(grouped):
        items = sorted(grouped[key], key=lambda row: str(row["id"]))
        rng.shuffle(items)

        if len(items) == 1:
            dev_count = 0
            eval_count = 0
        else:
            dev_count = split_count(len(items), dev_ratio)
            eval_count = split_count(len(items), eval_ratio)
            if dev_count + eval_count >= len(items):
                overflow = dev_count + eval_count - len(items) + 1
                if eval_count >= dev_count:
                    eval_count = max(0, eval_count - overflow)
                else:
                    dev_count = max(0, dev_count - overflow)

        dev_end = dev_count
        eval_end = dev_count + eval_count
        dev_rows.extend(items[:dev_end])
        eval_rows.extend(items[dev_end:eval_end])
        train_rows.extend(items[eval_end:])

    return train_rows, dev_rows, eval_rows


def split_count(total: int, ratio: float) -> int:
    if ratio <= 0:
        return 0
    return max(1, min(total - 1, round(total * ratio)))


def filter_candidates_by_dataset_and_bucket(
    candidates: list[Candidate],
    datasets: set[str],
    buckets: set[str],
) -> list[Candidate]:
    filtered = []
    for candidate in candidates:
        if candidate.dataset not in datasets:
            continue
        bucket = candidate_bucket_hint(candidate)
        if bucket != "unknown" and bucket not in buckets:
            continue
        filtered.append(candidate)
    return filtered


def item_duration(candidate: Candidate, waveform_frames: int, sample_rate: int) -> float:
    segment_duration = candidate_segment_duration(candidate)
    if segment_duration is not None:
        return segment_duration
    return waveform_frames / float(sample_rate)


def audio_segment_fields(candidate: Candidate) -> dict[str, float | None]:
    return {
        "audio_start": candidate.start_sec,
        "audio_end": candidate.end_sec,
        "source_audio_start": candidate.original_start_sec,
        "source_audio_end": candidate.original_end_sec,
    }


def make_manifest_row(
    candidate: Candidate,
    text: str,
    duration: float,
    bucket: str,
    source_sample_rate: int,
    project_root: Path,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_audio_path = candidate.original_audio_path or candidate.audio_path
    row = {
        "id": safe_id(candidate.dataset, candidate.utt_id),
        "audio": display_path(candidate.audio_path, project_root),
        "text": text,
        "text_raw": candidate.text_raw,
        "dataset": candidate.dataset,
        "duration": round(duration, 6),
        "bucket": bucket,
        "speaker": candidate.speaker,
        "split": candidate.split,
        "source_audio": display_path(source_audio_path, project_root),
        "source_text": display_path(candidate.source_text, project_root),
        "audio_sample_rate": source_sample_rate,
        "source_sample_rate": source_sample_rate,
        **audio_segment_fields(candidate),
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def should_skip_by_hours(
    used_seconds_by_dataset: dict[str, float],
    candidate: Candidate,
    duration: float,
    max_hours_per_dataset: float | None,
) -> bool:
    if max_hours_per_dataset is None:
        return False
    max_seconds = max_hours_per_dataset * 3600.0
    return used_seconds_by_dataset[candidate.dataset] + duration > max_seconds


def cap_rows_by_dataset(
    rows: list[dict[str, Any]],
    max_hours_per_dataset: float | None,
    dropped_handle,
    dropped_reasons: Counter,
    reason: str,
) -> list[dict[str, Any]]:
    if max_hours_per_dataset is None:
        return rows

    max_seconds = max_hours_per_dataset * 3600.0
    used_seconds_by_dataset: dict[str, float] = defaultdict(float)
    kept_rows = []
    for row in rows:
        dataset = str(row.get("dataset") or "unknown")
        duration = float(row.get("duration") or 0.0)
        if used_seconds_by_dataset[dataset] + duration > max_seconds:
            record_drop(
                dropped_handle,
                dropped_reasons,
                {
                    "id": row.get("id"),
                    "dataset": dataset,
                    "source_audio": row.get("source_audio", row.get("audio")),
                    "source_text": row.get("source_text", ""),
                },
                reason,
                duration=duration,
            )
            continue
        used_seconds_by_dataset[dataset] += duration
        kept_rows.append(row)
    return kept_rows


def prepare_manifest_rows(
    candidates: list[Candidate],
    project_root: Path,
    buckets: set[str],
    min_duration: float,
    max_duration: float | None,
    max_hours_per_dataset: float | None,
    dropped_handle,
    dropped_reasons: Counter,
    desc: str,
    extra_fields_for_candidate: Callable[[Candidate], dict[str, Any]] | None = None,
    max_hours_drop_reason: str = "exceeds_max_hours_per_dataset",
) -> list[dict[str, Any]]:
    rows = []
    used_seconds_by_dataset: dict[str, float] = defaultdict(float)
    cached_audio_path: Path | None = None
    cached_waveform = None
    cached_sample_rate: int | None = None

    for candidate in tqdm(candidates, desc=desc):
        text_raw = candidate.text_raw
        text = normalize_text(text_raw)
        drop_base = {
            "id": candidate.utt_id,
            "dataset": candidate.dataset,
            "source_audio": display_path(candidate.audio_path, project_root),
            "source_text": display_path(candidate.source_text, project_root),
        }

        if not text_raw.strip():
            record_drop(dropped_handle, dropped_reasons, drop_base, "empty_transcript")
            continue
        if len(text) <= 2:
            record_drop(dropped_handle, dropped_reasons, drop_base, "normalized_text_too_short")
            continue

        if cached_audio_path != candidate.audio_path:
            try:
                cached_waveform, cached_sample_rate = load_audio(candidate.audio_path)
                cached_audio_path = candidate.audio_path
            except Exception as exc:
                record_drop(dropped_handle, dropped_reasons, drop_base, "torchaudio_load_failed", error=str(exc))
                cached_audio_path = None
                cached_waveform = None
                cached_sample_rate = None
                continue

        if cached_waveform is None or cached_sample_rate is None:
            record_drop(dropped_handle, dropped_reasons, drop_base, "audio_cache_empty")
            continue

        duration = item_duration(candidate, cached_waveform.size(1), cached_sample_rate)
        if duration <= min_duration:
            record_drop(dropped_handle, dropped_reasons, drop_base, "duration_too_short", duration=duration)
            continue
        if max_duration is not None and duration > max_duration:
            record_drop(dropped_handle, dropped_reasons, drop_base, "duration_too_long", duration=duration)
            continue

        bucket = bucket_for_duration(duration)
        if bucket not in buckets:
            record_drop(dropped_handle, dropped_reasons, drop_base, "bucket_filtered", duration=duration, bucket=bucket)
            continue
        if should_skip_by_hours(used_seconds_by_dataset, candidate, duration, max_hours_per_dataset):
            record_drop(dropped_handle, dropped_reasons, drop_base, max_hours_drop_reason, duration=duration)
            continue

        used_seconds_by_dataset[candidate.dataset] += duration
        extra_fields = extra_fields_for_candidate(candidate) if extra_fields_for_candidate else None
        rows.append(
            make_manifest_row(
                candidate=candidate,
                text=text,
                duration=duration,
                bucket=bucket,
                source_sample_rate=cached_sample_rate,
                project_root=project_root,
                extra_fields=extra_fields,
            )
        )

    return rows


def per_group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return {
        name: {"samples": len(items), "duration_hours": duration_hours(items)}
        for name, items in sorted(grouped.items())
    }
