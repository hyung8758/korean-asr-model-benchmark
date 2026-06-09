from collections import defaultdict
from statistics import mean, median
from typing import Any

from core.io import read_jsonl, write_json, write_jsonl
from core.text import char_tokens, word_tokens


BUCKETS = ("short", "mid", "long")


def edit_distance(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = int(round((len(values) - 1) * q))
    return values[index]


def new_stats() -> dict[str, Any]:
    return {
        "samples": 0,
        "duration_sec": 0.0,
        "decode_time_sec": 0.0,
        "char_errors": 0,
        "char_total": 0,
        "word_errors": 0,
        "word_total": 0,
        "rtfs": [],
    }


def add_row(stats: dict[str, Any], row: dict[str, Any]) -> None:
    ref_chars = char_tokens(row.get("reference", ""))
    hyp_chars = char_tokens(row.get("prediction", ""))
    ref_words = word_tokens(row.get("reference", ""))
    hyp_words = word_tokens(row.get("prediction", ""))

    stats["samples"] += 1
    stats["duration_sec"] += float(row.get("duration") or 0.0)
    stats["decode_time_sec"] += float(row.get("decode_time") or 0.0)
    stats["char_errors"] += edit_distance(ref_chars, hyp_chars)
    stats["char_total"] += len(ref_chars)
    stats["word_errors"] += edit_distance(ref_words, hyp_words)
    stats["word_total"] += len(ref_words)
    if row.get("rtf") is not None:
        stats["rtfs"].append(float(row["rtf"]))


def finish_stats(stats: dict[str, Any]) -> dict[str, Any]:
    rtfs = stats.pop("rtfs")
    char_total = stats["char_total"]
    word_total = stats["word_total"]
    duration_sec = stats["duration_sec"]
    decode_time_sec = stats["decode_time_sec"]

    stats["duration_hours"] = round(duration_sec / 3600.0, 6)
    stats["decode_time_hours"] = round(decode_time_sec / 3600.0, 6)
    stats["cer"] = 100.0 * stats["char_errors"] / char_total if char_total else None
    stats["wer"] = 100.0 * stats["word_errors"] / word_total if word_total else None
    stats["avg_rtf"] = mean(rtfs) if rtfs else None
    stats["p50_rtf"] = median(rtfs) if rtfs else None
    stats["p90_rtf"] = percentile(rtfs, 0.90)
    stats["avg_decode_time"] = decode_time_sec / stats["samples"] if stats["samples"] else None
    return stats


def collect_outputs(result_dir, prefix: str) -> list[dict[str, Any]]:
    rows = read_jsonl(result_dir / f"{prefix}.jsonl")
    shard_dir = result_dir / "shards"
    if shard_dir.exists():
        for path in sorted(shard_dir.glob(f"{prefix}.*.jsonl")):
            rows.extend(read_jsonl(path))
    return rows


def deduplicate_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {}
    for row in rows:
        by_id[row["id"]] = row
    return list(by_id.values())


def sort_by_manifest(rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {row["id"]: index for index, row in enumerate(manifest_rows)}
    return sorted(rows, key=lambda row: order.get(row["id"], len(order)))


def build_metrics(predictions: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    overall = new_stats()
    per_dataset = defaultdict(new_stats)
    per_bucket = {bucket: new_stats() for bucket in BUCKETS}
    error_count_by_reason = defaultdict(int)

    for row in predictions:
        add_row(overall, row)
        add_row(per_dataset[row["dataset"]], row)
        add_row(per_bucket[row["bucket"]], row)

    for row in errors:
        error_count_by_reason[row.get("reason", "unknown")] += 1

    return {
        "overall": finish_stats(overall),
        "per_dataset": {dataset: finish_stats(stats) for dataset, stats in sorted(per_dataset.items())},
        "per_bucket": {bucket: finish_stats(stats) for bucket, stats in per_bucket.items()},
        "error_count": len(errors),
        "error_count_by_reason": dict(sorted(error_count_by_reason.items())),
    }


def evaluate_result_dir(manifest_path, result_dir) -> None:
    manifest_rows = read_jsonl(manifest_path)
    predictions = sort_by_manifest(deduplicate_by_id(collect_outputs(result_dir, "predictions")), manifest_rows)
    errors = sort_by_manifest(deduplicate_by_id(collect_outputs(result_dir, "errors")), manifest_rows)
    prediction_ids = {row["id"] for row in predictions}
    errors = [row for row in errors if row["id"] not in prediction_ids]

    write_jsonl(result_dir / "predictions.jsonl", predictions)
    write_jsonl(result_dir / "errors.jsonl", errors)
    write_json(result_dir / "metrics.json", build_metrics(predictions, errors))
