import re
from collections import Counter, defaultdict
from typing import Any

from core.text import char_tokens, normalize_text


SUBTITLE_PATTERNS = (
    "자막 제공",
    "자막제공",
    "시청해 주셔서 감사합니다",
    "시청해주셔서 감사합니다",
    "구독",
)


def is_empty_prediction(row: dict[str, Any]) -> bool:
    return not normalize_text(str(row.get("prediction") or row.get("prediction_raw") or ""))


def length_ratio(row: dict[str, Any]) -> float | None:
    reference_length = len(char_tokens(str(row.get("reference") or "")))
    prediction_length = len(char_tokens(str(row.get("prediction") or "")))
    if reference_length == 0:
        return None
    return prediction_length / reference_length


def is_very_long_prediction(row: dict[str, Any], ratio_threshold: float = 2.5, min_extra_chars: int = 20) -> bool:
    reference_length = len(char_tokens(str(row.get("reference") or "")))
    prediction_length = len(char_tokens(str(row.get("prediction") or "")))
    if reference_length == 0:
        return prediction_length >= min_extra_chars
    return prediction_length >= reference_length + min_extra_chars and prediction_length / reference_length >= ratio_threshold


def is_very_short_prediction(row: dict[str, Any], ratio_threshold: float = 0.3, min_missing_chars: int = 10) -> bool:
    reference_length = len(char_tokens(str(row.get("reference") or "")))
    prediction_length = len(char_tokens(str(row.get("prediction") or "")))
    if reference_length < min_missing_chars:
        return False
    return reference_length - prediction_length >= min_missing_chars and prediction_length / reference_length <= ratio_threshold


def has_repetition(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False

    compact = normalized.replace(" ", "")
    for size in range(2, 7):
        pattern = re.compile(rf"(.{{{size}}})\1{{3,}}")
        if pattern.search(compact):
            return True

    words = normalized.split()
    if len(words) < 6:
        return False
    counts = Counter(words)
    return any(count >= 5 for count in counts.values())


def has_subtitle_phrase(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return any(pattern.replace(" ", "") in compact for pattern in SUBTITLE_PATTERNS)


def has_non_korean_foreign_text(text: str) -> bool:
    cleaned = re.sub(r"[0-9a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ\s.,!?;:'\"()\[\]{}<>/%+-]", "", text)
    return bool(cleaned.strip())


def analyze_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counters = Counter()
    per_dataset = defaultdict(Counter)
    per_bucket = defaultdict(Counter)
    dataset_totals = Counter()
    bucket_totals = Counter()

    for row in rows:
        prediction_raw = str(row.get("prediction_raw") or row.get("prediction") or "")
        checks = {
            "empty_prediction": is_empty_prediction(row),
            "very_long_prediction": is_very_long_prediction(row),
            "very_short_prediction": is_very_short_prediction(row),
            "repetition_pattern": has_repetition(prediction_raw),
            "subtitle_phrase": has_subtitle_phrase(prediction_raw),
            "non_korean_foreign_text": has_non_korean_foreign_text(prediction_raw),
        }
        dataset = str(row.get("dataset", "unknown"))
        bucket = str(row.get("bucket", "unknown"))
        dataset_totals[dataset] += 1
        bucket_totals[bucket] += 1
        for name, matched in checks.items():
            if matched:
                counters[name] += 1
                per_dataset[dataset][name] += 1
                per_bucket[bucket][name] += 1

    return {
        "total_samples": total,
        "overall": format_counter(counters, total),
        "per_dataset": {
            key: {
                "total_samples": dataset_totals[key],
                "patterns": format_counter(per_dataset[key], dataset_totals[key]),
            }
            for key in sorted(dataset_totals)
        },
        "per_bucket": {
            key: {
                "total_samples": bucket_totals[key],
                "patterns": format_counter(per_bucket[key], bucket_totals[key]),
            }
            for key in sorted(bucket_totals)
        },
    }


def format_counter(counter: Counter, total: int) -> dict[str, dict[str, float | int]]:
    names = (
        "empty_prediction",
        "very_long_prediction",
        "very_short_prediction",
        "repetition_pattern",
        "subtitle_phrase",
        "non_korean_foreign_text",
    )
    return {
        name: {
            "count": int(counter.get(name, 0)),
            "percent": round(100.0 * counter.get(name, 0) / total, 4) if total else 0.0,
        }
        for name in names
    }


def collect_examples(rows: list[dict[str, Any]], max_examples: int) -> dict[str, list[dict[str, Any]]]:
    examples = defaultdict(list)
    for row in rows:
        prediction_raw = str(row.get("prediction_raw") or row.get("prediction") or "")
        checks = {
            "empty_prediction": is_empty_prediction(row),
            "very_long_prediction": is_very_long_prediction(row),
            "very_short_prediction": is_very_short_prediction(row),
            "repetition_pattern": has_repetition(prediction_raw),
            "subtitle_phrase": has_subtitle_phrase(prediction_raw),
            "non_korean_foreign_text": has_non_korean_foreign_text(prediction_raw),
        }
        for name, matched in checks.items():
            if matched and len(examples[name]) < max_examples:
                examples[name].append(
                    {
                        "id": row.get("id"),
                        "dataset": row.get("dataset"),
                        "bucket": row.get("bucket"),
                        "reference": row.get("reference"),
                        "prediction_raw": prediction_raw,
                        "length_ratio": length_ratio(row),
                    }
                )
    return dict(examples)
