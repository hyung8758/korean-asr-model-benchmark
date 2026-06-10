from collections import Counter
from typing import Any


REQUIRED_FIELDS = (
    "id",
    "audio",
    "reference",
    "prediction",
    "prediction_raw",
    "dataset",
    "bucket",
    "duration",
    "decode_time",
    "rtf",
)


def validate_prediction_rows(rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    issues = []
    id_counts = Counter(str(row.get("id", "")) for row in rows)
    duplicate_ids = sorted(row_id for row_id, count in id_counts.items() if row_id and count > 1)
    manifest_ids = {str(row["id"]) for row in manifest_rows or []}

    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id", ""))
        for field in REQUIRED_FIELDS:
            if field not in row:
                issues.append({"line": index, "id": row_id, "issue": "missing_field", "field": field})

        if not row_id:
            issues.append({"line": index, "id": row_id, "issue": "empty_id"})
        if manifest_ids and row_id and row_id not in manifest_ids:
            issues.append({"line": index, "id": row_id, "issue": "id_not_in_manifest"})
        if not isinstance(row.get("segments", []), list):
            issues.append({"line": index, "id": row_id, "issue": "segments_not_list"})

        for field in ("duration", "decode_time"):
            value = row.get(field)
            if value is None:
                continue
            try:
                if float(value) < 0:
                    issues.append({"line": index, "id": row_id, "issue": "negative_value", "field": field})
            except (TypeError, ValueError):
                issues.append({"line": index, "id": row_id, "issue": "not_numeric", "field": field})

    for row_id in duplicate_ids:
        issues.append({"id": row_id, "issue": "duplicate_id", "count": id_counts[row_id]})

    missing_manifest_ids = sorted(manifest_ids - set(id_counts)) if manifest_ids else []
    return {
        "valid": not issues and not missing_manifest_ids,
        "total_predictions": len(rows),
        "total_manifest_samples": len(manifest_rows or []),
        "duplicate_ids": duplicate_ids,
        "missing_manifest_ids": missing_manifest_ids,
        "issues": issues,
    }
