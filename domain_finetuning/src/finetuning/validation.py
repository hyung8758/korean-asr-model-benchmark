import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from finetuning.manifest import FINETUNING_BUCKETS, FINETUNING_MANIFEST_FIELDS, FINETUNING_SPLITS

REQUIRED_FIELDS = FINETUNING_MANIFEST_FIELDS


def resolve_audio_path(project_root: Path, audio: str) -> Path:
    path = Path(audio)
    if path.is_absolute():
        return path
    return project_root / path


def validate_jsonl_file(
    data_path: Path,
    project_root: Path,
    max_errors: int,
) -> tuple[dict[str, Any], list[str]]:
    errors = []
    samples = 0
    first_row: dict[str, Any] | None = None
    required = set(REQUIRED_FIELDS)

    if not data_path.exists():
        return {"samples": 0, "first_row": None}, [f"missing data file: {data_path}"]

    with data_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except JSONDecodeError as exc:
                errors.append(f"{data_path}:{line_number} invalid json: {exc}")
                if len(errors) >= max_errors:
                    break
                continue

            if not isinstance(row, dict):
                errors.append(f"{data_path}:{line_number} row must be a json object")
                if len(errors) >= max_errors:
                    break
                continue

            samples += 1
            if first_row is None:
                first_row = row

            keys = set(row)
            missing = sorted(required - keys)
            extra = sorted(keys - required)
            if missing:
                errors.append(f"{data_path}:{line_number} missing fields: {missing}")
            if extra:
                errors.append(f"{data_path}:{line_number} extra fields: {extra}")
            if not str(row.get("id", "")).strip():
                errors.append(f"{data_path}:{line_number} empty id")
            if not str(row.get("text", "")).strip():
                errors.append(f"{data_path}:{line_number} empty text")
            if not str(row.get("dataset", "")).strip():
                errors.append(f"{data_path}:{line_number} empty dataset")
            if row.get("bucket") not in FINETUNING_BUCKETS:
                errors.append(f"{data_path}:{line_number} invalid bucket: {row.get('bucket')}")
            if row.get("split") != data_path.stem:
                errors.append(f"{data_path}:{line_number} invalid split: {row.get('split')}")

            try:
                duration = float(row.get("duration"))
            except (TypeError, ValueError):
                errors.append(f"{data_path}:{line_number} invalid duration: {row.get('duration')}")
            else:
                if duration <= 0.0:
                    errors.append(f"{data_path}:{line_number} non-positive duration: {duration}")

            audio = row.get("audio")
            if not isinstance(audio, str) or not audio.strip():
                errors.append(f"{data_path}:{line_number} empty audio")
            elif not resolve_audio_path(project_root, audio).exists():
                errors.append(f"{data_path}:{line_number} audio not found: {audio}")

            if len(errors) >= max_errors:
                break

    if samples == 0:
        errors.append(f"empty data file: {data_path}")

    return {"samples": samples, "first_row": first_row}, errors


def validate_finetuning_data(
    output_root: Path,
    project_root: Path,
    max_errors: int = 20,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"output_root": str(output_root), "splits": {}, "errors": []}
    for split in FINETUNING_SPLITS:
        if len(summary["errors"]) >= max_errors:
            break
        data_path = output_root / f"{split}.jsonl"
        split_summary, errors = validate_jsonl_file(
            data_path=data_path,
            project_root=project_root,
            max_errors=max_errors - len(summary["errors"]),
        )
        summary["splits"][split] = split_summary
        summary["errors"].extend(errors)
        if len(summary["errors"]) >= max_errors:
            break
    summary["ok"] = not summary["errors"]
    return summary
