from typing import Any


FINETUNING_MANIFEST_FIELDS = ("id", "audio", "text", "duration", "dataset", "bucket", "split")
FINETUNING_BUCKETS = {"short", "mid", "long"}
FINETUNING_SPLITS = ("train", "dev", "eval")


def finetuning_manifest_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "id": row["id"],
        "audio": row["audio"],
        "text": row["text"],
        "duration": row["duration"],
        "dataset": row["dataset"],
        "bucket": row["bucket"],
        "split": split,
    }
