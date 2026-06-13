import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from core.io import append_jsonl
from decoding.run_utils import make_error_row, make_prediction_row


@dataclass
class DecodeOutput:
    prediction_raw: str
    segments: list[dict[str, Any]]
    decode_time: float | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)


DecodeOne = Callable[[dict[str, Any]], DecodeOutput]


def decode_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    prediction_path: Path,
    error_path: Path,
    done_ids: set[str],
    limit: int | None,
    decode_one: DecodeOne,
    logger: logging.Logger,
) -> tuple[int, int]:
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    decoded_count = 0
    error_count = 0

    with prediction_path.open("a", encoding="utf-8") as prediction_file, error_path.open(
        "a", encoding="utf-8"
    ) as error_file:
        for item in tqdm(rows, desc="Decoding"):
            if item["id"] in done_ids:
                continue
            if limit is not None and decoded_count + error_count >= limit:
                break

            start = time.perf_counter()
            try:
                output = decode_one(item)
                decode_time = output.decode_time
                if decode_time is None:
                    decode_time = time.perf_counter() - start

                row = make_prediction_row(
                    item=item,
                    prediction_raw=output.prediction_raw,
                    segments=output.segments,
                    decode_time=decode_time,
                    config=config,
                )
                row.update(output.extra_fields)
                append_jsonl(prediction_file, row)
                decoded_count += 1
            except Exception as exc:
                decode_time = time.perf_counter() - start
                append_jsonl(error_file, make_error_row(item, "decode_failed", str(exc), decode_time, config))
                logger.exception("Decode failed for id=%s", item["id"])
                error_count += 1

    return decoded_count, error_count
