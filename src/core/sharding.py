from typing import Any


def select_shard(rows: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]
