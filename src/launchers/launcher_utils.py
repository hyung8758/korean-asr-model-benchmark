import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from core.config import find_experiment, load_config, result_dir_for
from core.io import read_jsonl, write_jsonl
from core.logging_utils import setup_logging


def parse_gpu_ids(args, config: dict[str, Any]) -> list[int]:
    if args.gpus:
        return [int(gpu_id.strip()) for gpu_id in args.gpus.split(",") if gpu_id.strip()]
    return [int(gpu_id) for gpu_id in config.get("launcher", {}).get("gpu_ids", [0])]


def validate_gpu_launcher_args(args, gpu_ids: list[int]) -> None:
    if args.result_dir is not None:
        raise ValueError("--result_dir can be used only with --experiment. Use --result_root in launcher mode.")
    if getattr(args, "device", None) == "cpu":
        raise ValueError("CPU launcher mode is not supported. Use --experiment with --device cpu for a small test.")
    if not gpu_ids:
        raise ValueError("At least one GPU id is required in launcher mode.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"GPU ids must be unique in launcher mode, got {gpu_ids}.")


def validate_launcher_args(args, worker_ids: list[int]) -> None:
    if args.result_dir is not None:
        raise ValueError("--result_dir can be used only with --experiment. Use --result_root in launcher mode.")
    if not worker_ids:
        raise ValueError("At least one worker id is required in launcher mode.")
    if len(set(worker_ids)) != len(worker_ids):
        raise ValueError(f"Worker ids must be unique in launcher mode, got {worker_ids}.")


def apply_result_root_override(config: dict[str, Any], args) -> None:
    if args.result_root is not None:
        config["result_root"] = str(args.result_root)


def setup_launcher_logging(config: dict[str, Any]) -> None:
    setup_logging(Path(config.get("result_root", "results")) / config["engine"] / "_launcher" / "logs" / "launcher.log")


def prepare_merged_manifest(config: dict[str, Any], logger: logging.Logger) -> None:
    if config.get("manifest_paths") and not config.get("manifest_path"):
        raise ValueError("Config with manifest_paths must also set manifest_path for the merged manifest output.")

    manifest_paths = [Path(path) for path in config.get("manifest_paths", [])]
    if not manifest_paths:
        return

    output_path = Path(config["manifest_path"])
    rows = []
    seen_ids = set()
    duplicates = []
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            logger.warning("Manifest path does not exist and will be skipped: %s", manifest_path)
            continue
        for row in read_jsonl(manifest_path):
            row_id = row["id"]
            if row_id in seen_ids:
                duplicates.append(row_id)
                continue
            seen_ids.add(row_id)
            rows.append(row)

    if duplicates:
        example = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate utterance ids found while merging manifests: {example}")
    if not rows:
        raise ValueError(f"No manifest rows loaded from manifest_paths={manifest_paths}")

    write_jsonl(output_path, rows)
    config["manifest_path"] = str(output_path)
    logger.info("Prepared merged manifest: %s (%s samples)", output_path, len(rows))


def run_single_experiment(
    args,
    apply_overrides: Callable[[dict[str, Any], dict[str, Any], Any], None],
    build_run_config: Callable[[dict[str, Any], dict[str, Any], Path | None], dict[str, Any]],
    run_decode: Callable[[dict[str, Any], Any], None],
) -> None:
    base_config = load_config(args.config)
    experiment = find_experiment(base_config, args.experiment)
    apply_overrides(base_config, experiment, args)
    run_config = build_run_config(base_config, experiment, args.result_dir)
    log_path = Path(run_config["result_dir"]) / "logs" / (
        "run.log" if args.num_shards == 1 else f"run.shard_{args.shard_index:03d}.log"
    )
    setup_logging(log_path)
    run_decode(run_config, args)


def add_optional_arg(command: list[str], name: str, value) -> None:
    if value is not None:
        command.extend([f"--{name}", str(value)])


def add_optional_args(command: list[str], args, names: list[str]) -> None:
    for name in names:
        add_optional_arg(command, name, getattr(args, name, None))


def add_flags(command: list[str], args, names: list[str]) -> None:
    for name in names:
        if getattr(args, name, False):
            command.append(f"--{name}")


def launch_shards(
    script_name: str,
    config_path: Path,
    experiment_name: str,
    worker_ids: list[int],
    args,
    logger: logging.Logger,
    project_root: Path,
    worker_options: Callable[[int, int], list[str]],
    optional_arg_names: list[str],
    flag_names: list[str],
) -> None:
    processes = []
    for shard_index, worker_id in enumerate(worker_ids):
        command = [
            sys.executable,
            script_name,
            "--config",
            str(config_path),
            "--experiment",
            experiment_name,
            "--num_shards",
            str(len(worker_ids)),
            "--shard_index",
            str(shard_index),
            "--worker",
            *worker_options(shard_index, worker_id),
        ]
        add_optional_args(command, args, optional_arg_names)
        add_flags(command, args, flag_names)
        logger.info("Launching shard %s on worker=%s for %s", shard_index, worker_id, experiment_name)
        processes.append((shard_index, subprocess.Popen(command, cwd=project_root)))

    wait_for_processes(processes, experiment_name)


def run_logged_command(command: list[str], logger: logging.Logger, project_root: Path) -> None:
    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=project_root, check=True)


def evaluate_experiment(
    config_path: Path,
    experiment_name: str,
    args,
    apply_overrides: Callable[[dict[str, Any], dict[str, Any], Any], None],
    logger: logging.Logger,
    project_root: Path,
) -> None:
    config = load_config(config_path)
    experiment = find_experiment(config, experiment_name)
    apply_overrides(config, experiment, args)
    result_dir = args.result_dir if args.result_dir is not None else Path(result_dir_for(config, experiment))
    command = [
        sys.executable,
        "scripts/evaluate_predictions.py",
        "--manifest_path",
        str(config["manifest_path"]),
        "--result_dir",
        str(result_dir),
    ]
    run_logged_command(command, logger, project_root)


def wait_for_processes(processes: list[tuple[int, subprocess.Popen]], experiment_name: str) -> None:
    failed = []
    for shard_index, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((shard_index, return_code))
    if failed:
        raise RuntimeError(f"Experiment {experiment_name} failed shards: {failed}")
