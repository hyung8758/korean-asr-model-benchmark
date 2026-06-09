import argparse
import logging
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stt_benchmark.config import find_experiment, load_config
from stt_benchmark.launcher_utils import (
    add_optional_arg,
    apply_result_root_override,
    evaluate_experiment,
    parse_gpu_ids,
    run_single_experiment as run_configured_experiment,
    setup_launcher_logging,
    validate_launcher_args,
    validate_gpu_launcher_args,
    wait_for_processes,
)
from stt_benchmark.whisper_cpp_runner import apply_overrides, build_run_config, run_whisper_cpp, validate_runtime


LOGGER = logging.getLogger("whisper_cpp_launcher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode benchmark manifest with whisper.cpp.")
    parser.add_argument("--config", type=Path, default=Path("configs/whisper_cpp_experiments.json"))
    parser.add_argument("--experiment")
    parser.add_argument("--manifest_path", type=Path)
    parser.add_argument("--result_root", type=Path)
    parser.add_argument("--result_dir", type=Path)
    parser.add_argument("--binary_path", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--beam_size", type=int)
    parser.add_argument("--quantization")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--device_index", type=int)
    parser.add_argument("--language")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--retry_errors", action="store_true")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids for launcher mode.")
    parser.add_argument("--no_evaluate", action="store_true")
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


def validate_experiment_runtime(config_path: Path, experiment_name: str, args: argparse.Namespace) -> None:
    config = load_config(config_path)
    experiment = find_experiment(config, experiment_name)
    apply_overrides(config, experiment, args)
    run_config = build_run_config(config, experiment, args.result_dir)
    validate_runtime(run_config)


def launch_experiment(config_path: Path, experiment_name: str, gpu_ids: list[int], args: argparse.Namespace) -> None:
    validate_experiment_runtime(config_path, experiment_name, args)

    processes = []
    config = load_config(config_path)
    device = args.device or config.get("device", "cuda")
    for shard_index, gpu_id in enumerate(gpu_ids):
        command = [
            sys.executable,
            "scripts/run_whisper_cpp.py",
            "--config",
            str(config_path),
            "--experiment",
            experiment_name,
            "--num_shards",
            str(len(gpu_ids)),
            "--shard_index",
            str(shard_index),
            "--device",
            device,
            "--worker",
        ]
        if device != "cpu":
            add_optional_arg(command, "device_index", gpu_id)
        add_optional_arg(command, "manifest_path", args.manifest_path)
        add_optional_arg(command, "result_root", args.result_root)
        add_optional_arg(command, "binary_path", args.binary_path)
        add_optional_arg(command, "model", args.model)
        add_optional_arg(command, "model_path", args.model_path)
        add_optional_arg(command, "beam_size", args.beam_size)
        add_optional_arg(command, "quantization", args.quantization)
        add_optional_arg(command, "language", args.language)
        add_optional_arg(command, "limit", args.limit)
        if args.no_resume:
            command.append("--no_resume")
        if args.retry_errors:
            command.append("--retry_errors")
        LOGGER.info("Launching shard %s on %s worker=%s for %s", shard_index, device, gpu_id, experiment_name)
        processes.append((shard_index, subprocess.Popen(command, cwd=PROJECT_ROOT)))

    wait_for_processes(processes, experiment_name)


def launch_all_experiments(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    gpu_ids = parse_gpu_ids(args, config)
    device = args.device or config.get("device", "cuda")
    if device == "cpu":
        validate_launcher_args(args, gpu_ids)
    else:
        validate_gpu_launcher_args(args, gpu_ids)
    apply_result_root_override(config, args)
    setup_launcher_logging(config)
    LOGGER.info("Launcher mode: %s experiments, GPUs=%s", len(config["experiments"]), gpu_ids)

    for experiment in config["experiments"]:
        experiment_name = experiment["name"]
        LOGGER.info("Starting experiment: %s", experiment_name)
        launch_experiment(args.config, experiment_name, gpu_ids, args)
        should_evaluate = config.get("launcher", {}).get("evaluate_after_decode", True) and not args.no_evaluate
        if should_evaluate:
            evaluate_experiment(args.config, experiment_name, args, apply_overrides, LOGGER, PROJECT_ROOT)
        LOGGER.info("Finished experiment: %s", experiment_name)


def main() -> None:
    args = parse_args()
    if args.worker or args.experiment:
        run_configured_experiment(args, apply_overrides, build_run_config, run_whisper_cpp)
        return
    launch_all_experiments(args)


if __name__ == "__main__":
    main()
