import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.config import load_config
from runners.faster_whisper import apply_overrides, build_run_config, run_faster_whisper
from launchers.launcher_utils import (
    apply_result_root_override,
    evaluate_experiment,
    launch_shards,
    parse_gpu_ids,
    run_single_experiment as run_configured_experiment,
    setup_launcher_logging,
    validate_gpu_launcher_args,
)


LOGGER = logging.getLogger("faster_whisper_launcher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="faster-whisper로 벤치마크 manifest를 디코딩한다.")
    parser.add_argument("--config", type=Path, default=Path("configs/engines/faster_whisper_experiments.json"))
    parser.add_argument("--experiment")
    parser.add_argument("--manifest_path", type=Path)
    parser.add_argument("--result_root", type=Path)
    parser.add_argument("--result_dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--beam_size", type=int)
    parser.add_argument("--compute_type", choices=("float16", "float32", "int8", "int8_float16"))
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--device_index", type=int)
    parser.add_argument("--language")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids for launcher mode.")
    parser.add_argument("--no_evaluate", action="store_true")
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


def launch_experiment(config_path: Path, experiment_name: str, gpu_ids: list[int], args: argparse.Namespace) -> None:
    launch_shards(
        script_name="scripts/run_faster_whisper.py",
        config_path=config_path,
        experiment_name=experiment_name,
        worker_ids=gpu_ids,
        args=args,
        logger=LOGGER,
        project_root=PROJECT_ROOT,
        worker_options=lambda _shard_index, gpu_id: ["--device", "cuda", "--device_index", str(gpu_id)],
        optional_arg_names=["manifest_path", "result_root", "model", "beam_size", "compute_type", "language", "limit"],
        flag_names=["no_resume"],
    )


def launch_all_experiments(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    gpu_ids = parse_gpu_ids(args, config)
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
        run_configured_experiment(args, apply_overrides, build_run_config, run_faster_whisper)
        return
    launch_all_experiments(args)


if __name__ == "__main__":
    main()
