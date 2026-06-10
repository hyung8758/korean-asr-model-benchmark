import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.cuda import validate_cuda_devices
from core.config import load_config
from launchers.launcher_utils import (
    apply_result_root_override,
    evaluate_experiment,
    launch_shards,
    parse_gpu_ids,
    prepare_merged_manifest,
    run_single_experiment as run_configured_experiment,
    setup_launcher_logging,
    validate_gpu_launcher_args,
)
from runners.qwen_speech_recognition import apply_overrides, build_run_config, run_qwen_speech_recognition


LOGGER = logging.getLogger("qwen_speech_recognition_launcher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen Speech Recognition 모델로 벤치마크 manifest를 디코딩한다.")
    parser.add_argument("--config", type=Path, default=Path("configs/engines/qwen_speech_recognition_experiments.json"))
    parser.add_argument("--experiment")
    parser.add_argument("--manifest_path", type=Path)
    parser.add_argument("--result_root", type=Path)
    parser.add_argument("--result_dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--beam_size", type=int)
    parser.add_argument("--precision", choices=("float16", "float32", "bfloat16"))
    parser.add_argument("--device")
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


def launch_experiment(config_path: Path, experiment_name: str, gpu_ids: list[int], args: argparse.Namespace) -> None:
    launch_shards(
        script_name="scripts/run_qwen_speech_recognition.py",
        config_path=config_path,
        experiment_name=experiment_name,
        worker_ids=gpu_ids,
        args=args,
        logger=LOGGER,
        project_root=PROJECT_ROOT,
        worker_options=lambda _shard_index, gpu_id: ["--device", f"cuda:{gpu_id}"],
        optional_arg_names=["manifest_path", "result_root", "model", "beam_size", "precision", "language", "limit"],
        flag_names=["no_resume", "retry_errors"],
    )


def launch_all_experiments(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    gpu_ids = parse_gpu_ids(args, config)
    validate_gpu_launcher_args(args, gpu_ids)
    apply_result_root_override(config, args)
    setup_launcher_logging(config)
    validate_cuda_devices(gpu_ids)
    prepare_merged_manifest(config, LOGGER)
    LOGGER.info("Launcher mode: %s experiments, GPUs=%s", len(config["experiments"]), gpu_ids)

    for experiment in config["experiments"]:
        experiment_name = experiment["name"]
        LOGGER.info("Starting experiment: %s", experiment_name)
        launch_experiment(args.config, experiment_name, gpu_ids, args)
        should_evaluate = config.get("launcher", {}).get("evaluate_after_decode", True) and not args.no_evaluate
        if should_evaluate:
            evaluate_experiment(args.config, experiment_name, args, apply_overrides, LOGGER, PROJECT_ROOT)
        LOGGER.info("Finished experiment: %s", experiment_name)


def run_worker_or_single(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prepare_merged_manifest(config, LOGGER)
    if args.manifest_path is None:
        args.manifest_path = Path(config["manifest_path"])
    run_configured_experiment(args, apply_overrides, build_run_config, run_qwen_speech_recognition)


def main() -> None:
    args = parse_args()
    if args.worker or args.experiment:
        run_worker_or_single(args)
        return
    launch_all_experiments(args)


if __name__ == "__main__":
    main()
