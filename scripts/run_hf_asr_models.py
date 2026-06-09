import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.cuda import validate_cuda_devices
from core.config import load_config
from runners.hf_asr import apply_overrides, build_run_config, run_hf_asr
from core.io import read_jsonl, write_jsonl
from launchers.launcher_utils import (
    apply_result_root_override,
    evaluate_experiment,
    launch_shards,
    parse_gpu_ids,
    run_single_experiment as run_configured_experiment,
    setup_launcher_logging,
    validate_gpu_launcher_args,
)


LOGGER = logging.getLogger("hf_asr_launcher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode benchmark manifests with selected Hugging Face ASR models.")
    parser.add_argument("--config", type=Path, default=Path("configs/hf_asr_experiments.json"))
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


def ensure_manifest(config: dict) -> None:
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
            LOGGER.warning("Manifest path does not exist and will be skipped: %s", manifest_path)
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
    LOGGER.info("Prepared merged manifest: %s (%s samples)", output_path, len(rows))


def launch_experiment(config_path: Path, experiment_name: str, gpu_ids: list[int], args: argparse.Namespace) -> None:
    launch_shards(
        script_name="scripts/run_hf_asr_models.py",
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
    ensure_manifest(config)
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
    ensure_manifest(config)
    if args.manifest_path is None:
        args.manifest_path = Path(config["manifest_path"])
    run_configured_experiment(args, apply_overrides, build_run_config, run_hf_asr)


def main() -> None:
    args = parse_args()
    if args.worker or args.experiment:
        run_worker_or_single(args)
        return
    launch_all_experiments(args)


if __name__ == "__main__":
    main()
