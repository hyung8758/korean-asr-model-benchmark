import logging
import shutil
from pathlib import Path
from typing import Any

from core.io import read_json, write_json
from core.precision import torch_dtype_from_precision
from finetuning.config import resolve_project_path


LOGGER = logging.getLogger(__name__)


def resolve_adapter_path(train_dir: Path, adapter_path: Path | None, checkpoint_path: Path | None) -> Path:
    if adapter_path is not None:
        return adapter_path
    if checkpoint_path is not None:
        return checkpoint_path

    best_checkpoint = find_best_checkpoint(train_dir)
    if best_checkpoint is not None:
        return best_checkpoint

    final_adapter_path = train_dir / "adapter"
    if is_adapter_dir(final_adapter_path):
        return final_adapter_path

    latest_checkpoint = find_latest_checkpoint(train_dir)
    if latest_checkpoint is not None:
        return latest_checkpoint

    raise FileNotFoundError(
        f"No LoRA adapter found. Expected {final_adapter_path} or a checkpoint under {train_dir / 'checkpoints'}."
    )


def is_adapter_dir(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").exists()


def find_best_checkpoint(train_dir: Path) -> Path | None:
    checkpoint_dirs = list_checkpoint_dirs(train_dir)
    if not checkpoint_dirs:
        return None
    best_checkpoint = find_best_checkpoint_from_trainer_state(train_dir, checkpoint_dirs)
    if best_checkpoint is not None and is_adapter_dir(best_checkpoint):
        return best_checkpoint
    return None


def find_latest_checkpoint(train_dir: Path) -> Path | None:
    checkpoint_dirs = list_checkpoint_dirs(train_dir)
    if not checkpoint_dirs:
        return None
    return checkpoint_dirs[-1]


def list_checkpoint_dirs(train_dir: Path) -> list[Path]:
    checkpoints_dir = train_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return []

    return sorted(
        [path for path in checkpoints_dir.glob("checkpoint-*") if is_adapter_dir(path)],
        key=checkpoint_step,
    )


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", maxsplit=1)[-1])
    except ValueError:
        return -1


def find_best_checkpoint_from_trainer_state(train_dir: Path, checkpoint_dirs: list[Path]) -> Path | None:
    for checkpoint_dir in reversed(checkpoint_dirs):
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        if not trainer_state_path.exists():
            continue
        trainer_state = read_json(trainer_state_path)
        best_checkpoint = trainer_state.get("best_model_checkpoint")
        if best_checkpoint:
            return resolve_checkpoint_from_state(train_dir, Path(best_checkpoint))
    return None


def resolve_checkpoint_from_state(train_dir: Path, checkpoint_path: Path) -> Path:
    if is_adapter_dir(checkpoint_path):
        return checkpoint_path
    current_train_checkpoint = train_dir / "checkpoints" / checkpoint_path.name
    if is_adapter_dir(current_train_checkpoint):
        return current_train_checkpoint
    return checkpoint_path


def read_run_config(train_dir: Path) -> dict[str, Any]:
    config_path = train_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    return read_json(config_path)


def default_output_dir(project_root: Path, train_dir: Path) -> Path:
    return project_root / "exp" / "merged" / train_dir.name


def validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)


def save_processor(base_model_name_or_path: str, train_dir: Path, adapter_path: Path, output_dir: Path) -> str:
    from transformers import WhisperProcessor

    for processor_path in (adapter_path, train_dir / "processor"):
        if (processor_path / "preprocessor_config.json").exists():
            processor = WhisperProcessor.from_pretrained(processor_path)
            processor.save_pretrained(output_dir)
            return str(processor_path)

    processor = WhisperProcessor.from_pretrained(base_model_name_or_path)
    processor.save_pretrained(output_dir)
    return base_model_name_or_path


def merge_whisper_lora(
    project_root: Path,
    train_dir: Path,
    adapter_path: Path | None = None,
    checkpoint_path: Path | None = None,
    output_dir: Path | None = None,
    base_model_name_or_path: str | None = None,
    precision: str = "float32",
    device: str = "cpu",
    overwrite: bool = False,
) -> dict[str, Any]:
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration

    train_dir = resolve_project_path(project_root, train_dir) or train_dir
    adapter_path = resolve_project_path(project_root, adapter_path) if adapter_path is not None else None
    checkpoint_path = resolve_project_path(project_root, checkpoint_path) if checkpoint_path is not None else None
    run_config = read_run_config(train_dir)

    selected_adapter_path = resolve_adapter_path(train_dir, adapter_path, checkpoint_path)
    if output_dir is None:
        output_dir = default_output_dir(project_root, train_dir)
    else:
        output_dir = resolve_project_path(project_root, output_dir) or output_dir

    base_model = base_model_name_or_path or str(run_config["base_model_name_or_path"])
    dtype = torch_dtype_from_precision(precision)

    validate_output_dir(output_dir, overwrite=overwrite)
    LOGGER.info("Loading base model: %s", base_model)
    model = WhisperForConditionalGeneration.from_pretrained(base_model, dtype=dtype)
    LOGGER.info("Loading LoRA adapter: %s", selected_adapter_path)
    model = PeftModel.from_pretrained(model, selected_adapter_path, is_trainable=False)
    if device != "cpu":
        model.to(device)

    LOGGER.info("Merging adapter into base model")
    merged_model = model.merge_and_unload()
    output_dir.mkdir(parents=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    processor_source = save_processor(base_model, train_dir, selected_adapter_path, output_dir)

    summary = {
        "train_dir": str(train_dir),
        "adapter_path": str(selected_adapter_path),
        "base_model_name_or_path": base_model,
        "output_dir": str(output_dir),
        "precision": precision,
        "device": device,
        "processor_source": processor_source,
    }
    write_json(output_dir / "merge_summary.json", summary)
    LOGGER.info("Wrote merged model: %s", output_dir)
    return summary
