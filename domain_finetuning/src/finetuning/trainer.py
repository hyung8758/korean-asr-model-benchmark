import logging
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from core.io import write_json
from finetuning.config import make_train_dir, resolve_project_path, write_resolved_config
from finetuning.dataset import WhisperDataCollator, WhisperManifestDataset


LOGGER = logging.getLogger(__name__)


class EpochCheckpointCallback(TrainerCallback):
    def __init__(self, train_dir: Path) -> None:
        self.train_dir = train_dir
        self.epoch_dir = train_dir / "epochs"

    def on_save(
        self,
        args: Seq2SeqTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if state.epoch is None:
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            return control

        epoch_index = int(round(state.epoch))
        epoch_path = self.epoch_dir / f"epoch_{epoch_index:03d}"
        self.epoch_dir.mkdir(parents=True, exist_ok=True)
        if epoch_path.exists() or epoch_path.is_symlink():
            if epoch_path.is_dir() and not epoch_path.is_symlink():
                shutil.rmtree(epoch_path)
            else:
                epoch_path.unlink()
        epoch_path.symlink_to(Path(os.path.relpath(checkpoint_dir, epoch_path.parent)))
        return control


def setup_whisper_generation(model, processor, language: str, task: str) -> None:
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []


def build_lora_config(config: dict[str, Any]) -> LoraConfig:
    lora = config["lora"]
    return LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias=str(lora["bias"]),
    )


def build_training_arguments(config: dict[str, Any], train_dir: Path) -> Seq2SeqTrainingArguments:
    training = config["training"]
    return Seq2SeqTrainingArguments(
        output_dir=str(train_dir / "checkpoints"),
        run_name=train_dir.name,
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        learning_rate=float(training["learning_rate"]),
        warmup_steps=int(training["warmup_steps"]),
        num_train_epochs=float(training["epoch"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        fp16=bool(training["fp16"]),
        bf16=bool(training["bf16"]),
        logging_steps=int(training["logging_steps"]),
        save_total_limit=training.get("save_total_limit"),
        dataloader_num_workers=int(training["dataloader_num_workers"]),
        predict_with_generate=bool(training["predict_with_generate"]),
        generation_max_length=int(training["generation_max_length"]),
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
    )


def list_epoch_checkpoints(train_dir: Path) -> list[Path]:
    epoch_dir = train_dir / "epochs"
    if not epoch_dir.exists():
        return []
    return sorted(path for path in epoch_dir.glob("epoch_*") if path.exists())


def find_best_epoch_checkpoint(train_dir: Path, best_checkpoint: str | None) -> Path | None:
    if best_checkpoint is None:
        return None

    best_path = Path(best_checkpoint).resolve()
    for epoch_path in list_epoch_checkpoints(train_dir):
        if epoch_path.resolve() == best_path:
            return epoch_path
    return None


def run_whisper_lora_training(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    torch.manual_seed(int(config["seed"]))

    train_path = resolve_project_path(project_root, config["train_path"])
    dev_path = resolve_project_path(project_root, config["dev_path"])
    if train_path is None or dev_path is None:
        raise ValueError("train_path and dev_path are required.")
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")
    if not dev_path.exists():
        raise FileNotFoundError(f"Dev data not found: {dev_path}")

    train_dir = make_train_dir(config, project_root)
    train_dir.mkdir(parents=True, exist_ok=False)
    write_resolved_config(train_dir, config)

    LOGGER.info("=== Whisper LoRA training started ===")
    LOGGER.info("Train dir: %s", train_dir)
    LOGGER.info("Base model: %s", config["base_model_name_or_path"])
    LOGGER.info("Train/dev/eval: %s / %s / %s", train_path, dev_path, resolve_project_path(project_root, config.get("eval_path")))
    LOGGER.info(
        "Epochs=%s save_strategy=epoch eval_strategy=epoch train_batch=%s eval_batch=%s grad_accum=%s fp16=%s bf16=%s",
        config["training"]["epoch"],
        config["training"]["per_device_train_batch_size"],
        config["training"]["per_device_eval_batch_size"],
        config["training"]["gradient_accumulation_steps"],
        config["training"]["fp16"],
        config["training"]["bf16"],
    )

    LOGGER.info("Loading processor: %s", config["base_model_name_or_path"])
    processor = WhisperProcessor.from_pretrained(
        config["base_model_name_or_path"],
        language=config["language"],
        task=config["task"],
    )

    LOGGER.info("Loading base model: %s", config["base_model_name_or_path"])
    model = WhisperForConditionalGeneration.from_pretrained(config["base_model_name_or_path"])
    setup_whisper_generation(model, processor, language=str(config["language"]), task=str(config["task"]))

    if bool(config["training"]["gradient_checkpointing"]):
        model.config.use_cache = False
        model.generation_config.use_cache = False
        model.enable_input_require_grads()

    LOGGER.info("Attaching LoRA adapters")
    model = get_peft_model(model, build_lora_config(config))
    model.print_trainable_parameters()

    train_dataset = WhisperManifestDataset(
        manifest_path=train_path,
        processor=processor,
        project_root=project_root,
        sampling_rate=int(config["sampling_rate"]),
    )
    dev_dataset = WhisperManifestDataset(
        manifest_path=dev_path,
        processor=processor,
        project_root=project_root,
        sampling_rate=int(config["sampling_rate"]),
    )
    collator = WhisperDataCollator(processor)

    trainer = Seq2SeqTrainer(
        args=build_training_arguments(config, train_dir),
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        processing_class=processor,
        callbacks=[EpochCheckpointCallback(train_dir)],
    )

    resume_from_checkpoint = config["training"].get("resume_from_checkpoint")
    if resume_from_checkpoint:
        resume_from_checkpoint = str(resolve_project_path(project_root, resume_from_checkpoint))

    LOGGER.info("Starting LoRA training: train=%s dev=%s output=%s", train_path, dev_path, train_dir)
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(train_dir / "adapter"))
    processor.save_pretrained(train_dir / "processor")

    eval_path = resolve_project_path(project_root, config.get("eval_path"))
    epoch_checkpoints = list_epoch_checkpoints(train_dir)
    best_epoch_checkpoint = find_best_epoch_checkpoint(train_dir, trainer.state.best_model_checkpoint)
    summary = {
        "train_dir": str(train_dir),
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "eval_path": str(eval_path) if eval_path is not None else None,
        "train_samples": len(train_dataset),
        "dev_samples": len(dev_dataset),
        "train_metrics": train_result.metrics,
        "global_step": trainer.state.global_step,
        "best_metric": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_epoch_checkpoint": str(best_epoch_checkpoint) if best_epoch_checkpoint is not None else None,
        "epoch_checkpoints": [str(path) for path in epoch_checkpoints],
        "adapter_path": str(train_dir / "adapter"),
        "processor_path": str(train_dir / "processor"),
        "training_summary_path": str(train_dir / "training_summary.json"),
    }
    write_json(train_dir / "training_summary.json", summary)
    if trainer.state.best_metric is not None:
        LOGGER.info("Best eval loss: %s", trainer.state.best_metric)
    if trainer.state.best_model_checkpoint is not None:
        LOGGER.info("Best epoch checkpoint: %s", trainer.state.best_model_checkpoint)
    if best_epoch_checkpoint is not None:
        LOGGER.info("Best epoch alias: %s", best_epoch_checkpoint)
    LOGGER.info("Best epoch adapter: %s", train_dir / "adapter")
    LOGGER.info("Training summary: %s", train_dir / "training_summary.json")
    LOGGER.info("=== DONE Whisper LoRA training ===")
    return summary
