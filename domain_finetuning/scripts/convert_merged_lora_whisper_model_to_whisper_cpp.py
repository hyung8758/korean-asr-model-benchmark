"""Merge된 Hugging Face Whisper 모델을 whisper.cpp용 ggml 모델로 변환한다."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any


DOMAIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DOMAIN_ROOT.parent
sys.path.insert(0, str(DOMAIN_ROOT / "src"))


LOGGER = logging.getLogger("convert_merged_lora_whisper_model_to_whisper_cpp")
DEFAULT_QUANTIZATIONS = ("q8_0", "q5_0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge된 Whisper 모델을 whisper.cpp ggml/q8_0/q5_0 모델로 변환한다.")
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--whisper_cpp_dir", type=Path, default=REPO_ROOT / "third_party" / "whisper.cpp")
    parser.add_argument("--whisper_python_root", type=Path, default=None)
    parser.add_argument("--quantizations", nargs="+", default=list(DEFAULT_QUANTIZATIONS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return DOMAIN_ROOT / path


def default_output_dir(model_dir: Path) -> Path:
    return DOMAIN_ROOT / "exp" / "converted_model" / model_dir.name


def find_whisper_python_root() -> Path:
    import whisper

    return Path(whisper.__file__).resolve().parents[1]


def validate_whisper_python_root(path: Path) -> None:
    mel_filters = path / "whisper" / "assets" / "mel_filters.npz"
    if not mel_filters.exists():
        raise FileNotFoundError(f"OpenAI Whisper assets not found: {mel_filters}")


def validate_model_dir(path: Path) -> None:
    required_files = ("config.json", "vocab.json", "added_tokens.json")
    missing = [name for name in required_files if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Merged model directory is missing files: {missing}")
    if not (path / "model.safetensors").exists() and not (path / "pytorch_model.bin").exists():
        raise FileNotFoundError(f"Merged model weights not found under: {path}")


def find_convert_script(whisper_cpp_dir: Path) -> Path:
    script = whisper_cpp_dir / "models" / "convert-h5-to-ggml.py"
    if not script.exists():
        raise FileNotFoundError(f"whisper.cpp conversion script not found: {script}")
    return script


def find_quantize_binary(whisper_cpp_dir: Path) -> Path:
    candidates = (
        whisper_cpp_dir / "build" / "bin" / "whisper-quantize",
        whisper_cpp_dir / "build" / "bin" / "quantize",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "whisper.cpp quantize binary not found. Build whisper.cpp first, expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def run_command(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    LOGGER.info("Running: %s", " ".join(command))
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def ensure_command_succeeded(completed: subprocess.CompletedProcess) -> None:
    if completed.returncode == 0:
        return

    output = completed.stdout or ""
    tail = "\n".join(output.splitlines()[-40:])
    raise RuntimeError(
        "Command failed with exit code "
        f"{completed.returncode}: {' '.join(str(part) for part in completed.args)}\n{tail}"
    )


def write_command_log(log_path: Path, title: str, completed: subprocess.CompletedProcess) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== {title} =====\n")
        handle.write(" ".join(str(part) for part in completed.args))
        handle.write("\n")
        handle.write(f"returncode={completed.returncode}\n")
        handle.write(completed.stdout or "")
        handle.write("\n")


def convert_to_ggml(
    model_dir: Path,
    whisper_cpp_dir: Path,
    whisper_python_root: Path,
    output_dir: Path,
    overwrite: bool,
    log_path: Path,
) -> Path:
    ggml_path = output_dir / "ggml-model.bin"
    if ggml_path.exists() and not overwrite:
        LOGGER.info("Reusing existing ggml model: %s", ggml_path)
        return ggml_path
    if ggml_path.exists():
        ggml_path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(find_convert_script(whisper_cpp_dir)),
        str(model_dir),
        str(whisper_python_root),
        str(output_dir),
    ]
    completed = run_command(command, cwd=REPO_ROOT)
    write_command_log(log_path, "convert-h5-to-ggml", completed)
    ensure_command_succeeded(completed)
    if not ggml_path.exists() or ggml_path.stat().st_size == 0:
        raise RuntimeError(f"ggml conversion did not create a valid file: {ggml_path}")
    return ggml_path


def quantize_model(
    quantize_binary: Path,
    ggml_path: Path,
    output_dir: Path,
    quantization: str,
    overwrite: bool,
    log_path: Path,
    command_env: dict[str, str],
) -> Path:
    quantized_path = output_dir / f"ggml-model-{quantization}.bin"
    if quantized_path.exists() and not overwrite:
        LOGGER.info("Reusing existing quantized model: %s", quantized_path)
        return quantized_path
    if quantized_path.exists():
        quantized_path.unlink()

    command = [
        str(quantize_binary),
        str(ggml_path),
        str(quantized_path),
        quantization,
    ]
    completed = run_command(command, cwd=REPO_ROOT, env=command_env)
    write_command_log(log_path, f"quantize-{quantization}", completed)
    ensure_command_succeeded(completed)
    if not quantized_path.exists() or quantized_path.stat().st_size == 0:
        raise RuntimeError(f"Quantization did not create a valid file: {quantized_path}")
    return quantized_path


def model_file_summary(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
    }


def convert_merged_lora_whisper_model_to_whisper_cpp(args: argparse.Namespace) -> dict[str, Any]:
    from core.io import write_json
    from whisper_cpp.runtime import env_with_library_dirs, library_dirs_from_project

    model_dir = resolve_project_path(args.model_dir)
    whisper_cpp_dir = resolve_project_path(args.whisper_cpp_dir)
    output_dir = resolve_project_path(args.output_dir) if args.output_dir is not None else default_output_dir(model_dir)
    whisper_python_root = (
        resolve_project_path(args.whisper_python_root)
        if args.whisper_python_root is not None
        else find_whisper_python_root()
    )

    validate_model_dir(model_dir)
    validate_whisper_python_root(whisper_python_root)
    quantize_binary = find_quantize_binary(whisper_cpp_dir)
    library_dirs = library_dirs_from_project(whisper_cpp_dir)
    whisper_cpp_env = env_with_library_dirs(library_dirs)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "convert.log"

    ggml_path = convert_to_ggml(
        model_dir=model_dir,
        whisper_cpp_dir=whisper_cpp_dir,
        whisper_python_root=whisper_python_root,
        output_dir=output_dir,
        overwrite=args.overwrite,
        log_path=log_path,
    )
    quantized_paths = [
        quantize_model(
            quantize_binary=quantize_binary,
            ggml_path=ggml_path,
            output_dir=output_dir,
            quantization=quantization,
            overwrite=args.overwrite,
            log_path=log_path,
            command_env=whisper_cpp_env,
        )
        for quantization in args.quantizations
    ]

    summary = {
        "model_dir": str(model_dir),
        "whisper_cpp_dir": str(whisper_cpp_dir),
        "whisper_cpp_library_dirs": [str(path) for path in library_dirs],
        "whisper_python_root": str(whisper_python_root),
        "output_dir": str(output_dir),
        "ggml_model": model_file_summary(ggml_path),
        "quantized_models": [model_file_summary(path) for path in quantized_paths],
        "log_path": str(log_path),
    }
    write_json(output_dir / "conversion_summary.json", summary)
    return summary


def main() -> None:
    from core.logging_utils import setup_logging

    args = parse_args()
    output_dir = resolve_project_path(args.output_dir) if args.output_dir is not None else default_output_dir(resolve_project_path(args.model_dir))
    setup_logging(output_dir / "convert.log")
    LOGGER.info("=== whisper.cpp conversion started ===")
    summary = convert_merged_lora_whisper_model_to_whisper_cpp(args)
    LOGGER.info("ggml model: %s", summary["ggml_model"]["path"])
    for model in summary["quantized_models"]:
        LOGGER.info("quantized model: %s", model["path"])
    LOGGER.info("summary: %s", Path(summary["output_dir"]) / "conversion_summary.json")
    LOGGER.info("=== DONE whisper.cpp conversion ===")


if __name__ == "__main__":
    main()
