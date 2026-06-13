import os
from pathlib import Path


def library_dirs_from_build_dir(build_dir: Path) -> list[Path]:
    candidates = [
        build_dir / "src",
        build_dir / "ggml" / "src",
        build_dir / "ggml" / "src" / "ggml-cuda",
    ]
    return [path for path in candidates if path.exists()]


def library_dirs_from_project(whisper_cpp_dir: Path) -> list[Path]:
    return library_dirs_from_build_dir(whisper_cpp_dir / "build")


def env_with_library_dirs(library_dirs: list[Path]) -> dict[str, str]:
    env = dict(os.environ)
    values = [str(path) for path in library_dirs]
    current = env.get("LD_LIBRARY_PATH")
    if current:
        values.append(current)
    env["LD_LIBRARY_PATH"] = ":".join(values)
    return env
