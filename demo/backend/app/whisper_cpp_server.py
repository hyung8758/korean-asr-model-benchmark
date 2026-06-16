import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

from app.engine_models import PROJECT_ROOT, EngineSpec, project_path


class WhisperCppServerController:
    def __init__(self) -> None:
        self.processes: dict[str, subprocess.Popen] = {}

    def ensure_running(self, spec: EngineSpec, gpu_index: int) -> None:
        if self.is_available(spec):
            return

        binary = project_path(spec.server_binary)
        model_path = project_path(spec.server_model_path)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise FileNotFoundError(f"whisper.cpp server binary not found or not executable: {binary}")
        if not model_path.is_file():
            raise FileNotFoundError(f"whisper.cpp model not found: {model_path}")

        log_dir = Path(os.getenv("DEMO_LOG_DIR", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{spec.id}.server.log"
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = self.library_path(spec)

        log_file = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            self.command(spec, gpu_index),
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        setattr(process, "_stt_log_file", log_file)
        self.processes[spec.id] = process
        try:
            self.wait_until_ready(spec, process)
        except Exception:
            self.stop(spec.id)
            raise

    def command(self, spec: EngineSpec, gpu_index: int) -> list[str]:
        command = [
            str(project_path(spec.server_binary)),
            "--model",
            str(project_path(spec.server_model_path)),
            "--host",
            spec.server_host,
            "--port",
            str(spec.server_port),
            "--language",
            "ko",
            "--beam-size",
            "1",
            "--threads",
            str(spec.server_threads),
            "--processors",
            str(spec.server_processors),
            "--device",
            str(gpu_index),
            "--no-language-probabilities",
        ]
        command.append("--flash-attn" if spec.server_flash_attention else "--no-flash-attn")
        return command

    def library_path(self, spec: EngineSpec) -> str:
        build_dir = project_path(spec.server_binary).resolve().parents[1]
        paths = [build_dir / "src", build_dir / "ggml" / "src", build_dir / "ggml" / "src" / "ggml-cuda"]
        existing = [str(path) for path in paths if path.exists()]
        current = os.environ.get("LD_LIBRARY_PATH")
        if current:
            existing.append(current)
        return ":".join(existing)

    def wait_until_ready(self, spec: EngineSpec, process: subprocess.Popen) -> None:
        deadline = time.monotonic() + 120.0
        last_error = ""
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"whisper.cpp server exited early with code {process.returncode}")
            if self.is_available(spec):
                return
            try:
                response = requests.get(health_url(spec.server_url), timeout=2.0)
                last_error = response.text
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(0.5)
        raise TimeoutError(f"whisper.cpp server did not become ready at {spec.server_url}. Last error: {last_error}")

    def stop(self, engine_id: str) -> None:
        process = self.processes.pop(engine_id, None)
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        log_file = getattr(process, "_stt_log_file", None)
        if log_file is not None:
            log_file.close()

    def is_available(self, spec: EngineSpec) -> bool:
        try:
            response = requests.get(health_url(spec.server_url), timeout=0.5)
        except requests.RequestException:
            return False
        if response.status_code != 200:
            return False
        try:
            return response.json().get("status") == "ok"
        except ValueError:
            return False


def health_url(server_url: str) -> str:
    parts = urlsplit(server_url)
    return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))
