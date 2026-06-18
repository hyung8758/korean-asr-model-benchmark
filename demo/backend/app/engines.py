import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from app.config import DEFAULT_CONFIG_PATH, DEMO_CONFIG
from app.engine_models import EngineSpec, TranscriptionResult, engine_specs_from_config, parse_gpu_indices
from app.schemas import EngineInfo, EngineStatus
from app.worker_client import WorkerClient


LOGGER = logging.getLogger(__name__)
BACKEND_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class WorkerRecord:
    process: subprocess.Popen
    client: WorkerClient
    log_handle: object
    port: int


class EngineManager:
    def __init__(self, specs: list[EngineSpec], gpu_indices: list[int]):
        self.spec_order = [spec.id for spec in specs]
        self.specs = {spec.id: spec for spec in specs}
        self.gpu_indices = gpu_indices
        self.assigned_gpus: dict[str, int] = {}
        self.workers: dict[str, WorkerRecord] = {}
        self.worker_lock = threading.RLock()
        self.engine_locks = {spec.id: threading.RLock() for spec in specs}
        self.active_streams = {spec.id: 0 for spec in specs}
        self.status_lock = threading.RLock()
        self.statuses = {spec.id: self.inactive_status(spec.id) for spec in specs}
        self.cancelling_workers: set[str] = set()
        worker_config = DEMO_CONFIG.get("workers", {})
        self.worker_host = str(worker_config.get("host", "127.0.0.1"))
        self.worker_base_port = int(worker_config.get("base_port", 17000))
        self.worker_startup_timeout = float(worker_config.get("startup_timeout_seconds", 30))
        self.worker_load_timeout = float(worker_config.get("load_timeout_seconds", 600))
        self.parallel_preload = positive_int(worker_config.get("parallel_preload", 1), "workers.parallel_preload")
        self.assign_initial_gpus()

    def assign_initial_gpus(self) -> None:
        for engine_id, gpu_index in zip(self.spec_order, self.gpu_indices, strict=False):
            self.assigned_gpus[engine_id] = gpu_index
            self.statuses[engine_id] = self.waiting_status(engine_id)

    def inactive_status(self, engine_id: str) -> EngineStatus:
        return EngineStatus(id=engine_id, state="inactive", label="비활성화")

    def waiting_status(self, engine_id: str) -> EngineStatus:
        label = "모델 대기 중"
        return EngineStatus(id=engine_id, state="not_loaded", label=label, assigned_gpu=self.assigned_gpus.get(engine_id))

    def free_gpu_indices(self) -> list[int]:
        used = set(self.assigned_gpus.values())
        return [gpu for gpu in self.gpu_indices if gpu not in used]

    def can_activate(self, engine_id: str) -> bool:
        return engine_id not in self.assigned_gpus and bool(self.free_gpu_indices())

    def runtime_spec(self, engine_id: str) -> EngineSpec:
        spec = self.get_base_spec(engine_id)
        gpu_index = self.assigned_gpus.get(engine_id)
        if gpu_index is None:
            return spec
        if spec.kind == "whisper_cpp_server":
            server_url = f"http://{spec.server_host}:{spec.server_port}/inference"
            return replace(spec, device=f"cuda:{gpu_index}/server", server_url=server_url)
        return replace(spec, device=f"cuda:{gpu_index}")

    def engine_info(self, engine_id: str) -> EngineInfo:
        spec = self.runtime_spec(engine_id)
        assigned_gpu = self.assigned_gpus.get(engine_id)
        return EngineInfo(
            id=spec.id,
            name=spec.name,
            provider=spec.provider,
            model=spec.model,
            device=spec.server_url if spec.kind == "whisper_cpp_server" and assigned_gpu is not None else spec.device,
            theme=spec.theme,
            active=assigned_gpu is not None,
            assigned_gpu=assigned_gpu,
            can_activate=self.can_activate(engine_id),
            model_options=list(spec.model_options),
            language_options=list(spec.language_options),
            note=spec.note,
        )

    def status_with_resource_fields(self, status: EngineStatus) -> EngineStatus:
        return EngineStatus(
            id=status.id,
            state=status.state,
            label=status.label,
            assigned_gpu=self.assigned_gpus.get(status.id),
            can_activate=self.can_activate(status.id),
            load_time=status.load_time,
            error=status.error,
        )

    def list_engines(self) -> list[EngineInfo]:
        return [self.engine_info(engine_id) for engine_id in self.spec_order]

    def list_statuses(self) -> list[EngineStatus]:
        self.refresh_worker_statuses()
        with self.status_lock:
            return [self.status_with_resource_fields(self.statuses[engine_id]) for engine_id in self.spec_order]

    def refresh_worker_statuses(self) -> None:
        for engine_id in self.spec_order:
            if engine_id not in self.assigned_gpus:
                continue
            record = self.worker_record(engine_id)
            current = self.statuses[engine_id]
            if record is None:
                continue
            if record.process.poll() is not None:
                if current.state not in {"inactive", "not_loaded"}:
                    self.set_status(engine_id, "error", "엔진 종료", error=f"worker exited: {record.process.returncode}")
                continue
            if current.state in {"loading", "decoding"}:
                continue
            try:
                health = record.client.health()
            except Exception:
                continue
            health_error = self.validate_worker_health(engine_id, record, health)
            if health_error:
                self.set_status(engine_id, "error", "엔진 포트 충돌", error=health_error)
                continue
            worker_state = str(health.get("state", current.state))
            if worker_state == "ready":
                self.set_status(engine_id, "ready", "준비 완료", load_time=health.get("model_load_time"))
            elif worker_state == "error":
                self.set_status(engine_id, "error", "엔진 오류", error=str(health.get("error", "")))

    def set_status(
        self,
        engine_id: str,
        state: str,
        label: str,
        load_time: float | None = None,
        error: str = "",
    ) -> None:
        with self.status_lock:
            previous = self.statuses.get(engine_id)
            self.statuses[engine_id] = EngineStatus(
                id=engine_id,
                state=state,
                label=label,
                assigned_gpu=self.assigned_gpus.get(engine_id),
                can_activate=self.can_activate(engine_id),
                load_time=load_time if load_time is not None else previous.load_time if previous else None,
                error=error,
            )

    def preload_all(self, event_callback=None) -> None:
        thread = threading.Thread(
            target=self.preload_all_configured,
            args=(event_callback,),
            daemon=True,
        )
        thread.start()

    def preload_all_configured(self, event_callback=None) -> None:
        engine_ids = self.assigned_engine_ids()
        if self.parallel_preload <= 1 or len(engine_ids) <= 1:
            for engine_id in engine_ids:
                self.preload_one(engine_id, event_callback)
            return

        worker_count = min(self.parallel_preload, len(engine_ids))
        LOGGER.info("Preloading %d engines with %d parallel workers", len(engine_ids), worker_count)
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="model-preload") as executor:
            futures = {
                executor.submit(self.preload_one, engine_id, event_callback): engine_id
                for engine_id in engine_ids
            }
            for future in as_completed(futures):
                engine_id = futures[future]
                try:
                    future.result()
                except Exception:
                    LOGGER.exception("Unexpected preload failure for engine_id=%s", engine_id)

    def assigned_engine_ids(self) -> list[str]:
        return [engine_id for engine_id in self.spec_order if engine_id in self.assigned_gpus]

    def preload_one(self, engine_id: str, event_callback=None, raise_errors: bool = False) -> None:
        spec = self.runtime_spec(engine_id)
        if engine_id not in self.assigned_gpus:
            self.set_status(engine_id, "inactive", "비활성화")
            return

        started_at = time.perf_counter()
        try:
            self.set_status(engine_id, "loading", "모델 로딩 중")
            record = self.ensure_worker(engine_id)
            health = record.client.load()
            health_error = self.validate_worker_health(engine_id, record, health)
            if health_error:
                raise RuntimeError(health_error)
            elapsed = float(health.get("model_load_time") or (time.perf_counter() - started_at))
            self.set_status(engine_id, "ready", "준비 완료", load_time=elapsed)
            if event_callback:
                event_callback(
                    {
                        "engine_id": spec.id,
                        "engine": spec.name,
                        "model": spec.model,
                        "device": spec.device,
                        "assigned_gpu": self.assigned_gpus.get(engine_id),
                        "worker_url": record.client.base_url,
                        "model_load_time": round(elapsed, 6),
                        "status": "ok",
                    }
                )
        except Exception as exc:
            LOGGER.exception("Failed to preload worker for engine_id=%s", engine_id)
            self.stop_worker(engine_id)
            self.set_status(engine_id, "error", "로딩 실패", error=str(exc))
            if event_callback:
                event_callback(
                    {
                        "engine_id": spec.id,
                        "engine": spec.name,
                        "model": spec.model,
                        "device": spec.device,
                        "assigned_gpu": self.assigned_gpus.get(engine_id),
                        "status": "error",
                        "error": str(exc),
                    }
                )
            if raise_errors:
                raise

    def activate_engine(self, engine_id: str) -> EngineStatus:
        self.get_base_spec(engine_id)
        with self.status_lock:
            if engine_id in self.assigned_gpus:
                return self.status_with_resource_fields(self.statuses[engine_id])
            free_gpus = self.free_gpu_indices()
            if not free_gpus:
                raise RuntimeError("현재 모든 GPU가 사용 중입니다. 이 모델을 로딩하려면 다른 모델을 먼저 내려주세요.")
            self.assigned_gpus[engine_id] = free_gpus[0]
            self.statuses[engine_id] = self.waiting_status(engine_id)
        try:
            self.preload_one(engine_id, raise_errors=True)
        except Exception:
            self.unload_engine(engine_id)
            raise
        return self.status_with_resource_fields(self.statuses[engine_id])

    def deactivate_engine(self, engine_id: str) -> EngineStatus:
        self.get_base_spec(engine_id)
        with self.engine_locks[engine_id]:
            current = self.statuses[engine_id]
            if current.state in {"loading", "decoding"} or self.active_streams[engine_id] > 0:
                raise RuntimeError("모델이 로딩 또는 인식 중입니다. 작업이 끝난 뒤 다시 시도하세요.")
            self.unload_engine(engine_id)
            return self.status_with_resource_fields(self.statuses[engine_id])

    def cancel_current_work(self, engine_id: str) -> EngineStatus:
        self.get_base_spec(engine_id)
        if engine_id not in self.assigned_gpus:
            return self.status_with_resource_fields(self.statuses[engine_id])

        with self.status_lock:
            self.cancelling_workers.add(engine_id)
        self.active_streams[engine_id] = 0
        self.set_status(engine_id, "loading", "작업 중지 후 모델 재초기화 중")
        self.stop_worker(engine_id)

        thread = threading.Thread(
            target=self.reload_after_cancel,
            args=(engine_id,),
            daemon=True,
            name=f"reload-after-cancel-{safe_name(engine_id)}",
        )
        thread.start()
        return self.status_with_resource_fields(self.statuses[engine_id])

    def reload_after_cancel(self, engine_id: str) -> None:
        try:
            if engine_id in self.assigned_gpus:
                self.preload_one(engine_id)
        finally:
            with self.status_lock:
                self.cancelling_workers.discard(engine_id)

    def is_cancelling_worker(self, engine_id: str) -> bool:
        with self.status_lock:
            return engine_id in self.cancelling_workers

    def unload_engine(self, engine_id: str) -> None:
        self.set_status(engine_id, "unloading", "비활성화 중")
        self.stop_worker(engine_id)
        with self.status_lock:
            self.cancelling_workers.discard(engine_id)
        self.assigned_gpus.pop(engine_id, None)
        self.statuses[engine_id] = self.inactive_status(engine_id)

    def stop_all(self) -> None:
        for engine_id in self.worker_engine_ids():
            self.stop_worker(engine_id)

    def get_base_spec(self, engine_id: str) -> EngineSpec:
        if engine_id not in self.specs:
            names = ", ".join(self.specs)
            raise ValueError(f"알 수 없는 엔진: {engine_id}. 사용 가능: {names}")
        return self.specs[engine_id]

    def get_spec(self, engine_id: str) -> EngineSpec:
        spec = self.runtime_spec(engine_id)
        if engine_id not in self.assigned_gpus:
            raise RuntimeError(f"{spec.name}은 비활성화 상태입니다. 먼저 모델을 활성화하세요.")
        return spec

    def transcribe(
        self,
        engine_id: str,
        audio_path: Path,
        language: str,
        beam_size: int,
        temperature: float,
    ) -> TranscriptionResult:
        with self.engine_locks[engine_id]:
            self.get_spec(engine_id)
            record = self.require_ready_worker(engine_id)
            with self.status_lock:
                self.set_status(engine_id, "decoding", "인식 중")
            try:
                return record.client.transcribe(audio_path, language, beam_size, temperature)
            except Exception as exc:
                if self.is_cancelling_worker(engine_id):
                    LOGGER.info("Decode cancelled for engine_id=%s", engine_id)
                else:
                    LOGGER.exception("Decode failed for engine_id=%s", engine_id)
                    self.set_status(engine_id, "error", "인식 실패", error=str(exc))
                raise
            finally:
                if not self.is_cancelling_worker(engine_id) and self.statuses[engine_id].state != "error":
                    self.set_status(engine_id, "ready", "준비 완료")

    def start_stream(self, engine_id: str, language: str, beam_size: int, temperature: float) -> str:
        with self.engine_locks[engine_id]:
            record = self.require_ready_worker(engine_id)
            session_id = record.client.stream_start(language, beam_size, temperature)
            self.active_streams[engine_id] += 1
            return session_id

    def stream_chunk(
        self,
        engine_id: str,
        session_id: str,
        pcm: bytes,
        settings: tuple[int, int, int],
    ) -> TranscriptionResult:
        with self.engine_locks[engine_id]:
            record = self.require_ready_worker(engine_id)
            self.set_status(engine_id, "decoding", "인식 중")
            try:
                return record.client.stream_chunk(session_id, pcm, settings)
            finally:
                if not self.is_cancelling_worker(engine_id) and self.statuses[engine_id].state != "error":
                    self.set_status(engine_id, "ready", "준비 완료")

    def stream_finish(self, engine_id: str, session_id: str) -> TranscriptionResult:
        with self.engine_locks[engine_id]:
            record = self.require_ready_worker(engine_id)
            self.set_status(engine_id, "decoding", "인식 중")
            try:
                return record.client.stream_finish(session_id)
            finally:
                self.decrement_active_stream(engine_id)
                if not self.is_cancelling_worker(engine_id) and self.statuses[engine_id].state != "error":
                    self.set_status(engine_id, "ready", "준비 완료")

    def stream_cancel(self, engine_id: str, session_id: str) -> None:
        with self.engine_locks[engine_id]:
            record = self.worker_record(engine_id)
            if record and record.process.poll() is None:
                record.client.stream_cancel(session_id)
            self.decrement_active_stream(engine_id)

    def require_ready_worker(self, engine_id: str) -> WorkerRecord:
        current = self.statuses[engine_id]
        if current.state == "error":
            raise RuntimeError(current.error or f"{engine_id} worker is in error state.")
        record = self.ensure_worker(engine_id)
        health = record.client.health()
        health_error = self.validate_worker_health(engine_id, record, health)
        if health_error:
            self.set_status(engine_id, "error", "엔진 포트 충돌", error=health_error)
            raise RuntimeError(health_error)
        if health.get("state") != "ready":
            self.preload_one(engine_id, raise_errors=True)
            record = self.ensure_worker(engine_id)
        return record

    def ensure_worker(self, engine_id: str) -> WorkerRecord:
        with self.engine_locks[engine_id]:
            record = self.worker_record(engine_id)
            if record is not None and record.process.poll() is None:
                return record
            self.stop_worker(engine_id)
            return self.start_worker(engine_id)

    def start_worker(self, engine_id: str) -> WorkerRecord:
        gpu_index = self.assigned_gpus[engine_id]
        port = self.worker_port(engine_id)
        log_path = self.worker_log_path(engine_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "app.worker_process",
            "--engine-id",
            engine_id,
            "--gpu-index",
            str(gpu_index),
            "--host",
            self.worker_host,
            "--port",
            str(port),
        ]
        env = os.environ.copy()
        env["DEMO_CONFIG_PATH"] = str(Path(os.getenv("DEMO_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))).resolve())
        process = subprocess.Popen(
            command,
            cwd=str(BACKEND_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        record = WorkerRecord(
            process=process,
            client=WorkerClient(self.worker_url(port), timeout=self.worker_load_timeout),
            log_handle=log_handle,
            port=port,
        )
        with self.worker_lock:
            self.workers[engine_id] = record
        try:
            self.wait_for_worker_health(engine_id, record)
            return record
        except Exception:
            self.stop_worker(engine_id)
            raise

    def wait_for_worker_health(self, engine_id: str, record: WorkerRecord) -> None:
        deadline = time.perf_counter() + self.worker_startup_timeout
        last_error = ""
        while time.perf_counter() < deadline:
            if record.process.poll() is not None:
                raise RuntimeError(f"{engine_id} worker exited early: {record.process.returncode}")
            try:
                health = record.client.health()
                health_error = self.validate_worker_health(engine_id, record, health)
                if health_error:
                    raise RuntimeError(health_error)
                return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.25)
        raise TimeoutError(f"{engine_id} worker did not become ready. Last error: {last_error}")

    def validate_worker_health(self, engine_id: str, record: WorkerRecord, health: dict) -> str:
        if health.get("engine_id") != engine_id:
            return f"unexpected worker on port {record.port}: {health.get('engine_id')}"
        pid = health.get("pid")
        if pid is not None and int(pid) != record.process.pid:
            return f"stale worker on port {record.port}: pid={pid}, expected={record.process.pid}"
        return ""

    def stop_worker(self, engine_id: str) -> None:
        with self.worker_lock:
            record = self.workers.pop(engine_id, None)
        if record is None:
            return
        process = record.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        try:
            record.log_handle.close()
        except Exception:
            LOGGER.exception("Failed to close worker log for engine_id=%s", engine_id)

    def worker_port(self, engine_id: str) -> int:
        return self.worker_base_port + self.spec_order.index(engine_id)

    def worker_url(self, port: int) -> str:
        return f"http://{self.worker_host}:{port}"

    def worker_log_path(self, engine_id: str) -> Path:
        log_dir = Path(os.getenv("DEMO_LOG_DIR", "logs"))
        return log_dir / "workers" / f"{safe_name(engine_id)}.log"

    def worker_record(self, engine_id: str) -> WorkerRecord | None:
        with self.worker_lock:
            return self.workers.get(engine_id)

    def worker_engine_ids(self) -> list[str]:
        with self.worker_lock:
            return list(self.workers)

    def decrement_active_stream(self, engine_id: str) -> None:
        self.active_streams[engine_id] = max(0, self.active_streams[engine_id] - 1)


def safe_name(text: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)


def positive_int(value: object, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc
    if result < 1:
        raise ValueError(f"{field_name} must be a positive integer.")
    return result


engine_manager = EngineManager(
    specs=engine_specs_from_config(DEMO_CONFIG),
    gpu_indices=parse_gpu_indices(DEMO_CONFIG),
)
