from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from pydantic import BaseModel, Field
from tqdm.auto import tqdm as base_tqdm


APP_HOST = os.getenv("MLX_STUDIO_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("MLX_STUDIO_PORT", "8111"))
INFERENCE_PORT = int(os.getenv("MLX_STUDIO_INFERENCE_PORT", "8112"))
INFERENCE_URL = f"http://127.0.0.1:{INFERENCE_PORT}"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StartRequest(BaseModel):
    model_id: str = Field(min_length=3, max_length=300)
    backend: Literal["lm", "vlm"] = "lm"
    trust_remote_code: bool = False
    draft_model: str | None = Field(default=None, max_length=300)
    num_draft_tokens: int = Field(default=3, ge=1, le=16)
    decode_concurrency: int = Field(default=32, ge=1, le=128)
    prompt_concurrency: int = Field(default=8, ge=1, le=64)
    prefill_step_size: int = Field(default=2048, ge=128, le=32768)
    prompt_cache_size: int = Field(default=10, ge=0, le=100)
    prompt_cache_gb: int = Field(default=0, ge=0, le=1024)
    pipeline: bool = False
    default_max_tokens: int = Field(default=4096, ge=1, le=262144)
    kv_bits: Literal["none", "8", "3.5"] = "none"
    kv_group_size: int = Field(default=64, ge=16, le=256)
    max_kv_size: int = Field(default=0, ge=0, le=1048576)
    quantized_kv_start: int = Field(default=0, ge=0, le=1048576)
    expert_cache_gb: int = Field(default=0, ge=0, le=1024)
    max_num_seqs: int = Field(default=0, ge=0, le=128)
    vision_cache_size: int = Field(default=20, ge=0, le=128)
    apc_enabled: bool = False
    apc_num_blocks: int = Field(default=4096, ge=128, le=65536)
    draft_kind: Literal["auto", "dflash", "eagle3", "mtp"] = "auto"
    draft_block_size: int = Field(default=0, ge=0, le=64)


class DownloadCancelled(Exception):
    """Raised inside Hugging Face progress callbacks when a download is cancelled."""


class RuntimeController:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.ps_process: psutil.Process | None = None
        self.model_id: str | None = None
        self.backend: str | None = None
        self.phase = "stopped"
        self.started_at: float | None = None
        self.last_error: str | None = None
        self.logs: deque[dict[str, Any]] = deque(maxlen=600)
        self.history: deque[dict[str, Any]] = deque(maxlen=120)
        self.lock = threading.RLock()
        self.request_count = 0
        self.error_count = 0
        self.active_requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_latency_ms = 0.0
        self.last_ttft_ms = 0.0
        self.total_ttft_ms = 0.0
        self.ttft_samples = 0
        self.last_decode_tps = 0.0
        self.total_decode_tps = 0.0
        self.decode_tps_samples = 0
        self.last_prefill_tps = 0.0
        self.total_prefill_tps = 0.0
        self.prefill_tps_samples = 0
        self.cache_hit_tokens = 0
        self.last_draft_kind: str | None = None
        self.last_draft_rounds = 0
        self.last_draft_tokens = 0
        self.last_draft_accepted_tokens = 0
        self.draft_rounds = 0
        self.draft_tokens = 0
        self.draft_accepted_tokens = 0
        self.performance: dict[str, Any] = {}
        self.job_id = 0
        self.download_thread: threading.Thread | None = None
        self.download_cancel = threading.Event()
        self.download_transfer_started_at: float | None = None
        self.download_cached_bytes = 0
        self.download = self._empty_download()

    @staticmethod
    def _empty_download() -> dict[str, Any]:
        return {
            "status": "idle",
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "remaining_bytes": 0,
            "percent": 0.0,
            "files_total": 0,
            "files_remaining": 0,
            "speed_bytes_per_second": 0,
            "eta_seconds": None,
            "started_at": None,
            "repository": None,
            "repository_index": 0,
            "repositories_total": 0,
        }

    def _log(self, message: str, level: str = "info") -> None:
        clean = message.rstrip()
        if not clean:
            return
        with self.lock:
            self.logs.append({"time": time.time(), "level": level, "message": clean})

    def _reader(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            level = "error" if any(word in line.lower() for word in ("error", "traceback", "exception")) else "info"
            self._log(line, level)
        return_code = process.wait()
        with self.lock:
            if self.process is process:
                if self.phase not in {"stopping", "stopped"} and return_code != 0:
                    self.phase = "error"
                    self.last_error = f"Inference server exited with code {return_code}."
                else:
                    self.phase = "stopped"
                self.process = None
                self.ps_process = None
        self._log(f"Inference process exited ({return_code}).", "error" if return_code else "info")

    def _wait_until_ready(self, process: subprocess.Popen[str]) -> None:
        deadline = time.time() + 1800
        while process.poll() is None and time.time() < deadline:
            try:
                response = httpx.get(f"{INFERENCE_URL}/v1/models", timeout=1.5)
                if response.status_code < 500:
                    with self.lock:
                        if self.process is process:
                            self.phase = "ready"
                    self._log("OpenAI-compatible endpoint is ready.")
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)

    def _record_download_progress(self, job_id: int, downloaded_missing_bytes: int) -> None:
        with self.lock:
            if job_id != self.job_id or self.phase != "downloading":
                return
            now = time.time()
            total = int(self.download["total_bytes"])
            downloaded = min(total, self.download_cached_bytes + max(0, int(downloaded_missing_bytes)))
            remaining = max(0, total - downloaded)
            elapsed = max(0.001, now - (self.download_transfer_started_at or now))
            transferred = max(0, downloaded - self.download_cached_bytes)
            speed = round(transferred / elapsed)
            self.download.update(
                {
                    "status": "downloading",
                    "downloaded_bytes": downloaded,
                    "remaining_bytes": remaining,
                    "percent": round((downloaded / total) * 100, 1) if total else 0.0,
                    "speed_bytes_per_second": speed,
                    "eta_seconds": round(remaining / speed) if speed else None,
                }
            )

    def _launch_process(self, request: StartRequest, job_id: int) -> None:
        package = "mlx_lm" if request.backend == "lm" else "mlx_vlm"
        module = f"{package}.server"
        command = [
            sys.executable,
            "-m",
            module,
            "--model",
            request.model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(INFERENCE_PORT),
        ]
        if request.trust_remote_code:
            command.append("--trust-remote-code")
        if request.draft_model:
            command.extend(["--draft-model", request.draft_model])
        command.extend(
            [
                "--prefill-step-size",
                str(request.prefill_step_size),
                "--max-tokens",
                str(request.default_max_tokens),
            ]
        )

        if request.backend == "lm":
            command.extend(
                [
                    "--num-draft-tokens",
                    str(request.num_draft_tokens),
                    "--decode-concurrency",
                    str(request.decode_concurrency),
                    "--prompt-concurrency",
                    str(request.prompt_concurrency),
                    "--prompt-cache-size",
                    str(request.prompt_cache_size),
                ]
            )
            if request.prompt_cache_gb:
                command.extend(
                    ["--prompt-cache-bytes", str(request.prompt_cache_gb * 1024**3)]
                )
            if request.pipeline:
                command.append("--pipeline")
        else:
            command.extend(
                [
                    "--vision-cache-size",
                    str(request.vision_cache_size),
                    "--log-progress-interval",
                    "0",
                ]
            )
            if request.max_num_seqs:
                command.extend(["--max-num-seqs", str(request.max_num_seqs)])
            if request.kv_bits != "none":
                command.extend(
                    [
                        "--kv-bits",
                        request.kv_bits,
                        "--kv-group-size",
                        str(request.kv_group_size),
                        "--quantized-kv-start",
                        str(request.quantized_kv_start),
                    ]
                )
                if request.kv_bits == "3.5":
                    command.extend(["--kv-quant-scheme", "turboquant"])
            if request.max_kv_size:
                command.extend(["--max-kv-size", str(request.max_kv_size)])
            if request.expert_cache_gb:
                command.extend(["--expert-cache-gb", str(request.expert_cache_gb)])
            if request.draft_model and request.draft_kind != "auto":
                command.extend(["--draft-kind", request.draft_kind])
            if request.draft_model and request.draft_block_size:
                command.extend(["--draft-block-size", str(request.draft_block_size)])

        with self.lock:
            if job_id != self.job_id or self.download_cancel.is_set():
                return
            self.phase = "starting"
            self.started_at = time.time()
            self._log(f"Loading {request.model_id} with {package.replace('_', '-')}…")

            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            if request.backend == "vlm":
                environment["APC_ENABLED"] = "1" if request.apc_enabled else "0"
                if request.apc_enabled:
                    environment["APC_NUM_BLOCKS"] = str(request.apc_num_blocks)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
            except OSError as error:
                self.phase = "error"
                self.last_error = str(error)
                self._log(f"Could not start inference server: {error}", "error")
                return

            self.process = process
            self.ps_process = psutil.Process(process.pid)
            self.ps_process.cpu_percent(None)
            threading.Thread(target=self._reader, args=(process,), daemon=True).start()
            threading.Thread(target=self._wait_until_ready, args=(process,), daemon=True).start()

    def _download_and_launch(self, request: StartRequest, job_id: int, cancel: threading.Event) -> None:
        try:
            token = os.getenv("HF_TOKEN")
            repositories = [request.model_id]
            if request.draft_model and request.draft_model not in repositories:
                repositories.append(request.draft_model)
            repositories = [
                repo for repo in repositories if "/" in repo and not Path(repo).expanduser().exists()
            ]

            manifests: list[tuple[str, list[Any], int]] = []
            for repository in repositories:
                with self.lock:
                    if cancel.is_set() or job_id != self.job_id:
                        raise DownloadCancelled
                    self.download["repository"] = repository
                files = snapshot_download(repository, token=token, dry_run=True)
                repository_remaining = sum(
                    max(0, int(file.file_size or 0)) for file in files if file.will_download
                )
                manifests.append((repository, files, repository_remaining))

            all_files = [file for _, files, _ in manifests for file in files]
            total_bytes = sum(max(0, int(file.file_size or 0)) for file in all_files)
            remaining_files = [file for file in all_files if file.will_download]
            remaining_bytes = sum(remaining for _, _, remaining in manifests)
            cached_bytes = max(0, total_bytes - remaining_bytes)
            with self.lock:
                if cancel.is_set() or job_id != self.job_id:
                    raise DownloadCancelled
                self.download_cached_bytes = cached_bytes
                self.download_transfer_started_at = time.time()
                self.download.update(
                    {
                        "status": "downloading" if remaining_bytes else "complete",
                        "downloaded_bytes": cached_bytes if remaining_bytes else total_bytes,
                        "total_bytes": total_bytes,
                        "remaining_bytes": remaining_bytes,
                        "percent": round((cached_bytes / total_bytes) * 100, 1) if total_bytes else 100.0,
                        "files_total": len(all_files),
                        "files_remaining": len(remaining_files),
                        "repository": repositories[0] if repositories else None,
                        "repository_index": 1 if repositories else 0,
                        "repositories_total": len(repositories),
                    }
                )
                if remaining_bytes:
                    self._log(
                        f"Downloading {len(remaining_files)} files ({remaining_bytes / 1024**3:.2f} GB remaining; "
                        f"{cached_bytes / 1024**3:.2f} GB cached)…"
                    )
                else:
                    self._log("Model files are already cached.")

            controller = self
            completed_missing_bytes = 0
            completed_missing_files = 0
            for index, (repository, files, repository_remaining) in enumerate(manifests, start=1):
                progress_offset = completed_missing_bytes

                class StudioProgress(base_tqdm):
                    def __init__(self, *args: Any, **kwargs: Any) -> None:
                        self._studio_download = str(kwargs.get("desc", "")).startswith("Reconstructing")
                        super().__init__(*args, **kwargs)

                    def display(self, msg: str | None = None, pos: int | None = None) -> bool:
                        return True

                    def refresh(self, nolock: bool = False, lock_args: tuple[Any, ...] | None = None) -> bool:
                        if getattr(self, "_studio_download", False) and cancel.is_set():
                            raise DownloadCancelled
                        return super().refresh(nolock=nolock, lock_args=lock_args)

                    def update(self, n: int | float | None = 1) -> bool | None:
                        if cancel.is_set():
                            raise DownloadCancelled
                        result = super().update(n)
                        if self._studio_download:
                            controller._record_download_progress(job_id, progress_offset + int(self.n))
                        return result

                with self.lock:
                    self.download.update(
                        {
                            "repository": repository,
                            "repository_index": index,
                            "files_remaining": max(0, len(remaining_files) - completed_missing_files),
                        }
                    )
                snapshot_download(repository, token=token, tqdm_class=StudioProgress)
                completed_missing_bytes += repository_remaining
                completed_missing_files += sum(1 for file in files if file.will_download)
                self._record_download_progress(job_id, completed_missing_bytes)

            with self.lock:
                if cancel.is_set() or job_id != self.job_id:
                    raise DownloadCancelled
                self.download.update(
                    {
                        "status": "complete",
                        "downloaded_bytes": total_bytes,
                        "remaining_bytes": 0,
                        "percent": 100.0,
                        "eta_seconds": 0,
                    }
                )
                self._log("Model download complete.")
            self._launch_process(request, job_id)
        except DownloadCancelled:
            with self.lock:
                if job_id == self.job_id:
                    self.phase = "stopped"
                    self.download["status"] = "cancelled"
                    self._log("Model download cancelled.")
        except Exception as error:
            with self.lock:
                if job_id == self.job_id:
                    self.phase = "error"
                    self.last_error = f"Model download failed: {error}"
                    self.download["status"] = "error"
                    self._log(self.last_error, "error")
        finally:
            with self.lock:
                if self.download_thread is threading.current_thread():
                    self.download_thread = None

    def start(self, request: StartRequest) -> dict[str, Any]:
        with self.lock:
            if (self.process and self.process.poll() is None) or self.phase in {"downloading", "starting", "stopping"}:
                raise HTTPException(status_code=409, detail="Stop the current model before starting another one.")

            package = "mlx_lm" if request.backend == "lm" else "mlx_vlm"
            if importlib.util.find_spec(package) is None:
                raise HTTPException(
                    status_code=424,
                    detail=f"{package.replace('_', '-')} is not installed. Run ./setup.sh first.",
                )

            self.model_id = request.model_id
            self.backend = request.backend
            self.started_at = None
            self.last_error = None
            self.request_count = 0
            self.error_count = 0
            self.active_requests = 0
            self.input_tokens = 0
            self.output_tokens = 0
            self.total_latency_ms = 0
            self.last_ttft_ms = 0
            self.total_ttft_ms = 0
            self.ttft_samples = 0
            self.last_decode_tps = 0
            self.total_decode_tps = 0
            self.decode_tps_samples = 0
            self.last_prefill_tps = 0
            self.total_prefill_tps = 0
            self.prefill_tps_samples = 0
            self.cache_hit_tokens = 0
            self.last_draft_kind = None
            self.last_draft_rounds = 0
            self.last_draft_tokens = 0
            self.last_draft_accepted_tokens = 0
            self.draft_rounds = 0
            self.draft_tokens = 0
            self.draft_accepted_tokens = 0
            self.performance = request.model_dump(exclude={"model_id", "backend", "trust_remote_code"})
            self.history.clear()
            self.logs.clear()
            self.job_id += 1
            job_id = self.job_id
            cancel = threading.Event()
            self.download_cancel = cancel
            self.download_transfer_started_at = None
            self.download_cached_bytes = 0
            self.download = self._empty_download()

            requested_repositories = [request.model_id, request.draft_model]
            is_remote_model = any(
                value and "/" in value and not Path(value).expanduser().exists()
                for value in requested_repositories
            )
            if is_remote_model:
                self.phase = "downloading"
                self.download.update({"status": "checking", "started_at": time.time()})
                self._log(f"Checking the Hugging Face cache for {request.model_id}…")
                thread = threading.Thread(
                    target=self._download_and_launch,
                    args=(request, job_id, cancel),
                    daemon=True,
                )
                self.download_thread = thread
                thread.start()
            else:
                self.phase = "starting"

        if not is_remote_model:
            self._launch_process(request, job_id)

        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            process = self.process
            if self.phase == "downloading" or (self.download_thread and self.download_thread.is_alive()):
                self.phase = "stopping"
                self.download_cancel.set()
                self.job_id += 1
                self.download["status"] = "cancelled"
                self._log("Cancelling model download…")
                self.phase = "stopped"
                return self.snapshot()
            if not process or process.poll() is not None:
                self.phase = "stopped"
                self.process = None
                self.ps_process = None
                return self.snapshot()
            self.phase = "stopping"
            self._log("Stopping inference server…")

        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
        except ProcessLookupError:
            pass

        with self.lock:
            if self.process is process:
                self.process = None
                self.ps_process = None
                self.phase = "stopped"
        return self.snapshot()

    def begin_request(self) -> None:
        with self.lock:
            self.active_requests += 1

    def add_stream_output_tokens(self, count: int = 1) -> None:
        with self.lock:
            self.output_tokens += max(0, count)

    def finish_request(
        self,
        started: float,
        status_code: int,
        usage: dict[str, Any] | None = None,
        provisional_output_tokens: int = 0,
        first_token_at: float | None = None,
        last_token_at: float | None = None,
        timings: dict[str, Any] | None = None,
    ) -> None:
        with self.lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.request_count += 1
            self.total_latency_ms += (time.perf_counter() - started) * 1000
            if status_code >= 400:
                self.error_count += 1
            prompt_tokens = 0
            reported_output = 0
            if usage:
                prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                self.input_tokens += prompt_tokens
                reported_output = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
                self.output_tokens += max(0, reported_output - provisional_output_tokens)
            if first_token_at is not None:
                self.last_ttft_ms = max(0, (first_token_at - started) * 1000)
                self.total_ttft_ms += self.last_ttft_ms
                self.ttft_samples += 1

            native_decode_tps = float((timings or {}).get("predicted_per_second") or 0)
            output_count = reported_output or provisional_output_tokens
            measured_decode_tps = 0.0
            if output_count > 1 and first_token_at is not None and last_token_at and last_token_at > first_token_at:
                measured_decode_tps = (output_count - 1) / (last_token_at - first_token_at)
            decode_tps = native_decode_tps or measured_decode_tps
            if decode_tps > 0:
                self.last_decode_tps = decode_tps
                self.total_decode_tps += decode_tps
                self.decode_tps_samples += 1

            prefill_tps = float((timings or {}).get("prompt_per_second") or 0)
            if prefill_tps > 0:
                self.last_prefill_tps = prefill_tps
                self.total_prefill_tps += prefill_tps
                self.prefill_tps_samples += 1
            self.cache_hit_tokens += int((timings or {}).get("cache_n") or 0)

            draft_tokens = max(0, int((timings or {}).get("draft_n") or 0))
            if draft_tokens:
                draft_accepted_tokens = min(
                    draft_tokens,
                    max(0, int((timings or {}).get("draft_n_accepted") or 0)),
                )
                self.last_draft_kind = str((timings or {}).get("draft_kind") or "speculative")
                self.last_draft_rounds = max(0, int((timings or {}).get("draft_rounds") or 0))
                self.last_draft_tokens = draft_tokens
                self.last_draft_accepted_tokens = draft_accepted_tokens
                self.draft_rounds += self.last_draft_rounds
                self.draft_tokens += draft_tokens
                self.draft_accepted_tokens += draft_accepted_tokens

    def snapshot(self) -> dict[str, Any]:
        cpu = 0.0
        rss = 0
        process = self.ps_process
        if process:
            try:
                cpu = process.cpu_percent(None)
                rss = process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        cpu += child.cpu_percent(None)
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        memory = psutil.virtual_memory()
        now = time.time()
        point = {
            "time": now,
            "cpu_percent": round(cpu, 1),
            "rss_bytes": rss,
            "memory_percent": round((rss / memory.total) * 100, 2) if memory.total else 0,
        }
        with self.lock:
            if not self.history or now - self.history[-1]["time"] >= 0.8:
                self.history.append(point)
            average_latency = self.total_latency_ms / self.request_count if self.request_count else 0
            last_acceptance_rate = (
                self.last_draft_accepted_tokens / self.last_draft_tokens * 100
                if self.last_draft_tokens
                else 0
            )
            average_acceptance_rate = (
                self.draft_accepted_tokens / self.draft_tokens * 100
                if self.draft_tokens
                else 0
            )
            return {
                "phase": self.phase,
                "model_id": self.model_id,
                "backend": self.backend,
                "pid": process.pid if process else None,
                "started_at": self.started_at,
                "uptime_seconds": round(now - self.started_at) if self.started_at and self.phase != "stopped" else 0,
                "last_error": self.last_error,
                "endpoint": f"http://{APP_HOST}:{APP_PORT}/v1",
                "download": dict(self.download),
                "performance": dict(self.performance),
                "inference": point,
                "system": {
                    "memory_total_bytes": memory.total,
                    "memory_used_bytes": memory.used,
                    "memory_percent": memory.percent,
                },
                "requests": {
                    "total": self.request_count,
                    "errors": self.error_count,
                    "active": self.active_requests,
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "average_latency_ms": round(average_latency, 1),
                    "last_ttft_ms": round(self.last_ttft_ms, 1),
                    "average_ttft_ms": round(self.total_ttft_ms / self.ttft_samples, 1) if self.ttft_samples else 0,
                    "last_decode_tps": round(self.last_decode_tps, 1),
                    "average_decode_tps": round(self.total_decode_tps / self.decode_tps_samples, 1) if self.decode_tps_samples else 0,
                    "last_prefill_tps": round(self.last_prefill_tps, 1),
                    "average_prefill_tps": round(self.total_prefill_tps / self.prefill_tps_samples, 1) if self.prefill_tps_samples else 0,
                    "cache_hit_tokens": self.cache_hit_tokens,
                    "draft_kind": self.last_draft_kind,
                    "last_draft_rounds": self.last_draft_rounds,
                    "last_draft_tokens": self.last_draft_tokens,
                    "last_draft_accepted_tokens": self.last_draft_accepted_tokens,
                    "draft_rounds": self.draft_rounds,
                    "draft_tokens": self.draft_tokens,
                    "draft_accepted_tokens": self.draft_accepted_tokens,
                    "last_speculative_acceptance_rate": round(last_acceptance_rate, 1),
                    "average_speculative_acceptance_rate": round(average_acceptance_rate, 1),
                },
                "history": list(self.history),
            }


controller = RuntimeController()
hf_api = HfApi(token=os.getenv("HF_TOKEN"))


def _infer_backend(model_id: str, pipeline_tag: str | None, tags: list[str]) -> str | None:
    combined = " ".join([model_id, pipeline_tag or "", *tags]).lower()
    unsupported_markers = (
        "mlx-audio",
        "speech-to-text",
        "text-to-speech",
        "audio-classification",
        "image-generation",
        "text-to-image",
        "feature-extraction",
        "sentence-similarity",
        "reranking",
    )
    if any(marker in combined for marker in unsupported_markers):
        return None
    vision_markers = ("mlx-vlm", "image-text-to-text", "visual-question-answering", "vision", "-vl-", "vlm")
    return "vlm" if any(marker in combined for marker in vision_markers) else "lm"


def _is_mlx_model(model_id: str, tags: list[str]) -> bool:
    combined = " ".join([model_id, *tags]).lower()
    return model_id.lower().startswith("mlx-community/") or "mlx" in combined


def _parameter_count(model: Any) -> int | None:
    safetensors = getattr(model, "safetensors", None)
    parameters = getattr(safetensors, "parameters", None) if safetensors else None
    if isinstance(parameters, dict):
        return int(sum(value for value in parameters.values() if isinstance(value, int)))
    total = getattr(safetensors, "total", None) if safetensors else None
    return int(total) if isinstance(total, int) else None


def _quantization(model_id: str, tags: list[str]) -> str | None:
    text = " ".join([model_id, *tags])
    match = re.search(r"(?<!\d)([2-8](?:\.\d)?)\s*[-_ ]?bit", text, re.IGNORECASE)
    return f"{match.group(1)}-bit" if match else None


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    timeout = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=30.0)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as proxy_client:
        app_instance.state.proxy_client = proxy_client
        try:
            yield
        finally:
            controller.stop()


app = FastAPI(
    title="MLX Studio",
    description="Local controller and OpenAI-compatible proxy for mlx-lm and mlx-vlm.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        f"http://localhost:{APP_PORT}",
        f"http://127.0.0.1:{APP_PORT}",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "runtime": controller.phase}


@app.get("/api/system")
def system_info() -> dict[str, Any]:
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        chip = platform.processor() or platform.machine()
    return {
        "chip": chip,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx_lm_installed": importlib.util.find_spec("mlx_lm") is not None,
        "mlx_vlm_installed": importlib.util.find_spec("mlx_vlm") is not None,
    }


@app.get("/api/status")
def status() -> dict[str, Any]:
    return controller.snapshot()


@app.post("/api/runtime/start")
def start_runtime(request: StartRequest) -> dict[str, Any]:
    return controller.start(request)


@app.post("/api/runtime/stop")
def stop_runtime() -> dict[str, Any]:
    return controller.stop()


@app.get("/api/logs")
def logs(after: float = 0) -> dict[str, Any]:
    with controller.lock:
        entries = [entry for entry in controller.logs if entry["time"] > after]
    return {"data": entries}


@app.get("/api/models/search")
def search_models(q: str = "", backend: Literal["all", "lm", "vlm"] = "all", limit: int = 30) -> dict[str, Any]:
    limit = max(1, min(limit, 60))
    try:
        query = q.strip() or None
        primary = hf_api.list_models(
            search=query,
            author="mlx-community",
            sort="downloads",
            limit=180,
            full=True,
            cardData=True,
        )
        candidates = list(primary)
        if query:
            candidates.extend(
                hf_api.list_models(
                    search=query,
                    filter="mlx",
                    sort="downloads",
                    limit=120,
                    full=True,
                    cardData=True,
                )
            )
        results = []
        seen: set[str] = set()
        for model in candidates:
            if model.id in seen:
                continue
            seen.add(model.id)
            tags = list(model.tags or [])
            if not _is_mlx_model(model.id, tags):
                continue
            inferred = _infer_backend(model.id, model.pipeline_tag, tags)
            if inferred is None:
                continue
            if backend != "all" and inferred != backend:
                continue
            cache_name = f"models--{model.id.replace('/', '--')}"
            cached = (Path(HF_HUB_CACHE) / cache_name / "snapshots").exists()
            results.append(
                {
                    "id": model.id,
                    "backend": inferred,
                    "pipeline_tag": model.pipeline_tag,
                    "downloads": model.downloads or 0,
                    "likes": model.likes or 0,
                    "last_modified": model.last_modified,
                    "tags": tags[:24],
                    "parameters": _parameter_count(model),
                    "quantization": _quantization(model.id, tags),
                    "cached": cached,
                    "url": f"https://huggingface.co/{model.id}",
                }
            )
            if len(results) >= limit:
                break
        return {"data": results, "query": q, "backend": backend}
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Hugging Face search failed: {error}") from error


@app.get("/api/opencode-config")
def opencode_config() -> dict[str, Any]:
    model_id = controller.model_id or "mlx-community/Qwen3-4B-4bit"
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "mlx-studio": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "MLX Studio (local)",
                "options": {"baseURL": f"http://127.0.0.1:{APP_PORT}/v1", "apiKey": "local"},
                "models": {model_id: {"name": model_id.split("/")[-1]}},
            }
        },
    }


@app.get("/v1/models")
def openai_models() -> dict[str, Any]:
    data = []
    if controller.model_id:
        data.append(
            {
                "id": controller.model_id,
                "object": "model",
                "created": int(controller.started_at or time.time()),
                "owned_by": "local",
            }
        )
    return {"object": "list", "data": data}


def _usage_from_value(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    usage = value.get("usage")
    if isinstance(usage, dict):
        return usage
    response = value.get("response")
    if isinstance(response, dict):
        return _usage_from_value(response)
    return None


def _usage_from_payload(payload: bytes) -> dict[str, Any] | None:
    try:
        return _usage_from_value(json.loads(payload))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _timings_from_value(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        timings = value.get("timings")
        if isinstance(timings, dict):
            return timings
        for nested in value.values():
            found = _timings_from_value(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _timings_from_value(nested)
            if found:
                return found
    return None


def _timings_from_payload(payload: bytes) -> dict[str, Any] | None:
    try:
        return _timings_from_value(json.loads(payload))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _stream_token_increment(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    if value.get("type") in {"response.output_text.delta", "response.reasoning_text.delta"}:
        return 1 if value.get("delta") else 0
    choices = value.get("choices")
    if not isinstance(choices, list):
        return 0
    count = 0
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        if isinstance(delta, dict) and any(
            delta.get(key) for key in ("content", "reasoning", "reasoning_content", "tool_calls")
        ):
            count += 1
        elif choice.get("text"):
            count += 1
    return count


def _ensure_stream_usage(path: str, payload: bytes) -> bytes:
    if path != "chat/completions" or not payload:
        return payload
    if b'"stream_options"' in payload:
        return payload
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return payload
    if not isinstance(value, dict) or value.get("stream") is not True:
        return payload
    options = value.get("stream_options")
    if not isinstance(options, dict):
        options = {}
        value["stream_options"] = options
    options["include_usage"] = True
    return json.dumps(value).encode()


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_openai(path: str, request: Request):
    if controller.phase != "ready":
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "No model is ready. Start one in MLX Studio first.",
                    "type": "server_unavailable",
                    "code": "model_not_ready",
                }
            },
        )

    body = _ensure_stream_usage(path, await request.body())
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length", "connection", "authorization"}
    }
    target = f"{INFERENCE_URL}/v1/{path}"
    started = time.perf_counter()
    controller.begin_request()

    client: httpx.AsyncClient = request.app.state.proxy_client
    try:
        upstream_request = client.build_request(request.method, target, content=body, headers=headers)
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as error:
        controller.finish_request(started, 502)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(error), "type": "upstream_error", "code": "mlx_upstream_error"}},
        )

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in {"content-length", "connection", "content-encoding", "transfer-encoding"}
    }
    content_type = upstream.headers.get("content-type", "application/json")

    async def stream_body():
        usage: dict[str, Any] | None = None
        timings: dict[str, Any] | None = None
        buffer = b""
        provisional_output_tokens = 0
        first_token_at: float | None = None
        last_token_at: float | None = None
        last_value: Any = None
        try:
            async for chunk in upstream.aiter_raw():
                buffer += chunk
                lines = buffer.split(b"\n")
                buffer = lines.pop()
                for line in lines:
                    if line.startswith(b"data:"):
                        candidate = line[5:].strip()
                        if candidate == b"[DONE]":
                            usage = _usage_from_value(last_value)
                            timings = _timings_from_value(last_value)
                        elif candidate:
                            try:
                                value = json.loads(candidate)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                value = None
                            last_value = value
                            increment = _stream_token_increment(value)
                            if increment:
                                token_at = time.perf_counter()
                                first_token_at = first_token_at or token_at
                                last_token_at = token_at
                                provisional_output_tokens += increment
                                controller.add_stream_output_tokens(increment)
                yield chunk
        finally:
            await upstream.aclose()
            if last_value is not None:
                usage = usage or _usage_from_value(last_value)
                timings = timings or _timings_from_value(last_value)
            controller.finish_request(
                started,
                upstream.status_code,
                usage,
                provisional_output_tokens=provisional_output_tokens,
                first_token_at=first_token_at,
                last_token_at=last_token_at,
                timings=timings,
            )

    if "text/event-stream" in content_type:
        return StreamingResponse(
            stream_body(),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type="text/event-stream",
        )

    payload = await upstream.aread()
    await upstream.aclose()
    controller.finish_request(
        started,
        upstream.status_code,
        _usage_from_payload(payload),
        timings=_timings_from_payload(payload),
    )
    return JSONResponse(
        status_code=upstream.status_code,
        content=json.loads(payload) if payload else {},
        headers=response_headers,
    )


static_directory = PROJECT_ROOT / "dist" / "client"
if static_directory.exists():
    app.mount("/", StaticFiles(directory=static_directory, html=True), name="ui")


def main() -> None:
    import uvicorn

    uvicorn.run("mlx_studio.app:app", host=APP_HOST, port=APP_PORT, reload=False)


if __name__ == "__main__":
    main()
