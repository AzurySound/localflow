"""Managed background llama-server.exe process with health checks and idle VRAM eviction."""

from __future__ import annotations

import logging
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import httpx

logger = logging.getLogger("localflow.llm")

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*$", re.IGNORECASE)


def parse_keep_alive_seconds(keep_alive: str | None) -> float:
    """Parse Ollama-style duration string ('0', '60s', '2m', '10m', '1h') into seconds."""
    if not keep_alive:
        return 120.0
    m = _DURATION_RE.match(str(keep_alive))
    if not m:
        return 120.0
    val = float(m.group(1))
    unit = (m.group(2) or "s").lower()
    if unit == "m":
        return val * 60.0
    if unit == "h":
        return val * 3600.0
    return val


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LlamaServerManager:
    """Starts, health-checks, idle-unloads, and stops a local llama-server subprocess."""

    def __init__(
        self,
        binary_path: Path,
        model_path: Path,
        *,
        num_ctx: int = 2048,
        num_gpu: int | None = None,
        keep_alive: str = "5m",
        startup_timeout_sec: float = 90.0,
        popen_fn: Any = None,
        health_client: httpx.Client | None = None,
    ) -> None:
        self.binary_path = Path(binary_path)
        self.model_path = Path(model_path)
        self.num_ctx = max(512, int(num_ctx))
        self.num_gpu = num_gpu
        self.idle_timeout_sec = parse_keep_alive_seconds(keep_alive)
        self.startup_timeout_sec = startup_timeout_sec

        self._popen_fn = popen_fn or subprocess.Popen
        self._owns_health_client = health_client is None
        self._health_client = health_client or httpx.Client(timeout=2.0, trust_env=False)

        self._lock = threading.RLock()
        self._proc: Any = None
        self._port: int | None = None
        self._idle_timer: threading.Timer | None = None
        self._closed = False

    @property
    def base_url(self) -> str | None:
        with self._lock:
            if self._port is None:
                return None
            return f"http://127.0.0.1:{self._port}"

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def build_command(self, port: int, *, include_reasoning_flag: bool = True) -> list[str]:
        ngl = "0" if self.num_gpu == 0 else "99"
        cmd = [
            str(self.binary_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-m",
            str(self.model_path),
            "-c",
            str(self.num_ctx),
            "-ngl",
            ngl,
        ]
        if include_reasoning_flag:
            cmd.extend(["--reasoning-budget", "0"])
        return cmd

    def _cancel_idle_timer_locked(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def touch_activity(self) -> None:
        """Reset the idle-eviction timer after a completed LLM request."""
        with self._lock:
            self._cancel_idle_timer_locked()
            if self._closed or self._proc is None:
                return
            if self.idle_timeout_sec <= 0:
                self._stop_locked()
                return
            timer = threading.Timer(self.idle_timeout_sec, self.stop)
            timer.daemon = True
            self._idle_timer = timer
            timer.start()

    def ensure_started(self) -> str:
        """Start llama-server if not running, wait until /health is ready, and return base URL."""
        with self._lock:
            if self._closed:
                raise RuntimeError("LlamaServerManager is closed")
            if self._proc is not None and self._proc.poll() is None and self._port is not None:
                self._cancel_idle_timer_locked()
                return f"http://127.0.0.1:{self._port}"

            if not self.binary_path.is_file():
                raise FileNotFoundError(f"llama-server не найден: {self.binary_path}")
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Файл модели GGUF не найден: {self.model_path}")

            last_err = ""
            for include_reasoning in (True, False):
                port = _pick_free_port()
                cmd = self.build_command(port, include_reasoning_flag=include_reasoning)
                kwargs: dict[str, Any] = {
                    "cwd": str(self.binary_path.parent),
                    "stdin": subprocess.DEVNULL,
                    "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.PIPE,
                }
                if sys.platform == "win32":
                    kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

                logger.info("Starting embedded llama-server on 127.0.0.1:%d (%s)", port, self.model_path.name)
                proc = self._popen_fn(cmd, **kwargs)
                self._proc = proc
                self._port = port

                ok, err_detail = self._wait_for_health_locked(proc, port)
                if ok:
                    return f"http://127.0.0.1:{port}"

                last_err = err_detail
                self._stop_locked()
                # Only retry without --reasoning-budget if process exited immediately with an argument error
                if not include_reasoning or "reasoning" not in err_detail.lower():
                    break

            raise RuntimeError(f"Не удалось запустить встроенный llama-server: {last_err}")

    def _wait_for_health_locked(self, proc: Any, port: int) -> tuple[bool, str]:
        deadline = time.monotonic() + self.startup_timeout_sec
        url = f"http://127.0.0.1:{port}/health"
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr_txt = ""
                try:
                    if proc.stderr is not None:
                        stderr_txt = proc.stderr.read().decode("utf-8", errors="replace")[-400:].strip()
                except Exception:
                    pass
                return False, f"process exited with code {proc.returncode}: {stderr_txt}"
            try:
                resp = self._health_client.get(url)
                if resp.status_code == 200:
                    return True, ""
            except Exception:
                pass
            time.sleep(0.15)
        return False, f"startup timeout ({self.startup_timeout_sec:.0f}s)"

    def _stop_locked(self) -> None:
        self._cancel_idle_timer_locked()
        proc = self._proc
        self._proc = None
        self._port = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    proc.kill()
                    try:
                        proc.wait(timeout=1.0)
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            try:
                if getattr(proc, "stderr", None) is not None:
                    proc.stderr.close()
            except Exception:
                pass

    def stop(self) -> None:
        """Stop the background llama-server process and release VRAM/RAM."""
        with self._lock:
            self._stop_locked()

    def close(self) -> None:
        """Permanently stop server and close HTTP resources."""
        with self._lock:
            self._closed = True
            self._stop_locked()
        if self._owns_health_client:
            try:
                self._health_client.close()
            except Exception:
                pass
