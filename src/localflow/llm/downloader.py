"""Resumable HTTP downloader for llama.cpp runtime and GGUF LLM models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import zipfile

import httpx

from localflow.i18n import tr
from localflow.llm.model_catalog import (
    EmbeddedModelSpec,
    find_llama_server_binary,
    get_llama_runtime_dir,
    get_llm_models_dir,
)

LLAMA_CPP_LATEST_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
LLAMA_CPP_RELEASES_LIST_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8"


class DownloadCancelledError(RuntimeError):
    """Raised when the user cancels an active download."""


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    stage: str
    downloaded_bytes: int
    total_bytes: int
    speed_mb_s: float = 0.0

    @property
    def percent(self) -> int:
        if self.total_bytes <= 0:
            return 0
        return max(0, min(100, int(round(self.downloaded_bytes * 100.0 / self.total_bytes))))


ProgressCallback = Callable[[DownloadProgress], None]


def download_file(
    url: str,
    dest_path: Path,
    *,
    stage: str = "Downloading",
    on_progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    client: httpx.Client | None = None,
    chunk_size: int = 262_144,
) -> Path:
    """Download a file with .part resume support, progress reporting, and cancellation."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_name(dest_path.name + ".part")

    existing = part_path.stat().st_size if part_path.is_file() else 0
    headers = {"User-Agent": "LocalFlow/0.2.0"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    owns_client = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(connect=20.0, read=60.0, write=30.0, pool=20.0),
        follow_redirects=True,
        trust_env=True,
    )

    try:
        with http.stream("GET", url, headers=headers) as resp:
            if resp.status_code == 416 and existing > 0:
                # Stale partial file larger than remote: restart from byte 0
                part_path.unlink(missing_ok=True)
                existing = 0
                headers.pop("Range", None)
                return download_file(
                    url,
                    dest_path,
                    stage=stage,
                    on_progress=on_progress,
                    cancel_event=cancel_event,
                    client=http,
                    chunk_size=chunk_size,
                )

            resp.raise_for_status()

            is_resumed = resp.status_code == 206 and existing > 0
            if not is_resumed:
                existing = 0

            content_len_hdr = resp.headers.get("content-length")
            remaining_len = int(content_len_hdr) if content_len_hdr and content_len_hdr.isdigit() else 0
            total_bytes = (existing + remaining_len) if remaining_len > 0 else 0

            mode = "ab" if is_resumed else "wb"
            downloaded = existing
            started_at = time.monotonic()
            session_bytes = 0
            last_emit = 0.0

            if on_progress is not None:
                on_progress(DownloadProgress(stage=stage, downloaded_bytes=downloaded, total_bytes=total_bytes))

            with part_path.open(mode) as fh:
                for chunk in resp.iter_bytes(chunk_size=chunk_size):
                    if cancel_event is not None and cancel_event.is_set():
                        raise DownloadCancelledError("Загрузка отменена пользователем")
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    session_bytes += len(chunk)

                    now = time.monotonic()
                    if on_progress is not None and (now - last_emit >= 0.12 or (total_bytes and downloaded >= total_bytes)):
                        elapsed = max(0.001, now - started_at)
                        speed = (session_bytes / (1024 * 1024)) / elapsed
                        on_progress(
                            DownloadProgress(
                                stage=stage,
                                downloaded_bytes=downloaded,
                                total_bytes=total_bytes,
                                speed_mb_s=speed,
                            )
                        )
                        last_emit = now

            if cancel_event is not None and cancel_event.is_set():
                raise DownloadCancelledError("Загрузка отменена пользователем")

        part_path.replace(dest_path)
        if on_progress is not None:
            final_size = dest_path.stat().st_size
            on_progress(
                DownloadProgress(
                    stage=stage,
                    downloaded_bytes=final_size,
                    total_bytes=final_size,
                    speed_mb_s=0.0,
                )
            )
        return dest_path
    finally:
        if owns_client:
            http.close()


def _select_llama_windows_asset(assets: list[dict[str, object]]) -> tuple[str, str] | None:
    """Pick the best official Windows x64 llama.cpp asset (preferring Vulkan, fallback to CPU)."""
    preferred_patterns = (
        ("win-vulkan-x64.zip",),
        ("win-cpu-x64.zip",),
        ("win-avx2-x64.zip",),
    )
    for patterns in preferred_patterns:
        for asset in assets:
            name = str(asset.get("name") or "")
            url = str(asset.get("browser_download_url") or "")
            low = name.lower()
            if url and all(p in low for p in patterns):
                return name, url
    return None


def download_llama_runtime(
    *,
    base_dir: Path | None = None,
    on_progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    client: httpx.Client | None = None,
    lang: str | None = None,
) -> Path:
    """Download and extract llama-server.exe (+ DLLs) from official ggml-org/llama.cpp releases."""
    existing = find_llama_server_binary(base_dir)
    if existing is not None:
        return existing

    runtime_dir = get_llama_runtime_dir(base_dir)
    owns_client = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(connect=20.0, read=60.0, write=30.0, pool=20.0),
        follow_redirects=True,
        trust_env=True,
    )

    try:
        if on_progress is not None:
            on_progress(DownloadProgress(stage=tr("dl.stage_search_llama", lang), downloaded_bytes=0, total_bytes=0))

        gh_headers = {"User-Agent": "LocalFlow/0.2.0", "Accept": "application/vnd.github+json"}
        picked: tuple[str, str] | None = None

        rel_resp = http.get(LLAMA_CPP_LATEST_RELEASE_API, headers=gh_headers)
        rel_resp.raise_for_status()
        release_data = rel_resp.json()
        assets = release_data.get("assets") if isinstance(release_data, dict) else None
        if isinstance(assets, list):
            picked = _select_llama_windows_asset(assets)

        # Fallback: latest release may still be uploading CI binaries; check recent releases
        if picked is None:
            list_resp = http.get(LLAMA_CPP_RELEASES_LIST_API, headers=gh_headers)
            list_resp.raise_for_status()
            releases = list_resp.json()
            if isinstance(releases, list):
                for rel in releases:
                    if not isinstance(rel, dict) or rel.get("draft"):
                        continue
                    rel_assets = rel.get("assets")
                    if isinstance(rel_assets, list):
                        picked = _select_llama_windows_asset(rel_assets)
                        if picked is not None:
                            break

        if picked is None:
            raise RuntimeError("Windows x64 llama.cpp release archive (Vulkan/CPU) not found")

        asset_name, asset_url = picked
        archive_path = runtime_dir / asset_name
        try:
            download_file(
                asset_url,
                archive_path,
                stage=tr("dl.stage_llama", lang),
                on_progress=on_progress,
                cancel_event=cancel_event,
                client=http,
            )

            if on_progress is not None:
                on_progress(DownloadProgress(stage=tr("dl.stage_extract_llama", lang), downloaded_bytes=0, total_bytes=0))

            extracted_server: Path | None = None
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    fname = Path(member.filename).name
                    low = fname.lower()
                    if low == "llama-server.exe" or low.endswith(".dll"):
                        target = runtime_dir / fname
                        with zf.open(member, "r") as src, target.open("wb") as dst:
                            while True:
                                buf = src.read(262_144)
                                if not buf:
                                    break
                                dst.write(buf)
                        if low == "llama-server.exe":
                            extracted_server = target
            if extracted_server is None or not extracted_server.is_file():
                raise RuntimeError("llama-server.exe was not found in the llama.cpp archive")
            return extracted_server
        finally:
            archive_path.unlink(missing_ok=True)
    finally:
        if owns_client:
            http.close()


def download_embedded_llm_model(
    spec: EmbeddedModelSpec,
    *,
    base_dir: Path | None = None,
    ensure_runtime: bool = True,
    on_progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    client: httpx.Client | None = None,
    lang: str | None = None,
) -> Path:
    """Ensure llama-server runtime is installed, then download the selected GGUF model."""
    if ensure_runtime and find_llama_server_binary(base_dir) is None:
        download_llama_runtime(
            base_dir=base_dir,
            on_progress=on_progress,
            cancel_event=cancel_event,
            client=client,
            lang=lang,
        )

    dest = get_llm_models_dir(base_dir) / spec.filename
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest

    return download_file(
        spec.url,
        dest,
        stage=tr("dl.stage_model", lang, name=spec.name),
        on_progress=on_progress,
        cancel_event=cancel_event,
        client=client,
    )
