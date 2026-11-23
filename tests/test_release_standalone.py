from __future__ import annotations

import io
import json
from pathlib import Path
import threading
import zipfile

import httpx
import pytest

from localflow.autostart import build_autostart_command
from localflow.config import AppConfig, create_first_run_config, load_config, save_config
from localflow.injector.window_context import classify_window
from localflow.llm.downloader import (
    DownloadCancelledError,
    DownloadProgress,
    download_file,
    download_llama_runtime,
)
from localflow.llm.embedded_polisher import EmbeddedLlamaPolisher
from localflow.llm.llama_server import LlamaServerManager, parse_keep_alive_seconds
from localflow.llm.model_catalog import (
    DEFAULT_EMBEDDED_MODEL_ID,
    EMBEDDED_LLM_MODELS,
    RECOMMENDED_EMBEDDED_MODEL_ID,
    delete_model_file,
    get_llm_models_dir,
    get_model_spec,
    is_model_downloaded,
)
from localflow.single_instance import SingleInstanceGuard
from localflow.storage import HistoryStorage
from localflow.styles import STYLE_PROFILES, resolve_style_profile
from localflow.ui.settings_window import SettingsWindow


def test_first_run_config_uses_lightest_standalone_defaults(tmp_path: Path, monkeypatch) -> None:
    cfg_gpu = create_first_run_config(has_cuda=True)
    assert cfg_gpu.llm_backend == "embedded"
    assert cfg_gpu.embedded_llm_model_id == DEFAULT_EMBEDDED_MODEL_ID == "qwen25-1.5b"
    assert cfg_gpu.whisper_model == "small"
    assert cfg_gpu.performance_profile == "light_gpu"
    assert cfg_gpu.first_run_completed is False

    cfg_cpu = create_first_run_config(has_cuda=False)
    assert cfg_cpu.llm_backend == "embedded"
    assert cfg_cpu.whisper_model == "small"
    assert cfg_cpu.whisper_device == "cpu"
    assert cfg_cpu.ollama_num_gpu == 0
    assert cfg_cpu.performance_profile == "zero_vram"

    # Missing config.json triggers first-run standalone config
    monkeypatch.setattr("localflow.config._detect_cuda_available", lambda: False)
    fresh_path = tmp_path / "fresh" / "config.json"
    loaded = load_config(fresh_path)
    assert fresh_path.is_file()
    assert loaded.llm_backend == "embedded"
    assert loaded.whisper_model == "small"

    # Existing legacy config.json without llm_backend stays on ollama
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps({"ollama_model": "gemma4:e4b"}), encoding="utf-8")
    legacy = load_config(legacy_path)
    assert legacy.llm_backend == "ollama"
    assert legacy.first_run_completed is True


def test_model_catalog_status_and_delete(tmp_path: Path) -> None:
    spec = get_model_spec("qwen25-1.5b")
    assert spec is not None
    assert len(EMBEDDED_LLM_MODELS) >= 3
    assert is_model_downloaded("qwen25-1.5b", base_dir=tmp_path) is False

    target = get_llm_models_dir(tmp_path) / spec.filename
    target.write_bytes(b"\x00" * 1_100_000)
    assert is_model_downloaded("qwen25-1.5b", base_dir=tmp_path) is True
    assert delete_model_file("qwen25-1.5b", base_dir=tmp_path) is True
    assert is_model_downloaded("qwen25-1.5b", base_dir=tmp_path) is False


def test_download_file_resume_progress_and_cancel(tmp_path: Path) -> None:
    payload = b"0123456789" * 20  # 200 bytes
    dest = tmp_path / "model.gguf"
    part = tmp_path / "model.gguf.part"
    part.write_bytes(payload[:50])

    seen_ranges: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("range")
        seen_ranges.append(rng)
        if rng == "bytes=50-":
            return httpx.Response(
                206,
                content=payload[50:],
                headers={"content-length": str(len(payload) - 50)},
            )
        return httpx.Response(200, content=payload, headers={"content-length": str(len(payload))})

    progress_events: list[DownloadProgress] = []
    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = download_file(
        "https://example.test/model.gguf",
        dest,
        stage="Test",
        on_progress=progress_events.append,
        client=client,
        chunk_size=32,
    )
    assert out == dest
    assert dest.read_bytes() == payload
    assert not part.exists()
    assert seen_ranges == ["bytes=50-"]
    assert progress_events[-1].percent == 100

    # Cancel test
    cancel_ev = threading.Event()
    cancel_ev.set()
    dest2 = tmp_path / "cancelled.gguf"
    with pytest.raises(DownloadCancelledError):
        download_file(
            "https://example.test/model.gguf",
            dest2,
            cancel_event=cancel_ev,
            client=client,
            chunk_size=16,
        )
    assert not dest2.exists()


def test_download_llama_runtime_extracts_vulkan_server_and_dlls(tmp_path: Path) -> None:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("llama-b9999/llama-server.exe", b"MZ-FAKE-SERVER")
        zf.writestr("llama-b9999/ggml-vulkan.dll", b"MZ-FAKE-DLL")
        zf.writestr("llama-b9999/README.txt", b"ignore me")
    zip_bytes = zip_buf.getvalue()

    def handler(request: httpx.Request) -> httpx.Response:
        if "api.github.com" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "assets": [
                        {
                            "name": "llama-b9999-bin-win-cpu-x64.zip",
                            "browser_download_url": "https://example.test/cpu.zip",
                        },
                        {
                            "name": "llama-b9999-bin-win-vulkan-x64.zip",
                            "browser_download_url": "https://example.test/vulkan.zip",
                        },
                    ]
                },
            )
        if str(request.url).endswith("vulkan.zip"):
            return httpx.Response(
                200,
                content=zip_bytes,
                headers={"content-length": str(len(zip_bytes))},
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    exe = download_llama_runtime(base_dir=tmp_path, client=client)
    assert exe.name == "llama-server.exe"
    assert exe.read_bytes() == b"MZ-FAKE-SERVER"
    assert (exe.parent / "ggml-vulkan.dll").read_bytes() == b"MZ-FAKE-DLL"
    assert not (exe.parent / "README.txt").exists()


def test_download_llama_runtime_falls_back_when_latest_has_no_windows_assets(tmp_path: Path) -> None:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("llama-b8888/llama-server.exe", b"MZ-FALLBACK-SERVER")
    zip_bytes = zip_buf.getvalue()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/releases/latest"):
            # Simulate a newly tagged release whose Windows CI assets are not uploaded yet
            return httpx.Response(200, json={"tag_name": "b9999", "assets": []})
        if "/releases?per_page=" in url:
            return httpx.Response(
                200,
                json=[
                    {"tag_name": "b9999", "assets": []},
                    {
                        "tag_name": "b8888",
                        "assets": [
                            {
                                "name": "llama-b8888-bin-win-vulkan-x64.zip",
                                "browser_download_url": "https://example.test/fallback-vulkan.zip",
                            }
                        ],
                    },
                ],
            )
        if url.endswith("fallback-vulkan.zip"):
            return httpx.Response(
                200,
                content=zip_bytes,
                headers={"content-length": str(len(zip_bytes))},
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    exe = download_llama_runtime(base_dir=tmp_path, client=client)
    assert exe.name == "llama-server.exe"
    assert exe.read_bytes() == b"MZ-FALLBACK-SERVER"


def test_llama_server_manager_command_health_and_shutdown(tmp_path: Path) -> None:
    assert parse_keep_alive_seconds("0") == 0.0
    assert parse_keep_alive_seconds("45s") == 45.0
    assert parse_keep_alive_seconds("5m") == 300.0
    assert parse_keep_alive_seconds("1h") == 3600.0

    fake_bin = tmp_path / "llama-server.exe"
    fake_model = tmp_path / "model.gguf"
    fake_bin.write_bytes(b"bin")
    fake_model.write_bytes(b"gguf")

    class _FakeProc:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminated = False
            self.stderr = io.BytesIO(b"")

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout=None) -> int:
            return 0

    spawned_cmds: list[list[str]] = []
    procs: list[_FakeProc] = []

    def fake_popen(cmd, **kwargs):
        spawned_cmds.append(list(cmd))
        p = _FakeProc()
        procs.append(p)
        return p

    health_client = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok"}))
    )
    mgr = LlamaServerManager(
        fake_bin,
        fake_model,
        num_ctx=2048,
        num_gpu=0,
        keep_alive="10m",
        popen_fn=fake_popen,
        health_client=health_client,
    )
    url = mgr.ensure_started()
    assert url.startswith("http://127.0.0.1:")
    assert "-ngl" in spawned_cmds[0]
    assert spawned_cmds[0][spawned_cmds[0].index("-ngl") + 1] == "0"
    assert "--reasoning-budget" in spawned_cmds[0]
    assert mgr.is_running is True

    # Subsequent ensure_started reuses existing process
    assert mgr.ensure_started() == url
    assert len(spawned_cmds) == 1

    mgr.close()
    assert procs[0].terminated is True
    assert mgr.is_running is False


def test_embedded_polisher_styles_vocabulary_and_missing_model_fallback(tmp_path: Path) -> None:
    # 1. Missing runtime/model returns graceful fallback without crashing
    polisher_missing = EmbeddedLlamaPolisher(
        model_id="qwen25-1.5b",
        vocabulary="вайп кожу -> вайбкожу",
        base_dir=tmp_path,
    )
    res_fb = polisher_missing.polish("я вайп кожу")
    assert res_fb.used_llm is False
    assert res_fb.text == "я вайбкожу"
    assert res_fb.error

    # 2. Mocked running server handles polish, styles, short guard, and transform_selection
    class _StubManager:
        def ensure_started(self) -> str:
            return "http://127.0.0.1:19999"

        def touch_activity(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.append(body)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Ок, вайбкожу дальше."}}]},
        )

    polisher = EmbeddedLlamaPolisher(
        model_id="gemma3-4b",
        system_prompt="BASE",
        vocabulary="вайп кожу -> вайбкожу",
        server_manager=_StubManager(),  # type: ignore[arg-type]
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    tg = classify_window("Telegram.exe", "Telegram")
    alive = resolve_style_profile(tg)
    res = polisher.polish("ну ок вайп кожу дальше", window_context=tg, style_profile=alive)
    assert res.used_llm is True
    assert res.text == "Ок, вайбкожу дальше"  # stripped trailing period for short messenger
    assert "вайбкожу" in captured[0]["messages"][1]["content"]  # pre-LLM vocab replacement
    assert captured[0]["chat_template_kwargs"] == {"enable_thinking": False}

    # Short 1-word prompt in ai_prompt uses gentle default instruction
    ai = classify_window("Antigravity.exe", "chat — Antigravity")
    polisher.polish("Проанализируй.", window_context=ai, style_profile=resolve_style_profile(ai))
    assert "Style (general)" in captured[1]["messages"][0]["content"]

    # RAW skips HTTP completely
    res_raw = polisher.polish("я вайп кожу", style_profile=STYLE_PROFILES["raw"])
    assert res_raw.used_llm is False
    assert res_raw.text == "я вайбкожу"
    assert len(captured) == 2

    # transform_selection works via /v1/chat/completions
    res_tr = polisher.transform_selection("hello world", "переведи")
    assert res_tr.used_llm is True
    assert "<selected_text>\nhello world\n</selected_text>" in captured[2]["messages"][1]["content"]


def test_single_instance_guard_and_autostart_command(qtbot) -> None:
    cmd = build_autostart_command()
    assert "localflow" in cmd.lower()

    srv_name = "LocalFlow_Test_SingleInstance_Unique"
    g1 = SingleInstanceGuard(server_name=srv_name)
    g2 = SingleInstanceGuard(server_name=srv_name)
    activated: list[bool] = []
    g1.second_instance_activated.connect(lambda: activated.append(True))

    try:
        assert g1.try_acquire() is True
        assert g2.try_acquire() is False
        qtbot.waitUntil(lambda: len(activated) == 1, timeout=1500)
    finally:
        g2.release()
        g1.release()


def test_settings_window_embedded_backend_and_recommended_download_ui(
    tmp_path: Path, monkeypatch, qtbot
) -> None:
    monkeypatch.setattr("localflow.ui.settings_window.list_input_devices", lambda: [])
    storage = HistoryStorage(db_path=tmp_path / "hist.db")
    cfg = create_first_run_config(has_cuda=True)

    win = SettingsWindow(cfg, storage)
    qtbot.addWidget(win)
    win.resize(960, 640)
    win._navigate_to("settings")
    win.show()
    qtbot.wait(40)

    # Embedded backend visible in Settings, Ollama controls hidden, zero horizontal overflow
    assert win._llm_backend_combo.currentData() == "embedded"
    assert win._embedded_panel.isVisible() is True
    assert win._ollama_url_edit.isVisible() is False
    assert win._settings_scroll.horizontalScrollBar().maximum() == 0

    # Mock download of recommended models -> switches to gemma3-4b + large-v3-turbo
    downloaded_specs: list[str] = []

    def fake_download(spec, *, on_progress=None, cancel_event=None, lang=None):
        downloaded_specs.append(spec.id)
        if on_progress:
            on_progress(DownloadProgress(stage="Mock", downloaded_bytes=100, total_bytes=100, speed_mb_s=50.0))
        return tmp_path / spec.filename

    monkeypatch.setattr("localflow.ui.settings_window.download_embedded_llm_model", fake_download)

    with qtbot.waitSignal(win.config_saved, timeout=2000) as blocker:
        win._download_recommended_btn.click()

    assert downloaded_specs == [RECOMMENDED_EMBEDDED_MODEL_ID]
    saved: AppConfig = blocker.args[0]
    assert saved.llm_backend == "embedded"
    assert saved.embedded_llm_model_id == RECOMMENDED_EMBEDDED_MODEL_ID
    assert saved.whisper_model == "large-v3-turbo"

    # Toggle autostart & switch to Ollama backend roundtrip
    win._autostart_check.setChecked(True)
    win._llm_backend_combo.setCurrentIndex(win._llm_backend_combo.findData("ollama"))
    assert win._ollama_url_edit.isVisible() is True
    assert win._embedded_panel.isVisible() is False
    with qtbot.waitSignal(win.config_saved, timeout=1000) as blocker2:
        win._save_btn.click()
    assert blocker2.args[0].autostart_enabled is True
    assert blocker2.args[0].llm_backend == "ollama"
