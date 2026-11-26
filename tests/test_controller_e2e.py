from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from localflow.app import AppState, LocalFlowController
from localflow.asr.whisper_engine import ASRStatus
from localflow.config import AppConfig, load_config, save_config
from localflow.hotkey.capslock_hook import (
    VK_CAPITAL,
    VK_CONTROL,
    WM_KEYDOWN,
    WM_KEYUP,
    GlobalHotkeyListener,
)
from localflow.injector.window_context import ActiveWindowContext, classify_window
from localflow.llm.ollama_polisher import OllamaPolisher
from localflow.storage import HistoryStorage
from localflow.ui.overlay_pill import PillMode


class _FakeClipboard:
    def __init__(self) -> None:
        self.text_value: str = ""
        self.set_calls: list[str] = []

    def setText(self, text: str) -> None:
        self.text_value = text
        self.set_calls.append(text)

    def text(self) -> str:
        return self.text_value


class _FakeHotkeyListener(QObject):
    toggle_requested = Signal()
    command_requested = Signal()
    cancel_requested = Signal()

    def __init__(self, threshold_ms: int = 320, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.threshold_ms = threshold_ms
        self.recording_active = False
        self.started = False
        self.stopped = False

    def set_threshold_ms(self, threshold_ms: int) -> None:
        self.threshold_ms = threshold_ms

    def set_recording_active(self, active: bool) -> None:
        self.recording_active = active

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class _FakeAudioRecorder:
    def __init__(self) -> None:
        self.is_recording = False
        self.latest_level = 0.0
        self.start_calls: list[dict[str, Any]] = []
        self.stop_calls = 0
        self.stop_chunked_calls = 0
        self.on_segment = None
        self.on_silence_timeout = None
        self.next_audio = np.ones(16000, dtype=np.float32) * 0.05
        self.next_duration = 1.0
        self.next_tail_audio = np.zeros(0, dtype=np.float32)
        self.next_chunked_duration = 1.0
        self.start_error: Exception | None = None

    def start(
        self,
        device_index: int | None = None,
        *,
        pause_ms: int = 400,
        silence_auto_stop_sec: float | None = None,
        on_segment=None,
        on_silence_timeout=None,
    ) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.is_recording = True
        self.on_segment = on_segment
        self.on_silence_timeout = on_silence_timeout
        self.start_calls.append(
            {
                "device_index": device_index,
                "pause_ms": pause_ms,
                "silence_auto_stop_sec": silence_auto_stop_sec,
                "has_on_segment": on_segment is not None,
                "has_on_silence_timeout": on_silence_timeout is not None,
            }
        )

    def emit_segment(self, segment: np.ndarray) -> None:
        assert self.on_segment is not None
        self.on_segment(segment)

    def trigger_silence_timeout_from_worker(self) -> threading.Thread:
        cb = self.on_silence_timeout
        assert cb is not None
        t = threading.Thread(target=cb, name="fake-portaudio-cb", daemon=True)
        t.start()
        return t

    def stop(self) -> tuple[np.ndarray, float]:
        self.stop_calls += 1
        self.is_recording = False
        return self.next_audio.copy(), self.next_duration

    def stop_chunked(self) -> tuple[np.ndarray, float]:
        self.stop_chunked_calls += 1
        self.is_recording = False
        return self.next_tail_audio.copy(), self.next_chunked_duration


class _FakeWhisperEngine:
    instances: list["_FakeWhisperEngine"] = []

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "int8_float16",
        beam_size: int = 2,
        initial_prompt: str = "",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.prompt_updates: list[str] = [initial_prompt]
        self.status = ASRStatus(
            ready=False,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
        )
        self.transcribe_calls: list[tuple[int, str, str]] = []
        self.transcribe_fn = None
        self.unloaded = False
        _FakeWhisperEngine.instances.append(self)

    def load(self) -> ASRStatus:
        self.status = ASRStatus(
            ready=True,
            model_name=self.model_name,
            device="cpu",
            compute_type=self.compute_type,
        )
        return self.status

    def unload(self) -> None:
        self.unloaded = True
        self.status = ASRStatus(ready=False)

    def set_initial_prompt(self, initial_prompt: str) -> None:
        self.initial_prompt = initial_prompt
        self.prompt_updates.append(initial_prompt)

    def transcribe(self, audio: np.ndarray, *, initial_prompt: str | None = None) -> str:
        used_prompt = self.initial_prompt if initial_prompt is None else initial_prompt
        self.transcribe_calls.append(
            (int(audio.size), used_prompt, threading.current_thread().name)
        )
        if self.transcribe_fn is not None:
            return self.transcribe_fn(audio, used_prompt)
        return "распознанный текст"


class _FakeInjector:
    def __init__(self) -> None:
        self.selected_text: str | None = "исходный выделенный текст"
        self.capture_calls: list[str] = []
        self.inject_calls: list[dict[str, Any]] = []

    def capture_selected_text(
        self, timeout_ms: int = 180, restore_clipboard: bool = True
    ) -> str | None:
        self.capture_calls.append(threading.current_thread().name)
        return self.selected_text

    def inject(
        self,
        text: str,
        auto_space: bool = True,
        restore_clipboard: bool = True,
        paste_delay_ms: int = 120,
    ) -> bool:
        self.inject_calls.append(
            {
                "text": text,
                "auto_space": auto_space,
                "restore_clipboard": restore_clipboard,
                "paste_delay_ms": paste_delay_ms,
            }
        )
        return True


class _ControllerHarness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.cfg_path = tmp_path / "config.json"
        self.db_path = tmp_path / "history.db"
        self.cues: list[str] = []
        self.clipboard = _FakeClipboard()
        self.context_calls: list[str] = []
        self.active_window_ctx: ActiveWindowContext | None = classify_window(
            "Antigravity.exe", "chat — Antigravity"
        )
        self.ollama_requests: list[dict[str, Any]] = []
        self.ollama_handler = self._default_ollama_handler

        _FakeWhisperEngine.instances.clear()

        monkeypatch.setattr("localflow.app.get_app_data_dir", lambda: tmp_path)
        monkeypatch.setattr("localflow.app._setup_logging", lambda: None)
        monkeypatch.setattr(
            "localflow.app.load_config", lambda path=None: load_config(self.cfg_path)
        )
        monkeypatch.setattr(
            "localflow.app.save_config", lambda cfg, path=None: save_config(cfg, self.cfg_path)
        )
        monkeypatch.setattr(
            "localflow.app.HistoryStorage",
            lambda db_path=None, max_entries=500: HistoryStorage(
                db_path=self.db_path, max_entries=max_entries
            ),
        )
        monkeypatch.setattr("localflow.app.play_cue", lambda kind: self.cues.append(kind))
        monkeypatch.setattr(
            "localflow.ui.settings_window.list_input_devices", lambda: []
        )
        monkeypatch.setattr(
            QApplication, "clipboard", staticmethod(lambda: self.clipboard)
        )

        def _fake_get_ctx() -> ActiveWindowContext | None:
            self.context_calls.append(threading.current_thread().name)
            return self.active_window_ctx

        monkeypatch.setattr("localflow.app.get_active_window_context", _fake_get_ctx)
        monkeypatch.setattr("localflow.app.GlobalHotkeyListener", _FakeHotkeyListener)
        monkeypatch.setattr("localflow.app.AudioRecorder", _FakeAudioRecorder)
        monkeypatch.setattr("localflow.app.WhisperEngine", _FakeWhisperEngine)
        monkeypatch.setattr("localflow.app.Win32TextInjector", _FakeInjector)

        harness = self

        def _make_polisher(**kwargs) -> OllamaPolisher:
            transport = httpx.MockTransport(harness._dispatch_ollama)
            client = httpx.Client(transport=transport)
            return OllamaPolisher(client=client, **kwargs)

        monkeypatch.setattr("localflow.app.OllamaPolisher", _make_polisher)

    def _default_ollama_handler(self, payload: dict[str, Any]) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"content": "Очищенный текст"}},
        )

    def _dispatch_ollama(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/chat"):
            payload = json.loads(request.content.decode("utf-8"))
            self.ollama_requests.append(payload)
            return self.ollama_handler(payload)
        if request.url.path.endswith("/api/tags"):
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, json={})

    def create_controller(
        self, qapp: QApplication, qtbot, initial_config: AppConfig | None = None
    ) -> LocalFlowController:
        if initial_config is not None:
            save_config(initial_config, self.cfg_path)
        ctrl = LocalFlowController(qapp)
        qtbot.waitUntil(lambda: ctrl._asr.status.ready, timeout=3000)
        return ctrl


@pytest.fixture
def make_harness(tmp_path, monkeypatch, qapp, qtbot):
    controllers: list[LocalFlowController] = []
    harness = _ControllerHarness(tmp_path, monkeypatch)

    def _start(initial_config: AppConfig | None = None) -> tuple[LocalFlowController, _ControllerHarness]:
        ctrl = harness.create_controller(qapp, qtbot, initial_config=initial_config)
        controllers.append(ctrl)
        return ctrl, harness

    yield _start

    for ctrl in controllers:
        ctrl.shutdown()
        ctrl._pill.close()
        ctrl._pill.deleteLater()
        ctrl._settings.close()
        ctrl._settings.deleteLater()
        ctrl._tray.deleteLater()
    qapp.processEvents()


def test_r1_captures_active_window_context_on_main_thread_and_passes_to_polisher(
    make_harness, qtbot
) -> None:
    ctrl, h = make_harness(AppConfig(ollama_warmup_on_start=False))
    h.active_window_ctx = classify_window("Antigravity.exe", "main.py — Antigravity")
    ctrl._asr.transcribe_fn = lambda audio, prompt: "проверь camelCase и snake_case в новом модуле сейчас"

    ctrl._hotkey.toggle_requested.emit()
    assert ctrl._state == AppState.RECORDING
    assert h.context_calls == [threading.main_thread().name]

    # Even if the foreground window changes before stop, the captured start context is used.
    h.active_window_ctx = classify_window("Telegram.exe", "Telegram")
    h.cues.clear()
    ctrl._recorder.next_chunked_duration = 1.2
    ctrl._recorder.next_tail_audio = np.ones(8000, dtype=np.float32) * 0.05
    ctrl._hotkey.toggle_requested.emit()

    qtbot.waitUntil(lambda: ctrl._state == AppState.IDLE, timeout=3000)
    assert len(h.ollama_requests) == 1
    sys_msg = h.ollama_requests[0]["messages"][0]["content"]
    assert "Active window: Antigravity" in sys_msg
    assert "camelCase" in sys_msg
    assert "snake_case" in sys_msg


def test_r2_vocabulary_prompt_sync_on_init_and_apply_config_without_whisper_reload(
    make_harness, qtbot
) -> None:
    cfg = AppConfig(
        custom_vocabulary="Antigravity, PySide6\nкурс -> курсор",
        ollama_warmup_on_start=False,
    )
    ctrl, h = make_harness(cfg)
    initial_asr = ctrl._asr
    assert "Antigravity" in initial_asr.initial_prompt
    assert "PySide6" in initial_asr.initial_prompt
    assert "курсор" in initial_asr.initial_prompt

    updated_cfg = cfg.model_copy(
        update={"custom_vocabulary": "FastAPI, LocalFlow\nдокер -> Docker"}
    )
    ctrl._on_config_saved(updated_cfg)

    assert ctrl._asr is initial_asr
    assert not initial_asr.unloaded
    assert "FastAPI" in initial_asr.initial_prompt
    assert "Docker" in initial_asr.initial_prompt

    ctrl._asr.transcribe_fn = lambda audio, prompt: "проверь пожалуйста наш новый FastAPI сервис"
    ctrl._hotkey.toggle_requested.emit()
    ctrl._recorder.next_chunked_duration = 1.0
    ctrl._recorder.next_tail_audio = np.ones(8000, dtype=np.float32) * 0.05
    ctrl._hotkey.toggle_requested.emit()

    qtbot.waitUntil(lambda: ctrl._state == AppState.IDLE, timeout=3000)
    sys_msg = h.ollama_requests[-1]["messages"][0]["content"]
    assert "FastAPI" in sys_msg
    assert '"докер" -> "Docker"' in sys_msg


def test_r2_deterministic_replacements_when_llm_disabled(make_harness, qtbot) -> None:
    cfg = AppConfig(
        llm_enabled=False,
        custom_vocabulary="курс -> курсор",
        ollama_warmup_on_start=False,
    )
    ctrl, h = make_harness(cfg)
    ctrl._asr.transcribe_fn = lambda audio, prompt: "перемести курс в конец"

    ctrl._hotkey.toggle_requested.emit()
    ctrl._recorder.next_chunked_duration = 1.1
    ctrl._recorder.next_tail_audio = np.ones(8000, dtype=np.float32) * 0.05
    ctrl._hotkey.toggle_requested.emit()

    qtbot.waitUntil(lambda: ctrl._state == AppState.IDLE, timeout=3000)
    assert h.ollama_requests == []
    assert len(ctrl._injector.inject_calls) == 1
    assert ctrl._injector.inject_calls[0]["text"] == "перемести курсор в конец"
    recent = ctrl._storage.list_recent(limit=1)
    assert recent[0].final_text == "перемести курсор в конец"
    assert recent[0].used_llm is False


def test_r3_command_mode_ctrl_double_capslock_transforms_and_pastes_without_auto_space(
    make_harness, qtbot
) -> None:
    ctrl, h = make_harness(AppConfig(ollama_warmup_on_start=False))
    h.active_window_ctx = classify_window("Code.exe", "app.py — Visual Studio Code")
    ctrl._injector.selected_text = "hello world"
    ctrl._asr.transcribe_fn = lambda audio, prompt: "переведи на русский"
    h.ollama_handler = lambda payload: httpx.Response(
        200, json={"message": {"content": "привет мир"}}
    )

    # Sub-check: real GlobalHotkeyListener produces command_requested on Ctrl+CapsLockx2
    # and toggle_requested on single CapsLock when recording is active.
    real_hook = GlobalHotkeyListener(threshold_ms=320, ctrl_async_state_fn=lambda: False)
    hook_events: list[str] = []
    real_hook.command_requested.connect(lambda: hook_events.append("command"))
    real_hook.toggle_requested.connect(lambda: hook_events.append("toggle"))
    real_hook._handle_key_event(WM_KEYDOWN, VK_CONTROL, timestamp=1.00)
    real_hook._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.02)
    real_hook._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.05)
    real_hook._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.15)
    real_hook._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.18)
    real_hook._handle_key_event(WM_KEYUP, VK_CONTROL, timestamp=1.20)
    real_hook.set_recording_active(True)
    real_hook._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=2.00)
    real_hook._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=2.03)
    assert hook_events == ["command", "toggle"]

    ctrl._hotkey.command_requested.emit()
    assert ctrl._state == AppState.RECORDING
    assert ctrl._hotkey.recording_active is True
    assert ctrl._injector.capture_calls == [threading.main_thread().name]

    ctrl._recorder.next_duration = 1.2
    ctrl._hotkey.toggle_requested.emit()

    qtbot.waitUntil(lambda: ctrl._state == AppState.IDLE, timeout=3000)
    assert len(h.ollama_requests) == 1
    user_msg = h.ollama_requests[0]["messages"][1]["content"]
    sys_msg = h.ollama_requests[0]["messages"][0]["content"]
    assert "hello world" in user_msg
    assert "переведи на русский" in user_msg
    assert "Visual Studio Code" in sys_msg

    assert len(ctrl._injector.inject_calls) == 1
    assert ctrl._injector.inject_calls[0]["text"] == "привет мир"
    assert ctrl._injector.inject_calls[0]["auto_space"] is False
    assert ctrl._last_dictation_text == "привет мир"
    assert ctrl._storage.list_recent(limit=1)[0].final_text == "привет мир"


def test_r3_command_mode_aborts_on_empty_or_none_selection(make_harness) -> None:
    ctrl, _ = make_harness(AppConfig(ollama_warmup_on_start=False))

    for sel in (None, "  \n\t "):
        ctrl._injector.selected_text = sel
        ctrl._hotkey.command_requested.emit()
        assert ctrl._state == AppState.IDLE
        assert ctrl._recorder.start_calls == []
        assert ctrl._pill._notice_text == "Select text for a command"


@pytest.mark.parametrize(
    "bad_response",
    [
        httpx.Response(503, text="Service Unavailable"),
        httpx.Response(200, json={"message": {"content": ""}}),
    ],
)
def test_r3_command_mode_aborts_paste_on_llm_error_or_empty_response(
    make_harness, qtbot, bad_response
) -> None:
    ctrl, h = make_harness(AppConfig(ollama_warmup_on_start=False))
    ctrl._last_dictation_text = "предыдущая диктовка"
    ctrl._injector.selected_text = "keep this text"
    ctrl._asr.transcribe_fn = lambda audio, prompt: "сократи"
    h.ollama_handler = lambda payload: bad_response

    ctrl._hotkey.command_requested.emit()
    ctrl._recorder.next_duration = 1.0
    ctrl._hotkey.toggle_requested.emit()

    qtbot.waitUntil(lambda: ctrl._state == AppState.IDLE, timeout=3000)
    assert ctrl._injector.inject_calls == []
    assert ctrl._storage.list_recent(limit=10) == []
    assert ctrl._last_dictation_text == "предыдущая диктовка"
    assert ctrl._pill._notice_is_error is True


def test_r3_command_ignored_while_busy_and_esc_clears_pending_command_state(
    make_harness, qtbot
) -> None:
    ctrl, h = make_harness(AppConfig(ollama_warmup_on_start=False))
    ctrl._injector.selected_text = "selected text"
    ctrl._hotkey.command_requested.emit()
    assert ctrl._state == AppState.RECORDING
    assert ctrl._is_command_mode is True

    # Ignored while RECORDING
    calls_before = len(ctrl._injector.capture_calls)
    ctrl._hotkey.command_requested.emit()
    assert len(ctrl._injector.capture_calls) == calls_before

    # Esc cancels and clears command state
    ctrl._hotkey.cancel_requested.emit()
    assert ctrl._state == AppState.IDLE
    assert ctrl._is_command_mode is False
    assert ctrl._pending_selected_text is None
    assert ctrl._pending_window_context is None

    # Subsequent normal dictation calls polish, NOT transform_selection
    ctrl._asr.transcribe_fn = lambda audio, prompt: "ну давай проверим обычную диктовку"
    h.ollama_handler = lambda payload: httpx.Response(
        200, json={"message": {"content": "Обычная диктовка."}}
    )
    ctrl._hotkey.toggle_requested.emit()
    ctrl._recorder.next_chunked_duration = 1.0
    ctrl._recorder.next_tail_audio = np.ones(8000, dtype=np.float32) * 0.05
    ctrl._hotkey.toggle_requested.emit()

    qtbot.waitUntil(lambda: ctrl._state == AppState.IDLE, timeout=3000)
    assert len(h.ollama_requests) == 1
    assert "selected text" not in h.ollama_requests[0]["messages"][1]["content"]
    assert ctrl._injector.inject_calls[0]["text"] == "Обычная диктовка"


def test_r4_chunked_asr_processes_segments_before_stop_and_handles_empty_tail(
    make_harness, qtbot
) -> None:
    ctrl, h = make_harness(AppConfig(llm_enabled=False, ollama_warmup_on_start=False))
    seg_outputs = iter(["seg1", "seg2"])
    ctrl._asr.transcribe_fn = lambda audio, prompt: next(seg_outputs)

    ctrl._hotkey.toggle_requested.emit()
    assert ctrl._recorder.start_calls[-1]["pause_ms"] == 750
    assert ctrl._recorder.start_calls[-1]["has_on_segment"] is True

    ctrl._recorder.emit_segment(np.ones(8000, dtype=np.float32) * 0.05)
    ctrl._recorder.emit_segment(np.ones(8000, dtype=np.float32) * 0.05)

    # Wait for both background segment transcriptions to finish BEFORE stopping!
    qtbot.waitUntil(lambda: len(ctrl._asr.transcribe_calls) == 2, timeout=3000)
    assert all(
        call[2] == "localflow-chunked-asr" for call in ctrl._asr.transcribe_calls
    )

    # Trailing pause flushed everything: tail_audio is empty, duration is 2.5s
    ctrl._recorder.next_tail_audio = np.zeros(0, dtype=np.float32)
    ctrl._recorder.next_chunked_duration = 2.5
    ctrl._hotkey.toggle_requested.emit()

    qtbot.waitUntil(lambda: ctrl._state == AppState.IDLE, timeout=3000)
    assert ctrl._recorder.stop_chunked_calls == 1
    assert len(ctrl._asr.transcribe_calls) == 2
    assert ctrl._injector.inject_calls[0]["text"] == "seg1 seg2"


def test_r4_short_audio_abort_and_esc_cancel_call_session_cancel(
    make_harness, qtbot
) -> None:
    ctrl, _ = make_harness(AppConfig(min_duration_sec=0.3, ollama_warmup_on_start=False))

    # Short audio (< min_duration_sec) cancels ChunkedASRSession and skips ASR/paste
    ctrl._hotkey.toggle_requested.emit()
    session_short = ctrl._chunked_session
    assert session_short is not None
    ctrl._recorder.next_chunked_duration = 0.1
    ctrl._recorder.next_tail_audio = np.zeros(0, dtype=np.float32)
    ctrl._hotkey.toggle_requested.emit()

    assert ctrl._state == AppState.IDLE
    assert session_short._cancelled is True
    assert ctrl._pill._mode == PillMode.IDLE_BAR
    assert ctrl._asr.transcribe_calls == []
    assert ctrl._injector.inject_calls == []

    # Esc cancel cancels ChunkedASRSession and transitions to IDLE
    ctrl._hotkey.toggle_requested.emit()
    session_esc = ctrl._chunked_session
    assert session_esc is not None
    ctrl._hotkey.cancel_requested.emit()

    assert ctrl._state == AppState.IDLE
    assert session_esc._cancelled is True
    assert ctrl._pill._notice_text == "Cancelled"


def test_r5_live_preview_queued_cross_thread_updates_enabled_toggle_and_stale_guard(
    make_harness, qtbot
) -> None:
    ctrl, _ = make_harness(AppConfig(live_preview_enabled=True, ollama_warmup_on_start=False))
    ctrl._asr.transcribe_fn = lambda audio, prompt: "живое превью"

    ctrl._hotkey.toggle_requested.emit()
    gen = ctrl._session_gen
    ctrl._recorder.emit_segment(np.ones(8000, dtype=np.float32) * 0.05)

    qtbot.waitUntil(lambda: ctrl._pill._preview_text == "живое превью", timeout=3000)
    assert ctrl._pill._top_offset() > 0

    # Stale session_gen update is ignored
    ctrl._partial_text_ready.emit(gen - 1, "устаревший текст")
    qapp = QApplication.instance()
    assert qapp is not None
    qapp.processEvents()
    assert ctrl._pill._preview_text == "живое превью"

    # Disabling live preview in config updates pill immediately
    ctrl._on_config_saved(ctrl._config.model_copy(update={"live_preview_enabled": False}))
    assert ctrl._pill._live_preview_enabled is False
    assert ctrl._pill._top_offset() == 0

    # Cancel so non-RECORDING partial signal is also ignored
    ctrl._hotkey.cancel_requested.emit()
    ctrl._partial_text_ready.emit(gen, "после отмены")
    qapp.processEvents()
    assert ctrl._pill._preview_text == ""


def test_r6_silence_auto_stop_default_off_cross_thread_trigger_and_copy_last_dictation(
    make_harness, qtbot, tmp_path, monkeypatch
) -> None:
    # Seed pre-existing HistoryStorage before controller initialization
    seed_db = tmp_path / "history.db"
    seed_storage = HistoryStorage(db_path=seed_db)
    seed_storage.add_entry(
        duration_sec=1.5,
        raw_text="ранее сохранённая",
        final_text="Ранее сохранённая диктовка.",
        llm_model=None,
        used_llm=False,
    )

    ctrl, h = make_harness(AppConfig(llm_enabled=False, ollama_warmup_on_start=False))

    # Pre-existing history recovers _last_dictation_text and copies via tray
    assert ctrl._last_dictation_text == "Ранее сохранённая диктовка."
    ctrl._tray.copy_last_dictation_requested.emit()
    assert h.clipboard.text() == "Ранее сохранённая диктовка."
    assert ctrl._pill._notice_text == "Last dictation copied"

    # Default config has silence auto-stop disabled (silence_auto_stop_sec=None)
    ctrl._hotkey.toggle_requested.emit()
    assert ctrl._recorder.start_calls[-1]["silence_auto_stop_sec"] is None
    assert ctrl._recorder.start_calls[-1]["has_on_silence_timeout"] is False
    ctrl._hotkey.cancel_requested.emit()

    # Enable silence auto-stop = 2.5s and trigger timeout from worker thread
    ctrl._on_config_saved(
        ctrl._config.model_copy(
            update={"silence_auto_stop_enabled": True, "silence_auto_stop_sec": 2.5}
        )
    )
    ctrl._asr.transcribe_fn = lambda audio, prompt: "авто стоп сработал"
    ctrl._hotkey.toggle_requested.emit()
    assert ctrl._recorder.start_calls[-1]["silence_auto_stop_sec"] == pytest.approx(2.5)
    assert ctrl._recorder.start_calls[-1]["has_on_silence_timeout"] is True

    ctrl._recorder.next_tail_audio = np.ones(8000, dtype=np.float32) * 0.05
    ctrl._recorder.next_chunked_duration = 2.6
    worker_thread = ctrl._recorder.trigger_silence_timeout_from_worker()
    worker_thread.join(timeout=1.0)

    qtbot.waitUntil(lambda: ctrl._state == AppState.IDLE, timeout=3000)
    assert ctrl._injector.inject_calls[-1]["text"] == "авто стоп сработал"
    assert ctrl._last_dictation_text == "авто стоп сработал"

    ctrl._tray.copy_last_dictation_requested.emit()
    assert h.clipboard.text() == "авто стоп сработал"

    # Empty history notice when no dictation exists
    ctrl._last_dictation_text = ""
    ctrl._storage.clear()
    ctrl._tray.copy_last_dictation_requested.emit()
    assert ctrl._pill._notice_text == "No saved dictation"


def test_flow_bar_idle_snooze_mic_switch_and_reset_position(make_harness, monkeypatch) -> None:
    from localflow.audio.recorder import AudioDeviceInfo

    monkeypatch.setattr(
        "localflow.app.list_input_devices",
        lambda: [AudioDeviceInfo(index=3, name="HyperX Mic", max_input_channels=1, is_default=False)],
    )
    ctrl, harness = make_harness(
        AppConfig(show_idle_bar=True, pill_x=300, pill_y=400, ollama_warmup_on_start=False)
    )
    assert ctrl._pill._mode == PillMode.IDLE_BAR
    assert ctrl._pill.isVisible()

    # Snooze hides the resting bar; unsnooze brings it back
    ctrl._on_snooze_1h()
    assert not ctrl._pill.isVisible()
    assert ctrl._snooze_timer.isActive()
    ctrl._on_unsnooze()
    assert ctrl._pill._mode == PillMode.IDLE_BAR
    assert ctrl._pill.isVisible()

    # Switching microphone updates config + tray + settings and persists
    ctrl._on_mic_selected(3)
    assert ctrl._config.audio_device_index == 3
    assert load_config(harness.cfg_path).audio_device_index == 3
    assert "HyperX Mic" in ctrl._pill._notice_text

    # Resetting pill position clears pill_x/pill_y
    ctrl._on_reset_pill_position()
    assert ctrl._config.pill_x is None
    assert ctrl._config.pill_y is None
    assert load_config(harness.cfg_path).pill_x is None

    # Disabling idle bar hides pill completely
    ctrl._on_toggle_idle_bar(False)
    assert ctrl._config.show_idle_bar is False
    assert not ctrl._pill.isVisible()


def test_cuda_two_pass_asr_uses_full_audio_instead_of_stitched_chunks(
    make_harness, qtbot
) -> None:
    ctrl, _ = make_harness(AppConfig(llm_enabled=False, ollama_warmup_on_start=False))
    # Simulate CUDA ASR device
    ctrl._asr.status = ASRStatus(
        ready=True,
        model_name="large-v3-turbo",
        device="cuda",
        compute_type="float16",
    )
    full_buf = np.ones(32000, dtype=np.float32) * 0.08
    ctrl._recorder.stop_chunked_with_full_audio = lambda: (  # type: ignore[attr-defined]
        np.ones(4000, dtype=np.float32) * 0.05,
        full_buf,
        2.0,
    )
    ctrl._asr.transcribe_fn = lambda audio, prompt: (
        "цельный контекстный проход" if audio.size == 32000 else "рваный чанк"
    )

    ctrl._hotkey.toggle_requested.emit()
    session = ctrl._chunked_session
    assert session is not None
    # Live preview still runs on chunked background session
    ctrl._recorder.emit_segment(np.ones(8000, dtype=np.float32) * 0.05)
    qtbot.waitUntil(lambda: ctrl._pill._preview_text == "рваный чанк", timeout=3000)

    # Stop triggers single full-context GPU pass on full_buf
    ctrl._hotkey.toggle_requested.emit()
    qtbot.waitUntil(lambda: ctrl._state == AppState.IDLE, timeout=3000)
    assert session._cancelled is True
    assert ctrl._injector.inject_calls[-1]["text"] == "цельный контекстный проход"


def test_warming_pill_queues_autostart_and_cancels_on_esc(make_harness, qtbot) -> None:
    ctrl, _ = make_harness(AppConfig(llm_enabled=False, ollama_warmup_on_start=False))

    # Simulate ASR still loading when the user presses 2x CapsLock
    ctrl._asr.status = ASRStatus(
        ready=False,
        model_name="large-v3-turbo",
        device="cuda",
        compute_type="int8_float16",
    )
    ctrl._hotkey.toggle_requested.emit()
    assert ctrl._state == AppState.IDLE
    assert ctrl._start_recording_when_ready is True
    assert ctrl._pill._mode == PillMode.WARMING

    # Esc cancels the queued auto-start cleanly
    ctrl._hotkey.cancel_requested.emit()
    assert ctrl._start_recording_when_ready is False
    assert ctrl._state == AppState.IDLE
    assert ctrl._pill._mode != PillMode.WARMING

    # Queue again, then finish ASR loading -> recording starts automatically
    ctrl._hotkey.toggle_requested.emit()
    assert ctrl._start_recording_when_ready is True
    assert ctrl._pill._mode == PillMode.WARMING

    ready_status = ASRStatus(
        ready=True,
        model_name="large-v3-turbo",
        device="cuda",
        compute_type="int8_float16",
    )
    ctrl._asr.status = ready_status
    ctrl._asr_loaded.emit(ctrl._asr_load_gen, ready_status)
    qtbot.waitUntil(lambda: ctrl._state == AppState.RECORDING, timeout=3000)
    assert ctrl._start_recording_when_ready is False

