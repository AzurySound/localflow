"""Main application controller and state machine for LocalFlow."""

from __future__ import annotations

import logging
import math
from enum import Enum
import threading
import time

import numpy as np
from PySide6.QtCore import QObject, QPoint, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from localflow.asr.chunked_session import ChunkedASRSession
from localflow.asr.whisper_engine import ASRStatus, WhisperEngine
from localflow.audio.loopback_recorder import AudioMode, MeetingAudioRecorder
from localflow.audio.recorder import AudioRecorder, list_input_devices
from localflow.audio.sound_fx import play_cue
from localflow.autostart import set_autostart_enabled
from localflow.config import (
    COMMAND_WHISPER_PROMPT,
    AppConfig,
    apply_performance_profile,
    get_app_data_dir,
    is_portable_mode,
    load_config,
    save_config,
)
from localflow.hotkey.capslock_hook import GlobalHotkeyListener
from localflow.i18n import LanguageCode, normalize_language, tr
from localflow.injector.win32_paste import Win32TextInjector
from localflow.injector.window_context import ActiveWindowContext, get_active_window_context
from localflow.llm.downloader import (
    DownloadProgress,
    find_local_whisper_model_path,
    get_whisper_model_dir,
)
from localflow.llm.embedded_polisher import EmbeddedLlamaPolisher
from localflow.llm.ollama_polisher import (
    OllamaPolisher,
    find_ollama_launch_command,
    is_local_ollama_url,
    launch_ollama_background,
    should_bypass_llm,
    strip_long_dashes,
)
from localflow.storage import HistoryStorage
from localflow.notetaker.session import MeetingSession
from localflow.notetaker.storage import MeetingStorage
from localflow.notetaker.summarizer import MeetingSummarizer
from localflow.styles import apply_style_postprocessing, resolve_style_profile, strip_final_period
from localflow.ui.context_menu import QuickMenuCallbacks, QuickMenuState, build_quick_menu
from localflow.ui.meeting_window import MeetingSideWindow
from localflow.ui.overlay_pill import FloatingPillWidget
from localflow.ui.settings_window import SettingsWindow
from localflow.ui.theme import resolve_theme
from localflow.ui.tray import LocalFlowTray
from localflow.vocabulary import (
    apply_vocabulary_replacements_with_count,
    build_whisper_initial_prompt,
    parse_vocabulary,
)

logger = logging.getLogger("localflow")


def _setup_logging() -> None:
    if logger.handlers:
        return
    log_path = get_app_data_dir() / "localflow.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AppState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    POLISHING = "polishing"


class _FnWorker(QRunnable):
    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        self._fn()


class LocalFlowController(QObject):
    """Coordinates hotkey, audio capture, ASR, Ollama polishing, UI, and Win32 paste."""

    _asr_loaded = Signal(int, object)
    _asr_failed = Signal(int, str)
    _asr_download_progress = Signal(int, object)  # gen, DownloadProgress
    _stage_polishing = Signal(str)
    _stage_injecting = Signal()
    _pipeline_done = Signal(bool, str, str, str)  # llm_fallback_needed, llm_err, final_text, raw_text
    _pipeline_empty = Signal(str)
    _pipeline_error = Signal(str)
    _profile_ready = Signal(str, object)  # profile, available_models (or None)
    _partial_text_ready = Signal(int, str)  # session_gen, partial_text
    _auto_stop_requested = Signal(int)  # session_gen
    _ollama_start_status = Signal(bool, str)  # ready_ok, notice_text
    _meeting_stopped_done = Signal(int)  # meeting_id
    _meeting_warning_signal = Signal(str)
    _meeting_summary_step = Signal(int, str, int, int)  # meeting_id, stage, step, total
    _meeting_summary_done = Signal(int, bool, str)  # meeting_id, too_short, err

    def __init__(self, app: QApplication) -> None:
        super().__init__()
        _setup_logging()
        self._app = app
        self._config = load_config()
        self._lang: LanguageCode = normalize_language(self._config.language)
        self._storage = HistoryStorage()
        self._meeting_storage = MeetingStorage()
        self._state = AppState.IDLE
        self._asr_load_gen = 0
        self._session_gen = 0
        self._chunked_session: ChunkedASRSession | None = None
        self._dictation_using_meeting_tap = False
        self._pending_window_context: ActiveWindowContext | None = None
        self._pending_selected_text: str | None = None
        self._is_command_mode = False
        self._shutting_down = False
        self._asr_status_kind = "loading"
        self._asr_status_obj: ASRStatus | None = None
        self._asr_downloading = False
        self._asr_download_pct = -1
        self._asr_load_started_at = 0.0
        self._start_recording_when_ready = False
        self._ollama_start_prompted = False
        self._asr_status_text = tr("menu.asr_loading_model", self._lang, model=self._config.whisper_model)
        recent = self._storage.list_recent(limit=1)
        self._last_dictation_text: str = recent[0].final_text if recent else ""
        self._last_raw_text: str = recent[0].raw_text if recent else ""
        self._pool = QThreadPool.globalInstance()

        self._snooze_timer = QTimer(self)
        self._snooze_timer.setSingleShot(True)
        self._snooze_timer.timeout.connect(self._on_unsnooze)

        self._warming_timer = QTimer(self)
        self._warming_timer.setInterval(250)
        self._warming_timer.timeout.connect(self._update_warming_pill)

        self._history_cleanup_timer = QTimer(self)
        self._history_cleanup_timer.setInterval(60_000)
        self._history_cleanup_timer.timeout.connect(self._run_history_auto_cleanup)
        self._history_cleanup_timer.start()

        self._recorder = AudioRecorder()
        self._meeting_recorder = MeetingAudioRecorder()
        self._injector = Win32TextInjector()
        self._asr = self._build_asr(self._config)
        self._polisher = self._build_polisher(self._config)
        self._meeting_session = MeetingSession(
            storage=self._meeting_storage,
            whisper_engine=self._asr,
            recorder=self._meeting_recorder,
            vocabulary=parse_vocabulary(self._config.custom_vocabulary),
            whisper_prompt=self._config.whisper_initial_prompt,
            diarization_enabled=self._config.notetaker_diarization_enabled,
            lang=self._lang,
        )
        self._meeting_pill_timer = QTimer(self)
        self._meeting_pill_timer.setInterval(1000)
        self._meeting_pill_timer.timeout.connect(self._on_meeting_pill_tick)

        initial_pos = (
            (self._config.pill_x, self._config.pill_y)
            if self._config.pill_x is not None and self._config.pill_y is not None
            else None
        )
        resolved_theme = resolve_theme(self._config.theme)
        self._pill = FloatingPillWidget(initial_pos=initial_pos)
        self._pill.set_theme(resolved_theme)
        self._pill.set_language(self._lang)
        self._pill.set_level_provider(
            lambda: (
                self._meeting_recorder.mic_level
                if self._dictation_using_meeting_tap
                else self._recorder.latest_level
            )
        )
        self._pill.set_live_preview_enabled(self._config.live_preview_enabled)
        self._pill.set_idle_bar_enabled(self._config.show_idle_bar)
        self._pill.set_has_last_dictation(bool(self._last_dictation_text))
        self._tray = LocalFlowTray(
            llm_enabled=self._config.llm_enabled,
            profile=self._config.performance_profile,
            show_idle_bar=self._config.show_idle_bar,
            selected_mic_index=self._config.audio_device_index,
            theme=resolved_theme,
            lang=self._lang,
        )
        if self._last_dictation_text:
            self._tray.set_last_dictation_preview(self._last_dictation_text)
        if self._last_raw_text:
            self._tray.set_last_raw_preview(self._last_raw_text)
        self._settings = SettingsWindow(self._config, self._storage, self._meeting_storage)
        self._meeting_window = MeetingSideWindow(
            self._meeting_storage,
            llm_factory=lambda: self._polisher if self._config.llm_enabled else None,
            lang=self._lang,
            theme=resolved_theme,
        )
        nt = self._settings.notetaker_tab
        mw = self._meeting_window.workspace
        llm_getter = lambda: self._polisher if self._config.llm_enabled else None
        lvl_getter = lambda: (
            self._meeting_recorder.mic_level,
            self._meeting_recorder.system_level,
            self._meeting_recorder.elapsed_sec,
        )
        nt.set_llm_factory(llm_getter)
        nt.set_level_provider(lvl_getter)
        mw.set_llm_factory(llm_getter)
        mw.set_level_provider(lvl_getter)

        def _on_utt(utt, spk) -> None:
            nt.notify_utterance(utt.meeting_id, utt, spk)
            mw.notify_utterance(utt.meeting_id, utt, spk)

        def _on_interim(mid: int, ch: str, spk_key: str, txt: str) -> None:
            nt.notify_interim_text(mid, ch, spk_key, txt)
            mw.notify_interim_text(mid, ch, spk_key, txt)

        def _on_spks(mid: int) -> None:
            nt.notify_speakers_updated(mid)
            mw.notify_speakers_updated(mid)

        self._meeting_session.on_utterance = _on_utt
        self._meeting_session.on_interim_text = _on_interim
        self._meeting_session.on_speakers_updated = _on_spks
        self._meeting_session.on_warning = self._meeting_warning_signal.emit
        self._hotkey = GlobalHotkeyListener(threshold_ms=self._config.double_tap_ms)

        self._connect_signals()
        self._run_history_auto_cleanup()
        self._tray.show()
        self._restore_pill_idle()
        self._hotkey.start()
        self._load_asr_async()

        if not self._config.first_run_completed:
            self._config = self._config.model_copy(update={"first_run_completed": True})
            save_config(self._config)

    def show_hub(self) -> None:
        """Bring the Hub/Settings window to the foreground (also called on second instance launch)."""
        self._settings.show_and_focus()

    def _open_first_run_setup(self) -> None:
        self._settings.show_and_focus()
        self._settings._navigate_to("settings")

    def _refresh_asr_status_label(self) -> None:
        if self._asr_status_kind == "ready" and self._asr_status_obj is not None:
            st = self._asr_status_obj
            self._asr_status_text = f"ASR: {st.model_name} ({st.device}/{st.compute_type})"
        elif self._asr_status_kind == "error":
            self._asr_status_text = tr("menu.asr_load_error", self._lang)
        else:
            self._asr_status_text = tr(
                "menu.asr_loading_model", self._lang, model=self._config.whisper_model
            )
        self._tray.set_asr_status(self._asr_status_text)

    def _apply_language(self, lang: str) -> None:
        self._lang = normalize_language(lang)
        self._pill.set_language(self._lang)
        self._tray.set_language(self._lang)
        self._meeting_window.retranslate(self._lang)
        self._refresh_asr_status_label()

    def _on_language_preview(self, lang: str) -> None:
        self._apply_language(lang)

    def _apply_resolved_theme(self, theme_mode: str | None = None) -> None:
        mode = theme_mode if theme_mode is not None else self._config.theme
        resolved = resolve_theme(mode)
        self._pill.set_theme(resolved)
        self._tray.set_theme(resolved)
        self._settings.apply_theme(mode)
        self._meeting_window.set_theme(resolved)

    def _on_theme_preview(self, mode: str) -> None:
        resolved = resolve_theme(mode)
        self._pill.set_theme(resolved)
        self._tray.set_theme(resolved)
        self._meeting_window.set_theme(resolved)

    def _on_system_theme_changed(self, *_args) -> None:
        if not self._shutting_down and self._config.theme == "system":
            self._apply_resolved_theme("system")

    @staticmethod
    def _active_llm_label(cfg: AppConfig) -> str:
        if cfg.llm_backend == "embedded":
            if cfg.embedded_llm_custom_path.strip():
                from pathlib import Path

                return Path(cfg.embedded_llm_custom_path.strip()).stem
            return cfg.embedded_llm_model_id
        return cfg.ollama_model

    def _build_asr(self, cfg: AppConfig) -> WhisperEngine:
        vocab = parse_vocabulary(cfg.custom_vocabulary)
        effective_prompt = build_whisper_initial_prompt(cfg.whisper_initial_prompt, vocab)
        return WhisperEngine(
            model_name=cfg.whisper_model,
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
            beam_size=cfg.whisper_beam_size,
            initial_prompt=effective_prompt,
        )

    def _build_polisher(self, cfg: AppConfig) -> OllamaPolisher | EmbeddedLlamaPolisher:
        vocab = parse_vocabulary(cfg.custom_vocabulary)
        if cfg.llm_backend == "embedded":
            return EmbeddedLlamaPolisher(
                model_id=cfg.embedded_llm_model_id,
                custom_model_path=cfg.embedded_llm_custom_path,
                system_prompt=cfg.llm_system_prompt,
                timeout_sec=cfg.ollama_timeout_sec,
                keep_alive=cfg.ollama_keep_alive,
                num_ctx=cfg.ollama_num_ctx,
                num_gpu=cfg.ollama_num_gpu,
                vocabulary=vocab,
                lang=cfg.language,
            )
        return OllamaPolisher(
            base_url=cfg.ollama_url,
            model=cfg.ollama_model,
            system_prompt=cfg.llm_system_prompt,
            timeout_sec=cfg.ollama_timeout_sec,
            keep_alive=cfg.ollama_keep_alive,
            num_ctx=cfg.ollama_num_ctx,
            num_gpu=cfg.ollama_num_gpu,
            vocabulary=vocab,
            lang=cfg.language,
        )

    def _connect_signals(self) -> None:
        self._hotkey.toggle_requested.connect(self._on_toggle)
        self._hotkey.command_requested.connect(self._on_command_toggle)
        self._hotkey.cancel_requested.connect(self._on_cancel)

        self._asr_loaded.connect(self._on_asr_loaded)
        self._asr_failed.connect(self._on_asr_failed)
        self._asr_download_progress.connect(
            self._on_asr_download_progress, Qt.ConnectionType.QueuedConnection
        )
        self._stage_polishing.connect(self._on_stage_polishing)
        self._stage_injecting.connect(self._on_stage_injecting)
        self._pipeline_done.connect(self._on_pipeline_done)
        self._pipeline_empty.connect(self._on_pipeline_empty)
        self._pipeline_error.connect(self._on_pipeline_error)
        self._profile_ready.connect(self._apply_fetched_profile)
        self._partial_text_ready.connect(
            self._on_partial_text_ready, Qt.ConnectionType.QueuedConnection
        )
        self._auto_stop_requested.connect(
            self._on_auto_stop, Qt.ConnectionType.QueuedConnection
        )
        self._ollama_start_status.connect(
            self._on_ollama_start_status, Qt.ConnectionType.QueuedConnection
        )

        self._tray.open_settings_requested.connect(self._settings.show_and_focus)
        self._tray.toggle_llm_requested.connect(self._on_toggle_llm_from_tray)
        self._tray.profile_selected.connect(self._on_profile_selected)
        self._tray.mic_selected.connect(self._on_mic_selected)
        self._tray.toggle_idle_bar_requested.connect(self._on_toggle_idle_bar)
        self._tray.snooze_1h_requested.connect(self._on_snooze_1h)
        self._tray.unsnooze_requested.connect(self._on_unsnooze)
        self._tray.reset_position_requested.connect(self._on_reset_pill_position)
        self._tray.copy_last_dictation_requested.connect(self._on_copy_last_dictation)
        self._tray.copy_last_raw_requested.connect(self._on_copy_last_raw)
        self._tray.meeting_toggle_requested.connect(self._on_toggle_meeting_default)
        self._tray.quit_requested.connect(self.shutdown)

        self._pill.position_changed.connect(self._on_pill_moved)
        self._pill.toggle_requested.connect(self._on_toggle)
        self._pill.cancel_requested.connect(self._on_cancel)
        self._pill.copy_last_requested.connect(self._on_copy_last_dictation)
        self._pill.download_clicked.connect(self._open_first_run_setup)
        self._pill.meeting_clicked.connect(self._open_active_or_hub_meeting)
        self._pill.context_menu_requested.connect(self._on_pill_context_menu)

        nt = self._settings.notetaker_tab
        mw = self._meeting_window.workspace
        nt.start_meeting_requested.connect(self.start_meeting)
        nt.stop_meeting_requested.connect(self.stop_meeting)
        nt.resume_meeting_requested.connect(self.resume_meeting)
        nt.regenerate_summary_requested.connect(self._run_meeting_summary_async)
        nt.open_side_window_requested.connect(
            lambda mid: self._meeting_window.open_for_meeting(
                mid, recording=(self._meeting_session.is_active and self._meeting_session.active_meeting is not None and self._meeting_session.active_meeting.id == mid)
            )
        )
        nt.meeting_updated.connect(self._sync_side_window_from_hub)

        mw.start_meeting_requested.connect(self.start_meeting)
        mw.stop_meeting_requested.connect(self.stop_meeting)
        mw.resume_meeting_requested.connect(self.resume_meeting)
        mw.regenerate_summary_requested.connect(self._run_meeting_summary_async)
        mw.meeting_updated.connect(self._sync_hub_from_side_window)
        mw.delete_meeting_requested.connect(lambda _mid: nt.refresh_meetings())
        self._meeting_window.back_to_hub_requested.connect(self._settings.show_notetaker)

        self._meeting_stopped_done.connect(self._on_meeting_stopped_ui)
        self._meeting_summary_step.connect(self._on_meeting_summary_step_ui, Qt.ConnectionType.QueuedConnection)
        self._meeting_summary_done.connect(self._on_meeting_summary_done_ui, Qt.ConnectionType.QueuedConnection)
        self._meeting_warning_signal.connect(
            lambda msg: self._pill.show_notice(msg[:48], is_error=True, duration_ms=2600)
        )

        self._settings.config_saved.connect(self._on_config_saved)
        self._settings.history_retention_changed.connect(self._on_history_retention_changed)
        self._settings.reset_pill_position_requested.connect(self._on_reset_pill_position)
        self._settings.theme_preview_requested.connect(self._on_theme_preview)
        self._settings.language_preview_requested.connect(self._on_language_preview)
        self._settings.llm_download_progress_changed.connect(self._on_llm_download_progress)
        self._settings.llm_download_ended.connect(self._on_llm_download_ended)
        try:
            self._app.styleHints().colorSchemeChanged.connect(self._on_system_theme_changed)
        except Exception:
            pass

    def _on_history_retention_changed(self, cfg: AppConfig) -> None:
        self._config = self._config.model_copy(
            update={
                "history_auto_delete_enabled": cfg.history_auto_delete_enabled,
                "history_auto_delete_value": cfg.history_auto_delete_value,
                "history_auto_delete_unit": cfg.history_auto_delete_unit,
                "history_hard_delete": cfg.history_hard_delete,
            }
        )
        self._run_history_auto_cleanup()

    def _run_history_auto_cleanup(self) -> None:
        if not self._config.history_auto_delete_enabled:
            return
        try:
            removed = self._storage.apply_auto_cleanup(
                enabled=self._config.history_auto_delete_enabled,
                value=self._config.history_auto_delete_value,
                unit=self._config.history_auto_delete_unit,
                hard=self._config.history_hard_delete,
            )
            if removed > 0 and self._settings.isVisible():
                self._settings.refresh_history()
                if self._config.history_hard_delete:
                    self._settings.refresh_insights()
        except Exception:
            logger.exception("Failed to run automatic history cleanup")

    def _load_asr_async(self) -> None:
        self._asr_load_gen += 1
        gen = self._asr_load_gen
        asr_ref = self._asr
        model_name = self._config.whisper_model
        dl_lang = self._lang
        self._asr_status_kind = "loading"
        self._asr_status_obj = None
        self._asr_load_started_at = time.monotonic()
        self._refresh_asr_status_label()

        needs_download = find_local_whisper_model_path(model_name) is None
        if needs_download:
            self._asr_downloading = True
            self._asr_download_pct = 0
            dest_dir = get_whisper_model_dir(model_name)
            label = f"Whisper {model_name}"
            self._pill.show_downloading(label, 0)
            self._settings.set_external_download_progress(
                label,
                DownloadProgress(stage=label, downloaded_bytes=0, total_bytes=0),
                dest_dir,
            )
            try:
                msg_key = (
                    "onboarding.tray_whisper_portable"
                    if is_portable_mode()
                    else "onboarding.tray_whisper_installed"
                )
                self._tray.showMessage(
                    tr("onboarding.tray_title", dl_lang),
                    tr(msg_key, dl_lang, model=model_name, path=str(get_app_data_dir())),
                    self._tray.MessageIcon.Information,
                    6000,
                )
            except Exception:
                pass
        else:
            self._asr_downloading = False
            self._asr_download_pct = -1

        def _work() -> None:
            try:
                try:
                    status = asr_ref.load(
                        on_progress=lambda prog: self._asr_download_progress.emit(gen, prog),
                        lang=dl_lang,
                    )
                except TypeError:
                    status = asr_ref.load()
                self._asr_loaded.emit(gen, status)
            except Exception as exc:
                self._asr_failed.emit(gen, str(exc))

        self._pool.start(_FnWorker(_work))

    def _on_asr_download_progress(self, gen: int, progress: object) -> None:
        if gen != self._asr_load_gen or not isinstance(progress, DownloadProgress):
            return
        self._asr_downloading = True
        pct = progress.percent if progress.total_bytes > 0 else -1
        self._asr_download_pct = pct
        label = f"Whisper {self._config.whisper_model}"
        self._settings.set_external_download_progress(
            label, progress, get_whisper_model_dir(self._config.whisper_model)
        )
        if self._state == AppState.IDLE:
            self._pill.show_downloading(label, pct)

    def _on_llm_download_progress(self, label: str, percent: int) -> None:
        if self._state == AppState.IDLE and not self._asr_downloading:
            self._pill.show_downloading(label, percent)

    def _on_llm_download_ended(self) -> None:
        if self._state == AppState.IDLE and not self._asr_downloading:
            self._restore_pill_idle()

    def _update_warming_pill(self) -> None:
        if self._shutting_down or self._asr.status.ready or self._asr_downloading:
            self._warming_timer.stop()
            return
        if self._state != AppState.IDLE:
            return
        elapsed = max(0.0, time.monotonic() - self._asr_load_started_at)
        remaining = max(0.0, float(self._config.last_asr_load_sec) - elapsed)
        sec = int(math.ceil(remaining))
        if sec >= 1:
            base = tr("pill.warming_queued" if self._start_recording_when_ready else "pill.warming", self._lang)
            text = tr("pill.warming_sec", self._lang, label=base, sec=sec)
        else:
            text = tr("pill.warming_almost", self._lang)
        self._pill.show_warming(text, queued=self._start_recording_when_ready)

    def _on_asr_loaded(self, gen: int, status: ASRStatus) -> None:
        if gen != self._asr_load_gen:
            return
        was_downloading = self._asr_downloading
        queued_start = self._start_recording_when_ready
        was_warming = self._warming_timer.isActive()
        self._warming_timer.stop()
        self._start_recording_when_ready = False

        if self._asr_load_started_at > 0 and not was_downloading:
            measured = max(1.5, min(30.0, time.monotonic() - self._asr_load_started_at))
            if abs(measured - self._config.last_asr_load_sec) >= 0.5:
                self._config = self._config.model_copy(update={"last_asr_load_sec": round(measured, 1)})
                try:
                    save_config(self._config)
                except Exception:
                    pass

        self._asr_downloading = False
        self._asr_download_pct = -1
        self._asr_load_started_at = 0.0
        self._ollama_start_prompted = False
        self._settings.clear_external_download()
        self._asr_status_kind = "ready"
        self._asr_status_obj = status
        self._refresh_asr_status_label()
        if self._config.llm_enabled and self._config.ollama_warmup_on_start:
            self._pool.start(_FnWorker(self._polisher.warmup))

        if queued_start and self._state == AppState.IDLE:
            self._hotkey.set_recording_active(False)
            self._start_recording()
            return

        if was_downloading and self._state == AppState.IDLE:
            self._pill.show_notice(tr("onboarding.pill_ready", self._lang), duration_ms=3000)
        elif was_warming and self._state == AppState.IDLE:
            self._pill.show_notice(tr("pill.ready", self._lang), duration_ms=1400)

    def _on_asr_failed(self, gen: int, err: str) -> None:
        if gen != self._asr_load_gen:
            return
        self._warming_timer.stop()
        self._start_recording_when_ready = False
        self._hotkey.set_recording_active(False)
        self._asr_downloading = False
        self._asr_download_pct = -1
        self._asr_load_started_at = 0.0
        self._ollama_start_prompted = False
        self._settings.clear_external_download()
        self._asr_status_kind = "error"
        self._asr_status_obj = None
        self._refresh_asr_status_label()
        self._pill.show_notice(
            tr("notice.asr_error", self._lang, err=err[:42]),
            is_error=True,
            duration_ms=3500,
        )

    def _should_show_idle_bar(self) -> bool:
        return bool(self._config.show_idle_bar) and not self._snooze_timer.isActive()

    def _restore_pill_idle(self) -> None:
        show_idle = self._should_show_idle_bar()
        self._pill.set_idle_bar_enabled(show_idle)
        if self._state != AppState.IDLE:
            return
        if self._meeting_session.is_active:
            self._pill.show_meeting(self._meeting_recorder.elapsed_sec)
            return
        if show_idle:
            self._pill.show_idle_bar()
        else:
            self._pill.hide_pill()

    def _on_meeting_pill_tick(self) -> None:
        if self._meeting_session.is_active and self._state == AppState.IDLE:
            self._pill.show_meeting(self._meeting_recorder.elapsed_sec)

    def _on_toggle_meeting_default(self) -> None:
        if self._meeting_session.is_active:
            self.stop_meeting()
        else:
            self.start_meeting(self._config.notetaker_audio_mode)

    def start_meeting(self, audio_mode: str = "dual") -> None:
        if self._meeting_session.is_active:
            return
        if not self._asr.status.ready:
            self._pill.show_notice(tr("notice.model_loading", self._lang), is_error=True, duration_ms=2000)
            return
        mode: AudioMode = "mic_only" if audio_mode == "mic_only" else "dual"
        self._meeting_session.whisper_engine = self._asr
        self._meeting_session.vocabulary = parse_vocabulary(self._config.custom_vocabulary)
        self._meeting_session.whisper_prompt = self._config.whisper_initial_prompt
        self._meeting_session.diarization_enabled = self._config.notetaker_diarization_enabled
        self._meeting_session.lang = self._lang
        try:
            meeting = self._meeting_session.start(
                audio_mode=mode,
                mic_device_index=self._config.audio_device_index,
                loopback_device_index=self._config.notetaker_loopback_device_index,
            )
        except Exception as exc:
            logger.exception("Failed to start meeting")
            self._pill.show_notice(
                tr("notice.mic_error", self._lang, err=str(exc)[:30]),
                is_error=True,
                duration_ms=2500,
            )
            return
        self._tray.set_meeting_active(True)
        self._settings.notetaker_tab.set_recording_state(True, meeting.id)
        self._meeting_window.open_for_meeting(meeting.id, recording=True)
        self._meeting_pill_timer.start()
        self._restore_pill_idle()

    def resume_meeting(self, meeting_id: int) -> None:
        if self._meeting_session.is_active:
            return
        if not self._asr.status.ready:
            self._pill.show_notice(tr("notice.model_loading", self._lang), is_error=True, duration_ms=2000)
            return
        self._meeting_session.whisper_engine = self._asr
        self._meeting_session.vocabulary = parse_vocabulary(self._config.custom_vocabulary)
        self._meeting_session.whisper_prompt = self._config.whisper_initial_prompt
        self._meeting_session.diarization_enabled = self._config.notetaker_diarization_enabled
        self._meeting_session.lang = self._lang
        try:
            meeting = self._meeting_session.resume(
                meeting_id,
                mic_device_index=self._config.audio_device_index,
                loopback_device_index=self._config.notetaker_loopback_device_index,
            )
        except Exception as exc:
            logger.exception("Failed to resume meeting")
            self._pill.show_notice(
                tr("notice.mic_error", self._lang, err=str(exc)[:30]),
                is_error=True,
                duration_ms=2500,
            )
            return
        if meeting is None:
            return
        self._tray.set_meeting_active(True)
        self._settings.notetaker_tab.set_recording_state(True, meeting.id)
        self._meeting_window.open_for_meeting(meeting.id, recording=True)
        self._meeting_pill_timer.start()
        self._restore_pill_idle()

    def _open_active_or_hub_meeting(self) -> None:
        active = self._meeting_session.active_meeting
        if active is not None:
            self._meeting_window.open_for_meeting(active.id, recording=True)
        else:
            self._settings.show_notetaker()

    def _sync_hub_from_side_window(self, mid: int) -> None:
        nt = self._settings.notetaker_tab
        if nt.workspace.current_meeting_id == mid:
            nt.workspace.set_meeting(mid)
        nt._refresh_list_labels_only()

    def _sync_side_window_from_hub(self, mid: int) -> None:
        mw = self._meeting_window.workspace
        if mw.current_meeting_id == mid:
            mw.set_meeting(mid)

    def stop_meeting(self) -> None:
        if not self._meeting_session.is_active:
            return
        active = self._meeting_session.active_meeting
        mid = active.id if active is not None else None
        self._meeting_pill_timer.stop()
        self._tray.set_meeting_active(False)
        self._settings.notetaker_tab.set_recording_state(False, None)
        self._meeting_window.workspace.set_recording_state(False, None)
        if mid is not None:
            self._settings.notetaker_tab.notify_summary_step(mid, "refine", 1, 2)
            self._meeting_window.workspace.set_summary_step(mid, "refine", 1, 2)
        self._restore_pill_idle()

        def _worker() -> None:
            rec = self._meeting_session.stop()
            if rec is not None:
                self._meeting_stopped_done.emit(rec.id)

        self._pool.start(_FnWorker(_worker))

    def _on_meeting_stopped_ui(self, meeting_id: int) -> None:
        self._restore_pill_idle()
        self._settings.notetaker_tab.refresh_meetings(select_id=meeting_id)
        self._meeting_window.workspace.set_meeting(meeting_id)
        if self._config.llm_enabled:
            self._run_meeting_summary_async(meeting_id, force=False)
        else:
            self._meeting_summary_done.emit(meeting_id, False, "")

    def _run_meeting_summary_async(self, meeting_id: int, force: bool = False) -> None:
        if not self._config.llm_enabled:
            self._meeting_summary_done.emit(meeting_id, False, tr("notetaker.llm_unavailable", self._lang))
            return
        target_lang = self._lang
        polisher = self._polisher
        storage = self._meeting_storage
        self._meeting_storage.update_meeting(meeting_id, status="summarizing")
        self._meeting_summary_step.emit(meeting_id, "analyze", 1, 2)

        def _worker() -> None:
            try:
                summarizer = MeetingSummarizer(polisher)  # type: ignore[arg-type]
                res = summarizer.summarize_meeting(
                    storage,
                    meeting_id,
                    lang=target_lang,
                    force=force,
                    on_step=lambda stage, step, total: self._meeting_summary_step.emit(
                        meeting_id, stage, step, total
                    ),
                )
                self._meeting_summary_done.emit(meeting_id, res.too_short, "")
            except Exception as exc:
                storage.update_meeting(meeting_id, status="error")
                self._meeting_summary_done.emit(meeting_id, False, str(exc))

        threading.Thread(target=_worker, name="LocalFlow-MeetingSummary", daemon=True).start()

    def _on_meeting_summary_step_ui(self, meeting_id: int, stage: str, step: int, total: int) -> None:
        self._settings.notetaker_tab.notify_summary_step(meeting_id, stage, step, total)
        self._meeting_window.workspace.set_summary_step(meeting_id, stage, step, total)

    def _on_meeting_summary_done_ui(self, meeting_id: int, too_short: bool, err: str) -> None:
        self._settings.notetaker_tab.notify_summary_done(meeting_id, too_short=too_short, error=err)
        self._meeting_window.workspace.set_summary_done(meeting_id, too_short=too_short, error=err)

    def _set_state(self, state: AppState) -> None:
        self._state = state
        self._tray.set_state_icon(state.value)
        self._hotkey.set_recording_active(state == AppState.RECORDING)

    def _safe_get_active_window_context(self) -> ActiveWindowContext | None:
        try:
            return get_active_window_context()
        except Exception:
            logger.exception("Failed to read active window context")
            return None

    def _on_partial_text_ready(self, session_gen: int, text: str) -> None:
        if session_gen != self._session_gen:
            return
        if self._state != AppState.RECORDING or self._is_command_mode:
            return
        if self._config.live_preview_enabled:
            self._pill.set_preview_text(text)

    def _on_auto_stop(self, session_gen: int) -> None:
        if session_gen != self._session_gen:
            return
        if self._state == AppState.RECORDING:
            self._stop_and_process()

    def _on_copy_last_dictation(self) -> None:
        text = self._last_dictation_text
        if not text:
            recent = self._storage.list_recent(limit=1)
            if recent:
                text = recent[0].final_text
                self._last_dictation_text = text
        if not text:
            self._pill.show_notice(tr("notice.no_last_dictation", self._lang), duration_ms=1600)
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self._pill.show_notice(tr("notice.last_copied", self._lang), duration_ms=1600)

    def _on_copy_last_raw(self) -> None:
        text = self._last_raw_text
        if not text:
            recent = self._storage.list_recent(limit=1)
            if recent:
                text = recent[0].raw_text
                self._last_raw_text = text
        if not text:
            self._pill.show_notice(tr("notice.no_last_dictation", self._lang), duration_ms=1600)
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self._pill.show_notice(tr("notice.raw_copied", self._lang), duration_ms=1600)

    def _on_toggle(self) -> None:
        if self._state == AppState.IDLE:
            self._start_recording()
        elif self._state == AppState.RECORDING:
            self._stop_and_process()

    def _prompt_start_ollama(self, *, can_launch: bool) -> str:
        """Show a topmost modal prompt asking the user to start Ollama."""
        box = QMessageBox()
        box.setWindowTitle(tr("ollama_prompt.title", self._lang))
        box.setText(
            tr(
                "ollama_prompt.body",
                self._lang,
                model=self._config.ollama_model,
                url=self._config.ollama_url,
            )
        )
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowFlags(box.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        start_btn = (
            box.addButton(tr("ollama_prompt.btn_start", self._lang), QMessageBox.ButtonRole.AcceptRole)
            if can_launch
            else None
        )
        continue_btn = box.addButton(
            tr("ollama_prompt.btn_continue", self._lang), QMessageBox.ButtonRole.ActionRole
        )
        box.addButton(tr("ollama_prompt.btn_cancel", self._lang), QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if start_btn is not None and clicked == start_btn:
            return "start"
        if clicked == continue_btn:
            return "continue"
        return "cancel"

    def _on_ollama_start_status(self, ready_ok: bool, msg: str) -> None:
        if self._state == AppState.IDLE:
            self._pill.show_notice(msg, is_error=not ready_ok, duration_ms=2600)

    def _check_ollama_ready_on_first_dictation(self) -> bool:
        """On first dictation with Ollama selected, prompt to start Ollama if unreachable."""
        if self._ollama_start_prompted:
            return True
        if not self._config.llm_enabled or self._config.llm_backend != "ollama":
            return True
        if not self._config.ollama_model.strip():
            return True
        if not hasattr(self._polisher, "is_reachable"):
            return True

        if self._polisher.is_reachable(timeout=0.5):
            self._ollama_start_prompted = True
            return True

        launch_cmd = (
            find_ollama_launch_command()
            if is_local_ollama_url(self._config.ollama_url)
            else None
        )
        choice = self._prompt_start_ollama(can_launch=launch_cmd is not None)
        if choice == "continue":
            self._ollama_start_prompted = True
            return True
        if choice != "start" or launch_cmd is None:
            return False

        self._ollama_start_prompted = True
        if not launch_ollama_background(launch_cmd):
            self._pill.show_notice(
                tr("notice.ollama_start_failed", self._lang), is_error=True, duration_ms=2600
            )
            return False

        self._pill.show_notice(tr("notice.ollama_starting", self._lang), duration_ms=6000)
        polisher = self._polisher
        lang = self._lang

        def _wait_ready() -> None:
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline and not self._shutting_down:
                if polisher.is_reachable(timeout=0.5):
                    try:
                        polisher.warmup()
                    except Exception:
                        pass
                    self._ollama_start_status.emit(True, tr("notice.ollama_ready", lang))
                    return
                time.sleep(0.5)
            if not self._shutting_down:
                self._ollama_start_status.emit(False, tr("notice.ollama_start_failed", lang))

        self._pool.start(_FnWorker(_wait_ready))
        return False

    def _on_command_toggle(self) -> None:
        if self._state != AppState.IDLE:
            return
        if not self._asr.status.ready:
            if self._asr_downloading:
                self._pill.show_downloading(
                    f"Whisper {self._config.whisper_model}", self._asr_download_pct
                )
            else:
                self._pill.show_notice(
                    tr("notice.model_loading", self._lang), is_error=True, duration_ms=1800
                )
            return
        if not self._check_ollama_ready_on_first_dictation():
            return

        window_ctx = self._safe_get_active_window_context()
        selected = self._injector.capture_selected_text()
        if not selected or not selected.strip():
            logger.info(
                "Command mode aborted: no selected text captured (app=%s)",
                window_ctx.app_name if window_ctx is not None else "unknown",
            )
            self._pill.show_notice(tr("notice.select_text_cmd", self._lang), duration_ms=2000)
            return

        self._session_gen += 1
        session_gen = self._session_gen
        silence_sec = (
            self._config.silence_auto_stop_sec if self._config.silence_auto_stop_enabled else None
        )
        on_timeout = (
            (lambda gen=session_gen: self._auto_stop_requested.emit(gen))
            if silence_sec is not None
            else None
        )

        try:
            if self._meeting_session.is_active:
                self._meeting_recorder.start_dictation_tap()
                self._dictation_using_meeting_tap = True
            else:
                self._dictation_using_meeting_tap = False
                self._recorder.start(
                    device_index=self._config.audio_device_index,
                    silence_auto_stop_sec=silence_sec,
                    on_silence_timeout=on_timeout,
                )
        except Exception as exc:
            logger.exception("Microphone start error")
            self._pill.show_notice(
                tr("notice.mic_error", self._lang, err=str(exc)[:30]),
                is_error=True,
                duration_ms=2500,
            )
            return

        self._chunked_session = None
        self._pending_window_context = window_ctx
        self._pending_selected_text = selected
        self._is_command_mode = True
        self._set_state(AppState.RECORDING)
        self._pill.set_preview_text("")
        self._pill.show_recording(command_mode=True)
        if self._config.play_sounds:
            play_cue("start")
        self._pool.start(_FnWorker(self._polisher.warmup))

    def _on_cancel(self) -> None:
        if self._start_recording_when_ready and self._state == AppState.IDLE:
            self._start_recording_when_ready = False
            self._warming_timer.stop()
            self._hotkey.set_recording_active(False)
            self._restore_pill_idle()
            return
        if self._state != AppState.RECORDING:
            return
        self._session_gen += 1
        session = self._chunked_session
        self._chunked_session = None
        self._pending_window_context = None
        self._pending_selected_text = None
        self._is_command_mode = False
        if session is not None:
            session.cancel()
        if self._dictation_using_meeting_tap:
            self._dictation_using_meeting_tap = False
            self._meeting_recorder.stop_dictation_tap()
        else:
            self._recorder.stop()
        self._set_state(AppState.IDLE)
        self._pill.show_notice(tr("notice.cancelled", self._lang), duration_ms=1100)
        if self._config.play_sounds:
            play_cue("cancel")

    def _start_recording(self) -> None:
        if not self._asr.status.ready:
            if self._asr_downloading:
                self._pill.show_downloading(
                    f"Whisper {self._config.whisper_model}", self._asr_download_pct
                )
            else:
                self._start_recording_when_ready = True
                self._hotkey.set_recording_active(True)
                self._warming_timer.start()
                self._update_warming_pill()
            return
        if not self._check_ollama_ready_on_first_dictation():
            return

        window_ctx = self._safe_get_active_window_context()
        self._session_gen += 1
        session_gen = self._session_gen
        session = ChunkedASRSession(
            self._asr,
            on_partial_text=lambda text, gen=session_gen: self._partial_text_ready.emit(gen, text),
        )
        silence_sec = (
            self._config.silence_auto_stop_sec if self._config.silence_auto_stop_enabled else None
        )
        on_timeout = (
            (lambda gen=session_gen: self._auto_stop_requested.emit(gen))
            if silence_sec is not None
            else None
        )

        try:
            if self._meeting_session.is_active:
                self._meeting_recorder.start_dictation_tap()
                self._dictation_using_meeting_tap = True
                session.cancel()
                session = None
            else:
                self._dictation_using_meeting_tap = False
                self._recorder.start(
                    device_index=self._config.audio_device_index,
                    pause_ms=750,
                    silence_auto_stop_sec=silence_sec,
                    on_segment=session.submit,
                    on_silence_timeout=on_timeout,
                )
        except Exception as exc:
            if session is not None:
                session.cancel()
            logger.exception("Microphone start error")
            self._pill.show_notice(
                tr("notice.mic_error", self._lang, err=str(exc)[:30]),
                is_error=True,
                duration_ms=2500,
            )
            return

        self._chunked_session = session
        self._pending_window_context = window_ctx
        self._pending_selected_text = None
        self._is_command_mode = False
        self._set_state(AppState.RECORDING)
        self._pill.set_preview_text("")
        self._pill.show_recording()
        if self._config.play_sounds:
            play_cue("start")
        if self._config.llm_enabled:
            self._pool.start(_FnWorker(self._polisher.warmup))

    def _stop_and_process(self) -> None:
        if self._state != AppState.RECORDING:
            return
        self._session_gen += 1

        session = self._chunked_session
        window_ctx = self._pending_window_context
        selected_text = self._pending_selected_text
        is_command = self._is_command_mode

        self._chunked_session = None
        self._pending_window_context = None
        self._pending_selected_text = None
        self._is_command_mode = False

        full_audio = np.zeros(0, dtype=np.float32)
        if self._dictation_using_meeting_tap:
            self._dictation_using_meeting_tap = False
            audio = self._meeting_recorder.stop_dictation_tap()
            duration_sec = float(audio.size) / 16000.0
            if session is not None:
                session.cancel()
                session = None
        elif session is not None:
            if hasattr(self._recorder, "stop_chunked_with_full_audio"):
                audio, full_audio, duration_sec = self._recorder.stop_chunked_with_full_audio()
            else:
                audio, duration_sec = self._recorder.stop_chunked()
        else:
            audio, duration_sec = self._recorder.stop()

        if self._config.play_sounds:
            play_cue("stop")

        if duration_sec < self._config.min_duration_sec or (session is None and audio.size == 0):
            if session is not None:
                session.cancel()
            self._set_state(AppState.IDLE)
            self._restore_pill_idle()
            return

        self._set_state(AppState.TRANSCRIBING)
        self._pill.show_transcribing()

        cfg = self._config
        lang = self._lang
        asr = self._asr
        polisher = self._polisher
        injector = self._injector
        storage = self._storage

        def _work() -> None:
            try:
                if session is not None:
                    use_full_gpu_pass = (
                        asr.status.device == "cuda"
                        and full_audio.size > 0
                        and duration_sec <= 60.0
                    )
                    if use_full_gpu_pass:
                        session.cancel()
                        raw_text = asr.transcribe(full_audio)
                    else:
                        raw_text = session.finalize(audio)
                        if (not raw_text or not raw_text.strip()) and session.errors:
                            self._pipeline_error.emit(str(session.errors[0]))
                            return
                elif is_command:
                    raw_text = asr.transcribe(audio, initial_prompt=COMMAND_WHISPER_PROMPT)
                else:
                    raw_text = asr.transcribe(audio)

                if not raw_text or not raw_text.strip():
                    msg = (
                        tr("notice.cmd_not_recognized", lang)
                        if is_command
                        else tr("notice.speech_not_recognized", lang)
                    )
                    self._pipeline_empty.emit(msg)
                    return

                if is_command:
                    self._stage_polishing.emit(tr("pill.command", lang))
                    res = polisher.transform_selection(
                        selected_text or "",
                        raw_text,
                        window_context=window_ctx,
                    )
                    if not res.used_llm or not res.text or not res.text.strip():
                        self._pipeline_error.emit(
                            res.error or tr("notice.llm_cmd_unavailable", lang)
                        )
                        return
                    final_text = res.text
                    self._stage_injecting.emit()
                    injector.inject(
                        final_text,
                        auto_space=False,
                        restore_clipboard=cfg.paste_restore_clipboard,
                        paste_delay_ms=cfg.paste_delay_ms,
                    )
                    storage.add_entry(
                        duration_sec=duration_sec,
                        raw_text=raw_text,
                        final_text=final_text,
                        llm_model=self._active_llm_label(cfg),
                        used_llm=True,
                        app_category=window_ctx.category if window_ctx is not None else "general",
                        app_name=window_ctx.app_name if window_ctx is not None else "",
                        is_command=True,
                        dict_fixes=0,
                        style_profile="command",
                    )
                    self._pipeline_done.emit(False, "", final_text, raw_text)
                    return

                final_text = raw_text
                used_llm = False
                llm_err = ""
                vocab = parse_vocabulary(cfg.custom_vocabulary)
                dict_fixes = 0
                active_style = resolve_style_profile(
                    window_ctx,
                    auto_style_by_app=cfg.auto_style_by_app,
                    category_styles=cfg.category_styles,
                    app_style_overrides=cfg.app_style_overrides,
                    custom_instructions=cfg.style_custom_instructions,
                    messenger_strip_trailing_period=cfg.messenger_strip_trailing_period,
                    lang=lang,
                )
                effective_style_key = active_style.key if cfg.llm_enabled else "raw"

                if cfg.llm_enabled and active_style.use_llm and not should_bypass_llm(raw_text, active_style):
                    self._stage_polishing.emit(active_style.pill_label)
                    res = polisher.polish(
                        raw_text,
                        window_context=window_ctx,
                        style_profile=active_style,
                    )
                    final_text = res.text
                    used_llm = res.used_llm
                    llm_err = res.error or ""
                    if vocab.replacements:
                        _post, dict_fixes = apply_vocabulary_replacements_with_count(
                            raw_text, vocab.replacements
                        )
                    if vocab.terms and final_text:
                        low_raw = raw_text.casefold()
                        low_final = final_text.casefold()
                        for term in vocab.terms:
                            t_low = term.casefold()
                            if t_low in low_final and term not in raw_text and t_low in low_raw:
                                dict_fixes += 1
                else:
                    if vocab.replacements:
                        replaced, dict_fixes = apply_vocabulary_replacements_with_count(
                            raw_text, vocab.replacements
                        )
                        if replaced.strip() and replaced != raw_text:
                            final_text = replaced
                    if cfg.llm_enabled and active_style.use_llm:
                        final_text = apply_style_postprocessing(final_text, active_style)
                        final_text = strip_long_dashes(final_text)

                final_text = strip_final_period(final_text)
                is_term = bool(
                    window_ctx is not None
                    and window_ctx.app_name in ("Terminal", "PowerShell", "Command Prompt")
                )
                self._stage_injecting.emit()
                try:
                    injector.inject(
                        final_text,
                        auto_space=True,
                        restore_clipboard=cfg.paste_restore_clipboard,
                        paste_delay_ms=cfg.paste_delay_ms,
                        is_terminal=is_term,
                        smart_inline_case=cfg.smart_inline_case_enabled,
                        preserve_words=vocab.terms,
                    )
                except TypeError:
                    injector.inject(
                        final_text,
                        auto_space=True,
                        restore_clipboard=cfg.paste_restore_clipboard,
                        paste_delay_ms=cfg.paste_delay_ms,
                    )
                storage.add_entry(
                    duration_sec=duration_sec,
                    raw_text=raw_text,
                    final_text=final_text,
                    llm_model=self._active_llm_label(cfg) if used_llm else None,
                    used_llm=used_llm,
                    app_category=window_ctx.category if window_ctx is not None else "general",
                    app_name=window_ctx.app_name if window_ctx is not None else "",
                    is_command=False,
                    dict_fixes=dict_fixes,
                    style_profile=effective_style_key,
                )
                self._pipeline_done.emit(
                    bool(llm_err and cfg.llm_enabled and active_style.use_llm),
                    llm_err,
                    final_text,
                    raw_text,
                )
            except Exception as exc:
                if session is not None:
                    session.cancel()
                logger.exception("Pipeline error")
                self._pipeline_error.emit(str(exc))

        self._pool.start(_FnWorker(_work))

    def _on_stage_polishing(self, label: str = "") -> None:
        if self._state == AppState.TRANSCRIBING:
            self._set_state(AppState.POLISHING)
            self._pill.show_polishing(label or tr("pill.polishing", self._lang))

    def _on_stage_injecting(self) -> None:
        self._pill.hide_pill()

    def _on_pipeline_empty(self, msg: str) -> None:
        self._set_state(AppState.IDLE)
        self._pill.show_notice(msg, duration_ms=1400)

    def _on_pipeline_error(self, err: str) -> None:
        self._set_state(AppState.IDLE)
        self._pill.show_notice(
            tr("notice.error", self._lang, err=err[:36]), is_error=True, duration_ms=2600
        )

    def _on_pipeline_done(
        self, llm_fallback: bool, llm_err: str, final_text: str, raw_text: str = ""
    ) -> None:
        self._set_state(AppState.IDLE)
        self._run_history_auto_cleanup()
        if final_text:
            self._last_dictation_text = final_text
            self._tray.set_last_dictation_preview(final_text)
            self._pill.set_has_last_dictation(True)
        if raw_text:
            self._last_raw_text = raw_text
            self._tray.set_last_raw_preview(raw_text)
        if llm_fallback:
            logger.warning("LLM polish fallback: %s", llm_err)
            if self._config.llm_backend == "embedded":
                low_err = llm_err.lower()
                is_missing = any(
                    tok in low_err
                    for tok in ("не загружена", "не установлен", "not downloaded", "not installed")
                )
                msg = (
                    tr("notice.download_llm_raw", self._lang)
                    if is_missing
                    else tr("notice.embedded_llm_raw", self._lang)
                )
            else:
                msg = tr("notice.ollama_raw", self._lang)
            self._pill.show_notice(msg, duration_ms=2200)
        else:
            self._restore_pill_idle()

    def _on_pill_context_menu(self, global_pos: QPoint) -> None:
        state = QuickMenuState(
            asr_status=self._asr_status_text,
            selected_mic_index=self._config.audio_device_index,
            profile=self._config.performance_profile,
            llm_enabled=self._config.llm_enabled,
            live_preview_enabled=self._config.live_preview_enabled,
            show_idle_bar=self._config.show_idle_bar,
            is_snoozed=self._snooze_timer.isActive(),
            last_dictation_preview=self._last_dictation_text,
            last_raw_preview=self._last_raw_text,
            is_meeting_active=self._meeting_session.is_active,
        )
        callbacks = QuickMenuCallbacks(
            on_select_mic=self._on_mic_selected,
            on_select_profile=self._on_profile_selected,
            on_toggle_llm=self._on_toggle_llm_from_tray,
            on_toggle_live_preview=self._on_toggle_live_preview,
            on_toggle_idle_bar=self._on_toggle_idle_bar,
            on_snooze_1h=self._on_snooze_1h,
            on_unsnooze=self._on_unsnooze,
            on_reset_position=self._on_reset_pill_position,
            on_copy_last_dictation=self._on_copy_last_dictation,
            on_copy_last_raw=self._on_copy_last_raw,
            on_open_settings=self._settings.show_and_focus,
            on_toggle_meeting=self._on_toggle_meeting_default,
            on_quit=self.shutdown,
        )
        menu = build_quick_menu(
            state,
            callbacks,
            parent=self._pill,
            theme=resolve_theme(self._config.theme),
            lang=self._lang,
        )
        self._pill.set_context_menu_open(True)
        try:
            menu.exec(global_pos)
        finally:
            self._pill.set_context_menu_open(False)

    def _on_mic_selected(self, device_index: object) -> None:
        idx = device_index if isinstance(device_index, int) else None
        self._config = self._config.model_copy(update={"audio_device_index": idx})
        save_config(self._config)
        self._tray.set_selected_mic(idx)
        self._settings.sync_from_config(self._config)
        mic_label = tr("notice.mic_system_short", self._lang)
        if idx is not None:
            try:
                for dev in list_input_devices():
                    if dev.index == idx:
                        mic_label = dev.name[:24]
                        break
            except Exception:
                mic_label = f"#{idx}"
        self._pill.show_notice(
            tr("notice.mic_switched", self._lang, name=mic_label), duration_ms=1600
        )

    def _on_toggle_live_preview(self, enabled: bool) -> None:
        self._config = self._config.model_copy(update={"live_preview_enabled": bool(enabled)})
        save_config(self._config)
        self._pill.set_live_preview_enabled(bool(enabled))
        self._settings.sync_from_config(self._config)

    def _on_toggle_idle_bar(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and self._snooze_timer.isActive():
            self._snooze_timer.stop()
            self._tray.set_snoozed(False)
        self._config = self._config.model_copy(update={"show_idle_bar": enabled})
        save_config(self._config)
        self._tray.set_idle_bar_checked(enabled)
        self._settings.sync_from_config(self._config)
        self._restore_pill_idle()

    def _on_snooze_1h(self) -> None:
        self._snooze_timer.start(3600 * 1000)
        self._tray.set_snoozed(True)
        self._restore_pill_idle()

    def _on_unsnooze(self) -> None:
        self._snooze_timer.stop()
        self._tray.set_snoozed(False)
        self._restore_pill_idle()

    def _on_reset_pill_position(self) -> None:
        self._config = self._config.model_copy(update={"pill_x": None, "pill_y": None})
        save_config(self._config)
        self._pill.reset_position()
        self._restore_pill_idle()

    def _on_toggle_llm_from_tray(self, enabled: bool) -> None:
        if enabled:
            self._ollama_start_prompted = False
        self._config = self._config.model_copy(update={"llm_enabled": enabled})
        save_config(self._config)
        self._tray.set_llm_checked(enabled)
        self._settings.sync_from_config(self._config)

    def _on_pill_moved(self, x: int, y: int) -> None:
        self._config = self._config.model_copy(update={"pill_x": x, "pill_y": y})
        save_config(self._config)

    def _on_profile_selected(self, profile: str) -> None:
        polisher = self._polisher

        def _fetch() -> None:
            models: list[str] | None = None
            try:
                models = polisher.list_models()
            except Exception:
                pass
            self._profile_ready.emit(profile, models)

        self._pool.start(_FnWorker(_fetch))

    def _apply_fetched_profile(self, profile: str, models: object) -> None:
        avail = models if isinstance(models, list) else None
        new_cfg = apply_performance_profile(self._config, profile, available_ollama_models=avail)  # type: ignore[arg-type]
        self._apply_config(new_cfg)
        self._settings.sync_from_config(new_cfg)
        self._pill.show_notice(
            tr("notice.mode_switched", self._lang, profile=profile), duration_ms=1600
        )

    def _on_config_saved(self, new_cfg: AppConfig) -> None:
        self._apply_config(new_cfg)

    def _apply_config(self, new_cfg: AppConfig) -> None:
        old_cfg = self._config
        old_polisher = self._polisher

        whisper_changed = (
            new_cfg.whisper_model != old_cfg.whisper_model
            or new_cfg.whisper_device != old_cfg.whisper_device
            or new_cfg.whisper_compute_type != old_cfg.whisper_compute_type
            or new_cfg.whisper_beam_size != old_cfg.whisper_beam_size
        )
        should_evict_llm = (
            new_cfg.ollama_num_gpu == 0
            or new_cfg.llm_backend != old_cfg.llm_backend
            or new_cfg.ollama_model != old_cfg.ollama_model
            or new_cfg.embedded_llm_model_id != old_cfg.embedded_llm_model_id
            or new_cfg.embedded_llm_custom_path != old_cfg.embedded_llm_custom_path
            or not new_cfg.llm_enabled
        )

        self._config = new_cfg
        if (
            (new_cfg.llm_enabled and not old_cfg.llm_enabled)
            or new_cfg.llm_backend != old_cfg.llm_backend
            or new_cfg.ollama_url != old_cfg.ollama_url
        ):
            self._ollama_start_prompted = False
        save_config(new_cfg)
        if new_cfg.autostart_enabled != old_cfg.autostart_enabled:
            set_autostart_enabled(new_cfg.autostart_enabled)
        self._hotkey.set_threshold_ms(new_cfg.double_tap_ms)
        self._pill.set_live_preview_enabled(new_cfg.live_preview_enabled)
        self._polisher = self._build_polisher(new_cfg)
        self._tray.set_llm_checked(new_cfg.llm_enabled)
        self._tray.set_profile_checked(new_cfg.performance_profile)
        self._tray.set_selected_mic(new_cfg.audio_device_index)
        self._tray.set_idle_bar_checked(new_cfg.show_idle_bar)
        self._apply_language(new_cfg.language)
        self._apply_resolved_theme(new_cfg.theme)
        self._restore_pill_idle()

        def _cleanup_old_polisher() -> None:
            if should_evict_llm:
                old_polisher.unload()
            old_polisher.close()

        self._pool.start(_FnWorker(_cleanup_old_polisher))

        if not should_evict_llm and new_cfg.llm_enabled and new_cfg.ollama_warmup_on_start:
            self._pool.start(_FnWorker(self._polisher.warmup))

        if whisper_changed:
            self._asr.unload()
            self._asr = self._build_asr(new_cfg)
            self._load_asr_async()
        else:
            vocab = parse_vocabulary(new_cfg.custom_vocabulary)
            effective_prompt = build_whisper_initial_prompt(new_cfg.whisper_initial_prompt, vocab)
            self._asr.set_initial_prompt(effective_prompt)

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        try:
            self._app.styleHints().colorSchemeChanged.disconnect(self._on_system_theme_changed)
        except Exception:
            pass
        self._session_gen += 1
        self._warming_timer.stop()
        self._start_recording_when_ready = False
        self._hotkey.stop()
        session = self._chunked_session
        self._chunked_session = None
        if session is not None:
            session.cancel()
        if self._recorder.is_recording:
            self._recorder.stop()
        if self._meeting_session.is_active:
            self._meeting_pill_timer.stop()
            self._meeting_session.stop()
        self._pending_window_context = None
        self._pending_selected_text = None
        self._is_command_mode = False
        self._polisher.close()
        self._tray.hide()
        self._pill.hide()
        self._meeting_window.hide()
        self._pool.waitForDone(1000)
        self._app.quit()
