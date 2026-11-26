from __future__ import annotations

from pathlib import Path
import wave

import pytest

from localflow.audio import sound_fx
from localflow.config import AppConfig, load_config, save_config
from localflow.storage import HistoryStorage
from localflow.ui.overlay_pill import FloatingPillWidget
from localflow.ui.settings_window import SettingsWindow
from localflow.ui.theme import build_hub_stylesheet, build_menu_stylesheet, resolve_theme


def test_sound_fx_writes_cached_wav_and_uses_filename_async(tmp_path: Path, monkeypatch) -> None:
    sound_fx._WAV_FILE_CACHE.clear()
    monkeypatch.setattr(sound_fx.tempfile, "gettempdir", lambda: str(tmp_path))

    played: list[tuple[object, int]] = []

    class _FakeWinsound:
        SND_FILENAME = 0x20000
        SND_ASYNC = 0x1
        SND_NODEFAULT = 0x2
        SND_MEMORY = 0x4

        @staticmethod
        def PlaySound(sound, flags):
            played.append((sound, flags))

    monkeypatch.setattr(sound_fx, "winsound", _FakeWinsound)

    for kind in ("start", "stop", "cancel"):
        sound_fx.play_cue(kind)  # type: ignore[arg-type]

    assert len(played) == 3
    for path_str, flags in played:
        assert isinstance(path_str, str)
        p = Path(path_str)
        assert p.is_file()
        assert flags & _FakeWinsound.SND_FILENAME
        assert flags & _FakeWinsound.SND_ASYNC
        assert not (flags & _FakeWinsound.SND_MEMORY)
        with wave.open(str(p), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 44100
            assert wf.getnframes() > 2000


def test_theme_resolution_and_stylesheets(monkeypatch) -> None:
    monkeypatch.setattr("localflow.ui.theme.detect_system_theme", lambda: "light")
    assert resolve_theme("system") == "light"
    assert resolve_theme("dark") == "dark"
    assert resolve_theme("light") == "light"

    monkeypatch.setattr("localflow.ui.theme.detect_system_theme", lambda: "dark")
    assert resolve_theme("system") == "dark"
    assert resolve_theme("light") == "light"

    from localflow.ui.theme import build_dialog_stylesheet

    assert "#f2f2f5" in build_hub_stylesheet("light")
    assert "#111113" in build_hub_stylesheet("dark")
    assert "#ffffff" in build_menu_stylesheet("light")
    assert "#18181b" in build_menu_stylesheet("dark")
    assert "#18181b" in build_dialog_stylesheet("dark")
    assert "#ffffff" in build_dialog_stylesheet("light")


def test_settings_window_theme_switch_roundtrip_and_sound_test_button(
    tmp_path: Path, monkeypatch, qtbot
) -> None:
    monkeypatch.setattr("localflow.ui.settings_window.list_input_devices", lambda: [])
    cues: list[str] = []
    monkeypatch.setattr("localflow.ui.settings_window.play_cue", lambda kind: cues.append(kind))

    storage = HistoryStorage(db_path=tmp_path / "hist.db")
    win = SettingsWindow(AppConfig(theme="dark"), storage)
    qtbot.addWidget(win)
    win._navigate_to("settings")
    win.show()

    assert win._resolved_theme == "dark"
    assert win._theme_combo.currentData() == "dark"

    # Clicking test sound button plays the start cue
    win._test_sound_btn.click()
    assert cues == ["start"]

    # Switching dropdown to light previews immediately and emits theme_preview_requested
    with qtbot.waitSignal(win.theme_preview_requested, timeout=1000) as preview_blocker:
        win._theme_combo.setCurrentIndex(win._theme_combo.findData("light"))
    assert preview_blocker.args == ["light"]
    assert win._resolved_theme == "light"
    assert win._insights_tab._resolved_theme == "light"
    assert "#f2f2f5" in win.styleSheet()

    # Save & reload persists light theme
    with qtbot.waitSignal(win.config_saved, timeout=1000) as save_blocker:
        win._save_btn.click()
    saved: AppConfig = save_blocker.args[0]
    assert saved.theme == "light"

    cfg_path = tmp_path / "config.json"
    save_config(saved, cfg_path)
    reloaded = load_config(cfg_path)
    assert reloaded.theme == "light"

    win2 = SettingsWindow(reloaded, storage)
    qtbot.addWidget(win2)
    assert win2._resolved_theme == "light"
    assert win2._theme_combo.currentData() == "light"


def test_overlay_pill_renders_both_light_and_dark_themes(qtbot) -> None:
    pill = FloatingPillWidget()
    qtbot.addWidget(pill)

    for theme in ("light", "dark"):
        pill.set_theme(theme)  # type: ignore[arg-type]
        pill.show_idle_bar()
        pill.grab()

        pill._hovered = True
        pill._hover_progress = 1.0
        pill._apply_idle_geometry()
        pill.grab()

        pill.show_recording(command_mode=False)
        pill.set_preview_text("Привет светлая тема")
        pill.grab()

        pill.show_transcribing()
        pill.grab()

        pill.show_polishing("Живой")
        pill.grab()

        pill.show_notice("Скопировано")
        pill.grab()

    pill.hide_pill()
