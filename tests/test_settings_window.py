from __future__ import annotations

from dataclasses import dataclass

import pytest
from PySide6.QtWidgets import QScrollArea

from localflow.config import AppConfig, load_config, save_config
from localflow.storage import HistoryStorage
from localflow.ui.settings_window import SettingsWindow
from localflow.vocabulary import parse_vocabulary


@dataclass(frozen=True)
class _FakeDevice:
    index: int
    name: str
    is_default: bool = False


@pytest.fixture
def make_settings(tmp_path, monkeypatch, qtbot):
    monkeypatch.setattr(
        "localflow.ui.settings_window.list_input_devices",
        lambda: [
            _FakeDevice(index=1, name="Mic One", is_default=True),
            _FakeDevice(index=2, name="USB Mic", is_default=False),
        ],
    )
    storage = HistoryStorage(db_path=tmp_path / "history.db")
    created: list[SettingsWindow] = []

    def _factory(config: AppConfig | None = None) -> SettingsWindow:
        win = SettingsWindow(config or AppConfig(), storage)
        qtbot.addWidget(win)
        created.append(win)
        return win

    yield _factory

    for win in created:
        win.close()


def test_separate_pages_scroll_areas_and_pinned_save_bar(make_settings, qtbot) -> None:
    win = make_settings()
    win.show()

    # History & Insights hide the config save footer
    win._navigate_to("history")
    assert win._tabs.currentWidget() is win._history_tab
    assert win._config_footer.isVisible() is False

    win._navigate_to("insights")
    assert win._tabs.currentWidget() is win._insights_tab
    assert win._config_footer.isVisible() is False

    # Dictionary, Style & Settings are separate stacked pages with own scroll areas
    assert len({win._vocab_tab, win._style_tab, win._settings_tab}) == 3

    win._navigate_to("vocab")
    assert win._tabs.currentWidget() is win._vocab_tab
    vocab_scroll = win._vocab_tab.findChild(QScrollArea)
    assert vocab_scroll is not None and vocab_scroll.widgetResizable() is True
    assert vocab_scroll.widget().isAncestorOf(win._vocab_edit)
    assert not win._settings_tab.isAncestorOf(win._vocab_edit)
    assert not vocab_scroll.isAncestorOf(win._save_btn)
    assert win._save_btn.isVisible()

    win._navigate_to("style")
    assert win._tabs.currentWidget() is win._style_tab
    style_scroll = win._style_tab.findChild(QScrollArea)
    assert style_scroll is not None and style_scroll.widgetResizable() is True
    assert style_scroll.widget().isAncestorOf(win._styles_card)
    assert style_scroll.widget().isAncestorOf(win._base_prompt_card)
    assert not style_scroll.widget().isAncestorOf(win._llm_card)
    assert not win._settings_tab.isAncestorOf(win._styles_card)
    assert not style_scroll.isAncestorOf(win._save_btn)
    assert win._save_btn.isVisible()

    win._navigate_to("settings")
    assert win._tabs.currentWidget() is win._settings_tab
    scroll = win._settings_tab.findChild(QScrollArea)
    assert scroll is not None
    assert scroll.widgetResizable() is True
    container = scroll.widget()
    assert container is not None

    for widget in (
        win._llm_card,
        win._theme_combo,
        win._live_preview_check,
        win._silence_auto_stop_check,
        win._silence_auto_stop_spin,
        win._sounds_check,
        win._test_sound_btn,
    ):
        assert container.isAncestorOf(widget)

    # Sidebar splits Profile & Hotkeys into two separate cards with all shortcuts
    assert win._profile_status_card is not win._hotkeys_card
    assert win._sidebar.isAncestorOf(win._profile_status_card)
    assert win._sidebar.isAncestorOf(win._hotkeys_card)
    from PySide6.QtWidgets import QLabel

    hk_texts = [lbl.text() for lbl in win._hotkeys_card.findChildren(QLabel)]
    assert any("HOTKEYS" in t for t in hk_texts)
    assert any("Start" in t for t in hk_texts)
    assert any("Stop" in t for t in hk_texts)
    assert any("Cancel" in t for t in hk_texts)
    assert any("Command" in t for t in hk_texts)

    assert not scroll.isAncestorOf(win._save_btn)
    assert not scroll.isAncestorOf(win._save_status)

    win.resize(740, 360)
    qtbot.waitUntil(lambda: scroll.verticalScrollBar().maximum() > 0, timeout=1000)
    assert win._save_btn.isVisible()


def test_clear_history_requires_confirmation(make_settings, monkeypatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    win = make_settings()
    win.refresh_history()
    assert win._clear_history_btn.isEnabled() is False

    confirm_calls: list[int] = []
    monkeypatch.setattr(win, "_confirm_clear_history", lambda n: confirm_calls.append(n) or True)

    # Empty history does not prompt
    win._clear_history()
    assert confirm_calls == []

    # Add two entries
    win._storage.add_entry(
        duration_sec=2.0,
        raw_text="раз",
        final_text="Раз",
        llm_model="gemma4:e4b",
        used_llm=True,
    )
    win._storage.add_entry(
        duration_sec=3.0,
        raw_text="два",
        final_text="Два",
        llm_model="gemma4:e4b",
        used_llm=True,
    )
    win.refresh_history()
    assert win._clear_history_btn.isEnabled() is True
    assert win._history_list.count() == 2

    # Cancel keeps all entries
    monkeypatch.setattr(win, "_confirm_clear_history", lambda n: confirm_calls.append(n) or False)
    win._clear_history_btn.click()
    assert confirm_calls == [2]
    assert win._history_list.count() == 2
    assert len(win._storage.list_all()) == 2

    # Confirm clears all entries and disables button
    monkeypatch.setattr(win, "_confirm_clear_history", lambda n: confirm_calls.append(n) or True)
    win._clear_history_btn.click()
    assert confirm_calls == [2, 2]
    assert win._history_list.count() == 0
    assert win._storage.list_all() == []
    assert win._clear_history_btn.isEnabled() is False

    # Verify QMessageBox dialog itself defaults to Cancel and distinguishes Cancel vs Clear
    monkeypatch.undo()
    observed_default: list[str] = []

    def _fake_exec_cancel(box: QMessageBox) -> int:
        default_btn = box.defaultButton()
        assert default_btn is not None
        observed_default.append(default_btn.text())
        default_btn.click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", _fake_exec_cancel)
    assert win._confirm_clear_history(5) is False
    assert observed_default == ["Cancel"]

    def _fake_exec_confirm(box: QMessageBox) -> int:
        clear_btn = next(b for b in box.buttons() if b.text() == "Clear")
        clear_btn.click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", _fake_exec_confirm)
    assert win._confirm_clear_history(5) is True


def test_selective_single_and_multi_delete_history_entries(make_settings, monkeypatch) -> None:
    from PySide6.QtWidgets import QAbstractItemView

    win = make_settings()
    assert win._history_list.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection

    id1 = win._storage.add_entry(
        duration_sec=1.5,
        raw_text="первая запись",
        final_text="Первая запись",
        llm_model="gemma4:e4b",
        used_llm=True,
    )
    id2 = win._storage.add_entry(
        duration_sec=2.5,
        raw_text="вторая команда",
        final_text="Вторая команда",
        llm_model="gemma4:e4b",
        used_llm=True,
        is_command=True,
        style_profile="command",
    )
    id3 = win._storage.add_entry(
        duration_sec=3.5,
        raw_text="третья запись",
        final_text="Третья запись",
        llm_model=None,
        used_llm=False,
    )
    win.refresh_history()
    assert win._history_list.count() == 3
    # Row 0 is automatically selected on refresh -> single selection enabled
    assert win._delete_selected_btn.isEnabled() is True
    assert win._delete_selected_btn.text() == "Delete"

    # Select 2 items (row 0 = id3, row 2 = id1)
    win._history_list.clearSelection()
    win._history_list.item(0).setSelected(True)
    win._history_list.item(2).setSelected(True)
    assert win._delete_selected_btn.isEnabled() is True
    assert win._delete_selected_btn.text() == "Delete (2)"
    assert win._inspector_title.text() == "Selected entries: 2"

    # Cancel keeps all 3 entries
    confirm_counts: list[int] = []
    monkeypatch.setattr(
        win, "_confirm_delete_entries", lambda n: confirm_counts.append(n) or False
    )
    win._delete_selected_btn.click()
    assert confirm_counts == [2]
    assert len(win._storage.list_all()) == 3

    # Confirm removes only id1 & id3, preserving id2 (the command entry)
    monkeypatch.setattr(
        win, "_confirm_delete_entries", lambda n: confirm_counts.append(n) or True
    )
    win._delete_selected_btn.click()
    assert confirm_counts == [2, 2]
    remaining = win._storage.list_all()
    assert [e.id for e in remaining] == [id2]
    assert win._history_list.count() == 1
    assert win._delete_selected_btn.text() == "Delete"

    # Delete remaining single entry
    win._history_list.item(0).setSelected(True)
    win._delete_selected_btn.click()
    assert confirm_counts == [2, 2, 1]
    assert win._storage.list_all() == []
    assert win._history_list.count() == 0
    assert win._delete_selected_btn.isEnabled() is False
    assert win._clear_history_btn.isEnabled() is False


def test_initial_values_from_default_config(make_settings) -> None:
    win = make_settings(AppConfig())
    assert win._vocab_edit.toPlainText() == ""
    assert win._vocab_edit.placeholderText() == "Antigravity, Wispr Flow, Ollama, PySide6\ncurse -> cursor"
    assert win._live_preview_check.isChecked() is True
    assert win._silence_auto_stop_check.isChecked() is False
    assert win._silence_auto_stop_spin.isEnabled() is False
    assert win._silence_auto_stop_spin.value() == pytest.approx(2.5)
    assert win._silence_auto_stop_spin.minimum() == pytest.approx(0.5)
    assert win._silence_auto_stop_spin.maximum() == pytest.approx(15.0)
    assert win._silence_auto_stop_spin.singleStep() == pytest.approx(0.5)
    assert win._silence_auto_stop_spin.decimals() == 1
    assert win._silence_auto_stop_spin.suffix() == " s"


def test_silence_auto_stop_toggle_enables_disables_spinbox(make_settings, qtbot) -> None:
    win = make_settings(AppConfig())
    assert win._silence_auto_stop_spin.isEnabled() is False

    win._silence_auto_stop_check.setChecked(True)
    assert win._silence_auto_stop_spin.isEnabled() is True

    win._silence_auto_stop_check.setChecked(False)
    assert win._silence_auto_stop_spin.isEnabled() is False

    win_enabled = make_settings(
        AppConfig(silence_auto_stop_enabled=True, silence_auto_stop_sec=6.0)
    )
    assert win_enabled._silence_auto_stop_check.isChecked() is True
    assert win_enabled._silence_auto_stop_spin.isEnabled() is True
    assert win_enabled._silence_auto_stop_spin.value() == pytest.approx(6.0)


def test_vocabulary_multiline_and_replacements_roundtrip(make_settings, tmp_path, qtbot) -> None:
    win = make_settings(AppConfig())
    raw_input = "  Antigravity, Wispr Flow\nкурс -> курсор\nLocalFlow → localflow  "
    win._vocab_edit.setPlainText(raw_input)

    with qtbot.waitSignal(win.config_saved, timeout=1000) as blocker:
        win._save_btn.click()

    saved_cfg: AppConfig = blocker.args[0]
    expected_vocab = "Antigravity, Wispr Flow\nкурс -> курсор\nLocalFlow → localflow"
    assert saved_cfg.custom_vocabulary == expected_vocab

    cfg_path = tmp_path / "config.json"
    save_config(saved_cfg, cfg_path)
    reloaded = load_config(cfg_path)
    assert reloaded.custom_vocabulary == expected_vocab

    win2 = make_settings(reloaded)
    assert win2._vocab_edit.toPlainText() == expected_vocab
    parsed = parse_vocabulary(win2._vocab_edit.toPlainText())
    assert "Antigravity" in parsed.terms
    assert "Wispr Flow" in parsed.terms
    assert ("курс", "курсор") in parsed.replacements
    assert ("LocalFlow", "localflow") in parsed.replacements


def test_live_preview_and_silence_auto_stop_save_load_roundtrip(
    make_settings, tmp_path, qtbot
) -> None:
    cfg_path = tmp_path / "config.json"

    for val in (4.5, 0.5, 15.0):
        win = make_settings(AppConfig())
        win._live_preview_check.setChecked(False)
        win._silence_auto_stop_check.setChecked(True)
        win._silence_auto_stop_spin.setValue(val)

        with qtbot.waitSignal(win.config_saved, timeout=1000) as blocker:
            win._save_btn.click()

        emitted: AppConfig = blocker.args[0]
        assert emitted.live_preview_enabled is False
        assert emitted.silence_auto_stop_enabled is True
        assert emitted.silence_auto_stop_sec == pytest.approx(val)

        save_config(emitted, cfg_path)
        reloaded = load_config(cfg_path)
        win2 = make_settings(reloaded)
        assert win2._live_preview_check.isChecked() is False
        assert win2._silence_auto_stop_check.isChecked() is True
        assert win2._silence_auto_stop_spin.isEnabled() is True
        assert win2._silence_auto_stop_spin.value() == pytest.approx(val)


def test_sync_from_config_updates_all_fields(make_settings) -> None:
    win = make_settings(AppConfig())
    custom = AppConfig(
        performance_profile="quality",
        audio_device_index=2,
        double_tap_ms=450,
        whisper_model="large-v3",
        whisper_device="cpu",
        whisper_compute_type="int8",
        llm_enabled=False,
        ollama_url="http://localhost:11435",
        ollama_model="gemma4:e4b",
        ollama_keep_alive="30m",
        llm_system_prompt="Custom prompt",
        custom_vocabulary="PySide6\nтест -> test",
        live_preview_enabled=False,
        silence_auto_stop_enabled=True,
        silence_auto_stop_sec=7.5,
        auto_space_before_paste=False,
        paste_restore_clipboard=False,
        play_sounds=False,
    )

    win.sync_from_config(custom)

    assert win._profile_combo.currentData() == "quality"
    assert win._mic_combo.currentData() == 2
    assert win._double_tap_spin.value() == 450
    assert win._whisper_combo.currentText() == "large-v3"
    assert win._whisper_dev_combo.currentText() == "cpu"
    assert win._whisper_ct_combo.currentText() == "int8"
    assert win._llm_check.isChecked() is False
    assert win._ollama_url_edit.text() == "http://localhost:11435"
    assert win._ollama_model_combo.isEditable() is False
    assert win._embedded_model_combo.isEditable() is False
    assert win._ollama_model_combo.currentText() == "gemma4:e4b"
    assert win._keep_alive_combo.currentText() == "30m"
    assert win._prompt_edit.toPlainText() == "Custom prompt"
    assert win._vocab_edit.toPlainText() == "PySide6\nтест -> test"
    assert win._live_preview_check.isChecked() is False
    assert win._silence_auto_stop_check.isChecked() is True
    assert win._silence_auto_stop_spin.value() == pytest.approx(7.5)
    assert win._silence_auto_stop_spin.isEnabled() is True
    assert not hasattr(win, "_auto_space_check")
    assert win._restore_cb_check.isChecked() is False
    assert win._sounds_check.isChecked() is False
    assert win._save_status.text() == ""

    reset_cfg = AppConfig(silence_auto_stop_enabled=False, audio_device_index=None)
    win.sync_from_config(reset_cfg)
    assert win._silence_auto_stop_spin.isEnabled() is False
    assert win._mic_combo.currentIndex() == 0


def test_config_saved_signal_and_status_message(make_settings, qtbot) -> None:
    cfg = AppConfig(
        performance_profile="light_gpu",
        custom_vocabulary="LocalFlow\nкурс -> курсор",
        live_preview_enabled=False,
        silence_auto_stop_enabled=True,
        silence_auto_stop_sec=3.5,
    )
    win = make_settings(cfg)
    idx_quality = win._profile_combo.findData("quality")
    win._profile_combo.setCurrentIndex(idx_quality)

    with qtbot.waitSignal(win.config_saved, timeout=1000) as blocker:
        win._save_btn.click()

    emitted: AppConfig = blocker.args[0]
    assert win._config == emitted
    assert win._save_status.text() == "Settings saved."
    assert emitted.performance_profile == "quality"
    assert emitted.custom_vocabulary == "LocalFlow\nкурс -> курсор"
    assert emitted.live_preview_enabled is False
    assert emitted.silence_auto_stop_enabled is True
    assert emitted.silence_auto_stop_sec == pytest.approx(3.5)


def test_frameless_embedded_header_and_sidebar_vector_icons(make_settings, qtbot) -> None:
    from PySide6.QtCore import Qt

    win = make_settings(AppConfig(theme="dark"))
    assert bool(win.windowFlags() & Qt.WindowType.FramelessWindowHint)
    assert win._shell.objectName() == "appShell"
    assert not win._min_btn.icon().isNull()
    assert not win._max_btn.icon().isNull()
    assert not win._close_btn.icon().isNull()
    for btn, _name in win._nav_items:
        assert not btn.icon().isNull()

    win.show()
    qtbot.waitUntil(win.isVisible, timeout=2000)
    win._toggle_maximize()
    assert win.isMaximized()
    assert win._shell.property("windowMaximized") is True
    win._toggle_maximize()
    assert not win.isMaximized()
    assert win._shell.property("windowMaximized") is False

