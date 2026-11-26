from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QMenu

from localflow.config import AppConfig
from localflow.storage import HistoryStorage
from localflow.ui.context_menu import (
    QuickMenuCallbacks,
    QuickMenuState,
    build_quick_menu,
    populate_microphone_menu,
)
from localflow.ui.settings_window import SettingsWindow


@dataclass(frozen=True)
class _FakeDevice:
    index: int
    name: str
    max_input_channels: int = 1
    is_default: bool = False


def test_populate_microphone_menu_checks_selected_and_fires_callback(qtbot) -> None:
    menu = QMenu()
    selected: list[int | None] = []
    devices = [
        _FakeDevice(index=1, name="Built-in Mic", is_default=True),
        _FakeDevice(index=4, name="USB Headset", is_default=False),
    ]
    populate_microphone_menu(menu, selected_index=4, on_select=selected.append, devices=devices)

    actions = [a for a in menu.actions() if not a.isSeparator()]
    assert [a.data() for a in actions] == [None, 1, 4]
    assert actions[0].isChecked() is False
    assert actions[1].isChecked() is False
    assert actions[2].isChecked() is True

    actions[1].trigger()
    assert selected == [1]


def test_build_quick_menu_exposes_mic_vram_flow_bar_and_triggers_callbacks(qtbot) -> None:
    events: list[tuple[str, object]] = []
    state = QuickMenuState(
        asr_status="ASR: large-v3-turbo (cuda/int8_float16)",
        selected_mic_index=2,
        profile="light_gpu",
        llm_enabled=True,
        live_preview_enabled=True,
        show_idle_bar=True,
        is_snoozed=False,
        last_dictation_preview="тестовая диктовка",
    )
    callbacks = QuickMenuCallbacks(
        on_select_mic=lambda idx: events.append(("mic", idx)),
        on_select_profile=lambda p: events.append(("profile", p)),
        on_toggle_llm=lambda v: events.append(("llm", v)),
        on_toggle_live_preview=lambda v: events.append(("preview", v)),
        on_toggle_idle_bar=lambda v: events.append(("idle_bar", v)),
        on_snooze_1h=lambda: events.append(("snooze", True)),
        on_unsnooze=lambda: events.append(("unsnooze", True)),
        on_reset_position=lambda: events.append(("reset_pos", True)),
        on_copy_last_dictation=lambda: events.append(("copy_last", True)),
        on_open_settings=lambda: events.append(("settings", True)),
        on_quit=lambda: events.append(("quit", True)),
    )
    devices = [_FakeDevice(index=2, name="Studio Mic", is_default=True)]
    menu = build_quick_menu(state, callbacks, devices=devices)

    top_texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert "ASR: large-v3-turbo (cuda/int8_float16)" in top_texts
    assert "Microphone" in top_texts
    assert "VRAM Mode" in top_texts
    assert "Flow Bar" in top_texts
    assert "Copy to clipboard" in top_texts
    assert "Settings" in top_texts

    # Russian locale menu check
    ru_menu = build_quick_menu(state, callbacks, devices=devices, lang="ru")
    ru_texts = [a.text() for a in ru_menu.actions() if not a.isSeparator()]
    assert "Скопировать в буфер" in ru_texts
    assert "Настройки" in ru_texts
    assert "Скопировать в буфер диктовку" not in ru_texts
    assert "Вернуть последнюю диктовку" not in ru_texts
    assert "История и настройки..." not in ru_texts

    light_menu = build_quick_menu(state, callbacks, devices=devices, theme="light")
    assert "#ffffff" in light_menu.styleSheet()

    # Trigger snooze & reset position from Flow Bar submenu
    bar_action = next(a for a in menu.actions() if a.text() == "Flow Bar")
    bar_menu = bar_action.menu()
    assert bar_menu is not None
    snooze_act = next(a for a in bar_menu.actions() if a.text() == "Hide for 1 hour")
    reset_act = next(a for a in bar_menu.actions() if a.text() == "Reset position")
    snooze_act.trigger()
    reset_act.trigger()
    assert ("snooze", True) in events
    assert ("reset_pos", True) in events


def test_settings_window_history_search_stats_and_show_idle_bar(tmp_path, monkeypatch, qtbot) -> None:
    monkeypatch.setattr(
        "localflow.ui.settings_window.list_input_devices",
        lambda: [_FakeDevice(index=1, name="Mic One", is_default=True)],
    )
    storage = HistoryStorage(db_path=tmp_path / "history.db")
    storage.add_entry(
        duration_sec=2.0,
        raw_text="привет мир",
        final_text="Привет мир",
        llm_model="gemma4:e4b",
        used_llm=True,
    )
    storage.add_entry(
        duration_sec=3.0,
        raw_text="ну типа деплой backend",
        final_text="Деплой backend сервиса",
        llm_model="gemma4:e4b",
        used_llm=True,
    )

    win = SettingsWindow(AppConfig(show_idle_bar=True), storage)
    qtbot.addWidget(win)
    win.refresh_history()

    assert win._history_list.count() == 2
    assert "Dictations: 2" in win._history_stats_label.text()
    assert "Words: 5" in win._history_stats_label.text()

    win._history_search.setText("backend")
    assert win._history_list.count() == 1
    win._history_search.setText("")
    assert win._history_list.count() == 2

    assert win._show_idle_bar_check.isChecked() is True
    win._show_idle_bar_check.setChecked(False)
    with qtbot.waitSignal(win.config_saved, timeout=1000) as blocker:
        win._save_btn.click()
    assert blocker.args[0].show_idle_bar is False
    win.close()


def test_build_pill_menu_matches_wispr_structure_and_callbacks(qtbot) -> None:
    from localflow.ui.context_menu import build_pill_menu

    events: list[tuple[str, object]] = []
    state = QuickMenuState(
        asr_status="ASR ready",
        selected_mic_index=1,
        profile="light_gpu",
        llm_enabled=True,
        live_preview_enabled=True,
        show_idle_bar=True,
        is_snoozed=False,
    )
    callbacks = QuickMenuCallbacks(
        on_select_mic=lambda idx: events.append(("mic", idx)),
        on_select_profile=lambda p: events.append(("profile", p)),
        on_toggle_llm=lambda v: events.append(("llm", v)),
        on_toggle_live_preview=lambda v: events.append(("preview", v)),
        on_toggle_idle_bar=lambda v: events.append(("idle_bar", v)),
        on_snooze_1h=lambda: events.append(("snooze", True)),
        on_unsnooze=lambda: events.append(("unsnooze", True)),
        on_reset_position=lambda: events.append(("reset_pos", True)),
        on_copy_last_dictation=lambda: events.append(("copy_last", True)),
        on_open_settings=lambda: events.append(("settings", True)),
        on_open_history=lambda: events.append(("history", True)),
        on_paste_last_dictation=lambda: events.append(("paste_last", True)),
        on_quit=lambda: events.append(("quit", True)),
    )
    devices = [_FakeDevice(index=1, name="Mic One", is_default=True)]
    menu = build_pill_menu(state, callbacks, devices=devices, theme="dark", lang="en")
    actions = [a for a in menu.actions() if not a.isSeparator()]
    assert [a.text() for a in actions] == [
        "Hide for 1 hour",
        "Reset position",
        "Settings",
        "Quit",
        "Microphone",
        "VRAM Mode",
        "Paste last transcript",
    ]
    assert all(not a.icon().isNull() for a in actions)
    assert "#141417" in menu.styleSheet()

    # Reset position sits directly above Settings
    actions[1].trigger()
    assert ("reset_pos", True) in events

    # Quit is directly under Settings and triggers on_quit
    actions[3].trigger()
    assert ("quit", True) in events

    # VRAM Mode submenu allows switching profiles
    mode_menu = actions[5].menu()
    assert mode_menu is not None
    quality_act = next(a for a in mode_menu.actions() if "Quality" in a.text())
    quality_act.trigger()
    assert ("profile", "quality") in events

    # Paste last transcript invokes immediate-paste callback
    actions[6].trigger()
    assert ("paste_last", True) in events

    light_menu = build_pill_menu(state, callbacks, devices=devices, theme="light", lang="en")
    assert "#ffffff" in light_menu.styleSheet()
