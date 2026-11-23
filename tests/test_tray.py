from __future__ import annotations

import pytest
from PySide6.QtGui import QAction

from localflow.ui.tray import LocalFlowTray


@pytest.fixture
def make_tray(qtbot):
    trays: list[LocalFlowTray] = []

    def _factory(llm_enabled: bool = True, profile: str = "light_gpu") -> LocalFlowTray:
        tray = LocalFlowTray(llm_enabled=llm_enabled, profile=profile)
        trays.append(tray)
        return tray

    yield _factory

    for tray in trays:
        tray.hide()
        tray.deleteLater()


def _find_action(tray: LocalFlowTray, text: str) -> QAction | None:
    menu = tray.contextMenu()
    assert menu is not None
    for act in menu.actions():
        if act.text() == text:
            return act
    return None


def test_tray_has_copy_last_dictation_action_enabled_by_default(make_tray) -> None:
    tray = make_tray()
    act = _find_action(tray, "Copy to clipboard")
    assert act is not None
    assert act.isEnabled() is True
    assert act.isCheckable() is False


def test_triggering_copy_last_dictation_emits_signal(make_tray, qtbot) -> None:
    tray = make_tray()
    act = _find_action(tray, "Copy to clipboard")
    assert act is not None

    with qtbot.waitSignal(tray.copy_last_dictation_requested, timeout=1000):
        act.trigger()


def test_optional_preview_helper_does_not_disable_action(make_tray) -> None:
    tray = make_tray()
    act = _find_action(tray, "Copy to clipboard")
    assert act is not None

    tray.set_last_dictation_preview(None)
    assert act.isEnabled() is True
    tray.set_last_dictation_preview("")
    assert act.isEnabled() is True
    tray.set_last_dictation_preview("Very long dictation text " * 5)
    assert act.isEnabled() is True


def test_existing_tray_actions_regression(make_tray, qtbot) -> None:
    tray = make_tray(llm_enabled=True, profile="light_gpu")

    open_act = _find_action(tray, "Settings")
    llm_act = _find_action(tray, "Smart cleanup (LLM)")
    quit_act = _find_action(tray, "Quit")
    profile_act = _find_action(tray, "VRAM Mode")

    assert open_act is not None
    assert llm_act is not None
    assert quit_act is not None
    assert profile_act is not None and profile_act.menu() is not None

    tray.set_theme("light")
    assert "#ffffff" in tray.contextMenu().styleSheet()
    tray.set_theme("dark")
    assert "#14171f" in tray.contextMenu().styleSheet()

    # Live Russian language switch updates tray actions immediately
    tray.set_language("ru")
    assert _find_action(tray, "Скопировать в буфер") is not None
    assert _find_action(tray, "Настройки") is not None
    tray.set_language("en")
    assert _find_action(tray, "Copy to clipboard") is not None

    with qtbot.waitSignal(tray.open_settings_requested, timeout=1000):
        open_act.trigger()

    with qtbot.waitSignal(tray.toggle_llm_requested, timeout=1000) as llm_blocker:
        llm_act.trigger()
    assert llm_blocker.args == [False]

    quality_act = next(
        a for a in profile_act.menu().actions() if "Quality" in a.text()
    )
    with qtbot.waitSignal(tray.profile_selected, timeout=1000) as prof_blocker:
        quality_act.trigger()
    assert prof_blocker.args == ["quality"]

    with qtbot.waitSignal(tray.quit_requested, timeout=1000):
        quit_act.trigger()
