"""Shared themed quick context menu for the Flow Bar pill and tray."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu, QWidget

from localflow.audio.recorder import AudioDeviceInfo, list_input_devices
from localflow.i18n import DEFAULT_LANGUAGE, tr
from localflow.ui.theme import ResolvedTheme, build_menu_stylesheet


def get_profile_labels(lang: str | None = None) -> dict[str, str]:
    return {
        "quality": tr("profile.quality.short", lang),
        "light_gpu": tr("profile.light_gpu.short", lang),
        "zero_vram": tr("profile.zero_vram.short", lang),
    }


PROFILE_LABELS = get_profile_labels(DEFAULT_LANGUAGE)

DARK_MENU_STYLESHEET = build_menu_stylesheet("dark")


@dataclass(frozen=True, slots=True)
class QuickMenuState:
    asr_status: str
    selected_mic_index: int | None
    profile: str
    llm_enabled: bool
    live_preview_enabled: bool
    show_idle_bar: bool
    is_snoozed: bool
    last_dictation_preview: str = ""
    last_raw_preview: str = ""


@dataclass(frozen=True, slots=True)
class QuickMenuCallbacks:
    on_select_mic: Callable[[int | None], None]
    on_select_profile: Callable[[str], None]
    on_toggle_llm: Callable[[bool], None]
    on_toggle_live_preview: Callable[[bool], None]
    on_toggle_idle_bar: Callable[[bool], None]
    on_snooze_1h: Callable[[], None]
    on_unsnooze: Callable[[], None]
    on_reset_position: Callable[[], None]
    on_copy_last_dictation: Callable[[], None]
    on_open_settings: Callable[[], None]
    on_quit: Callable[[], None]
    on_copy_last_raw: Callable[[], None] | None = None


def populate_microphone_menu(
    menu: QMenu,
    selected_index: int | None,
    on_select: Callable[[int | None], None],
    devices: list[AudioDeviceInfo] | None = None,
    lang: str | None = None,
) -> QActionGroup:
    """Populate a microphone selection QMenu with checkable input devices."""
    menu.clear()
    group = QActionGroup(menu)
    group.setExclusive(True)

    default_act = QAction(tr("settings.mic_default", lang), menu)
    default_act.setCheckable(True)
    default_act.setChecked(selected_index is None)
    default_act.setData(None)
    default_act.triggered.connect(lambda checked: checked and on_select(None))
    group.addAction(default_act)
    menu.addAction(default_act)

    if devices is None:
        try:
            devices = list_input_devices()
        except Exception:
            devices = []

    if devices:
        menu.addSeparator()
        for dev in devices:
            suffix = tr("settings.mic_default_suffix", lang) if dev.is_default else ""
            act = QAction(f"{dev.name}{suffix}", menu)
            act.setCheckable(True)
            act.setChecked(selected_index == dev.index)
            act.setData(dev.index)
            act.triggered.connect(lambda checked, idx=dev.index: checked and on_select(idx))
            group.addAction(act)
            menu.addAction(act)

    return group


def build_quick_menu(
    state: QuickMenuState,
    callbacks: QuickMenuCallbacks,
    *,
    parent: QWidget | None = None,
    devices: list[AudioDeviceInfo] | None = None,
    theme: ResolvedTheme = "dark",
    lang: str | None = None,
) -> QMenu:
    """Build the themed quick context menu shown on right-clicking the Flow Bar pill."""
    menu_sheet = build_menu_stylesheet(theme)
    menu = QMenu(parent)
    menu.setStyleSheet(menu_sheet)

    status_act = QAction(state.asr_status or tr("menu.asr_ready", lang), menu)
    status_act.setEnabled(False)
    menu.addAction(status_act)
    menu.addSeparator()

    # Microphone submenu
    mic_menu = QMenu(tr("menu.microphone", lang), menu)
    mic_menu.setStyleSheet(menu_sheet)
    populate_microphone_menu(
        mic_menu,
        state.selected_mic_index,
        callbacks.on_select_mic,
        devices=devices,
        lang=lang,
    )
    menu.addMenu(mic_menu)

    # VRAM profile submenu
    profile_menu = QMenu(tr("menu.vram_mode", lang), menu)
    profile_menu.setStyleSheet(menu_sheet)
    profile_group = QActionGroup(profile_menu)
    profile_group.setExclusive(True)
    for key, label in get_profile_labels(lang).items():
        act = QAction(label, profile_menu)
        act.setCheckable(True)
        act.setChecked(key == state.profile)
        act.triggered.connect(lambda checked, k=key: checked and callbacks.on_select_profile(k))
        profile_group.addAction(act)
        profile_menu.addAction(act)
    menu.addMenu(profile_menu)

    menu.addSeparator()

    llm_act = QAction(tr("menu.smart_cleanup", lang), menu)
    llm_act.setCheckable(True)
    llm_act.setChecked(state.llm_enabled)
    llm_act.toggled.connect(callbacks.on_toggle_llm)
    menu.addAction(llm_act)

    preview_act = QAction(tr("menu.live_preview", lang), menu)
    preview_act.setCheckable(True)
    preview_act.setChecked(state.live_preview_enabled)
    preview_act.toggled.connect(callbacks.on_toggle_live_preview)
    menu.addAction(preview_act)

    # Flow Bar visibility submenu
    bar_menu = QMenu(tr("menu.flow_bar", lang), menu)
    bar_menu.setStyleSheet(menu_sheet)

    show_idle_act = QAction(tr("menu.show_idle_bar", lang), bar_menu)
    show_idle_act.setCheckable(True)
    show_idle_act.setChecked(state.show_idle_bar)
    show_idle_act.toggled.connect(callbacks.on_toggle_idle_bar)
    bar_menu.addAction(show_idle_act)

    if state.is_snoozed:
        unsnooze_act = QAction(tr("menu.show_bar_now", lang), bar_menu)
        unsnooze_act.triggered.connect(callbacks.on_unsnooze)
        bar_menu.addAction(unsnooze_act)
    else:
        snooze_act = QAction(tr("menu.snooze_1h", lang), bar_menu)
        snooze_act.triggered.connect(callbacks.on_snooze_1h)
        bar_menu.addAction(snooze_act)

    bar_menu.addSeparator()
    reset_pos_act = QAction(tr("menu.reset_pos", lang), bar_menu)
    reset_pos_act.triggered.connect(callbacks.on_reset_position)
    bar_menu.addAction(reset_pos_act)

    menu.addMenu(bar_menu)

    menu.addSeparator()

    copy_last_act = QAction(tr("menu.copy_to_clipboard", lang), menu)
    if state.last_dictation_preview:
        snippet = " ".join(state.last_dictation_preview.split())
        if len(snippet) > 48:
            snippet = snippet[:47] + "…"
        copy_last_act.setToolTip(tr("menu.copy_tooltip", lang, snippet=snippet))
    copy_last_act.triggered.connect(callbacks.on_copy_last_dictation)
    menu.addAction(copy_last_act)

    has_distinct_raw = bool(
        state.last_raw_preview.strip()
        and state.last_raw_preview.strip() != state.last_dictation_preview.strip()
    )
    if callbacks.on_copy_last_raw is not None and has_distinct_raw:
        copy_raw_act = QAction(tr("menu.copy_raw_asr", lang), menu)
        raw_snippet = " ".join(state.last_raw_preview.split())
        if len(raw_snippet) > 48:
            raw_snippet = raw_snippet[:47] + "…"
        copy_raw_act.setToolTip(tr("menu.copy_tooltip", lang, snippet=raw_snippet))
        copy_raw_act.triggered.connect(callbacks.on_copy_last_raw)
        menu.addAction(copy_raw_act)

    open_settings_act = QAction(tr("menu.settings", lang), menu)
    open_settings_act.triggered.connect(callbacks.on_open_settings)
    menu.addAction(open_settings_act)

    menu.addSeparator()

    quit_act = QAction(tr("menu.quit", lang), menu)
    quit_act.triggered.connect(callbacks.on_quit)
    menu.addAction(quit_act)

    return menu
