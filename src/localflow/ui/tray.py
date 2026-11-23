"""System tray icon and menu for LocalFlow."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

from localflow.i18n import DEFAULT_LANGUAGE, LanguageCode, normalize_language, tr
from localflow.ui.context_menu import (
    get_profile_labels,
    populate_microphone_menu,
)
from localflow.ui.theme import ResolvedTheme, build_menu_stylesheet


def _build_tray_icon(badge_color: QColor | None = None) -> QIcon:
    """Build recognizable LocalFlow 5-bar waveform tray icon with an optional state badge."""
    pix = QPixmap(64, 64)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)

    # Rounded dark badge background matching the main window & .ico brand mark
    p.setBrush(QColor("#181a22"))
    p.drawRoundedRect(4, 4, 56, 56, 15, 15)

    # 5 vertical waveform bars
    p.setBrush(QColor("#fafafa"))
    heights = (16.0, 30.0, 22.0, 34.0, 14.0)
    for i, h in enumerate(heights):
        x = 13.0 + i * 8.4
        y = (64.0 - h) / 2.0
        p.drawRoundedRect(QRectF(x, y, 4.4, h), 2.2, 2.2)

    # Crisp bottom-right status dot for active recording / processing states
    if badge_color is not None:
        p.setBrush(QColor("#181a22"))
        p.drawEllipse(QRectF(39.0, 39.0, 23.0, 23.0))
        p.setBrush(badge_color)
        p.drawEllipse(QRectF(42.5, 42.5, 16.0, 16.0))

    p.end()
    return QIcon(pix)


class LocalFlowTray(QSystemTrayIcon):
    """System tray icon with quick toggles, microphone picker, VRAM profiles, and Flow Bar controls."""

    open_settings_requested = Signal()
    toggle_llm_requested = Signal(bool)
    profile_selected = Signal(str)
    mic_selected = Signal(object)  # int | None
    toggle_idle_bar_requested = Signal(bool)
    snooze_1h_requested = Signal()
    unsnooze_requested = Signal()
    reset_position_requested = Signal()
    copy_last_dictation_requested = Signal()
    quit_requested = Signal()

    def __init__(
        self,
        llm_enabled: bool,
        profile: str = "light_gpu",
        show_idle_bar: bool = True,
        selected_mic_index: int | None = None,
        theme: ResolvedTheme = "dark",
        lang: str = DEFAULT_LANGUAGE,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._icon_idle = _build_tray_icon(None)
        self._icon_recording = _build_tray_icon(QColor(239, 68, 68))
        self._icon_busy = _build_tray_icon(QColor(167, 139, 250))
        self._selected_mic_index = selected_mic_index
        self._is_snoozed = False
        self._mic_group: QActionGroup | None = None
        self._resolved_theme: ResolvedTheme = theme
        self._lang: LanguageCode = normalize_language(lang)
        self._last_preview_raw: str = ""

        self.setIcon(self._icon_idle)
        self.setToolTip(tr("tray.tooltip", self._lang))

        menu_sheet = build_menu_stylesheet(theme)
        menu = QMenu()
        menu.setStyleSheet(menu_sheet)
        self._menu = menu
        self._status_action = QAction(tr("menu.asr_loading_generic", self._lang), menu)
        self._status_action.setEnabled(False)
        menu.addAction(self._status_action)

        menu.addSeparator()

        # Microphone submenu (populated dynamically on menu open)
        self._mic_menu = QMenu(tr("menu.microphone", self._lang), menu)
        self._mic_menu.setStyleSheet(menu_sheet)
        self._mic_menu.aboutToShow.connect(self._refresh_mic_menu)
        menu.addMenu(self._mic_menu)

        # VRAM / Performance profile submenu
        self._profile_menu = QMenu(tr("menu.vram_mode", self._lang), menu)
        self._profile_menu.setStyleSheet(menu_sheet)
        self._profile_group = QActionGroup(self._profile_menu)
        self._profile_group.setExclusive(True)
        self._profile_actions: dict[str, QAction] = {}
        for key, label in get_profile_labels(self._lang).items():
            act = QAction(label, self._profile_menu)
            act.setCheckable(True)
            act.setChecked(key == profile)
            act.triggered.connect(lambda checked, k=key: checked and self.profile_selected.emit(k))
            self._profile_group.addAction(act)
            self._profile_menu.addAction(act)
            self._profile_actions[key] = act
        menu.addMenu(self._profile_menu)

        # Flow Bar submenu
        self._bar_menu = QMenu(tr("menu.flow_bar", self._lang), menu)
        self._bar_menu.setStyleSheet(menu_sheet)
        self._idle_bar_action = QAction(tr("menu.show_idle_bar", self._lang), self._bar_menu)
        self._idle_bar_action.setCheckable(True)
        self._idle_bar_action.setChecked(show_idle_bar)
        self._idle_bar_action.toggled.connect(self.toggle_idle_bar_requested.emit)
        self._bar_menu.addAction(self._idle_bar_action)

        self._snooze_action = QAction(tr("menu.snooze_1h", self._lang), self._bar_menu)
        self._snooze_action.triggered.connect(self._on_snooze_clicked)
        self._bar_menu.addAction(self._snooze_action)

        self._bar_menu.addSeparator()
        self._reset_pos_action = QAction(tr("menu.reset_pos", self._lang), self._bar_menu)
        self._reset_pos_action.triggered.connect(self.reset_position_requested.emit)
        self._bar_menu.addAction(self._reset_pos_action)
        menu.addMenu(self._bar_menu)

        self._llm_action = QAction(tr("menu.smart_cleanup", self._lang), menu)
        self._llm_action.setCheckable(True)
        self._llm_action.setChecked(llm_enabled)
        self._llm_action.toggled.connect(self.toggle_llm_requested.emit)
        menu.addAction(self._llm_action)

        self._copy_last_action = QAction(tr("menu.copy_to_clipboard", self._lang), menu)
        self._copy_last_action.triggered.connect(self.copy_last_dictation_requested.emit)
        menu.addAction(self._copy_last_action)

        self._open_action = QAction(tr("menu.settings", self._lang), menu)
        self._open_action.triggered.connect(self.open_settings_requested.emit)
        menu.addAction(self._open_action)

        menu.addSeparator()

        self._quit_action = QAction(tr("menu.quit", self._lang), menu)
        self._quit_action.triggered.connect(self.quit_requested.emit)
        menu.addAction(self._quit_action)

        self.setContextMenu(menu)
        self.activated.connect(self._on_activated)

    def set_language(self, lang: str) -> None:
        self._lang = normalize_language(lang)
        self.setToolTip(tr("tray.tooltip", self._lang))
        self._mic_menu.setTitle(tr("menu.microphone", self._lang))
        self._profile_menu.setTitle(tr("menu.vram_mode", self._lang))
        for key, label in get_profile_labels(self._lang).items():
            if key in self._profile_actions:
                self._profile_actions[key].setText(label)
        self._bar_menu.setTitle(tr("menu.flow_bar", self._lang))
        self._idle_bar_action.setText(tr("menu.show_idle_bar", self._lang))
        self._snooze_action.setText(
            tr("menu.show_bar_now", self._lang) if self._is_snoozed else tr("menu.snooze_1h", self._lang)
        )
        self._reset_pos_action.setText(tr("menu.reset_pos", self._lang))
        self._llm_action.setText(tr("menu.smart_cleanup", self._lang))
        self._copy_last_action.setText(tr("menu.copy_to_clipboard", self._lang))
        self._open_action.setText(tr("menu.settings", self._lang))
        self._quit_action.setText(tr("menu.quit", self._lang))
        self.set_last_dictation_preview(self._last_preview_raw)

    def set_theme(self, theme: ResolvedTheme) -> None:
        self._resolved_theme = theme
        sheet = build_menu_stylesheet(theme)
        for m in (self._menu, self._mic_menu, self._profile_menu, self._bar_menu):
            m.setStyleSheet(sheet)

    def _refresh_mic_menu(self) -> None:
        self._mic_group = populate_microphone_menu(
            self._mic_menu,
            self._selected_mic_index,
            lambda idx: self.mic_selected.emit(idx),
            lang=self._lang,
        )

    def _on_snooze_clicked(self) -> None:
        if self._is_snoozed:
            self.unsnooze_requested.emit()
        else:
            self.snooze_1h_requested.emit()

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.open_settings_requested.emit()

    def set_asr_status(self, text: str) -> None:
        self._status_action.setText(text)

    def set_llm_checked(self, enabled: bool) -> None:
        self._llm_action.blockSignals(True)
        self._llm_action.setChecked(enabled)
        self._llm_action.blockSignals(False)

    def set_profile_checked(self, profile: str) -> None:
        for key, act in self._profile_actions.items():
            act.blockSignals(True)
            act.setChecked(key == profile)
            act.blockSignals(False)

    def set_selected_mic(self, device_index: int | None) -> None:
        self._selected_mic_index = device_index

    def set_idle_bar_checked(self, enabled: bool) -> None:
        self._idle_bar_action.blockSignals(True)
        self._idle_bar_action.setChecked(enabled)
        self._idle_bar_action.blockSignals(False)

    def set_snoozed(self, snoozed: bool) -> None:
        self._is_snoozed = bool(snoozed)
        self._snooze_action.setText(
            tr("menu.show_bar_now", self._lang) if snoozed else tr("menu.snooze_1h", self._lang)
        )

    def set_last_dictation_preview(self, text: str | None) -> None:
        self._last_preview_raw = text or ""
        snippet = " ".join(self._last_preview_raw.split())
        if snippet:
            if len(snippet) > 48:
                snippet = snippet[:47] + "…"
            self._copy_last_action.setToolTip(tr("menu.copy_tooltip", self._lang, snippet=snippet))
        else:
            self._copy_last_action.setToolTip("")

    def set_state_icon(self, state: str) -> None:
        if state == "recording":
            self.setIcon(self._icon_recording)
        elif state in ("transcribing", "polishing"):
            self.setIcon(self._icon_busy)
        else:
            self.setIcon(self._icon_idle)
