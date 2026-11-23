"""Theme detection (Windows system dark/light) and stylesheets for LocalFlow."""

from __future__ import annotations

import ctypes
import sys
from typing import Literal

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget

from localflow.i18n import tr

ThemeMode = Literal["system", "dark", "light"]
ResolvedTheme = Literal["dark", "light"]


def get_theme_choice_items(lang: str | None = None) -> tuple[tuple[str, ThemeMode], ...]:
    return (
        (tr("theme.system", lang), "system"),
        (tr("theme.dark", lang), "dark"),
        (tr("theme.light", lang), "light"),
    )


THEME_CHOICE_ITEMS: tuple[tuple[str, ThemeMode], ...] = get_theme_choice_items("en")


def detect_system_theme() -> ResolvedTheme:
    """Detect OS light/dark theme (Windows registry AppsUseLightTheme + Qt ColorScheme fallback)."""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
                0,
                winreg.KEY_READ,
            ) as key:
                val, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                return "light" if int(val) == 1 else "dark"
        except Exception:
            pass

    app = QGuiApplication.instance()
    if app is not None:
        try:
            scheme = app.styleHints().colorScheme()
            if scheme == Qt.ColorScheme.Light:
                return "light"
            if scheme == Qt.ColorScheme.Dark:
                return "dark"
        except Exception:
            pass
    return "dark"


def resolve_theme(mode: str | None) -> ResolvedTheme:
    """Resolve user theme preference ('system' | 'dark' | 'light') to 'dark' or 'light'."""
    norm = (mode or "system").strip().lower()
    if norm == "light":
        return "light"
    if norm == "dark":
        return "dark"
    return detect_system_theme()


def apply_windows_titlebar_theme(widget: QWidget, resolved: ResolvedTheme) -> None:
    """Apply dark or light native Windows 10/11 window titlebar via DWM."""
    if sys.platform != "win32":
        return
    try:
        hwnd = int(widget.winId())
        use_dark = ctypes.c_int(1 if resolved == "dark" else 0)
        dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
        # 20 = DWMWA_USE_IMMERSIVE_DARK_MODE (Win10 2004+ / Win11), 19 = older Win10
        for attr in (20, 19):
            res = dwmapi.DwmSetWindowAttribute(
                ctypes.c_void_p(hwnd),
                ctypes.c_uint(attr),
                ctypes.byref(use_dark),
                ctypes.sizeof(use_dark),
            )
            if res == 0:
                break
    except Exception:
        pass


_DARK_TOKENS: dict[str, str] = {
    "win_bg": "#111113",
    "text": "#e4e4e7",
    "text_strong": "#fafafa",
    "text_secondary": "#d4d4d8",
    "text_muted": "#8a8a93",
    "section_hdr": "#71717a",
    "canvas_bg": "#1a1a1c",
    "canvas_border": "#262629",
    "inspector_bg": "#151517",
    "inspector_border": "#252528",
    "status_bg": "#09090b",
    "status_border": "#222226",
    "card_bg": "#202023",
    "card_border": "#2a2a2e",
    "divider": "#252528",
    "badge_bg": "#4c1d95",
    "badge_fg": "#ede9fe",
    "nav_fg": "#a1a1aa",
    "nav_hover_bg": "#1c1c1f",
    "nav_hover_fg": "#f4f4f5",
    "nav_active_bg": "#262629",
    "nav_active_fg": "#ffffff",
    "tab_fg": "#8a8a93",
    "tab_hover_fg": "#e4e4e7",
    "tab_active_fg": "#fafafa",
    "primary_bg": "#ffffff",
    "primary_fg": "#111113",
    "primary_hover": "#e4e4e7",
    "primary_pressed": "#d4d4d8",
    "secondary_bg": "#262629",
    "secondary_fg": "#f4f4f5",
    "secondary_border": "#323237",
    "secondary_hover_bg": "#303036",
    "secondary_hover_border": "#3f3f46",
    "ghost_fg": "#a1a1aa",
    "ghost_hover_bg": "#262629",
    "ghost_hover_fg": "#fafafa",
    "ghost_disabled": "#52525b",
    "input_bg": "#141416",
    "input_border": "#2c2c31",
    "input_fg": "#f4f4f5",
    "input_sel": "#3f3f46",
    "input_focus": "#52525b",
    "input_dis_fg": "#52525b",
    "input_dis_bg": "#121214",
    "input_dis_border": "#222226",
    "popup_bg": "#18181b",
    "popup_border": "#2e2e34",
    "popup_sel": "#2c2c32",
    "check_border": "#3f3f46",
    "check_bg": "#141416",
    "check_hover": "#71717a",
    "list_bg": "#202023",
    "list_border": "#29292d",
    "list_hover_bg": "#252529",
    "list_hover_border": "#34343a",
    "list_sel_bg": "#29292e",
    "list_sel_border": "#46464f",
    "list_sel_fg": "#ffffff",
    "scroll_track": "#141417",
    "scroll_handle": "#52525b",
    "scroll_hover": "#71717a",
    "msgbox_bg": "#18181b",
}

_LIGHT_TOKENS: dict[str, str] = {
    "win_bg": "#f2f2f5",
    "text": "#27272a",
    "text_strong": "#111113",
    "text_secondary": "#3f3f46",
    "text_muted": "#63636e",
    "section_hdr": "#71717a",
    "canvas_bg": "#ffffff",
    "canvas_border": "#e4e4e7",
    "inspector_bg": "#f9f9fb",
    "inspector_border": "#e4e4e7",
    "status_bg": "#ffffff",
    "status_border": "#e2e2e7",
    "card_bg": "#f7f7fa",
    "card_border": "#e4e4e9",
    "divider": "#e4e4e7",
    "badge_bg": "#ede9fe",
    "badge_fg": "#5b21b6",
    "nav_fg": "#52525b",
    "nav_hover_bg": "#e7e7ec",
    "nav_hover_fg": "#111113",
    "nav_active_bg": "#e2e2e8",
    "nav_active_fg": "#111113",
    "tab_fg": "#71717a",
    "tab_hover_fg": "#18181b",
    "tab_active_fg": "#111113",
    "primary_bg": "#18181b",
    "primary_fg": "#ffffff",
    "primary_hover": "#27272a",
    "primary_pressed": "#3f3f46",
    "secondary_bg": "#ffffff",
    "secondary_fg": "#18181b",
    "secondary_border": "#d4d4d8",
    "secondary_hover_bg": "#f4f4f6",
    "secondary_hover_border": "#a1a1aa",
    "ghost_fg": "#52525b",
    "ghost_hover_bg": "#e9e9ee",
    "ghost_hover_fg": "#111113",
    "ghost_disabled": "#a1a1aa",
    "input_bg": "#ffffff",
    "input_border": "#d4d4d8",
    "input_fg": "#18181b",
    "input_sel": "#dbeafe",
    "input_focus": "#71717a",
    "input_dis_fg": "#a1a1aa",
    "input_dis_bg": "#f4f4f6",
    "input_dis_border": "#e4e4e7",
    "popup_bg": "#ffffff",
    "popup_border": "#d4d4d8",
    "popup_sel": "#ededf2",
    "check_border": "#a1a1aa",
    "check_bg": "#ffffff",
    "check_hover": "#52525b",
    "list_bg": "#f8f8fb",
    "list_border": "#e4e4e9",
    "list_hover_bg": "#f0f0f5",
    "list_hover_border": "#d4d4dc",
    "list_sel_bg": "#e8e8f0",
    "list_sel_border": "#a1a1aa",
    "list_sel_fg": "#111113",
    "scroll_track": "#e8e8ee",
    "scroll_handle": "#a1a1aa",
    "scroll_hover": "#71717a",
    "msgbox_bg": "#ffffff",
}


def build_hub_stylesheet(theme: ResolvedTheme) -> str:
    t = _LIGHT_TOKENS if theme == "light" else _DARK_TOKENS
    return f"""
QWidget {{
    background-color: transparent;
    color: {t["text"]};
    font-family: 'Segoe UI';
    font-size: 9.5pt;
}}
SettingsWindow {{
    background-color: {t["win_bg"]};
}}
QFrame#sidebar {{
    background-color: {t["win_bg"]};
    border: none;
}}
QFrame#contentCanvas {{
    background-color: {t["canvas_bg"]};
    border: 1px solid {t["canvas_border"]};
    border-radius: 12px;
}}
QFrame#inspectorPane {{
    background-color: {t["inspector_bg"]};
    border-left: 1px solid {t["inspector_border"]};
    border-top-right-radius: 12px;
    border-bottom-right-radius: 12px;
}}
QFrame#statusCard {{
    background-color: {t["status_bg"]};
    border: 1px solid {t["status_border"]};
    border-radius: 10px;
}}
QFrame#sectionCard {{
    background-color: {t["card_bg"]};
    border: 1px solid {t["card_border"]};
    border-radius: 10px;
}}
QFrame#divider {{
    background-color: {t["divider"]};
    max-height: 1px;
    min-height: 1px;
}}
QLabel#brandTitle {{
    color: {t["text_strong"]};
    font-size: 13.5pt;
    font-weight: 700;
}}
QLabel#brandBadge {{
    background-color: {t["badge_bg"]};
    color: {t["badge_fg"]};
    border-radius: 6px;
    padding: 2px 7px;
    font-size: 7.5pt;
    font-weight: 700;
}}
QLabel#pageTitle {{
    color: {t["text_strong"]};
    font-size: 16pt;
    font-weight: 700;
}}
QLabel#inspectorTitle {{
    color: {t["text_strong"]};
    font-size: 12pt;
    font-weight: 600;
}}
QLabel#sectionHeader {{
    color: {t["section_hdr"]};
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 0.8px;
}}
QLabel#mutedMeta {{
    color: {t["text_muted"]};
    font-size: 8.5pt;
}}
QLabel#statusItem {{
    color: {t["text_secondary"]};
    font-size: 8.5pt;
}}
QPushButton#navBtn {{
    background-color: transparent;
    color: {t["nav_fg"]};
    border: none;
    border-radius: 8px;
    padding: 8px 12px;
    text-align: left;
    font-size: 9.5pt;
    font-weight: 500;
}}
QPushButton#navBtn:hover {{
    background-color: {t["nav_hover_bg"]};
    color: {t["nav_hover_fg"]};
}}
QPushButton#navBtn:checked {{
    background-color: {t["nav_active_bg"]};
    color: {t["nav_active_fg"]};
    font-weight: 600;
}}
QPushButton#filterTab {{
    background-color: transparent;
    color: {t["tab_fg"]};
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0px;
    padding: 6px 2px 8px 2px;
    margin-right: 16px;
    font-size: 9pt;
    font-weight: 500;
}}
QPushButton#filterTab:hover {{
    color: {t["tab_hover_fg"]};
}}
QPushButton#filterTab:checked {{
    color: {t["tab_active_fg"]};
    border-bottom: 2px solid {t["tab_active_fg"]};
    font-weight: 600;
}}
QPushButton#primaryBtn {{
    background-color: {t["primary_bg"]};
    color: {t["primary_fg"]};
    border: none;
    border-radius: 8px;
    padding: 7px 16px;
    font-weight: 600;
}}
QPushButton#primaryBtn:hover {{
    background-color: {t["primary_hover"]};
}}
QPushButton#primaryBtn:pressed {{
    background-color: {t["primary_pressed"]};
}}
QPushButton#secondary {{
    background-color: {t["secondary_bg"]};
    color: {t["secondary_fg"]};
    border: 1px solid {t["secondary_border"]};
    border-radius: 8px;
    padding: 6px 13px;
    font-weight: 500;
}}
QPushButton#secondary:hover {{
    background-color: {t["secondary_hover_bg"]};
    border-color: {t["secondary_hover_border"]};
}}
QPushButton#ghostBtn {{
    background-color: transparent;
    color: {t["ghost_fg"]};
    border: none;
    border-radius: 6px;
    padding: 5px 10px;
    font-weight: 500;
}}
QPushButton#ghostBtn:hover {{
    background-color: {t["ghost_hover_bg"]};
    color: {t["ghost_hover_fg"]};
}}
QPushButton#ghostBtn:disabled {{
    color: {t["ghost_disabled"]};
}}
QPushButton#dangerBtn {{
    background-color: #dc2626;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 6px 14px;
    font-weight: 600;
}}
QPushButton#dangerBtn:hover {{
    background-color: #ef4444;
}}
QPushButton#dangerBtn:pressed {{
    background-color: #b91c1c;
}}
QMessageBox {{
    background-color: {t["msgbox_bg"]};
}}
QMessageBox QLabel {{
    color: {t["text"]};
    background-color: transparent;
}}
QProgressBar {{
    background-color: {t["input_bg"]};
    border: 1px solid {t["input_border"]};
    border-radius: 4px;
    max-height: 8px;
    min-height: 8px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: #10b981;
    border-radius: 3px;
}}
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {t["input_bg"]};
    border: 1px solid {t["input_border"]};
    border-radius: 7px;
    padding: 6px 9px;
    color: {t["input_fg"]};
    selection-background-color: {t["input_sel"]};
}}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {t["input_focus"]};
}}
QComboBox {{
    padding-right: 22px;
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox::down-arrow {{
    image: none;
    width: 0px;
    height: 0px;
}}
QComboBox QAbstractItemView {{
    background-color: {t["popup_bg"]};
    border: 1px solid {t["popup_border"]};
    selection-background-color: {t["popup_sel"]};
    color: {t["input_fg"]};
    outline: 0;
}}
QDoubleSpinBox:disabled, QSpinBox:disabled, QComboBox:disabled, QTextEdit:disabled {{
    color: {t["input_dis_fg"]};
    background-color: {t["input_dis_bg"]};
    border-color: {t["input_dis_border"]};
}}
QCheckBox:disabled {{
    color: {t["input_dis_fg"]};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 18px;
}}
QCheckBox {{
    spacing: 8px;
    color: {t["text"]};
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 5px;
    border: 1.5px solid {t["check_border"]};
    background-color: {t["check_bg"]};
}}
QCheckBox::indicator:hover {{
    border-color: {t["check_hover"]};
}}
QCheckBox::indicator:checked {{
    background-color: #10b981;
    border: 3px solid {t["check_bg"]};
}}
QListWidget#historyList {{
    background-color: transparent;
    border: none;
    outline: 0;
}}
QListWidget#historyList::item {{
    background-color: {t["list_bg"]};
    border: 1px solid {t["list_border"]};
    border-radius: 9px;
    padding: 10px 12px;
    margin-bottom: 6px;
    color: {t["text"]};
}}
QListWidget#historyList::item:hover {{
    background-color: {t["list_hover_bg"]};
    border-color: {t["list_hover_border"]};
}}
QListWidget#historyList::item:selected {{
    background-color: {t["list_sel_bg"]};
    border: 1px solid {t["list_sel_border"]};
    color: {t["list_sel_fg"]};
}}
QTextEdit#inspectorText {{
    background-color: transparent;
    border: none;
    padding: 0px;
    color: {t["text_secondary"]};
    font-size: 9.5pt;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: {t["scroll_track"]};
    width: 8px;
    margin: 2px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {t["scroll_handle"]};
    border-radius: 4px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{
    background: {t["scroll_hover"]};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
    background: none;
    border: none;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
    border: none;
}}
QScrollBar:horizontal {{
    background: {t["scroll_track"]};
    height: 8px;
    margin: 2px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {t["scroll_handle"]};
    border-radius: 4px;
    min-width: 28px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {t["scroll_hover"]};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
    background: none;
    border: none;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
    border: none;
}}
"""


def build_menu_stylesheet(theme: ResolvedTheme) -> str:
    if theme == "light":
        return """
QMenu {
    background-color: #ffffff;
    color: #18181b;
    border: 1px solid #d4d4d8;
    border-radius: 8px;
    padding: 5px;
    font-family: 'Segoe UI';
    font-size: 9pt;
}
QMenu::item {
    padding: 6px 24px 6px 24px;
    border-radius: 5px;
}
QMenu::item:selected {
    background-color: #2563eb;
    color: #ffffff;
}
QMenu::item:disabled {
    color: #71717a;
}
QMenu::separator {
    height: 1px;
    background: #e4e4e7;
    margin: 4px 8px;
}
"""
    return """
QMenu {
    background-color: #14171f;
    color: #e5e7eb;
    border: 1px solid #2a2f3d;
    border-radius: 8px;
    padding: 5px;
    font-family: 'Segoe UI';
    font-size: 9pt;
}
QMenu::item {
    padding: 6px 24px 6px 24px;
    border-radius: 5px;
}
QMenu::item:selected {
    background-color: #2563eb;
    color: #ffffff;
}
QMenu::item:disabled {
    color: #9ca3af;
}
QMenu::separator {
    height: 1px;
    background: #262b38;
    margin: 4px 8px;
}
"""
