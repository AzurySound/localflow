"""Compact, draggable Wispr Flow-style Flow Bar overlay that never steals focus."""

from __future__ import annotations

import ctypes
import math
from enum import Enum
from typing import Callable

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QWidget

from localflow.i18n import DEFAULT_LANGUAGE, LanguageCode, normalize_language, tr
from localflow.ui.theme import ResolvedTheme

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080

COMPACT_WIDTH = 136
PILL_HEIGHT = 32
PREVIEW_WIDTH = 280
PREVIEW_HEIGHT = 22
PREVIEW_GAP = 6

IDLE_HITBOX_WIDTH = 68
IDLE_HITBOX_HEIGHT = 14
IDLE_BAR_WIDTH = 54
IDLE_BAR_HEIGHT = 7
HOVER_WIDTH = 86
HOVER_RIGHT_EXTRA = 32
DRAG_THRESHOLD_PX = 4


class _PillTooltipPopup(QWidget):
    """Floating non-activating obsidian/light capsule tooltip shown above the Flow Bar."""

    HEIGHT = 30

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.WindowTransparentForInput,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._text = ""
        self._bold_suffix = ""
        self._theme: ResolvedTheme = "dark"
        self._font_reg = QFont("Segoe UI", 9, QFont.Weight.Normal)
        self._font_bold = QFont("Segoe UI", 9, QFont.Weight.Bold)
        self._fm_reg = QFontMetrics(self._font_reg)
        self._fm_bold = QFontMetrics(self._font_bold)

    def show_above(
        self,
        anchor_rect: QRect,
        text: str,
        bold_suffix: str = "",
        theme: ResolvedTheme = "dark",
    ) -> None:
        self._text = text.strip()
        self._bold_suffix = bold_suffix.strip()
        self._theme = theme
        if not self._text:
            self.hide()
            return

        reg_w = self._fm_reg.horizontalAdvance(self._text)
        bold_w = (
            self._fm_reg.horizontalAdvance(" ") + self._fm_bold.horizontalAdvance(self._bold_suffix)
            if self._bold_suffix
            else 0
        )
        w = max(72, reg_w + bold_w + 26)
        h = self.HEIGHT
        self.setFixedSize(w, h)

        x = anchor_rect.center().x() - w // 2
        y = anchor_rect.top() - h - 8
        screen = QGuiApplication.screenAt(anchor_rect.center()) or QGuiApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            x = max(geo.x() + 4, min(x, geo.x() + geo.width() - w - 4))
            y = max(geo.y() + 4, y)
        self.move(x, y)
        if not self.isVisible():
            self.show()
        self.raise_()
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            rect = QRectF(0.75, 0.75, float(self.width()) - 1.5, float(self.height()) - 1.5)
            radius = rect.height() / 2.0
            path = QPainterPath()
            path.addRoundedRect(rect, radius, radius)

            is_light = self._theme == "light"
            bg = QColor(255, 255, 255, 248) if is_light else QColor(12, 12, 15, 245)
            border = QColor(24, 24, 27, 45) if is_light else QColor(255, 255, 255, 42)
            fg_reg = QColor(39, 39, 42) if is_light else QColor(228, 228, 231)
            fg_bold = QColor(17, 17, 19) if is_light else QColor(250, 250, 250)

            p.fillPath(path, bg)
            p.setPen(QPen(border, 1.0))
            p.drawPath(path)

            reg_w = self._fm_reg.horizontalAdvance(self._text)
            space_w = self._fm_reg.horizontalAdvance(" ") if self._bold_suffix else 0
            bold_w = self._fm_bold.horizontalAdvance(self._bold_suffix) if self._bold_suffix else 0
            total_w = reg_w + space_w + bold_w
            start_x = (float(self.width()) - total_w) / 2.0

            p.setFont(self._font_reg)
            p.setPen(fg_reg)
            p.drawText(
                QRectF(start_x, 0.0, float(reg_w) + 2.0, float(self.height())),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                self._text,
            )
            if self._bold_suffix:
                p.setFont(self._font_bold)
                p.setPen(fg_bold)
                p.drawText(
                    QRectF(start_x + reg_w + space_w, 0.0, float(bold_w) + 2.0, float(self.height())),
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                    self._bold_suffix,
                )
        finally:
            p.end()


class PillMode(str, Enum):
    HIDDEN = "hidden"
    IDLE_BAR = "idle_bar"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    POLISHING = "polishing"
    DOWNLOADING = "downloading"
    WARMING = "warming"
    NOTICE = "notice"
    MEETING = "meeting"


class FloatingPillWidget(QWidget):
    """Wispr Flow-style Flow Bar: thin idle strip, hover expansion, live waveform, and drag/click."""

    position_changed = Signal(int, int)
    toggle_requested = Signal()
    cancel_requested = Signal()
    copy_last_requested = Signal()
    download_clicked = Signal()
    meeting_clicked = Signal()
    context_menu_requested = Signal(QPoint)

    def __init__(self, initial_pos: tuple[int, int] | None = None) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setFixedSize(COMPACT_WIDTH, PILL_HEIGHT)

        self._mode = PillMode.HIDDEN
        self._command_mode = False
        self._lang: LanguageCode = DEFAULT_LANGUAGE
        self._processing_label = tr("pill.polishing", self._lang)
        self._notice_text = ""
        self._notice_is_error = False
        self._download_label = ""
        self._download_percent = -1
        self._warming_text = ""
        self._warming_queued = False
        self._meeting_elapsed_sec = 0.0
        self._live_preview_enabled = True
        self._preview_text = ""
        self._idle_bar_enabled = False
        self._idle_hint_text = "2× CapsLock"
        self._pill_offset_x = 0
        self._pill_body_width = COMPACT_WIDTH
        self._elapsed_sec = 0.0
        self._phase = 0.0
        self._target_level = 0.0
        self._level_provider: Callable[[], float] | None = None
        self._bars = [0.14] * 9

        # Hover expansion state for IDLE_BAR
        self._hovered = False
        self._hover_progress = 0.0
        self._right_expand_progress = 0.0
        self._context_menu_open = False
        self._cancel_hovered = False
        self._idle_hover_zone = ""  # "mic" | "notetaker" | "menu" | ""
        self._has_last_dictation = True
        self._resolved_theme: ResolvedTheme = "dark"
        self._tooltip_popup = _PillTooltipPopup()

        # Custom user position (saved across sessions; always anchored to COMPACT_WIDTH x PILL_HEIGHT)
        self._saved_pos: QPoint | None = (
            QPoint(initial_pos[0], initial_pos[1]) if initial_pos is not None else None
        )
        self._drag_offset: QPoint | None = None
        self._press_global_pos: QPoint | None = None
        self._did_drag = False

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)  # ~60 FPS for crisp fluid hover & waveform animation
        self._anim_timer.timeout.connect(self._on_tick)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._on_notice_timeout)

        self._leave_timer = QTimer(self)
        self._leave_timer.setSingleShot(True)
        self._leave_timer.setInterval(260)
        self._leave_timer.timeout.connect(self._collapse_idle_hover)

        # Fallback hover watcher: Windows TrackMouseEvent(TME_LEAVE) is one-shot, so
        # swallowing a premature leaveEvent (e.g. grazing top/right edge) disarms OS leave events.
        self._hover_watch_timer = QTimer(self)
        self._hover_watch_timer.setInterval(80)
        self._hover_watch_timer.timeout.connect(self._check_idle_hover_cursor)

        self._font_label = QFont("Segoe UI", 8, QFont.Weight.Medium)
        self._font_metrics = QFontMetrics(self._font_label)
        self._font_preview = QFont("Segoe UI", 8)
        self._preview_metrics = QFontMetrics(self._font_preview)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._apply_win32_noactivate()

    def _apply_win32_noactivate(self) -> None:
        try:
            hwnd = int(self.winId())
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            ex_style = get_long(hwnd, GWL_EXSTYLE)
            set_long(hwnd, GWL_EXSTYLE, ex_style | WS_EX_NOACTIVATE | WS_EX_TOPMOST | WS_EX_TOOLWINDOW)
        except Exception:
            pass

    def _is_point_on_any_screen(self, pt: QPoint) -> bool:
        for screen in QGuiApplication.screens():
            if screen.availableGeometry().adjusted(-40, -20, 40, 20).contains(pt):
                return True
        return False

    def _screen_for_point(self, pt: QPoint):
        for screen in QGuiApplication.screens():
            if screen.availableGeometry().adjusted(-40, -20, 40, 20).contains(pt):
                return screen
        return QGuiApplication.screenAt(pt) or QGuiApplication.primaryScreen()

    def _has_preview(self) -> bool:
        return (
            self._mode == PillMode.RECORDING
            and self._live_preview_enabled
            and bool(self._preview_text)
        )

    def _top_offset(self) -> int:
        return (PREVIEW_HEIGHT + PREVIEW_GAP) if self._has_preview() else 0

    def _idle_dimensions(self) -> tuple[int, int]:
        base_w, _extra, w, h = self._idle_layout_metrics()
        _ = base_w
        return w, h

    def _idle_layout_metrics(self) -> tuple[int, int, int, int]:
        t = 1.0 - (1.0 - self._hover_progress) ** 3
        r = 1.0 - (1.0 - self._right_expand_progress) ** 3
        base_w = round(IDLE_HITBOX_WIDTH + (HOVER_WIDTH - IDLE_HITBOX_WIDTH) * t)
        extra_w = round(HOVER_RIGHT_EXTRA * r)
        h = round(IDLE_HITBOX_HEIGHT + (PILL_HEIGHT - IDLE_HITBOX_HEIGHT) * t)
        return base_w, extra_w, base_w + extra_w, h

    def _target_width_for_mode(self) -> int:
        if self._mode == PillMode.IDLE_BAR:
            return self._idle_dimensions()[0]
        if self._mode == PillMode.NOTICE:
            return max(COMPACT_WIDTH, min(260, self._font_metrics.horizontalAdvance(self._notice_text) + 28))
        if self._has_preview():
            return PREVIEW_WIDTH
        return COMPACT_WIDTH

    def _elided_preview_text(self, max_width: int) -> str:
        if max_width <= 0 or not self._preview_text:
            return ""
        return self._preview_metrics.elidedText(
            self._preview_text,
            Qt.TextElideMode.ElideLeft,
            max_width,
        )

    def _cancel_button_rect(self) -> QRectF:
        """Return the clickable cancel ('×') hitbox in widget coordinates during RECORDING."""
        pill_x = float(self._pill_offset_x)
        pill_y = float(self._top_offset())
        return QRectF(pill_x + 4.0, pill_y + 4.0, 24.0, 24.0)

    def _mic_button_rect(self) -> QRectF:
        """Left capsule hitbox for voice dictation during expanded IDLE_BAR."""
        return QRectF(1.0, 1.0, 47.0, float(self.height()) - 2.0)

    def _notetaker_button_rect(self) -> QRectF:
        """Right circular Notetaker button hitbox inside the dual-button IDLE_BAR."""
        return QRectF(52.0, 1.0, 33.0, float(self.height()) - 2.0)

    def _menu_button_rect(self) -> QRectF:
        """Chevron menu hitbox revealed to the right of Notetaker when hovering right button."""
        return QRectF(85.0, 1.0, 32.0, float(self.height()) - 2.0)

    def _hide_hover_tooltip(self) -> None:
        if hasattr(self, "_tooltip_popup") and self._tooltip_popup.isVisible():
            self._tooltip_popup.hide()

    def _sync_hover_tooltip(self) -> None:
        if (
            self._mode != PillMode.IDLE_BAR
            or self._hover_progress < 0.8
            or self._context_menu_open
            or self._drag_offset is not None
            or not self._idle_hover_zone
        ):
            self._hide_hover_tooltip()
            return
        if self._idle_hover_zone == "mic":
            self._tooltip_popup.show_above(
                self.frameGeometry(),
                tr("pill.tip.dictate", self._lang),
                self._idle_hint_text,
                self._resolved_theme,
            )
        elif self._idle_hover_zone == "notetaker":
            self._tooltip_popup.show_above(
                self.frameGeometry(),
                tr("pill.tip.notetaker", self._lang),
                "",
                self._resolved_theme,
            )
        elif self._idle_hover_zone == "menu":
            self._tooltip_popup.show_above(
                self.frameGeometry(),
                tr("pill.tip.menu", self._lang),
                "",
                self._resolved_theme,
            )
        else:
            self._hide_hover_tooltip()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._hover_watch_timer.stop()
        self._hide_hover_tooltip()
        super().closeEvent(event)

    def _apply_idle_geometry(self) -> None:
        base_w, _extra_w, w, h = self._idle_layout_metrics()
        self._pill_offset_x = 0
        self._pill_body_width = w

        if self._drag_offset is not None:
            old_pos = self.pos()
            old_w, old_h = self.width(), self.height()
            new_x = old_pos.x()
            new_y = old_pos.y() + (old_h - h)
            self.setFixedSize(w, h)
            self.move(QPoint(new_x, new_y))
            self._drag_offset += QPoint(old_pos.x() - new_x, old_pos.y() - new_y)
            return

        if self._saved_pos is not None and self._is_point_on_any_screen(self._saved_pos):
            anchor_x = self._saved_pos.x()
            anchor_y = self._saved_pos.y()
            x = anchor_x + (COMPACT_WIDTH - base_w) // 2
            y = anchor_y + PILL_HEIGHT - h
            screen = self._screen_for_point(self._saved_pos)
            if screen is not None:
                geo = screen.availableGeometry()
                if geo.width() >= w + 8:
                    x = max(geo.x() + 4, min(x, geo.x() + geo.width() - w - 4))
                if geo.height() >= h:
                    y = max(geo.y(), min(y, geo.y() + geo.height() - h))
            self.setFixedSize(w, h)
            self.move(QPoint(x, y))
            return

        screen = (
            (
                QGuiApplication.screenAt(self.frameGeometry().center())
                or self.screen()
                or QGuiApplication.primaryScreen()
            )
            if self.isVisible()
            else (QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen())
        )
        if screen is None:
            self.setFixedSize(w, h)
            return
        geo = screen.availableGeometry()
        anchor_x = geo.x() + (geo.width() - COMPACT_WIDTH) // 2
        anchor_y = geo.y() + geo.height() - PILL_HEIGHT - 22
        x = anchor_x + (COMPACT_WIDTH - base_w) // 2
        y = anchor_y + PILL_HEIGHT - h
        self.setFixedSize(w, h)
        self.move(QPoint(x, y))

    def _apply_width_and_position(self, width: int) -> None:
        if self._mode == PillMode.IDLE_BAR:
            self._apply_idle_geometry()
            return

        top_offset = self._top_offset()
        total_height = PILL_HEIGHT + top_offset
        old_pill_offset_x = self._pill_offset_x
        old_top_offset = max(0, self.height() - PILL_HEIGHT)
        old_pill_w = self._pill_body_width
        new_pill_body_width = width if self._mode == PillMode.NOTICE else COMPACT_WIDTH

        if self._drag_offset is not None:
            new_pill_offset_x = 0 if self._mode == PillMode.NOTICE else (width - COMPACT_WIDTH) // 2
            old_anchor_offset_x = (
                -((COMPACT_WIDTH - old_pill_w) // 2)
                if old_pill_w != COMPACT_WIDTH
                else old_pill_offset_x
            )
            new_anchor_offset_x = (
                -((COMPACT_WIDTH - width) // 2)
                if self._mode == PillMode.NOTICE
                else new_pill_offset_x
            )
            old_pos = self.pos()
            old_bottom_offset = max(0, PILL_HEIGHT - self.height())
            new_x = old_pos.x() + old_anchor_offset_x - new_anchor_offset_x
            new_y = old_pos.y() + old_top_offset - old_bottom_offset - top_offset
            self._pill_offset_x = new_pill_offset_x
            self._pill_body_width = new_pill_body_width
            self.setFixedSize(width, total_height)
            self.move(QPoint(new_x, new_y))
            self._drag_offset += QPoint(old_pos.x() - new_x, old_pos.y() - new_y)
            return

        if self._saved_pos is not None and self._is_point_on_any_screen(self._saved_pos):
            target_pill_x = self._saved_pos.x()
            target_pill_y = self._saved_pos.y()
            screen = self._screen_for_point(self._saved_pos)
            desired_x = target_pill_x + (COMPACT_WIDTH - width) // 2
            desired_y = target_pill_y - top_offset
            widget_x = desired_x
            widget_y = desired_y
            if screen is not None:
                geo = screen.availableGeometry()
                if geo.width() >= width + 8:
                    widget_x = max(geo.x() + 4, min(desired_x, geo.x() + geo.width() - width - 4))
                if geo.height() >= total_height:
                    widget_y = max(geo.y(), min(desired_y, geo.y() + geo.height() - total_height))
            self._pill_offset_x = (
                0
                if self._mode == PillMode.NOTICE
                else max(0, min(width - COMPACT_WIDTH, target_pill_x - widget_x))
            )
            self._pill_body_width = new_pill_body_width
            self.setFixedSize(width, total_height)
            self.move(QPoint(widget_x, widget_y))
            return

        old_pill_center_x = self.x() + old_pill_offset_x + old_pill_w // 2
        screen = (
            (
                QGuiApplication.screenAt(self.frameGeometry().center())
                or self.screen()
                or QGuiApplication.primaryScreen()
            )
            if self.isVisible()
            else (QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen())
        )
        if screen is None:
            self._pill_offset_x = 0 if self._mode == PillMode.NOTICE else (width - COMPACT_WIDTH) // 2
            self._pill_body_width = new_pill_body_width
            self.setFixedSize(width, total_height)
            return

        geo = screen.availableGeometry()
        target_pill_y = geo.y() + geo.height() - PILL_HEIGHT - 22
        if self.isVisible():
            target_pill_x = old_pill_center_x - COMPACT_WIDTH // 2
            if geo.width() >= width + 16:
                widget_x = max(
                    geo.x() + 8,
                    min(old_pill_center_x - width // 2, geo.x() + geo.width() - width - 8),
                )
            else:
                widget_x = old_pill_center_x - width // 2
            self._pill_offset_x = (
                0
                if self._mode == PillMode.NOTICE
                else max(0, min(width - COMPACT_WIDTH, target_pill_x - widget_x))
            )
        else:
            widget_x = geo.x() + (geo.width() - width) // 2
            self._pill_offset_x = 0 if self._mode == PillMode.NOTICE else (width - COMPACT_WIDTH) // 2

        widget_y = max(geo.y(), target_pill_y - top_offset)
        self._pill_body_width = new_pill_body_width
        self.setFixedSize(width, total_height)
        self.move(QPoint(widget_x, widget_y))

    # ------------------------------------------------------------------
    # Hover, Drag-and-Drop, Click & Context Menu (without stealing focus)
    # ------------------------------------------------------------------
    def enterEvent(self, event) -> None:  # type: ignore[override]
        self._leave_timer.stop()
        if self._mode == PillMode.IDLE_BAR:
            self._hovered = True
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            if not self._hover_watch_timer.isActive():
                self._hover_watch_timer.start()
            if not self._anim_timer.isActive():
                self._anim_timer.start()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        if (
            self._mode == PillMode.IDLE_BAR
            and not self._context_menu_open
            and self._drag_offset is None
            and self.isVisible()
            and self.frameGeometry().contains(QCursor.pos())
        ):
            self._hovered = True
            self._leave_timer.stop()
            if not self._hover_watch_timer.isActive():
                self._hover_watch_timer.start()
            super().leaveEvent(event)
            return
        if self._cancel_hovered or self._idle_hover_zone:
            self._cancel_hovered = False
            self._idle_hover_zone = ""
            self.setToolTip("")
            if self._right_expand_progress > 0.0 and not self._anim_timer.isActive():
                self._anim_timer.start()
            self.update()
        self._hide_hover_tooltip()
        if self._mode == PillMode.IDLE_BAR and not self._context_menu_open and self._drag_offset is None:
            self._leave_timer.start()
        super().leaveEvent(event)

    def _check_idle_hover_cursor(self) -> None:
        if self._mode != PillMode.IDLE_BAR or not self._hovered or not self.isVisible():
            self._hover_watch_timer.stop()
            return
        if self._context_menu_open or self._drag_offset is not None:
            return
        if self.frameGeometry().contains(QCursor.pos()):
            if self._leave_timer.isActive():
                self._leave_timer.stop()
            return
        if self._idle_hover_zone:
            self._idle_hover_zone = ""
            self.setToolTip("")
            if self._right_expand_progress > 0.0 and not self._anim_timer.isActive():
                self._anim_timer.start()
            self.update()
        self._hide_hover_tooltip()
        if not self._leave_timer.isActive():
            self._leave_timer.start()

    def _collapse_idle_hover(self) -> None:
        if self._mode == PillMode.IDLE_BAR and not self._context_menu_open and self._drag_offset is None:
            if self.isVisible() and self.frameGeometry().contains(QCursor.pos()):
                self._hovered = True
                self._leave_timer.stop()
                if not self._hover_watch_timer.isActive():
                    self._hover_watch_timer.start()
                return
            self._hover_watch_timer.stop()
            self._hovered = False
            self._idle_hover_zone = ""
            self._hide_hover_tooltip()
            self.setToolTip("")
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            if not self._anim_timer.isActive():
                self._anim_timer.start()

    def set_context_menu_open(self, is_open: bool) -> None:
        self._context_menu_open = bool(is_open)
        if is_open:
            self._hide_hover_tooltip()
        elif self._mode == PillMode.IDLE_BAR and not self.frameGeometry().contains(QCursor.pos()):
            self._leave_timer.start()

    def contextMenuEvent(self, event) -> None:  # type: ignore[override]
        self._leave_timer.stop()
        self._hide_hover_tooltip()
        self.context_menu_requested.emit(event.globalPos())
        event.accept()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._hide_hover_tooltip()
            self._press_global_pos = event.globalPosition().toPoint()
            self._drag_offset = self._press_global_pos - self.frameGeometry().topLeft()
            self._did_drag = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._drag_offset is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            global_pt = event.globalPosition().toPoint()
            if not self._did_drag and self._press_global_pos is not None:
                if (global_pt - self._press_global_pos).manhattanLength() > DRAG_THRESHOLD_PX:
                    self._did_drag = True
            if self._did_drag:
                self.move(global_pt - self._drag_offset)
            event.accept()
            return

        if self._mode == PillMode.RECORDING:
            over_cancel = self._cancel_button_rect().contains(event.position())
            if over_cancel != self._cancel_hovered:
                self._cancel_hovered = over_cancel
                self.setCursor(
                    Qt.CursorShape.PointingHandCursor if over_cancel else Qt.CursorShape.OpenHandCursor
                )
                self.update()
        elif self._mode == PillMode.IDLE_BAR:
            self._leave_timer.stop()
            if not self._hovered:
                self._hovered = True
                if not self._anim_timer.isActive():
                    self._anim_timer.start()
            if not self._hover_watch_timer.isActive():
                self._hover_watch_timer.start()
            if self._hover_progress >= 0.5:
                pos = event.position()
                if self._notetaker_button_rect().contains(pos):
                    zone = "notetaker"
                elif self._right_expand_progress >= 0.45 and self._menu_button_rect().contains(pos):
                    zone = "menu"
                elif self._mic_button_rect().contains(pos):
                    zone = "mic"
                elif self.rect().contains(pos.toPoint()):
                    zone = self._idle_hover_zone if self._idle_hover_zone in ("mic", "notetaker") else "mic"
                else:
                    zone = ""
                if zone != self._idle_hover_zone:
                    self._idle_hover_zone = zone
                    self._sync_hover_tooltip()
                    right_target = 1.0 if zone in ("notetaker", "menu") else 0.0
                    if abs(self._right_expand_progress - right_target) > 1e-4 and not self._anim_timer.isActive():
                        self._anim_timer.start()
                    self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self._drag_offset is not None:
            was_drag = self._did_drag
            self._drag_offset = None
            self._press_global_pos = None
            self._did_drag = False
            self.setCursor(
                Qt.CursorShape.PointingHandCursor
                if (self._mode == PillMode.IDLE_BAR and self._hovered)
                else Qt.CursorShape.OpenHandCursor
            )

            if was_drag:
                pos = self.pos()
                if self._mode == PillMode.IDLE_BAR:
                    base_w = self._idle_layout_metrics()[0]
                    anchor_x = pos.x() - (COMPACT_WIDTH - base_w) // 2
                    anchor_y = pos.y() - (PILL_HEIGHT - self.height())
                elif self._mode in (PillMode.NOTICE, PillMode.DOWNLOADING, PillMode.WARMING):
                    anchor_x = pos.x() - (COMPACT_WIDTH - self.width()) // 2
                    anchor_y = pos.y()
                else:
                    anchor_x = pos.x() + self._pill_offset_x
                    anchor_y = pos.y() + self._top_offset()
                self._saved_pos = QPoint(anchor_x, anchor_y)
                self.position_changed.emit(self._saved_pos.x(), self._saved_pos.y())
            else:
                if self._mode == PillMode.IDLE_BAR:
                    if self._hover_progress >= 0.5:
                        if self._notetaker_button_rect().contains(event.position()):
                            self._hide_hover_tooltip()
                            self.meeting_clicked.emit()
                        elif self._right_expand_progress >= 0.45 and self._menu_button_rect().contains(
                            event.position()
                        ):
                            self._leave_timer.stop()
                            self._hide_hover_tooltip()
                            anchor = self.mapToGlobal(QPoint(self.width() // 2, 0))
                            self.context_menu_requested.emit(anchor)
                        else:
                            self._hide_hover_tooltip()
                            self.toggle_requested.emit()
                    else:
                        self._hide_hover_tooltip()
                        self.toggle_requested.emit()
                elif self._mode == PillMode.RECORDING:
                    if self._cancel_button_rect().contains(event.position()):
                        self.cancel_requested.emit()
                    else:
                        self.toggle_requested.emit()
                elif self._mode == PillMode.DOWNLOADING:
                    self.download_clicked.emit()
                elif self._mode == PillMode.WARMING and self._warming_queued:
                    self.cancel_requested.emit()
                elif self._mode == PillMode.MEETING:
                    self.meeting_clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Public state transitions & settings
    # ------------------------------------------------------------------
    def set_level_provider(self, provider: Callable[[], float] | None) -> None:
        self._level_provider = provider

    def set_audio_level(self, level: float) -> None:
        self._target_level = max(0.0, min(1.0, level))

    def set_idle_bar_enabled(self, enabled: bool) -> None:
        self._idle_bar_enabled = bool(enabled)

    def set_idle_hint(self, text: str) -> None:
        self._idle_hint_text = text or "2× CapsLock"
        if self._mode == PillMode.IDLE_BAR:
            self.update()

    def set_has_last_dictation(self, has_text: bool) -> None:
        self._has_last_dictation = bool(has_text)
        if self._mode == PillMode.IDLE_BAR:
            self.update()

    def set_theme(self, theme: ResolvedTheme) -> None:
        self._resolved_theme = theme
        self.update()

    def set_language(self, lang: str) -> None:
        self._lang = normalize_language(lang)
        self.update()

    def reset_position(self) -> None:
        self._saved_pos = None
        if self.isVisible():
            self._apply_width_and_position(self._target_width_for_mode())
            self.update()

    def set_live_preview_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._live_preview_enabled:
            return
        self._live_preview_enabled = enabled
        if self._mode == PillMode.RECORDING:
            self._apply_width_and_position(self._target_width_for_mode())
            self.update()

    def set_preview_text(self, text: str) -> None:
        normalized = " ".join((text or "").split())
        if normalized == self._preview_text:
            return
        had_preview = self._has_preview()
        self._preview_text = normalized
        if self._mode == PillMode.RECORDING:
            if had_preview != self._has_preview():
                self._apply_width_and_position(self._target_width_for_mode())
            self.update()

    def show_idle_bar(self) -> None:
        self._hide_timer.stop()
        self._leave_timer.stop()
        self._hover_watch_timer.stop()
        self._hide_hover_tooltip()
        self._mode = PillMode.IDLE_BAR
        self._command_mode = False
        self._preview_text = ""
        self._cancel_hovered = False
        self._idle_hover_zone = ""
        self._hovered = False
        self._hover_progress = 0.0
        self._right_expand_progress = 0.0
        self._anim_timer.stop()
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._apply_idle_geometry()
        self.show()
        self.raise_()
        self.update()

    def show_recording(self, *, command_mode: bool = False) -> None:
        self._hide_timer.stop()
        self._leave_timer.stop()
        self._hover_watch_timer.stop()
        self._hide_hover_tooltip()
        self._mode = PillMode.RECORDING
        self._command_mode = bool(command_mode)
        self._preview_text = ""
        self._hovered = False
        self._hover_progress = 0.0
        self._right_expand_progress = 0.0
        self._cancel_hovered = False
        self._idle_hover_zone = ""
        self._elapsed_sec = 0.0
        self._target_level = 0.0
        self._bars = [0.14] * 9
        self._apply_width_and_position(COMPACT_WIDTH)
        self.show()
        self.raise_()
        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.update()

    def show_transcribing(self) -> None:
        self._hide_timer.stop()
        self._leave_timer.stop()
        self._hover_watch_timer.stop()
        self._hide_hover_tooltip()
        self._mode = PillMode.TRANSCRIBING
        self._preview_text = ""
        self._cancel_hovered = False
        self._apply_width_and_position(COMPACT_WIDTH)
        self.show()
        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.update()

    def show_polishing(self, label: str | None = None) -> None:
        self._hide_timer.stop()
        self._leave_timer.stop()
        self._hover_watch_timer.stop()
        self._hide_hover_tooltip()
        self._mode = PillMode.POLISHING
        default_lbl = tr("pill.polishing", self._lang)
        self._processing_label = (label or default_lbl).strip() or default_lbl
        self._preview_text = ""
        self._cancel_hovered = False
        self._apply_width_and_position(COMPACT_WIDTH)
        self.show()
        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.update()

    def show_downloading(self, label: str, percent: int = -1) -> None:
        """Show compact download progress inside the Flow Bar (clicking opens Settings)."""
        self._hide_timer.stop()
        self._leave_timer.stop()
        self._hover_watch_timer.stop()
        self._hide_hover_tooltip()
        self._mode = PillMode.DOWNLOADING
        self._command_mode = False
        self._preview_text = ""
        self._cancel_hovered = False
        self._download_label = (label or "Model").strip()
        self._download_percent = -1 if percent < 0 else max(0, min(100, int(percent)))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        display = (
            f"↓ {self._download_label} · {self._download_percent}%"
            if self._download_percent >= 0
            else f"↓ {self._download_label}..."
        )
        width = max(156, min(260, self._font_metrics.horizontalAdvance(display) + 30))
        if self.width() != width or not self.isVisible():
            self._apply_width_and_position(width)
            self.show()
        if self._download_percent < 0 and not self._anim_timer.isActive():
            self._anim_timer.start()
        elif self._download_percent >= 0 and self._anim_timer.isActive():
            self._anim_timer.stop()
        self.update()

    def show_warming(self, text: str, *, queued: bool = False) -> None:
        """Show calm animated model warm-up countdown (never alarming red)."""
        self._hide_timer.stop()
        self._leave_timer.stop()
        self._hover_watch_timer.stop()
        self._hide_hover_tooltip()
        self._mode = PillMode.WARMING
        self._command_mode = False
        self._preview_text = ""
        self._cancel_hovered = False
        self._warming_text = (text or tr("pill.warming", self._lang)).strip()
        self._warming_queued = queued
        self.setCursor(Qt.CursorShape.PointingHandCursor if queued else Qt.CursorShape.OpenHandCursor)
        width = max(148, min(250, self._font_metrics.horizontalAdvance(self._warming_text) + 46))
        if self.width() != width or not self.isVisible():
            self._apply_width_and_position(width)
            self.show()
        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.update()

    def show_notice(self, text: str, *, is_error: bool = False, duration_ms: int = 2200) -> None:
        self._leave_timer.stop()
        self._hover_watch_timer.stop()
        self._hide_hover_tooltip()
        self._mode = PillMode.NOTICE
        self._command_mode = False
        self._preview_text = ""
        self._cancel_hovered = False
        self._notice_text = text
        self._notice_is_error = is_error
        self._anim_timer.stop()
        width = max(COMPACT_WIDTH, min(260, self._font_metrics.horizontalAdvance(text) + 28))
        self._apply_width_and_position(width)
        self.show()
        self.update()
        self._hide_timer.start(duration_ms)

    def _on_notice_timeout(self) -> None:
        if self._mode != PillMode.NOTICE:
            return
        if self._idle_bar_enabled:
            self.show_idle_bar()
        else:
            self.hide_pill()

    def hide_pill(self) -> None:
        self._hide_timer.stop()
        self._leave_timer.stop()
        self._hover_watch_timer.stop()
        self._anim_timer.stop()
        self._hide_hover_tooltip()
        self._mode = PillMode.HIDDEN
        self._command_mode = False
        self._preview_text = ""
        self._hovered = False
        self._hover_progress = 0.0
        self._right_expand_progress = 0.0
        self._cancel_hovered = False
        self._idle_hover_zone = ""
        self._pill_offset_x = 0
        self._pill_body_width = COMPACT_WIDTH
        self._drag_offset = None
        self._press_global_pos = None
        self._did_drag = False
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.hide()
        self.setFixedSize(COMPACT_WIDTH, PILL_HEIGHT)

    # ------------------------------------------------------------------
    # Animation & Painting
    # ------------------------------------------------------------------
    def _on_tick(self) -> None:
        self._phase += 0.22
        if self._mode == PillMode.IDLE_BAR:
            target = 1.0 if self._hovered else 0.0
            step = 0.14 if self._hovered else 0.15
            if abs(self._hover_progress - target) <= step:
                self._hover_progress = target
            elif self._hover_progress < target:
                self._hover_progress += step
            else:
                self._hover_progress -= step

            right_want = self._hovered and self._idle_hover_zone in ("notetaker", "menu")
            right_target = 1.0 if right_want else 0.0
            right_step = 0.14 if right_want else 0.12
            if abs(self._right_expand_progress - right_target) <= right_step:
                self._right_expand_progress = right_target
            elif self._right_expand_progress < right_target:
                self._right_expand_progress += right_step
            else:
                self._right_expand_progress -= right_step

            if self._hover_progress == target and self._right_expand_progress == right_target:
                self._anim_timer.stop()

            self._apply_idle_geometry()
            self._sync_hover_tooltip()
            self.update()
            return

        if self._mode == PillMode.RECORDING:
            self._elapsed_sec += 0.016
            if self._level_provider is not None:
                self._target_level = max(0.0, min(1.0, self._level_provider()))

            center = 4
            lvl = self._target_level
            for i in range(len(self._bars)):
                dist = abs(i - center) / 4.0
                envelope = 1.0 - 0.32 * dist
                harmonic = 0.55 + 0.45 * math.sin(self._phase * 1.45 - abs(i - center) * 0.95 + i * 0.35)
                target = 0.14 + (0.86 * lvl * envelope * harmonic)
                alpha = 0.50 if target > self._bars[i] else 0.21
                self._bars[i] = self._bars[i] * (1.0 - alpha) + target * alpha
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._mode == PillMode.IDLE_BAR:
            self._paint_idle_bar(painter)
            return

        top_offset = self._top_offset()
        if top_offset > 0:
            self._paint_preview(painter)

        pill_x = 0.0 if self._mode in (PillMode.NOTICE, PillMode.WARMING) else float(self._pill_offset_x)
        pill_w = float(self.width() if self._mode in (PillMode.NOTICE, PillMode.WARMING) else COMPACT_WIDTH)

        painter.save()
        try:
            painter.translate(pill_x, float(top_offset))

            rect = QRectF(0.75, 0.75, pill_w - 1.5, PILL_HEIGHT - 1.5)
            radius = rect.height() / 2.0

            path = QPainterPath()
            path.addRoundedRect(rect, radius, radius)

            # Glass surface with subtle vertical tint
            is_light = self._resolved_theme == "light"
            bg_grad = QLinearGradient(0.0, rect.top(), 0.0, rect.bottom())
            if is_light:
                bg_grad.setColorAt(0.0, QColor(255, 255, 255, 246))
                bg_grad.setColorAt(1.0, QColor(242, 243, 247, 248))
            else:
                bg_grad.setColorAt(0.0, QColor(24, 27, 36, 242))
                bg_grad.setColorAt(1.0, QColor(12, 14, 19, 246))
            painter.fillPath(path, bg_grad)

            # Gradient 1px edge stroke
            border_grad = QLinearGradient(0.0, rect.top(), 0.0, rect.bottom())
            if self._mode == PillMode.POLISHING or (self._mode == PillMode.RECORDING and self._command_mode):
                if is_light:
                    border_grad.setColorAt(0.0, QColor(124, 58, 237, 175))
                    border_grad.setColorAt(1.0, QColor(99, 102, 241, 90))
                else:
                    border_grad.setColorAt(0.0, QColor(167, 139, 250, 175))
                    border_grad.setColorAt(1.0, QColor(99, 102, 241, 80))
            elif self._mode in (PillMode.TRANSCRIBING, PillMode.WARMING):
                if is_light:
                    border_grad.setColorAt(0.0, QColor(13, 148, 136, 165))
                    border_grad.setColorAt(1.0, QColor(99, 102, 241, 80))
                else:
                    border_grad.setColorAt(0.0, QColor(45, 212, 191, 165))
                    border_grad.setColorAt(1.0, QColor(167, 139, 250, 75))
            elif self._mode == PillMode.RECORDING:
                if is_light:
                    border_grad.setColorAt(0.0, QColor(24, 24, 27, 48))
                    border_grad.setColorAt(1.0, QColor(24, 24, 27, 28))
                else:
                    border_grad.setColorAt(0.0, QColor(255, 255, 255, 68))
                    border_grad.setColorAt(1.0, QColor(255, 255, 255, 18))
            else:
                if is_light:
                    border_grad.setColorAt(0.0, QColor(24, 24, 27, 42))
                    border_grad.setColorAt(1.0, QColor(24, 24, 27, 24))
                else:
                    border_grad.setColorAt(0.0, QColor(255, 255, 255, 48))
                    border_grad.setColorAt(1.0, QColor(255, 255, 255, 14))
            painter.setPen(QPen(border_grad, 1.0))
            painter.drawPath(path)

            if self._mode == PillMode.RECORDING:
                self._paint_recording(painter)
            elif self._mode in (PillMode.TRANSCRIBING, PillMode.POLISHING):
                self._paint_processing(painter)
            elif self._mode == PillMode.DOWNLOADING:
                self._paint_downloading(painter)
            elif self._mode == PillMode.WARMING:
                self._paint_warming(painter)
            elif self._mode == PillMode.NOTICE:
                self._paint_notice(painter)
            elif self._mode == PillMode.MEETING:
                self._paint_meeting(painter)
        finally:
            painter.restore()

    def show_meeting(self, elapsed_sec: float = 0.0) -> None:
        """Show compact active meeting indicator inside the Flow Bar."""
        self._hide_timer.stop()
        self._leave_timer.stop()
        self._hover_watch_timer.stop()
        self._hide_hover_tooltip()
        self._mode = PillMode.MEETING
        self._command_mode = False
        self._preview_text = ""
        self._meeting_elapsed_sec = max(0.0, float(elapsed_sec))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if self.width() != COMPACT_WIDTH or not self.isVisible():
            self._apply_width_and_position(COMPACT_WIDTH)
            self.show()
        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.update()

    def _paint_meeting(self, painter: QPainter) -> None:
        is_light = self._resolved_theme == "light"
        total = max(0, int(self._meeting_elapsed_sec))
        mins, secs = divmod(total, 60)
        hrs, mins = divmod(mins, 60)
        ts = f"{hrs:02d}:{mins:02d}:{secs:02d}" if hrs > 0 else f"{mins:02d}:{secs:02d}"

        pulse = 0.55 + 0.45 * math.sin(self._phase * 0.30)
        dot_color = QColor(239, 68, 68, int(round(255 * pulse)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(dot_color)
        painter.drawEllipse(QRectF(14.0, (PILL_HEIGHT - 8.0) / 2.0, 8.0, 8.0))

        painter.setPen(QColor(24, 24, 27) if is_light else QColor(248, 250, 252))
        painter.setFont(self._font_label)
        painter.drawText(
            QRectF(28.0, 0.0, COMPACT_WIDTH - 36.0, float(PILL_HEIGHT)),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"Rec {ts}",
        )

    def _paint_idle_bar(self, painter: QPainter) -> None:
        is_light = self._resolved_theme == "light"
        t = 1.0 - (1.0 - self._hover_progress) ** 3
        r = 1.0 - (1.0 - self._right_expand_progress) ** 3
        if t < 0.05:
            # Thin ambient resting bar centered inside the bottom hitbox
            bx = (self.width() - IDLE_BAR_WIDTH) / 2.0
            by = self.height() - IDLE_BAR_HEIGHT - 1.5
            rect = QRectF(bx, by, IDLE_BAR_WIDTH, IDLE_BAR_HEIGHT)
            radius = rect.height() / 2.0
            path = QPainterPath()
            path.addRoundedRect(rect, radius, radius)
            if is_light:
                painter.fillPath(path, QColor(248, 249, 252, 232))
                painter.setPen(QPen(QColor(24, 24, 27, 52), 0.9))
            else:
                painter.fillPath(path, QColor(16, 19, 26, 205))
                painter.setPen(QPen(QColor(255, 255, 255, 62), 0.9))
            painter.drawPath(path)

            # Subtle center accent pill dot
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(
                QColor(82, 82, 91, 175) if is_light else QColor(148, 163, 184, 150)
            )
            painter.drawRoundedRect(QRectF(rect.center().x() - 7.0, rect.center().y() - 1.0, 14.0, 2.0), 1.0, 1.0)
            return

        # Expanded dual-button bar: [mic] + (◉) in resting hover, expanding to [mic] + [◉ | ^] on right hover
        content_alpha = int(255 * max(0.0, min(1.0, (t - 0.15) / 0.85)))
        h = float(self.height())
        w = float(self.width())
        # Nearly invisible hit-test bridge across the full expanded rect so Windows
        # layered hit-testing never drops hover in the 4px transparent gap between capsules.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 1))
        painter.drawRect(QRectF(0.0, 0.0, w, h))
        gap = 4.0
        base_w = IDLE_HITBOX_WIDTH + (HOVER_WIDTH - IDLE_HITBOX_WIDTH) * t
        left_w = min(47.0, max(24.0, (base_w - gap) * 0.57))
        left_rect = QRectF(0.75, 0.75, left_w, h - 1.5)
        right_x = left_rect.right() + gap
        right_w = max(22.0, w - right_x - 0.75)
        right_rect = QRectF(right_x, 0.75, right_w, h - 1.5)

        if is_light:
            cap_base = QColor(255, 255, 255, int(235 + 15 * t))
            cap_hover = QColor(238, 238, 243, int(245))
            cap_border = QColor(24, 24, 27, int(42 * t + 18))
            fg_bright = QColor(24, 24, 27, max(40, content_alpha))
            circle_hover = QColor(24, 24, 27, 26)
        else:
            cap_base = QColor(22, 22, 26, int(230 + 20 * t))
            cap_hover = QColor(48, 48, 54, int(245))
            cap_border = QColor(255, 255, 255, int(44 * t + 14))
            fg_bright = QColor(248, 250, 252, max(40, content_alpha))
            circle_hover = QColor(255, 255, 255, 42)

        # Left capsule (Dictate microphone)
        left_path = QPainterPath()
        left_path.addRoundedRect(left_rect, left_rect.height() / 2.0, left_rect.height() / 2.0)
        painter.fillPath(
            left_path,
            cap_hover if self._idle_hover_zone == "mic" else cap_base,
        )
        painter.setPen(QPen(cap_border, 1.0))
        painter.drawPath(left_path)

        # Right capsule (Notetaker circle at rest; expands rightward to reveal menu chevron on hover)
        right_path = QPainterPath()
        right_path.addRoundedRect(right_rect, right_rect.height() / 2.0, right_rect.height() / 2.0)
        painter.fillPath(right_path, cap_base)

        nt_cx = right_rect.left() + min(16.0, right_w / 2.0)
        cy = h / 2.0
        if content_alpha > 0 and self._idle_hover_zone in ("notetaker", "menu"):
            painter.save()
            try:
                painter.setClipPath(right_path)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(circle_hover)
                if self._idle_hover_zone == "notetaker":
                    painter.drawEllipse(QPointF(nt_cx, cy), 13.0, 13.0)
                elif self._idle_hover_zone == "menu" and r > 0.35:
                    painter.drawEllipse(QPointF(right_rect.right() - 16.0, cy), 13.0, 13.0)
            finally:
                painter.restore()

        painter.setPen(QPen(cap_border, 1.0))
        painter.drawPath(right_path)

        if content_alpha <= 0:
            return

        # Mic icon inside left capsule
        mx = left_rect.center().x()
        my = h / 2.0
        painter.setPen(QPen(fg_bright, 1.45, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(mx - 2.3, my - 6.0, 4.6, 7.6), 2.3, 2.3)
        painter.drawArc(QRectF(mx - 4.7, my - 3.0, 9.4, 7.4), 190 * 16, 160 * 16)
        painter.drawLine(QPointF(mx, my + 4.5), QPointF(mx, my + 6.7))
        painter.drawLine(QPointF(mx - 2.2, my + 6.7), QPointF(mx + 2.2, my + 6.7))

        # Concentric Notetaker record glyph inside right circle
        painter.setPen(QPen(fg_bright, 1.45))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(nt_cx, my), 6.0, 6.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fg_bright)
        painter.drawEllipse(QPointF(nt_cx, my), 2.7, 2.7)

        # Up-chevron fades/slides in only while hovering the right button (r > 0.25)
        chevron_alpha = int(255 * max(0.0, min(1.0, (r - 0.25) / 0.75)) * (content_alpha / 255.0))
        if chevron_alpha > 0 and right_w >= 46.0:
            ch_color = QColor(82, 82, 91, chevron_alpha) if is_light else QColor(212, 212, 216, chevron_alpha)
            ch_cx = right_rect.right() - 16.0
            painter.setPen(
                QPen(ch_color, 1.45, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolyline([QPointF(ch_cx - 3.6, my + 1.6), QPointF(ch_cx, my - 2.0), QPointF(ch_cx + 3.6, my + 1.6)])

    def _paint_preview(self, painter: QPainter) -> None:
        is_light = self._resolved_theme == "light"
        text_advance = self._preview_metrics.horizontalAdvance(self._preview_text)
        capsule_w = min(self.width() - 1.5, max(float(COMPACT_WIDTH), float(text_advance + 22)))
        pill_center = self._pill_offset_x + COMPACT_WIDTH / 2.0
        capsule_x = pill_center - capsule_w / 2.0
        capsule_x = max(0.75, min(float(self.width()) - capsule_w - 0.75, capsule_x))

        rect = QRectF(capsule_x, 0.75, capsule_w, PREVIEW_HEIGHT - 1.5)
        radius = rect.height() / 2.0

        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        if is_light:
            painter.fillPath(path, QColor(252, 252, 254, 236))
            painter.setPen(QPen(QColor(24, 24, 27, 36), 1.0))
        else:
            painter.fillPath(path, QColor(14, 16, 21, 185))
            painter.setPen(QPen(QColor(255, 255, 255, 36), 1.0))
        painter.drawPath(path)

        text_rect = QRectF(rect.x() + 10.0, rect.y(), rect.width() - 20.0, rect.height())
        elided = self._elided_preview_text(int(text_rect.width()))
        painter.setPen(
            QColor(24, 24, 27, 240) if is_light else QColor(229, 231, 235, 235)
        )
        painter.setFont(self._font_preview)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine,
            elided,
        )

    def _paint_recording(self, painter: QPainter) -> None:
        is_light = self._resolved_theme == "light"
        # Left interactive cancel ('×') button
        btn_rect = QRectF(6.0, 6.0, 20.0, 20.0)
        painter.setPen(Qt.PenStyle.NoPen)
        if is_light:
            cancel_bg = QColor(239, 68, 68, 42) if self._cancel_hovered else QColor(24, 24, 27, 14)
            x_color = QColor(220, 38, 38) if self._cancel_hovered else QColor(82, 82, 91, 220)
        else:
            cancel_bg = QColor(239, 68, 68, 70) if self._cancel_hovered else QColor(255, 255, 255, 16)
            x_color = QColor(252, 165, 165) if self._cancel_hovered else QColor(156, 163, 175, 220)
        painter.setBrush(cancel_bg)
        painter.drawEllipse(btn_rect)

        painter.setPen(QPen(x_color, 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(13.0, 13.0), QPointF(19.0, 19.0))
        painter.drawLine(QPointF(19.0, 13.0), QPointF(13.0, 19.0))

        # 9 crisp waveform bars (violet accent in command mode, dark/silver in dictation)
        bar_w = 2.4
        gap = 3.0
        start_x = 32.0
        max_h = 19.0
        center_y = PILL_HEIGHT / 2.0

        painter.setPen(Qt.PenStyle.NoPen)
        if is_light:
            bar_color = QColor(124, 58, 237, 235) if self._command_mode else QColor(24, 24, 27, 225)
        else:
            bar_color = QColor(196, 181, 253, 245) if self._command_mode else QColor(245, 246, 250, 242)
        painter.setBrush(bar_color)
        for i, val in enumerate(self._bars):
            h = max(3.0, val * max_h)
            x = start_x + i * (bar_w + gap)
            y = center_y - h / 2.0
            painter.drawRoundedRect(QRectF(x, y, bar_w, h), 1.2, 1.2)

        # Pulsing active dot + compact timer on the right
        pulse = 0.55 + 0.45 * math.sin(self._phase * 0.28)
        dot_rgb = (124, 58, 237) if (self._command_mode and is_light) else (
            (167, 139, 250) if self._command_mode else (239, 68, 68)
        )
        painter.setBrush(QColor(dot_rgb[0], dot_rgb[1], dot_rgb[2], int(255 * pulse)))
        painter.drawEllipse(QRectF(83.5, 13.0, 6.0, 6.0))

        total = int(self._elapsed_sec)
        mins, secs = divmod(total, 60)
        painter.setPen(QColor(82, 82, 91, 230) if is_light else QColor(156, 163, 175, 225))
        painter.setFont(self._font_label)
        painter.drawText(
            QRectF(92.0, 0.0, 35.0, float(PILL_HEIGHT)),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            f"{mins}:{secs:02d}",
        )

    def _paint_processing(self, painter: QPainter) -> None:
        is_light = self._resolved_theme == "light"
        is_polish = self._mode == PillMode.POLISHING
        if is_light:
            accent = QColor(124, 58, 237) if is_polish else QColor(2, 132, 199)
            text_color = QColor(24, 24, 27)
        else:
            accent = QColor(167, 139, 250) if is_polish else QColor(56, 189, 248)
            text_color = QColor(243, 244, 246)
        label = self._processing_label if is_polish else tr("pill.transcribing", self._lang)

        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(3):
            alpha = int(95 + 160 * (0.5 + 0.5 * math.sin(self._phase * 1.15 - i * 0.9)))
            c = QColor(accent)
            c.setAlpha(alpha)
            painter.setBrush(c)
            painter.drawEllipse(QRectF(14 + i * 8, 13.5, 5, 5))

        painter.setPen(text_color)
        painter.setFont(self._font_label)
        painter.drawText(
            QRectF(44, 0, COMPACT_WIDTH - 52, PILL_HEIGHT),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            label,
        )

    def _paint_downloading(self, painter: QPainter) -> None:
        is_light = self._resolved_theme == "light"
        w = float(self.width())
        h = float(PILL_HEIGHT)

        # Clip progress bar inside the rounded pill body
        clip_path = QPainterPath()
        clip_path.addRoundedRect(QRectF(1.0, 1.0, w - 2.0, h - 2.0), (h - 2.0) / 2.0, (h - 2.0) / 2.0)
        painter.save()
        painter.setClipPath(clip_path)

        # Track + fill bar along bottom edge
        bar_h = 3.0
        bar_y = h - bar_h - 1.0
        track_color = QColor(24, 24, 27, 24) if is_light else QColor(255, 255, 255, 22)
        painter.fillRect(QRectF(0.0, bar_y, w, bar_h), track_color)

        fill_grad = QLinearGradient(0.0, 0.0, w, 0.0)
        if is_light:
            fill_grad.setColorAt(0.0, QColor(13, 148, 136))
            fill_grad.setColorAt(1.0, QColor(124, 58, 237))
        else:
            fill_grad.setColorAt(0.0, QColor(45, 212, 191))
            fill_grad.setColorAt(1.0, QColor(167, 139, 250))

        if self._download_percent >= 0:
            fill_w = w * (self._download_percent / 100.0)
            painter.fillRect(QRectF(0.0, bar_y, fill_w, bar_h), fill_grad)
        else:
            # Smooth indeterminate sliding bar
            seg_w = w * 0.34
            pos = ((math.sin(self._phase * 0.7) + 1.0) / 2.0) * (w - seg_w)
            painter.fillRect(QRectF(pos, bar_y, seg_w, bar_h), fill_grad)
        painter.restore()

        display = (
            f"↓ {self._download_label} · {self._download_percent}%"
            if self._download_percent >= 0
            else f"↓ {self._download_label}..."
        )
        text_color = QColor(24, 24, 27) if is_light else QColor(243, 244, 246)
        painter.setPen(text_color)
        painter.setFont(self._font_label)
        painter.drawText(
            QRectF(12.0, 0.0, w - 24.0, h - 2.0),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine,
            display,
        )

    def _paint_warming(self, painter: QPainter) -> None:
        is_light = self._resolved_theme == "light"
        w = float(self.width())
        h = float(PILL_HEIGHT)

        # Soft animated shimmer bar along bottom edge inside rounded clip
        clip_path = QPainterPath()
        clip_path.addRoundedRect(QRectF(1.0, 1.0, w - 2.0, h - 2.0), (h - 2.0) / 2.0, (h - 2.0) / 2.0)
        painter.save()
        try:
            painter.setClipPath(clip_path)
            bar_h = 2.5
            bar_y = h - bar_h - 1.0
            track_color = QColor(24, 24, 27, 20) if is_light else QColor(255, 255, 255, 18)
            painter.fillRect(QRectF(0.0, bar_y, w, bar_h), track_color)
            fill_grad = QLinearGradient(0.0, 0.0, w, 0.0)
            if is_light:
                fill_grad.setColorAt(0.0, QColor(13, 148, 136))
                fill_grad.setColorAt(1.0, QColor(99, 102, 241))
            else:
                fill_grad.setColorAt(0.0, QColor(45, 212, 191))
                fill_grad.setColorAt(1.0, QColor(167, 139, 250))
            seg_w = w * 0.36
            pos = ((math.sin(self._phase * 0.65) + 1.0) / 2.0) * (w - seg_w)
            painter.fillRect(QRectF(pos, bar_y, seg_w, bar_h), fill_grad)
        finally:
            painter.restore()

        dot_color = QColor(13, 148, 136) if is_light else QColor(45, 212, 191)
        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(3):
            alpha = int(90 + 165 * (0.5 + 0.5 * math.sin(self._phase * 1.1 - i * 0.85)))
            c = QColor(dot_color)
            c.setAlpha(alpha)
            painter.setBrush(c)
            painter.drawEllipse(QRectF(13.0 + i * 7.0, 13.5, 4.5, 4.5))

        text_color = QColor(24, 24, 27) if is_light else QColor(243, 244, 246)
        painter.setPen(text_color)
        painter.setFont(self._font_label)
        painter.drawText(
            QRectF(36.0, 0.0, w - 46.0, h - 2.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextSingleLine,
            self._warming_text,
        )

    def _paint_notice(self, painter: QPainter) -> None:
        is_light = self._resolved_theme == "light"
        if is_light:
            color = QColor(220, 38, 38) if self._notice_is_error else QColor(5, 150, 105)
        else:
            color = QColor(248, 113, 113) if self._notice_is_error else QColor(167, 243, 208)
        painter.setPen(color)
        painter.setFont(self._font_label)
        painter.drawText(
            QRectF(12, 0, self.width() - 24, PILL_HEIGHT),
            Qt.AlignmentFlag.AlignCenter,
            self._notice_text,
        )
