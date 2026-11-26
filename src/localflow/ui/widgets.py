"""Shared themed widgets: vector checkbox style, dropdown combo, and frameless window shell helper."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QLineEdit,
    QProxyStyle,
    QPushButton,
    QStyle,
    QStyleOption,
    QWidget,
)

from localflow.ui.theme import ResolvedTheme, get_theme_tokens


class CheckmarkProxyStyle(QProxyStyle):
    """Paints crisp rounded checkboxes with an explicit white checkmark across all themes."""

    def __init__(self, theme: ResolvedTheme = "dark") -> None:
        super().__init__()
        self._theme: ResolvedTheme = theme

    def set_theme(self, theme: ResolvedTheme) -> None:
        self._theme = theme

    def pixelMetric(  # type: ignore[override]
        self,
        metric: QStyle.PixelMetric,
        option: QStyleOption | None = None,
        widget: QWidget | None = None,
    ) -> int:
        if metric in (
            QStyle.PixelMetric.PM_IndicatorWidth,
            QStyle.PixelMetric.PM_IndicatorHeight,
        ):
            return 16
        return super().pixelMetric(metric, option, widget)

    def drawPrimitive(  # type: ignore[override]
        self,
        element: QStyle.PrimitiveElement,
        option: QStyleOption,
        painter: QPainter,
        widget: QWidget | None = None,
    ) -> None:
        if element != QStyle.PrimitiveElement.PE_IndicatorCheckBox:
            super().drawPrimitive(element, option, painter, widget)
            return

        t = get_theme_tokens(self._theme)
        r = QRectF(option.rect)
        box_size = 16.0
        box = QRectF(
            r.center().x() - box_size / 2.0 + 0.5,
            r.center().y() - box_size / 2.0 + 0.5,
            box_size - 1.0,
            box_size - 1.0,
        )

        enabled = bool(option.state & QStyle.StateFlag.State_Enabled)
        checked = bool(option.state & QStyle.StateFlag.State_On)
        indeterminate = bool(option.state & QStyle.StateFlag.State_NoChange)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            if checked or indeterminate:
                fill = QColor(t["accent"] if enabled else t["check_border"])
                painter.setPen(QPen(fill, 1.2))
                painter.setBrush(fill)
                painter.drawRoundedRect(box, 4.5, 4.5)

                painter.setBrush(Qt.BrushStyle.NoBrush)
                pen = QPen(
                    QColor("#ffffff"),
                    2.0,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
                painter.setPen(pen)
                if checked:
                    painter.drawPolyline(
                        [
                            QPointF(box.left() + box.width() * 0.25, box.top() + box.height() * 0.52),
                            QPointF(box.left() + box.width() * 0.43, box.top() + box.height() * 0.70),
                            QPointF(box.left() + box.width() * 0.76, box.top() + box.height() * 0.30),
                        ]
                    )
                else:
                    mid_y = box.center().y()
                    painter.drawLine(
                        QPointF(box.left() + box.width() * 0.26, mid_y),
                        QPointF(box.right() - box.width() * 0.26, mid_y),
                    )
            else:
                bg = QColor(t["check_bg"] if enabled else t["input_dis_bg"])
                border_col = t["check_hover"] if (hovered and enabled) else (
                    t["check_border"] if enabled else t["input_dis_border"]
                )
                painter.setPen(QPen(QColor(border_col), 1.4))
                painter.setBrush(bg)
                painter.drawRoundedRect(box, 4.5, 4.5)
        finally:
            painter.restore()


class DropdownCombo(QComboBox):
    """QComboBox with crisp anti-aliased chevron indicator and disabled wheel scrolling."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            win = self.window()
            theme = getattr(win, "_resolved_theme", getattr(win, "_theme", "dark"))
            t = get_theme_tokens("light" if theme == "light" else "dark")
            color = QColor(t["nav_fg"] if self.isEnabled() else t["input_dis_fg"])
            p.setPen(
                QPen(
                    color,
                    1.5,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            cx = float(self.width()) - 14.0
            cy = float(self.height()) / 2.0 + 1.0
            p.drawLine(QPointF(cx - 3.6, cy - 2.2), QPointF(cx, cy + 1.6))
            p.drawLine(QPointF(cx, cy + 1.6), QPointF(cx + 3.6, cy - 2.2))
        finally:
            p.end()


def _edges_for_pos(pos: QPoint, rect: QRect, margin: int) -> Qt.Edge:
    edges = Qt.Edge(0)
    if pos.x() <= margin:
        edges |= Qt.Edge.LeftEdge
    elif pos.x() >= rect.width() - margin:
        edges |= Qt.Edge.RightEdge
    if pos.y() <= margin:
        edges |= Qt.Edge.TopEdge
    elif pos.y() >= rect.height() - margin:
        edges |= Qt.Edge.BottomEdge
    return edges


def _cursor_for_edges(edges: Qt.Edge) -> Qt.CursorShape | None:
    if edges in (
        Qt.Edge.LeftEdge | Qt.Edge.TopEdge,
        Qt.Edge.RightEdge | Qt.Edge.BottomEdge,
    ):
        return Qt.CursorShape.SizeFDiagCursor
    if edges in (
        Qt.Edge.RightEdge | Qt.Edge.TopEdge,
        Qt.Edge.LeftEdge | Qt.Edge.BottomEdge,
    ):
        return Qt.CursorShape.SizeBDiagCursor
    if edges in (Qt.Edge.LeftEdge, Qt.Edge.RightEdge):
        return Qt.CursorShape.SizeHorCursor
    if edges in (Qt.Edge.TopEdge, Qt.Edge.BottomEdge):
        return Qt.CursorShape.SizeVerCursor
    return None


class FramelessWindowHelper(QObject):
    """Adds native-feeling drag, double-click maximize, 8-edge resize, and shell radius sync."""

    RESIZE_MARGIN = 6

    def __init__(
        self,
        window: QWidget,
        shell: QFrame,
        *,
        titlebar_height: int = 40,
        on_maximized_changed: Callable[[bool], None] | None = None,
    ) -> None:
        super().__init__(window)
        self._window = window
        self._shell = shell
        self._titlebar_height = titlebar_height
        self._on_maximized_changed = on_maximized_changed
        self._drag_offset: QPoint | None = None

        window.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        window.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        window.setMouseTracking(True)
        shell.setMouseTracking(True)

        window.installEventFilter(self)
        shell.installEventFilter(self)
        self.sync_maximized_state()

    def _reset_cursor(self) -> None:
        if self._window.cursor().shape() != Qt.CursorShape.ArrowCursor:
            self._window.unsetCursor()

    def _update_edge_cursor(self, wpos: QPoint) -> None:
        if self._window.isMaximized():
            self._reset_cursor()
            return
        edges = _edges_for_pos(wpos, self._window.rect(), self.RESIZE_MARGIN)
        cur = _cursor_for_edges(edges)
        if cur is not None:
            self._window.setCursor(cur)
        else:
            self._reset_cursor()

    def sync_maximized_state(self) -> None:
        is_max = self._window.isMaximized()
        if is_max:
            self._reset_cursor()
        if self._shell.property("windowMaximized") != is_max:
            self._shell.setProperty("windowMaximized", is_max)
            self._shell.style().unpolish(self._shell)
            self._shell.style().polish(self._shell)
            self._shell.update()
        if self._on_maximized_changed is not None:
            self._on_maximized_changed(is_max)

    def _win_pos_from_event(self, watched: QObject, event) -> QPoint:
        gpos = event.globalPosition().toPoint()
        return gpos - self._window.frameGeometry().topLeft()

    def _is_interactive_child(self, child: QWidget | None) -> bool:
        cur = child
        while cur is not None and cur is not self._window and cur is not self._shell:
            if isinstance(cur, (QPushButton, QLineEdit)):
                return True
            cur = cur.parentWidget()
        return False

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        etype = event.type()

        if etype == QEvent.Type.WindowStateChange and watched is self._window:
            self.sync_maximized_state()
            return False

        if etype == QEvent.Type.MouseButtonDblClick and hasattr(event, "button"):
            if event.button() == Qt.MouseButton.LeftButton:  # type: ignore[attr-defined]
                wpos = self._win_pos_from_event(watched, event)
                if wpos.y() <= self._titlebar_height:
                    child = self._window.childAt(wpos)
                    if not self._is_interactive_child(child):
                        if self._window.isMaximized():
                            self._window.showNormal()
                        else:
                            self._window.showMaximized()
                        return True

        elif etype == QEvent.Type.MouseButtonPress and hasattr(event, "button"):
            if event.button() == Qt.MouseButton.LeftButton:  # type: ignore[attr-defined]
                wpos = self._win_pos_from_event(watched, event)
                if not self._window.isMaximized():
                    edges = _edges_for_pos(wpos, self._window.rect(), self.RESIZE_MARGIN)
                    if edges != Qt.Edge(0):
                        handle = self._window.windowHandle()
                        if handle is not None and handle.startSystemResize(edges):
                            return True
                if wpos.y() <= self._titlebar_height:
                    child = self._window.childAt(wpos)
                    if not self._is_interactive_child(child):
                        handle = self._window.windowHandle()
                        if handle is not None and handle.startSystemMove():
                            return True
                        self._drag_offset = (
                            event.globalPosition().toPoint()  # type: ignore[attr-defined]
                            - self._window.frameGeometry().topLeft()
                        )
                        return True

        elif etype == QEvent.Type.HoverMove and watched is self._window:
            if self._drag_offset is None:
                wpos = event.position().toPoint() if hasattr(event, "position") else event.pos()  # type: ignore[attr-defined]
                self._update_edge_cursor(wpos)

        elif etype in (QEvent.Type.HoverLeave, QEvent.Type.Leave) and watched is self._window:
            self._reset_cursor()

        elif etype == QEvent.Type.MouseMove and hasattr(event, "globalPosition"):
            if self._drag_offset is not None and (event.buttons() & Qt.MouseButton.LeftButton):  # type: ignore[attr-defined]
                self._window.move(event.globalPosition().toPoint() - self._drag_offset)  # type: ignore[attr-defined]
                return True
            self._update_edge_cursor(self._win_pos_from_event(watched, event))

        elif etype == QEvent.Type.MouseButtonRelease:
            if self._drag_offset is not None:
                self._drag_offset = None
                return True

        return False
