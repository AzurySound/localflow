"""Vector QPainter icon factory for crisp HiDPI monochrome UI icons."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def render_icon(name: str, color: str, size: int = 16) -> QIcon:
    """Render a crisp 2x-HiDPI monochrome vector icon from normalized 16x16 paths."""
    dpr = 2.0
    pix = QPixmap(int(round(size * dpr)), int(round(size * dpr)))
    pix.fill(Qt.GlobalColor.transparent)
    pix.setDevicePixelRatio(dpr)

    p = QPainter(pix)
    try:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        scale = float(size) / 16.0
        p.scale(scale, scale)

        qcolor = QColor(color)
        pen = QPen(
            qcolor,
            1.45,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        if name == "history":
            p.drawArc(QRectF(2.5, 2.5, 11.0, 11.0), 40 * 16, 290 * 16)
            p.drawPolyline([QPointF(2.0, 4.2), QPointF(2.9, 6.2), QPointF(4.8, 5.2)])
            p.drawLine(QPointF(8.0, 5.0), QPointF(8.0, 8.2))
            p.drawLine(QPointF(8.0, 8.2), QPointF(10.3, 9.6))

        elif name == "notetaker":
            p.drawRoundedRect(QRectF(2.5, 2.0, 11.0, 12.0), 2.0, 2.0)
            p.drawLine(QPointF(5.2, 5.4), QPointF(10.8, 5.4))
            p.drawLine(QPointF(5.2, 8.0), QPointF(10.8, 8.0))
            p.drawLine(QPointF(5.2, 10.6), QPointF(8.4, 10.6))

        elif name == "insights":
            p.drawRoundedRect(QRectF(2.2, 8.5, 2.8, 5.0), 0.9, 0.9)
            p.drawRoundedRect(QRectF(6.6, 5.2, 2.8, 8.3), 0.9, 0.9)
            p.drawRoundedRect(QRectF(11.0, 2.5, 2.8, 11.0), 0.9, 0.9)

        elif name == "dictionary":
            p.drawRoundedRect(QRectF(3.0, 2.0, 10.0, 12.0), 1.8, 1.8)
            p.drawLine(QPointF(5.6, 2.2), QPointF(5.6, 13.8))
            p.drawLine(QPointF(7.6, 5.6), QPointF(10.8, 5.6))
            p.drawLine(QPointF(7.6, 8.4), QPointF(10.2, 8.4))

        elif name == "style":
            # Vertical slider toggles representing formatting styles
            p.drawLine(QPointF(4.0, 2.5), QPointF(4.0, 13.5))
            p.drawLine(QPointF(8.0, 2.5), QPointF(8.0, 13.5))
            p.drawLine(QPointF(12.0, 2.5), QPointF(12.0, 13.5))
            p.setBrush(qcolor)
            p.drawEllipse(QPointF(4.0, 5.5), 1.5, 1.5)
            p.drawEllipse(QPointF(8.0, 10.2), 1.5, 1.5)
            p.drawEllipse(QPointF(12.0, 6.5), 1.5, 1.5)

        elif name == "settings":
            p.drawEllipse(QPointF(8.0, 8.0), 2.4, 2.4)
            p.drawEllipse(QPointF(8.0, 8.0), 5.0, 5.0)
            for idx in range(6):
                ang = idx * math.pi / 3.0
                p.drawLine(
                    QPointF(8.0 + 5.1 * math.cos(ang), 8.0 + 5.1 * math.sin(ang)),
                    QPointF(8.0 + 6.5 * math.cos(ang), 8.0 + 6.5 * math.sin(ang)),
                )

        elif name == "search":
            p.drawEllipse(QPointF(7.0, 7.0), 4.3, 4.3)
            p.drawLine(QPointF(10.2, 10.2), QPointF(13.6, 13.6))

        elif name == "speakers":
            p.drawEllipse(QPointF(6.2, 5.2), 2.3, 2.3)
            p.drawArc(QRectF(2.2, 9.2, 8.0, 5.2), 15 * 16, 150 * 16)
            p.drawLine(QPointF(12.2, 4.5), QPointF(12.2, 8.5))
            p.drawLine(QPointF(10.2, 6.5), QPointF(14.2, 6.5))

        elif name == "copy":
            p.drawRoundedRect(QRectF(5.5, 5.5, 7.8, 8.3), 1.5, 1.5)
            path = QPainterPath()
            path.moveTo(5.5, 10.2)
            path.lineTo(3.8, 10.2)
            path.quadTo(2.7, 10.2, 2.7, 9.1)
            path.lineTo(2.7, 3.6)
            path.quadTo(2.7, 2.5, 3.8, 2.5)
            path.lineTo(9.4, 2.5)
            path.quadTo(10.5, 2.5, 10.5, 3.6)
            path.lineTo(10.5, 5.5)
            p.drawPath(path)

        elif name == "check":
            p.drawPolyline([QPointF(3.2, 8.3), QPointF(6.5, 11.6), QPointF(12.8, 4.6)])

        elif name == "back":
            p.drawPolyline([QPointF(10.0, 3.5), QPointF(5.5, 8.0), QPointF(10.0, 12.5)])

        elif name == "more":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qcolor)
            for cx in (4.0, 8.0, 12.0):
                p.drawEllipse(QPointF(cx, 8.0), 1.15, 1.15)

        elif name == "popout":
            p.drawPolyline(
                [
                    QPointF(9.0, 3.2),
                    QPointF(3.2, 3.2),
                    QPointF(3.2, 12.8),
                    QPointF(12.8, 12.8),
                    QPointF(12.8, 7.2),
                ]
            )
            p.drawLine(QPointF(7.8, 8.2), QPointF(13.2, 2.8))
            p.drawPolyline([QPointF(9.6, 2.8), QPointF(13.2, 2.8), QPointF(13.2, 6.4)])

        elif name == "export":
            p.drawPolyline(
                [
                    QPointF(3.0, 10.2),
                    QPointF(3.0, 13.0),
                    QPointF(13.0, 13.0),
                    QPointF(13.0, 10.2),
                ]
            )
            p.drawLine(QPointF(8.0, 10.2), QPointF(8.0, 3.0))
            p.drawPolyline([QPointF(5.2, 5.8), QPointF(8.0, 3.0), QPointF(10.8, 5.8)])

        elif name == "trash":
            p.drawLine(QPointF(3.2, 4.4), QPointF(12.8, 4.4))
            p.drawRoundedRect(QRectF(6.2, 2.3, 3.6, 2.1), 0.7, 0.7)
            p.drawRoundedRect(QRectF(4.2, 4.4, 7.6, 9.2), 1.4, 1.4)
            p.drawLine(QPointF(6.7, 7.0), QPointF(6.7, 11.2))
            p.drawLine(QPointF(9.3, 7.0), QPointF(9.3, 11.2))

        elif name == "minimize":
            p.drawLine(QPointF(3.6, 8.0), QPointF(12.4, 8.0))

        elif name == "maximize":
            p.drawRoundedRect(QRectF(3.5, 3.5, 9.0, 9.0), 1.5, 1.5)

        elif name == "restore":
            p.drawRoundedRect(QRectF(3.0, 5.2, 7.6, 7.6), 1.2, 1.2)
            path = QPainterPath()
            path.moveTo(5.4, 5.2)
            path.lineTo(5.4, 3.6)
            path.quadTo(5.4, 3.0, 6.0, 3.0)
            path.lineTo(12.4, 3.0)
            path.quadTo(13.0, 3.0, 13.0, 3.6)
            path.lineTo(13.0, 10.0)
            path.quadTo(13.0, 10.6, 12.4, 10.6)
            path.lineTo(10.6, 10.6)
            p.drawPath(path)

        elif name == "close":
            p.drawLine(QPointF(4.0, 4.0), QPointF(12.0, 12.0))
            p.drawLine(QPointF(12.0, 4.0), QPointF(4.0, 12.0))

        elif name == "record_dot":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qcolor)
            p.drawEllipse(QPointF(8.0, 8.0), 4.4, 4.4)

        elif name == "stop_square":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qcolor)
            p.drawRoundedRect(QRectF(4.2, 4.2, 7.6, 7.6), 1.6, 1.6)

        elif name == "clock":
            p.drawEllipse(QPointF(8.0, 8.0), 5.4, 5.4)
            p.drawLine(QPointF(8.0, 4.8), QPointF(8.0, 8.2))
            p.drawLine(QPointF(8.0, 8.2), QPointF(10.4, 9.6))

        elif name == "sparkles":
            star = QPainterPath()
            star.moveTo(8.0, 2.2)
            star.quadTo(8.3, 7.7, 13.8, 8.0)
            star.quadTo(8.3, 8.3, 8.0, 13.8)
            star.quadTo(7.7, 8.3, 2.2, 8.0)
            star.quadTo(7.7, 7.7, 8.0, 2.2)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qcolor)
            p.drawPath(star)

        elif name == "warning":
            tri = QPainterPath()
            tri.moveTo(8.0, 2.4)
            tri.lineTo(14.0, 13.0)
            tri.lineTo(2.0, 13.0)
            tri.closeSubpath()
            p.drawPath(tri)
            p.drawLine(QPointF(8.0, 6.2), QPointF(8.0, 9.2))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qcolor)
            p.drawEllipse(QPointF(8.0, 11.1), 0.9, 0.9)

        elif name == "mic":
            p.drawRoundedRect(QRectF(6.1, 2.0, 3.8, 6.8), 1.9, 1.9)
            p.drawArc(QRectF(4.2, 4.8, 7.6, 6.0), 180 * 16, 180 * 16)
            p.drawLine(QPointF(8.0, 10.8), QPointF(8.0, 13.4))
            p.drawLine(QPointF(5.8, 13.4), QPointF(10.2, 13.4))

        elif name == "notetaker_rec":
            p.drawEllipse(QPointF(8.0, 8.0), 5.6, 5.6)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qcolor)
            p.drawEllipse(QPointF(8.0, 8.0), 2.7, 2.7)

        elif name == "chevron_up":
            p.drawPolyline([QPointF(4.2, 10.0), QPointF(8.0, 6.0), QPointF(11.8, 10.0)])

        elif name == "snooze":
            p.drawEllipse(QPointF(8.0, 8.8), 4.7, 4.7)
            p.drawLine(QPointF(3.2, 4.4), QPointF(5.0, 2.8))
            p.drawLine(QPointF(12.8, 4.4), QPointF(11.0, 2.8))
            p.drawPolyline([QPointF(6.1, 8.9), QPointF(7.4, 10.3), QPointF(10.1, 7.4)])

        elif name == "clipboard_list":
            p.drawRoundedRect(QRectF(3.3, 3.3, 9.4, 10.6), 1.6, 1.6)
            p.drawRoundedRect(QRectF(5.7, 2.0, 4.6, 2.2), 0.8, 0.8)
            p.drawLine(QPointF(5.5, 6.8), QPointF(10.5, 6.8))
            p.drawLine(QPointF(5.5, 9.2), QPointF(10.5, 9.2))
            p.drawLine(QPointF(5.5, 11.6), QPointF(8.8, 11.6))

        elif name == "clipboard_paste":
            p.drawRoundedRect(QRectF(3.0, 3.3, 8.2, 10.6), 1.6, 1.6)
            p.drawRoundedRect(QRectF(5.1, 2.0, 4.0, 2.2), 0.8, 0.8)
            p.drawLine(QPointF(7.6, 9.6), QPointF(13.4, 9.6))
            p.drawPolyline([QPointF(11.2, 7.4), QPointF(13.4, 9.6), QPointF(11.2, 11.8)])

        elif name == "logout":
            p.drawPolyline(
                [
                    QPointF(6.8, 2.6),
                    QPointF(3.2, 2.6),
                    QPointF(3.2, 13.4),
                    QPointF(6.8, 13.4),
                ]
            )
            p.drawLine(QPointF(6.2, 8.0), QPointF(13.2, 8.0))
            p.drawPolyline([QPointF(10.5, 5.2), QPointF(13.2, 8.0), QPointF(10.5, 10.8)])
    finally:
        p.end()

    return QIcon(pix)
