"""Generate multi-resolution Windows .ico (16..256 px) using PySide6 without Pillow."""

from __future__ import annotations

from pathlib import Path
import struct

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPixmap


def _render_png_bytes(size: int) -> bytes:
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)

    scale = size / 64.0
    margin = 4.0 * scale
    radius = 14.0 * scale
    p.setBrush(QColor("#14161e"))
    p.drawRoundedRect(QRectF(margin, margin, size - 2 * margin, size - 2 * margin), radius, radius)

    p.setBrush(QColor("#fafafa"))
    heights = (16.0, 30.0, 22.0, 34.0, 14.0)
    bar_w = 4.6 * scale
    for i, h in enumerate(heights):
        bh = h * scale
        bx = (12.8 + i * 8.4) * scale
        by = (size - bh) / 2.0
        p.drawRoundedRect(QRectF(bx, by, bar_w, bh), bar_w / 2.0, bar_w / 2.0)
    p.end()

    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    pix.save(buf, "PNG")
    buf.close()
    return bytes(ba.data())


def build_ico(output_path: Path, sizes: tuple[int, ...] = (16, 24, 32, 48, 64, 128, 256)) -> Path:
    app = QGuiApplication.instance() or QGuiApplication([])
    _ = app
    images = [(sz, _render_png_bytes(sz)) for sz in sizes]

    count = len(images)
    header = struct.pack("<HHH", 0, 1, count)
    offset = 6 + 16 * count
    entries = bytearray()
    payload = bytearray()

    for sz, png in images:
        dim = 0 if sz >= 256 else sz
        entries.extend(struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(png), offset))
        payload.extend(png)
        offset += len(png)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(header + bytes(entries) + bytes(payload))
    return output_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = build_ico(root / "assets" / "localflow.ico")
    print(f"Generated {out} ({out.stat().st_size} bytes)")
