"""Tests for Portable mode storage, Whisper progress downloader, and pill/settings onboarding UI."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent

from localflow.config import (
    PORTABLE_MARKER_FILENAME,
    AppConfig,
    get_app_data_dir,
    get_whisper_models_dir,
    is_portable_mode,
)
from localflow.llm.downloader import (
    DownloadProgress,
    download_whisper_model,
    find_local_whisper_model_path,
    get_whisper_model_dir,
    is_whisper_model_downloaded,
)
from localflow.storage import HistoryStorage
from localflow.ui.overlay_pill import FloatingPillWidget, PillMode
from localflow.ui.settings_window import SettingsWindow


def test_portable_mode_detection_and_data_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LOCALFLOW_PORTABLE", raising=False)
    monkeypatch.delenv("LOCALFLOW_DATA_DIR", raising=False)

    exe_dir = tmp_path / "LocalFlowPortable"
    exe_dir.mkdir()
    assert is_portable_mode(exe_dir) is False

    # Adding portable.flag activates portable mode and routes data into <exe_dir>/data
    (exe_dir / PORTABLE_MARKER_FILENAME).write_text("portable", encoding="ascii")
    assert is_portable_mode(exe_dir) is True

    data_dir = get_app_data_dir(exe_dir)
    assert data_dir == exe_dir / "data"
    assert data_dir.is_dir()

    whisper_dir = get_whisper_models_dir(data_dir)
    assert whisper_dir == exe_dir / "data" / "models" / "whisper"
    assert whisper_dir.is_dir()


def test_download_whisper_model_stores_in_target_dir_with_progress(tmp_path: Path) -> None:
    data_dir = tmp_path / "portable_data"
    fake_model_bin = b"W" * 1_200_000  # > 1 MB so is_whisper_model_downloaded passes
    fake_config = b'{"model": "small"}'

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/api/models/" in url:
            payload = {
                "siblings": [
                    {"rfilename": "config.json", "size": len(fake_config)},
                    {"rfilename": "model.bin", "size": len(fake_model_bin)},
                    {"rfilename": "README.md", "size": 500},  # ignored
                ]
            }
            return httpx.Response(200, content=json.dumps(payload).encode("utf-8"))
        if url.endswith("/config.json"):
            return httpx.Response(
                200,
                content=fake_config,
                headers={"content-length": str(len(fake_config))},
            )
        if url.endswith("/model.bin"):
            return httpx.Response(
                200,
                content=fake_model_bin,
                headers={"content-length": str(len(fake_model_bin))},
            )
        return httpx.Response(404)

    events: list[DownloadProgress] = []
    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        dest = download_whisper_model(
            "small",
            base_dir=data_dir,
            on_progress=events.append,
            client=client,
        )
    finally:
        client.close()

    assert dest == get_whisper_model_dir("small", base_dir=data_dir)
    assert is_whisper_model_downloaded("small", base_dir=data_dir) is True
    assert not (dest / "README.md").exists()
    assert find_local_whisper_model_path("small", base_dir=data_dir, allow_global_cache=False) == dest
    assert any(ev.percent == 100 for ev in events)


def test_pill_downloading_mode_and_click_signal(qapp) -> None:
    pill = FloatingPillWidget()
    clicked: list[bool] = []
    pill.download_clicked.connect(lambda: clicked.append(True))

    pill.show_downloading("Whisper small", 42)
    assert pill._mode == PillMode.DOWNLOADING
    assert pill._download_percent == 42
    assert pill.width() >= 156

    # Render paintEvent cleanly in both dark and light themes
    pill.set_theme("dark")
    pill.grab()
    pill.set_theme("light")
    pill.grab()

    # Clicking the downloading pill emits download_clicked
    pos = QPointF(pill.width() / 2.0, pill.height() / 2.0)
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        pos,
        pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        pos,
        pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    pill.mousePressEvent(press)
    pill.mouseReleaseEvent(release)
    assert clicked == [True]
    pill.close()


def test_settings_active_download_card_updates_and_clears(qapp, tmp_path: Path) -> None:
    storage = HistoryStorage(tmp_path / "history.db")
    win = SettingsWindow(AppConfig(language="en"), storage)
    assert win._active_dl_card.isHidden() is True

    dest = tmp_path / "data" / "models" / "whisper" / "small"
    prog = DownloadProgress(
        stage="Whisper small",
        downloaded_bytes=200 * 1024 * 1024,
        total_bytes=460 * 1024 * 1024,
        speed_mb_s=12.5,
    )
    win.set_external_download_progress("Whisper small", prog, dest)
    assert win._active_dl_card.isHidden() is False
    assert "43%" in win._active_dl_status_lbl.text()
    assert str(dest) in win._active_dl_path_lbl.text()

    win.clear_external_download()
    assert win._active_dl_card.isHidden() is True
    win.close()


def test_dropdowns_and_spinboxes_ignore_mouse_wheel_and_scrollbar_theme(qapp, tmp_path: Path) -> None:
    from localflow.ui.theme import build_hub_stylesheet

    storage = HistoryStorage(tmp_path / "history.db")
    win = SettingsWindow(AppConfig(language="en"), storage)

    def _make_wheel() -> QWheelEvent:
        return QWheelEvent(
            QPointF(10.0, 10.0),
            QPointF(10.0, 10.0),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )

    orig_combo_idx = win._profile_combo.currentIndex()
    w_combo = _make_wheel()
    win._profile_combo.wheelEvent(w_combo)
    assert w_combo.isAccepted() is False
    assert win._profile_combo.currentIndex() == orig_combo_idx

    orig_spin_val = win._double_tap_spin.value()
    w_spin = _make_wheel()
    win._double_tap_spin.wheelEvent(w_spin)
    assert w_spin.isAccepted() is False
    assert win._double_tap_spin.value() == orig_spin_val

    orig_dspin_val = win._silence_auto_stop_spin.value()
    w_dspin = _make_wheel()
    win._silence_auto_stop_spin.wheelEvent(w_dspin)
    assert w_dspin.isAccepted() is False
    assert win._silence_auto_stop_spin.value() == orig_dspin_val

    dark_qss = build_hub_stylesheet("dark")
    assert "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical" in dark_qss
    assert "#141417" in dark_qss  # dark track background
    assert "#52525b" in dark_qss  # lighter moving handle
    win.close()
