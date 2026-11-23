"""CLI / GUI entry point for LocalFlow."""

from __future__ import annotations

import ctypes
import logging
import os
import sys

from PySide6.QtWidgets import QApplication

from localflow import __version__
from localflow.app import LocalFlowController
from localflow.config import get_app_data_dir
from localflow.single_instance import SingleInstanceGuard


def _set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("LocalFlow.Desktop.1")
    except Exception:
        pass


def _install_crash_hook() -> None:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

    def _handle_exception(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        try:
            log_path = get_app_data_dir() / "localflow.log"
            logger = logging.getLogger("localflow")
            if not logger.handlers:
                handler = logging.FileHandler(log_path, encoding="utf-8")
                handler.setFormatter(
                    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
                )
                logger.addHandler(handler)
            logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))
        except Exception:
            pass

    sys.excepthook = _handle_exception


def main() -> int:
    if "--version" in sys.argv[1:]:
        if sys.stdout is not None:
            print(f"LocalFlow {__version__}")
        return 0

    _install_crash_hook()
    _set_windows_app_id()

    app = QApplication(sys.argv)
    app.setApplicationName("LocalFlow")
    app.setApplicationVersion(__version__)
    app.setQuitOnLastWindowClosed(False)

    guard = SingleInstanceGuard()
    if not guard.try_acquire():
        return 0

    controller = LocalFlowController(app)
    guard.second_instance_activated.connect(controller.show_hub)

    def _on_quit() -> None:
        controller.shutdown()
        guard.release()

    app.aboutToQuit.connect(_on_quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
