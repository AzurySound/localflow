"""Windows per-user autostart management (HKCU\\...\\Run\\LocalFlow, no admin required)."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

logger = logging.getLogger("localflow")

RUN_REGISTRY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
APP_VALUE_NAME = "LocalFlow"


def build_autostart_command() -> str:
    """Return the exact command line to register in Windows Run key."""
    if getattr(sys, "frozen", False):
        return f'"{Path(sys.executable).resolve()}"'
    pyw = Path(sys.executable).resolve().with_name("pythonw.exe")
    exe = pyw if pyw.is_file() else Path(sys.executable).resolve()
    return f'"{exe}" -m localflow'


def is_autostart_enabled() -> bool:
    """Check whether LocalFlow is currently registered in HKCU Run."""
    if sys.platform != "win32":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_REGISTRY_PATH, 0, winreg.KEY_READ) as key:
            val, _ = winreg.QueryValueEx(key, APP_VALUE_NAME)
            return bool(str(val).strip())
    except FileNotFoundError:
        return False
    except OSError:
        return False


def set_autostart_enabled(enabled: bool, *, command: str | None = None) -> bool:
    """Enable or disable Windows login autostart for the current user."""
    if sys.platform != "win32":
        return False
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            RUN_REGISTRY_PATH,
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            if enabled:
                cmd = command or build_autostart_command()
                winreg.SetValueEx(key, APP_VALUE_NAME, 0, winreg.REG_SZ, cmd)
            else:
                try:
                    winreg.DeleteValue(key, APP_VALUE_NAME)
                except FileNotFoundError:
                    pass
        return True
    except OSError:
        logger.exception("Failed to update Windows autostart registry entry")
        return False
