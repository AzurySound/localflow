"""Pure double-tap / single-tap state machine (no OS or GUI dependencies)."""

from __future__ import annotations

import time
from typing import Literal

ActionType = Literal["dictate", "command", "stop"]


class DoubleTapDetector:
    """Detects double-tap to start (plain or Ctrl-modified) and optional single-tap to stop."""

    def __init__(self, threshold_ms: int = 320) -> None:
        self.threshold_sec = threshold_ms / 1000.0
        self._is_down = False
        self._last_tap_ts: float | None = None
        self._last_tap_ctrl = False
        self._single_tap_mode = False
        self.last_action: ActionType | None = None

    def set_threshold_ms(self, threshold_ms: int) -> None:
        self.threshold_sec = threshold_ms / 1000.0

    def set_single_tap_mode(self, enabled: bool) -> None:
        """When True, the next single key press triggers immediately (used to stop recording)."""
        self._single_tap_mode = enabled
        self._last_tap_ts = None
        self._last_tap_ctrl = False

    def reset(self) -> None:
        self._is_down = False
        self._last_tap_ts = None
        self._last_tap_ctrl = False
        self.last_action = None

    def on_key_event(
        self,
        is_down: bool,
        timestamp: float | None = None,
        *,
        ctrl_held: bool = False,
    ) -> bool:
        """Process a key event. Returns True when triggered and updates last_action."""
        now = time.monotonic() if timestamp is None else timestamp

        if not is_down:
            self._is_down = False
            return False

        if self._is_down:
            return False

        self._is_down = True

        if self._single_tap_mode:
            self._single_tap_mode = False
            self._last_tap_ts = None
            self._last_tap_ctrl = False
            self.last_action = "stop"
            return True

        if self._last_tap_ts is not None and (now - self._last_tap_ts) <= self.threshold_sec:
            if self._last_tap_ctrl == ctrl_held:
                self._last_tap_ts = None
                self._last_tap_ctrl = False
                self.last_action = "command" if ctrl_held else "dictate"
                return True
            self._last_tap_ts = now
            self._last_tap_ctrl = ctrl_held
            return False

        self._last_tap_ts = now
        self._last_tap_ctrl = ctrl_held
        return False
