"""Global hotkey listener for double-tap CapsLock and Esc."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

from localflow.hotkey.detector import DoubleTapDetector

WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105
WM_QUIT = 0x0012

VK_CONTROL = 0x11
VK_CAPITAL = 0x14
VK_ESCAPE = 0x1B
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
_CTRL_VKS = frozenset({VK_CONTROL, VK_LCONTROL, VK_RCONTROL})
LLKHF_INJECTED = 0x00000010


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


LRESULT = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
user32.SetWindowsHookExW.restype = wintypes.HHOOK

user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.CallNextHookEx.restype = LRESULT

user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
user32.UnhookWindowsHookEx.restype = wintypes.BOOL

user32.GetMessageW.argtypes = [
    ctypes.POINTER(wintypes.MSG),
    wintypes.HWND,
    wintypes.UINT,
    wintypes.UINT,
]
user32.GetMessageW.restype = wintypes.BOOL

user32.PostThreadMessageW.argtypes = [
    wintypes.DWORD,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
]
user32.PostThreadMessageW.restype = wintypes.BOOL

kernel32.GetCurrentThreadId.argtypes = []
kernel32.GetCurrentThreadId.restype = wintypes.DWORD

user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short


def _is_ctrl_async_down() -> bool:
    return bool(user32.GetAsyncKeyState(VK_CONTROL) & 0x8000)


class GlobalHotkeyListener(QObject):
    """Low-level Windows keyboard hook emitting Qt signals on double-tap CapsLock, Ctrl+double-tap CapsLock, and Esc."""

    toggle_requested = Signal()
    command_requested = Signal()
    cancel_requested = Signal()

    def __init__(
        self,
        threshold_ms: int = 320,
        parent: QObject | None = None,
        *,
        ctrl_async_state_fn: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._detector = DoubleTapDetector(threshold_ms=threshold_ms)
        self._ctrl_async_state_fn = ctrl_async_state_fn or _is_ctrl_async_down
        self._pressed_ctrl_vks: set[int] = set()
        self._cancel_enabled = False
        self._single_tap_stop = False
        self._swallow_next_capslock_up = False
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._hook_handle: wintypes.HHOOK | None = None
        self._hook_proc: HOOKPROC | None = None
        self._ready = threading.Event()

    def set_threshold_ms(self, threshold_ms: int) -> None:
        with self._lock:
            self._detector.set_threshold_ms(threshold_ms)

    def set_cancel_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._cancel_enabled = enabled

    def set_recording_active(self, active: bool) -> None:
        """Enable single-tap CapsLock stop and Esc cancel while recording."""
        with self._lock:
            self._cancel_enabled = active
            self._single_tap_stop = active
            self._detector.set_single_tap_mode(active)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._thread = threading.Thread(target=self._run_loop, name="CapsLockHookThread", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2.0)

    def stop(self) -> None:
        if self._thread_id is not None:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        self._thread_id = None
        with self._lock:
            self._pressed_ctrl_vks.clear()
            self._swallow_next_capslock_up = False
            self._detector.reset()

    def _handle_key_event(
        self,
        w_param: int,
        vk_code: int,
        flags: int = 0,
        *,
        timestamp: float | None = None,
    ) -> bool:
        """Handle a keyboard hook event. Returns True if the event should be swallowed."""
        if flags & LLKHF_INJECTED:
            return False

        is_down = w_param in (WM_KEYDOWN, WM_SYSKEYDOWN)
        is_up = w_param in (WM_KEYUP, WM_SYSKEYUP)
        if not is_down and not is_up:
            return False

        if vk_code in _CTRL_VKS:
            with self._lock:
                if is_down:
                    self._pressed_ctrl_vks.add(vk_code)
                elif vk_code == VK_CONTROL:
                    self._pressed_ctrl_vks.clear()
                else:
                    self._pressed_ctrl_vks.discard(vk_code)
                    self._pressed_ctrl_vks.discard(VK_CONTROL)
            return False

        if vk_code == VK_CAPITAL:
            if is_down:
                action_to_emit: str | None = None
                with self._lock:
                    ctrl_held = bool(self._pressed_ctrl_vks) or bool(self._ctrl_async_state_fn())
                    swallow = self._single_tap_stop or self._swallow_next_capslock_up
                    if self._detector.on_key_event(
                        is_down=True,
                        timestamp=timestamp,
                        ctrl_held=ctrl_held,
                    ):
                        action_to_emit = self._detector.last_action
                        self._single_tap_stop = False
                        if swallow:
                            self._swallow_next_capslock_up = True
                if action_to_emit == "command":
                    self.command_requested.emit()
                elif action_to_emit in ("dictate", "stop"):
                    self.toggle_requested.emit()
                return swallow

            with self._lock:
                self._detector.on_key_event(is_down=False, timestamp=timestamp)
                swallow_up = self._swallow_next_capslock_up
                if swallow_up:
                    self._swallow_next_capslock_up = False
            return swallow_up

        if vk_code == VK_ESCAPE:
            with self._lock:
                can_cancel = self._cancel_enabled
            if can_cancel and is_down:
                self.cancel_requested.emit()
                return True

        return False

    def _process_hook_message(self, n_code: int, w_param: int, l_param: int) -> int:
        if n_code >= 0 and l_param:
            kb = ctypes.cast(l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            if self._handle_key_event(w_param=w_param, vk_code=kb.vkCode, flags=kb.flags):
                return 1
        return user32.CallNextHookEx(self._hook_handle, n_code, w_param, l_param)

    def _run_loop(self) -> None:
        self._thread_id = kernel32.GetCurrentThreadId()

        def _low_level_handler(n_code: int, w_param: int, l_param: int) -> int:
            return self._process_hook_message(n_code, w_param, l_param)

        self._hook_proc = HOOKPROC(_low_level_handler)
        self._hook_handle = user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._hook_proc, None, 0)
        self._ready.set()

        if not self._hook_handle:
            return

        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        user32.UnhookWindowsHookEx(self._hook_handle)
        self._hook_handle = None
        self._hook_proc = None
