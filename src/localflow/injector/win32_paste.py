"""Win32 clipboard-based text injection into the currently focused window."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import time
from collections.abc import Callable

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_CAPITAL = 0x14
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_INSERT = 0x2D
VK_C = 0x43
VK_V = 0x56
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4
VK_RMENU = 0xA5

_EXTENDED_VKS = frozenset({VK_RCONTROL, VK_RMENU, VK_LEFT, VK_RIGHT, VK_INSERT})
_MODIFIER_GROUPS: tuple[tuple[int, int, int], ...] = (
    (VK_SHIFT, VK_LSHIFT, VK_RSHIFT),
    (VK_MENU, VK_LMENU, VK_RMENU),
    (VK_CONTROL, VK_LCONTROL, VK_RCONTROL),
)

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
kernel32.GlobalAlloc.restype = wintypes.HGLOBAL

kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalLock.restype = ctypes.c_void_p

kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalUnlock.restype = wintypes.BOOL

kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalFree.restype = wintypes.HGLOBAL

user32.OpenClipboard.argtypes = [wintypes.HWND]
user32.OpenClipboard.restype = wintypes.BOOL

user32.CloseClipboard.argtypes = []
user32.CloseClipboard.restype = wintypes.BOOL

user32.EmptyClipboard.argtypes = []
user32.EmptyClipboard.restype = wintypes.BOOL

user32.GetClipboardData.argtypes = [wintypes.UINT]
user32.GetClipboardData.restype = wintypes.HANDLE

user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
user32.SetClipboardData.restype = wintypes.HANDLE

user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
user32.IsClipboardFormatAvailable.restype = wintypes.BOOL

user32.GetClipboardSequenceNumber.argtypes = []
user32.GetClipboardSequenceNumber.restype = wintypes.DWORD

user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short

user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
user32.MapVirtualKeyW.restype = wintypes.UINT

user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
user32.RegisterClipboardFormatW.restype = wintypes.UINT


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("_input",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("_input", _INPUT_UNION),
    ]


user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT


def _open_clipboard_with_retry(retries: int = 8, delay_sec: float = 0.015) -> bool:
    for _ in range(retries):
        if user32.OpenClipboard(None):
            return True
        time.sleep(delay_sec)
    return False


def get_clipboard_text() -> str | None:
    """Read current CF_UNICODETEXT from clipboard if available."""
    if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
        return None
    if not _open_clipboard_with_retry():
        return None
    try:
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return None
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            return None
        try:
            return ctypes.wstring_at(ptr)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def set_clipboard_text(text: str) -> bool:
    """Write Unicode text into the Windows clipboard."""
    if not _open_clipboard_with_retry():
        return False
    try:
        user32.EmptyClipboard()
        encoded = (text + "\0").encode("utf-16-le")
        h_mem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(encoded))
        if not h_mem:
            return False
        ptr = kernel32.GlobalLock(h_mem)
        if not ptr:
            kernel32.GlobalFree(h_mem)
            return False
        ctypes.memmove(ptr, encoded, len(encoded))
        kernel32.GlobalUnlock(h_mem)
        if not user32.SetClipboardData(CF_UNICODETEXT, h_mem):
            kernel32.GlobalFree(h_mem)
            return False
        return True
    finally:
        user32.CloseClipboard()


def _send_keys(events: list[tuple[int, bool]]) -> None:
    """Send a sequence of (vk_code, is_keyup) keyboard events via SendInput."""
    inputs = (INPUT * len(events))()
    for i, (vk, is_up) in enumerate(events):
        flags = KEYEVENTF_KEYUP if is_up else 0
        if vk in _EXTENDED_VKS:
            flags |= KEYEVENTF_EXTENDEDKEY
        scan = int(user32.MapVirtualKeyW(vk, 0))
        inputs[i].type = INPUT_KEYBOARD
        inputs[i].ki = KEYBDINPUT(
            wVk=vk,
            wScan=scan,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )
    user32.SendInput(len(events), inputs, ctypes.sizeof(INPUT))


def _release_held_modifiers(
    *,
    get_async_key_state: Callable[[int], int] | None = None,
    send_keys: Callable[[list[tuple[int, bool]]], None] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """Release held Shift/Alt/Ctrl modifiers before sending synthetic shortcuts."""
    key_state_fn = get_async_key_state or user32.GetAsyncKeyState
    send_fn = send_keys or _send_keys
    sleep_fn = sleep or time.sleep

    to_release: list[tuple[int, bool]] = []
    for generic_vk, left_vk, right_vk in _MODIFIER_GROUPS:
        released_specific = False
        for vk in (left_vk, right_vk):
            if key_state_fn(vk) & 0x8000:
                to_release.append((vk, True))
                released_specific = True
        if not released_specific and (key_state_fn(generic_vk) & 0x8000):
            to_release.append((generic_vk, True))

    if to_release:
        send_fn(to_release)
        sleep_fn(0.015)


def send_ctrl_c(
    *,
    get_async_key_state: Callable[[int], int] | None = None,
    send_keys: Callable[[list[tuple[int, bool]]], None] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """Send Ctrl+C key combination via Win32 SendInput with inter-event settling delays."""
    _release_held_modifiers(
        get_async_key_state=get_async_key_state,
        send_keys=send_keys,
        sleep=sleep,
    )
    send_fn = send_keys or _send_keys
    sleep_fn = sleep or time.sleep
    send_fn([(VK_CONTROL, False)])
    sleep_fn(0.015)
    send_fn([(VK_C, False), (VK_C, True)])
    sleep_fn(0.015)
    send_fn([(VK_CONTROL, True)])


def send_ctrl_insert(
    *,
    get_async_key_state: Callable[[int], int] | None = None,
    send_keys: Callable[[list[tuple[int, bool]]], None] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """Send Ctrl+Insert (universal Windows copy shortcut) via Win32 SendInput."""
    _release_held_modifiers(
        get_async_key_state=get_async_key_state,
        send_keys=send_keys,
        sleep=sleep,
    )
    send_fn = send_keys or _send_keys
    sleep_fn = sleep or time.sleep
    send_fn([(VK_CONTROL, False)])
    sleep_fn(0.015)
    send_fn([(VK_INSERT, False), (VK_INSERT, True)])
    sleep_fn(0.015)
    send_fn([(VK_CONTROL, True)])


def send_ctrl_v(
    *,
    get_async_key_state: Callable[[int], int] | None = None,
    send_keys: Callable[[list[tuple[int, bool]]], None] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """Send Ctrl+V key combination via Win32 SendInput with inter-event settling delays."""
    _release_held_modifiers(
        get_async_key_state=get_async_key_state,
        send_keys=send_keys,
        sleep=sleep,
    )
    send_fn = send_keys or _send_keys
    sleep_fn = sleep or time.sleep
    send_fn([(VK_CONTROL, False)])
    sleep_fn(0.015)
    send_fn([(VK_V, False), (VK_V, True)])
    sleep_fn(0.015)
    send_fn([(VK_CONTROL, True)])


def clear_clipboard() -> bool:
    if not _open_clipboard_with_retry():
        return False
    try:
        user32.EmptyClipboard()
        return True
    finally:
        user32.CloseClipboard()


def should_prepend_space(left_char: str | None, text_to_paste: str) -> bool:
    """Return True if a leading space should be added before pasting text."""
    if not left_char or not text_to_paste:
        return False
    if left_char.isspace() or left_char in "([{\"'«„":
        return False
    if text_to_paste[0].isspace() or text_to_paste[0] in ".,!?:;)]}»\"":
        return False
    return True


def _is_vscode_empty_selection_copy() -> bool:
    """Return True if clipboard holds a VS Code/Monaco whole-line copy from an empty selection."""
    try:
        fmt = int(user32.RegisterClipboardFormatW("vscode-editor-data"))
        if not fmt or not user32.IsClipboardFormatAvailable(fmt):
            return False
        if not _open_clipboard_with_retry(retries=3, delay_sec=0.005):
            return False
        try:
            handle = user32.GetClipboardData(fmt)
            if not handle:
                return False
            ptr = kernel32.GlobalLock(handle)
            if not ptr:
                return False
            try:
                raw = ctypes.string_at(ptr)
            finally:
                kernel32.GlobalUnlock(handle)
        finally:
            user32.CloseClipboard()
        text = raw.decode("utf-8", errors="ignore").replace(" ", "")
        return '"isFromEmptySelection":true' in text
    except Exception:
        return False


def _has_active_selection_win32(
    *,
    is_terminal: bool = False,
    get_clipboard_text_fn: Callable[[], str | None] | None = None,
    get_seq_fn: Callable[[], int] | None = None,
    get_async_key_state: Callable[[int], int] | None = None,
    send_keys: Callable[[list[tuple[int, bool]]], None] | None = None,
    monotonic: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> bool:
    """Probe whether the active window has selected text without moving the caret."""
    read_clip = get_clipboard_text_fn or get_clipboard_text
    seq_fn = get_seq_fn or user32.GetClipboardSequenceNumber
    mono_fn = monotonic or time.monotonic
    sleep_fn = sleep or time.sleep

    seq_before = seq_fn()
    if is_terminal:
        send_ctrl_insert(
            get_async_key_state=get_async_key_state,
            send_keys=send_keys,
            sleep=sleep_fn,
        )
    else:
        send_ctrl_c(
            get_async_key_state=get_async_key_state,
            send_keys=send_keys,
            sleep=sleep_fn,
        )

    deadline = mono_fn() + 0.060
    changed = False
    while True:
        if seq_fn() != seq_before:
            changed = True
            break
        now = mono_fn()
        if now >= deadline:
            break
        sleep_fn(min(0.004, deadline - now))

    if not changed:
        return False
    if get_clipboard_text_fn is None and _is_vscode_empty_selection_copy():
        return False
    copied = read_clip()
    return bool(copied)


def _peek_left_char(*, is_terminal: bool = False) -> str | None:
    """Peek at the single character immediately to the left of the caret."""
    if not clear_clipboard():
        return None

    _release_held_modifiers()
    seq_before = user32.GetClipboardSequenceNumber()

    # Select 1 char to the left using explicit LSHIFT + extended VK_LEFT (safe with NumLock ON)
    _send_keys([(VK_LSHIFT, False)])
    time.sleep(0.010)
    _send_keys([(VK_LEFT, False), (VK_LEFT, True)])
    time.sleep(0.010)
    _send_keys([(VK_LSHIFT, True)])
    time.sleep(0.015)

    if is_terminal:
        send_ctrl_insert()
    else:
        send_ctrl_c()

    deadline = time.monotonic() + 0.150
    while time.monotonic() < deadline:
        if user32.GetClipboardSequenceNumber() != seq_before:
            break
        time.sleep(0.004)

    copied = get_clipboard_text()
    if copied:
        # Collapse selection back to the right edge (original caret position after the character)
        _send_keys([(VK_RIGHT, False), (VK_RIGHT, True)])
        time.sleep(0.015)

    if copied == "\r\n":
        return "\n"
    if copied and len(copied) == 1:
        return copied
    return None


class Win32TextInjector:
    """Captures selected text and pastes text into the active window via clipboard + Ctrl+C/V."""

    def __init__(
        self,
        *,
        get_clipboard_text: Callable[[], str | None] | None = None,
        set_clipboard_text: Callable[[str], bool] | None = None,
        get_clipboard_sequence_number: Callable[[], int] | None = None,
        get_async_key_state: Callable[[int], int] | None = None,
        send_keys: Callable[[list[tuple[int, bool]]], None] | None = None,
        peek_left_char: Callable[[], str | None] | None = None,
        has_active_selection: Callable[[], bool] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._get_clipboard_text = get_clipboard_text
        self._set_clipboard_text = set_clipboard_text
        self._get_clipboard_sequence_number = get_clipboard_sequence_number
        self._get_async_key_state = get_async_key_state
        self._send_keys = send_keys
        self._peek_left_char = peek_left_char
        self._has_active_selection = has_active_selection
        self._monotonic = monotonic
        self._sleep = sleep

    def capture_selected_text(
        self,
        timeout_ms: int = 350,
        restore_clipboard: bool = True,
    ) -> str | None:
        """Copy selected text from the active window without pre-clearing clipboard."""
        get_text_fn = self._get_clipboard_text or get_clipboard_text
        set_text_fn = self._set_clipboard_text or set_clipboard_text
        get_seq_fn = self._get_clipboard_sequence_number or user32.GetClipboardSequenceNumber
        key_state_fn = self._get_async_key_state or user32.GetAsyncKeyState
        monotonic_fn = self._monotonic or time.monotonic
        sleep_fn = self._sleep or time.sleep

        # Wait briefly if CapsLock is still physically held from Ctrl + 2x CapsLock
        caps_deadline = monotonic_fn() + 0.12
        while (key_state_fn(VK_CAPITAL) & 0x8000) and monotonic_fn() < caps_deadline:
            sleep_fn(0.010)

        previous_text = get_text_fn() if restore_clipboard else None
        seq_before = get_seq_fn()

        total_sec = max(0.0, timeout_ms / 1000.0)
        start_ts = monotonic_fn()
        deadline = start_ts + total_sec
        first_phase_deadline = start_ts + (total_sec * 0.6 if total_sec >= 0.12 else total_sec)

        send_ctrl_c(
            get_async_key_state=self._get_async_key_state,
            send_keys=self._send_keys,
            sleep=self._sleep,
        )

        changed = False
        while True:
            if get_seq_fn() != seq_before:
                changed = True
                break
            now = monotonic_fn()
            if now >= first_phase_deadline:
                break
            sleep_fn(min(0.005, first_phase_deadline - now))

        # Fallback: Ctrl+Insert handles terminals, Electron/Chromium races, and physical Ctrl release
        if not changed and monotonic_fn() < deadline:
            send_ctrl_insert(
                get_async_key_state=self._get_async_key_state,
                send_keys=self._send_keys,
                sleep=self._sleep,
            )
            while True:
                if get_seq_fn() != seq_before:
                    changed = True
                    break
                now = monotonic_fn()
                if now >= deadline:
                    break
                sleep_fn(min(0.005, deadline - now))

        if not changed:
            return None

        try:
            copied = get_text_fn()
        finally:
            if restore_clipboard and previous_text is not None:
                set_text_fn(previous_text)

        if copied is None or not copied.strip():
            return None
        return copied

    def inject(
        self,
        text: str,
        auto_space: bool = True,
        restore_clipboard: bool = True,
        paste_delay_ms: int = 120,
        is_terminal: bool = False,
    ) -> bool:
        if not text:
            return False

        get_text_fn = self._get_clipboard_text or get_clipboard_text
        set_text_fn = self._set_clipboard_text or set_clipboard_text
        sleep_fn = self._sleep or time.sleep

        previous_text = get_text_fn() if restore_clipboard else None

        if auto_space:
            if self._has_active_selection is not None:
                has_selection = self._has_active_selection()
            elif (
                self._peek_left_char is not None
                and self._send_keys is None
                and self._get_clipboard_sequence_number is None
            ):
                # Preserve legacy unit tests that inject only a custom peek_left_char double
                has_selection = False
            else:
                has_selection = _has_active_selection_win32(
                    is_terminal=is_terminal,
                    get_clipboard_text_fn=self._get_clipboard_text,
                    get_seq_fn=self._get_clipboard_sequence_number,
                    get_async_key_state=self._get_async_key_state,
                    send_keys=self._send_keys,
                    monotonic=self._monotonic,
                    sleep=self._sleep,
                )

            if not has_selection:
                left_char = (
                    self._peek_left_char()
                    if self._peek_left_char is not None
                    else _peek_left_char(is_terminal=is_terminal)
                )
                if should_prepend_space(left_char, text):
                    text = " " + text

        if not set_text_fn(text):
            if restore_clipboard and previous_text is not None:
                set_text_fn(previous_text)
            return False

        sleep_fn(0.030)
        send_ctrl_v(
            get_async_key_state=self._get_async_key_state,
            send_keys=self._send_keys,
            sleep=self._sleep,
        )

        if restore_clipboard and previous_text is not None:
            sleep_fn(max(0.15, paste_delay_ms / 1000.0))
            set_text_fn(previous_text)

        return True


def capture_selected_text(
    timeout_ms: int = 350,
    restore_clipboard: bool = True,
) -> str | None:
    """Convenience wrapper for Win32TextInjector().capture_selected_text()."""
    return Win32TextInjector().capture_selected_text(
        timeout_ms=timeout_ms,
        restore_clipboard=restore_clipboard,
    )
