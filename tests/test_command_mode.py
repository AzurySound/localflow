import ctypes
import unittest
from unittest.mock import patch

from PySide6.QtCore import QCoreApplication

from localflow.hotkey import capslock_hook
from localflow.hotkey.capslock_hook import (
    KBDLLHOOKSTRUCT,
    LLKHF_INJECTED,
    VK_CAPITAL,
    VK_CONTROL,
    VK_ESCAPE,
    VK_LCONTROL,
    VK_RCONTROL,
    WM_KEYDOWN,
    WM_KEYUP,
    GlobalHotkeyListener,
)
from localflow.injector.win32_paste import (
    VK_C,
    VK_LMENU,
    VK_LSHIFT,
    VK_MENU,
    VK_RMENU,
    VK_RSHIFT,
    VK_SHIFT,
    VK_V,
    Win32TextInjector,
)


class _FakeClipboardHarness:
    """Self-contained Win32 clipboard + keyboard double for command-mode tests."""

    def __init__(
        self,
        *,
        initial_clipboard: str | None = "saved clip",
        selected_text: str | None = "selected",
        copy_succeeds: bool = True,
    ) -> None:
        self.clipboard_text = initial_clipboard
        self.selected_text = selected_text
        self.copy_succeeds = copy_succeeds
        self.seq = 10
        self.held_vks: set[int] = set()
        self.set_clipboard_calls: list[str] = []
        self.modifiers_during_v: list[set[int]] = []
        self.peek_calls = 0
        self.now = 0.0

    def get_clipboard_text(self) -> str | None:
        return self.clipboard_text

    def set_clipboard_text(self, text: str) -> bool:
        self.set_clipboard_calls.append(text)
        self.clipboard_text = text
        self.seq += 1
        return True

    def get_clipboard_sequence_number(self) -> int:
        return self.seq

    def get_async_key_state(self, vk: int) -> int:
        if vk in self.held_vks:
            return -32768
        if vk == VK_CONTROL and (VK_LCONTROL in self.held_vks or VK_RCONTROL in self.held_vks):
            return -32768
        if vk == VK_SHIFT and (VK_LSHIFT in self.held_vks or VK_RSHIFT in self.held_vks):
            return -32768
        if vk == VK_MENU and (VK_LMENU in self.held_vks or VK_RMENU in self.held_vks):
            return -32768
        return 0

    def send_keys(self, events: list[tuple[int, bool]]) -> None:
        for vk, is_up in events:
            if is_up:
                self.held_vks.discard(vk)
                if vk == VK_CONTROL:
                    self.held_vks.discard(VK_LCONTROL)
                    self.held_vks.discard(VK_RCONTROL)
            else:
                self.held_vks.add(vk)
                ctrl_down = bool(self.held_vks & {VK_CONTROL, VK_LCONTROL, VK_RCONTROL})
                shift_or_alt = bool(
                    self.held_vks & {VK_SHIFT, VK_LSHIFT, VK_RSHIFT, VK_MENU, VK_LMENU, VK_RMENU}
                )
                if vk == VK_C and self.copy_succeeds and ctrl_down and not shift_or_alt:
                    self.clipboard_text = self.selected_text
                    self.seq += 1
                elif vk == VK_V:
                    self.modifiers_during_v.append(set(self.held_vks))
        self.held_vks.discard(VK_C)
        self.held_vks.discard(VK_V)

    def peek_left_char(self) -> str | None:
        self.peek_calls += 1
        return "."

    def monotonic(self) -> float:
        return self.now

    def sleep(self, sec: float) -> None:
        self.now += max(0.0, sec)

    def make_injector(self) -> Win32TextInjector:
        return Win32TextInjector(
            get_clipboard_text=self.get_clipboard_text,
            set_clipboard_text=self.set_clipboard_text,
            get_clipboard_sequence_number=self.get_clipboard_sequence_number,
            get_async_key_state=self.get_async_key_state,
            send_keys=self.send_keys,
            peek_left_char=self.peek_left_char,
            monotonic=self.monotonic,
            sleep=self.sleep,
        )


def _ensure_qt_app() -> QCoreApplication:
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


class _SignalRecorder:
    def __init__(self, listener: GlobalHotkeyListener) -> None:
        self.toggles = 0
        self.commands = 0
        self.cancels = 0
        listener.toggle_requested.connect(self._on_toggle)
        listener.command_requested.connect(self._on_command)
        listener.cancel_requested.connect(self._on_cancel)

    def _on_toggle(self) -> None:
        self.toggles += 1

    def _on_command(self) -> None:
        self.commands += 1

    def _on_cancel(self) -> None:
        self.cancels += 1


class TestGlobalHotkeyListenerCommandMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._qt_app = _ensure_qt_app()

    def _make_listener(self, *, async_ctrl: bool = False) -> tuple[GlobalHotkeyListener, _SignalRecorder]:
        listener = GlobalHotkeyListener(
            threshold_ms=320,
            ctrl_async_state_fn=lambda: async_ctrl,
        )
        return listener, _SignalRecorder(listener)

    def test_ctrl_double_capslock_emits_command_requested_for_all_ctrl_vks(self) -> None:
        for ctrl_vk in (VK_LCONTROL, VK_RCONTROL, VK_CONTROL):
            with self.subTest(ctrl_vk=hex(ctrl_vk)):
                listener, rec = self._make_listener()
                self.assertFalse(listener._handle_key_event(WM_KEYDOWN, ctrl_vk, timestamp=1.00))
                self.assertFalse(listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.02))
                self.assertFalse(listener._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.05))
                self.assertFalse(listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.15))
                self.assertFalse(listener._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.18))
                self.assertFalse(listener._handle_key_event(WM_KEYUP, ctrl_vk, timestamp=1.20))

                self.assertEqual(rec.commands, 1)
                self.assertEqual(rec.toggles, 0)
                self.assertEqual(rec.cancels, 0)

    def test_plain_double_capslock_emits_toggle_requested(self) -> None:
        listener, rec = self._make_listener()
        self.assertFalse(listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.00))
        self.assertFalse(listener._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.04))
        self.assertFalse(listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.15))
        self.assertFalse(listener._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.18))

        self.assertEqual(rec.toggles, 1)
        self.assertEqual(rec.commands, 0)

    def test_async_ctrl_state_fallback_triggers_command(self) -> None:
        listener, rec = self._make_listener(async_ctrl=True)
        listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.00)
        listener._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.03)
        listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.15)
        listener._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.18)

        self.assertEqual(rec.commands, 1)
        self.assertEqual(rec.toggles, 0)

    def test_both_ctrl_keys_held_releasing_one_still_commands(self) -> None:
        listener, rec = self._make_listener()
        listener._handle_key_event(WM_KEYDOWN, VK_LCONTROL, timestamp=1.00)
        listener._handle_key_event(WM_KEYDOWN, VK_RCONTROL, timestamp=1.01)
        listener._handle_key_event(WM_KEYUP, VK_LCONTROL, timestamp=1.02)
        listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.05)
        listener._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.08)
        listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.18)
        listener._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.20)
        listener._handle_key_event(WM_KEYUP, VK_RCONTROL, timestamp=1.25)

        self.assertEqual(rec.commands, 1)
        self.assertEqual(rec.toggles, 0)

    def test_mixed_ctrl_release_between_taps_does_not_emit(self) -> None:
        listener, rec = self._make_listener()
        listener._handle_key_event(WM_KEYDOWN, VK_LCONTROL, timestamp=1.00)
        listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.02)
        listener._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.05)
        listener._handle_key_event(WM_KEYUP, VK_LCONTROL, timestamp=1.08)
        listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.15)
        listener._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.18)

        self.assertEqual(rec.commands, 0)
        self.assertEqual(rec.toggles, 0)

    def test_injected_events_are_ignored(self) -> None:
        listener, rec = self._make_listener()
        listener.set_recording_active(True)
        self.assertFalse(
            listener._handle_key_event(WM_KEYDOWN, VK_LCONTROL, flags=LLKHF_INJECTED, timestamp=1.00)
        )
        self.assertFalse(
            listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, flags=LLKHF_INJECTED, timestamp=1.02)
        )
        self.assertFalse(
            listener._handle_key_event(WM_KEYDOWN, VK_ESCAPE, flags=LLKHF_INJECTED, timestamp=1.04)
        )
        self.assertEqual(rec.commands, 0)
        self.assertEqual(rec.toggles, 0)
        self.assertEqual(rec.cancels, 0)
        self.assertEqual(listener._pressed_ctrl_vks, set())

    def test_single_tap_stop_swallows_down_repeat_and_up_even_with_ctrl_held(self) -> None:
        listener, rec = self._make_listener()
        listener.set_recording_active(True)
        listener._handle_key_event(WM_KEYDOWN, VK_LCONTROL, timestamp=1.00)

        # Single-tap stop swallows down, autorepeat down, and matching up
        self.assertTrue(listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.05))
        self.assertTrue(listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.08))
        self.assertTrue(listener._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.12))
        # Next CapsLock is no longer swallowed
        self.assertFalse(listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=2.00))

        self.assertEqual(rec.toggles, 1)
        self.assertEqual(rec.commands, 0)

        # Escape cancels while cancel enabled
        listener.set_recording_active(True)
        self.assertTrue(listener._handle_key_event(WM_KEYDOWN, VK_ESCAPE, timestamp=2.50))
        self.assertEqual(rec.cancels, 1)

    def test_signal_emitted_outside_lock_allows_reentrant_set_recording_active(self) -> None:
        listener, rec = self._make_listener()
        listener.command_requested.connect(lambda: listener.set_recording_active(True))

        listener._handle_key_event(WM_KEYDOWN, VK_LCONTROL, timestamp=1.00)
        listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.02)
        listener._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.05)
        listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.15)
        listener._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.18)

        self.assertEqual(rec.commands, 1)
        self.assertTrue(listener._single_tap_stop)

    def test_process_hook_message_unpacks_kbdllhookstruct(self) -> None:
        listener, rec = self._make_listener()
        listener.set_recording_active(True)

        kb_stop = KBDLLHOOKSTRUCT(vkCode=VK_CAPITAL, scanCode=0, flags=0, time=0, dwExtraInfo=0)
        kb_other = KBDLLHOOKSTRUCT(vkCode=0x41, scanCode=0, flags=0, time=0, dwExtraInfo=0)

        with patch.object(capslock_hook.user32, "CallNextHookEx", return_value=42) as mocked_next:
            swallow_ret = listener._process_hook_message(0, WM_KEYDOWN, ctypes.addressof(kb_stop))
            pass_ret = listener._process_hook_message(0, WM_KEYDOWN, ctypes.addressof(kb_other))

        self.assertEqual(swallow_ret, 1)
        self.assertEqual(pass_ret, 42)
        self.assertEqual(rec.toggles, 1)
        mocked_next.assert_called_once()


class TestCommandModeHotkeyCapturePasteFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._qt_app = _ensure_qt_app()

    def test_ctrl_double_capslock_capture_and_overwrite_paste_roundtrip(self) -> None:
        listener = GlobalHotkeyListener(threshold_ms=320, ctrl_async_state_fn=lambda: False)
        triggered: list[str] = []
        listener.command_requested.connect(lambda: triggered.append("command"))

        harness = _FakeClipboardHarness(initial_clipboard="saved clip", selected_text="привет мир")
        harness.held_vks.add(VK_LCONTROL)
        injector = harness.make_injector()

        listener._handle_key_event(WM_KEYDOWN, VK_LCONTROL, timestamp=1.00)
        listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.02)
        listener._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.05)
        listener._handle_key_event(WM_KEYDOWN, VK_CAPITAL, timestamp=1.15)
        listener._handle_key_event(WM_KEYUP, VK_CAPITAL, timestamp=1.18)

        self.assertEqual(triggered, ["command"])

        selected = injector.capture_selected_text(timeout_ms=180, restore_clipboard=True)
        self.assertEqual(selected, "привет мир")
        self.assertEqual(harness.clipboard_text, "saved clip")
        self.assertEqual(harness.held_vks, set())

        pasted = injector.inject("Hello world", auto_space=False, restore_clipboard=True)
        self.assertTrue(pasted)
        self.assertEqual(harness.peek_calls, 0)
        self.assertEqual(harness.clipboard_text, "saved clip")
        self.assertEqual(harness.set_clipboard_calls, ["saved clip", "Hello world", "saved clip"])

    def test_command_aborts_when_no_selection_or_whitespace_only(self) -> None:
        # Case 1: No selection (clipboard sequence number unchanged)
        harness_none = _FakeClipboardHarness(initial_clipboard="keep me", copy_succeeds=False)
        injector_none = harness_none.make_injector()
        self.assertIsNone(injector_none.capture_selected_text())
        self.assertEqual(harness_none.clipboard_text, "keep me")
        self.assertEqual(harness_none.set_clipboard_calls, [])

        # Case 2: Blank line copied ("\r\n") -> returns None, restores clipboard, no paste
        harness_blank = _FakeClipboardHarness(initial_clipboard="keep me", selected_text="\r\n")
        injector_blank = harness_blank.make_injector()
        self.assertIsNone(injector_blank.capture_selected_text())
        self.assertEqual(harness_blank.clipboard_text, "keep me")
        self.assertEqual(harness_blank.modifiers_during_v, [])


if __name__ == "__main__":
    unittest.main()
