import unittest
from unittest.mock import patch

from localflow.injector import win32_paste
from localflow.injector.win32_paste import (
    KEYEVENTF_EXTENDEDKEY,
    KEYEVENTF_KEYUP,
    VK_C,
    VK_CAPITAL,
    VK_CONTROL,
    VK_INSERT,
    VK_LCONTROL,
    VK_LEFT,
    VK_LMENU,
    VK_LSHIFT,
    VK_MENU,
    VK_RCONTROL,
    VK_RIGHT,
    VK_RMENU,
    VK_RSHIFT,
    VK_SHIFT,
    VK_V,
    Win32TextInjector,
    send_ctrl_c,
    send_ctrl_insert,
    send_ctrl_v,
    should_prepend_space,
)


class FakeWin32Harness:
    """In-memory Win32 clipboard + keyboard state double for deterministic injector tests."""

    def __init__(
        self,
        *,
        initial_clipboard: str | None = "prior clipboard",
        selected_text: str | None = "selected text",
        copy_succeeds: bool = True,
        copy_delay_polls: int = 0,
        set_clipboard_ok: bool = True,
    ) -> None:
        self.clipboard_text = initial_clipboard
        self.selected_text = selected_text
        self.copy_succeeds = copy_succeeds
        self.copy_delay_polls = copy_delay_polls
        self.set_clipboard_ok = set_clipboard_ok
        self.seq = 100
        self.pending_copy_polls: int | None = None
        self.held_vks: set[int] = set()
        self.sent_batches: list[list[tuple[int, bool]]] = []
        self.modifiers_during_c: list[set[int]] = []
        self.modifiers_during_v: list[set[int]] = []
        self.set_clipboard_calls: list[str] = []
        self.get_clipboard_calls = 0
        self.peek_calls = 0
        self.left_char: str | None = None
        self.now = 10.0

    def get_clipboard_text(self) -> str | None:
        self.get_clipboard_calls += 1
        return self.clipboard_text

    def set_clipboard_text(self, text: str) -> bool:
        self.set_clipboard_calls.append(text)
        if not self.set_clipboard_ok:
            return False
        self.clipboard_text = text
        self.seq += 1
        return True

    def get_clipboard_sequence_number(self) -> int:
        if self.pending_copy_polls is not None:
            if self.pending_copy_polls <= 0:
                self.pending_copy_polls = None
                self.clipboard_text = self.selected_text
                self.seq += 1
            else:
                self.pending_copy_polls -= 1
        return self.seq

    def get_async_key_state(self, vk: int) -> int:
        if vk in self.held_vks:
            return -32768
        if vk == VK_SHIFT and (VK_LSHIFT in self.held_vks or VK_RSHIFT in self.held_vks):
            return -32768
        if vk == VK_MENU and (VK_LMENU in self.held_vks or VK_RMENU in self.held_vks):
            return -32768
        if vk == VK_CONTROL and (VK_LCONTROL in self.held_vks or VK_RCONTROL in self.held_vks):
            return -32768
        return 0

    def _clear_modifier(self, vk: int) -> None:
        self.held_vks.discard(vk)
        if vk == VK_CONTROL:
            self.held_vks.discard(VK_LCONTROL)
            self.held_vks.discard(VK_RCONTROL)
        elif vk == VK_SHIFT:
            self.held_vks.discard(VK_LSHIFT)
            self.held_vks.discard(VK_RSHIFT)
        elif vk == VK_MENU:
            self.held_vks.discard(VK_LMENU)
            self.held_vks.discard(VK_RMENU)

    def send_keys(self, events: list[tuple[int, bool]]) -> None:
        self.sent_batches.append(list(events))
        for vk, is_up in events:
            if is_up:
                self._clear_modifier(vk)
            else:
                self.held_vks.add(vk)
                if vk == VK_C:
                    self.modifiers_during_c.append(
                        set(self.held_vks) & {
                            VK_SHIFT,
                            VK_LSHIFT,
                            VK_RSHIFT,
                            VK_MENU,
                            VK_LMENU,
                            VK_RMENU,
                            VK_CONTROL,
                            VK_LCONTROL,
                            VK_RCONTROL,
                        }
                    )
                    if self.copy_succeeds and self._ctrl_down() and not self._shift_or_alt_down():
                        if self.copy_delay_polls > 0:
                            self.pending_copy_polls = self.copy_delay_polls
                        else:
                            self.clipboard_text = self.selected_text
                            self.seq += 1
                elif vk == VK_V:
                    self.modifiers_during_v.append(
                        set(self.held_vks) & {
                            VK_SHIFT,
                            VK_LSHIFT,
                            VK_RSHIFT,
                            VK_MENU,
                            VK_LMENU,
                            VK_RMENU,
                            VK_CONTROL,
                            VK_LCONTROL,
                            VK_RCONTROL,
                        }
                    )
        self.held_vks.discard(VK_C)
        self.held_vks.discard(VK_V)

    def _ctrl_down(self) -> bool:
        return bool(self.held_vks & {VK_CONTROL, VK_LCONTROL, VK_RCONTROL})

    def _shift_or_alt_down(self) -> bool:
        return bool(self.held_vks & {VK_SHIFT, VK_LSHIFT, VK_RSHIFT, VK_MENU, VK_LMENU, VK_RMENU})

    def peek_left_char(self) -> str | None:
        self.peek_calls += 1
        return self.left_char

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


class TestAutoSpace(unittest.TestCase):
    def test_adds_space_when_adjacent_to_word_or_punctuation(self) -> None:
        self.assertTrue(should_prepend_space(".", "Привет"))
        self.assertTrue(should_prepend_space("?", "Но в целом выглядит неплохо."))
        self.assertTrue(should_prepend_space("!", "Отлично."))
        self.assertTrue(should_prepend_space("а", "привет"))
        self.assertTrue(should_prepend_space("1", "шаг"))

    def test_no_space_when_already_spaced_or_opening_bracket(self) -> None:
        self.assertFalse(should_prepend_space(" ", "Привет"))
        self.assertFalse(should_prepend_space("\n", "Привет"))
        self.assertFalse(should_prepend_space("(", "Привет"))
        self.assertFalse(should_prepend_space(None, "Привет"))

    def test_no_space_before_closing_punctuation(self) -> None:
        self.assertFalse(should_prepend_space("слово", "."))
        self.assertFalse(should_prepend_space("слово", ", далее"))


class TestCaptureSelectedText(unittest.TestCase):
    def test_captures_selection_and_restores_prior_clipboard(self) -> None:
        h = FakeWin32Harness(initial_clipboard="old clip", selected_text="hello world")
        res = h.make_injector().capture_selected_text(timeout_ms=180, restore_clipboard=True)
        self.assertEqual(res, "hello world")
        self.assertEqual(h.clipboard_text, "old clip")
        self.assertEqual(h.set_clipboard_calls, ["old clip"])

    def test_preserves_leading_and_trailing_whitespace_of_valid_selection(self) -> None:
        h = FakeWin32Harness(initial_clipboard="old", selected_text="  foo bar  ")
        res = h.make_injector().capture_selected_text()
        self.assertEqual(res, "  foo bar  ")
        self.assertEqual(h.clipboard_text, "old")

    def test_restore_clipboard_false_skips_read_and_restore(self) -> None:
        h = FakeWin32Harness(initial_clipboard="old", selected_text="copied")
        res = h.make_injector().capture_selected_text(restore_clipboard=False)
        self.assertEqual(res, "copied")
        self.assertEqual(h.clipboard_text, "copied")
        self.assertEqual(h.set_clipboard_calls, [])
        self.assertEqual(h.get_clipboard_calls, 1)

    def test_previous_none_does_not_overwrite_on_restore(self) -> None:
        h = FakeWin32Harness(initial_clipboard=None, selected_text="picked")
        res = h.make_injector().capture_selected_text(restore_clipboard=True)
        self.assertEqual(res, "picked")
        self.assertEqual(h.set_clipboard_calls, [])

    def test_unchanged_sequence_returns_none_and_never_touches_clipboard(self) -> None:
        h = FakeWin32Harness(initial_clipboard="non-empty prior", copy_succeeds=False)
        res = h.make_injector().capture_selected_text(timeout_ms=30, restore_clipboard=True)
        self.assertIsNone(res)
        self.assertEqual(h.clipboard_text, "non-empty prior")
        self.assertEqual(h.set_clipboard_calls, [])
        self.assertEqual(h.get_clipboard_calls, 1)

    def test_delayed_sequence_change_within_timeout_succeeds(self) -> None:
        h = FakeWin32Harness(initial_clipboard="old", selected_text="delayed", copy_delay_polls=3)
        res = h.make_injector().capture_selected_text(timeout_ms=100, restore_clipboard=True)
        self.assertEqual(res, "delayed")
        self.assertEqual(h.clipboard_text, "old")

    def test_empty_or_whitespace_only_copied_returns_none_and_restores(self) -> None:
        for blank in ("", "   ", "\r\n", "\t\n"):
            with self.subTest(blank=repr(blank)):
                h = FakeWin32Harness(initial_clipboard="saved", selected_text=blank)
                res = h.make_injector().capture_selected_text(restore_clipboard=True)
                self.assertIsNone(res)
                self.assertEqual(h.clipboard_text, "saved")

    def test_copied_none_when_sequence_changed_restores_prior_text(self) -> None:
        h = FakeWin32Harness(initial_clipboard="saved", selected_text=None)
        res = h.make_injector().capture_selected_text(restore_clipboard=True)
        self.assertIsNone(res)
        self.assertEqual(h.clipboard_text, "saved")

    def test_falls_back_to_ctrl_insert_when_ctrl_c_does_not_copy(self) -> None:
        h = FakeWin32Harness(initial_clipboard="old", selected_text="from insert", copy_succeeds=False)
        orig_send = h.send_keys

        def _send_with_insert(events: list[tuple[int, bool]]) -> None:
            orig_send(events)
            for vk, is_up in events:
                if vk == VK_INSERT and not is_up and (VK_CONTROL in h.held_vks):
                    h.clipboard_text = h.selected_text
                    h.seq += 1

        h.send_keys = _send_with_insert  # type: ignore[method-assign]
        res = h.make_injector().capture_selected_text(timeout_ms=300, restore_clipboard=True)
        self.assertEqual(res, "from insert")
        self.assertEqual(h.clipboard_text, "old")
        self.assertTrue(any(vk == VK_INSERT for batch in h.sent_batches for vk, _ in batch))

    def test_waits_for_capslock_release_before_copy(self) -> None:
        h = FakeWin32Harness(initial_clipboard="old", selected_text="after caps")
        h.held_vks.add(VK_CAPITAL)
        orig_sleep = h.sleep
        sleep_calls = 0

        def _sleep_releases_caps(sec: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls == 2:
                h.held_vks.discard(VK_CAPITAL)
            orig_sleep(sec)

        h.sleep = _sleep_releases_caps  # type: ignore[method-assign]
        res = h.make_injector().capture_selected_text(timeout_ms=300, restore_clipboard=True)
        self.assertEqual(res, "after caps")
        self.assertGreaterEqual(sleep_calls, 2)


class TestModifierHandling(unittest.TestCase):
    def test_held_ctrl_variants_released_cleanly_for_ctrl_c(self) -> None:
        for ctrl_vk in (VK_LCONTROL, VK_RCONTROL, VK_CONTROL):
            with self.subTest(ctrl_vk=hex(ctrl_vk)):
                h = FakeWin32Harness(initial_clipboard="old", selected_text="text")
                h.held_vks.add(ctrl_vk)
                res = h.make_injector().capture_selected_text()
                self.assertEqual(res, "text")
                self.assertEqual(h.held_vks, set())
                self.assertEqual(h.modifiers_during_c, [{VK_CONTROL}])

    def test_held_shift_and_alt_released_before_ctrl_c(self) -> None:
        h = FakeWin32Harness(initial_clipboard="old", selected_text="clean")
        h.held_vks.update({VK_LCONTROL, VK_RSHIFT, VK_LMENU})
        res = h.make_injector().capture_selected_text()
        self.assertEqual(res, "clean")
        self.assertEqual(h.modifiers_during_c, [{VK_CONTROL}])
        self.assertEqual(h.held_vks, set())

    def test_send_ctrl_v_releases_held_modifiers_first(self) -> None:
        h = FakeWin32Harness()
        h.held_vks.update({VK_RCONTROL, VK_LSHIFT, VK_RMENU})
        send_ctrl_v(
            get_async_key_state=h.get_async_key_state,
            send_keys=h.send_keys,
            sleep=h.sleep,
        )
        self.assertEqual(h.modifiers_during_v, [{VK_CONTROL}])
        self.assertEqual(h.held_vks, set())

    def test_send_ctrl_c_releases_held_modifiers_first(self) -> None:
        h = FakeWin32Harness()
        h.held_vks.update({VK_LCONTROL, VK_SHIFT, VK_MENU})
        send_ctrl_c(
            get_async_key_state=h.get_async_key_state,
            send_keys=h.send_keys,
            sleep=h.sleep,
        )
        self.assertEqual(h.modifiers_during_c, [{VK_CONTROL}])
        self.assertEqual(h.held_vks, set())

    def test_send_keys_sets_extended_flag_for_rcontrol_and_rmenu(self) -> None:
        captured: list[tuple[int, int]] = []

        def fake_send_input(count, inputs, size):
            for idx in range(count):
                captured.append((inputs[idx].ki.wVk, inputs[idx].ki.dwFlags))
            return count

        with patch.object(win32_paste.user32, "SendInput", side_effect=fake_send_input):
            win32_paste._send_keys([(VK_RCONTROL, True), (VK_RMENU, True), (VK_LCONTROL, True)])

        self.assertEqual(
            captured,
            [
                (VK_RCONTROL, KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY),
                (VK_RMENU, KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY),
                (VK_LCONTROL, KEYEVENTF_KEYUP),
            ],
        )

    def test_send_keys_sets_extended_flag_and_scan_code_for_arrows(self) -> None:
        captured: list[tuple[int, int, int]] = []

        def fake_send_input(count, inputs, size):
            for idx in range(count):
                ki = inputs[idx].ki
                captured.append((ki.wVk, ki.dwFlags, ki.wScan))
            return count

        with patch.object(win32_paste.user32, "SendInput", side_effect=fake_send_input):
            win32_paste._send_keys([(VK_LEFT, False), (VK_LEFT, True), (VK_RIGHT, False)])

        self.assertEqual(
            [(vk, flags) for vk, flags, _ in captured],
            [
                (VK_LEFT, KEYEVENTF_EXTENDEDKEY),
                (VK_LEFT, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP),
                (VK_RIGHT, KEYEVENTF_EXTENDEDKEY),
            ],
        )
        # Extended arrow keys must include hardware scan codes to avoid NumLock/Electron shift drop
        self.assertTrue(all(scan > 0 for _, _, scan in captured))


class TestWin32TextInjectorOverwrite(unittest.TestCase):
    def test_inject_auto_space_false_skips_peek_and_restores_clipboard(self) -> None:
        h = FakeWin32Harness(initial_clipboard="before")
        h.left_char = "."
        ok = h.make_injector().inject("replaced", auto_space=False, restore_clipboard=True)
        self.assertTrue(ok)
        self.assertEqual(h.peek_calls, 0)
        self.assertEqual(h.set_clipboard_calls, ["replaced", "before"])
        self.assertEqual(h.clipboard_text, "before")
        self.assertEqual(len(h.modifiers_during_v), 1)

    def test_inject_auto_space_true_peeks_and_prepends_space(self) -> None:
        h = FakeWin32Harness(initial_clipboard="before", copy_succeeds=False)
        h.left_char = "."
        ok = h.make_injector().inject("привет", auto_space=True, restore_clipboard=True)
        self.assertTrue(ok)
        self.assertEqual(h.peek_calls, 1)
        self.assertEqual(h.set_clipboard_calls, [" привет", "before"])

    def test_inject_empty_text_returns_false_without_side_effects(self) -> None:
        h = FakeWin32Harness(initial_clipboard="before")
        ok = h.make_injector().inject("")
        self.assertFalse(ok)
        self.assertEqual(h.set_clipboard_calls, [])
        self.assertEqual(h.sent_batches, [])

    def test_inject_returns_false_when_set_clipboard_fails(self) -> None:
        h = FakeWin32Harness(initial_clipboard="before", set_clipboard_ok=False)
        ok = h.make_injector().inject("hello", auto_space=False)
        self.assertFalse(ok)
        self.assertEqual(h.sent_batches, [])

    def test_inject_auto_space_with_active_selection_replaces_without_collapsing(self) -> None:
        h = FakeWin32Harness(
            initial_clipboard="before",
            selected_text="old selection",
            copy_succeeds=True,
        )
        h.left_char = "."
        ok = h.make_injector().inject("replacement", auto_space=True, restore_clipboard=True)
        self.assertTrue(ok)
        self.assertEqual(h.peek_calls, 0)
        self.assertEqual(h.set_clipboard_calls, ["replacement", "before"])
        self.assertEqual(h.clipboard_text, "before")
        # Must never send arrow keys or Shift that would collapse the active selection
        sent_vks = {vk for batch in h.sent_batches for vk, _ in batch}
        self.assertNotIn(VK_LEFT, sent_vks)
        self.assertNotIn(VK_RIGHT, sent_vks)
        self.assertNotIn(VK_LSHIFT, sent_vks)
        self.assertIn(VK_C, sent_vks)
        self.assertIn(VK_V, sent_vks)

    def test_inject_auto_space_without_selection_peeks_and_prepends_space(self) -> None:
        h = FakeWin32Harness(initial_clipboard="before", copy_succeeds=False)
        h.left_char = "."
        ok = h.make_injector().inject("привет", auto_space=True, restore_clipboard=True)
        self.assertTrue(ok)
        self.assertEqual(h.peek_calls, 1)
        self.assertEqual(h.set_clipboard_calls, [" привет", "before"])

    def test_inject_terminal_uses_ctrl_insert_instead_of_ctrl_c(self) -> None:
        h = FakeWin32Harness(initial_clipboard="before", copy_succeeds=False)
        h.left_char = " "
        ok = h.make_injector().inject("ls", auto_space=True, restore_clipboard=True, is_terminal=True)
        self.assertTrue(ok)
        sent_vks = {vk for batch in h.sent_batches for vk, _ in batch}
        self.assertIn(VK_INSERT, sent_vks)
        self.assertNotIn(VK_C, sent_vks)


class TestCaptureSelectedTextWrapper(unittest.TestCase):
    def test_module_wrapper_delegates_to_win32_text_injector(self) -> None:
        with patch.object(Win32TextInjector, "capture_selected_text", return_value="abc") as mocked:
            result = win32_paste.capture_selected_text(timeout_ms=250, restore_clipboard=False)
        self.assertEqual(result, "abc")
        mocked.assert_called_once_with(timeout_ms=250, restore_clipboard=False)


if __name__ == "__main__":
    unittest.main()
