import unittest

from localflow.hotkey.detector import DoubleTapDetector


class TestDoubleTapDetector(unittest.TestCase):
    def test_double_tap_triggers_within_threshold(self) -> None:
        d = DoubleTapDetector(threshold_ms=300)
        self.assertFalse(d.on_key_event(True, 1.00))
        self.assertFalse(d.on_key_event(False, 1.05))
        self.assertTrue(d.on_key_event(True, 1.20))
        self.assertFalse(d.on_key_event(False, 1.25))

    def test_slow_second_tap_does_not_trigger(self) -> None:
        d = DoubleTapDetector(threshold_ms=300)
        self.assertFalse(d.on_key_event(True, 1.00))
        self.assertFalse(d.on_key_event(False, 1.05))
        self.assertFalse(d.on_key_event(True, 1.50))

    def test_key_autorepeat_is_ignored(self) -> None:
        d = DoubleTapDetector(threshold_ms=300)
        self.assertFalse(d.on_key_event(True, 1.00))
        self.assertFalse(d.on_key_event(True, 1.05))
        self.assertFalse(d.on_key_event(True, 1.10))

    def test_triple_tap_does_not_trigger_twice(self) -> None:
        d = DoubleTapDetector(threshold_ms=300)
        self.assertFalse(d.on_key_event(True, 1.00))
        self.assertFalse(d.on_key_event(False, 1.02))
        self.assertTrue(d.on_key_event(True, 1.10))
        self.assertFalse(d.on_key_event(False, 1.12))
        self.assertFalse(d.on_key_event(True, 1.20))

    def test_single_tap_mode_triggers_on_first_press_and_resets(self) -> None:
        d = DoubleTapDetector(threshold_ms=300)
        d.set_single_tap_mode(True)
        self.assertTrue(d.on_key_event(True, 2.00))
        # Autorepeat while held down must not re-trigger
        self.assertFalse(d.on_key_event(True, 2.05))
        self.assertFalse(d.on_key_event(False, 2.10))
        # Back in double-tap mode: single press should not trigger
        self.assertFalse(d.on_key_event(True, 3.00))

    def test_plain_double_tap_sets_last_action_dictate(self) -> None:
        d = DoubleTapDetector(threshold_ms=300)
        self.assertIsNone(d.last_action)
        self.assertFalse(d.on_key_event(True, 1.00, ctrl_held=False))
        self.assertFalse(d.on_key_event(False, 1.05, ctrl_held=False))
        self.assertTrue(d.on_key_event(True, 1.15, ctrl_held=False))
        self.assertEqual(d.last_action, "dictate")
        self.assertFalse(d.on_key_event(False, 1.20, ctrl_held=False))
        self.assertEqual(d.last_action, "dictate")
        d.reset()
        self.assertIsNone(d.last_action)

    def test_ctrl_double_tap_sets_last_action_command(self) -> None:
        d = DoubleTapDetector(threshold_ms=300)
        self.assertFalse(d.on_key_event(True, 1.00, ctrl_held=True))
        self.assertFalse(d.on_key_event(False, 1.05, ctrl_held=True))
        self.assertTrue(d.on_key_event(True, 1.20, ctrl_held=True))
        self.assertEqual(d.last_action, "command")

    def test_ctrl_slow_second_tap_does_not_trigger(self) -> None:
        d = DoubleTapDetector(threshold_ms=300)
        self.assertFalse(d.on_key_event(True, 1.00, ctrl_held=True))
        self.assertFalse(d.on_key_event(False, 1.05, ctrl_held=True))
        self.assertFalse(d.on_key_event(True, 1.50, ctrl_held=True))
        self.assertIsNone(d.last_action)
        self.assertFalse(d.on_key_event(False, 1.55, ctrl_held=True))
        self.assertTrue(d.on_key_event(True, 1.65, ctrl_held=True))
        self.assertEqual(d.last_action, "command")

    def test_mixed_ctrl_taps_do_not_trigger_and_reanchor(self) -> None:
        d = DoubleTapDetector(threshold_ms=300)
        # Ctrl+tap followed by plain tap -> does not trigger, re-anchors on plain tap
        self.assertFalse(d.on_key_event(True, 1.00, ctrl_held=True))
        self.assertFalse(d.on_key_event(False, 1.05, ctrl_held=True))
        self.assertFalse(d.on_key_event(True, 1.15, ctrl_held=False))
        self.assertIsNone(d.last_action)
        self.assertFalse(d.on_key_event(False, 1.20, ctrl_held=False))
        self.assertTrue(d.on_key_event(True, 1.30, ctrl_held=False))
        self.assertEqual(d.last_action, "dictate")

        d.reset()
        # Plain tap followed by Ctrl+tap -> does not trigger, re-anchors on Ctrl+tap
        self.assertFalse(d.on_key_event(True, 2.00, ctrl_held=False))
        self.assertFalse(d.on_key_event(False, 2.05, ctrl_held=False))
        self.assertFalse(d.on_key_event(True, 2.15, ctrl_held=True))
        self.assertIsNone(d.last_action)
        self.assertFalse(d.on_key_event(False, 2.20, ctrl_held=True))
        self.assertTrue(d.on_key_event(True, 2.30, ctrl_held=True))
        self.assertEqual(d.last_action, "command")

    def test_ctrl_autorepeat_is_ignored(self) -> None:
        d = DoubleTapDetector(threshold_ms=300)
        self.assertFalse(d.on_key_event(True, 1.00, ctrl_held=True))
        self.assertFalse(d.on_key_event(True, 1.05, ctrl_held=False))
        self.assertFalse(d.on_key_event(True, 1.10, ctrl_held=True))
        self.assertFalse(d.on_key_event(False, 1.12, ctrl_held=True))
        self.assertTrue(d.on_key_event(True, 1.20, ctrl_held=True))
        self.assertEqual(d.last_action, "command")

    def test_ctrl_triple_tap_does_not_trigger_twice(self) -> None:
        d = DoubleTapDetector(threshold_ms=300)
        self.assertFalse(d.on_key_event(True, 1.00, ctrl_held=True))
        self.assertFalse(d.on_key_event(False, 1.02, ctrl_held=True))
        self.assertTrue(d.on_key_event(True, 1.10, ctrl_held=True))
        self.assertEqual(d.last_action, "command")
        self.assertFalse(d.on_key_event(False, 1.12, ctrl_held=True))
        self.assertFalse(d.on_key_event(True, 1.20, ctrl_held=True))

    def test_single_tap_stop_ignores_ctrl_held(self) -> None:
        d = DoubleTapDetector(threshold_ms=300)
        d.set_single_tap_mode(True)
        self.assertTrue(d.on_key_event(True, 2.00, ctrl_held=True))
        self.assertEqual(d.last_action, "stop")
        self.assertFalse(d.on_key_event(True, 2.05, ctrl_held=True))
        self.assertFalse(d.on_key_event(False, 2.10, ctrl_held=True))

        d.set_single_tap_mode(True)
        self.assertTrue(d.on_key_event(True, 3.00, ctrl_held=False))
        self.assertEqual(d.last_action, "stop")


if __name__ == "__main__":
    unittest.main()
