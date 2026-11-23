import threading
import time
import unittest

import numpy as np

from localflow.asr.chunked_session import ChunkedASRSession, merge_transcript_parts


class _RecordingFakeASR:
    def __init__(self, delay_sec: float = 0.02, fail_values: set[int] | None = None) -> None:
        self.delay_sec = delay_sec
        self.fail_values = fail_values or set()
        self.calls: list[int] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()
        self.started_event = threading.Event()
        self.release_event: threading.Event | None = None

    def transcribe(self, audio: np.ndarray) -> str:
        tag = int(round(float(audio[0]) * 1000)) if audio.size > 0 else -1
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started_event.set()
        try:
            if self.release_event is not None:
                self.release_event.wait(timeout=2.0)
            elif self.delay_sec > 0:
                time.sleep(self.delay_sec)
            with self._lock:
                self.calls.append(tag)
            if tag in self.fail_values:
                raise RuntimeError(f"boom on {tag}")
            if tag == 99:
                return "   "
            return f"part{tag}"
        finally:
            with self._lock:
                self.active -= 1


def _tagged_audio(tag: int, length: int = 160) -> np.ndarray:
    return np.full(length, float(tag) / 1000.0, dtype=np.float32)


class TestMergeTranscriptParts(unittest.TestCase):
    def test_merge_strips_drops_empty_and_joins_with_single_space(self) -> None:
        self.assertEqual(merge_transcript_parts(["  hello ", "", "   ", "world\n"]), "hello world")
        self.assertEqual(merge_transcript_parts([]), "")
        self.assertEqual(ChunkedASRSession.merge_transcript_parts(["a", " b "]), "a b")


class TestChunkedASRSession(unittest.TestCase):
    def test_sequential_fifo_execution_and_cumulative_partial_callbacks(self) -> None:
        asr = _RecordingFakeASR(delay_sec=0.025)
        partials: list[str] = []
        session = ChunkedASRSession(asr, on_partial_text=partials.append)

        t0 = time.monotonic()
        session.submit(_tagged_audio(1))
        session.submit(_tagged_audio(99))  # blank output skipped from partial
        session.submit(_tagged_audio(2))
        submit_elapsed = time.monotonic() - t0
        self.assertLess(submit_elapsed, 0.05)

        final = session.finalize(_tagged_audio(3))
        self.assertEqual(final, "part1 part2 part3")
        self.assertEqual(asr.calls, [1, 99, 2, 3])
        self.assertEqual(asr.max_active, 1)
        self.assertEqual(partials, ["part1", "part1 part2"])

    def test_finalize_with_empty_tail_skips_transcribe(self) -> None:
        asr = _RecordingFakeASR(delay_sec=0.01)
        session = ChunkedASRSession(asr)
        session.submit(_tagged_audio(1))
        session.submit(_tagged_audio(2))

        result = session.finalize(np.zeros(0, dtype=np.float32))
        self.assertEqual(result, "part1 part2")
        self.assertEqual(asr.calls, [1, 2])
        # Second finalize returns cached result cleanly
        self.assertEqual(session.finalize(_tagged_audio(4)), "part1 part2")
        self.assertEqual(asr.calls, [1, 2])

    def test_partial_callback_exception_is_isolated(self) -> None:
        asr = _RecordingFakeASR(delay_sec=0.01)

        def flaky_cb(_text: str) -> None:
            raise ValueError("ui error")

        session = ChunkedASRSession(asr, on_partial_text=flaky_cb)
        session.submit(_tagged_audio(1))
        self.assertEqual(session.finalize(_tagged_audio(2)), "part1 part2")

    def test_segment_error_isolation_does_not_deadlock_or_drop_other_chunks(self) -> None:
        asr = _RecordingFakeASR(delay_sec=0.01, fail_values={2})
        session = ChunkedASRSession(asr)
        session.submit(_tagged_audio(1))
        session.submit(_tagged_audio(2))
        session.submit(_tagged_audio(3))

        result = session.finalize(_tagged_audio(4))
        self.assertEqual(result, "part1 part3 part4")
        self.assertEqual(len(session.errors), 1)
        self.assertEqual(asr.calls, [1, 2, 3, 4])

    def test_all_segments_fail_with_empty_tail_returns_empty_without_deadlock(self) -> None:
        asr = _RecordingFakeASR(delay_sec=0.01, fail_values={1})
        session = ChunkedASRSession(asr)
        session.submit(_tagged_audio(1))
        self.assertEqual(session.finalize(np.zeros(0, dtype=np.float32)), "")
        self.assertEqual(len(session.errors), 1)

    def test_tail_error_preserves_prior_parts_or_reraises_when_empty(self) -> None:
        asr_with_prior = _RecordingFakeASR(delay_sec=0.005, fail_values={2})
        s1 = ChunkedASRSession(asr_with_prior)
        s1.submit(_tagged_audio(1))
        self.assertEqual(s1.finalize(_tagged_audio(2)), "part1")

        asr_empty = _RecordingFakeASR(delay_sec=0.005, fail_values={2})
        s2 = ChunkedASRSession(asr_empty)
        with self.assertRaises(RuntimeError):
            s2.finalize(_tagged_audio(2))

    def test_cancel_drops_queued_segments_suppresses_callbacks_and_finalize_returns_empty(self) -> None:
        asr = _RecordingFakeASR(delay_sec=0.0)
        asr.release_event = threading.Event()
        partials: list[str] = []
        session = ChunkedASRSession(asr, on_partial_text=partials.append)

        session.submit(_tagged_audio(1))
        self.assertTrue(asr.started_event.wait(timeout=1.0))
        session.submit(_tagged_audio(2))
        session.submit(_tagged_audio(3))

        session.cancel()
        asr.release_event.set()

        result = session.finalize(_tagged_audio(4))
        session._worker.join(timeout=1.0)
        self.assertFalse(session._worker.is_alive())
        self.assertEqual(result, "")
        self.assertEqual(partials, [])
        self.assertEqual(asr.calls, [1])


if __name__ == "__main__":
    unittest.main()
