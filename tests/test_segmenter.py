import unittest
from unittest.mock import patch

import numpy as np

from localflow.audio.recorder import AudioRecorder
from localflow.audio.segmenter import SegmentEvent, SpeechSegmenter

SAMPLE_RATE = 16000
BLOCK = 512  # 32 ms


def _tone(duration_ms: int, rms: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    samples = int(round(sample_rate * duration_ms / 1000.0))
    if samples <= 0:
        return np.zeros(0, dtype=np.float32)
    # Deterministic square-like alternating signal with exact RMS = rms
    arr = np.empty(samples, dtype=np.float32)
    arr[0::2] = rms
    arr[1::2] = -rms
    return arr


def _feed_blocks(segmenter: SpeechSegmenter, audio: np.ndarray, block_size: int = BLOCK) -> list[SegmentEvent]:
    events: list[SegmentEvent] = []
    for idx in range(0, len(audio), block_size):
        events.append(segmenter.process_block(audio[idx : idx + block_size]))
    return events


class TestSpeechSegmenter(unittest.TestCase):
    def test_invalid_sample_rate_raises(self) -> None:
        with self.assertRaises(ValueError):
            SpeechSegmenter(sample_rate=0)

    def test_pure_silence_produces_no_segments_and_no_speech_flag(self) -> None:
        seg = SpeechSegmenter()
        silence = _tone(2000, rms=0.001)
        events = _feed_blocks(seg, silence)
        self.assertTrue(all(ev.segment is None and not ev.auto_stop_triggered for ev in events))
        self.assertFalse(seg.has_heard_speech)
        tail, duration = seg.finish()
        self.assertEqual(len(tail), len(silence))
        self.assertAlmostEqual(duration, 2.0, places=5)

    def test_sub_onset_transient_does_not_trigger_speech(self) -> None:
        seg = SpeechSegmenter(min_speech_onset_ms=64)
        stream = np.concatenate([
            _tone(32, rms=0.01),
            _tone(600, rms=0.001),
        ])
        events = _feed_blocks(seg, stream)
        self.assertTrue(all(ev.segment is None for ev in events))
        self.assertFalse(seg.has_heard_speech)

    def test_hysteresis_band_keeps_speech_active_and_blocks_false_onset(self) -> None:
        # From silence, 0.0035 RMS (< 0.004) must NOT trigger speech onset
        seg_silent = SpeechSegmenter()
        _feed_blocks(seg_silent, _tone(500, rms=0.0035))
        self.assertFalse(seg_silent.has_heard_speech)

        # Once in speech (>= 0.004 for >= 64 ms), dipping to 0.0035 (>= 0.003) stays in speech
        seg_active = SpeechSegmenter()
        stream = np.concatenate([
            _tone(128, rms=0.006),
            _tone(500, rms=0.0035),
        ])
        events = _feed_blocks(seg_active, stream)
        self.assertTrue(seg_active.has_heard_speech)
        self.assertTrue(all(ev.segment is None for ev in events))

    def test_two_bursts_separated_by_pause_stitch_with_zero_overlap_and_gap(self) -> None:
        seg = SpeechSegmenter()
        stream = np.concatenate([
            _tone(160, rms=0.001),
            _tone(608, rms=0.02),
            _tone(416, rms=0.001),
            _tone(512, rms=0.03),
            _tone(128, rms=0.001),
        ])
        events = _feed_blocks(seg, stream)
        emitted = [ev.segment for ev in events if ev.segment is not None]
        self.assertEqual(len(emitted), 1)
        tail, duration = seg.finish()
        stitched = np.concatenate([*emitted, tail])
        self.assertTrue(np.array_equal(stitched, stream))
        self.assertAlmostEqual(duration, len(stream) / SAMPLE_RATE, places=6)

    def test_three_bursts_produce_two_emitted_segments_plus_tail(self) -> None:
        seg = SpeechSegmenter()
        stream = np.concatenate([
            _tone(512, rms=0.02),
            _tone(448, rms=0.001),
            _tone(512, rms=0.03),
            _tone(448, rms=0.001),
            _tone(512, rms=0.04),
        ])
        events = _feed_blocks(seg, stream)
        emitted = [ev.segment for ev in events if ev.segment is not None]
        self.assertEqual(len(emitted), 2)
        tail, duration = seg.finish()
        self.assertGreater(tail.size, 0)
        self.assertTrue(np.array_equal(np.concatenate([*emitted, tail]), stream))
        self.assertAlmostEqual(duration, len(stream) / SAMPLE_RATE, places=6)

    def test_short_speech_below_min_segment_is_not_emitted_as_tiny_chunk(self) -> None:
        seg = SpeechSegmenter(min_segment_speech_ms=250)
        stream = np.concatenate([
            _tone(160, rms=0.02),   # < 250 ms
            _tone(448, rms=0.001),  # pause
            _tone(320, rms=0.03),   # subsequent speech combines
            _tone(448, rms=0.001),
        ])
        events = _feed_blocks(seg, stream)
        emitted = [ev.segment for ev in events if ev.segment is not None]
        self.assertEqual(len(emitted), 1)
        tail, _ = seg.finish()
        self.assertTrue(np.array_equal(np.concatenate([*emitted, tail]), stream))

    def test_auto_stop_does_not_fire_before_speech(self) -> None:
        seg = SpeechSegmenter(silence_auto_stop_sec=2.5)
        events = _feed_blocks(seg, _tone(4000, rms=0.001))
        self.assertFalse(any(ev.auto_stop_triggered for ev in events))
        self.assertFalse(seg.has_heard_speech)

    def test_auto_stop_is_not_reset_by_pause_cut_and_fires_once(self) -> None:
        seg = SpeechSegmenter(pause_ms=400, silence_auto_stop_sec=2.5)
        speech_blocks = 16  # 512 ms
        silence_blocks = 100  # 3200 ms

        emitted_at_block: list[int] = []
        auto_stop_at_block: list[int] = []

        for idx in range(speech_blocks):
            ev = seg.process_block(_tone(32, rms=0.02))
            self.assertFalse(ev.auto_stop_triggered)
            if ev.segment is not None:
                emitted_at_block.append(idx)

        for idx in range(silence_blocks):
            ev = seg.process_block(_tone(32, rms=0.001))
            if ev.segment is not None:
                emitted_at_block.append(speech_blocks + idx)
            if ev.auto_stop_triggered:
                auto_stop_at_block.append(idx)

        self.assertEqual(len(emitted_at_block), 1)
        # 400 ms = 12.5 blocks -> block index 12 (13th silence block = 416 ms)
        self.assertEqual(emitted_at_block[0], speech_blocks + 12)
        # 2.5 s = 2500 ms = 78.125 blocks -> block index 78 (79th silence block = 2528 ms)
        self.assertEqual(auto_stop_at_block, [78])

    def test_auto_stop_timer_resets_only_on_confirmed_speech(self) -> None:
        seg = SpeechSegmenter(pause_ms=400, silence_auto_stop_sec=2.5)
        # 512 ms speech + 1504 ms silence (47 blocks) + 512 ms speech + 1504 ms silence -> no trigger yet
        _feed_blocks(seg, _tone(512, rms=0.02))
        events_s1 = _feed_blocks(seg, _tone(1504, rms=0.001))
        self.assertFalse(any(ev.auto_stop_triggered for ev in events_s1))

        _feed_blocks(seg, _tone(512, rms=0.02))
        events_s2 = _feed_blocks(seg, _tone(1504, rms=0.001))
        self.assertFalse(any(ev.auto_stop_triggered for ev in events_s2))

        # Additional 1024 ms silence (total 2528 ms since 2nd speech) -> triggers once
        events_s3 = _feed_blocks(seg, _tone(1024, rms=0.001))
        self.assertEqual(sum(1 for ev in events_s3 if ev.auto_stop_triggered), 1)

    def test_auto_stop_none_never_triggers(self) -> None:
        seg = SpeechSegmenter(silence_auto_stop_sec=None)
        _feed_blocks(seg, _tone(512, rms=0.02))
        events = _feed_blocks(seg, _tone(5000, rms=0.001))
        self.assertFalse(any(ev.auto_stop_triggered for ev in events))

    def test_block_shapes_varying_sizes_and_idempotent_finish(self) -> None:
        seg = SpeechSegmenter()
        ev_empty = seg.process_block(np.zeros(0, dtype=np.float32))
        self.assertIsNone(ev_empty.segment)
        self.assertFalse(ev_empty.auto_stop_triggered)

        mono_2d = _tone(32, rms=0.02).reshape(-1, 1)
        seg.process_block(mono_2d)

        # Stereo with speech only in channel 1
        stereo = np.zeros((512, 2), dtype=np.float32)
        stereo[:, 1] = _tone(32, rms=0.04)
        seg.process_block(stereo)
        self.assertTrue(seg.has_heard_speech)

        # Varying block sizes
        for n in (128, 777, 1024):
            seg.process_block(np.full(n, 0.01, dtype=np.float32))

        tail1, dur1 = seg.finish()
        tail2, dur2 = seg.finish()
        self.assertGreater(tail1.size, 0)
        self.assertEqual(tail2.size, 0)
        self.assertEqual(dur1, dur2)


class _FakeInputStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class TestAudioRecorderChunkedIntegration(unittest.TestCase):
    def test_legacy_start_and_stop_remain_unchanged(self) -> None:
        with patch("localflow.audio.recorder.sd.InputStream", _FakeInputStream):
            rec = AudioRecorder(sample_rate=16000)
            rec.start()
            self.assertTrue(rec.is_recording)
            b1 = _tone(32, rms=0.02).reshape(-1, 1)
            b2 = _tone(32, rms=0.001).reshape(-1, 1)
            rec._audio_callback(b1, len(b1), None, None)
            rec._audio_callback(b2, len(b2), None, None)
            audio, dur = rec.stop()
        self.assertFalse(rec.is_recording)
        self.assertEqual(len(audio), 1024)
        self.assertAlmostEqual(dur, 1024 / 16000, places=6)

    def test_chunked_callbacks_run_outside_lock_and_stop_chunked_stitches_cleanly(self) -> None:
        segments: list[np.ndarray] = []
        timeout_results: list[tuple[np.ndarray, float]] = []

        with patch("localflow.audio.recorder.sd.InputStream", _FakeInputStream):
            rec = AudioRecorder(sample_rate=16000)

            def on_timeout() -> None:
                # Synchronous stop_chunked() inside callback must NOT deadlock
                timeout_results.append(rec.stop_chunked())

            rec.start(
                pause_ms=400,
                silence_auto_stop_sec=1.0,
                on_segment=segments.append,
                on_silence_timeout=on_timeout,
            )

            full_stream = np.concatenate([
                _tone(512, rms=0.02),
                _tone(1024, rms=0.001),
            ])
            for idx in range(0, len(full_stream), BLOCK):
                blk = full_stream[idx : idx + BLOCK].reshape(-1, 1)
                rec._audio_callback(blk, len(blk), None, None)

        self.assertEqual(len(segments), 1)
        self.assertEqual(len(timeout_results), 1)
        tail, total_dur = timeout_results[0]
        self.assertFalse(rec.is_recording)
        self.assertTrue(np.array_equal(np.concatenate([*segments, tail]), full_stream))
        self.assertAlmostEqual(total_dur, len(full_stream) / 16000, places=6)

    def test_start_failure_resets_state(self) -> None:
        class _FailingStream(_FakeInputStream):
            def start(self) -> None:
                raise RuntimeError("device busy")

        with patch("localflow.audio.recorder.sd.InputStream", _FailingStream):
            rec = AudioRecorder()
            with self.assertRaises(RuntimeError):
                rec.start(on_segment=lambda _: None)
            self.assertFalse(rec.is_recording)


if __name__ == "__main__":
    unittest.main()
