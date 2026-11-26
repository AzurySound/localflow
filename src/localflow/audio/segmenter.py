"""Frame-based pause speech segmentation and post-speech silence auto-stop tracking."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SegmentEvent:
    """Result returned by SpeechSegmenter.process_block()."""

    segment: np.ndarray | None
    auto_stop_triggered: bool


def _to_mono_float32(block: np.ndarray) -> np.ndarray:
    """Normalize an audio block into a contiguous 1D float32 mono copy."""
    arr = np.asarray(block, dtype=np.float32)
    if arr.ndim == 0 or arr.size == 0 or arr.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    if arr.ndim == 1:
        return np.ascontiguousarray(arr, dtype=np.float32).copy()
    if arr.ndim == 2:
        if arr.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)
        if arr.shape[1] == 1:
            return np.ascontiguousarray(arr[:, 0], dtype=np.float32).copy()
        return np.ascontiguousarray(arr.mean(axis=1, dtype=np.float32), dtype=np.float32)
    reshaped = arr.reshape(arr.shape[0], -1)
    if reshaped.shape[1] == 0:
        return np.zeros(0, dtype=np.float32)
    return np.ascontiguousarray(reshaped.mean(axis=1, dtype=np.float32), dtype=np.float32)


class SpeechSegmenter:
    """Splits speech on natural pauses (>400 ms) with zero-overlap/zero-gap sample stitching."""

    def __init__(
        self,
        sample_rate: int = 16000,
        speech_rms_threshold: float = 0.004,
        silence_rms_threshold: float = 0.003,
        min_speech_onset_ms: int = 64,
        pause_ms: int = 400,
        min_segment_speech_ms: int = 250,
        edge_pad_ms: int = 150,
        silence_auto_stop_sec: float | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")

        self.sample_rate = sample_rate
        self.speech_rms_threshold = speech_rms_threshold
        self.silence_rms_threshold = silence_rms_threshold

        self._min_speech_onset_samples = max(0, int(round(sample_rate * min_speech_onset_ms / 1000.0)))
        self._pause_samples = max(1, int(round(sample_rate * pause_ms / 1000.0)))
        self._min_segment_speech_samples = max(0, int(round(sample_rate * min_segment_speech_ms / 1000.0)))
        self._edge_pad_samples = max(0, int(round(sample_rate * edge_pad_ms / 1000.0)))
        self._silence_auto_stop_samples = (
            max(1, int(round(sample_rate * silence_auto_stop_sec)))
            if silence_auto_stop_sec is not None and silence_auto_stop_sec > 0
            else None
        )

        self._chunks: deque[np.ndarray] = deque()
        self._total_samples = 0
        self._committed_sample = 0

        self._in_speech = False
        self._has_heard_speech = False
        self._onset_speech_samples = 0
        self._segment_speech_samples = 0
        self._pause_silence_samples = 0
        self._last_speech_end_sample: int | None = None

        self._session_trailing_silence_samples = 0
        self._auto_stop_fired = False

    @property
    def has_heard_speech(self) -> bool:
        return self._has_heard_speech

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def peek_active_audio(self, max_seconds: float = 12.0, min_speech_ms: int = 400) -> np.ndarray | None:
        """Return a non-destructive copy of the currently active in-progress speech buffer."""
        min_samples = int(round(self.sample_rate * min_speech_ms / 1000.0))
        if not self._in_speech or self._segment_speech_samples < min_samples or not self._chunks:
            return None
        max_samples = max(min_samples, int(round(self.sample_rate * max_seconds)))
        parts: list[np.ndarray] = []
        collected = 0
        for chunk in reversed(self._chunks):
            parts.append(chunk)
            collected += len(chunk)
            if collected >= max_samples:
                break
        parts.reverse()
        if len(parts) == 1:
            arr = parts[0]
        else:
            arr = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        if len(arr) > max_samples:
            arr = arr[-max_samples:]
        return arr.copy()

    def _pop_samples_until(self, target_sample: int) -> np.ndarray:
        need = target_sample - self._committed_sample
        if need <= 0 or not self._chunks:
            return np.zeros(0, dtype=np.float32)

        parts: list[np.ndarray] = []
        remaining = need
        while remaining > 0 and self._chunks:
            head = self._chunks.popleft()
            if len(head) <= remaining:
                parts.append(head)
                remaining -= len(head)
            else:
                parts.append(head[:remaining])
                self._chunks.appendleft(head[remaining:])
                remaining = 0

        consumed = need - remaining
        self._committed_sample += consumed
        if not parts:
            return np.zeros(0, dtype=np.float32)
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def process_block(self, block: np.ndarray) -> SegmentEvent:
        """Process one audio block and return any emitted segment and auto-stop flag."""
        mono = _to_mono_float32(block)
        n = len(mono)
        if n == 0:
            return SegmentEvent(segment=None, auto_stop_triggered=False)

        self._chunks.append(mono)
        self._total_samples += n

        rms = math.sqrt(float(np.dot(mono, mono)) / float(n))
        segment: np.ndarray | None = None

        if not self._in_speech:
            if rms >= self.speech_rms_threshold:
                self._onset_speech_samples += n
                if self._onset_speech_samples >= self._min_speech_onset_samples:
                    self._in_speech = True
                    self._has_heard_speech = True
                    self._segment_speech_samples += self._onset_speech_samples
                    self._onset_speech_samples = 0
                    self._pause_silence_samples = 0
                    self._last_speech_end_sample = self._total_samples
                    self._session_trailing_silence_samples = 0
            else:
                self._onset_speech_samples = 0
                if self._has_heard_speech:
                    self._session_trailing_silence_samples += n
        else:
            if rms >= self.silence_rms_threshold:
                self._segment_speech_samples += n
                self._last_speech_end_sample = self._total_samples
                self._pause_silence_samples = 0
                self._session_trailing_silence_samples = 0
            else:
                self._pause_silence_samples += n
                self._session_trailing_silence_samples += n
                if self._pause_silence_samples >= self._pause_samples:
                    if (
                        self._segment_speech_samples >= self._min_segment_speech_samples
                        and self._last_speech_end_sample is not None
                    ):
                        cut_sample = min(
                            self._total_samples,
                            self._last_speech_end_sample + self._edge_pad_samples,
                        )
                        if cut_sample > self._committed_sample:
                            segment = self._pop_samples_until(cut_sample)
                        self._in_speech = False
                        self._onset_speech_samples = 0
                        self._segment_speech_samples = 0
                        self._pause_silence_samples = 0
                        self._last_speech_end_sample = None
                    else:
                        # Sub-minimum speech burst: keep buffered audio and speech count intact
                        # so subsequent speech can merge with it or finish() returns it as tail.
                        self._in_speech = False
                        self._onset_speech_samples = 0
                        self._pause_silence_samples = 0

        auto_stop_triggered = False
        if (
            self._silence_auto_stop_samples is not None
            and self._has_heard_speech
            and not self._auto_stop_fired
            and self._session_trailing_silence_samples >= self._silence_auto_stop_samples
        ):
            self._auto_stop_fired = True
            auto_stop_triggered = True

        return SegmentEvent(segment=segment, auto_stop_triggered=auto_stop_triggered)

    def finish(self) -> tuple[np.ndarray, float]:
        """Flush remaining uncommitted audio and return (tail_audio, total_duration_sec)."""
        total_duration_sec = float(self._total_samples) / float(self.sample_rate)
        if self._committed_sample >= self._total_samples or not self._chunks:
            return np.zeros(0, dtype=np.float32), total_duration_sec
        tail_audio = self._pop_samples_until(self._total_samples)
        return tail_audio, total_duration_sec
