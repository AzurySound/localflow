"""Microphone audio recorder using sounddevice (16kHz mono float32)."""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from localflow.audio.segmenter import SegmentEvent, SpeechSegmenter

logger = logging.getLogger("localflow")

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 512  # ~32 ms at 16 kHz


def trim_edge_silence(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    threshold: float = 0.003,
    pad_ms: int = 120,
) -> np.ndarray:
    """Trim leading and trailing silence while preserving a short padding window."""
    if audio.size == 0:
        return audio
    above = np.flatnonzero(np.abs(audio) >= threshold)
    if above.size == 0:
        return np.zeros(0, dtype=np.float32)
    pad_samples = int(sample_rate * pad_ms / 1000)
    start = max(0, int(above[0]) - pad_samples)
    end = min(len(audio), int(above[-1]) + 1 + pad_samples)
    return audio[start:end]


@dataclass(frozen=True, slots=True)
class AudioDeviceInfo:
    index: int
    name: str
    max_input_channels: int
    is_default: bool


def list_input_devices() -> list[AudioDeviceInfo]:
    """Return all available input audio devices."""
    devices = sd.query_devices()
    default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
    result: list[AudioDeviceInfo] = []
    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) > 0:
            result.append(
                AudioDeviceInfo(
                    index=idx,
                    name=str(dev["name"]),
                    max_input_channels=int(dev["max_input_channels"]),
                    is_default=(idx == default_in),
                )
            )
    return result


class AudioRecorder:
    """Captures 16 kHz mono float32 audio and emits real-time RMS levels."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._recording = False
        self._latest_level = 0.0
        self._segmenter: SpeechSegmenter | None = None
        self._on_segment: Callable[[np.ndarray], None] | None = None
        self._on_silence_timeout: Callable[[], None] | None = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def latest_level(self) -> float:
        return self._latest_level

    def start(
        self,
        device_index: int | None = None,
        *,
        pause_ms: int = 400,
        silence_auto_stop_sec: float | None = None,
        on_segment: Callable[[np.ndarray], None] | None = None,
        on_silence_timeout: Callable[[], None] | None = None,
    ) -> None:
        with self._lock:
            if self._recording:
                return
            self._chunks = []
            self._latest_level = 0.0
            self._on_segment = on_segment
            self._on_silence_timeout = on_silence_timeout
            use_segmenter = (
                on_segment is not None
                or on_silence_timeout is not None
                or silence_auto_stop_sec is not None
            )
            self._segmenter = (
                SpeechSegmenter(
                    sample_rate=self.sample_rate,
                    pause_ms=pause_ms,
                    silence_auto_stop_sec=silence_auto_stop_sec,
                )
                if use_segmenter
                else None
            )
            self._recording = True
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=CHANNELS,
                    dtype="float32",
                    blocksize=BLOCK_SIZE,
                    device=device_index,
                    callback=self._audio_callback,
                )
                self._stream.start()
            except Exception:
                self._recording = False
                self._stream = None
                self._segmenter = None
                self._on_segment = None
                self._on_silence_timeout = None
                raise

    def _stop_stream(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
            self._recording = False
            self._latest_level = 0.0

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def stop(self) -> tuple[np.ndarray, float]:
        """Stop recording and return (mono_float32_audio, duration_seconds)."""
        self._stop_stream()

        with self._lock:
            chunks = self._chunks
            self._chunks = []
            self._segmenter = None
            self._on_segment = None
            self._on_silence_timeout = None

        if not chunks:
            return np.zeros(0, dtype=np.float32), 0.0
        audio = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        duration_sec = float(len(audio)) / float(self.sample_rate)
        return audio, duration_sec

    def stop_chunked(self) -> tuple[np.ndarray, float]:
        """Stop chunked recording and return (tail_audio, total_duration_seconds)."""
        tail, _full, duration_sec = self.stop_chunked_with_full_audio()
        return tail, duration_sec

    def stop_chunked_with_full_audio(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Stop chunked recording and return (tail_audio, full_audio, total_duration_seconds)."""
        self._stop_stream()

        with self._lock:
            segmenter = self._segmenter
            chunks = self._chunks
            self._segmenter = None
            self._chunks = []
            self._on_segment = None
            self._on_silence_timeout = None

        full_audio = (
            np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
            if chunks
            else np.zeros(0, dtype=np.float32)
        )

        if segmenter is not None:
            tail_audio, duration_sec = segmenter.finish()
            return tail_audio, full_audio, duration_sec

        duration_sec = float(len(full_audio)) / float(self.sample_rate) if full_audio.size > 0 else 0.0
        return full_audio, full_audio, duration_sec

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        if not self._recording or frames <= 0 or indata.size == 0:
            return
        mono = (
            indata.astype(np.float32, copy=True)
            if indata.ndim == 1
            else indata[:, 0].astype(np.float32, copy=True)
        )
        energy = float(np.dot(mono, mono))
        rms = math.sqrt(energy / max(1, frames))
        # High-sensitivity non-linear curve tuned for typical speech RMS (0.001 .. 0.05)
        linear = min(1.0, max(0.0, (rms - 0.0007) / 0.045))
        normalized = linear ** 0.55 if linear > 0.0 else 0.0

        event: SegmentEvent | None = None
        on_segment: Callable[[np.ndarray], None] | None = None
        on_silence_timeout: Callable[[], None] | None = None

        with self._lock:
            if not self._recording:
                return
            self._chunks.append(mono)
            self._latest_level = normalized
            segmenter = self._segmenter
            on_segment = self._on_segment
            on_silence_timeout = self._on_silence_timeout
            if segmenter is not None:
                event = segmenter.process_block(mono)

        if event is not None:
            if event.segment is not None and on_segment is not None:
                try:
                    on_segment(event.segment)
                except Exception:
                    logger.exception("AudioRecorder on_segment callback failed")
            if event.auto_stop_triggered and on_silence_timeout is not None:
                try:
                    on_silence_timeout()
                except Exception:
                    logger.exception("AudioRecorder on_silence_timeout callback failed")
