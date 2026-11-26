"""Dual-channel meeting audio recorder (Microphone via sounddevice + System Audio via PyAudioWPatch WASAPI Loopback)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Literal

import numpy as np
import sounddevice as sd

from localflow.audio.recorder import BLOCK_SIZE, CHANNELS, SAMPLE_RATE
from localflow.audio.segmenter import SpeechSegmenter

logger = logging.getLogger("localflow.loopback")

AudioChannel = Literal["mic", "system"]
AudioMode = Literal["dual", "mic_only"]


@dataclass(frozen=True, slots=True)
class LoopbackDeviceInfo:
    index: int
    name: str
    sample_rate: float
    channels: int
    is_default: bool = False


@dataclass(frozen=True, slots=True)
class AudioSegment:
    channel: AudioChannel
    audio: np.ndarray
    start_sec: float
    end_sec: float
    rms: float


def resample_to_16k_mono(
    audio: np.ndarray,
    src_rate: int,
    src_channels: int = 1,
    target_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Convert raw float32 audio (interleaved or 2D) into 16 kHz mono float32."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)

    if arr.ndim == 1:
        if src_channels > 1:
            usable = (arr.size // src_channels) * src_channels
            if usable == 0:
                return np.zeros(0, dtype=np.float32)
            mono = arr[:usable].reshape(-1, src_channels).mean(axis=1, dtype=np.float32)
        else:
            mono = arr
    elif arr.ndim == 2:
        mono = arr[:, 0] if arr.shape[1] == 1 else arr.mean(axis=1, dtype=np.float32)
    else:
        flat = arr.reshape(arr.shape[0], -1)
        mono = flat.mean(axis=1, dtype=np.float32)

    if mono.size == 0 or src_rate <= 0:
        return np.zeros(0, dtype=np.float32)
    if src_rate == target_rate:
        return np.ascontiguousarray(mono, dtype=np.float32)

    out_len = max(1, int(round(mono.size * float(target_rate) / float(src_rate))))
    x_old = np.linspace(0.0, 1.0, num=mono.size, endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False, dtype=np.float64)
    resampled = np.interp(x_new, x_old, mono.astype(np.float64, copy=False))
    return np.ascontiguousarray(resampled, dtype=np.float32)


def list_loopback_devices() -> list[LoopbackDeviceInfo]:
    """Enumerate Windows WASAPI loopback devices via PyAudioWPatch."""
    try:
        import pyaudiowpatch as pyaudio
    except Exception:
        return []

    pa = None
    try:
        pa = pyaudio.PyAudio()
        default_idx: int | None = None
        try:
            default_lb = pa.get_default_wasapi_loopback()
            if isinstance(default_lb, dict) and "index" in default_lb:
                default_idx = int(default_lb["index"])
        except Exception:
            default_idx = None

        devices: list[LoopbackDeviceInfo] = []
        for idx in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(idx)
            except Exception:
                continue
            if not info.get("isLoopbackDevice", False):
                continue
            channels = int(info.get("maxInputChannels", 2) or 2)
            rate = float(info.get("defaultSampleRate", 48000.0) or 48000.0)
            name = str(info.get("name", f"Loopback {idx}"))
            devices.append(
                LoopbackDeviceInfo(
                    index=int(info["index"]),
                    name=name,
                    sample_rate=rate,
                    channels=max(1, channels),
                    is_default=(int(info["index"]) == default_idx),
                )
            )
        return devices
    except Exception as exc:
        logger.warning("Failed to list WASAPI loopback devices: %s", exc)
        return []
    finally:
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass


class MeetingAudioRecorder:
    """Simultaneously captures Microphone (16 kHz mono) and WASAPI Loopback System Audio."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._recording = False
        self._audio_mode: AudioMode = "dual"
        self._started_at: float = 0.0
        self._time_offset_sec: float = 0.0

        self._mic_stream: sd.InputStream | None = None
        self._pa = None
        self._loopback_stream = None
        self._loopback_rate: int = 48000
        self._loopback_channels: int = 2

        self._mic_segmenter: SpeechSegmenter | None = None
        self._sys_segmenter: SpeechSegmenter | None = None

        self._on_segment: Callable[[AudioSegment], None] | None = None
        self._on_preview: Callable[[AudioChannel, np.ndarray], None] | None = None
        self._on_error: Callable[[str], None] | None = None
        self._preview_interval_sec: float = 1.15
        self._last_mic_preview_at: float = 0.0
        self._last_sys_preview_at: float = 0.0

        self._mic_level: float = 0.0
        self._system_level: float = 0.0

        # Parallel dictation tap (reuses open mic stream during an active meeting)
        self._tap_lock = threading.Lock()
        self._tap_active = False
        self._tap_chunks: list[np.ndarray] = []
        self._tap_callback: Callable[[np.ndarray], None] | None = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def audio_mode(self) -> AudioMode:
        return self._audio_mode

    @property
    def mic_level(self) -> float:
        return self._mic_level

    @property
    def system_level(self) -> float:
        return self._system_level

    @property
    def elapsed_sec(self) -> float:
        if not self._recording or self._started_at <= 0:
            return self._time_offset_sec
        return self._time_offset_sec + max(0.0, time.monotonic() - self._started_at)

    def start_dictation_tap(self, on_block: Callable[[np.ndarray], None] | None = None) -> None:
        """Start teeing microphone blocks for parallel dictation without closing the meeting stream."""
        with self._tap_lock:
            self._tap_chunks = []
            self._tap_callback = on_block
            self._tap_active = True

    def stop_dictation_tap(self) -> np.ndarray:
        """Stop parallel dictation tap and return collected 16 kHz mono float32 audio."""
        with self._tap_lock:
            self._tap_active = False
            self._tap_callback = None
            chunks = self._tap_chunks
            self._tap_chunks = []
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    def start(
        self,
        *,
        mic_device_index: int | None = None,
        loopback_device_index: int | None = None,
        audio_mode: AudioMode = "dual",
        pause_ms: int = 500,
        time_offset_sec: float = 0.0,
        on_segment: Callable[[AudioSegment], None] | None = None,
        on_preview: Callable[[AudioChannel, np.ndarray], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        with self._lock:
            if self._recording:
                return
            self._audio_mode = audio_mode
            self._time_offset_sec = max(0.0, float(time_offset_sec))
            self._on_segment = on_segment
            self._on_preview = on_preview
            self._on_error = on_error
            self._mic_level = 0.0
            self._system_level = 0.0
            self._started_at = time.monotonic()
            self._last_mic_preview_at = 0.0
            self._last_sys_preview_at = 0.0

            self._mic_segmenter = SpeechSegmenter(
                sample_rate=self.sample_rate,
                speech_rms_threshold=0.0045,
                silence_rms_threshold=0.0032,
                pause_ms=pause_ms,
                min_segment_speech_ms=280,
            )
            self._sys_segmenter = (
                SpeechSegmenter(
                    sample_rate=self.sample_rate,
                    speech_rms_threshold=0.004,
                    silence_rms_threshold=0.0028,
                    pause_ms=pause_ms,
                    min_segment_speech_ms=280,
                )
                if audio_mode == "dual"
                else None
            )
            self._recording = True

            try:
                self._mic_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=CHANNELS,
                    dtype="float32",
                    blocksize=BLOCK_SIZE,
                    device=mic_device_index,
                    callback=self._on_mic_audio,
                )
                self._mic_stream.start()
            except Exception:
                self._recording = False
                self._cleanup_streams_locked()
                raise

            if audio_mode == "dual":
                try:
                    self._start_loopback_locked(loopback_device_index)
                except Exception as exc:
                    logger.warning("WASAPI Loopback failed to start, continuing mic_only: %s", exc)
                    self._audio_mode = "mic_only"
                    self._sys_segmenter = None
                    if on_error is not None:
                        on_error(f"System audio loopback unavailable: {exc}")

    def _start_loopback_locked(self, loopback_device_index: int | None) -> None:
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        self._pa = pa
        target_info = None
        if loopback_device_index is not None:
            try:
                info = pa.get_device_info_by_index(loopback_device_index)
                if info.get("isLoopbackDevice", False):
                    target_info = info
            except Exception:
                target_info = None

        if target_info is None:
            target_info = pa.get_default_wasapi_loopback()

        self._loopback_rate = max(8000, int(round(float(target_info.get("defaultSampleRate", 48000)))))
        self._loopback_channels = max(1, int(target_info.get("maxInputChannels", 2)))
        frames_per_buffer = max(256, int(round(self._loopback_rate * 0.032)))

        self._loopback_stream = pa.open(
            format=pyaudio.paFloat32,
            channels=self._loopback_channels,
            rate=self._loopback_rate,
            input=True,
            input_device_index=int(target_info["index"]),
            frames_per_buffer=frames_per_buffer,
            stream_callback=self._on_loopback_audio,
        )
        self._loopback_stream.start_stream()

    def _emit_segment(self, channel: AudioChannel, seg: np.ndarray) -> None:
        if seg.size == 0:
            return
        cb = self._on_segment
        if cb is None:
            return
        end_sec = self._time_offset_sec + max(0.0, time.monotonic() - self._started_at)
        dur_sec = float(seg.size) / float(self.sample_rate)
        start_sec = max(0.0, end_sec - dur_sec)
        rms = math.sqrt(float(np.dot(seg, seg)) / float(seg.size))
        try:
            cb(
                AudioSegment(
                    channel=channel,
                    audio=seg,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    rms=rms,
                )
            )
        except Exception as exc:
            logger.debug("Segment callback error (%s): %s", channel, exc)

    def _maybe_emit_preview(self, channel: AudioChannel, seg_engine: SpeechSegmenter) -> None:
        cb = self._on_preview
        if cb is None or not seg_engine.in_speech:
            return
        now = time.monotonic()
        last = self._last_mic_preview_at if channel == "mic" else self._last_sys_preview_at
        if now - last < self._preview_interval_sec:
            return
        preview = seg_engine.peek_active_audio(max_seconds=12.0, min_speech_ms=420)
        if preview is None or preview.size == 0:
            return
        if channel == "mic":
            self._last_mic_preview_at = now
        else:
            self._last_sys_preview_at = now
        try:
            cb(channel, preview)
        except Exception:
            pass

    def _on_mic_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
        if not self._recording:
            return
        mono = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
        if mono.size == 0:
            return
        rms = math.sqrt(float(np.dot(mono, mono)) / float(mono.size))
        self._mic_level = min(1.0, rms * 10.0)

        with self._tap_lock:
            if self._tap_active:
                copy_block = mono.copy()
                self._tap_chunks.append(copy_block)
                tap_cb = self._tap_callback
            else:
                copy_block = None
                tap_cb = None

        if tap_cb is not None and copy_block is not None:
            try:
                tap_cb(copy_block)
            except Exception:
                pass

        seg_engine = self._mic_segmenter
        if seg_engine is not None:
            event = seg_engine.process_block(mono)
            if event.segment is not None:
                self._last_mic_preview_at = 0.0
                self._emit_segment("mic", event.segment)
            else:
                self._maybe_emit_preview("mic", seg_engine)

    def _on_loopback_audio(self, in_data: bytes, frame_count: int, time_info, status):  # noqa: ANN001
        import pyaudiowpatch as pyaudio

        if not self._recording:
            return (None, pyaudio.paComplete)
        if not in_data:
            return (None, pyaudio.paContinue)

        raw = np.frombuffer(in_data, dtype=np.float32)
        mono_16k = resample_to_16k_mono(
            raw,
            src_rate=self._loopback_rate,
            src_channels=self._loopback_channels,
            target_rate=self.sample_rate,
        )
        if mono_16k.size > 0:
            rms = math.sqrt(float(np.dot(mono_16k, mono_16k)) / float(mono_16k.size))
            self._system_level = min(1.0, rms * 10.0)
            seg_engine = self._sys_segmenter
            if seg_engine is not None:
                event = seg_engine.process_block(mono_16k)
                if event.segment is not None:
                    self._last_sys_preview_at = 0.0
                    self._emit_segment("system", event.segment)
                else:
                    self._maybe_emit_preview("system", seg_engine)

        return (None, pyaudio.paContinue)

    def _cleanup_streams_locked(self) -> None:
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop()
            except Exception:
                pass
            try:
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None

        if self._loopback_stream is not None:
            try:
                self._loopback_stream.stop_stream()
            except Exception:
                pass
            try:
                self._loopback_stream.close()
            except Exception:
                pass
            self._loopback_stream = None

        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def stop(self) -> None:
        """Stop capturing and flush any trailing speech segments."""
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._cleanup_streams_locked()
            mic_seg = self._mic_segmenter
            sys_seg = self._sys_segmenter
            self._mic_segmenter = None
            self._sys_segmenter = None

        if mic_seg is not None:
            tail, _dur = mic_seg.finish()
            if tail.size > 0:
                self._emit_segment("mic", tail)
        if sys_seg is not None:
            tail, _dur = sys_seg.finish()
            if tail.size > 0:
                self._emit_segment("system", tail)
        self._mic_level = 0.0
        self._system_level = 0.0
