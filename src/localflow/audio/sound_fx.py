"""Non-blocking soft audio cues generated in memory and played asynchronously on Windows."""

from __future__ import annotations

import io
import math
from pathlib import Path
import struct
import tempfile
import threading
import wave
import winsound

_SAMPLE_RATE = 44100


def _synthesize_chime_wav(
    notes: tuple[tuple[float, int], ...],
    volume: float = 0.38,
    gap_ms: int = 8,
) -> bytes:
    """Synthesize a warm, soft 16-bit mono PCM WAV chime with harmonic overtone and smooth envelope."""
    buf = io.BytesIO()
    gap_samples = int(_SAMPLE_RATE * gap_ms / 1000)

    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        frames = bytearray()

        for idx, (freq, dur_ms) in enumerate(notes):
            n_samples = max(1, int(_SAMPLE_RATE * dur_ms / 1000))
            attack = max(1, int(_SAMPLE_RATE * 0.008))  # 8 ms soft attack (no click)
            for i in range(n_samples):
                t = i / _SAMPLE_RATE
                # Fast smooth attack + gentle exponential decay like a soft marimba/glass chime
                if i < attack:
                    env = i / attack
                else:
                    progress = (i - attack) / max(1, n_samples - attack)
                    env = math.exp(-3.2 * progress) * (1.0 - 0.15 * progress)
                    if i > n_samples - attack:
                        env *= (n_samples - i) / attack

                tone = (
                    0.82 * math.sin(2.0 * math.pi * freq * t)
                    + 0.18 * math.sin(2.0 * math.pi * (freq * 2.0) * t)
                )
                sample = tone * env * volume
                frames.extend(struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767)))

            if idx < len(notes) - 1 and gap_samples > 0:
                frames.extend(b"\x00\x00" * gap_samples)

        wf.writeframes(bytes(frames))
    return buf.getvalue()


_SOUNDS: dict[str, bytes] = {
    # Bright ascending two-note chime on dictation start (D5 -> A5)
    "start": _synthesize_chime_wav(((587.33, 68), (880.0, 95)), volume=0.40),
    # Gentle descending two-note chime on dictation stop (G5 -> D5)
    "stop": _synthesize_chime_wav(((783.99, 65), (587.33, 95)), volume=0.38),
    # Short muted double tone on cancel (D4 -> A3)
    "cancel": _synthesize_chime_wav(((329.63, 58), (220.0, 82)), volume=0.34),
}

_WAV_CACHE_LOCK = threading.Lock()
_WAV_FILE_CACHE: dict[str, Path] = {}


def get_sound_buffer(kind: str) -> bytes | None:
    """Return pre-generated WAV bytes for the requested cue kind."""
    return _SOUNDS.get(kind)


def _get_cached_wav_path(kind: str, data: bytes) -> Path | None:
    with _WAV_CACHE_LOCK:
        cached = _WAV_FILE_CACHE.get(kind)
        if cached is not None and cached.is_file() and cached.stat().st_size == len(data):
            return cached
        try:
            sound_dir = Path(tempfile.gettempdir()) / "localflow_sounds"
            sound_dir.mkdir(parents=True, exist_ok=True)
            wav_path = sound_dir / f"cue_{kind}_v2.wav"
            if not wav_path.is_file() or wav_path.stat().st_size != len(data):
                wav_path.write_bytes(data)
            _WAV_FILE_CACHE[kind] = wav_path
            return wav_path
        except Exception:
            return None


def play_cue(kind: str) -> None:
    """Play a pre-generated sound asynchronously on Windows without blocking UI."""
    data = _SOUNDS.get(kind)
    if data is None:
        return

    # Note: winsound.PlaySound rejects SND_MEMORY | SND_ASYNC in CPython on Windows.
    # Playing from a tiny cached WAV file with SND_FILENAME | SND_ASYNC is native, instant, and non-blocking.
    wav_path = _get_cached_wav_path(kind, data)
    if wav_path is not None:
        try:
            winsound.PlaySound(
                str(wav_path),
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
            )
            return
        except Exception:
            pass

    # Fallback: synchronous memory playback on a background daemon thread
    def _fallback() -> None:
        try:
            winsound.PlaySound(data, winsound.SND_MEMORY | winsound.SND_NODEFAULT)
        except Exception:
            pass

    threading.Thread(target=_fallback, name=f"LocalFlowSound-{kind}", daemon=True).start()
