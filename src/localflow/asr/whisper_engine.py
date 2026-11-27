"""Local ASR engine powered by faster-whisper."""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from localflow.asr.hallucinations import clean_whisper_hallucinations
from localflow.audio.recorder import trim_edge_silence

logger = logging.getLogger("localflow.asr")


def _register_cuda_dll_directories() -> None:
    """Ensure CUDA 12 runtime DLLs (cublas64_12.dll, etc.) are discoverable on Windows."""
    if sys.platform != "win32":
        return

    candidates: list[Path] = []

    # 0. PyInstaller frozen bundle directories
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass))
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)

    # 1. PyPI nvidia-* packages inside the active environment
    for base in sys.path:
        if not base:
            continue
        nvidia_dir = Path(base) / "nvidia"
        if nvidia_dir.is_dir():
            for sub in nvidia_dir.iterdir():
                bin_dir = sub / "bin"
                if bin_dir.is_dir():
                    candidates.append(bin_dir)

    # 2. Local Ollama CUDA 12 runtime (often already installed on Windows)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        ollama_cuda = Path(local_app_data) / "Programs" / "Ollama" / "lib" / "ollama" / "cuda_v12"
        if ollama_cuda.is_dir():
            candidates.append(ollama_cuda)

    # 3. Standard CUDA_PATH
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cuda_bin = Path(cuda_path) / "bin"
        if cuda_bin.is_dir():
            candidates.append(cuda_bin)

    existing_path = os.environ.get("PATH", "")
    added: list[str] = []
    for directory in candidates:
        dir_str = str(directory)
        try:
            os.add_dll_directory(dir_str)
        except Exception:
            pass
        if dir_str.lower() not in existing_path.lower():
            added.append(dir_str)

    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + existing_path


@dataclass(frozen=True, slots=True)
class ASRStatus:
    ready: bool
    device: str
    compute_type: str
    model_name: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """Single timestamped segment returned by WhisperEngine.transcribe_segments()."""

    start: float
    end: float
    text: str


class WhisperEngine:
    """Thread-safe wrapper around faster-whisper with GPU auto-detection and CPU fallback."""

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 2,
        initial_prompt: str = "",
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.beam_size = max(1, beam_size)
        self.initial_prompt = initial_prompt

        self._lock = threading.Lock()
        self._model = None
        self._active_device = "unknown"
        self._active_compute_type = "unknown"
        self._last_error: str | None = None
        self._last_detected_language: tuple[str, float] | None = None

    @property
    def last_detected_language(self) -> tuple[str, float] | None:
        return self._last_detected_language

    @property
    def status(self) -> ASRStatus:
        return ASRStatus(
            ready=self._model is not None,
            device=self._active_device,
            compute_type=self._active_compute_type,
            model_name=self.model_name,
            error=self._last_error,
        )

    def load(
        self,
        *,
        on_progress=None,
        cancel_event: threading.Event | None = None,
        lang: str | None = None,
    ) -> ASRStatus:
        """Load and warm up the Whisper model into memory (downloading into local data dir if needed)."""
        _register_cuda_dll_directories()
        from faster_whisper import WhisperModel
        from localflow.llm.downloader import download_whisper_model, find_local_whisper_model_path

        local_dir = find_local_whisper_model_path(self.model_name)
        if local_dir is None:
            local_dir = download_whisper_model(
                self.model_name,
                on_progress=on_progress,
                cancel_event=cancel_event,
                lang=lang,
            )
        model_source = str(local_dir)

        with self._lock:
            candidates: list[tuple[str, str]]
            if self.requested_device == "auto":
                ct = "float16" if self.requested_compute_type == "auto" else self.requested_compute_type
                candidates = [("cuda", ct), ("cpu", "int8")]
            else:
                ct = self.requested_compute_type
                if ct == "auto":
                    ct = "float16" if self.requested_device == "cuda" else "int8"
                candidates = [(self.requested_device, ct)]

            last_err: Exception | None = None
            warmup_audio = np.zeros(3200, dtype=np.float32)  # 0.2s silence to verify CUDA kernels & DLLs

            for dev, ct in candidates:
                try:
                    model = WhisperModel(
                        model_source,
                        device=dev,
                        compute_type=ct,
                        local_files_only=True,
                    )
                    # Run warm-up inference so missing cublas/cudnn DLLs fail here and fall back cleanly
                    list(model.transcribe(warmup_audio, beam_size=1, vad_filter=False)[0])
                    self._model = model
                    self._active_device = dev
                    self._active_compute_type = ct
                    self._last_error = None
                    logger.info("Loaded Whisper model=%s device=%s compute_type=%s", self.model_name, dev, ct)
                    return self.status
                except Exception as exc:
                    logger.warning("Failed Whisper load/warmup on (%s, %s): %s", dev, ct, exc)
                    last_err = exc

            self._model = None
            self._last_error = str(last_err) if last_err else "Unknown ASR load error"
            raise RuntimeError(f"Failed to load Whisper model '{self.model_name}': {self._last_error}")

    def unload(self) -> None:
        """Release the loaded model from GPU/RAM."""
        with self._lock:
            self._model = None
            self._active_device = "unloaded"
            self._active_compute_type = "unloaded"
        gc.collect()

    def set_initial_prompt(self, initial_prompt: str) -> None:
        """Update the default Whisper initial_prompt in a thread-safe manner."""
        with self._lock:
            self.initial_prompt = initial_prompt

    def _run_model_transcribe(
        self,
        audio: np.ndarray,
        effective_prompt: str,
        *,
        language: str | None = None,
    ):
        """Invoke faster-whisper with anti-hallucination decoding guards."""
        assert self._model is not None
        vad_params = {
            "min_silence_duration_ms": 350,
            "speech_pad_ms": 180,
            "threshold": 0.5,
        }
        try:
            return self._model.transcribe(
                audio,
                language=language,
                beam_size=self.beam_size,
                vad_filter=True,
                vad_parameters=vad_params,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                hallucination_silence_threshold=1.5,
                initial_prompt=effective_prompt or None,
            )
        except TypeError:
            return self._model.transcribe(
                audio,
                language=language,
                beam_size=self.beam_size,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 350},
                condition_on_previous_text=False,
                initial_prompt=effective_prompt or None,
            )

    def _record_detected_language(self, info) -> None:
        lang = getattr(info, "language", None)
        prob = getattr(info, "language_probability", None)
        if isinstance(lang, str) and lang:
            self._last_detected_language = (lang.lower(), float(prob or 0.0))
        else:
            self._last_detected_language = None

    @staticmethod
    def _filter_segment_text(seg) -> str:
        """Drop low-confidence non-speech segments and clean known Whisper hallucinations."""
        no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
        avg_logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
        if no_speech > 0.65 and avg_logprob < -0.9:
            return ""
        raw_txt = seg.text.strip() if getattr(seg, "text", None) else ""
        if not raw_txt:
            return ""
        return clean_whisper_hallucinations(raw_txt)

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        initial_prompt: str | None = None,
        language: str | None = None,
    ) -> str:
        """Transcribe a 16 kHz mono float32 numpy array."""
        trimmed = trim_edge_silence(audio)
        if trimmed.size == 0:
            return ""

        with self._lock:
            if self._model is None:
                raise RuntimeError("ASR model is not loaded yet.")
            effective_prompt = self.initial_prompt if initial_prompt is None else initial_prompt
            segments, info = self._run_model_transcribe(trimmed, effective_prompt, language=language)
            self._record_detected_language(info)
            parts = [txt for seg in segments if (txt := self._filter_segment_text(seg))]
            return clean_whisper_hallucinations(" ".join(parts).strip())

    def transcribe_segments(
        self,
        audio: np.ndarray,
        *,
        initial_prompt: str | None = None,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> list[TranscriptSegment]:
        """Transcribe a 16 kHz mono float32 array and return timestamped segments."""
        if audio.size == 0:
            return []

        above = np.flatnonzero(np.abs(audio) >= 0.003)
        if above.size == 0:
            return []
        pad_samples = int(sample_rate * 120 / 1000)
        trim_start = max(0, int(above[0]) - pad_samples)
        trim_end = min(len(audio), int(above[-1]) + 1 + pad_samples)
        trimmed = audio[trim_start:trim_end]
        if trimmed.size == 0:
            return []
        offset_sec = float(trim_start) / float(max(1, sample_rate))

        with self._lock:
            if self._model is None:
                raise RuntimeError("ASR model is not loaded yet.")
            effective_prompt = self.initial_prompt if initial_prompt is None else initial_prompt
            segments, info = self._run_model_transcribe(trimmed, effective_prompt, language=language)
            self._record_detected_language(info)
            result: list[TranscriptSegment] = []
            for seg in segments:
                txt = self._filter_segment_text(seg)
                if not txt:
                    continue
                seg_start = max(0.0, float(getattr(seg, "start", 0.0)) + offset_sec)
                seg_end = max(seg_start, float(getattr(seg, "end", seg_start)) + offset_sec)
                result.append(TranscriptSegment(start=seg_start, end=seg_end, text=txt))
            return result
