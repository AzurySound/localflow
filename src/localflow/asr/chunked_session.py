"""Background sequential FIFO chunked ASR session."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

logger = logging.getLogger("localflow")


def merge_transcript_parts(parts: Sequence[str]) -> str:
    """Join non-empty stripped transcript parts with single spaces."""
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return " ".join(cleaned)


class ChunkedASRSession:
    """Transcribes speech segments sequentially on a single background worker thread."""

    merge_transcript_parts = staticmethod(merge_transcript_parts)

    def __init__(
        self,
        asr: Any,
        on_partial_text: Callable[[str], None] | None = None,
    ) -> None:
        self._asr = asr
        self._on_partial_text = on_partial_text
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._lock = threading.Lock()
        self._parts: list[str] = []
        self._errors: list[Exception] = []
        self._cancelled = False
        self._finalized = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="localflow-chunked-asr",
            daemon=True,
        )
        self._worker.start()

    @property
    def parts(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._parts)

    @property
    def errors(self) -> tuple[Exception, ...]:
        with self._lock:
            return tuple(self._errors)

    def submit(self, segment: np.ndarray) -> None:
        """Enqueue a speech segment for non-blocking background transcription."""
        if segment is None:
            return
        arr = np.asarray(segment, dtype=np.float32)
        if arr.size == 0:
            return
        contiguous = np.ascontiguousarray(arr, dtype=np.float32).copy()
        with self._lock:
            if self._cancelled or self._finalized:
                return
            self._queue.put(contiguous)

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                with self._lock:
                    if self._cancelled:
                        return
                try:
                    text = self._asr.transcribe(item)
                except Exception as exc:
                    logger.exception("Chunked ASR segment transcription failed")
                    with self._lock:
                        self._errors.append(exc)
                    continue

                cleaned = text.strip() if text else ""
                if not cleaned:
                    continue

                partial_text: str | None = None
                with self._lock:
                    if self._cancelled:
                        continue
                    self._parts.append(cleaned)
                    if self._on_partial_text is not None:
                        partial_text = merge_transcript_parts(self._parts)

                if partial_text is not None and self._on_partial_text is not None:
                    try:
                        self._on_partial_text(partial_text)
                    except Exception:
                        logger.exception("Chunked ASR partial text callback failed")
            finally:
                self._queue.task_done()

    def finalize(self, tail_audio: np.ndarray) -> str:
        """Drain queued segments, transcribe non-empty tail audio, and return merged transcript."""
        with self._lock:
            if self._cancelled:
                return ""
            if self._finalized:
                return merge_transcript_parts(self._parts)
            self._finalized = True
            self._queue.put(None)

        self._worker.join()

        with self._lock:
            if self._cancelled:
                return ""

        tail = np.zeros(0, dtype=np.float32) if tail_audio is None else np.asarray(tail_audio, dtype=np.float32)
        if tail.size > 0:
            try:
                tail_text = self._asr.transcribe(np.ascontiguousarray(tail, dtype=np.float32))
            except Exception as exc:
                logger.exception("Chunked ASR tail transcription failed")
                with self._lock:
                    self._errors.append(exc)
                    has_parts = bool(self._parts)
                    merged = merge_transcript_parts(self._parts)
                if has_parts:
                    return merged
                raise
            cleaned_tail = tail_text.strip() if tail_text else ""
            if cleaned_tail:
                with self._lock:
                    if not self._cancelled:
                        self._parts.append(cleaned_tail)

        with self._lock:
            if self._cancelled:
                return ""
            return merge_transcript_parts(self._parts)

    def cancel(self) -> None:
        """Cancel session without blocking; drops queued segments and suppresses callbacks."""
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            while True:
                try:
                    _ = self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    break
            self._queue.put(None)
