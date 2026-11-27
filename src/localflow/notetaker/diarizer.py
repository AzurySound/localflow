"""Lightweight local multi-speaker diarization using ONNX WeSpeaker embeddings (no PyTorch)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import re

import numpy as np

from localflow.llm.model_catalog import find_diarization_model_path

logger = logging.getLogger("localflow.diarizer")

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)


def _hz_to_mel(hz: float) -> float:
    return 1127.0 * math.log(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (math.exp(mel / 1127.0) - 1.0)


def _build_mel_filterbank(
    *,
    sample_rate: int = 16000,
    n_fft: int = 512,
    num_mel_bins: int = 80,
    low_freq: float = 20.0,
    high_freq: float = 7600.0,
) -> np.ndarray:
    """Build a Kaldi-style triangular mel filterbank matrix of shape (n_fft // 2 + 1, num_mel_bins)."""
    num_bins = n_fft // 2 + 1
    fft_bin_width = float(sample_rate) / float(n_fft)
    mel_low = _hz_to_mel(low_freq)
    mel_high = _hz_to_mel(min(high_freq, sample_rate * 0.5))
    mel_points = np.linspace(mel_low, mel_high, num_mel_bins + 2, dtype=np.float64)

    freqs = np.arange(num_bins, dtype=np.float64) * fft_bin_width
    mels = np.array([_hz_to_mel(float(f)) for f in freqs], dtype=np.float64)

    fb = np.zeros((num_bins, num_mel_bins), dtype=np.float32)
    for m in range(num_mel_bins):
        left = mel_points[m]
        center = mel_points[m + 1]
        right = mel_points[m + 2]
        up = (mels - left) / max(1e-8, center - left)
        down = (right - mels) / max(1e-8, right - center)
        fb[:, m] = np.maximum(0.0, np.minimum(up, down)).astype(np.float32)
    return fb


_MEL_FBANK_80 = _build_mel_filterbank()


def compute_kaldi_fbank(
    audio: np.ndarray,
    *,
    sample_rate: int = 16000,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    num_mel_bins: int = 80,
) -> np.ndarray:
    """Compute 80-bin Kaldi-compatible log-mel filterbank features with CMN."""
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    win_len = int(round(sample_rate * frame_length_ms / 1000.0))  # 400
    hop_len = int(round(sample_rate * frame_shift_ms / 1000.0))  # 160
    if wav.size < win_len:
        return np.zeros((0, num_mel_bins), dtype=np.float32)

    # Remove DC offset + pre-emphasis (0.97)
    wav = wav - float(np.mean(wav))
    num_frames = 1 + (wav.size - win_len) // hop_len
    if num_frames <= 0:
        return np.zeros((0, num_mel_bins), dtype=np.float32)

    indices = np.arange(win_len, dtype=np.int64)[None, :] + (
        np.arange(num_frames, dtype=np.int64)[:, None] * hop_len
    )
    frames = wav[indices].copy()
    frames[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
    frames[:, 0] = frames[:, 0] * 0.03

    # Povey window (Hann^0.85)
    window = np.hanning(win_len).astype(np.float32) ** 0.85
    frames *= window[None, :]

    spec = np.fft.rfft(frames, n=512, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    fb = _MEL_FBANK_80 if num_mel_bins == 80 else _build_mel_filterbank(
        sample_rate=sample_rate, num_mel_bins=num_mel_bins
    )
    mel_energies = np.maximum(1e-10, power @ fb)
    log_mel = np.log(mel_energies).astype(np.float32)
    # Utterance-level cepstral mean normalization
    log_mel -= np.mean(log_mel, axis=0, keepdims=True)
    return log_mel


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec.astype(np.float32, copy=False)
    return (vec / norm).astype(np.float32, copy=False)


class OnnxSpeakerEmbedder:
    """Extracts 256-dim L2-normalized speaker embeddings using WeSpeaker ONNX on CPU."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or find_diarization_model_path()
        self._session = None
        self._input_name = "feats"
        self._output_name = "embs"

    @property
    def is_available(self) -> bool:
        return self.model_path is not None and self.model_path.is_file()

    def _ensure_loaded(self) -> bool:
        if self._session is not None:
            return True
        if self.model_path is None or not self.model_path.is_file():
            return False
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = sess.get_inputs()[0].name
            self._output_name = sess.get_outputs()[0].name
            self._session = sess
            return True
        except Exception as exc:
            logger.warning("Failed to load ONNX speaker embedder: %s", exc)
            return False

    def embed(self, audio: np.ndarray, *, sample_rate: int = 16000) -> np.ndarray | None:
        """Compute L2-normalized speaker embedding for audio >= 0.6 s after stripping silence."""
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if wav.size < int(sample_rate * 0.6):
            return None
        if not self._ensure_loaded():
            return None
        frame_len = max(160, int(sample_rate * 0.02))
        n_frames = wav.size // frame_len
        if n_frames > 0:
            framed = wav[: n_frames * frame_len].reshape(n_frames, frame_len)
            rms = np.sqrt(np.mean(framed * framed, axis=1))
            peak_rms = float(np.max(rms)) if rms.size else 0.0
            if peak_rms > 0.0:
                thresh = max(0.005, 0.12 * peak_rms)
                voiced = framed[rms >= thresh].reshape(-1)
                if voiced.size >= int(sample_rate * 0.5):
                    wav = voiced
                elif voiced.size < int(sample_rate * 0.3):
                    return None
        fbank = compute_kaldi_fbank(wav, sample_rate=sample_rate)
        if fbank.shape[0] < 35:
            return None
        inp = fbank[None, :, :].astype(np.float32, copy=False)
        try:
            outputs = self._session.run([self._output_name], {self._input_name: inp})
            emb = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            return l2_normalize(emb)
        except Exception as exc:
            logger.debug("Speaker embedding inference failed: %s", exc)
            return None


@dataclass
class _SpeakerCentroid:
    speaker_key: str
    centroid: np.ndarray
    count: int = 1


class OnlineSpeakerTracker:
    """Real-time online cosine clustering of remote/offline speakers."""

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.38,
        max_speakers: int = 12,
        prefix: str = "spk_",
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_speakers = max(2, max_speakers)
        self.prefix = prefix
        self._clusters: list[_SpeakerCentroid] = []

    def assign(self, embedding: np.ndarray | None, *, allow_new_cluster: bool = True) -> str:
        """Assign an embedding to an existing speaker cluster or create a new one."""
        if embedding is None or embedding.size == 0:
            if self._clusters:
                return self._clusters[-1].speaker_key
            key = f"{self.prefix}1"
            return key

        emb = l2_normalize(embedding)
        if not self._clusters:
            key = f"{self.prefix}1"
            self._clusters.append(_SpeakerCentroid(speaker_key=key, centroid=emb, count=1))
            return key

        best_idx = -1
        best_sim = -1.0
        for idx, cluster in enumerate(self._clusters):
            sim = float(np.dot(emb, cluster.centroid))
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        should_reuse = best_idx >= 0 and (
            best_sim >= self.similarity_threshold
            or len(self._clusters) >= self.max_speakers
            or not allow_new_cluster
        )
        if should_reuse:
            cluster = self._clusters[best_idx]
            if best_sim >= 0.35:
                n = cluster.count
                updated = l2_normalize((cluster.centroid * float(n) + emb) / float(n + 1))
                cluster.centroid = updated
                cluster.count = n + 1
            return cluster.speaker_key

        key = f"{self.prefix}{len(self._clusters) + 1}"
        self._clusters.append(_SpeakerCentroid(speaker_key=key, centroid=emb, count=1))
        return key


def refine_speaker_clusters(
    items: list[tuple[int, np.ndarray]],
    *,
    similarity_threshold: float = 0.42,
    outlier_threshold: float = 0.25,
    prefix: str = "spk_",
) -> dict[int, str]:
    """Post-call centroid clustering to merge split speaker clusters and absorb single-turn outliers."""
    if not items:
        return {}
    if len(items) == 1:
        return {items[0][0]: f"{prefix}1"}

    embs = [l2_normalize(vec) for _, vec in items]
    clusters: list[list[int]] = [[i] for i in range(len(items))]

    def _centroid(indices: list[int]) -> np.ndarray:
        acc = np.sum([embs[i] for i in indices], axis=0)
        return l2_normalize(acc)

    centroids: list[np.ndarray] = [embs[i] for i in range(len(items))]

    while len(clusters) > 1:
        centroid_mat = np.vstack(centroids)
        sim_mat = centroid_mat @ centroid_mat.T
        sim_mat[np.tril_indices(len(clusters))] = -2.0
        flat = int(np.argmax(sim_mat))
        a, b = divmod(flat, len(clusters))
        best_sim = float(sim_mat[a, b])
        if best_sim < similarity_threshold:
            break
        clusters[a] = clusters[a] + clusters[b]
        clusters.pop(b)
        centroids[a] = _centroid(clusters[a])
        centroids.pop(b)

    if len(clusters) > 1 and any(len(cl) >= 2 for cl in clusters):
        merged = True
        while merged and len(clusters) > 1:
            merged = False
            for i, cl in enumerate(clusters):
                if len(cl) != 1:
                    continue
                best_target = -1
                best_sim = -1.0
                for j, other in enumerate(clusters):
                    if j == i or len(other) < 2:
                        continue
                    sim = float(np.dot(centroids[i], centroids[j]))
                    if sim > best_sim:
                        best_sim = sim
                        best_target = j
                if best_target >= 0 and best_sim >= outlier_threshold:
                    clusters[best_target] = clusters[best_target] + cl
                    centroids[best_target] = _centroid(clusters[best_target])
                    clusters.pop(i)
                    centroids.pop(i)
                    merged = True
                    break

    # Sort clusters by earliest utterance appearance so spk_1 speaks first
    clusters.sort(key=lambda cl: min(cl))
    mapping: dict[int, str] = {}
    for spk_num, cl in enumerate(clusters, start=1):
        key = f"{prefix}{spk_num}"
        for idx in cl:
            utt_id = items[idx][0]
            mapping[utt_id] = key
    return mapping


def _norm_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _char_bigrams(text: str) -> set[str]:
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    if len(cleaned) < 2:
        return set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def is_echo_bleed(
    mic_text: str,
    mic_start: float,
    mic_end: float,
    mic_rms: float,
    recent_system: list[tuple[float, float, str, float]],
    *,
    time_tolerance_sec: float = 1.8,
    overlap_threshold: float = 0.6,
) -> bool:
    """Detect whether a mic utterance is acoustic speaker bleed of a concurrent system utterance."""
    mic_words = _norm_words(mic_text)
    if not mic_words:
        return False
    mic_set = set(mic_words)
    mic_bi = _char_bigrams(mic_text)

    for sys_start, sys_end, sys_text, sys_rms in recent_system:
        if mic_end < sys_start - time_tolerance_sec or mic_start > sys_end + time_tolerance_sec:
            continue
        sys_words = set(_norm_words(sys_text))
        if not sys_words:
            continue
        intersection = len(mic_set & sys_words)
        ratio = float(intersection) / float(len(mic_set))
        if ratio >= overlap_threshold:
            if sys_rms >= mic_rms * 0.5 or len(mic_words) <= 8:
                return True
        if mic_bi and len(mic_words) <= 18:
            sys_bi = _char_bigrams(sys_text)
            if sys_bi:
                bi_jaccard = float(len(mic_bi & sys_bi)) / float(len(mic_bi | sys_bi))
                if bi_jaccard >= 0.45 and sys_rms >= mic_rms * 0.4:
                    return True
    return False
