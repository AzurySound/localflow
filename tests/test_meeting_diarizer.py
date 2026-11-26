from __future__ import annotations

import numpy as np

from localflow.audio.loopback_recorder import resample_to_16k_mono
from localflow.notetaker.diarizer import (
    OnlineSpeakerTracker,
    compute_kaldi_fbank,
    is_echo_bleed,
    l2_normalize,
    refine_speaker_clusters,
)


def test_compute_kaldi_fbank_shape_and_normalization() -> None:
    t = np.linspace(0.0, 1.0, 16000, endpoint=False, dtype=np.float32)
    wave = 0.2 * np.sin(2.0 * np.pi * 440.0 * t)
    feats = compute_kaldi_fbank(wave, sample_rate=16000)
    assert feats.ndim == 2
    assert feats.shape[1] == 80
    assert feats.shape[0] >= 95
    assert np.allclose(feats.mean(axis=0), 0.0, atol=1e-4)


def test_resample_to_16k_mono_stereo_48k() -> None:
    # 1 second of 48kHz stereo interleaved
    samples_48k = 48000
    stereo = np.ones((samples_48k, 2), dtype=np.float32) * 0.5
    out = resample_to_16k_mono(stereo, src_rate=48000, src_channels=2, target_rate=16000)
    assert out.ndim == 1
    assert out.shape[0] == 16000
    assert np.allclose(out, 0.5, atol=1e-4)


def test_online_and_post_call_speaker_clustering() -> None:
    rng = np.random.default_rng(42)
    base_a = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0] * 16, dtype=np.float32))
    base_b = l2_normalize(np.array([0.0, 1.0, 0.0, 0.0] * 16, dtype=np.float32))
    base_c = l2_normalize(np.array([0.0, 0.0, 1.0, 0.0] * 16, dtype=np.float32))

    tracker = OnlineSpeakerTracker(similarity_threshold=0.6)
    k1 = tracker.assign(base_a)
    k2 = tracker.assign(base_b)
    k3 = tracker.assign(base_a + 0.05 * rng.normal(size=64).astype(np.float32))
    k4 = tracker.assign(base_c)

    assert k1 == "spk_1"
    assert k2 == "spk_2"
    assert k3 == "spk_1"
    assert k4 == "spk_3"

    # Short utterance with allow_new_cluster=False reuses nearest existing speaker instead of spawning spk_4
    base_d = l2_normalize(np.array([0.1, 0.2, 0.2, 0.9] * 16, dtype=np.float32))
    short_key = tracker.assign(base_d, allow_new_cluster=False)
    assert short_key in {"spk_1", "spk_2", "spk_3"}

    # Post-call agglomerative clustering merges split clusters cleanly
    items = [
        (10, base_a),
        (11, base_b),
        (12, base_a + 0.03 * rng.normal(size=64).astype(np.float32)),
        (13, base_c),
        (14, base_b + 0.03 * rng.normal(size=64).astype(np.float32)),
    ]
    remap = refine_speaker_clusters(items, similarity_threshold=0.65)
    assert remap[10] == remap[12]
    assert remap[11] == remap[14]
    assert len({remap[10], remap[11], remap[13]}) == 3


def test_echo_bleed_suppression() -> None:
    recent_sys = [(10.0, 14.0, "Please send the deployment report by Friday", 0.05)]
    assert is_echo_bleed(
        "send the deployment report by Friday",
        10.4,
        13.8,
        0.02,
        recent_sys,
    )
    assert not is_echo_bleed(
        "Sure, I will prepare the staging metrics right now",
        10.4,
        13.8,
        0.04,
        recent_sys,
    )
    # Fuzzy bigram echo bleed with minor ASR distortion still suppressed
    ru_sys = [(20.0, 25.0, "Есть ли какой то процент ошибок", 0.06)]
    assert is_echo_bleed("есть ли какой-то процент ошиба", 21.0, 24.5, 0.03, ru_sys)


def test_meeting_session_live_flow_with_vocabulary_and_stop(tmp_path) -> None:
    from localflow.asr.whisper_engine import TranscriptSegment
    from localflow.audio.loopback_recorder import AudioSegment, MeetingAudioRecorder
    from localflow.audio.segmenter import SpeechSegmenter
    from localflow.notetaker.session import MeetingSession
    from localflow.notetaker.storage import MeetingStorage
    from localflow.vocabulary import parse_vocabulary

    class _FakeASR:
        def __init__(self) -> None:
            self.prompts: list[str | None] = []

        def transcribe_segments(self, audio, *, initial_prompt=None, sample_rate=16000):
            self.prompts.append(initial_prompt)
            return [TranscriptSegment(start=0.0, end=1.2, text="Testing LocalFlow whispr")]

    class _FakeRecorder(MeetingAudioRecorder):
        def start(self, **kwargs) -> None:
            self._recording = True
            self._on_segment = kwargs.get("on_segment")
            self._mic_segmenter = SpeechSegmenter(sample_rate=16000)
            self._sys_segmenter = None
            # Feed a 0.5s tone so finish() has a real tail to flush on stop()
            tone = (0.1 * np.ones(8000, dtype=np.float32))
            self._mic_segmenter.process_block(tone)

    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    asr = _FakeASR()
    rec = _FakeRecorder()
    vocab = parse_vocabulary("Kubernetes, whispr -> Whisper")
    emitted: list[tuple[str, str]] = []

    session = MeetingSession(
        storage=storage,
        whisper_engine=asr,  # type: ignore[arg-type]
        recorder=rec,
        vocabulary=vocab,
        whisper_prompt="API",
        diarization_enabled=False,
        lang="en",
        on_utterance=lambda utt, spk: emitted.append((spk.display_name, utt.text)),
    )
    meeting = session.start(audio_mode="dual")
    session._enqueue_segment(
        AudioSegment(
            channel="mic",
            audio=np.ones(16000, dtype=np.float32) * 0.1,
            start_sec=0.0,
            end_sec=1.0,
            rms=0.1,
        )
    )
    final_rec = session.stop()

    assert final_rec is not None
    assert final_rec.status == "ready"
    assert len(emitted) >= 1
    assert emitted[0] == ("You", "Testing LocalFlow Whisper")
    assert any("Kubernetes" in (p or "") for p in asr.prompts)
    assert len(storage.get_utterances(meeting.id)) >= 1


def test_segmenter_peek_and_session_interim_and_resume(tmp_path) -> None:
    from localflow.asr.whisper_engine import TranscriptSegment
    from localflow.audio.loopback_recorder import MeetingAudioRecorder
    from localflow.audio.segmenter import SpeechSegmenter
    from localflow.notetaker.session import MeetingSession
    from localflow.notetaker.storage import MeetingStorage

    # 1. Non-destructive peek_active_audio on SpeechSegmenter
    seg = SpeechSegmenter(sample_rate=16000, pause_ms=500)
    assert seg.peek_active_audio() is None
    for _ in range(16):  # ~0.51s of active speech (>64ms onset + >400ms min_speech_ms)
        seg.process_block(np.ones(512, dtype=np.float32) * 0.08)
    assert seg.in_speech is True
    peeked = seg.peek_active_audio()
    assert peeked is not None and peeked.size >= 6400
    tail, _dur = seg.finish()
    assert tail.size >= peeked.size  # peek did not consume samples!

    # 2. MeetingSession interim streaming + resume with preserved offset
    class _FakeASR:
        def transcribe(self, audio, *, initial_prompt=None):
            return "Hello interim world"

        def transcribe_segments(self, audio, *, initial_prompt=None, sample_rate=16000):
            return [TranscriptSegment(start=0.0, end=1.0, text="Committed second part")]

    class _FakeRecorder(MeetingAudioRecorder):
        def start(self, **kwargs) -> None:
            self._recording = True
            self._time_offset_sec = float(kwargs.get("time_offset_sec", 0.0))
            self._started_at = 0.0
            self._on_segment = kwargs.get("on_segment")
            self._on_preview = kwargs.get("on_preview")
            self._mic_segmenter = SpeechSegmenter(sample_rate=16000)
            self._sys_segmenter = None

        @property
        def elapsed_sec(self) -> float:
            return self._time_offset_sec + 1.5

    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    prior = storage.create_meeting(title="Resumable Call", audio_mode="dual", status="ready")
    storage.update_meeting(prior.id, duration_sec=42.0)
    storage.ensure_speaker(prior.id, "you", display_name="You", color_idx=0, is_self=True)

    interim_events: list[tuple[int, str, str, str]] = []
    rec = _FakeRecorder()
    session = MeetingSession(
        storage=storage,
        whisper_engine=_FakeASR(),  # type: ignore[arg-type]
        recorder=rec,
        diarization_enabled=False,
        lang="en",
        on_interim_text=lambda mid, ch, spk, txt: interim_events.append((mid, ch, spk, txt)),
    )
    resumed = session.resume(prior.id)
    assert resumed is not None and resumed.id == prior.id
    assert rec._time_offset_sec == 42.0

    session._enqueue_preview("mic", np.ones(8000, dtype=np.float32) * 0.08)
    # Wait briefly for worker to process preview before stopping
    import time
    for _ in range(40):
        if interim_events:
            break
        time.sleep(0.02)

    assert any(txt == "Hello interim world" and ch == "mic" and spk == "you" for _m, ch, spk, txt in interim_events)
    stopped = session.stop()
    assert stopped is not None
    assert stopped.duration_sec >= 43.0


