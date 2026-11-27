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


def test_diarizer_absorbs_singleton_outliers_and_remaps_short_utterances(tmp_path) -> None:
    from localflow.asr.whisper_engine import TranscriptSegment
    from localflow.audio.loopback_recorder import AudioSegment, MeetingAudioRecorder
    from localflow.notetaker.diarizer import OnnxSpeakerEmbedder
    from localflow.notetaker.session import MeetingSession
    from localflow.notetaker.storage import MeetingStorage

    # Singleton noisy outlier with similarity ~0.35 to a multi-turn cluster is absorbed (not left as extra speaker)
    base_a = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0] * 16, dtype=np.float32))
    base_b = l2_normalize(np.array([0.0, 1.0, 0.0, 0.0] * 16, dtype=np.float32))
    outlier_a = l2_normalize(0.35 * base_a + 0.93 * l2_normalize(np.array([0.0, 0.0, 1.0, 0.0] * 16, dtype=np.float32)))
    remap = refine_speaker_clusters([(1, base_a), (2, base_a), (3, base_b), (4, base_b), (5, outlier_a)])
    assert len(set(remap.values())) == 2
    assert remap[5] == remap[1]

    # Short <0.6s utterance with emb=None is remapped cleanly when post-call clustering merges speakers
    class _FakeASR:
        def transcribe_segments(self, audio, *, initial_prompt=None, sample_rate=16000):
            return [TranscriptSegment(start=0.0, end=max(0.4, audio.size / float(sample_rate)), text="Test line")]

    class _FakeEmbedder(OnnxSpeakerEmbedder):
        @property
        def is_available(self) -> bool:
            return True

        def embed(self, audio: np.ndarray, *, sample_rate: int = 16000) -> np.ndarray | None:
            if audio.size < int(sample_rate * 0.6):
                return None
            # Return two slightly drifted vectors of the same speaker that online splits (>0.38) or merges
            return l2_normalize(base_a + (0.05 if audio.size < 48000 else -0.05) * base_b)

    class _FakeRecorder(MeetingAudioRecorder):
        def start(self, **kwargs) -> None:
            self._recording = True
        def stop(self) -> None:
            self._recording = False
        @property
        def elapsed_sec(self) -> float:
            return 8.0

    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    sess = MeetingSession(
        storage=storage,
        whisper_engine=_FakeASR(),  # type: ignore[arg-type]
        recorder=_FakeRecorder(),
        embedder=_FakeEmbedder(),
        diarization_enabled=True,
    )
    m = sess.start(audio_mode="dual")
    # Force an initial split in online tracker, plus a short 0.4s utterance (emb=None) tagged to spk_2
    sess._tracker.similarity_threshold = 0.999
    sess._process_segment(AudioSegment(channel="system", audio=np.ones(16000 * 3, dtype=np.float32) * 0.1, start_sec=0.0, end_sec=3.0, rms=0.1))
    sess._process_segment(AudioSegment(channel="system", audio=np.ones(16000 * 3, dtype=np.float32) * 0.1, start_sec=3.0, end_sec=6.0, rms=0.1))
    sess._process_segment(AudioSegment(channel="system", audio=np.ones(int(16000 * 0.4), dtype=np.float32) * 0.1, start_sec=6.0, end_sec=6.4, rms=0.1))
    sess.stop()

    speakers = [s.speaker_key for s in storage.get_speakers(m.id) if not s.is_self]
    utts = storage.get_utterances(m.id)
    assert speakers == ["spk_1"]
    assert all(u.speaker_key == "spk_1" for u in utts)


def test_meeting_session_retries_unexpected_language_and_locks_dominant_language(tmp_path) -> None:
    from localflow.asr.whisper_engine import TranscriptSegment
    from localflow.audio.loopback_recorder import AudioSegment, MeetingAudioRecorder
    from localflow.notetaker.session import MeetingSession
    from localflow.notetaker.storage import MeetingStorage

    class _LangFakeASR:
        def __init__(self) -> None:
            self.calls: list[str | None] = []
            self.last_detected_language: tuple[str, float] | None = None
            self._turn = 0

        def transcribe_segments(self, audio, *, initial_prompt=None, sample_rate=16000, language=None):
            self.calls.append(language)
            if language is None and self._turn == 0:
                self._turn += 1
                self.last_detected_language = ("is", 0.42)
                return [TranscriptSegment(start=0.0, end=2.0, text="Við popalið vissu")]
            self._turn += 1
            self.last_detected_language = ("ru", 0.94)
            return [TranscriptSegment(start=0.0, end=2.0, text="Мы попали в среднюю лигу")]

    class _FakeRecorder(MeetingAudioRecorder):
        def start(self, **kwargs) -> None:
            self._recording = True
        def stop(self) -> None:
            self._recording = False
        @property
        def elapsed_sec(self) -> float:
            return 10.0

    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    asr = _LangFakeASR()
    sess = MeetingSession(
        storage=storage,
        whisper_engine=asr,  # type: ignore[arg-type]
        recorder=_FakeRecorder(),
        diarization_enabled=False,
        lang="ru",
    )
    m = sess.start(audio_mode="dual")
    seg = AudioSegment(channel="system", audio=np.ones(16000 * 2, dtype=np.float32) * 0.1, start_sec=0.0, end_sec=2.0, rms=0.1)

    # 1st segment misdetects Icelandic ("is", 0.42) -> retried immediately with fallback "ru"
    sess._process_segment(seg)
    assert asr.calls == [None, "ru"]

    # Next 3 high-confidence Russian segments lock language to "ru"
    for _ in range(3):
        sess._process_segment(seg)
    assert sess._locked_language == "ru"

    # Subsequent segment passes language="ru" directly (no auto-detect gamble)
    sess._process_segment(seg)
    assert asr.calls[-1] == "ru"
    utts = storage.get_utterances(m.id)
    assert all("Við" not in u.text for u in utts)
    sess.stop()


def test_session_preserves_custom_speaker_name_across_post_call_remap(tmp_path) -> None:
    from localflow.asr.whisper_engine import TranscriptSegment
    from localflow.audio.loopback_recorder import AudioSegment, MeetingAudioRecorder
    from localflow.notetaker.diarizer import OnnxSpeakerEmbedder
    from localflow.notetaker.session import MeetingSession
    from localflow.notetaker.storage import MeetingStorage

    base_a = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0] * 16, dtype=np.float32))

    class _FakeASR:
        def transcribe_segments(self, audio, *, initial_prompt=None, sample_rate=16000):
            return [TranscriptSegment(start=0.0, end=2.5, text="Segment speech")]

    class _FakeEmbedder(OnnxSpeakerEmbedder):
        @property
        def is_available(self) -> bool:
            return True

        def embed(self, audio: np.ndarray, *, sample_rate: int = 16000) -> np.ndarray | None:
            return base_a

    class _FakeRecorder(MeetingAudioRecorder):
        def start(self, **kwargs) -> None:
            self._recording = True
        def stop(self) -> None:
            self._recording = False
        @property
        def elapsed_sec(self) -> float:
            return 6.0

    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    sess = MeetingSession(
        storage=storage,
        whisper_engine=_FakeASR(),  # type: ignore[arg-type]
        recorder=_FakeRecorder(),
        embedder=_FakeEmbedder(),
        diarization_enabled=True,
        lang="ru",
    )
    m = sess.start(audio_mode="dual")
    seg = AudioSegment(channel="system", audio=np.ones(16000 * 3, dtype=np.float32) * 0.1, start_sec=0.0, end_sec=3.0, rms=0.1)
    sess._process_segment(seg)
    sess._process_segment(seg)

    # User renames spk_1 -> "Алексей" during the live meeting
    storage.rename_speaker(m.id, "spk_1", "Алексей")
    sess.stop()

    speakers = {s.speaker_key: s.display_name for s in storage.get_speakers(m.id) if not s.is_self}
    assert speakers == {"spk_1": "Алексей"}


def test_meeting_recorder_dictation_tap_isolates_mic_from_meeting() -> None:
    import sys
    import types
    from unittest.mock import patch

    from localflow.audio.loopback_recorder import MeetingAudioRecorder

    class _FakeSdStream:
        def __init__(self, **kwargs) -> None:
            self.active = True

        def start(self) -> None:
            self.active = True

        def stop(self) -> None:
            self.active = False

        def close(self) -> None:
            self.active = False

    class _FakeLbStream:
        def __init__(self) -> None:
            self.active = True

        def start_stream(self) -> None:
            self.active = True

        def is_active(self) -> bool:
            return self.active

        def stop_stream(self) -> None:
            self.active = False

        def close(self) -> None:
            self.active = False

    class _FakePyAudio:
        def get_default_wasapi_loopback(self) -> dict:
            return {
                "index": 99,
                "name": "Fake Loopback",
                "defaultSampleRate": 16000.0,
                "maxInputChannels": 1,
                "isLoopbackDevice": True,
            }

        def open(self, **kwargs) -> _FakeLbStream:
            return _FakeLbStream()

        def terminate(self) -> None:
            pass

    fake_pa_mod = types.SimpleNamespace(PyAudio=_FakePyAudio, paFloat32=1, paContinue=0, paComplete=1)

    segments = []
    previews: list[tuple[str, int]] = []

    rec = MeetingAudioRecorder(sample_rate=16000)
    rec._watchdog_interval_sec = 5.0

    def _feed_mic(seconds: float, val: float) -> None:
        total = int(round(16000 * seconds))
        for off in range(0, total, 512):
            blk = np.full((min(512, total - off), 1), val, dtype=np.float32)
            rec._on_mic_audio(blk, len(blk), None, None)

    def _feed_sys(seconds: float, val: float) -> None:
        total = int(round(16000 * seconds))
        for off in range(0, total, 512):
            blk = np.full(min(512, total - off), val, dtype=np.float32).tobytes()
            rec._on_loopback_audio(blk, 512, None, None)

    with (
        patch("localflow.audio.loopback_recorder.sd.InputStream", _FakeSdStream),
        patch.dict(sys.modules, {"pyaudiowpatch": fake_pa_mod}),
        patch("localflow.audio.loopback_recorder.get_default_render_endpoint_id", return_value="ep-1"),
    ):
        rec.start(
            audio_mode="dual",
            pause_ms=400,
            on_segment=segments.append,
            on_preview=lambda ch, arr: previews.append((ch, int(arr.size))),
        )

        # 1. Normal meeting mic speech emits a mic segment
        _feed_mic(0.512, 0.02)
        _feed_mic(0.448, 0.0005)
        mic_before = [s for s in segments if s.channel == "mic"]
        assert len(mic_before) == 1

        # 2. Active dictation tap blocks meeting mic segments & previews while system audio still works
        _feed_mic(0.48, 0.02)
        previews.clear()
        rec.start_dictation_tap()
        assert rec.is_dictation_tap_active is True
        assert previews and previews[-1] == ("mic", 0)

        previews.clear()
        _feed_mic(1.0, 0.03)
        _feed_mic(0.5, 0.0005)
        _feed_mic(1.0, 0.03)
        _feed_sys(0.512, 0.025)
        _feed_sys(0.448, 0.0005)

        assert len([s for s in segments if s.channel == "mic"]) == 1
        assert not any(ch == "mic" and sz > 0 for ch, sz in previews)
        assert len([s for s in segments if s.channel == "system"]) == 1

        # 3. Stopping dictation returns tapped audio and cooldown prevents trailing bleed
        tapped = rec.stop_dictation_tap()
        assert tapped.size == int(16000 * 2.5)
        assert rec.is_dictation_tap_active is False

        _feed_mic(0.192, 0.03)
        _feed_mic(0.448, 0.0005)
        assert len([s for s in segments if s.channel == "mic"]) == 1

        # 4. Once cooldown expires, meeting mic resumes transcribing cleanly
        rec._tap_cooldown_until = 0.0
        _feed_mic(0.512, 0.02)
        _feed_mic(0.448, 0.0005)
        assert len([s for s in segments if s.channel == "mic"]) == 2

        # 5. Cancelled dictation (stop_dictation_tap ignored) restores clean meeting mic state
        rec.start_dictation_tap()
        _feed_mic(0.512, 0.02)
        _ = rec.stop_dictation_tap()
        rec._tap_cooldown_until = 0.0
        _feed_mic(0.512, 0.02)
        _feed_mic(0.448, 0.0005)
        assert len([s for s in segments if s.channel == "mic"]) == 3

        rec.stop()


def test_meeting_recorder_loopback_watchdog_recovers_dead_stream_and_endpoint_switch() -> None:
    import sys
    import time
    import types
    from unittest.mock import patch

    from localflow.audio.loopback_recorder import MeetingAudioRecorder

    class _FakeSdStream:
        def __init__(self, **kwargs) -> None:
            self.active = True

        def start(self) -> None:
            self.active = True

        def stop(self) -> None:
            self.active = False

        def close(self) -> None:
            self.active = False

    lb_streams = []
    pa_instances = []
    endpoint_state = {"id": "ep-speakers"}

    class _FakeLbStream:
        def __init__(self) -> None:
            self.active = False
            self.stopped = False
            self.closed = False

        def start_stream(self) -> None:
            self.active = True

        def is_active(self) -> bool:
            return self.active

        def stop_stream(self) -> None:
            self.stopped = True
            self.active = False

        def close(self) -> None:
            self.closed = True
            self.active = False

    class _FakePyAudio:
        def __init__(self) -> None:
            self.terminated = False
            pa_instances.append(self)

        def get_default_wasapi_loopback(self) -> dict:
            return {
                "index": 50 + len(pa_instances),
                "name": f"Fake LB {endpoint_state['id']}",
                "defaultSampleRate": 16000.0,
                "maxInputChannels": 1,
                "isLoopbackDevice": True,
            }

        def open(self, **kwargs) -> _FakeLbStream:
            s = _FakeLbStream()
            lb_streams.append(s)
            return s

        def terminate(self) -> None:
            self.terminated = True

    fake_pa_mod = types.SimpleNamespace(PyAudio=_FakePyAudio, paFloat32=1, paContinue=0, paComplete=1)

    segments = []
    rec = MeetingAudioRecorder(sample_rate=16000)
    rec._watchdog_interval_sec = 0.03
    rec._loopback_stall_timeout_sec = 0.12
    rec._mic_stall_timeout_sec = 5.0

    def _wait_for(pred, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return True
            time.sleep(0.015)
        return False

    def _feed_sys(seconds: float, val: float) -> None:
        total = int(round(16000 * seconds))
        for off in range(0, total, 512):
            blk = np.full(min(512, total - off), val, dtype=np.float32).tobytes()
            rec._on_loopback_audio(blk, 512, None, None)

    with (
        patch("localflow.audio.loopback_recorder.sd.InputStream", _FakeSdStream),
        patch.dict(sys.modules, {"pyaudiowpatch": fake_pa_mod}),
        patch(
            "localflow.audio.loopback_recorder.get_default_render_endpoint_id",
            side_effect=lambda: endpoint_state["id"],
        ),
    ):
        rec.start(audio_mode="dual", pause_ms=300, on_segment=segments.append)
        assert len(lb_streams) == 1

        # Case 1: WASAPI loopback stream dies (e.g. Bluetooth disconnect)
        lb_streams[0].active = False
        assert _wait_for(lambda: len(lb_streams) >= 2)
        assert lb_streams[0].stopped and lb_streams[0].closed
        assert pa_instances[0].terminated is True

        _feed_sys(0.512, 0.03)
        _feed_sys(0.352, 0.0005)
        assert len([s for s in segments if s.channel == "system"]) == 1

        # Case 2: Windows default output switches to another endpoint
        rec._last_loopback_callback_at = time.monotonic()
        endpoint_state["id"] = "ep-sony-bt"
        assert _wait_for(lambda: len(lb_streams) >= 3)
        assert pa_instances[1].terminated is True

        # Case 3: Exception inside loopback callback never kills the stream
        with patch.object(rec._sys_segmenter, "process_block", side_effect=RuntimeError("boom")):
            res = rec._on_loopback_audio(np.full(512, 0.02, dtype=np.float32).tobytes(), 512, None, None)
        assert res == (None, fake_pa_mod.paContinue)

        rec.stop()
        assert rec.is_recording is False
        assert rec._watchdog_thread is None


def test_refine_speaker_clusters_fast_on_large_meeting() -> None:
    import time

    rng = np.random.default_rng(123)
    centers = [l2_normalize(rng.standard_normal(256).astype(np.float32)) for _ in range(4)]
    items = [
        (i, l2_normalize(centers[i % 4] + 0.08 * rng.standard_normal(256).astype(np.float32)))
        for i in range(380)
    ]
    t0 = time.perf_counter()
    remap = refine_speaker_clusters(items)
    elapsed = time.perf_counter() - t0
    assert len(set(remap.values())) == 4
    assert elapsed < 1.0



