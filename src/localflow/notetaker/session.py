"""Live MeetingSession orchestrator: dual capture, ASR, echo suppression, diarization, and storage."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
import logging
import threading

import numpy as np

from localflow.asr.whisper_engine import TranscriptSegment, WhisperEngine
from localflow.audio.loopback_recorder import AudioMode, AudioSegment, MeetingAudioRecorder
from localflow.i18n import tr
from localflow.notetaker.diarizer import (
    OnlineSpeakerTracker,
    OnnxSpeakerEmbedder,
    is_echo_bleed,
    refine_speaker_clusters,
)
from localflow.notetaker.storage import MeetingRecord, MeetingSpeaker, MeetingStorage, MeetingUtterance
from localflow.vocabulary import (
    ParsedVocabulary,
    apply_vocabulary_replacements,
    build_whisper_initial_prompt,
)

logger = logging.getLogger("localflow.notetaker.session")


class MeetingSession:
    """Runs a live meeting transcription + speaker diarization pipeline in the background."""

    def __init__(
        self,
        *,
        storage: MeetingStorage,
        whisper_engine: WhisperEngine,
        recorder: MeetingAudioRecorder | None = None,
        embedder: OnnxSpeakerEmbedder | None = None,
        vocabulary: ParsedVocabulary | None = None,
        whisper_prompt: str = "",
        diarization_enabled: bool = True,
        lang: str = "en",
        on_utterance: Callable[[MeetingUtterance, MeetingSpeaker], None] | None = None,
        on_interim_text: Callable[[int, str, str, str], None] | None = None,
        on_speakers_updated: Callable[[int], None] | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.storage = storage
        self.whisper_engine = whisper_engine
        self.recorder = recorder or MeetingAudioRecorder()
        self.embedder = embedder or OnnxSpeakerEmbedder()
        self.vocabulary = vocabulary
        self.whisper_prompt = whisper_prompt
        self.diarization_enabled = diarization_enabled
        self.lang = lang

        self.on_utterance = on_utterance
        self.on_interim_text = on_interim_text
        self.on_speakers_updated = on_speakers_updated
        self.on_warning = on_warning

        self._meeting: MeetingRecord | None = None
        self._cond = threading.Condition()
        self._segments_queue: deque[AudioSegment | None] = deque()
        self._pending_preview: tuple[str, np.ndarray] | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._tracker = OnlineSpeakerTracker()
        self._utterance_embeddings: list[tuple[int, np.ndarray]] = []
        self._remote_utterance_ids: list[int] = []
        self._recent_system: deque[tuple[float, float, str, float]] = deque(maxlen=24)
        self._last_remote_speaker_key: str = "spk_1"
        self._speaker_counter = 0
        self._lang_votes: dict[str, int] = {}
        self._locked_language: str | None = None

    @property
    def active_meeting(self) -> MeetingRecord | None:
        return self._meeting

    @property
    def is_active(self) -> bool:
        return self._meeting is not None and not self._stop_event.is_set()

    def start(
        self,
        *,
        title: str | None = None,
        audio_mode: AudioMode = "dual",
        mic_device_index: int | None = None,
        loopback_device_index: int | None = None,
    ) -> MeetingRecord:
        if self._meeting is not None:
            return self._meeting

        default_title = title or tr("notetaker.default_title", self.lang)
        meeting = self.storage.create_meeting(
            title=default_title,
            audio_mode=audio_mode,
            status="recording",
        )
        self._meeting = meeting
        self._stop_event.clear()
        self._tracker = OnlineSpeakerTracker()
        self._utterance_embeddings = []
        self._remote_utterance_ids = []
        self._recent_system.clear()
        self._last_remote_speaker_key = "spk_1"
        self._lang_votes.clear()
        self._locked_language = None
        with self._cond:
            self._segments_queue.clear()
            self._pending_preview = None

        if audio_mode == "dual":
            self.storage.ensure_speaker(
                meeting.id,
                "you",
                display_name=tr("notetaker.speaker_you", self.lang),
                color_idx=0,
                is_self=True,
            )

        self._worker = threading.Thread(
            target=self._process_loop,
            name="LocalFlow-MeetingWorker",
            daemon=True,
        )
        self._worker.start()

        self.recorder.start(
            mic_device_index=mic_device_index,
            loopback_device_index=loopback_device_index,
            audio_mode=audio_mode,
            time_offset_sec=0.0,
            on_segment=self._enqueue_segment,
            on_preview=self._enqueue_preview,
            on_error=self._handle_recorder_warning,
        )
        return meeting

    def resume(
        self,
        meeting_id: int,
        *,
        mic_device_index: int | None = None,
        loopback_device_index: int | None = None,
    ) -> MeetingRecord | None:
        """Resume recording into an existing meeting record with preserved timestamp offset."""
        if self._meeting is not None:
            return self._meeting
        meeting = self.storage.get_meeting(meeting_id)
        if meeting is None:
            return None

        self.storage.update_meeting(meeting.id, status="recording")
        meeting = self.storage.get_meeting(meeting.id) or meeting
        self._meeting = meeting
        self._stop_event.clear()
        self._recent_system.clear()
        with self._cond:
            self._segments_queue.clear()
            self._pending_preview = None

        mode: AudioMode = "mic_only" if meeting.audio_mode == "mic_only" else "dual"
        self._worker = threading.Thread(
            target=self._process_loop,
            name="LocalFlow-MeetingWorker",
            daemon=True,
        )
        self._worker.start()

        self.recorder.start(
            mic_device_index=mic_device_index,
            loopback_device_index=loopback_device_index,
            audio_mode=mode,
            time_offset_sec=meeting.duration_sec,
            on_segment=self._enqueue_segment,
            on_preview=self._enqueue_preview,
            on_error=self._handle_recorder_warning,
        )
        return meeting

    def _handle_recorder_warning(self, msg: str) -> None:
        if self.on_warning is not None:
            self.on_warning(msg)

    def _enqueue_segment(self, seg: AudioSegment) -> None:
        if not self._stop_event.is_set():
            with self._cond:
                self._segments_queue.append(seg)
                if self._pending_preview is not None and self._pending_preview[0] == seg.channel:
                    self._pending_preview = None
                self._cond.notify()

    def _enqueue_preview(self, channel: str, audio: np.ndarray) -> None:
        if not self._stop_event.is_set():
            with self._cond:
                self._pending_preview = (channel, audio)
                self._cond.notify()

    def _speaker_label_for_key(self, key: str) -> tuple[str, int, bool]:
        if key == "you":
            return (tr("notetaker.speaker_you", self.lang), 0, True)
        suffix = key.replace("spk_", "")
        idx = int(suffix) if suffix.isdigit() else 1
        template = tr("notetaker.speaker_n", self.lang)
        name = template.format(n=idx) if "{n}" in template else f"Speaker {idx}"
        return (name, idx, False)

    def _apply_vocab(self, text: str) -> str:
        if self.vocabulary is not None and self.vocabulary.replacements:
            return apply_vocabulary_replacements(text, self.vocabulary.replacements).strip()
        return text.strip()

    def _transcribe_segments_safe(
        self,
        audio: np.ndarray,
        *,
        prompt: str,
        language: str | None,
    ) -> list[TranscriptSegment]:
        if language is not None:
            try:
                return self.whisper_engine.transcribe_segments(
                    audio, initial_prompt=prompt, language=language
                )
            except TypeError:
                pass
        return self.whisper_engine.transcribe_segments(audio, initial_prompt=prompt)

    def _process_segment(self, seg: AudioSegment) -> None:
        meeting = self._meeting
        if meeting is None:
            return

        prompt = self._build_effective_prompt()

        try:
            asr_segments = self._transcribe_segments_safe(
                seg.audio, prompt=prompt, language=self._locked_language
            )
            if self._locked_language is None:
                detected = getattr(self.whisper_engine, "last_detected_language", None)
                if isinstance(detected, tuple) and len(detected) == 2:
                    det_lang, det_prob = str(detected[0]).lower(), float(detected[1])
                    fallback_lang = "ru" if self.lang.lower().startswith("ru") else "en"
                    if det_lang not in {"ru", "en"} and det_prob < 0.85:
                        logger.info(
                            "Retrying meeting segment: unexpected language %s (p=%.2f), retrying with %s",
                            det_lang,
                            det_prob,
                            fallback_lang,
                        )
                        asr_segments = self._transcribe_segments_safe(
                            seg.audio, prompt=prompt, language=fallback_lang
                        )
                    elif det_prob >= 0.70 and sum(len(s.text.split()) for s in asr_segments) >= 3:
                        self._lang_votes[det_lang] = self._lang_votes.get(det_lang, 0) + 1
                        if self._lang_votes[det_lang] >= 3:
                            self._locked_language = det_lang
                            logger.info("Locked meeting ASR language to %s", det_lang)
        except Exception as exc:
            logger.warning("Meeting ASR failed on %s segment: %s", seg.channel, exc)
            return

        if not asr_segments:
            return

        # Group contiguous Whisper sub-segments into speaker turns (~4-10s max before splitting)
        groups = self._group_subsegments(asr_segments)
        use_diarization = (
            self.diarization_enabled
            and self.embedder.is_available
            and (seg.channel == "system" or meeting.audio_mode == "mic_only")
        )

        for grp_start, grp_end, grp_text in groups:
            clean_text = self._apply_vocab(grp_text)
            if not clean_text:
                continue

            abs_start = seg.start_sec + grp_start
            abs_end = seg.start_sec + grp_end

            if seg.channel == "mic" and meeting.audio_mode == "dual":
                if is_echo_bleed(clean_text, abs_start, abs_end, seg.rms, list(self._recent_system)):
                    logger.debug("Filtered echo bleed on mic: %r", clean_text)
                    continue
                speaker_key = "you"
                emb = None
            else:
                s0 = max(0, int(round(grp_start * 16000)))
                s1 = min(seg.audio.size, int(round(grp_end * 16000)))
                sub_audio = seg.audio[s0:s1]
                emb = self.embedder.embed(sub_audio) if use_diarization else None
                allow_new = sub_audio.size >= int(16000 * 2.5)
                if use_diarization:
                    speaker_key = self._tracker.assign(emb, allow_new_cluster=allow_new)
                else:
                    speaker_key = "spk_1"

            disp_name, color_idx, is_self = self._speaker_label_for_key(speaker_key)
            speaker = self.storage.ensure_speaker(
                meeting.id,
                speaker_key,
                display_name=disp_name,
                color_idx=color_idx,
                is_self=is_self,
            )
            utt = self.storage.add_utterance(
                meeting.id,
                speaker_key=speaker_key,
                channel=seg.channel,
                start_sec=abs_start,
                end_sec=abs_end,
                text=clean_text,
            )
            if speaker_key != "you":
                self._remote_utterance_ids.append(utt.id)
            if emb is not None:
                self._utterance_embeddings.append((utt.id, emb))
            if seg.channel == "system":
                self._recent_system.append((abs_start, abs_end, clean_text, seg.rms))
                self._last_remote_speaker_key = speaker_key
            elif meeting.audio_mode == "mic_only":
                self._last_remote_speaker_key = speaker_key
            if self.on_utterance is not None:
                self.on_utterance(utt, speaker)

        if self.on_interim_text is not None:
            cleared_spk = "you" if (seg.channel == "mic" and meeting.audio_mode == "dual") else self._last_remote_speaker_key
            self.on_interim_text(meeting.id, seg.channel, cleared_spk, "")

    def _build_effective_prompt(self) -> str:
        return (
            build_whisper_initial_prompt(self.whisper_prompt, self.vocabulary)
            if self.vocabulary is not None
            else self.whisper_prompt
        )

    def _process_preview(self, channel: str, audio: np.ndarray) -> None:
        meeting = self._meeting
        if meeting is None or self._stop_event.is_set() or self.on_interim_text is None:
            return
        if audio.size == 0:
            spk_key = "you" if (channel == "mic" and meeting.audio_mode == "dual") else self._last_remote_speaker_key
            self.on_interim_text(meeting.id, channel, spk_key, "")
            return
        prompt = self._build_effective_prompt()
        text = ""
        try:
            if hasattr(self.whisper_engine, "transcribe"):
                if self._locked_language is not None:
                    try:
                        text = self.whisper_engine.transcribe(
                            audio, initial_prompt=prompt, language=self._locked_language
                        )
                    except TypeError:
                        text = self.whisper_engine.transcribe(audio, initial_prompt=prompt)
                else:
                    text = self.whisper_engine.transcribe(audio, initial_prompt=prompt)
            else:
                segs = self._transcribe_segments_safe(
                    audio, prompt=prompt, language=self._locked_language
                )
                text = " ".join(s.text for s in segs if s.text)
        except Exception as exc:
            logger.debug("Meeting interim preview ASR failed: %s", exc)
            return
        clean = self._apply_vocab(text)
        if not clean:
            return
        spk_key = "you" if (channel == "mic" and meeting.audio_mode == "dual") else self._last_remote_speaker_key
        if self._stop_event.is_set():
            return
        self.on_interim_text(meeting.id, channel, spk_key, clean)

    @staticmethod
    def _group_subsegments(segments: list[TranscriptSegment], max_group_sec: float = 8.0) -> list[tuple[float, float, str]]:
        """Merge tiny Whisper sub-segments while preserving splits on long turns for speaker changes."""
        if not segments:
            return []
        groups: list[tuple[float, float, str]] = []
        cur_start = segments[0].start
        cur_end = segments[0].end
        cur_parts = [segments[0].text]

        for seg in segments[1:]:
            gap = seg.start - cur_end
            span = seg.end - cur_start
            if gap >= 0.45 or span >= max_group_sec:
                groups.append((cur_start, cur_end, " ".join(cur_parts)))
                cur_start = seg.start
                cur_end = seg.end
                cur_parts = [seg.text]
            else:
                cur_end = max(cur_end, seg.end)
                cur_parts.append(seg.text)
        groups.append((cur_start, cur_end, " ".join(cur_parts)))
        return groups

    def _process_loop(self) -> None:
        while True:
            item: AudioSegment | None = None
            preview: tuple[str, np.ndarray] | None = None
            with self._cond:
                while not self._segments_queue and self._pending_preview is None:
                    self._cond.wait()
                if self._segments_queue:
                    item = self._segments_queue.popleft()
                    if item is not None and self._pending_preview is not None and self._pending_preview[0] == item.channel:
                        self._pending_preview = None
                elif self._pending_preview is not None:
                    preview = self._pending_preview
                    self._pending_preview = None

            if item is None and preview is None:
                break
            if item is not None:
                try:
                    self._process_segment(item)
                except Exception:
                    logger.exception("Unhandled error processing meeting audio segment")
            elif preview is not None:
                try:
                    self._process_preview(preview[0], preview[1])
                except Exception:
                    logger.exception("Unhandled error processing interim preview")

    def stop(self) -> MeetingRecord | None:
        """Stop recording, drain transcription queue, refine speaker clusters, and finalize record."""
        meeting = self._meeting
        if meeting is None:
            return None

        duration = self.recorder.elapsed_sec
        self.recorder.stop()
        self._stop_event.set()
        with self._cond:
            self._pending_preview = None
            self._segments_queue.append(None)
            self._cond.notify_all()

        if self._worker is not None:
            self._worker.join(timeout=30.0)
            self._worker = None

        if len(self._utterance_embeddings) >= 2:
            remap = refine_speaker_clusters(self._utterance_embeddings)
            full_remap: dict[int, str] = dict(remap)
            last_key: str | None = None
            for uid in self._remote_utterance_ids:
                if uid in full_remap:
                    last_key = full_remap[uid]
                elif last_key is not None:
                    full_remap[uid] = last_key
            first_key = next((full_remap[uid] for uid in self._remote_utterance_ids if uid in full_remap), None)
            if first_key is not None:
                for uid in self._remote_utterance_ids:
                    if uid not in full_remap:
                        full_remap[uid] = first_key

            # Preserve any user-assigned speaker names across post-call cluster renumbering
            existing_speakers = {s.speaker_key: s for s in self.storage.get_speakers(meeting.id)}
            existing_utts = {u.id: u.speaker_key for u in self.storage.get_utterances(meeting.id)}
            custom_by_old = {
                s.speaker_key: s.display_name.strip()
                for s in existing_speakers.values()
                if (
                    not s.is_self
                    and s.display_name.strip()
                    and s.display_name.strip() != self._speaker_label_for_key(s.speaker_key)[0]
                )
            }
            custom_votes: dict[str, dict[str, int]] = {}
            for uid, new_key in full_remap.items():
                old_key = existing_utts.get(uid)
                if old_key and old_key in custom_by_old:
                    cname = custom_by_old[old_key]
                    bucket = custom_votes.setdefault(new_key, {})
                    bucket[cname] = bucket.get(cname, 0) + 1

            speaker_names: dict[str, str] = {}
            for key in set(full_remap.values()):
                if key in custom_votes:
                    best_name = max(custom_votes[key].items(), key=lambda kv: kv[1])[0]
                    speaker_names[key] = best_name
                else:
                    speaker_names[key] = self._speaker_label_for_key(key)[0]

            self.storage.apply_speaker_remap(meeting.id, full_remap, speaker_names)
            if self.on_speakers_updated is not None:
                self.on_speakers_updated(meeting.id)

        ended_at = datetime.now(timezone.utc).isoformat()
        self.storage.update_meeting(
            meeting.id,
            ended_at=ended_at,
            duration_sec=duration,
            status="ready",
        )
        updated = self.storage.get_meeting(meeting.id)
        self._meeting = None
        return updated
