from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QLabel

from localflow.notetaker.storage import MeetingStorage
from localflow.ui.meeting_window import MeetingSideWindow
from localflow.ui.notetaker_page import NotetakerPage


def test_notetaker_page_tabs_actions_theme_and_i18n(qtbot, tmp_path: Path) -> None:
    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    m = storage.create_meeting(title="Product Sync")
    storage.ensure_speaker(m.id, "you", display_name="You", color_idx=0, is_self=True)
    storage.ensure_speaker(m.id, "spk_1", display_name="Speaker 1", color_idx=1, is_self=False)
    storage.add_utterance(m.id, speaker_key="you", channel="mic", start_sec=1.0, end_sec=3.0, text="Hello team")
    storage.add_utterance(m.id, speaker_key="spk_1", channel="system", start_sec=4.0, end_sec=7.0, text="Hi everyone")
    storage.update_meeting(m.id, summary_md="- Synced on roadmap.", my_thoughts="Follow up next week")
    storage.replace_action_items(m.id, [("Send roadmap", "You")])

    page = NotetakerPage(storage, lang="en", theme="dark")
    qtbot.addWidget(page)
    ws = page.workspace

    # 3 Wispr-order tabs: 0=My thoughts, 1=Transcript, 2=Summary
    assert ws._tabs.count() == 3
    assert ws.title_edit.text() == "Product Sync"
    assert "roadmap" in ws.summary_view.toPlainText()

    # Toggle Action Item checkbox persists to SQLite
    cbs = ws.action_container.findChildren(QCheckBox)
    assert cbs
    cbs[0].setChecked(True)
    assert storage.get_action_items(m.id)[0].done is True

    # Transcript renders clean editorial paragraphs when stopped
    ws._set_active_tab(1)
    labels_txt = " ".join(lbl.text() for lbl in ws._committed_container.findChildren(QLabel))
    assert "Hello team" in labels_txt
    assert "Hi everyone" in labels_txt

    # My thoughts tab
    ws._set_active_tab(0)
    assert ws.thoughts_edit.toPlainText() == "Follow up next week"

    # Retranslate to Russian & switch theme without stylesheet errors
    page.retranslate("ru")
    page.set_theme("light")
    assert ws._tab_transcript_btn.text() == "Транскрипт"
    assert ws._tab_thoughts_btn.text() == "Мои заметки"


def test_meeting_side_window_live_bubbles_progress_and_close(qtbot, tmp_path: Path) -> None:
    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    m = storage.create_meeting(title="New note", audio_mode="dual", status="recording")
    spk_you = storage.ensure_speaker(m.id, "you", display_name="You", color_idx=0, is_self=True)
    utt = storage.add_utterance(m.id, speaker_key="you", channel="mic", start_sec=0.5, end_sec=2.0, text="Committed bubble text")

    class _FakeLLM:
        def generate_text(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
            return "They discussed the rollout timeline."

    win = MeetingSideWindow(storage, llm_factory=lambda: _FakeLLM(), lang="en", theme="dark")
    qtbot.addWidget(win)
    win.open_for_meeting(m.id, recording=True)
    assert win.isVisible()

    ws = win.workspace
    assert ws._tabs.currentIndex() == 1
    assert ws._live_wave.isVisible()
    assert ws._qa_missed_btn.isVisible()

    # Trigger long multi-line live & interim bubbles + 4+ speakers (+N chip cap)
    storage.ensure_speaker(m.id, "spk_1", display_name="Speaker 1", color_idx=1)
    storage.ensure_speaker(m.id, "spk_2", display_name="Speaker 2", color_idx=2)
    storage.ensure_speaker(m.id, "spk_3", display_name="Speaker 3", color_idx=3)
    ws._handle_speakers_updated_ui(m.id)
    assert any(btn.text().startswith("+") for btn in ws._speakers_bar.findChildren(type(ws._regen_btn)))

    long_text = " ".join(["Очень длинная реплика для проверки переноса и отсутствия обрезания строк."] * 5)
    ws._handle_interim_ui(m.id, "mic", "you", long_text)
    interim_lbls = ws._interim_container.findChildren(QLabel)
    assert any("Очень длинная реплика" in lbl.text() for lbl in interim_lbls)

    # Clear interim and add another utterance incrementally
    utt2 = storage.add_utterance(m.id, speaker_key="you", channel="mic", start_sec=2.5, end_sec=4.0, text="Second line")
    ws._handle_interim_ui(m.id, "mic", "you", "")
    ws._handle_utterance_ui(m.id, utt2, spk_you)
    committed_txt = " ".join(lbl.text() for lbl in ws._committed_container.findChildren(QLabel))
    assert "Committed bubble text" in committed_txt
    assert "Second line" in committed_txt

    # Wrapped action items render cleanly inside container
    storage.replace_action_items(m.id, [("Очень длинная задача для проверки корректного переноса на узком окне", "Вы")])
    ws._render_action_items()
    act_lbls = [lbl.text() for lbl in ws.action_container.findChildren(QLabel)]
    assert any("Очень длинная задача" in t for t in act_lbls)

    # Stop recording and paint animated SummaryProgressCard & light theme cleanly
    ws.set_recording_state(False, None)
    ws._set_active_tab(2)
    ws.set_summary_step(m.id, "analyze", 1, 2)
    assert ws._progress_card.isVisible()
    pix = win.grab()
    assert not pix.isNull()

    # Too-short banner after stop
    ws.set_summary_done(m.id, too_short=True)
    assert ws._too_short_card.isVisible()

    win.set_theme("light")
    pix_light = win.grab()
    assert not pix_light.isNull()

    # Close hides rather than destroying
    win.close()
    assert not win.isVisible()


def test_overlay_pill_meeting_mode_paints_cleanly(qtbot) -> None:
    from localflow.ui.overlay_pill import FloatingPillWidget, PillMode

    pill = FloatingPillWidget()
    qtbot.addWidget(pill)
    pill.show_meeting(elapsed_sec=125.0)
    assert pill._mode == PillMode.MEETING
    pix = pill.grab()
    assert not pix.isNull()

    pill.show_warming("Готовлю микрофон… ~3 с", queued=True)
    assert pill._mode == PillMode.WARMING
    pix_w = pill.grab()
    assert not pix_w.isNull()
