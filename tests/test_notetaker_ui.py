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

    from localflow.ui.theme import build_hub_stylesheet
    from localflow.ui.widgets import DropdownCombo

    assert page._meeting_list.objectName() == "historyList"
    assert isinstance(page._mode_combo, DropdownCombo)
    assert page._record_btn.objectName() == "notetakerStartBtn"
    assert "#7c3aed" in build_hub_stylesheet("dark")
    assert "#991b1b" in build_hub_stylesheet("dark")
    assert "#7c3aed" in build_hub_stylesheet("light")
    assert "#991b1b" in build_hub_stylesheet("light")

    page.set_recording_state(True, m.id)
    assert page._record_btn.objectName() == "notetakerStopBtn"
    page.set_recording_state(False, None)
    assert page._record_btn.objectName() == "notetakerStartBtn"

    # Toggle Action Item checkbox persists to SQLite and strikes through label
    cbs = ws.action_container.findChildren(QCheckBox)
    assert cbs
    cbs[0].setChecked(True)
    assert storage.get_action_items(m.id)[0].done is True
    from localflow.ui.meeting_window import _ActionItemRow

    rows = ws.action_container.findChildren(_ActionItemRow)
    assert rows and rows[0]._lbl.font().strikeOut() is True
    cbs[0].setChecked(False)
    assert storage.get_action_items(m.id)[0].done is False
    assert rows[0]._lbl.font().strikeOut() is False

    # Check again for persistence test consistency
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


def test_short_meeting_modal_choices_and_render(qtbot) -> None:
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QWidget

    from localflow.ui.meeting_window import ShortMeetingModal

    # Even with an off-center side anchor, modal positions horizontally centered near bottom of screen
    side_anchor = QWidget()
    side_anchor.setGeometry(10, 10, 300, 400)
    side_anchor.show()
    qtbot.addWidget(side_anchor)

    dlg_dark = ShortMeetingModal(lang="en", theme="dark", anchor_widget=side_anchor)
    qtbot.addWidget(dlg_dark)
    screen = QGuiApplication.primaryScreen()
    assert screen is not None
    avail = screen.availableGeometry()
    assert abs((dlg_dark.x() + dlg_dark.width() // 2) - (avail.x() + avail.width() // 2)) <= 2
    assert dlg_dark.y() > avail.y() + avail.height() // 2

    assert dlg_dark.choice == "keep"
    pix_dark = dlg_dark.grab()
    assert not pix_dark.isNull()
    dlg_dark._discard_btn.click()
    assert dlg_dark.choice == "discard"

    dlg_light = ShortMeetingModal(lang="ru", theme="light")
    qtbot.addWidget(dlg_light)
    pix_light = dlg_light.grab()
    assert not pix_light.isNull()
    dlg_light._keep_btn.click()
    assert dlg_light.choice == "keep"

    dlg_close = ShortMeetingModal(lang="en", theme="dark")
    qtbot.addWidget(dlg_close)
    dlg_close._close_btn.click()
    assert dlg_close.choice == "keep"


def test_frameless_window_helper_resets_resize_cursor_inside_child_and_on_leave(qtbot) -> None:
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QHoverEvent
    from PySide6.QtWidgets import QApplication, QFrame, QScrollArea, QVBoxLayout, QWidget

    from localflow.ui.widgets import FramelessWindowHelper

    win = QWidget()
    qtbot.addWidget(win)
    win.resize(400, 300)
    lay = QVBoxLayout(win)
    lay.setContentsMargins(0, 0, 0, 0)
    shell = QFrame(win)
    lay.addWidget(shell)
    scroll = QScrollArea(shell)
    scroll.setGeometry(20, 20, 360, 260)
    child = QFrame()
    child.setMouseTracking(True)
    scroll.setWidget(child)
    scroll.setWidgetResizable(True)

    helper = FramelessWindowHelper(win, shell, titlebar_height=40)
    win.show()
    assert win.testAttribute(Qt.WidgetAttribute.WA_Hover)

    # Hovering left edge sets horizontal resize cursor
    edge_pt = QPointF(2.0, 120.0)
    edge_global = QPointF(win.mapToGlobal(edge_pt.toPoint()))
    QApplication.sendEvent(
        win,
        QHoverEvent(QEvent.Type.HoverMove, edge_pt, edge_global, edge_pt),
    )
    assert win.cursor().shape() == Qt.CursorShape.SizeHorCursor

    # Moving back inside over child area (WA_Hover delivers HoverMove to win) resets cursor
    inner_pt = QPointF(80.0, 120.0)
    inner_global = QPointF(win.mapToGlobal(inner_pt.toPoint()))
    QApplication.sendEvent(
        win,
        QHoverEvent(QEvent.Type.HoverMove, inner_pt, inner_global, edge_pt),
    )
    assert win.cursor().shape() == Qt.CursorShape.ArrowCursor
    assert child.cursor().shape() == Qt.CursorShape.ArrowCursor

    # Moving onto edge again then leaving window resets cursor on HoverLeave
    QApplication.sendEvent(
        win,
        QHoverEvent(QEvent.Type.HoverMove, edge_pt, edge_global, inner_pt),
    )
    assert win.cursor().shape() == Qt.CursorShape.SizeHorCursor
    QApplication.sendEvent(win, QEvent(QEvent.Type.HoverLeave))
    assert win.cursor().shape() == Qt.CursorShape.ArrowCursor
    assert helper is not None


def test_live_transcript_scroll_has_no_empty_void_and_respects_manual_scroll_up(qtbot, tmp_path: Path) -> None:
    from PySide6.QtWidgets import QApplication

    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    m = storage.create_meeting(title="Scroll Test", audio_mode="dual", status="recording")
    spk = storage.ensure_speaker(m.id, "spk_1", display_name="Speaker 1", color_idx=1, is_self=False)
    for i in range(15):
        storage.add_utterance(
            m.id,
            speaker_key="spk_1",
            channel="system",
            start_sec=i * 3.0,
            end_sec=i * 3.0 + 2.5,
            text=f"Long enough message number {i} to wrap across multiple lines in the live transcript panel cleanly.",
        )

    win = MeetingSideWindow(storage, lang="en", theme="dark")
    qtbot.addWidget(win)
    win.open_for_meeting(m.id, recording=True)
    QApplication.processEvents()

    ws = win.workspace
    body = ws._tr_scroll_body
    comm = ws._committed_container
    sb = ws._tr_scroll.verticalScrollBar()

    # Scroll body height must match actual content height without a giant empty spacer at the bottom
    assert abs(body.height() - comm.height()) <= 30
    assert sb.maximum() > 0

    # User scrolls up manually to read earlier messages -> interim updates must NOT force scroll back down
    sb.setValue(0)
    QApplication.processEvents()
    assert ws._stick_to_bottom is False

    ws._handle_interim_ui(m.id, "system", "spk_1", "Live interim preview text")
    qtbot.wait(30)
    assert sb.value() == 0

    # User scrolls back to bottom -> sticky auto-scroll resumes
    sb.setValue(sb.maximum())
    QApplication.processEvents()
    assert ws._stick_to_bottom is True
    assert spk.speaker_key == "spk_1"


def test_speech_bubble_never_clips_multiline_text_on_resize(qtbot) -> None:
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QFontMetrics

    from localflow.ui.meeting_window import _SpeechBubble

    text = (
        "Точнее, мы пососали одной слабой команде и выиграли одну сильную, "
        "а потом ещё откалибровали людей по рейтингу и разбили всех участников на три группы."
    )
    bubble = _SpeechBubble(text, is_self=False, is_interim=False, theme="dark", max_width=320)
    qtbot.addWidget(bubble)
    bubble.show()

    fm = QFontMetrics(bubble._lbl.font())
    assert bubble._lbl.font().pixelSize() == 13

    for width in (320, 240):
        bubble.sync_max_width(width)
        sh = bubble.sizeHint()
        m = bubble.layout().contentsMargins()
        inner_w = max(80, min(width, sh.width()) - m.left() - m.right())
        needed_h = fm.boundingRect(QRect(0, 0, inner_w, 10000), Qt.TextFlag.TextWordWrap, text).height()
        assert sh.height() >= needed_h + m.top() + m.bottom()


def test_transcript_speaker_right_click_opens_menu_and_renames_across_utterances(qtbot, tmp_path: Path, monkeypatch) -> None:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtWidgets import QApplication, QDialog, QInputDialog, QLabel, QMenu

    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    m = storage.create_meeting(title="Rename Test", audio_mode="dual", status="ready")
    storage.ensure_speaker(m.id, "spk_1", display_name="Speaker 1", color_idx=1, is_self=False)
    storage.ensure_speaker(m.id, "you", display_name="You", color_idx=0, is_self=True)
    storage.add_utterance(m.id, speaker_key="spk_1", channel="system", start_sec=1.0, end_sec=3.0, text="First line")
    storage.add_utterance(m.id, speaker_key="you", channel="mic", start_sec=4.0, end_sec=6.0, text="My reply")
    storage.add_utterance(m.id, speaker_key="spk_1", channel="system", start_sec=7.0, end_sec=9.0, text="Second line")

    win = MeetingSideWindow(storage, lang="en", theme="dark")
    qtbot.addWidget(win)
    win.open_for_meeting(m.id, recording=False)
    QApplication.processEvents()

    ws = win.workspace
    spk_headers = [
        lbl for lbl in ws._committed_container.findChildren(QLabel)
        if lbl.property("speaker_key") == "spk_1"
    ]
    assert len(spk_headers) == 2
    assert spk_headers[0].contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu

    class _FakeMenu(QMenu):
        def exec(self, *args, **kwargs):  # type: ignore[override]
            return self.actions()[0]

    monkeypatch.setattr("localflow.ui.meeting_window.QMenu", _FakeMenu)
    monkeypatch.setattr(QDialog, "exec", lambda self_dlg: int(QDialog.DialogCode.Accepted))
    monkeypatch.setattr(QInputDialog, "textValue", lambda self_dlg: "Alex")

    spk_headers[0].customContextMenuRequested.emit(QPoint(5, 5))
    QApplication.processEvents()

    updated_speakers = {s.speaker_key: s.display_name for s in storage.get_speakers(m.id)}
    assert updated_speakers["spk_1"] == "Alex"
    updated_headers = [
        lbl for lbl in ws._committed_container.findChildren(QLabel)
        if lbl.property("speaker_key") == "spk_1"
    ]
    assert len(updated_headers) == 2
    assert all("Alex" in lbl.text() for lbl in updated_headers)


def test_summary_copy_button_spacing_and_adaptive_action_items(qtbot, tmp_path: Path) -> None:
    from PySide6.QtWidgets import QApplication

    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    m_empty = storage.create_meeting(title="No Tasks Call", audio_mode="dual", status="ready")
    storage.update_meeting(
        m_empty.id,
        summary_md="Короткое введение по встрече.\n### Ключевые моменты\n- **Пункт 1**: важная деталь.",
    )

    m_tasks = storage.create_meeting(title="Many Tasks Call", audio_mode="dual", status="ready")
    storage.update_meeting(m_tasks.id, summary_md="Краткий итог.")
    storage.replace_action_items(
        m_tasks.id,
        [
            ("Подготовить длинную задачу с проверкой всех крайних случаев и интеграции", "Вы"),
            ("Согласовать макеты с командой дизайна до пятницы", "Алексей"),
            ("Обновить документацию релиза и список изменений", "Марина"),
        ],
    )

    win = MeetingSideWindow(storage, lang="ru", theme="dark")
    qtbot.addWidget(win)
    win.open_for_meeting(m_empty.id, recording=False)
    QApplication.processEvents()

    ws = win.workspace
    ws._set_active_tab(2)
    QApplication.processEvents()

    # Empty action items stays compact so Summary gets the maximum space
    assert ws._action_scroll.height() <= 32
    assert ws._copy_summary_btn.isEnabled()

    # Heading block after first paragraph gets an explicit top margin spacer (>= 12)
    doc = ws._summary_view.document()
    heading_blocks = []
    blk = doc.firstBlock()
    while blk.isValid():
        if blk.blockFormat().headingLevel() > 0:
            heading_blocks.append(blk)
        blk = blk.next()
    assert heading_blocks and heading_blocks[0].blockFormat().topMargin() >= 12.0

    # Copy Summary copies both formatted Markdown (with blank line before heading) and HTML
    ws._copy_summary_btn.click()
    cb = QApplication.clipboard()
    assert "### Ключевые моменты" in cb.text()
    assert "\n\n### Ключевые моменты" in cb.text()
    assert cb.mimeData().hasHtml()
    assert "<h3" in cb.mimeData().html().lower()

    # Switching to meeting with 3 action items expands Action Items height adaptively
    win.open_for_meeting(m_tasks.id, recording=False)
    QApplication.processEvents()
    assert ws._action_scroll.height() >= 80


def test_notetaker_meeting_list_wraps_long_titles_without_horizontal_overflow(qtbot, tmp_path: Path) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QListWidget

    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    long_title_a = "Игровые механики и географические термины"
    long_title_b = "Игровые темы и личные планы участников"
    m1 = storage.create_meeting(title=long_title_a, audio_mode="dual", status="ready")
    m2 = storage.create_meeting(title=long_title_b, audio_mode="dual", status="ready")
    for idx in range(6):
        storage.create_meeting(title=f"Meeting {idx}", audio_mode="dual", status="ready")

    page = NotetakerPage(storage, lang="ru", theme="dark")
    qtbot.addWidget(page)
    page.resize(1020, 520)
    page.show()
    QApplication.processEvents()

    ml = page._meeting_list
    assert ml.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert ml.wordWrap() is True
    assert ml.textElideMode() == Qt.TextElideMode.ElideRight
    assert ml.resizeMode() == QListWidget.ResizeMode.Adjust
    assert ml.horizontalScrollBar().maximum() == 0

    found = {}
    for row in range(ml.count()):
        it = ml.item(row)
        assert it is not None
        mid = it.data(Qt.ItemDataRole.UserRole)
        found[mid] = it
        rect = ml.visualItemRect(it)
        assert rect.width() <= ml.viewport().width()

    assert found[m1.id].toolTip() == long_title_a
    assert found[m2.id].toolTip() == long_title_b

    # Editing meeting title updates both list text and tooltip
    updated_title = "Новое длинное название встречи для проверки тултипа и переноса"
    storage.update_meeting(m1.id, title=updated_title)
    page._on_workspace_meeting_updated(m1.id)
    assert found[m1.id].toolTip() == updated_title
    assert updated_title in found[m1.id].text()

