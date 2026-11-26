"""Wispr Flow-inspired Hub Notetaker page wrapping the shared MeetingNoteWorkspace."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from localflow.i18n import DEFAULT_LANGUAGE, LanguageCode, normalize_language, tr
from localflow.notetaker.storage import (
    MeetingSpeaker,
    MeetingStorage,
    MeetingUtterance,
    format_timestamp,
)
from localflow.ui.meeting_window import MeetingNoteWorkspace
from localflow.ui.theme import ResolvedTheme


class NotetakerPage(QWidget):
    """Hub Notetaker workspace: meeting list sidebar + shared Wispr-style MeetingNoteWorkspace."""

    start_meeting_requested = Signal(str)  # audio_mode: "dual" | "mic_only"
    stop_meeting_requested = Signal()
    resume_meeting_requested = Signal(int)
    regenerate_summary_requested = Signal(int, bool)  # meeting_id, force
    open_side_window_requested = Signal(int)
    meeting_updated = Signal(int)

    def __init__(
        self,
        storage: MeetingStorage,
        *,
        llm_factory: Callable[[], object | None] | None = None,
        lang: LanguageCode = DEFAULT_LANGUAGE,
        theme: ResolvedTheme = "dark",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._storage = storage
        self._lang: LanguageCode = normalize_language(lang)
        self._theme: ResolvedTheme = theme
        self._recording_meeting_id: int | None = None
        self._is_recording = False
        self._loading_ui = False
        self._level_provider: Callable[[], tuple[float, float, float]] | None = None

        self._sidebar_timer = QTimer(self)
        self._sidebar_timer.setInterval(500)
        self._sidebar_timer.timeout.connect(self._update_record_button_text)

        self._build_ui(llm_factory)
        self._connect_signals()
        self.retranslate(self._lang)
        self.refresh_meetings()

    @property
    def workspace(self) -> MeetingNoteWorkspace:
        return self._workspace

    # Convenience proxies for tests & direct inspection
    @property
    def _tabs(self):
        return self._workspace._tabs

    @property
    def _title_edit(self):
        return self._workspace.title_edit

    @property
    def _summary_view(self):
        return self._workspace.summary_view

    @property
    def _action_container(self):
        return self._workspace.action_container

    @property
    def _thoughts_edit(self):
        return self._workspace.thoughts_edit

    def set_llm_factory(self, factory: Callable[[], object | None] | None) -> None:
        self._workspace.set_llm_factory(factory)

    def set_level_provider(self, provider: Callable[[], tuple[float, float, float]] | None) -> None:
        self._level_provider = provider
        self._workspace.set_level_provider(provider)

    def set_theme(self, theme: ResolvedTheme) -> None:
        self._theme = theme
        self._workspace.set_theme(theme)

    def retranslate(self, lang: str) -> None:
        self._lang = normalize_language(lang)
        self._update_record_button_text()
        self._mode_combo.setItemText(0, tr("notetaker.mode.dual", self._lang))
        self._mode_combo.setItemText(1, tr("notetaker.mode.mic_only", self._lang))
        self._popout_btn.setText(f"↗ {tr('notetaker.btn.popout', self._lang)}")
        self._export_btn.setText(tr("notetaker.btn.export", self._lang))
        self._delete_btn.setText(tr("notetaker.btn.delete", self._lang))
        self._search_edit.setPlaceholderText(tr("notetaker.search_placeholder", self._lang))
        self._workspace.retranslate(self._lang)

    def _build_ui(self, llm_factory: Callable[[], object | None] | None) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        root.addWidget(splitter)

        # Left sidebar
        left = QFrame()
        left.setObjectName("NotetakerSidebar")
        left.setMinimumWidth(240)
        left.setMaximumWidth(300)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(14, 14, 14, 14)
        left_lay.setSpacing(9)

        self._record_btn = QPushButton()
        self._record_btn.setMinimumHeight(36)
        self._record_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        left_lay.addWidget(self._record_btn)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Call (Mic + System Audio)", "dual")
        self._mode_combo.addItem("Offline (Mic Only)", "mic_only")
        left_lay.addWidget(self._mode_combo)

        util_row = QHBoxLayout()
        util_row.setSpacing(6)
        self._popout_btn = QPushButton("↗ Side Window")
        self._popout_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_btn = QPushButton("Export")
        self._export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        util_row.addWidget(self._popout_btn, 1)
        util_row.addWidget(self._export_btn)
        util_row.addWidget(self._delete_btn)
        left_lay.addLayout(util_row)

        from PySide6.QtWidgets import QLineEdit

        self._search_edit = QLineEdit()
        self._search_edit.setClearButtonEnabled(True)
        left_lay.addWidget(self._search_edit)

        self._meeting_list = QListWidget()
        self._meeting_list.setObjectName("HistoryList")
        self._meeting_list.setSpacing(4)
        left_lay.addWidget(self._meeting_list, 1)

        splitter.addWidget(left)

        # Right shared Wispr Flow workspace
        self._workspace = MeetingNoteWorkspace(
            self._storage,
            llm_factory=llm_factory,
            lang=self._lang,
            theme=self._theme,
            show_bottom_start_when_idle=False,
        )
        splitter.addWidget(self._workspace)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _connect_signals(self) -> None:
        self._record_btn.clicked.connect(self._on_record_clicked)
        self._popout_btn.clicked.connect(self._on_popout_clicked)
        self._export_btn.clicked.connect(self._on_export_clicked)
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        self._search_edit.textChanged.connect(lambda _t: self.refresh_meetings())
        self._meeting_list.currentItemChanged.connect(self._on_meeting_selected)

        self._workspace.start_meeting_requested.connect(self.start_meeting_requested.emit)
        self._workspace.stop_meeting_requested.connect(self.stop_meeting_requested.emit)
        self._workspace.resume_meeting_requested.connect(self.resume_meeting_requested.emit)
        self._workspace.regenerate_summary_requested.connect(self.regenerate_summary_requested.emit)
        self._workspace.meeting_updated.connect(self._on_workspace_meeting_updated)
        self._workspace.delete_meeting_requested.connect(lambda _mid: self.refresh_meetings())

    def set_recording_state(self, active: bool, meeting_id: int | None = None) -> None:
        self._is_recording = active
        self._recording_meeting_id = meeting_id if active else None
        self._mode_combo.setEnabled(not active)
        self._update_record_button_text()
        if active:
            self._sidebar_timer.start()
        else:
            self._sidebar_timer.stop()
        self._workspace.set_recording_state(active, meeting_id)
        self.refresh_meetings(select_id=(meeting_id if active else self._workspace.current_meeting_id))

    def notify_utterance(self, meeting_id: int, utt: MeetingUtterance, speaker: MeetingSpeaker) -> None:
        self._workspace.notify_utterance(meeting_id, utt, speaker)

    def notify_interim_text(self, meeting_id: int, channel: str, speaker_key: str, text: str) -> None:
        self._workspace.notify_interim_text(meeting_id, channel, speaker_key, text)

    def notify_speakers_updated(self, meeting_id: int) -> None:
        self._workspace.notify_speakers_updated(meeting_id)

    def notify_summary_step(self, meeting_id: int, stage: str, step: int, total: int) -> None:
        self._workspace.set_summary_step(meeting_id, stage, step, total)

    def notify_summary_done(self, meeting_id: int, *, too_short: bool = False, error: str = "") -> None:
        self._workspace.set_summary_done(meeting_id, too_short=too_short, error=error)
        self.refresh_meetings(select_id=self._workspace.current_meeting_id)

    def refresh_meetings(self, *, select_id: int | None = None) -> None:
        query = self._search_edit.text().strip()
        meetings = self._storage.list_meetings(search=query, limit=120)
        target_id = select_id if select_id is not None else self._workspace.current_meeting_id

        self._loading_ui = True
        chosen_id: int | None = None
        try:
            self._meeting_list.clear()
            selected_row = -1
            for idx, m in enumerate(meetings):
                date_str = m.created_at[:16].replace("T", " ")
                dur_str = format_timestamp(m.duration_sec) if m.duration_sec > 0 else "live"
                item = QListWidgetItem(f"{m.title}\n{date_str} · {dur_str}")
                item.setData(Qt.ItemDataRole.UserRole, m.id)
                self._meeting_list.addItem(item)
                if m.id == target_id:
                    selected_row = idx
                    chosen_id = m.id
            if selected_row >= 0:
                self._meeting_list.setCurrentRow(selected_row)
            elif meetings:
                self._meeting_list.setCurrentRow(0)
                chosen_id = meetings[0].id
        finally:
            self._loading_ui = False

        self._workspace.set_meeting(chosen_id)
        self._update_util_buttons()

    def _update_record_button_text(self) -> None:
        if self._is_recording:
            elapsed = self._level_provider()[2] if self._level_provider is not None else 0.0
            self._record_btn.setText(f"■ {tr('notetaker.btn.stop', self._lang)} ({format_timestamp(elapsed)})")
            self._record_btn.setStyleSheet(
                "background-color: #dc2626; color: #ffffff; font-weight: 600; border-radius: 8px; padding: 6px 12px;"
            )
        else:
            self._record_btn.setText(f"● {tr('notetaker.btn.start', self._lang)}")
            self._record_btn.setStyleSheet("")

    def _update_util_buttons(self) -> None:
        mid = self._workspace.current_meeting_id
        has_m = mid is not None
        self._popout_btn.setEnabled(has_m)
        self._export_btn.setEnabled(has_m)
        self._delete_btn.setEnabled(has_m and not (self._is_recording and mid == self._recording_meeting_id))

    def _on_record_clicked(self) -> None:
        if self._is_recording:
            self.stop_meeting_requested.emit()
        else:
            mode = str(self._mode_combo.currentData() or "dual")
            self.start_meeting_requested.emit(mode)

    def _on_meeting_selected(self, current: QListWidgetItem | None, _prev: QListWidgetItem | None) -> None:
        if self._loading_ui:
            return
        mid = int(current.data(Qt.ItemDataRole.UserRole)) if current is not None else None
        self._workspace.set_meeting(mid)
        self._update_util_buttons()

    def _on_workspace_meeting_updated(self, mid: int) -> None:
        self._refresh_list_labels_only()
        self.meeting_updated.emit(mid)

    def _refresh_list_labels_only(self) -> None:
        meetings = {m.id: m for m in self._storage.list_meetings(limit=120)}
        for row in range(self._meeting_list.count()):
            item = self._meeting_list.item(row)
            if item is None:
                continue
            mid = item.data(Qt.ItemDataRole.UserRole)
            m = meetings.get(mid)
            if m is not None:
                date_str = m.created_at[:16].replace("T", " ")
                dur_str = format_timestamp(m.duration_sec) if m.duration_sec > 0 else "live"
                item.setText(f"{m.title}\n{date_str} · {dur_str}")

    def _on_popout_clicked(self) -> None:
        if self._workspace.current_meeting_id is not None:
            self.open_side_window_requested.emit(self._workspace.current_meeting_id)

    def _on_export_clicked(self) -> None:
        mid = self._workspace.current_meeting_id
        if mid is None:
            return
        self._workspace.flush_pending_saves()
        md = self._storage.export_markdown(mid)
        if not md:
            return
        QApplication.clipboard().setText(md)
        meeting = self._storage.get_meeting(mid)
        safe_name = (meeting.title if meeting else "meeting").replace("/", "-").replace("\\", "-")[:48]
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("notetaker.btn.export", self._lang),
            f"{safe_name}.md",
            "Markdown (*.md)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(md)

    def _on_delete_clicked(self) -> None:
        mid = self._workspace.current_meeting_id
        if mid is None:
            return
        self._storage.delete_meeting(mid, hard=False)
        self.refresh_meetings()
        self.meeting_updated.emit(mid)
