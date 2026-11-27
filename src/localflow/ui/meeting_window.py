"""Wispr Flow-inspired Notetaker workspace & right-docked companion side window."""

from __future__ import annotations

from datetime import date
import html
import math
import re
import threading
from typing import Callable

from PySide6.QtCore import QEvent, QMimeData, QPoint, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPainterPath, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from localflow.i18n import DEFAULT_LANGUAGE, LanguageCode, normalize_language, tr
from localflow.llm.downloader import DownloadProgress, download_diarization_model
from localflow.llm.model_catalog import find_diarization_model_path
from localflow.notetaker.qa_engine import MeetingQAEngine, QAResult
from localflow.notetaker.summarizer import format_summary_markdown
from localflow.notetaker.storage import (
    MeetingActionItem,
    MeetingSpeaker,
    MeetingStorage,
    MeetingUtterance,
    format_timestamp,
)
from localflow.ui.icons import render_icon
from localflow.ui.theme import (
    ResolvedTheme,
    apply_windows_titlebar_theme,
    build_dialog_stylesheet,
    build_menu_stylesheet,
    get_theme_tokens,
)
from localflow.ui.widgets import CheckmarkProxyStyle, FramelessWindowHelper

_SPEAKER_COLORS_DARK = (
    "#fafafa",  # you (clean crisp label in editorial view; plum bubble during live)
    "#fb923c",  # spk_1 (warm Wispr amber/orange)
    "#60a5fa",  # spk_2 (sky blue)
    "#f472b6",  # spk_3 (rose)
    "#a78bfa",  # spk_4 (violet)
    "#34d399",  # spk_5 (emerald)
)
_SPEAKER_COLORS_LIGHT = (
    "#18181b",
    "#c2410c",
    "#1d4ed8",
    "#be185d",
    "#6d28d9",
    "#047857",
)


class LiveWaveIcon(QWidget):
    """Compact 3-bar emerald animated audio wave glyph shown on the live Transcript tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(16, 14)
        self._phase = 0.0
        self._theme: ResolvedTheme = "dark"
        self._active = False
        self._timer = QTimer(self)
        self._timer.setInterval(90)
        self._timer.timeout.connect(self._tick)
        self.setVisible(False)

    def set_theme(self, theme: ResolvedTheme) -> None:
        self._theme = theme
        self.update()

    def set_active(self, active: bool) -> None:
        self._active = active
        self.setVisible(active)
        if active:
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self._phase += 0.45
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if not self._active:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            color = QColor("#059669") if self._theme == "light" else QColor("#34d399")
            painter.setBrush(color)
            offsets = (0.0, 1.3, 2.5)
            for idx, off in enumerate(offsets):
                h = 4.0 + 7.5 * (0.5 + 0.5 * math.sin(self._phase + off))
                x = 2.0 + idx * 4.5
                y = (self.height() - h) / 2.0
                painter.drawRoundedRect(QRectF(x, y, 2.5, h), 1.25, 1.25)
        finally:
            painter.end()


class _SkeletonBarsWidget(QWidget):
    """Animated shimmering horizontal skeleton bars (matching Wispr Flow summary loader)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(136, 48)
        self._phase = 0.0
        self._theme: ResolvedTheme = "dark"
        self._timer = QTimer(self)
        self._timer.setInterval(55)
        self._timer.timeout.connect(self._tick)

    def set_theme(self, theme: ResolvedTheme) -> None:
        self._theme = theme
        self.update()

    def set_animating(self, running: bool) -> None:
        if running:
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()

    def _tick(self) -> None:
        self._phase += 0.16
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            rows = (
                ((0.0, 44.0), (50.0, 56.0)),
                ((0.0, 74.0), (80.0, 44.0)),
                ((0.0, 32.0), (38.0, 46.0), (90.0, 38.0)),
                ((0.0, 62.0), (68.0, 34.0)),
            )
            base_rgb = (24, 24, 27) if self._theme == "light" else (244, 244, 246)
            for r_idx, segments in enumerate(rows):
                y = 4.0 + r_idx * 11.5
                for s_idx, (x, w) in enumerate(segments):
                    wave = 0.5 + 0.5 * math.sin(self._phase - r_idx * 0.65 - s_idx * 0.45)
                    alpha = int(round(55 + 175 * wave))
                    painter.setBrush(QColor(base_rgb[0], base_rgb[1], base_rgb[2], alpha))
                    painter.drawRoundedRect(QRectF(x, y, w, 6.0), 3.0, 3.0)
        finally:
            painter.end()


class SummaryProgressCard(QFrame):
    """Multi-step Wispr-style summary progress banner with shimmering skeleton lines."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SummaryProgressCard")
        self.setFixedHeight(92)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(14)

        left = QVBoxLayout()
        left.setSpacing(4)
        self._title_lbl = QLabel("Generating summary...")
        self._title_lbl.setObjectName("SummaryProgressTitle")
        self._step_lbl = QLabel("Step 1 of 2")
        self._step_lbl.setObjectName("FieldHint")
        left.addStretch(1)
        left.addWidget(self._title_lbl)
        left.addWidget(self._step_lbl)
        left.addStretch(1)

        self._skeleton = _SkeletonBarsWidget()
        lay.addLayout(left, 1)
        lay.addWidget(self._skeleton, 0, Qt.AlignmentFlag.AlignVCenter)
        self.setVisible(False)

    def set_theme(self, theme: ResolvedTheme) -> None:
        self._skeleton.set_theme(theme)

    def set_progress(self, title: str, step_text: str) -> None:
        self._title_lbl.setText(title)
        self._step_lbl.setText(step_text)

    def set_active(self, active: bool) -> None:
        self.setVisible(active)
        self._skeleton.set_animating(active)


def _bubble_colors(is_self: bool, is_interim: bool, theme: ResolvedTheme) -> tuple[str, str]:
    t = get_theme_tokens(theme)
    if is_self:
        return (
            t["bubble_self_interim_bg"] if is_interim else t["bubble_self_bg"],
            t["bubble_self_interim_fg"] if is_interim else t["bubble_self_fg"],
        )
    return (
        t["bubble_other_interim_bg"] if is_interim else t["bubble_other_bg"],
        t["bubble_other_interim_fg"] if is_interim else t["bubble_other_fg"],
    )


class _SpeechBubble(QFrame):
    """Rounded chat bubble container whose inner QLabel wraps cleanly without vertical clipping."""

    def __init__(
        self,
        text: str,
        *,
        is_self: bool,
        is_interim: bool,
        theme: ResolvedTheme,
        max_width: int = 410,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        bg, fg = _bubble_colors(is_self=is_self, is_interim=is_interim, theme=theme)
        self.setObjectName("WisprSpeechBubble")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)
        self.setMaximumWidth(max(220, max_width))
        self.setStyleSheet(
            f"QFrame#WisprSpeechBubble {{ background-color: {bg}; border-radius: 12px; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(13, 9, 13, 10)
        lay.setSpacing(0)

        self._lbl = QLabel(text)
        lbl_font = QFont(self._lbl.font())
        lbl_font.setPixelSize(13)
        self._lbl.setFont(lbl_font)
        self._lbl.setWordWrap(True)
        self._lbl.setMinimumWidth(0)
        self._lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._lbl.setStyleSheet(
            f"QLabel {{ background: transparent; color: {fg}; font-size: 13px; border: none; }}"
        )
        lay.addWidget(self._lbl)

    def sync_max_width(self, max_width: int) -> None:
        self.setMaximumWidth(max(220, max_width))
        self.updateGeometry()

    def hasHeightForWidth(self) -> bool:  # type: ignore[override]
        return True

    def heightForWidth(self, width: int) -> int:  # type: ignore[override]
        m = self.layout().contentsMargins()
        eff_w = min(max(120, width), self.maximumWidth())
        inner_w = max(80, eff_w - m.left() - m.right())
        fm = QFontMetrics(self._lbl.font())
        fm_h = fm.boundingRect(
            QRect(0, 0, inner_w, 10000),
            Qt.TextFlag.TextWordWrap,
            self._lbl.text(),
        ).height()
        lbl_h = self._lbl.heightForWidth(inner_w)
        return max(lbl_h, fm_h) + m.top() + m.bottom()

    def sizeHint(self) -> QSize:  # type: ignore[override]
        m = self.layout().contentsMargins()
        fm = QFontMetrics(self._lbl.font())
        text_w = fm.horizontalAdvance(self._lbl.text())
        w = min(self.maximumWidth(), max(100, text_w + m.left() + m.right() + 10))
        return QSize(w, self.heightForWidth(w))

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return self.sizeHint()


class _ActionItemRow(QWidget):
    """Wrapped Action Item row: checkbox + clickable word-wrapped label."""

    toggled = Signal(bool)

    def __init__(
        self,
        label: str,
        *,
        checked: bool,
        theme: ResolvedTheme = "dark",
        check_style: CheckmarkProxyStyle | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme: ResolvedTheme = theme
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 3, 0, 3)
        lay.setSpacing(8)

        self._cb = QCheckBox("")
        if check_style is not None:
            self._cb.setStyle(check_style)
        self._cb.setChecked(checked)
        self._cb.setFixedSize(20, 20)
        self._cb.toggled.connect(self._on_cb_toggled)

        self._lbl = QLabel(label)
        self._lbl.setWordWrap(True)
        self._lbl.setMinimumWidth(0)
        self._lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sync_label_style(checked)

        lay.addWidget(self._cb, 0, Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self._lbl, 1)

    def set_theme(self, theme: ResolvedTheme) -> None:
        self._theme = theme
        self._sync_label_style(self._cb.isChecked())

    def _on_cb_toggled(self, checked: bool) -> None:
        self._sync_label_style(checked)
        self.toggled.emit(checked)

    def _sync_label_style(self, checked: bool) -> None:
        t = get_theme_tokens(self._theme)
        font = self._lbl.font()
        font.setStrikeOut(checked)
        self._lbl.setFont(font)
        col = t["text_muted"] if checked else t["text"]
        self._lbl.setStyleSheet(
            f"QLabel {{ font-size: 13px; color: {col}; background: transparent; border: none; }}"
        )

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._cb.toggle()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class MeetingNoteWorkspace(QWidget):
    """Shared Wispr Flow-style single-meeting note workspace used in Hub & Side Window."""

    start_meeting_requested = Signal(str)  # audio_mode
    stop_meeting_requested = Signal()
    resume_meeting_requested = Signal(int)  # meeting_id
    regenerate_summary_requested = Signal(int, bool)  # meeting_id, force
    meeting_updated = Signal(int)  # title/thoughts/speakers/actions changed
    delete_meeting_requested = Signal(int)

    _utterance_signal = Signal(int, object, object)  # meeting_id, utterance, speaker
    _interim_signal = Signal(int, str, str, str)  # meeting_id, channel, speaker_key, text
    _speakers_updated_signal = Signal(int)
    _qa_done_signal = Signal(object, str)  # QAResult | None, error_msg
    _diar_progress_signal = Signal(int, str)
    _diar_done_signal = Signal(bool, str)

    def __init__(
        self,
        storage: MeetingStorage,
        *,
        llm_factory: Callable[[], object | None] | None = None,
        lang: LanguageCode = DEFAULT_LANGUAGE,
        theme: ResolvedTheme = "dark",
        show_bottom_start_when_idle: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._storage = storage
        self._llm_factory = llm_factory
        self._lang: LanguageCode = normalize_language(lang)
        self._theme: ResolvedTheme = theme
        self._show_bottom_start = show_bottom_start_when_idle

        self._current_meeting_id: int | None = None
        self._recording_meeting_id: int | None = None
        self._is_recording = False
        self._summarizing_meeting_id: int | None = None
        self._too_short_meeting_id: int | None = None
        self._loading_ui = False
        self._live_banner_dismissed = False
        self._qa_scope: str = "current"
        self._diar_cancel: threading.Event | None = None
        self._level_provider: Callable[[], tuple[float, float, float]] | None = None
        self._check_style = CheckmarkProxyStyle(self._theme)

        # Incremental live-transcript rendering state
        self._rendered_live_meeting_id: int | None = None
        self._rendered_live_count: int = 0
        self._last_live_speaker_key: str | None = None
        self._interim_texts: dict[str, tuple[str, str]] = {}  # channel -> (speaker_key, text)

        self._thoughts_timer = QTimer(self)
        self._thoughts_timer.setSingleShot(True)
        self._thoughts_timer.setInterval(450)
        self._thoughts_timer.timeout.connect(self._save_thoughts_now)

        self._meter_timer = QTimer(self)
        self._meter_timer.setInterval(200)
        self._meter_timer.timeout.connect(self._poll_live_meters)

        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.setInterval(10)
        self._scroll_timer.timeout.connect(self._apply_scroll_to_bottom)
        self._stick_to_bottom = True
        self._programmatic_scroll = False

        self._copy_reset_timer = QTimer(self)
        self._copy_reset_timer.setSingleShot(True)
        self._copy_reset_timer.setInterval(1200)
        self._copy_reset_timer.timeout.connect(self._reset_copy_icon)

        self._copy_summary_reset_timer = QTimer(self)
        self._copy_summary_reset_timer.setSingleShot(True)
        self._copy_summary_reset_timer.setInterval(1200)
        self._copy_summary_reset_timer.timeout.connect(self._reset_copy_summary_icon)

        self._build_ui()
        self._connect_signals()
        self._apply_theme_styles()
        self.retranslate(self._lang)
        self._refresh_diarization_banner()

    @property
    def current_meeting_id(self) -> int | None:
        return self._current_meeting_id

    @property
    def thoughts_edit(self) -> QTextEdit:
        return self._thoughts_edit

    @property
    def title_edit(self) -> QLineEdit:
        return self._title_edit

    @property
    def summary_view(self) -> QTextEdit:
        return self._summary_view

    @property
    def action_container(self) -> QWidget:
        return self._action_container

    def set_llm_factory(self, factory: Callable[[], object | None] | None) -> None:
        self._llm_factory = factory

    def set_level_provider(self, provider: Callable[[], tuple[float, float, float]] | None) -> None:
        self._level_provider = provider

    def set_theme(self, theme: ResolvedTheme) -> None:
        self._theme = theme
        self._check_style.set_theme(theme)
        self._live_wave.set_theme(theme)
        self._progress_card.set_theme(theme)
        self._apply_theme_styles()
        self._refresh_workspace_icons()
        for row in self._action_container.findChildren(_ActionItemRow):
            row.set_theme(theme)
        self._render_current_meeting(force_transcript_rebuild=True)

    def retranslate(self, lang: str) -> None:
        self._lang = normalize_language(lang)
        self._title_edit.setPlaceholderText(tr("notetaker.title_placeholder", self._lang))
        self._tab_thoughts_btn.setText(tr("notetaker.tab.thoughts", self._lang))
        self._tab_transcript_btn.setText(tr("notetaker.tab.transcript", self._lang))
        self._tab_summary_btn.setText(tr("notetaker.tab.summary", self._lang))
        self._thoughts_edit.setPlaceholderText(tr("notetaker.thoughts_placeholder", self._lang))
        self._tr_search_edit.setPlaceholderText(tr("notetaker.transcript_search", self._lang))
        self._live_banner_lbl.setText(tr("notetaker.live_banner", self._lang))
        self._privacy_lbl.setText(tr("notetaker.privacy_footer", self._lang))
        self._too_short_lbl.setText(tr("notetaker.summary.too_short", self._lang))
        self._force_sum_btn.setText(tr("notetaker.summary.force", self._lang))
        self._copy_summary_btn.setText(tr("notetaker.btn.copy", self._lang))
        self._regen_btn.setText(tr("notetaker.btn.regenerate", self._lang))
        self._action_header.setText(tr("notetaker.action_items", self._lang))
        self._diar_label.setText(tr("notetaker.diar.banner", self._lang))
        self._diar_btn.setText(tr("notetaker.diar.download", self._lang))
        self._qa_input.setPlaceholderText(tr("notetaker.qa.placeholder", self._lang))
        self._qa_missed_btn.setText(tr("notetaker.qa.what_missed", self._lang))
        self._update_qa_scope_button()
        self._update_dock_controls()
        self._render_current_meeting(force_transcript_rebuild=True)

    def set_meeting(self, meeting_id: int | None, *, select_tab: int | None = None) -> None:
        if self._current_meeting_id != meeting_id:
            self._save_thoughts_now()
            self._interim_texts.clear()
            self._rendered_live_meeting_id = None
            self._rendered_live_count = 0
            self._qa_card.setVisible(False)
            self._stick_to_bottom = True
        self._current_meeting_id = meeting_id
        if select_tab is not None:
            self._set_active_tab(select_tab)
        self._render_current_meeting(force_transcript_rebuild=True)

    def set_recording_state(self, active: bool, meeting_id: int | None = None) -> None:
        was_recording = self._is_recording
        self._is_recording = active
        self._recording_meeting_id = meeting_id if active else None
        if active:
            self._live_banner_dismissed = False
            self._too_short_meeting_id = None
            self._stick_to_bottom = True
            self._interim_texts.clear()
            if meeting_id is not None and self._current_meeting_id != meeting_id:
                self._current_meeting_id = meeting_id
            if meeting_id is not None and meeting_id == self._current_meeting_id:
                self._set_active_tab(1)  # Live Transcript tab
            if not self._meter_timer.isActive():
                self._meter_timer.start()
        else:
            self._meter_timer.stop()
            self._mic_bar.setValue(0)
            self._sys_bar.setValue(0)
            self._interim_texts.clear()
            if was_recording:
                # Immediately prepare editorial post-meeting transition
                self._rendered_live_meeting_id = None
                self._rendered_live_count = 0

        self._live_wave.set_active(active and self._current_meeting_id == self._recording_meeting_id)
        self._update_dock_controls()
        self._render_current_meeting(force_transcript_rebuild=True)

    def set_summary_step(self, meeting_id: int, stage: str, step: int, total: int) -> None:
        self._summarizing_meeting_id = meeting_id
        self._too_short_meeting_id = None
        if self._current_meeting_id == meeting_id:
            key = {
                "refine": "notetaker.summary.step_refine",
                "analyze": "notetaker.summary.step_analyze",
                "write": "notetaker.summary.step_write",
            }.get(stage, "notetaker.summarizing")
            title = tr(key, self._lang)
            subtitle = tr("notetaker.summary.step_of", self._lang, step=step, total=total)
            self._progress_card.set_progress(title, subtitle)
            self._progress_card.set_active(True)
            self._too_short_card.setVisible(False)
            self._regen_btn.setEnabled(False)

    def set_summary_done(self, meeting_id: int, *, too_short: bool = False, error: str = "") -> None:
        if self._summarizing_meeting_id == meeting_id:
            self._summarizing_meeting_id = None
        self._too_short_meeting_id = meeting_id if too_short else None
        if self._current_meeting_id == meeting_id:
            self._progress_card.set_active(False)
            self._render_current_meeting(force_transcript_rebuild=True)
            if error:
                self._summary_view.setPlainText(f"{tr('notetaker.summary_error', self._lang)}: {error}")

    def notify_utterance(self, meeting_id: int, utt: MeetingUtterance, speaker: MeetingSpeaker) -> None:
        self._utterance_signal.emit(meeting_id, utt, speaker)

    def notify_interim_text(self, meeting_id: int, channel: str, speaker_key: str, text: str) -> None:
        self._interim_signal.emit(meeting_id, channel, speaker_key, text)

    def notify_speakers_updated(self, meeting_id: int) -> None:
        self._speakers_updated_signal.emit(meeting_id)

    def flush_pending_saves(self) -> None:
        self._save_thoughts_now()

    # ---- UI Construction ----

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 10, 20, 14)
        root.setSpacing(10)

        # Optional diarization download banner
        self._diar_banner = QFrame()
        self._diar_banner.setObjectName("WisprSubtleCard")
        diar_lay = QHBoxLayout(self._diar_banner)
        diar_lay.setContentsMargins(12, 8, 12, 8)
        diar_lay.setSpacing(10)
        self._diar_label = QLabel()
        self._diar_label.setWordWrap(True)
        self._diar_progress = QProgressBar()
        self._diar_progress.setRange(0, 100)
        self._diar_progress.setFixedWidth(100)
        self._diar_progress.setVisible(False)
        self._diar_btn = QPushButton()
        self._diar_btn.setObjectName("WisprPillBtn")
        self._diar_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        diar_lay.addWidget(self._diar_label, 1)
        diar_lay.addWidget(self._diar_progress)
        diar_lay.addWidget(self._diar_btn)
        root.addWidget(self._diar_banner)

        # Editorial serif title
        self._title_edit = QLineEdit()
        self._title_edit.setObjectName("WisprTitleEdit")
        self._title_edit.setMinimumWidth(0)
        root.addWidget(self._title_edit)

        # Row 1: Date chip + mode chip + live mini meters
        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.setSpacing(6)
        self._date_chip = QLabel("Today")
        self._date_chip.setObjectName("WisprMetaChip")
        self._mode_chip = QLabel("Mic + System")
        self._mode_chip.setObjectName("WisprMetaChip")
        meta_row.addWidget(self._date_chip)
        meta_row.addWidget(self._mode_chip)
        meta_row.addStretch(1)

        self._meters_widget = QWidget()
        m_lay = QHBoxLayout(self._meters_widget)
        m_lay.setContentsMargins(0, 0, 0, 0)
        m_lay.setSpacing(5)
        self._mic_bar = QProgressBar()
        self._mic_bar.setRange(0, 100)
        self._mic_bar.setTextVisible(False)
        self._mic_bar.setFixedSize(36, 5)
        self._sys_bar = QProgressBar()
        self._sys_bar.setRange(0, 100)
        self._sys_bar.setTextVisible(False)
        self._sys_bar.setFixedSize(36, 5)
        m_lay.addWidget(self._mic_bar)
        m_lay.addWidget(self._sys_bar)
        self._meters_widget.setVisible(False)
        meta_row.addWidget(self._meters_widget)
        root.addLayout(meta_row)

        # Row 2: Dedicated speaker chips row so chips never clip when side window narrows
        self._speakers_bar = QWidget()
        self._speakers_lay = QHBoxLayout(self._speakers_bar)
        self._speakers_lay.setContentsMargins(0, 0, 0, 2)
        self._speakers_lay.setSpacing(6)
        self._speakers_lay.addStretch(1)
        root.addWidget(self._speakers_bar)

        # Underline Tab Bar: My thoughts | [wave] Transcript | ✦ Summary
        tab_bar_wrap = QWidget()
        tab_bar_lay = QVBoxLayout(tab_bar_wrap)
        tab_bar_lay.setContentsMargins(0, 6, 0, 0)
        tab_bar_lay.setSpacing(0)

        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(0, 0, 0, 0)
        tab_row.setSpacing(18)

        self._tab_thoughts_btn = QPushButton("My thoughts")
        self._tab_thoughts_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        tr_btn_wrap = QWidget()
        tr_btn_lay = QHBoxLayout(tr_btn_wrap)
        tr_btn_lay.setContentsMargins(0, 0, 0, 0)
        tr_btn_lay.setSpacing(5)
        self._live_wave = LiveWaveIcon()
        self._tab_transcript_btn = QPushButton("Transcript")
        self._tab_transcript_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        tr_btn_lay.addWidget(self._live_wave, 0, Qt.AlignmentFlag.AlignVCenter)
        tr_btn_lay.addWidget(self._tab_transcript_btn)

        self._tab_summary_btn = QPushButton("Summary")
        self._tab_summary_btn.setIconSize(QSize(13, 13))
        self._tab_summary_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        tab_row.addWidget(self._tab_thoughts_btn)
        tab_row.addWidget(tr_btn_wrap)
        tab_row.addWidget(self._tab_summary_btn)
        tab_row.addStretch(1)
        tab_bar_lay.addLayout(tab_row)

        self._tab_divider = QFrame()
        self._tab_divider.setFrameShape(QFrame.Shape.HLine)
        self._tab_divider.setFixedHeight(1)
        tab_bar_lay.addWidget(self._tab_divider)
        root.addWidget(tab_bar_wrap)

        # 3-tab content stack
        self._tabs = QStackedWidget()

        # Page 0: My thoughts
        thoughts_page = QWidget()
        th_lay = QVBoxLayout(thoughts_page)
        th_lay.setContentsMargins(0, 12, 0, 0)
        self._thoughts_edit = QTextEdit()
        self._thoughts_edit.setObjectName("WisprPlainEditor")
        th_lay.addWidget(self._thoughts_edit, 1)
        self._tabs.addWidget(thoughts_page)

        # Page 1: Transcript
        tr_page = QWidget()
        tr_lay = QVBoxLayout(tr_page)
        tr_lay.setContentsMargins(0, 12, 0, 0)
        tr_lay.setSpacing(12)

        self._tr_card = QFrame()
        self._tr_card.setObjectName("WisprSubtleCard")
        self._tr_card.setMinimumWidth(0)
        tr_card_lay = QVBoxLayout(self._tr_card)
        tr_card_lay.setContentsMargins(14, 10, 14, 10)
        tr_card_lay.setSpacing(8)

        tr_top = QHBoxLayout()
        tr_top.setSpacing(6)
        self._tr_clock_lbl = QLabel()
        self._tr_clock_lbl.setFixedSize(14, 14)
        self._tr_meta_lbl = QLabel("0:00")
        self._tr_meta_lbl.setObjectName("FieldHint")
        self._tr_meta_lbl.setMinimumWidth(0)
        tr_top.addWidget(self._tr_clock_lbl)
        tr_top.addWidget(self._tr_meta_lbl)
        tr_top.addStretch(1)

        self._tr_search_edit = QLineEdit()
        self._tr_search_edit.setObjectName("WisprInlineSearch")
        self._tr_search_edit.setClearButtonEnabled(True)
        self._tr_search_edit.setMinimumWidth(0)
        self._tr_search_edit.setVisible(False)
        tr_top.addWidget(self._tr_search_edit, 2)

        self._tr_search_btn = QPushButton()
        self._tr_search_btn.setObjectName("WisprIconBtn")
        self._tr_search_btn.setFixedSize(26, 26)
        self._tr_search_btn.setIconSize(QSize(14, 14))
        self._tr_search_btn.setCheckable(True)
        self._tr_search_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self._tr_speakers_btn = QPushButton()
        self._tr_speakers_btn.setObjectName("WisprIconBtn")
        self._tr_speakers_btn.setFixedSize(26, 26)
        self._tr_speakers_btn.setIconSize(QSize(14, 14))
        self._tr_speakers_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self._tr_copy_btn = QPushButton()
        self._tr_copy_btn.setObjectName("WisprIconBtn")
        self._tr_copy_btn.setFixedSize(26, 26)
        self._tr_copy_btn.setIconSize(QSize(14, 14))
        self._tr_copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        tr_top.addWidget(self._tr_search_btn)
        tr_top.addWidget(self._tr_speakers_btn)
        tr_top.addWidget(self._tr_copy_btn)
        tr_card_lay.addLayout(tr_top)

        self._live_banner_row = QWidget()
        lb_lay = QHBoxLayout(self._live_banner_row)
        lb_lay.setContentsMargins(0, 4, 0, 0)
        lb_lay.setSpacing(8)
        self._live_banner_lbl = QLabel()
        self._live_banner_lbl.setObjectName("FieldHint")
        self._live_banner_lbl.setWordWrap(True)
        self._live_banner_close = QPushButton()
        self._live_banner_close.setObjectName("WisprIconBtn")
        self._live_banner_close.setFixedSize(22, 22)
        self._live_banner_close.setIconSize(QSize(11, 11))
        self._live_banner_close.setCursor(Qt.CursorShape.PointingHandCursor)
        lb_lay.addWidget(self._live_banner_lbl, 1)
        lb_lay.addWidget(self._live_banner_close)
        self._live_banner_row.setVisible(False)
        tr_card_lay.addWidget(self._live_banner_row)
        tr_lay.addWidget(self._tr_card)

        # Scrollable live bubbles / editorial transcript container
        self._tr_scroll = QScrollArea()
        self._tr_scroll.setWidgetResizable(True)
        self._tr_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._tr_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._tr_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self._tr_scroll_body = QWidget()
        self._tr_scroll_body.setStyleSheet("background: transparent;")
        self._tr_body_lay = QVBoxLayout(self._tr_scroll_body)
        self._tr_body_lay.setContentsMargins(2, 2, 6, 8)
        self._tr_body_lay.setSpacing(6)
        self._tr_body_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._committed_container = QWidget()
        self._committed_container.setStyleSheet("background: transparent;")
        self._committed_lay = QVBoxLayout(self._committed_container)
        self._committed_lay.setContentsMargins(0, 0, 0, 0)
        self._committed_lay.setSpacing(6)
        self._committed_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._interim_container = QWidget()
        self._interim_container.setStyleSheet("background: transparent;")
        self._interim_lay = QVBoxLayout(self._interim_container)
        self._interim_lay.setContentsMargins(0, 4, 0, 0)
        self._interim_lay.setSpacing(6)
        self._interim_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._tr_body_lay.addWidget(self._committed_container)
        self._tr_body_lay.addWidget(self._interim_container)
        self._tr_scroll.setWidget(self._tr_scroll_body)
        self._tr_scroll.viewport().installEventFilter(self)
        self._tr_scroll.verticalScrollBar().valueChanged.connect(self._on_transcript_scrolled)
        tr_lay.addWidget(self._tr_scroll, 1)
        self._tabs.addWidget(tr_page)

        # Page 2: Summary
        sum_page = QWidget()
        sum_lay = QVBoxLayout(sum_page)
        sum_lay.setContentsMargins(0, 12, 0, 0)
        sum_lay.setSpacing(10)

        self._progress_card = SummaryProgressCard()
        sum_lay.addWidget(self._progress_card)

        self._too_short_card = QFrame()
        self._too_short_card.setObjectName("WisprSubtleCard")
        ts_lay = QHBoxLayout(self._too_short_card)
        ts_lay.setContentsMargins(14, 12, 14, 12)
        ts_lay.setSpacing(10)
        self._too_short_lbl = QLabel("THE MEETING WAS TOO SHORT TO SUMMARIZE.")
        self._too_short_lbl.setObjectName("TooShortLabel")
        self._too_short_lbl.setWordWrap(True)
        self._force_sum_btn = QPushButton("Summarize anyway")
        self._force_sum_btn.setObjectName("WisprPillBtn")
        self._force_sum_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        ts_lay.addWidget(self._too_short_lbl, 1)
        ts_lay.addWidget(self._force_sum_btn)
        self._too_short_card.setVisible(False)
        sum_lay.addWidget(self._too_short_card)

        sum_hdr_row = QHBoxLayout()
        sum_hdr_row.setSpacing(8)
        sum_hdr_row.addStretch(1)
        self._copy_summary_btn = QPushButton("Copy")
        self._copy_summary_btn.setObjectName("WisprPillBtn")
        self._copy_summary_btn.setIconSize(QSize(13, 13))
        self._copy_summary_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        sum_hdr_row.addWidget(self._copy_summary_btn)
        self._regen_btn = QPushButton("Summarize")
        self._regen_btn.setObjectName("WisprPillBtn")
        self._regen_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        sum_hdr_row.addWidget(self._regen_btn)
        sum_lay.addLayout(sum_hdr_row)

        self._summary_view = QTextEdit()
        self._summary_view.setObjectName("WisprPlainEditor")
        self._summary_view.setReadOnly(True)
        sum_lay.addWidget(self._summary_view, 1)

        self._action_header = QLabel("Action Items")
        self._action_header.setObjectName("SectionHeader")
        sum_lay.addWidget(self._action_header)

        self._action_scroll = QScrollArea()
        self._action_scroll.setWidgetResizable(True)
        self._action_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._action_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._action_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self._action_container = QWidget()
        self._action_container.setStyleSheet("background: transparent;")
        self._action_lay = QVBoxLayout(self._action_container)
        self._action_lay.setContentsMargins(0, 2, 0, 2)
        self._action_lay.setSpacing(4)
        self._action_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._action_scroll.setWidget(self._action_container)
        self._action_scroll.setFixedHeight(28)
        sum_lay.addWidget(self._action_scroll, 0)
        self._tabs.addWidget(sum_page)

        root.addWidget(self._tabs, 1)

        # Collapsible Ask AI Answer Card
        self._qa_card = QFrame()
        self._qa_card.setObjectName("WisprSubtleCard")
        self._qa_card.setVisible(False)
        qa_card_lay = QVBoxLayout(self._qa_card)
        qa_card_lay.setContentsMargins(14, 10, 14, 10)
        qa_card_lay.setSpacing(6)
        qa_hdr = QHBoxLayout()
        self._qa_title_lbl = QLabel("✦ Ask AI")
        self._qa_title_lbl.setObjectName("SectionHeader")
        self._qa_close_btn = QPushButton("×")
        self._qa_close_btn.setObjectName("WisprIconBtn")
        self._qa_close_btn.setFixedSize(22, 22)
        self._qa_close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        qa_hdr.addWidget(self._qa_title_lbl)
        qa_hdr.addStretch(1)
        qa_hdr.addWidget(self._qa_close_btn)
        qa_card_lay.addLayout(qa_hdr)

        self._qa_answer = QTextEdit()
        self._qa_answer.setObjectName("WisprPlainEditor")
        self._qa_answer.setReadOnly(True)
        self._qa_answer.setMaximumHeight(150)
        qa_card_lay.addWidget(self._qa_answer)
        root.addWidget(self._qa_card)

        # Bottom Privacy Notice + Floating Dock
        self._privacy_lbl = QLabel("All transcription & AI stay 100% local on your PC.")
        self._privacy_lbl.setObjectName("PrivacyNotice")
        self._privacy_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._privacy_lbl.setVisible(False)
        root.addWidget(self._privacy_lbl)

        dock_row = QHBoxLayout()
        dock_row.setSpacing(10)

        self._dock_state_btn = QPushButton("● Start Meeting")
        self._dock_state_btn.setObjectName("WisprDockBtn")
        self._dock_state_btn.setFixedHeight(40)
        self._dock_state_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        dock_row.addWidget(self._dock_state_btn)

        self._qa_bar = QFrame()
        self._qa_bar.setObjectName("WisprAskBar")
        self._qa_bar.setFixedHeight(40)
        qa_bar_lay = QHBoxLayout(self._qa_bar)
        qa_bar_lay.setContentsMargins(8, 4, 6, 4)
        qa_bar_lay.setSpacing(6)

        self._qa_scope_btn = QPushButton("This")
        self._qa_scope_btn.setObjectName("WisprScopeChip")
        self._qa_scope_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        qa_bar_lay.addWidget(self._qa_scope_btn)

        self._qa_input = QLineEdit()
        self._qa_input.setObjectName("WisprAskInput")
        qa_bar_lay.addWidget(self._qa_input, 1)

        self._qa_missed_btn = QPushButton("What did I miss?")
        self._qa_missed_btn.setObjectName("WisprMissedChip")
        self._qa_missed_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        qa_bar_lay.addWidget(self._qa_missed_btn)

        dock_row.addWidget(self._qa_bar, 1)
        root.addLayout(dock_row)

        self._active_tab_idx = 2
        self._set_active_tab(2)

    def _connect_signals(self) -> None:
        self._tab_thoughts_btn.clicked.connect(lambda: self._set_active_tab(0))
        self._tab_transcript_btn.clicked.connect(lambda: self._set_active_tab(1))
        self._tab_summary_btn.clicked.connect(lambda: self._set_active_tab(2))

        self._title_edit.editingFinished.connect(self._on_title_edited)
        self._thoughts_edit.textChanged.connect(self._on_thoughts_changed)
        self._tr_search_btn.toggled.connect(self._on_search_toggled)
        self._tr_search_edit.textChanged.connect(lambda _t: self._render_transcript(force_rebuild=True))
        self._tr_speakers_btn.clicked.connect(self._on_speakers_menu_clicked)
        self._tr_copy_btn.clicked.connect(self._on_copy_transcript_clicked)
        self._live_banner_close.clicked.connect(self._dismiss_live_banner)

        self._copy_summary_btn.clicked.connect(self._on_copy_summary_clicked)
        self._regen_btn.clicked.connect(self._on_regen_clicked)
        self._force_sum_btn.clicked.connect(self._on_force_summarize_clicked)
        self._dock_state_btn.clicked.connect(self._on_dock_state_clicked)
        self._qa_scope_btn.clicked.connect(self._toggle_qa_scope)
        self._qa_input.returnPressed.connect(self._on_ask_ai_submit)
        self._qa_missed_btn.clicked.connect(self._on_what_missed_clicked)
        self._qa_close_btn.clicked.connect(lambda: self._qa_card.setVisible(False))
        self._diar_btn.clicked.connect(self._on_download_diarization)

        self._utterance_signal.connect(self._handle_utterance_ui)
        self._interim_signal.connect(self._handle_interim_ui)
        self._speakers_updated_signal.connect(self._handle_speakers_updated_ui)
        self._qa_done_signal.connect(self._handle_qa_done_ui)
        self._diar_progress_signal.connect(self._handle_diar_progress_ui)
        self._diar_done_signal.connect(self._handle_diar_done_ui)

    def _apply_theme_styles(self) -> None:
        t = get_theme_tokens(self._theme)
        bg = t["canvas_bg"]
        card_bg = t["subtle_card_bg"]
        card_border = t["subtle_card_border"]
        text = t["text_strong"]
        muted = t["text_muted"]
        chip_bg = t["chip_bg"]
        chip_border = t["chip_border"]
        hover_bg = t["hover_bg"]
        divider = t["divider"]

        self.setStyleSheet(
            f"""
            QLineEdit#WisprTitleEdit {{
                font-family: Georgia, "Times New Roman", serif;
                font-size: 22px;
                font-weight: 500;
                color: {text};
                background: transparent;
                border: none;
                padding: 2px 0;
            }}
            QLabel#WisprMetaChip {{
                background-color: {chip_bg};
                border: 1px solid {chip_border};
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 11px;
                color: {muted};
            }}
            QFrame#WisprSubtleCard, QFrame#SummaryProgressCard {{
                background-color: {card_bg};
                border: 1px solid {card_border};
                border-radius: 12px;
            }}
            QLabel#SummaryProgressTitle {{
                font-size: 14px;
                font-weight: 600;
                color: {text};
            }}
            QLabel#TooShortLabel {{
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.5px;
                color: {muted};
            }}
            QLabel#SectionHeader {{
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.6px;
                color: {t["section_hdr"]};
            }}
            QLabel#FieldHint {{
                font-size: 12px;
                color: {muted};
            }}
            QLabel#PrivacyNotice {{
                font-size: 11px;
                color: {muted};
            }}
            QTextEdit#WisprPlainEditor {{
                background: transparent;
                border: none;
                color: {text};
                font-size: 13.5px;
            }}
            QLineEdit#WisprInlineSearch {{
                background: transparent;
                border: none;
                border-bottom: 1px solid {chip_border};
                color: {text};
                font-size: 13px;
                padding: 2px 4px;
            }}
            QPushButton#WisprIconBtn {{
                background: transparent;
                border: none;
                border-radius: 6px;
                color: {muted};
                padding: 3px;
                font-size: 13px;
            }}
            QPushButton#WisprIconBtn:hover, QPushButton#WisprIconBtn:checked {{
                background-color: {hover_bg};
                color: {text};
            }}
            QPushButton#WisprPillBtn {{
                background-color: {chip_bg};
                border: 1px solid {chip_border};
                border-radius: 8px;
                color: {text};
                padding: 4px 10px;
                font-size: 12px;
            }}
            QPushButton#WisprPillBtn:hover {{
                background-color: {hover_bg};
            }}
            QPushButton#WisprDockBtn {{
                background-color: {card_bg};
                border: 1px solid {chip_border};
                border-radius: 20px;
                color: {text};
                padding: 0 16px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton#WisprDockBtn:hover {{
                background-color: {hover_bg};
            }}
            QFrame#WisprAskBar {{
                background-color: {bg};
                border: 1px solid {chip_border};
                border-radius: 20px;
            }}
            QLineEdit#WisprAskInput {{
                background: transparent;
                border: none;
                color: {text};
                font-size: 13px;
            }}
            QPushButton#WisprScopeChip, QPushButton#WisprMissedChip {{
                background-color: {chip_bg};
                border: 1px solid {chip_border};
                border-radius: 12px;
                color: {text};
                padding: 3px 10px;
                font-size: 11.5px;
                font-weight: 500;
            }}
            QPushButton#WisprScopeChip:hover, QPushButton#WisprMissedChip:hover {{
                background-color: {hover_bg};
            }}
            """
        )
        self._tab_divider.setStyleSheet(f"background-color: {divider}; border: none;")
        self._refresh_tab_styles()
        self._refresh_workspace_icons()

    def _refresh_workspace_icons(self) -> None:
        t = get_theme_tokens(self._theme)
        muted = t["text_muted"]
        self._tr_clock_lbl.setPixmap(render_icon("clock", muted, 13).pixmap(13, 13))
        self._tr_search_btn.setIcon(render_icon("search", muted, 14))
        self._tr_speakers_btn.setIcon(render_icon("speakers", muted, 14))
        self._tr_copy_btn.setIcon(render_icon("copy", muted, 14))
        self._copy_summary_btn.setIcon(render_icon("copy", t["text_secondary"], 13))
        self._live_banner_close.setIcon(render_icon("close", muted, 11))
        self._update_dock_controls()

    def _reset_copy_icon(self) -> None:
        t = get_theme_tokens(self._theme)
        self._tr_copy_btn.setIcon(render_icon("copy", t["text_muted"], 14))

    def _reset_copy_summary_icon(self) -> None:
        t = get_theme_tokens(self._theme)
        self._copy_summary_btn.setIcon(render_icon("copy", t["text_secondary"], 13))

    def _set_active_tab(self, idx: int) -> None:
        self._active_tab_idx = idx
        self._tabs.setCurrentIndex(idx)
        self._refresh_tab_styles()

    def _refresh_tab_styles(self) -> None:
        t = get_theme_tokens(self._theme)
        active_fg = t["tab_active_fg"]
        inactive_fg = t["tab_fg"]
        buttons = (self._tab_thoughts_btn, self._tab_transcript_btn, self._tab_summary_btn)
        for i, btn in enumerate(buttons):
            if i == self._active_tab_idx:
                btn.setStyleSheet(
                    f"QPushButton {{ background: transparent; border: none; "
                    f"border-bottom: 2px solid {active_fg}; color: {active_fg}; "
                    "font-size: 13px; font-weight: 600; padding: 6px 2px 8px 2px; }"
                )
            else:
                btn.setStyleSheet(
                    f"QPushButton {{ background: transparent; border: none; "
                    f"border-bottom: 2px solid transparent; color: {inactive_fg}; "
                    "font-size: 13px; font-weight: 500; padding: 6px 2px 8px 2px; }"
                )
        self._tab_summary_btn.setIcon(
            render_icon(
                "sparkles",
                active_fg if self._active_tab_idx == 2 else inactive_fg,
                13,
            )
        )

    def _update_dock_controls(self) -> None:
        t = get_theme_tokens(self._theme)
        is_current_recording = self._is_recording and self._current_meeting_id == self._recording_meeting_id
        self._privacy_lbl.setVisible(is_current_recording)
        self._qa_missed_btn.setVisible(is_current_recording and self._current_meeting_id is not None)
        self._meters_widget.setVisible(is_current_recording)

        self._dock_state_btn.setIconSize(QSize(12, 12))
        if self._is_recording:
            self._dock_state_btn.setVisible(True)
            self._dock_state_btn.setIcon(render_icon("stop_square", t["accent"], 12))
            self._dock_state_btn.setText(tr("notetaker.btn.stop_short", self._lang))
        elif self._current_meeting_id is not None:
            self._dock_state_btn.setVisible(True)
            self._dock_state_btn.setIcon(render_icon("record_dot", t["accent"], 12))
            self._dock_state_btn.setText(tr("notetaker.btn.resume", self._lang))
        else:
            self._dock_state_btn.setVisible(self._show_bottom_start)
            self._dock_state_btn.setIcon(render_icon("record_dot", t["accent"], 12))
            self._dock_state_btn.setText(tr("notetaker.btn.start", self._lang))

    def _poll_live_meters(self) -> None:
        if not self._is_recording or self._level_provider is None:
            return
        mic_lvl, sys_lvl, elapsed = self._level_provider()
        self._mic_bar.setValue(max(0, min(100, int(round(mic_lvl * 100)))))
        self._sys_bar.setValue(max(0, min(100, int(round(sys_lvl * 100)))))
        if self._current_meeting_id == self._recording_meeting_id:
            self._tr_meta_lbl.setText(format_timestamp(elapsed))

    def _render_current_meeting(self, *, force_transcript_rebuild: bool = False) -> None:
        mid = self._current_meeting_id
        meeting = self._storage.get_meeting(mid) if mid is not None else None
        has_meeting = meeting is not None
        is_rec_this = self._is_recording and mid is not None and mid == self._recording_meeting_id

        self._title_edit.setEnabled(has_meeting)
        self._thoughts_edit.setEnabled(has_meeting)
        self._copy_summary_btn.setEnabled(has_meeting and meeting is not None and bool(meeting.summary_md.strip()))
        self._regen_btn.setEnabled(has_meeting and not self._is_recording and self._summarizing_meeting_id != mid)
        self._live_wave.set_active(is_rec_this)
        self._update_dock_controls()

        self._loading_ui = True
        try:
            if meeting is None:
                self._title_edit.setText("")
                self._date_chip.setText(tr("notetaker.today", self._lang))
                self._mode_chip.setVisible(False)
                self._thoughts_edit.setPlainText("")
                self._summary_view.setPlainText(tr("notetaker.empty_hint", self._lang))
                self._too_short_card.setVisible(False)
                self._progress_card.set_active(False)
                self._clear_speakers_bar()
                self._clear_action_items()
                self._clear_layout(self._committed_lay)
                self._clear_layout(self._interim_lay)
                return

            self._title_edit.setText(meeting.title)
            self._title_edit.setCursorPosition(0)
            today_str = date.today().isoformat()
            self._date_chip.setText(
                tr("notetaker.today", self._lang) if meeting.created_at[:10] == today_str else meeting.created_at[:10]
            )
            self._mode_chip.setVisible(True)
            mode_key = "notetaker.mode_chip.mic_only" if meeting.audio_mode == "mic_only" else "notetaker.mode_chip.dual"
            self._mode_chip.setText(tr(mode_key, self._lang))
            if self._thoughts_edit.toPlainText() != meeting.my_thoughts:
                self._thoughts_edit.setPlainText(meeting.my_thoughts)

            is_sum_active = self._summarizing_meeting_id == mid or meeting.status == "summarizing"
            self._progress_card.set_active(is_sum_active)
            show_too_short = (
                not is_sum_active
                and not is_rec_this
                and not meeting.summary_md.strip()
                and self._too_short_meeting_id == mid
            )
            self._too_short_card.setVisible(show_too_short)
            self._summary_view.setVisible(not show_too_short)
            if meeting.summary_md.strip():
                self._set_summary_markdown(meeting.summary_md)
            else:
                self._summary_view.setPlainText("" if (show_too_short or is_sum_active) else tr("notetaker.no_summary", self._lang))
        finally:
            self._loading_ui = False

        self._render_speakers_bar()
        self._render_action_items()
        self._render_transcript(force_rebuild=force_transcript_rebuild)

    def _clear_speakers_bar(self) -> None:
        while self._speakers_lay.count() > 1:
            item = self._speakers_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.deleteLater()

    def _render_speakers_bar(self) -> None:
        self._clear_speakers_bar()
        if self._current_meeting_id is None:
            return
        speakers = self._storage.get_speakers(self._current_meeting_id)
        colors = _SPEAKER_COLORS_LIGHT if self._theme == "light" else _SPEAKER_COLORS_DARK
        max_inline = 3
        for spk in speakers[:max_inline]:
            color = colors[spk.color_idx % len(colors)]
            btn = QPushButton(spk.display_name)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(
                f"QPushButton {{ background: transparent; border: 1px solid {color}; color: {color}; "
                "border-radius: 10px; padding: 1px 9px; font-size: 11px; }"
            )
            btn.clicked.connect(lambda _checked=False, s=spk: self._on_speaker_chip_clicked(s))
            self._speakers_lay.insertWidget(self._speakers_lay.count() - 1, btn)
        if len(speakers) > max_inline:
            extra_cnt = len(speakers) - max_inline
            more_btn = QPushButton(f"+{extra_cnt}")
            more_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            more_btn.setObjectName("WisprMetaChip")
            more_btn.setStyleSheet(
                "QPushButton { background: transparent; border: 1px solid #52525b; color: #a1a1aa; "
                "border-radius: 10px; padding: 1px 8px; font-size: 11px; }"
            )
            more_btn.clicked.connect(self._on_speakers_menu_clicked)
            self._speakers_lay.insertWidget(self._speakers_lay.count() - 1, more_btn)

    def _on_speaker_chip_clicked(
        self,
        speaker: MeetingSpeaker,
        *,
        global_pos: QPoint | None = None,
    ) -> None:
        if self._current_meeting_id is None:
            return
        menu = QMenu(self)
        menu.setStyleSheet(build_menu_stylesheet(self._theme))
        rename_act = menu.addAction(tr("notetaker.speaker.rename", self._lang))
        merge_menu = menu.addMenu(tr("notetaker.speaker.merge", self._lang))
        others = [
            s for s in self._storage.get_speakers(self._current_meeting_id)
            if s.speaker_key != speaker.speaker_key
        ]
        merge_actions = {}
        for other in others:
            act = merge_menu.addAction(other.display_name)
            merge_actions[act] = other
        merge_menu.setEnabled(bool(others))

        chosen = menu.exec(global_pos if global_pos is not None else self.cursor().pos())
        if chosen is None:
            return
        if chosen == rename_act:
            dlg = QInputDialog(self)
            dlg.setWindowTitle(tr("notetaker.speaker.rename", self._lang))
            dlg.setLabelText(tr("notetaker.speaker.rename_prompt", self._lang))
            dlg.setTextValue(speaker.display_name)
            dlg.setStyleSheet(build_dialog_stylesheet(self._theme))
            apply_windows_titlebar_theme(dlg, self._theme)
            ok = dlg.exec() == int(QInputDialog.DialogCode.Accepted)
            new_name = dlg.textValue()
            if ok and new_name.strip():
                self._apply_speaker_change(
                    lambda mid: self._storage.rename_speaker(mid, speaker.speaker_key, new_name.strip())
                )
        elif chosen in merge_actions:
            target = merge_actions[chosen]
            self._apply_speaker_change(
                lambda mid: self._storage.merge_speakers(mid, speaker.speaker_key, target.speaker_key)
            )

    def _apply_speaker_change(self, mutator: Callable[[int], None]) -> None:
        mid = self._current_meeting_id
        if mid is None:
            return
        sb = self._tr_scroll.verticalScrollBar()
        saved_val = sb.value()
        was_stick = self._stick_to_bottom
        mutator(mid)
        self._render_speakers_bar()
        self._render_transcript(force_rebuild=True)
        if not was_stick:
            QTimer.singleShot(0, lambda v=saved_val: self._restore_transcript_scroll(v))
        self.meeting_updated.emit(mid)

    def _restore_transcript_scroll(self, val: int) -> None:
        sb = self._tr_scroll.verticalScrollBar()
        self._programmatic_scroll = True
        try:
            sb.setValue(min(val, sb.maximum()))
        finally:
            self._programmatic_scroll = False

    def _on_transcript_speaker_context_menu(self, speaker_key: str, global_pos: QPoint) -> None:
        if self._current_meeting_id is None:
            return
        speakers = {s.speaker_key: s for s in self._storage.get_speakers(self._current_meeting_id)}
        spk = speakers.get(speaker_key)
        if spk is None:
            return
        self._on_speaker_chip_clicked(spk, global_pos=global_pos)

    def _on_speakers_menu_clicked(self) -> None:
        if self._current_meeting_id is None:
            return
        speakers = self._storage.get_speakers(self._current_meeting_id)
        if not speakers:
            return
        menu = QMenu(self)
        menu.setStyleSheet(build_menu_stylesheet(self._theme))
        actions = {menu.addAction(s.display_name): s for s in speakers}
        chosen = menu.exec(self.cursor().pos())
        if chosen in actions:
            self._on_speaker_chip_clicked(actions[chosen])

    def _set_summary_markdown(self, md: str) -> None:
        normalized = format_summary_markdown(md)
        self._summary_view.setMarkdown(normalized)
        doc = self._summary_view.document()
        block = doc.firstBlock()
        while block.isValid():
            fmt = block.blockFormat()
            if fmt.headingLevel() > 0 and block.blockNumber() > 0:
                cursor = QTextCursor(block)
                new_fmt = block.blockFormat()
                new_fmt.setTopMargin(16.0)
                new_fmt.setBottomMargin(6.0)
                cursor.setBlockFormat(new_fmt)
            block = block.next()

    def _clear_action_items(self) -> None:
        while self._action_lay.count() > 0:
            item = self._action_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

    def _render_action_items(self) -> None:
        self._clear_action_items()
        if self._current_meeting_id is None:
            self._update_action_scroll_height()
            return
        items: list[MeetingActionItem] = self._storage.get_action_items(self._current_meeting_id)
        if not items:
            empty = QLabel(tr("notetaker.no_action_items", self._lang))
            empty.setObjectName("FieldHint")
            self._action_lay.addWidget(empty)
            self._update_action_scroll_height()
            return
        for item in items:
            label = f"{item.text} ({item.owner})" if item.owner else item.text
            row = _ActionItemRow(
                label,
                checked=item.done,
                theme=self._theme,
                check_style=self._check_style,
            )
            row.toggled.connect(
                lambda checked, item_id=item.id: self._on_action_toggled(item_id, checked)
            )
            self._action_lay.addWidget(row)
        self._update_action_scroll_height()

    def _update_action_scroll_height(self) -> None:
        rows = self._action_container.findChildren(_ActionItemRow)
        if not rows:
            self._action_scroll.setFixedHeight(28)
            return
        vp_w = self._action_scroll.viewport().width()
        eff_w = vp_w if vp_w > 120 else max(300, self.width() - 40)
        lbl_w = max(180, eff_w - 36)
        total_h = 4
        for row in rows:
            lbl = row._lbl
            fm = QFontMetrics(lbl.font())
            text_h = fm.boundingRect(
                QRect(0, 0, lbl_w, 10000),
                Qt.TextFlag.TextWordWrap,
                lbl.text(),
            ).height()
            row_h = max(26, max(text_h, lbl.heightForWidth(lbl_w)) + 6)
            total_h += row_h + 4
        max_cap = max(110, int(max(420, self.height()) * 0.34))
        self._action_scroll.setFixedHeight(max(36, min(total_h, max_cap)))

    def _on_action_toggled(self, item_id: int, checked: bool) -> None:
        self._storage.set_action_item_done(item_id, checked)
        if self._current_meeting_id is not None:
            self.meeting_updated.emit(self._current_meeting_id)

    @staticmethod
    def _clear_layout(lay: QVBoxLayout) -> None:
        while lay.count() > 0:
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                MeetingNoteWorkspace._clear_layout(item.layout())  # type: ignore[arg-type]

    def _render_transcript(self, *, force_rebuild: bool = False) -> None:
        mid = self._current_meeting_id
        if mid is None:
            self._clear_layout(self._committed_lay)
            self._clear_layout(self._interim_lay)
            return

        meeting = self._storage.get_meeting(mid)
        speakers = {s.speaker_key: s for s in self._storage.get_speakers(mid)}
        utterances = self._storage.get_utterances(mid)
        flt = self._tr_search_edit.text().strip().lower()
        is_live = self._is_recording and mid == self._recording_meeting_id

        # Card top timer & banner
        if is_live:
            elapsed = self._level_provider()[2] if self._level_provider is not None else 0.0
            self._tr_meta_lbl.setText(format_timestamp(elapsed))
        else:
            spk_count = max(1, len(speakers)) if utterances else len(speakers)
            spk_label = (
                tr("notetaker.speaker_single", self._lang)
                if spk_count == 1
                else tr("notetaker.speakers_count", self._lang, n=spk_count)
            )
            dur = meeting.duration_sec if meeting is not None else 0.0
            self._tr_meta_lbl.setText(f"{spk_label} · {format_timestamp(dur)}")
        self._live_banner_row.setVisible(is_live and not self._live_banner_dismissed)

        filtered = [
            u for u in utterances
            if not flt
            or flt in u.text.lower()
            or flt in (speakers[u.speaker_key].display_name.lower() if u.speaker_key in speakers else u.speaker_key.lower())
        ]

        # Incremental append during live recording when only 1 new utterance arrived
        if (
            is_live
            and not force_rebuild
            and not flt
            and self._rendered_live_meeting_id == mid
            and len(filtered) >= self._rendered_live_count
        ):
            for u in filtered[self._rendered_live_count :]:
                self._append_live_utterance_widget(u, speakers)
            self._rendered_live_count = len(filtered)
            self._render_interim_bubbles(speakers)
            self._scroll_transcript_to_bottom()
            return

        self._clear_layout(self._committed_lay)
        self._last_live_speaker_key = None
        if is_live:
            for u in filtered:
                self._append_live_utterance_widget(u, speakers)
            self._rendered_live_meeting_id = mid
            self._rendered_live_count = len(filtered)
            self._render_interim_bubbles(speakers)
            self._scroll_transcript_to_bottom()
        else:
            self._rendered_live_meeting_id = None
            self._rendered_live_count = 0
            self._clear_layout(self._interim_lay)
            self._render_editorial_paragraphs(filtered, speakers, search_filter=flt)

    def _bubble_max_width(self) -> int:
        vp_w = self._tr_scroll.viewport().width()
        if vp_w > 120:
            return max(220, vp_w - 32)
        return 390

    def _sync_bubble_widths(self) -> None:
        max_w = self._bubble_max_width()
        for b in self._committed_container.findChildren(_SpeechBubble):
            b.sync_max_width(max_w)
        for b in self._interim_container.findChildren(_SpeechBubble):
            b.sync_max_width(max_w)
        self._tr_scroll_body.updateGeometry()

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is self._tr_scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._sync_bubble_widths()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._sync_bubble_widths()
        self._update_action_scroll_height()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._update_action_scroll_height()

    def _make_speaker_header_label(
        self,
        speaker_key: str,
        display_name: str,
        color: str,
        *,
        timestamp_text: str = "",
        margin_top: int = 6,
    ) -> QLabel:
        if timestamp_text:
            ts_color = "#9ca3af" if self._theme == "light" else "#6e6e78"
            hdr = QLabel(
                f"<b style='color:{color};'>{html.escape(display_name)}</b> "
                f"<span style='color:{ts_color};font-size:11px;margin-left:6px;'>{timestamp_text}</span>"
            )
            hdr.setTextFormat(Qt.TextFormat.RichText)
            hdr.setStyleSheet("QLabel { font-size: 13px; margin-top: 10px; margin-bottom: 2px; }")
        else:
            hdr = QLabel(display_name)
            hdr.setStyleSheet(
                f"QLabel {{ color: {color}; font-size: 13px; font-weight: 600; margin-top: {margin_top}px; }}"
            )
        hdr.setProperty("speaker_key", speaker_key)
        hdr.setCursor(Qt.CursorShape.PointingHandCursor)
        hdr.setToolTip(tr("notetaker.speaker.rename", self._lang))
        hdr.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        hdr.customContextMenuRequested.connect(
            lambda pos, sk=speaker_key, w=hdr: self._on_transcript_speaker_context_menu(sk, w.mapToGlobal(pos))
        )
        return hdr

    def _append_live_utterance_widget(
        self,
        utt: MeetingUtterance,
        speakers: dict[str, MeetingSpeaker],
    ) -> None:
        spk = speakers.get(utt.speaker_key)
        spk_name = spk.display_name if spk is not None else utt.speaker_key
        is_self = (spk.is_self if spk is not None else utt.speaker_key == "you")
        colors = _SPEAKER_COLORS_LIGHT if self._theme == "light" else _SPEAKER_COLORS_DARK
        color = colors[(spk.color_idx if spk is not None else 1) % len(colors)]

        if utt.speaker_key != self._last_live_speaker_key:
            hdr = self._make_speaker_header_label(utt.speaker_key, spk_name, color, margin_top=6)
            self._committed_lay.addWidget(hdr)
            self._last_live_speaker_key = utt.speaker_key

        bubble = _SpeechBubble(
            utt.text,
            is_self=is_self,
            is_interim=False,
            theme=self._theme,
            max_width=self._bubble_max_width(),
        )
        self._committed_lay.addWidget(bubble)

    def _render_interim_bubbles(self, speakers: dict[str, MeetingSpeaker]) -> None:
        self._clear_layout(self._interim_lay)
        if not self._is_recording or self._current_meeting_id != self._recording_meeting_id:
            return
        colors = _SPEAKER_COLORS_LIGHT if self._theme == "light" else _SPEAKER_COLORS_DARK
        for _ch, (spk_key, text) in self._interim_texts.items():
            if not text.strip():
                continue
            spk = speakers.get(spk_key)
            spk_name = spk.display_name if spk is not None else spk_key
            is_self = (spk.is_self if spk is not None else spk_key == "you")
            if spk_key != self._last_live_speaker_key:
                color = colors[(spk.color_idx if spk is not None else 1) % len(colors)]
                hdr = self._make_speaker_header_label(spk_key, spk_name, color, margin_top=0)
                self._interim_lay.addWidget(hdr)
            bubble = _SpeechBubble(
                text,
                is_self=is_self,
                is_interim=True,
                theme=self._theme,
                max_width=self._bubble_max_width(),
            )
            self._interim_lay.addWidget(bubble)

    def _render_editorial_paragraphs(
        self,
        utterances: list[MeetingUtterance],
        speakers: dict[str, MeetingSpeaker],
        *,
        search_filter: str = "",
    ) -> None:
        if not utterances:
            muted = "#6b7280" if self._theme == "light" else "#8e8e98"
            empty = QLabel("...")
            empty.setStyleSheet(f"color: {muted}; font-size: 13px;")
            self._committed_lay.addWidget(empty)
            return

        colors = _SPEAKER_COLORS_LIGHT if self._theme == "light" else _SPEAKER_COLORS_DARK
        body_color = "#1f2937" if self._theme == "light" else "#e4e4e7"

        prev_spk: str | None = None
        for u in utterances:
            spk = speakers.get(u.speaker_key)
            spk_name = spk.display_name if spk is not None else u.speaker_key
            color = colors[(spk.color_idx if spk is not None else 1) % len(colors)]
            if u.speaker_key != prev_spk:
                ts = format_timestamp(u.start_sec)
                hdr = self._make_speaker_header_label(
                    u.speaker_key, spk_name, color, timestamp_text=ts
                )
                self._committed_lay.addWidget(hdr)
                prev_spk = u.speaker_key

            escaped_txt = html.escape(u.text)
            if search_filter:
                pat = re.compile(re.escape(html.escape(search_filter)), re.IGNORECASE)
                escaped_txt = pat.sub(
                    lambda m: f"<span style='background-color:#facc15;color:#111113;'>{m.group(0)}</span>",
                    escaped_txt,
                )
            p = QLabel(escaped_txt)
            p.setTextFormat(Qt.TextFormat.RichText)
            p.setWordWrap(True)
            p.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            p.setStyleSheet(f"QLabel {{ color: {body_color}; font-size: 13.5px; line-height: 1.45; margin-bottom: 4px; }}")
            self._committed_lay.addWidget(p)

    def _on_transcript_scrolled(self, value: int) -> None:
        if self._programmatic_scroll:
            return
        sb = self._tr_scroll.verticalScrollBar()
        self._stick_to_bottom = (sb.maximum() - value) <= 80

    def _scroll_transcript_to_bottom(self, *, force: bool = False) -> None:
        if force:
            self._stick_to_bottom = True
        if self._stick_to_bottom:
            self._scroll_timer.start()

    def _apply_scroll_to_bottom(self) -> None:
        if not self._stick_to_bottom:
            return
        sb = self._tr_scroll.verticalScrollBar()
        self._programmatic_scroll = True
        try:
            sb.setValue(sb.maximum())
        finally:
            self._programmatic_scroll = False

    # ---- Slots & Background Actions ----

    def _handle_utterance_ui(self, meeting_id: int, _utt: object, _spk: object) -> None:
        if self._current_meeting_id == meeting_id:
            self._render_speakers_bar()
            self._render_transcript(force_rebuild=False)

    def _handle_interim_ui(self, meeting_id: int, channel: str, speaker_key: str, text: str) -> None:
        if self._current_meeting_id != meeting_id or not self._is_recording:
            return
        if text.strip():
            self._interim_texts[channel] = (speaker_key, text)
        else:
            self._interim_texts.pop(channel, None)
        speakers = {s.speaker_key: s for s in self._storage.get_speakers(meeting_id)}
        self._render_interim_bubbles(speakers)
        self._scroll_transcript_to_bottom()

    def _handle_speakers_updated_ui(self, meeting_id: int) -> None:
        if self._current_meeting_id == meeting_id:
            self._render_speakers_bar()
            self._render_transcript(force_rebuild=True)

    def _on_title_edited(self) -> None:
        if self._loading_ui or self._current_meeting_id is None:
            return
        new_title = self._title_edit.text().strip()
        if not new_title:
            return
        self._storage.update_meeting(self._current_meeting_id, title=new_title)
        self.meeting_updated.emit(self._current_meeting_id)

    def _on_thoughts_changed(self) -> None:
        if self._loading_ui or self._current_meeting_id is None:
            return
        self._thoughts_timer.start()

    def _save_thoughts_now(self) -> None:
        if self._current_meeting_id is None:
            return
        if self._thoughts_timer.isActive():
            self._thoughts_timer.stop()
        self._storage.update_meeting(self._current_meeting_id, my_thoughts=self._thoughts_edit.toPlainText())
        self.meeting_updated.emit(self._current_meeting_id)

    def _on_search_toggled(self, checked: bool) -> None:
        self._tr_search_edit.setVisible(checked)
        if checked:
            self._tr_search_edit.setFocus()
        else:
            self._tr_search_edit.clear()

    def _dismiss_live_banner(self) -> None:
        self._live_banner_dismissed = True
        self._live_banner_row.setVisible(False)

    def _on_copy_transcript_clicked(self) -> None:
        if self._current_meeting_id is None:
            return
        speakers = {s.speaker_key: s.display_name for s in self._storage.get_speakers(self._current_meeting_id)}
        utterances = self._storage.get_utterances(self._current_meeting_id)
        txt = "\n".join(f"[{format_timestamp(u.start_sec)}] {speakers.get(u.speaker_key, u.speaker_key)}: {u.text}" for u in utterances)
        if txt:
            QApplication.clipboard().setText(txt)
            t = get_theme_tokens(self._theme)
            self._tr_copy_btn.setIcon(render_icon("check", t["accent"], 14))
            self._copy_reset_timer.start()

    def _on_copy_summary_clicked(self) -> None:
        if self._current_meeting_id is None:
            return
        meeting = self._storage.get_meeting(self._current_meeting_id)
        if meeting is None or not meeting.summary_md.strip():
            return
        formatted_md = format_summary_markdown(meeting.summary_md)
        mime = QMimeData()
        mime.setText(formatted_md)
        mime.setHtml(self._summary_view.toHtml())
        QApplication.clipboard().setMimeData(mime)
        t = get_theme_tokens(self._theme)
        self._copy_summary_btn.setIcon(render_icon("check", t["accent"], 13))
        self._copy_summary_reset_timer.start()

    def _on_regen_clicked(self) -> None:
        if self._current_meeting_id is None:
            return
        self._save_thoughts_now()
        self.regenerate_summary_requested.emit(self._current_meeting_id, False)

    def _on_force_summarize_clicked(self) -> None:
        if self._current_meeting_id is None:
            return
        self._save_thoughts_now()
        self.regenerate_summary_requested.emit(self._current_meeting_id, True)

    def _on_dock_state_clicked(self) -> None:
        if self._is_recording:
            self.stop_meeting_requested.emit()
        elif self._current_meeting_id is not None:
            self.resume_meeting_requested.emit(self._current_meeting_id)
        else:
            self.start_meeting_requested.emit("dual")

    def _toggle_qa_scope(self) -> None:
        self._qa_scope = "all" if self._qa_scope == "current" else "current"
        self._update_qa_scope_button()

    def _update_qa_scope_button(self) -> None:
        key = "notetaker.qa.scope_all_short" if self._qa_scope == "all" else "notetaker.qa.scope_this_short"
        self._qa_scope_btn.setText(tr(key, self._lang))

    def _on_what_missed_clicked(self) -> None:
        self._ask_question_text(
            tr("notetaker.qa.what_missed_prompt", self._lang),
            force_current=True,
            plain_bullets=True,
        )

    def _on_ask_ai_submit(self) -> None:
        q = self._qa_input.text().strip()
        if not q:
            return
        self._ask_question_text(q, force_current=False, plain_bullets=False)

    def _ask_question_text(
        self,
        question: str,
        *,
        force_current: bool,
        plain_bullets: bool = False,
    ) -> None:
        llm = self._llm_factory() if self._llm_factory is not None else None
        if llm is None:
            self._qa_card.setVisible(True)
            self._qa_answer.setPlainText(tr("notetaker.llm_unavailable", self._lang))
            return
        target_mid = self._current_meeting_id if (force_current or self._qa_scope == "current") else None
        if (force_current or self._qa_scope == "current") and target_mid is None:
            return

        self._qa_card.setVisible(True)
        self._qa_answer.setPlainText(tr("notetaker.qa.thinking", self._lang))
        self._qa_input.setEnabled(False)
        self._qa_missed_btn.setEnabled(False)
        lang_code = self._lang
        self._qa_plain_mode = plain_bullets

        def _worker() -> None:
            try:
                engine = MeetingQAEngine(self._storage, llm)  # type: ignore[arg-type]
                res = engine.answer_question(
                    question,
                    meeting_id=target_mid,
                    lang=lang_code,
                    plain_bullets=plain_bullets,
                )
                self._qa_done_signal.emit(res, "")
            except Exception as exc:
                self._qa_done_signal.emit(None, str(exc))

        threading.Thread(target=_worker, name="LocalFlow-AskAI", daemon=True).start()

    def _handle_qa_done_ui(self, result: object, err: str) -> None:
        self._qa_input.setEnabled(True)
        self._qa_missed_btn.setEnabled(True)
        if err:
            self._qa_answer.setPlainText(err)
            return
        if isinstance(result, QAResult):
            text = result.answer
            if not getattr(self, "_qa_plain_mode", False):
                cites = ", ".join(f"[{c.title} · {c.date_label}]" for c in result.citations)
                if cites:
                    text += f"\n\nSources: {cites}"
            self._qa_answer.setPlainText(text)
            self._qa_input.clear()

    def _refresh_diarization_banner(self) -> None:
        installed = find_diarization_model_path() is not None
        self._diar_banner.setVisible(not installed)

    def _on_download_diarization(self) -> None:
        self._diar_btn.setEnabled(False)
        self._diar_progress.setVisible(True)
        self._diar_progress.setValue(0)
        cancel = threading.Event()
        self._diar_cancel = cancel

        def _progress(p: DownloadProgress) -> None:
            self._diar_progress_signal.emit(p.percent, p.stage)

        def _worker() -> None:
            try:
                download_diarization_model(on_progress=_progress, cancel_event=cancel, lang=self._lang)
                self._diar_done_signal.emit(True, "")
            except Exception as exc:
                self._diar_done_signal.emit(False, str(exc))

        threading.Thread(target=_worker, name="LocalFlow-DiarDL", daemon=True).start()

    def _handle_diar_progress_ui(self, pct: int, _stage: str) -> None:
        self._diar_progress.setValue(pct)

    def _handle_diar_done_ui(self, ok: bool, err: str) -> None:
        self._diar_btn.setEnabled(True)
        self._diar_progress.setVisible(False)
        if ok:
            self._refresh_diarization_banner()
        elif err:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle(tr("notetaker.title", self._lang))
            box.setText(err)
            box.setStyleSheet(build_dialog_stylesheet(self._theme))
            apply_windows_titlebar_theme(box, self._theme)
            box.exec()


class MeetingSideWindow(QWidget):
    """Right-docked Wispr Flow-style companion window opened automatically when a meeting runs."""

    back_to_hub_requested = Signal()

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
        self._has_positioned = False

        outer_lay = QVBoxLayout(self)
        outer_lay.setContentsMargins(0, 0, 0, 0)

        self._shell = QFrame()
        self._shell.setObjectName("WisprSideShell")
        outer_lay.addWidget(self._shell)

        shell_lay = QVBoxLayout(self._shell)
        shell_lay.setContentsMargins(0, 0, 0, 0)
        shell_lay.setSpacing(0)

        # Custom draggable top bar
        self._topbar = QWidget()
        self._topbar.setFixedHeight(44)
        top_lay = QHBoxLayout(self._topbar)
        top_lay.setContentsMargins(14, 8, 10, 4)
        top_lay.setSpacing(6)

        self._back_btn = QPushButton()
        self._back_btn.setObjectName("WisprChromeBtn")
        self._back_btn.setFixedSize(28, 28)
        self._back_btn.setIconSize(QSize(14, 14))
        self._back_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self._more_btn = QPushButton()
        self._more_btn.setObjectName("WisprChromeBtn")
        self._more_btn.setFixedSize(28, 28)
        self._more_btn.setIconSize(QSize(14, 14))
        self._more_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self._export_btn = QPushButton("Export")
        self._export_btn.setObjectName("WisprPillBtn")
        self._export_btn.setFixedHeight(28)
        self._export_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self._copy_md_btn = QPushButton()
        self._copy_md_btn.setObjectName("WisprChromeBtn")
        self._copy_md_btn.setFixedSize(28, 28)
        self._copy_md_btn.setIconSize(QSize(14, 14))
        self._copy_md_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self._min_btn = QPushButton()
        self._min_btn.setObjectName("windowChromeBtn")
        self._min_btn.setFixedSize(28, 28)
        self._min_btn.setIconSize(QSize(14, 14))
        self._min_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self._max_btn = QPushButton()
        self._max_btn.setObjectName("windowChromeBtn")
        self._max_btn.setFixedSize(28, 28)
        self._max_btn.setIconSize(QSize(14, 14))
        self._max_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self._close_btn = QPushButton()
        self._close_btn.setObjectName("windowCloseBtn")
        self._close_btn.setFixedSize(28, 28)
        self._close_btn.setIconSize(QSize(14, 14))
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        top_lay.addWidget(self._back_btn)
        top_lay.addStretch(1)
        top_lay.addWidget(self._more_btn)
        top_lay.addWidget(self._export_btn)
        top_lay.addWidget(self._copy_md_btn)
        top_lay.addSpacing(2)
        top_lay.addWidget(self._min_btn)
        top_lay.addWidget(self._max_btn)
        top_lay.addWidget(self._close_btn)
        shell_lay.addWidget(self._topbar)

        self.workspace = MeetingNoteWorkspace(
            storage,
            llm_factory=llm_factory,
            lang=self._lang,
            theme=self._theme,
            show_bottom_start_when_idle=False,
        )
        shell_lay.addWidget(self.workspace, 1)

        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 4, 2)
        grip_row.addStretch(1)
        grip_row.addWidget(QSizeGrip(self))
        shell_lay.addLayout(grip_row)

        self._copy_md_reset_timer = QTimer(self)
        self._copy_md_reset_timer.setSingleShot(True)
        self._copy_md_reset_timer.setInterval(1200)
        self._copy_md_reset_timer.timeout.connect(self._reset_copy_md_icon)

        self._back_btn.clicked.connect(self.back_to_hub_requested.emit)
        self._more_btn.clicked.connect(self._on_more_clicked)
        self._export_btn.clicked.connect(self._on_export_clicked)
        self._copy_md_btn.clicked.connect(self._on_copy_md_clicked)
        self._min_btn.clicked.connect(self.showMinimized)
        self._max_btn.clicked.connect(self._toggle_maximize)
        self._close_btn.clicked.connect(self.hide)

        self._frameless_helper = FramelessWindowHelper(
            self,
            self._shell,
            titlebar_height=46,
            on_maximized_changed=self._on_maximized_changed,
        )

        self.resize(500, 820)
        self.setMinimumSize(400, 560)
        self.set_theme(self._theme)
        self.retranslate(self._lang)

    def _refresh_chrome_icons(self) -> None:
        t = get_theme_tokens(self._theme)
        fg = t["text_secondary"]
        self._back_btn.setIcon(render_icon("back", fg, 14))
        self._more_btn.setIcon(render_icon("more", fg, 14))
        self._copy_md_btn.setIcon(render_icon("copy", fg, 14))
        self._min_btn.setIcon(render_icon("minimize", fg, 14))
        self._max_btn.setIcon(render_icon("restore" if self.isMaximized() else "maximize", fg, 14))
        self._close_btn.setIcon(render_icon("close", fg, 14))

    def _reset_copy_md_icon(self) -> None:
        t = get_theme_tokens(self._theme)
        self._copy_md_btn.setIcon(render_icon("copy", t["text_secondary"], 14))

    def _on_maximized_changed(self, is_max: bool) -> None:
        t = get_theme_tokens(self._theme)
        self._max_btn.setIcon(render_icon("restore" if is_max else "maximize", t["text_secondary"], 14))

    def set_theme(self, theme: ResolvedTheme) -> None:
        self._theme = theme
        t = get_theme_tokens(theme)

        self._shell.setStyleSheet(
            f"""
            QFrame#WisprSideShell {{
                background-color: {t["shell_bg"]};
                border: 1px solid {t["shell_border"]};
                border-radius: 14px;
                color: {t["text"]};
            }}
            QFrame#WisprSideShell[windowMaximized="true"] {{
                border: none;
                border-radius: 0px;
            }}
            QPushButton#WisprChromeBtn {{
                background-color: {t["chrome_bg"]};
                border: 1px solid {t["shell_border"]};
                border-radius: 7px;
                color: {t["text"]};
            }}
            QPushButton#WisprChromeBtn:hover {{
                background-color: {t["chrome_hover"]};
            }}
            QPushButton#WisprPillBtn {{
                background-color: {t["chip_bg"]};
                border: 1px solid {t["chip_border"]};
                border-radius: 8px;
                color: {t["text_strong"]};
                padding: 4px 10px;
                font-size: 12px;
            }}
            QPushButton#WisprPillBtn:hover {{
                background-color: {t["hover_bg"]};
            }}
            QPushButton#windowChromeBtn, QPushButton#windowCloseBtn {{
                background-color: transparent;
                border: none;
                border-radius: 7px;
                color: {t["text_secondary"]};
            }}
            QPushButton#windowChromeBtn:hover {{
                background-color: {t["chrome_hover"]};
            }}
            QPushButton#windowCloseBtn:hover {{
                background-color: {t["close_hover_bg"]};
                color: {t["close_hover_fg"]};
            }}
            """
        )
        self._refresh_chrome_icons()
        self.workspace.set_theme(theme)

    def retranslate(self, lang: str) -> None:
        self._lang = normalize_language(lang)
        self._export_btn.setText(tr("notetaker.btn.export", self._lang))
        self.workspace.retranslate(self._lang)

    def dock_right_side(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        w = min(500, max(420, avail.width() - 40))
        h = min(840, max(580, avail.height() - 44))
        self.resize(w, h)
        x = avail.right() - w - 16
        y = avail.top() + max(12, (avail.height() - h) // 2)
        self.move(x, y)
        self._has_positioned = True

    def open_for_meeting(self, meeting_id: int, *, recording: bool = False) -> None:
        self.workspace.set_recording_state(recording, meeting_id if recording else None)
        self.workspace.set_meeting(meeting_id, select_tab=(1 if recording else None))
        if not self._has_positioned:
            self.dock_right_side()
        self.showNormal()
        self._frameless_helper.sync_maximized_state()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()
        self.hide()

    def _toggle_maximize(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
        self._frameless_helper.sync_maximized_state()

    def _on_export_clicked(self) -> None:
        mid = self.workspace.current_meeting_id
        if mid is None:
            return
        self.workspace.flush_pending_saves()
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

    def _on_copy_md_clicked(self) -> None:
        mid = self.workspace.current_meeting_id
        if mid is None:
            return
        self.workspace.flush_pending_saves()
        md = self._storage.export_markdown(mid)
        if md:
            QApplication.clipboard().setText(md)
            t = get_theme_tokens(self._theme)
            self._copy_md_btn.setIcon(render_icon("check", t["accent"], 14))
            self._copy_md_reset_timer.start()

    def _on_more_clicked(self) -> None:
        mid = self.workspace.current_meeting_id
        if mid is None:
            return
        menu = QMenu(self)
        menu.setStyleSheet(build_menu_stylesheet(self._theme))
        regen_act = menu.addAction(tr("notetaker.btn.regenerate", self._lang))
        force_act = menu.addAction(tr("notetaker.summary.force", self._lang))
        menu.addSeparator()
        del_act = menu.addAction(tr("notetaker.btn.delete", self._lang))
        del_act.setEnabled(not self.workspace._is_recording)

        chosen = menu.exec(self.cursor().pos())
        if chosen == regen_act:
            self.workspace.regenerate_summary_requested.emit(mid, False)
        elif chosen == force_act:
            self.workspace.regenerate_summary_requested.emit(mid, True)
        elif chosen == del_act:
            self._storage.delete_meeting(mid, hard=False)
            self.workspace.delete_meeting_requested.emit(mid)
            self.hide()


class ShortMeetingModal(QDialog):
    """Wispr Flow-style 'Started by mistake?' floating modal for ultra-short/empty meetings."""

    def __init__(
        self,
        *,
        lang: LanguageCode = DEFAULT_LANGUAGE,
        theme: ResolvedTheme = "dark",
        anchor_widget: QWidget | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Dialog,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setModal(True)
        self._lang: LanguageCode = normalize_language(lang)
        self._theme: ResolvedTheme = theme
        self.choice: str = "keep"

        self.setFixedWidth(400)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._card = QFrame()
        self._card.setObjectName("ShortMeetingCard")
        outer.addWidget(self._card)

        card_lay = QVBoxLayout(self._card)
        card_lay.setContentsMargins(20, 18, 20, 18)
        card_lay.setSpacing(12)

        # Header row: warning icon + title + circular close button
        hdr = QHBoxLayout()
        hdr.setSpacing(10)

        self._warn_lbl = QLabel()
        self._warn_lbl.setFixedSize(20, 20)
        hdr.addWidget(self._warn_lbl, 0, Qt.AlignmentFlag.AlignVCenter)

        self._title_lbl = QLabel(tr("notetaker.short_modal.title", self._lang))
        self._title_lbl.setObjectName("ShortModalTitle")
        hdr.addWidget(self._title_lbl, 1)

        self._close_btn = QPushButton()
        self._close_btn.setObjectName("ShortModalClose")
        self._close_btn.setFixedSize(28, 28)
        self._close_btn.setIconSize(QSize(12, 12))
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.clicked.connect(self.reject)
        hdr.addWidget(self._close_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        card_lay.addLayout(hdr)

        # Body message
        self._body_lbl = QLabel(tr("notetaker.short_modal.body", self._lang))
        self._body_lbl.setObjectName("ShortModalBody")
        self._body_lbl.setWordWrap(True)
        card_lay.addWidget(self._body_lbl)

        # Action buttons row
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)
        btn_row.setSpacing(8)
        btn_row.addStretch(1)

        self._discard_btn = QPushButton(tr("notetaker.short_modal.discard", self._lang))
        self._discard_btn.setObjectName("ShortModalDiscard")
        self._discard_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._discard_btn.clicked.connect(self._on_discard)

        self._keep_btn = QPushButton(tr("notetaker.short_modal.keep", self._lang))
        self._keep_btn.setObjectName("ShortModalKeep")
        self._keep_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._keep_btn.clicked.connect(self._on_keep)
        self._keep_btn.setDefault(True)

        btn_row.addWidget(self._discard_btn)
        btn_row.addWidget(self._keep_btn)
        card_lay.addLayout(btn_row)

        self._apply_styles()
        self.adjustSize()
        self._position_near(anchor_widget)

    def _on_discard(self) -> None:
        self.choice = "discard"
        self.accept()

    def _on_keep(self) -> None:
        self.choice = "keep"
        self.accept()

    def _apply_styles(self) -> None:
        is_light = self._theme == "light"
        bg = "#ffffff" if is_light else "#0c0c0e"
        border = "#d4d4d8" if is_light else "#26262b"
        title_fg = "#111113" if is_light else "#fafafa"
        body_fg = "#52525b" if is_light else "#a1a1aa"
        close_bg = "#f4f4f6" if is_light else "#141417"
        close_border = "#d4d4d8" if is_light else "#303036"
        close_hover = "#e4e4e7" if is_light else "#25252b"
        discard_bg = "#f4f4f6" if is_light else "#1d1d21"
        discard_hover = "#e4e4e7" if is_light else "#2a2a30"
        discard_fg = "#18181b" if is_light else "#fafafa"
        keep_bg = "#18181b" if is_light else "#efeee9"
        keep_hover = "#27272a" if is_light else "#ffffff"
        keep_fg = "#ffffff" if is_light else "#111113"

        self._warn_lbl.setPixmap(render_icon("warning", "#d97706" if is_light else "#f5c542", 20).pixmap(20, 20))
        self._close_btn.setIcon(render_icon("close", title_fg, 12))

        self._card.setStyleSheet(
            f"""
            QFrame#ShortMeetingCard {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 18px;
            }}
            QLabel#ShortModalTitle {{
                color: {title_fg};
                font-size: 14.5px;
                font-weight: 700;
                background: transparent;
                border: none;
            }}
            QLabel#ShortModalBody {{
                color: {body_fg};
                font-size: 13px;
                line-height: 1.4;
                background: transparent;
                border: none;
            }}
            QPushButton#ShortModalClose {{
                background-color: {close_bg};
                border: 1px solid {close_border};
                border-radius: 14px;
            }}
            QPushButton#ShortModalClose:hover {{
                background-color: {close_hover};
            }}
            QPushButton#ShortModalDiscard {{
                background-color: {discard_bg};
                color: {discard_fg};
                border: none;
                border-radius: 10px;
                padding: 8px 18px;
                font-size: 12.5px;
                font-weight: 600;
            }}
            QPushButton#ShortModalDiscard:hover {{
                background-color: {discard_hover};
            }}
            QPushButton#ShortModalKeep {{
                background-color: {keep_bg};
                color: {keep_fg};
                border: none;
                border-radius: 10px;
                padding: 8px 18px;
                font-size: 12.5px;
                font-weight: 600;
            }}
            QPushButton#ShortModalKeep:hover {{
                background-color: {keep_hover};
            }}
            """
        )

    def _position_near(self, anchor: QWidget | None) -> None:
        sz = self.sizeHint()
        w = max(self.width(), sz.width())
        h = max(self.height(), sz.height())
        screen = (
            QGuiApplication.screenAt(anchor.frameGeometry().center())
            if anchor is not None and anchor.isVisible()
            else None
        ) or QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            x = avail.x() + (avail.width() - w) // 2
            y = avail.y() + avail.height() - h - 68
            self.move(x, y)
