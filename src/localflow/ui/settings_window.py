"""Wispr Flow-inspired dark monochrome Hub window (History + Settings) for LocalFlow."""

from __future__ import annotations

from datetime import datetime
import threading

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from localflow.audio.recorder import list_input_devices
from localflow.audio.sound_fx import play_cue
from localflow.config import (
    AppConfig,
    DEFAULT_LLM_SYSTEM_PROMPT,
    apply_performance_profile,
    get_default_llm_prompt,
    is_builtin_llm_prompt,
)
from localflow.i18n import (
    DEFAULT_LANGUAGE,
    LANGUAGE_CHOICE_ITEMS,
    LanguageCode,
    normalize_language,
    tr,
)
from localflow.insights import tokenize_words
from localflow.llm.downloader import (
    DownloadCancelledError,
    DownloadProgress,
    download_embedded_llm_model,
)
from localflow.llm.model_catalog import (
    EMBEDDED_LLM_MODELS,
    RECOMMENDED_EMBEDDED_MODEL_ID,
    delete_model_file,
    get_model_badge,
    get_model_description,
    get_model_spec,
    is_llama_runtime_installed,
    is_model_downloaded,
)
from localflow.llm.ollama_polisher import OllamaPolisher
from localflow.storage import HistoryStorage
from localflow.styles import (
    DEFAULT_CATEGORY_STYLES,
    format_app_style_overrides,
    get_default_style_instruction,
    get_style_choice_items,
    get_style_label,
    is_builtin_style_instruction,
    parse_app_style_overrides,
)
from localflow.ui.insights_page import InsightsPage
from localflow.ui.theme import (
    ResolvedTheme,
    apply_windows_titlebar_theme,
    build_hub_stylesheet,
    build_menu_stylesheet,
    get_theme_choice_items,
    resolve_theme,
)

STYLE_CATEGORY_ROWS: tuple[tuple[str, str], ...] = (
    ("messenger", "style.cat.messenger"),
    ("ai_chat", "style.cat.ai_chat"),
    ("browser", "style.cat.browser"),
    ("ide_terminal", "style.cat.ide_terminal"),
    ("mail_docs", "style.cat.mail_docs"),
)


def get_profile_items(lang: str | None = None) -> list[tuple[str, str]]:
    return [
        (tr("profile.light_gpu.long", lang), "light_gpu"),
        (tr("profile.quality.long", lang), "quality"),
        (tr("profile.zero_vram.long", lang), "zero_vram"),
        (tr("profile.custom.long", lang), "custom"),
    ]


PROFILE_ITEMS = get_profile_items(DEFAULT_LANGUAGE)

PROFILE_BADGE_MAP = {
    "light_gpu": "LIGHT GPU",
    "quality": "QUALITY",
    "zero_vram": "0 VRAM",
    "custom": "CUSTOM",
}

WISPR_STYLESHEET = build_hub_stylesheet("dark")


def _build_window_icon() -> QIcon:
    pix = QPixmap(64, 64)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#181a22"))
    p.drawRoundedRect(4, 4, 56, 56, 14, 14)
    p.setBrush(QColor("#fafafa"))
    heights = (16, 30, 22, 34, 14)
    for i, h in enumerate(heights):
        x = 13.0 + i * 8.4
        y = (64.0 - h) / 2.0
        p.drawRoundedRect(QRectF(x, y, 4.4, float(h)), 2.2, 2.2)
    p.end()
    return QIcon(pix)


class _DropdownCombo(QComboBox):
    """QComboBox with a crisp anti-aliased vector chevron indicator on the right."""

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        is_light = getattr(self.window(), "_resolved_theme", "dark") == "light"
        if is_light:
            color = QColor("#52525b") if self.isEnabled() else QColor("#a1a1aa")
        else:
            color = QColor("#a1a1aa") if self.isEnabled() else QColor("#3f3f46")
        p.setPen(
            QPen(
                color,
                1.5,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        cx = float(self.width()) - 14.0
        cy = float(self.height()) / 2.0 + 1.0
        p.drawLine(QPointF(cx - 3.6, cy - 2.2), QPointF(cx, cy + 1.6))
        p.drawLine(QPointF(cx, cy + 1.6), QPointF(cx + 3.6, cy - 2.2))


class _FlowLogoGlyph(QWidget):
    """Minimalist vertical-bars brand mark inspired by Wispr Flow."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme: ResolvedTheme = "dark"
        self.setFixedSize(20, 18)

    def set_theme(self, theme: ResolvedTheme) -> None:
        self._theme = theme
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#111113") if self._theme == "light" else QColor("#fafafa"))
        heights = (8, 15, 11, 16, 7)
        for i, h in enumerate(heights):
            x = 1.0 + i * 3.8
            y = (self.height() - h) / 2.0
            p.drawRoundedRect(QRectF(x, y, 2.2, float(h)), 1.1, 1.1)


class SettingsWindow(QWidget):
    """Wispr Flow-styled Hub window with left sidebar, History + Inspector, and Settings cards."""

    config_saved = Signal(AppConfig)
    reset_pill_position_requested = Signal()
    theme_preview_requested = Signal(str)
    language_preview_requested = Signal(str)
    _models_loaded = Signal(list)
    _models_failed = Signal(str)
    _download_progress = Signal(object)
    _download_finished = Signal(str, bool)  # model_id, also_switch_whisper
    _download_failed = Signal(str)

    def __init__(self, config: AppConfig, storage: HistoryStorage) -> None:
        super().__init__()
        self._config = config
        self._storage = storage
        self._lang: LanguageCode = normalize_language(config.language)
        self._resolved_theme: ResolvedTheme = resolve_theme(config.theme)
        self._i18n_texts: list[tuple[QWidget, str]] = []
        self._i18n_placeholders: list[tuple[QWidget, str]] = []
        self._i18n_tooltips: list[tuple[QWidget, str]] = []
        self._cached_models: list[str] = []
        self._history_entries = []
        self._history_filter_mode = "all"  # "all" | "llm" | "raw"
        self._pending_style_instructions: dict[str, str] = dict(config.style_custom_instructions)
        self._editing_style_key: str = "messenger_alive"
        self._download_cancel_event: threading.Event | None = None
        self._download_active = False
        self._models_loaded.connect(self._on_models_loaded)
        self._models_failed.connect(self._on_models_failed)
        self._download_progress.connect(self._on_download_progress)
        self._download_finished.connect(self._on_download_finished)
        self._download_failed.connect(self._on_download_failed)

        self.setWindowTitle(tr("window.title", self._lang))
        self.setWindowIcon(_build_window_icon())
        self.resize(960, 630)
        self.setMinimumSize(680, 340)

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self._sidebar = self._build_sidebar()
        root.addWidget(self._sidebar)

        # Elevated rounded main workspace canvas (Wispr Flow style)
        canvas = QFrame()
        canvas.setObjectName("contentCanvas")
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)

        self._tabs = QStackedWidget()
        self._history_tab = self._build_history_tab()
        self._insights_tab = InsightsPage()
        self._vocab_tab = self._build_vocab_tab()
        self._style_tab = self._build_style_tab()
        self._settings_tab = self._build_settings_tab()
        self._tabs.addWidget(self._history_tab)
        self._tabs.addWidget(self._insights_tab)
        self._tabs.addWidget(self._vocab_tab)
        self._tabs.addWidget(self._style_tab)
        self._tabs.addWidget(self._settings_tab)
        self._tabs.currentChanged.connect(self._sync_sidebar_for_stack_index)
        canvas_layout.addWidget(self._tabs, stretch=1)

        self._config_footer = self._build_config_footer()
        self._config_footer.setVisible(False)
        canvas_layout.addWidget(self._config_footer)

        root.addWidget(canvas, stretch=1)
        self.apply_theme(self._config.theme)
        self.apply_language(self._config.language)

    def _bind_text(self, widget: QWidget, key: str) -> None:
        self._i18n_texts.append((widget, key))

    def _bind_placeholder(self, widget: QWidget, key: str) -> None:
        self._i18n_placeholders.append((widget, key))

    def _bind_tooltip(self, widget: QWidget, key: str) -> None:
        self._i18n_tooltips.append((widget, key))

    def _add_form_row(self, form: QFormLayout, label_key: str, field) -> QLabel:
        lbl = QLabel(tr(label_key, self._lang))
        self._bind_text(lbl, label_key)
        form.addRow(lbl, field)
        return lbl

    @staticmethod
    def _repopulate_data_combo(combo: QComboBox, items: list[tuple[str, str]] | tuple[tuple[str, str], ...]) -> None:
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for label, key in items:
            combo.addItem(label, key)
        idx = combo.findData(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        apply_windows_titlebar_theme(self, self._resolved_theme)

    def apply_theme(self, theme_mode: str | None = None) -> ResolvedTheme:
        resolved = resolve_theme(theme_mode if theme_mode is not None else self._config.theme)
        self._resolved_theme = resolved
        self.setStyleSheet(build_hub_stylesheet(resolved))
        self._logo_glyph.set_theme(resolved)
        self._insights_tab.set_theme(resolved)
        self._update_status_card(self._config)
        for combo in self.findChildren(_DropdownCombo):
            combo.update()
        if self.isVisible():
            apply_windows_titlebar_theme(self, resolved)
        return resolved

    def apply_language(self, lang: str | None = None) -> LanguageCode:
        new_lang = normalize_language(lang if lang is not None else self._config.language)
        prev_lang = self._lang
        self._lang = new_lang
        self.setWindowTitle(tr("window.title", new_lang))

        for widget, key in self._i18n_texts:
            widget.setText(tr(key, new_lang))  # type: ignore[attr-defined]
        for widget, key in self._i18n_placeholders:
            widget.setPlaceholderText(tr(key, new_lang))  # type: ignore[attr-defined]
        for widget, key in self._i18n_tooltips:
            widget.setToolTip(tr(key, new_lang))

        # Repopulate localized dropdowns preserving selected data values
        self._repopulate_data_combo(self._profile_combo, get_profile_items(new_lang))
        self._repopulate_data_combo(
            self._llm_backend_combo,
            [
                (tr("settings.llm_backend_embedded", new_lang), "embedded"),
                (tr("settings.llm_backend_ollama", new_lang), "ollama"),
            ],
        )
        self._repopulate_data_combo(self._theme_combo, get_theme_choice_items(new_lang))

        style_items = get_style_choice_items(new_lang)
        for combo in self._cat_style_combos.values():
            self._repopulate_data_combo(combo, style_items)
        self._repopulate_data_combo(
            self._edit_style_combo,
            [(lbl, key) for lbl, key in style_items if key != "raw"],
        )

        # Update spinbox suffixes & dynamic lists
        self._double_tap_spin.setSuffix(tr("settings.ms_suffix", new_lang))
        self._silence_auto_stop_spin.setSuffix(tr("settings.sec_suffix", new_lang))

        selected_mic = self._mic_combo.currentData()
        self._populate_mics(select_data=selected_mic)
        self._refresh_embedded_model_items()

        # Swap default LLM prompt / style instruction if not customized by user
        if prev_lang != new_lang and is_builtin_llm_prompt(self._prompt_edit.toPlainText()):
            self._prompt_edit.setPlainText(get_default_llm_prompt(new_lang))
        if self._editing_style_key not in self._pending_style_instructions:
            self._load_style_instruction_editor(self._editing_style_key)

        self._insights_tab.set_language(new_lang)
        self._update_status_card(self._config)
        self._update_history_stats_label()
        if self._history_entries:
            self._apply_history_filter(self._history_search.text())
        else:
            self._inspector_title.setText(tr("history.inspector_title", new_lang))
            self._inspector_meta.setText(tr("history.inspector_empty_meta", new_lang))
            self._delete_selected_btn.setText(tr("history.delete", new_lang))
        return new_lang

    # ------------------------------------------------------------------
    # Left Navigation Sidebar (Wispr Flow Hub aesthetic)
    # ------------------------------------------------------------------
    def _build_sidebar(self) -> QFrame:
        sb = QFrame()
        sb.setObjectName("sidebar")
        sb.setFixedWidth(212)
        layout = QVBoxLayout(sb)
        layout.setContentsMargins(6, 10, 6, 6)
        layout.setSpacing(4)

        # Clean brand header (matching "|||| Flow" in Wispr Flow)
        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(10, 2, 6, 16)
        brand_row.setSpacing(9)
        self._logo_glyph = _FlowLogoGlyph()
        brand_row.addWidget(self._logo_glyph)
        brand_title = QLabel("LocalFlow")
        brand_title.setObjectName("brandTitle")
        brand_row.addWidget(brand_title)
        brand_row.addStretch()
        layout.addLayout(brand_row)

        # Navigation buttons
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)

        self._nav_history = self._make_nav_button(tr("nav.history", self._lang), checked=True)
        self._nav_insights = self._make_nav_button(tr("nav.insights", self._lang))
        self._nav_vocab = self._make_nav_button(tr("nav.vocab", self._lang))
        self._nav_style = self._make_nav_button(tr("nav.style", self._lang))
        self._nav_settings = self._make_nav_button(tr("nav.settings", self._lang))
        self._bind_text(self._nav_history, "nav.history")
        self._bind_text(self._nav_insights, "nav.insights")
        self._bind_text(self._nav_vocab, "nav.vocab")
        self._bind_text(self._nav_style, "nav.style")
        self._bind_text(self._nav_settings, "nav.settings")

        self._nav_history.clicked.connect(lambda: self._navigate_to("history"))
        self._nav_insights.clicked.connect(lambda: self._navigate_to("insights"))
        self._nav_vocab.clicked.connect(lambda: self._navigate_to("vocab"))
        self._nav_style.clicked.connect(lambda: self._navigate_to("style"))
        self._nav_settings.clicked.connect(lambda: self._navigate_to("settings"))

        for btn in (
            self._nav_history,
            self._nav_insights,
            self._nav_vocab,
            self._nav_style,
            self._nav_settings,
        ):
            self._nav_group.addButton(btn)
            layout.addWidget(btn)

        layout.addStretch()

        # Block 1: Active Profile card
        self._profile_status_card = QFrame()
        self._profile_status_card.setObjectName("statusCard")
        sc_layout = QVBoxLayout(self._profile_status_card)
        sc_layout.setContentsMargins(12, 10, 12, 10)
        sc_layout.setSpacing(5)

        sc_hdr_row = QHBoxLayout()
        sc_hdr_row.setContentsMargins(0, 0, 0, 2)
        sc_title = QLabel(tr("sidebar.profile_hdr", self._lang))
        sc_title.setObjectName("sectionHeader")
        self._bind_text(sc_title, "sidebar.profile_hdr")
        sc_hdr_row.addWidget(sc_title)
        sc_hdr_row.addStretch()
        self._brand_badge = QLabel(PROFILE_BADGE_MAP.get(self._config.performance_profile, "LOCAL"))
        self._brand_badge.setObjectName("brandBadge")
        sc_hdr_row.addWidget(self._brand_badge)
        sc_layout.addLayout(sc_hdr_row)

        self._status_asr_lbl = QLabel("")
        self._status_asr_lbl.setObjectName("statusItem")
        sc_layout.addWidget(self._status_asr_lbl)

        self._status_llm_lbl = QLabel("")
        self._status_llm_lbl.setObjectName("statusItem")
        sc_layout.addWidget(self._status_llm_lbl)
        layout.addWidget(self._profile_status_card)

        layout.addSpacing(4)

        # Block 2: Hotkeys card (all shortcuts)
        self._hotkeys_card = QFrame()
        self._hotkeys_card.setObjectName("statusCard")
        hk_layout = QVBoxLayout(self._hotkeys_card)
        hk_layout.setContentsMargins(12, 10, 12, 10)
        hk_layout.setSpacing(4)

        hk_title = QLabel(tr("sidebar.hotkeys_hdr", self._lang))
        hk_title.setObjectName("sectionHeader")
        self._bind_text(hk_title, "sidebar.hotkeys_hdr")
        hk_layout.addWidget(hk_title)

        for shortcut_key in (
            "sidebar.hk_start",
            "sidebar.hk_stop",
            "sidebar.hk_cancel",
            "sidebar.hk_command",
        ):
            lbl = QLabel(tr(shortcut_key, self._lang))
            lbl.setObjectName("mutedMeta")
            self._bind_text(lbl, shortcut_key)
            hk_layout.addWidget(lbl)

        layout.addWidget(self._hotkeys_card)
        return sb

    def _make_nav_button(self, text: str, *, checked: bool = False) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("navBtn")
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn

    def _navigate_to(self, target: str) -> None:
        if target == "history":
            self._tabs.setCurrentWidget(self._history_tab)
        elif target == "insights":
            self.refresh_insights()
            self._tabs.setCurrentWidget(self._insights_tab)
        elif target == "vocab":
            self._tabs.setCurrentWidget(self._vocab_tab)
        elif target == "style":
            self._tabs.setCurrentWidget(self._style_tab)
        else:
            self._tabs.setCurrentWidget(self._settings_tab)

    def _sync_sidebar_for_stack_index(self, index: int) -> None:
        w = self._tabs.widget(index)
        if w is self._history_tab:
            self._nav_history.setChecked(True)
        elif w is self._insights_tab:
            self._nav_insights.setChecked(True)
        elif w is self._vocab_tab:
            self._nav_vocab.setChecked(True)
        elif w is self._style_tab:
            self._nav_style.setChecked(True)
        else:
            self._nav_settings.setChecked(True)
        self._config_footer.setVisible(
            w in (self._vocab_tab, self._style_tab, self._settings_tab)
        )

    def refresh_insights(self) -> None:
        self._insights_tab.refresh(self._storage.list_all())

    def _update_status_card(self, cfg: AppConfig) -> None:
        is_light = self._resolved_theme == "light"
        asr_dot = "#059669" if is_light else "#10b981"
        llm_dot = "#7c3aed" if is_light else "#a78bfa"
        self._brand_badge.setText(PROFILE_BADGE_MAP.get(cfg.performance_profile, "LOCAL"))
        self._status_asr_lbl.setText(
            f"<span style='color:{asr_dot};'>●</span>  {tr('sidebar.asr_status', self._lang, model=cfg.whisper_model)}"
        )
        if cfg.llm_enabled:
            if cfg.llm_backend == "embedded":
                spec = get_model_spec(cfg.embedded_llm_model_id)
                llm_label = spec.id if spec is not None else cfg.embedded_llm_model_id
                self._status_llm_lbl.setText(
                    f"<span style='color:{llm_dot};'>●</span>  {tr('sidebar.llm_local', self._lang, model=llm_label)}"
                )
            else:
                self._status_llm_lbl.setText(
                    f"<span style='color:{llm_dot};'>●</span>  {tr('sidebar.llm_ollama', self._lang, model=cfg.ollama_model)}"
                )
        else:
            self._status_llm_lbl.setText(
                f"<span style='color:#71717a;'>○</span>  {tr('sidebar.llm_off', self._lang)}"
            )

    # ------------------------------------------------------------------
    # History Page (Main List + Right Detail Inspector like Wispr Flow)
    # ------------------------------------------------------------------
    def _build_history_tab(self) -> QWidget:
        w = QWidget()
        root = QHBoxLayout(w)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Left/center column: feed
        feed = QWidget()
        layout = QVBoxLayout(feed)
        layout.setContentsMargins(24, 20, 20, 18)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel(tr("history.title", self._lang))
        title.setObjectName("pageTitle")
        self._bind_text(title, "history.title")
        header.addWidget(title)
        header.addStretch()

        refresh_btn = QPushButton(tr("history.refresh", self._lang))
        refresh_btn.setObjectName("secondary")
        self._bind_text(refresh_btn, "history.refresh")
        refresh_btn.clicked.connect(self.refresh_history)
        header.addWidget(refresh_btn)

        self._delete_selected_btn = QPushButton(tr("history.delete", self._lang))
        self._delete_selected_btn.setObjectName("secondary")
        self._delete_selected_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_selected_btn.setEnabled(False)
        self._delete_selected_btn.setToolTip(tr("history.delete_tooltip", self._lang))
        self._bind_tooltip(self._delete_selected_btn, "history.delete_tooltip")
        self._delete_selected_btn.clicked.connect(self._delete_selected_history)
        header.addWidget(self._delete_selected_btn)

        self._clear_history_btn = QPushButton(tr("history.clear", self._lang))
        self._clear_history_btn.setObjectName("ghostBtn")
        self._clear_history_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._bind_text(self._clear_history_btn, "history.clear")
        self._clear_history_btn.clicked.connect(self._clear_history)
        header.addWidget(self._clear_history_btn)
        layout.addLayout(header)

        # Underline filter bar + inline search (mirroring Past notes | Shared | Search)
        filter_row = QHBoxLayout()
        filter_row.setSpacing(0)
        self._filter_group = QButtonGroup(self)
        self._filter_group.setExclusive(True)

        for key, label_key in (
            ("all", "history.filter_all"),
            ("llm", "history.filter_llm"),
            ("raw", "history.filter_raw"),
        ):
            btn = QPushButton(tr(label_key, self._lang))
            btn.setObjectName("filterTab")
            btn.setCheckable(True)
            btn.setChecked(key == "all")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._bind_text(btn, label_key)
            btn.clicked.connect(lambda checked, m=key: checked and self._set_history_filter_mode(m))
            self._filter_group.addButton(btn)
            filter_row.addWidget(btn)

        filter_row.addStretch()

        self._history_search = QLineEdit()
        self._history_search.setPlaceholderText(tr("history.search_placeholder", self._lang))
        self._bind_placeholder(self._history_search, "history.search_placeholder")
        self._history_search.setClearButtonEnabled(True)
        self._history_search.setFixedWidth(185)
        self._history_search.textChanged.connect(self._apply_history_filter)
        filter_row.addWidget(self._history_search)
        layout.addLayout(filter_row)

        divider = QFrame()
        divider.setObjectName("divider")
        layout.addWidget(divider)

        self._history_stats_label = QLabel(tr("history.stats", self._lang, count=0, words=0))
        self._history_stats_label.setObjectName("sectionHeader")
        layout.addWidget(self._history_stats_label)

        self._history_list = QListWidget()
        self._history_list.setObjectName("historyList")
        self._history_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._history_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._history_list.setWordWrap(True)
        self._history_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._history_list.setSpacing(2)
        self._history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._history_list.customContextMenuRequested.connect(self._on_history_context_menu)
        self._history_list.itemDoubleClicked.connect(self._copy_selected_history)
        self._history_list.currentItemChanged.connect(self._on_history_selected)
        self._history_list.itemSelectionChanged.connect(self._on_history_selection_changed)
        del_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self._history_list)
        del_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        del_shortcut.activated.connect(self._delete_selected_history)
        layout.addWidget(self._history_list, stretch=1)

        root.addWidget(feed, stretch=3)

        # Right inspector panel (matching Wispr Flow right column)
        inspector = QFrame()
        inspector.setObjectName("inspectorPane")
        inspector.setFixedWidth(268)
        ins_layout = QVBoxLayout(inspector)
        ins_layout.setContentsMargins(18, 22, 18, 18)
        ins_layout.setSpacing(10)

        self._inspector_title = QLabel(tr("history.inspector_title", self._lang))
        self._inspector_title.setObjectName("inspectorTitle")
        ins_layout.addWidget(self._inspector_title)

        self._inspector_meta = QLabel(tr("history.inspector_empty_meta", self._lang))
        self._inspector_meta.setObjectName("mutedMeta")
        self._inspector_meta.setWordWrap(True)
        ins_layout.addWidget(self._inspector_meta)

        ins_div = QFrame()
        ins_div.setObjectName("divider")
        ins_layout.addWidget(ins_div)

        overview_hdr = QLabel(tr("history.inspector_overview", self._lang))
        overview_hdr.setObjectName("sectionHeader")
        self._bind_text(overview_hdr, "history.inspector_overview")
        ins_layout.addWidget(overview_hdr)

        self._raw_preview = QTextEdit()
        self._raw_preview.setObjectName("inspectorText")
        self._raw_preview.setReadOnly(True)
        self._raw_preview.setPlaceholderText(tr("history.inspector_placeholder", self._lang))
        self._bind_placeholder(self._raw_preview, "history.inspector_placeholder")
        ins_layout.addWidget(self._raw_preview, stretch=1)

        copy_btn = QPushButton(tr("history.copy_text", self._lang))
        copy_btn.setObjectName("primaryBtn")
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._bind_text(copy_btn, "history.copy_text")
        copy_btn.clicked.connect(self._copy_selected_history)
        ins_layout.addWidget(copy_btn)

        root.addWidget(inspector)
        return w

    def _set_history_filter_mode(self, mode: str) -> None:
        self._history_filter_mode = mode
        self._apply_history_filter(self._history_search.text())

    def _update_history_stats_label(self) -> None:
        total_words = sum(len(tokenize_words(e.final_text)) for e in self._history_entries)
        self._history_stats_label.setText(
            tr("history.stats", self._lang, count=len(self._history_entries), words=total_words)
        )

    def refresh_history(self) -> None:
        self._history_entries = self._storage.list_recent(limit=200)
        self._clear_history_btn.setEnabled(bool(self._history_entries))
        self._update_history_stats_label()
        self._apply_history_filter(self._history_search.text())

    def _apply_history_filter(self, query: str) -> None:
        self._history_list.clear()
        self._raw_preview.clear()
        self._inspector_title.setText(tr("history.inspector_title", self._lang))
        self._inspector_meta.setText(tr("history.inspector_empty_meta", self._lang))
        self._delete_selected_btn.setEnabled(False)
        self._delete_selected_btn.setText(tr("history.delete", self._lang))
        needle = query.strip().lower()
        for entry in self._history_entries:
            if self._history_filter_mode == "llm" and not entry.used_llm:
                continue
            if self._history_filter_mode == "raw" and entry.used_llm:
                continue
            if needle and needle not in entry.final_text.lower() and needle not in entry.raw_text.lower():
                continue
            try:
                dt = datetime.fromisoformat(entry.created_at).astimezone()
                ts = dt.strftime("%d.%m %H:%M")
            except Exception:
                ts = entry.created_at[:16]
            badge = "✨ LLM" if entry.used_llm else "RAW"
            style_lbl = get_style_label(entry.style_profile, self._lang)
            meta_parts = [ts, f"{entry.duration_sec:.1f}s", badge]
            if style_lbl and entry.style_profile not in ("default", "raw"):
                meta_parts.append(style_lbl)
            elif entry.app_name:
                meta_parts.append(entry.app_name)
            preview = " ".join(entry.final_text.split())
            if len(preview) > 84:
                preview = preview[:83].rstrip() + "…"
            item = QListWidgetItem(f"{preview}\n" + "  ·  ".join(meta_parts))
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self._history_list.addItem(item)
        if self._history_list.count() > 0:
            self._history_list.setCurrentRow(0)

    def _on_history_selection_changed(self) -> None:
        selected = self._history_list.selectedItems()
        count = len(selected)
        self._delete_selected_btn.setEnabled(count > 0)
        self._delete_selected_btn.setText(
            tr("history.delete_count", self._lang, count=count)
            if count > 1
            else tr("history.delete", self._lang)
        )

        if count == 0:
            self._raw_preview.clear()
            self._inspector_title.setText(tr("history.inspector_title", self._lang))
            self._inspector_meta.setText(tr("history.inspector_empty_meta", self._lang))
            return

        if count > 1:
            entries = [item.data(Qt.ItemDataRole.UserRole) for item in selected]
            total_sec = sum(e.duration_sec for e in entries)
            total_words = sum(len(tokenize_words(e.final_text)) for e in entries)
            self._inspector_title.setText(tr("history.multi_title", self._lang, count=count))
            self._inspector_meta.setText(
                tr("history.multi_meta", self._lang, words=total_words, sec=total_sec)
            )
            self._raw_preview.setPlainText(tr("history.multi_hint", self._lang))
            return

        self._on_history_selected(selected[0])

    def _on_history_selected(self, current: QListWidgetItem | None) -> None:
        if len(self._history_list.selectedItems()) > 1:
            return
        if current is None:
            self._raw_preview.clear()
            self._inspector_title.setText(tr("history.inspector_title", self._lang))
            self._inspector_meta.setText(tr("history.inspector_empty_meta", self._lang))
            return
        entry = current.data(Qt.ItemDataRole.UserRole)
        try:
            dt = datetime.fromisoformat(entry.created_at).astimezone()
            ts = dt.strftime("%d.%m.%Y · %H:%M")
        except Exception:
            ts = entry.created_at[:16]
        model_info = entry.llm_model if entry.used_llm and entry.llm_model else "RAW ASR"
        self._inspector_title.setText(
            tr("history.type_command", self._lang)
            if entry.is_command
            else (
                tr("history.type_dictation", self._lang)
                if entry.used_llm
                else tr("history.type_raw", self._lang)
            )
        )
        meta_bits = [ts, f"{entry.duration_sec:.1f}s", model_info]
        if entry.app_name:
            meta_bits.append(entry.app_name)
        style_lbl = get_style_label(entry.style_profile, self._lang)
        if style_lbl:
            meta_bits.append(tr("history.meta_style", self._lang, label=style_lbl))
        self._inspector_meta.setText(" · ".join(meta_bits))
        self._raw_preview.setPlainText(
            tr("history.preview_body", self._lang, final=entry.final_text, raw=entry.raw_text)
        )

    def _on_history_context_menu(self, pos) -> None:
        item = self._history_list.itemAt(pos)
        if item is not None and not item.isSelected():
            self._history_list.setCurrentItem(item)
        selected = self._history_list.selectedItems()
        if not selected:
            return
        menu = QMenu(self)
        menu.setStyleSheet(build_menu_stylesheet(self._resolved_theme))
        if len(selected) == 1:
            copy_act = menu.addAction(tr("history.ctx_copy", self._lang))
            copy_act.triggered.connect(self._copy_selected_history)
        del_label = (
            tr("history.ctx_delete_many", self._lang, count=len(selected))
            if len(selected) > 1
            else tr("history.ctx_delete_one", self._lang)
        )
        del_act = menu.addAction(del_label)
        del_act.triggered.connect(self._delete_selected_history)
        menu.exec(self._history_list.viewport().mapToGlobal(pos))

    def _copy_selected_history(self) -> None:
        item = self._history_list.currentItem()
        if item is None:
            return
        entry = item.data(Qt.ItemDataRole.UserRole)
        QApplication.clipboard().setText(entry.final_text)

    def _confirm_delete_dialog(
        self, title: str, text: str, info_text: str, confirm_label: str
    ) -> bool:
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(text)
        box.setInformativeText(info_text)
        cancel_btn = box.addButton(tr("common.cancel", self._lang), QMessageBox.ButtonRole.RejectRole)
        confirm_btn = box.addButton(confirm_label, QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn.setObjectName("secondary")
        confirm_btn.setObjectName("dangerBtn")
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        confirm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        box.setDefaultButton(cancel_btn)
        box.setEscapeButton(cancel_btn)
        box.setStyleSheet(build_hub_stylesheet(self._resolved_theme))
        apply_windows_titlebar_theme(box, self._resolved_theme)
        box.exec()
        return box.clickedButton() is confirm_btn

    def _confirm_delete_entries(self, count: int) -> bool:
        text = (
            tr("history.confirm_delete_one", self._lang)
            if count == 1
            else tr("history.confirm_delete_many", self._lang, count=count)
        )
        return self._confirm_delete_dialog(
            tr("history.confirm_delete_title", self._lang),
            text,
            tr("history.confirm_delete_info", self._lang),
            tr("history.delete", self._lang),
        )

    def _delete_selected_history(self) -> None:
        selected = self._history_list.selectedItems()
        if not selected:
            return
        rows = sorted(self._history_list.row(item) for item in selected)
        entry_ids = [item.data(Qt.ItemDataRole.UserRole).id for item in selected]
        if not self._confirm_delete_entries(len(entry_ids)):
            return
        self._storage.delete_entries(entry_ids)
        self.refresh_history()
        self.refresh_insights()
        if self._history_list.count() > 0:
            target_row = min(rows[0], self._history_list.count() - 1)
            self._history_list.setCurrentRow(target_row)

    def _confirm_clear_history(self, count: int) -> bool:
        return self._confirm_delete_dialog(
            tr("history.confirm_clear_title", self._lang),
            tr("history.confirm_clear_text", self._lang, count=count),
            tr("history.confirm_clear_info", self._lang),
            tr("history.clear", self._lang),
        )

    def _clear_history(self) -> None:
        count = len(self._history_entries) or len(self._storage.list_all())
        if count == 0:
            return
        if not self._confirm_clear_history(count):
            return
        self._storage.clear()
        self.refresh_history()
        self.refresh_insights()

    # ------------------------------------------------------------------
    # Shared Configuration Page Helpers & Pinned Save Footer
    # ------------------------------------------------------------------
    def _configure_fluid_combo(self, combo: QComboBox, *, min_chars: int = 8) -> None:
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(min_chars)
        combo.setMinimumWidth(0)
        combo.setCursor(Qt.CursorShape.PointingHandCursor)

    @staticmethod
    def _set_combo_text(combo: QComboBox, text: str) -> None:
        val = (text or "").strip()
        if not val:
            return
        idx = combo.findText(val)
        if idx < 0:
            combo.addItem(val)
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)

    def _make_section_card(self, title_key: str) -> tuple[QFrame, QFormLayout]:
        card = QFrame()
        card.setObjectName("sectionCard")
        card.setMinimumWidth(0)
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(16, 14, 16, 14)
        vbox.setSpacing(10)

        hdr = QLabel(tr(title_key, self._lang))
        hdr.setObjectName("sectionHeader")
        self._bind_text(hdr, title_key)
        vbox.addWidget(hdr)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(9)
        vbox.addLayout(form)
        return card, form

    def _make_scroll_page(
        self, title_key: str, subtitle_key: str = ""
    ) -> tuple[QWidget, QScrollArea, QVBoxLayout]:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(24, 20, 24, 12)
        outer.setSpacing(10)

        title = QLabel(tr(title_key, self._lang))
        title.setObjectName("pageTitle")
        self._bind_text(title, title_key)
        outer.addWidget(title)

        if subtitle_key:
            subtitle = QLabel(tr(subtitle_key, self._lang))
            subtitle.setObjectName("mutedMeta")
            subtitle.setWordWrap(True)
            self._bind_text(subtitle, subtitle_key)
            outer.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        cards_container = QWidget()
        cards_container.setMinimumWidth(0)
        cards_layout = QVBoxLayout(cards_container)
        cards_layout.setContentsMargins(0, 4, 10, 8)
        cards_layout.setSpacing(12)

        scroll.setWidget(cards_container)
        outer.addWidget(scroll, stretch=1)
        return page, scroll, cards_layout

    def _build_config_footer(self) -> QFrame:
        footer = QFrame()
        footer.setObjectName("configFooter")
        vbox = QVBoxLayout(footer)
        vbox.setContentsMargins(24, 0, 24, 14)
        vbox.setSpacing(10)

        footer_div = QFrame()
        footer_div.setObjectName("divider")
        vbox.addWidget(footer_div)

        self._save_status = QLabel("")
        self._save_status.setObjectName("mutedMeta")
        self._save_btn = QPushButton(tr("settings.save_btn", self._lang))
        self._save_btn.setObjectName("primaryBtn")
        self._save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._bind_text(self._save_btn, "settings.save_btn")
        self._save_btn.clicked.connect(self._save_clicked)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addWidget(self._save_status, stretch=1)
        bottom.addWidget(self._save_btn)
        vbox.addLayout(bottom)
        return footer

    # ------------------------------------------------------------------
    # Dictionary Page (Personal Vocabulary & Replacements)
    # ------------------------------------------------------------------
    def _build_vocab_tab(self) -> QWidget:
        page, self._vocab_scroll, cards_layout = self._make_scroll_page(
            "vocab.title",
            "vocab.subtitle",
        )

        self._vocab_card, vocab_form = self._make_section_card("vocab.card_hdr")

        vocab_box = QVBoxLayout()
        vocab_box.setSpacing(6)
        self._vocab_edit = QTextEdit()
        self._vocab_edit.setMinimumWidth(0)
        self._vocab_edit.setPlainText(self._config.custom_vocabulary)
        self._vocab_edit.setMinimumHeight(230)
        self._vocab_edit.setPlaceholderText(tr("vocab.placeholder", self._lang))
        self._bind_placeholder(self._vocab_edit, "vocab.placeholder")
        vocab_box.addWidget(self._vocab_edit)

        vocab_hint = QLabel(tr("vocab.hint", self._lang))
        vocab_hint.setObjectName("mutedMeta")
        vocab_hint.setWordWrap(True)
        self._bind_text(vocab_hint, "vocab.hint")
        vocab_box.addWidget(vocab_hint)

        self._add_form_row(vocab_form, "vocab.label", vocab_box)
        cards_layout.addWidget(self._vocab_card)
        cards_layout.addStretch()
        return page

    # ------------------------------------------------------------------
    # Style & Prompt Page (App-Aware Styles + Base System Prompt)
    # ------------------------------------------------------------------
    def _build_style_tab(self) -> QWidget:
        page, self._style_scroll, cards_layout = self._make_scroll_page(
            "style.title",
            "style.subtitle",
        )

        # Card 1: App-Aware Style Profiles
        self._styles_card, styles_form = self._make_section_card("style.card_hdr")

        self._auto_style_check = QCheckBox(tr("style.auto_check", self._lang))
        self._auto_style_check.setChecked(self._config.auto_style_by_app)
        self._bind_text(self._auto_style_check, "style.auto_check")
        self._auto_style_check.toggled.connect(self._update_auto_style_controls_enabled)
        styles_form.addRow(self._auto_style_check)

        self._messenger_strip_period_check = QCheckBox(
            tr("style.strip_period_check", self._lang)
        )
        self._messenger_strip_period_check.setChecked(self._config.messenger_strip_trailing_period)
        self._bind_text(self._messenger_strip_period_check, "style.strip_period_check")
        styles_form.addRow(self._messenger_strip_period_check)

        self._cat_style_combos: dict[str, QComboBox] = {}
        style_items = get_style_choice_items(self._lang)
        for cat_key, cat_label_key in STYLE_CATEGORY_ROWS:
            combo = _DropdownCombo()
            self._configure_fluid_combo(combo, min_chars=10)
            for style_label, style_key in style_items:
                combo.addItem(style_label, style_key)
            current_style = self._config.category_styles.get(
                cat_key, DEFAULT_CATEGORY_STYLES[cat_key]
            )
            cidx = combo.findData(current_style)
            if cidx >= 0:
                combo.setCurrentIndex(cidx)
            self._cat_style_combos[cat_key] = combo
            self._add_form_row(styles_form, cat_label_key, combo)

        overrides_box = QVBoxLayout()
        overrides_box.setSpacing(4)
        self._app_overrides_edit = QTextEdit()
        self._app_overrides_edit.setMinimumWidth(0)
        self._app_overrides_edit.setMinimumHeight(74)
        self._app_overrides_edit.setMaximumHeight(110)
        self._app_overrides_edit.setPlaceholderText(
            "Antigravity = ai_prompt\nCursor = ai_prompt\nTerminal = raw"
        )
        self._app_overrides_edit.setPlainText(
            format_app_style_overrides(self._config.app_style_overrides)
        )
        overrides_box.addWidget(self._app_overrides_edit)
        overrides_hint = QLabel(tr("style.overrides_hint", self._lang))
        overrides_hint.setObjectName("mutedMeta")
        overrides_hint.setWordWrap(True)
        self._bind_text(overrides_hint, "style.overrides_hint")
        overrides_box.addWidget(overrides_hint)
        self._add_form_row(styles_form, "style.overrides_label", overrides_box)

        style_editor_box = QVBoxLayout()
        style_editor_box.setSpacing(6)
        style_sel_row = QHBoxLayout()
        style_sel_row.setSpacing(8)
        self._edit_style_combo = _DropdownCombo()
        self._configure_fluid_combo(self._edit_style_combo, min_chars=10)
        for style_label, style_key in style_items:
            if style_key != "raw":
                self._edit_style_combo.addItem(style_label, style_key)
        self._edit_style_combo.currentIndexChanged.connect(self._on_edit_style_changed)
        style_sel_row.addWidget(self._edit_style_combo, stretch=1)

        self._reset_style_btn = QPushButton(tr("style.reset_btn", self._lang))
        self._reset_style_btn.setObjectName("secondary")
        self._reset_style_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._bind_text(self._reset_style_btn, "style.reset_btn")
        self._reset_style_btn.clicked.connect(self._reset_current_style_instruction)
        style_sel_row.addWidget(self._reset_style_btn)
        style_editor_box.addLayout(style_sel_row)

        self._style_instruction_edit = QTextEdit()
        self._style_instruction_edit.setMinimumWidth(0)
        self._style_instruction_edit.setMinimumHeight(88)
        self._style_instruction_edit.setMaximumHeight(130)
        self._load_style_instruction_editor(self._editing_style_key)
        style_editor_box.addWidget(self._style_instruction_edit)
        self._add_form_row(styles_form, "style.instruction_label", style_editor_box)

        self._update_auto_style_controls_enabled(self._config.auto_style_by_app)
        cards_layout.addWidget(self._styles_card)

        # Card 2: Base System Prompt
        self._base_prompt_card, prompt_form = self._make_section_card("style.base_prompt_hdr")
        prompt_box = QVBoxLayout()
        prompt_box.setSpacing(6)
        self._prompt_edit = QTextEdit()
        self._prompt_edit.setMinimumWidth(0)
        self._prompt_edit.setPlainText(self._config.llm_system_prompt)
        self._prompt_edit.setMinimumHeight(130)
        prompt_box.addWidget(self._prompt_edit)

        reset_prompt_btn = QPushButton(tr("style.reset_prompt_btn", self._lang))
        reset_prompt_btn.setObjectName("secondary")
        reset_prompt_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._bind_text(reset_prompt_btn, "style.reset_prompt_btn")
        reset_prompt_btn.clicked.connect(
            lambda: self._prompt_edit.setPlainText(get_default_llm_prompt(self._lang))
        )
        prompt_box.addWidget(reset_prompt_btn, alignment=Qt.AlignmentFlag.AlignRight)
        self._add_form_row(prompt_form, "style.prompt_label", prompt_box)
        cards_layout.addWidget(self._base_prompt_card)

        cards_layout.addStretch()
        return page

    # ------------------------------------------------------------------
    # Settings Page (ASR, Microphone, LLM Engine & Flow Bar Behavior)
    # ------------------------------------------------------------------
    def _build_settings_tab(self) -> QWidget:
        page, self._settings_scroll, cards_layout = self._make_scroll_page(
            "settings.title",
            "settings.subtitle",
        )

        # Card 1: Performance & ASR
        perf_card, perf_form = self._make_section_card("settings.perf_hdr")

        self._profile_combo = _DropdownCombo()
        self._configure_fluid_combo(self._profile_combo, min_chars=12)
        for label, key in get_profile_items(self._lang):
            self._profile_combo.addItem(label, key)
        idx = self._profile_combo.findData(self._config.performance_profile)
        if idx >= 0:
            self._profile_combo.setCurrentIndex(idx)
        self._profile_combo.currentIndexChanged.connect(self._on_profile_combo_changed)
        self._add_form_row(perf_form, "settings.vram_label", self._profile_combo)

        whisper_row = QHBoxLayout()
        whisper_row.setSpacing(8)
        self._whisper_combo = _DropdownCombo()
        self._whisper_combo.setEditable(True)
        self._configure_fluid_combo(self._whisper_combo, min_chars=8)
        for m in ("large-v3-turbo", "large-v3", "medium", "small"):
            self._whisper_combo.addItem(m)
        self._whisper_combo.setCurrentText(self._config.whisper_model)
        whisper_row.addWidget(self._whisper_combo, stretch=2)

        self._whisper_dev_combo = _DropdownCombo()
        self._configure_fluid_combo(self._whisper_dev_combo, min_chars=4)
        for dev in ("auto", "cuda", "cpu"):
            self._whisper_dev_combo.addItem(dev)
        self._whisper_dev_combo.setCurrentText(self._config.whisper_device)
        whisper_row.addWidget(self._whisper_dev_combo, stretch=1)

        self._whisper_ct_combo = _DropdownCombo()
        self._configure_fluid_combo(self._whisper_ct_combo, min_chars=6)
        for ct in ("int8_float16", "float16", "int8", "auto"):
            self._whisper_ct_combo.addItem(ct)
        self._whisper_ct_combo.setCurrentText(self._config.whisper_compute_type)
        whisper_row.addWidget(self._whisper_ct_combo, stretch=1)
        self._add_form_row(perf_form, "settings.whisper_label", whisper_row)

        whisper_hint = QLabel(tr("settings.whisper_hint", self._lang))
        whisper_hint.setObjectName("mutedMeta")
        whisper_hint.setWordWrap(True)
        self._bind_text(whisper_hint, "settings.whisper_hint")
        perf_form.addRow("", whisper_hint)
        cards_layout.addWidget(perf_card)

        # Card 2: Audio & Hotkey
        audio_card, audio_form = self._make_section_card("settings.audio_hdr")

        self._mic_combo = _DropdownCombo()
        self._configure_fluid_combo(self._mic_combo, min_chars=12)
        self._populate_mics()
        self._add_form_row(audio_form, "settings.mic_label", self._mic_combo)

        self._double_tap_spin = QSpinBox()
        self._double_tap_spin.setRange(150, 800)
        self._double_tap_spin.setSuffix(tr("settings.ms_suffix", self._lang))
        self._double_tap_spin.setFixedWidth(110)
        self._double_tap_spin.setValue(self._config.double_tap_ms)
        self._add_form_row(audio_form, "settings.double_tap_label", self._double_tap_spin)

        silence_row = QHBoxLayout()
        silence_row.setSpacing(10)
        self._silence_auto_stop_check = QCheckBox(tr("settings.silence_check", self._lang))
        self._silence_auto_stop_check.setChecked(self._config.silence_auto_stop_enabled)
        self._bind_text(self._silence_auto_stop_check, "settings.silence_check")
        self._silence_auto_stop_spin = QDoubleSpinBox()
        self._silence_auto_stop_spin.setRange(0.5, 15.0)
        self._silence_auto_stop_spin.setSingleStep(0.5)
        self._silence_auto_stop_spin.setDecimals(1)
        self._silence_auto_stop_spin.setSuffix(tr("settings.sec_suffix", self._lang))
        self._silence_auto_stop_spin.setFixedWidth(92)
        self._silence_auto_stop_spin.setValue(self._config.silence_auto_stop_sec)
        self._silence_auto_stop_spin.setEnabled(self._config.silence_auto_stop_enabled)
        self._silence_auto_stop_check.toggled.connect(self._silence_auto_stop_spin.setEnabled)
        silence_row.addWidget(self._silence_auto_stop_check)
        silence_row.addWidget(self._silence_auto_stop_spin)
        silence_row.addStretch()
        self._add_form_row(audio_form, "settings.silence_label", silence_row)
        cards_layout.addWidget(audio_card)

        # Card 3: Smart Cleanup & LLM Engine
        self._llm_card, llm_form = self._make_section_card("settings.llm_hdr")

        self._llm_check = QCheckBox(tr("settings.llm_enable", self._lang))
        self._llm_check.setChecked(self._config.llm_enabled)
        self._bind_text(self._llm_check, "settings.llm_enable")
        llm_form.addRow(self._llm_check)

        self._llm_backend_combo = _DropdownCombo()
        self._configure_fluid_combo(self._llm_backend_combo, min_chars=12)
        self._llm_backend_combo.addItem(tr("settings.llm_backend_embedded", self._lang), "embedded")
        self._llm_backend_combo.addItem(tr("settings.llm_backend_ollama", self._lang), "ollama")
        b_idx = self._llm_backend_combo.findData(self._config.llm_backend)
        if b_idx >= 0:
            self._llm_backend_combo.setCurrentIndex(b_idx)
        self._llm_backend_combo.currentIndexChanged.connect(self._update_llm_backend_visibility)
        self._add_form_row(llm_form, "settings.llm_engine_label", self._llm_backend_combo)

        # --- Embedded llama.cpp controls ---
        self._embedded_panel = QWidget()
        self._embedded_panel.setMinimumWidth(0)
        emb_layout = QVBoxLayout(self._embedded_panel)
        emb_layout.setContentsMargins(0, 2, 0, 2)
        emb_layout.setSpacing(7)

        emb_model_row = QHBoxLayout()
        emb_model_row.setSpacing(8)
        self._embedded_model_combo = _DropdownCombo()
        self._configure_fluid_combo(self._embedded_model_combo, min_chars=12)
        self._embedded_model_combo.currentIndexChanged.connect(self._on_embedded_model_changed)
        emb_model_row.addWidget(self._embedded_model_combo, stretch=1)

        self._download_model_btn = QPushButton(tr("settings.download_model", self._lang))
        self._download_model_btn.setObjectName("secondary")
        self._download_model_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._download_model_btn.clicked.connect(self._download_selected_model_clicked)
        emb_model_row.addWidget(self._download_model_btn)

        self._delete_model_btn = QPushButton(tr("history.delete", self._lang))
        self._delete_model_btn.setObjectName("ghostBtn")
        self._delete_model_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._bind_text(self._delete_model_btn, "history.delete")
        self._delete_model_btn.clicked.connect(self._delete_selected_model_clicked)
        emb_model_row.addWidget(self._delete_model_btn)
        emb_layout.addLayout(emb_model_row)

        self._embedded_model_desc = QLabel("")
        self._embedded_model_desc.setObjectName("mutedMeta")
        self._embedded_model_desc.setWordWrap(True)
        emb_layout.addWidget(self._embedded_model_desc)

        rec_row = QHBoxLayout()
        rec_row.setSpacing(8)
        self._download_recommended_btn = QPushButton(tr("settings.download_recommended", self._lang))
        self._download_recommended_btn.setObjectName("secondary")
        self._download_recommended_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._download_recommended_btn.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self._bind_text(self._download_recommended_btn, "settings.download_recommended")
        self._download_recommended_btn.setToolTip(tr("settings.download_recommended_tip", self._lang))
        self._bind_tooltip(self._download_recommended_btn, "settings.download_recommended_tip")
        self._download_recommended_btn.clicked.connect(self._download_recommended_clicked)
        rec_row.addWidget(self._download_recommended_btn)

        self._browse_gguf_btn = QPushButton(tr("settings.browse_gguf", self._lang))
        self._browse_gguf_btn.setObjectName("ghostBtn")
        self._browse_gguf_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._bind_text(self._browse_gguf_btn, "settings.browse_gguf")
        self._browse_gguf_btn.clicked.connect(self._browse_custom_gguf)
        rec_row.addWidget(self._browse_gguf_btn)
        rec_row.addStretch()
        emb_layout.addLayout(rec_row)

        self._custom_gguf_edit = QLineEdit(self._config.embedded_llm_custom_path)
        self._custom_gguf_edit.setMinimumWidth(0)
        self._custom_gguf_edit.setPlaceholderText(tr("settings.custom_gguf_placeholder", self._lang))
        self._bind_placeholder(self._custom_gguf_edit, "settings.custom_gguf_placeholder")
        self._custom_gguf_edit.setVisible(bool(self._config.embedded_llm_custom_path.strip()))
        emb_layout.addWidget(self._custom_gguf_edit)

        self._download_box = QWidget()
        self._download_box.setVisible(False)
        dl_v = QVBoxLayout(self._download_box)
        dl_v.setContentsMargins(0, 2, 0, 2)
        dl_v.setSpacing(4)
        dl_top = QHBoxLayout()
        self._download_status_lbl = QLabel("")
        self._download_status_lbl.setObjectName("mutedMeta")
        dl_top.addWidget(self._download_status_lbl, stretch=1)
        self._download_cancel_btn = QPushButton(tr("common.cancel", self._lang))
        self._download_cancel_btn.setObjectName("ghostBtn")
        self._download_cancel_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._bind_text(self._download_cancel_btn, "common.cancel")
        self._download_cancel_btn.clicked.connect(self._cancel_model_download)
        dl_top.addWidget(self._download_cancel_btn)
        dl_v.addLayout(dl_top)

        self._download_bar = QProgressBar()
        self._download_bar.setRange(0, 100)
        self._download_bar.setTextVisible(False)
        dl_v.addWidget(self._download_bar)
        emb_layout.addWidget(self._download_box)

        self._embedded_label = self._add_form_row(llm_form, "settings.model_gguf_label", self._embedded_panel)

        # --- External Ollama controls ---
        self._ollama_url_edit = QLineEdit(self._config.ollama_url)
        self._ollama_url_edit.setMinimumWidth(0)
        self._ollama_url_label = self._add_form_row(llm_form, "settings.ollama_url_label", self._ollama_url_edit)

        self._ollama_model_row_widget = QWidget()
        self._ollama_model_row_widget.setMinimumWidth(0)
        model_row = QHBoxLayout(self._ollama_model_row_widget)
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(8)
        self._ollama_model_combo = _DropdownCombo()
        self._configure_fluid_combo(self._ollama_model_combo, min_chars=10)
        self._set_combo_text(self._ollama_model_combo, self._config.ollama_model)
        model_row.addWidget(self._ollama_model_combo, stretch=1)

        refresh_models_btn = QPushButton(tr("settings.refresh_models", self._lang))
        refresh_models_btn.setObjectName("secondary")
        refresh_models_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._bind_text(refresh_models_btn, "settings.refresh_models")
        refresh_models_btn.clicked.connect(self._refresh_ollama_models)
        model_row.addWidget(refresh_models_btn)
        self._ollama_model_label = self._add_form_row(
            llm_form, "settings.ollama_model_label", self._ollama_model_row_widget
        )

        self._keep_alive_combo = _DropdownCombo()
        self._keep_alive_combo.setEditable(True)
        self._keep_alive_combo.setFixedWidth(110)
        for ka in ("0", "60s", "2m", "5m", "10m", "30m"):
            self._keep_alive_combo.addItem(ka)
        self._keep_alive_combo.setCurrentText(self._config.ollama_keep_alive)
        self._add_form_row(llm_form, "settings.keep_alive_label", self._keep_alive_combo)
        cards_layout.addWidget(self._llm_card)

        self._refresh_embedded_model_items(select_id=self._config.embedded_llm_model_id)
        self._update_llm_backend_visibility()

        # Card 4: Appearance, Flow Bar & Behavior
        ux_card, ux_form = self._make_section_card("settings.ux_hdr")

        self._language_combo = _DropdownCombo()
        self._configure_fluid_combo(self._language_combo, min_chars=10)
        for lang_label, lang_code in LANGUAGE_CHOICE_ITEMS:
            self._language_combo.addItem(lang_label, lang_code)
        l_idx = self._language_combo.findData(self._lang)
        self._language_combo.setCurrentIndex(l_idx if l_idx >= 0 else 0)
        self._language_combo.currentIndexChanged.connect(self._on_language_combo_changed)
        self._add_form_row(ux_form, "settings.language_label", self._language_combo)

        self._theme_combo = _DropdownCombo()
        self._configure_fluid_combo(self._theme_combo, min_chars=10)
        for theme_label, theme_key in get_theme_choice_items(self._lang):
            self._theme_combo.addItem(theme_label, theme_key)
        t_idx = self._theme_combo.findData(self._config.theme)
        self._theme_combo.setCurrentIndex(t_idx if t_idx >= 0 else 0)
        self._theme_combo.currentIndexChanged.connect(self._on_theme_combo_changed)
        self._add_form_row(ux_form, "settings.theme_label", self._theme_combo)

        bar_row = QHBoxLayout()
        bar_row.setSpacing(8)
        self._show_idle_bar_check = QCheckBox(tr("settings.show_idle_bar", self._lang))
        self._show_idle_bar_check.setChecked(self._config.show_idle_bar)
        self._bind_text(self._show_idle_bar_check, "settings.show_idle_bar")
        bar_row.addWidget(self._show_idle_bar_check)
        bar_row.addStretch()
        self._reset_pill_pos_btn = QPushButton(tr("settings.reset_pos", self._lang))
        self._reset_pill_pos_btn.setObjectName("secondary")
        self._reset_pill_pos_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._bind_text(self._reset_pill_pos_btn, "settings.reset_pos")
        self._reset_pill_pos_btn.clicked.connect(self.reset_pill_position_requested.emit)
        bar_row.addWidget(self._reset_pill_pos_btn)
        ux_form.addRow(bar_row)

        self._live_preview_check = QCheckBox(tr("settings.live_preview", self._lang))
        self._live_preview_check.setChecked(self._config.live_preview_enabled)
        self._bind_text(self._live_preview_check, "settings.live_preview")
        ux_form.addRow(self._live_preview_check)

        self._restore_cb_check = QCheckBox(tr("settings.restore_cb", self._lang))
        self._restore_cb_check.setChecked(self._config.paste_restore_clipboard)
        self._bind_text(self._restore_cb_check, "settings.restore_cb")
        ux_form.addRow(self._restore_cb_check)

        sounds_row = QHBoxLayout()
        sounds_row.setSpacing(8)
        self._sounds_check = QCheckBox(tr("settings.sounds_check", self._lang))
        self._sounds_check.setChecked(self._config.play_sounds)
        self._bind_text(self._sounds_check, "settings.sounds_check")
        sounds_row.addWidget(self._sounds_check)
        sounds_row.addStretch()
        self._test_sound_btn = QPushButton(tr("settings.test_sound", self._lang))
        self._test_sound_btn.setObjectName("secondary")
        self._test_sound_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._test_sound_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._bind_text(self._test_sound_btn, "settings.test_sound")
        self._test_sound_btn.clicked.connect(lambda: play_cue("start"))
        sounds_row.addWidget(self._test_sound_btn)
        ux_form.addRow(sounds_row)

        self._autostart_check = QCheckBox(tr("settings.autostart", self._lang))
        self._autostart_check.setChecked(self._config.autostart_enabled)
        self._bind_text(self._autostart_check, "settings.autostart")
        ux_form.addRow(self._autostart_check)
        cards_layout.addWidget(ux_card)

        cards_layout.addStretch()
        return page

    def _populate_mics(self, *, select_data: object = None) -> None:
        target = select_data if select_data is not None else self._config.audio_device_index
        self._mic_combo.clear()
        self._mic_combo.addItem(tr("settings.mic_default", self._lang), None)
        try:
            for dev in list_input_devices():
                suffix = tr("settings.mic_default_suffix", self._lang) if dev.is_default else ""
                self._mic_combo.addItem(f"{dev.name}{suffix}", dev.index)
        except Exception:
            pass
        if target is not None:
            idx = self._mic_combo.findData(target)
            if idx >= 0:
                self._mic_combo.setCurrentIndex(idx)

    def _refresh_ollama_models(self) -> None:
        url = self._ollama_url_edit.text().strip()
        self._save_status.setText(tr("settings.loading_ollama", self._lang))

        def _fetch() -> None:
            polisher = OllamaPolisher(base_url=url, timeout_sec=3.0)
            try:
                models = polisher.list_models()
                self._models_loaded.emit(models)
            except Exception as exc:
                self._models_failed.emit(str(exc))
            finally:
                polisher.close()

        threading.Thread(target=_fetch, name="OllamaModelsFetch", daemon=True).start()

    def _on_models_loaded(self, models: list) -> None:
        self._cached_models = [str(m) for m in models]
        current = self._ollama_model_combo.currentText()
        self._ollama_model_combo.clear()
        for m in self._cached_models:
            self._ollama_model_combo.addItem(m)
        if current:
            self._set_combo_text(self._ollama_model_combo, current)
        self._save_status.setText(
            tr("settings.found_ollama", self._lang, count=len(self._cached_models))
        )

    def _on_models_failed(self, err: str) -> None:
        self._save_status.setText(tr("settings.ollama_err", self._lang, err=err))

    def _on_language_combo_changed(self) -> None:
        lang = str(self._language_combo.currentData() or DEFAULT_LANGUAGE)
        self.apply_language(lang)
        self.language_preview_requested.emit(lang)

    def _on_theme_combo_changed(self) -> None:
        mode = str(self._theme_combo.currentData() or "system")
        self.apply_theme(mode)
        self.theme_preview_requested.emit(mode)

    def _on_profile_combo_changed(self) -> None:
        profile = self._profile_combo.currentData()
        if profile == "custom" or not profile:
            return
        preview = apply_performance_profile(
            self._config,
            profile,
            available_ollama_models=self._cached_models or None,
        )
        self._whisper_combo.setCurrentText(preview.whisper_model)
        self._whisper_dev_combo.setCurrentText(preview.whisper_device)
        self._whisper_ct_combo.setCurrentText(preview.whisper_compute_type)
        self._set_combo_text(self._ollama_model_combo, preview.ollama_model)
        self._keep_alive_combo.setCurrentText(preview.ollama_keep_alive)
        self._save_status.setText(
            tr("settings.preset_applied", self._lang, name=self._profile_combo.currentText())
        )

    def _update_auto_style_controls_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self._messenger_strip_period_check.setEnabled(enabled)
        for combo in self._cat_style_combos.values():
            combo.setEnabled(enabled)
        self._app_overrides_edit.setEnabled(enabled)

    def _load_style_instruction_editor(self, style_key: str) -> None:
        self._editing_style_key = style_key
        text = self._pending_style_instructions.get(style_key) or get_default_style_instruction(
            style_key, self._lang
        )
        self._style_instruction_edit.setPlainText(text)

    def _flush_current_style_instruction(self) -> None:
        key = self._editing_style_key
        if not key or key == "raw":
            return
        text = self._style_instruction_edit.toPlainText().strip()
        if not text or is_builtin_style_instruction(key, text):
            self._pending_style_instructions.pop(key, None)
        else:
            self._pending_style_instructions[key] = text

    def _on_edit_style_changed(self) -> None:
        new_key = self._edit_style_combo.currentData()
        if not new_key or new_key == self._editing_style_key:
            return
        self._flush_current_style_instruction()
        self._load_style_instruction_editor(str(new_key))

    def _reset_current_style_instruction(self) -> None:
        key = self._editing_style_key
        self._pending_style_instructions.pop(key, None)
        self._style_instruction_edit.setPlainText(
            get_default_style_instruction(key, self._lang)
        )

    def _update_llm_backend_visibility(self) -> None:
        is_embedded = self._llm_backend_combo.currentData() == "embedded"
        self._embedded_label.setVisible(is_embedded)
        self._embedded_panel.setVisible(is_embedded)
        self._ollama_url_label.setVisible(not is_embedded)
        self._ollama_url_edit.setVisible(not is_embedded)
        self._ollama_model_label.setVisible(not is_embedded)
        self._ollama_model_row_widget.setVisible(not is_embedded)

    def _refresh_embedded_model_items(self, *, select_id: str | None = None) -> None:
        target_id = select_id or self._embedded_model_combo.currentData() or self._config.embedded_llm_model_id
        self._embedded_model_combo.blockSignals(True)
        self._embedded_model_combo.clear()
        for spec in EMBEDDED_LLM_MODELS:
            status = "✓" if is_model_downloaded(spec.id) else "↓"
            short_name = spec.name.replace(" (Q4_K_M)", "")
            badge = get_model_badge(spec, self._lang)
            self._embedded_model_combo.addItem(f"{status} {short_name} · {badge}", spec.id)
        idx = self._embedded_model_combo.findData(target_id)
        self._embedded_model_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._embedded_model_combo.blockSignals(False)
        self._on_embedded_model_changed()

    def _on_embedded_model_changed(self) -> None:
        model_id = str(self._embedded_model_combo.currentData() or "")
        spec = get_model_spec(model_id)
        if spec is None:
            return
        downloaded = is_model_downloaded(spec.id)
        runtime_ok = is_llama_runtime_installed()
        status_note = (
            tr("settings.model_ready", self._lang)
            if (downloaded and runtime_ok)
            else (
                tr("settings.model_no_runtime", self._lang)
                if downloaded
                else tr("settings.model_not_downloaded", self._lang)
            )
        )
        desc = get_model_description(spec, self._lang)
        self._embedded_model_desc.setText(f"{desc}\n{status_note}")
        self._download_model_btn.setText(
            tr("settings.downloaded_ok", self._lang)
            if (downloaded and runtime_ok)
            else tr("settings.download_model", self._lang)
        )
        self._download_model_btn.setEnabled(not self._download_active and not (downloaded and runtime_ok))
        self._delete_model_btn.setEnabled(not self._download_active and downloaded)
        self._download_recommended_btn.setEnabled(not self._download_active)

    def _browse_custom_gguf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("settings.browse_gguf_title", self._lang),
            "",
            "GGUF Models (*.gguf);;All Files (*)",
        )
        if path:
            self._custom_gguf_edit.setText(path)
            self._custom_gguf_edit.setVisible(True)
            self._save_status.setText(tr("settings.gguf_selected", self._lang))

    def _start_embedded_download(self, model_id: str, *, also_switch_whisper: bool = False) -> None:
        if self._download_active:
            return
        spec = get_model_spec(model_id)
        if spec is None:
            return
        self._download_active = True
        self._download_cancel_event = threading.Event()
        cancel_ev = self._download_cancel_event
        dl_lang = self._lang

        self._download_box.setVisible(True)
        self._download_bar.setRange(0, 0)  # Indeterminate until size is known
        self._download_status_lbl.setText(tr("settings.dl_preparing", dl_lang, name=spec.name))
        self._on_embedded_model_changed()

        def _worker() -> None:
            try:
                download_embedded_llm_model(
                    spec,
                    on_progress=lambda prog: self._download_progress.emit(prog),
                    cancel_event=cancel_ev,
                    lang=dl_lang,
                )
                self._download_finished.emit(spec.id, also_switch_whisper)
            except DownloadCancelledError:
                self._download_failed.emit(tr("settings.dl_cancelled", dl_lang))
            except Exception as exc:
                self._download_failed.emit(tr("settings.dl_error", dl_lang, err=exc))

        threading.Thread(target=_worker, name="EmbeddedModelDownload", daemon=True).start()

    def _download_selected_model_clicked(self) -> None:
        model_id = str(self._embedded_model_combo.currentData() or "")
        if model_id:
            self._start_embedded_download(model_id, also_switch_whisper=False)

    def _download_recommended_clicked(self) -> None:
        self._start_embedded_download(RECOMMENDED_EMBEDDED_MODEL_ID, also_switch_whisper=True)

    def _delete_selected_model_clicked(self) -> None:
        model_id = str(self._embedded_model_combo.currentData() or "")
        spec = get_model_spec(model_id)
        if spec is None or not is_model_downloaded(spec.id):
            return
        if not self._confirm_delete_dialog(
            tr("settings.delete_model_title", self._lang),
            tr("settings.delete_model_text", self._lang, name=spec.name),
            tr("settings.delete_model_info", self._lang),
            tr("history.delete", self._lang),
        ):
            return
        delete_model_file(spec.id)
        self._refresh_embedded_model_items(select_id=spec.id)
        self._save_status.setText(tr("settings.model_deleted", self._lang, name=spec.name))

    def _cancel_model_download(self) -> None:
        if self._download_cancel_event is not None:
            self._download_cancel_event.set()
            self._download_status_lbl.setText(tr("settings.dl_cancelling", self._lang))

    def _on_download_progress(self, progress: object) -> None:
        if not isinstance(progress, DownloadProgress):
            return
        if progress.total_bytes > 0:
            if self._download_bar.maximum() == 0:
                self._download_bar.setRange(0, 100)
            self._download_bar.setValue(progress.percent)
            dl_mb = progress.downloaded_bytes / (1024 * 1024)
            tot_mb = progress.total_bytes / (1024 * 1024)
            spd = f" · {progress.speed_mb_s:.1f} MB/s" if progress.speed_mb_s > 0.05 else ""
            self._download_status_lbl.setText(
                f"{progress.stage}: {progress.percent}% ({dl_mb:.0f} / {tot_mb:.0f} MB{spd})"
            )
        else:
            self._download_bar.setRange(0, 0)
            self._download_status_lbl.setText(progress.stage)

    def _on_download_finished(self, model_id: str, also_switch_whisper: bool) -> None:
        self._download_active = False
        self._download_cancel_event = None
        self._download_box.setVisible(False)
        self._refresh_embedded_model_items(select_id=model_id)
        b_idx = self._llm_backend_combo.findData("embedded")
        if b_idx >= 0:
            self._llm_backend_combo.setCurrentIndex(b_idx)
        self._custom_gguf_edit.clear()
        self._custom_gguf_edit.setVisible(False)
        if also_switch_whisper:
            self._whisper_combo.setCurrentText("large-v3-turbo")
        self._save_clicked()
        spec = get_model_spec(model_id)
        label = spec.name if spec is not None else model_id
        self._save_status.setText(
            tr("settings.dl_rec_done", self._lang, label=label)
            if also_switch_whisper
            else tr("settings.dl_one_done", self._lang, label=label)
        )

    def _on_download_failed(self, err: str) -> None:
        self._download_active = False
        self._download_cancel_event = None
        self._download_box.setVisible(False)
        self._on_embedded_model_changed()
        self._save_status.setText(err)

    def sync_from_config(self, cfg: AppConfig) -> None:
        self._config = cfg
        self._language_combo.blockSignals(True)
        l_idx = self._language_combo.findData(normalize_language(cfg.language))
        self._language_combo.setCurrentIndex(l_idx if l_idx >= 0 else 0)
        self._language_combo.blockSignals(False)
        self.apply_language(cfg.language)

        self._profile_combo.blockSignals(True)
        idx = self._profile_combo.findData(cfg.performance_profile)
        if idx < 0:
            idx = self._profile_combo.findData("custom")
        if idx >= 0:
            self._profile_combo.setCurrentIndex(idx)
        self._profile_combo.blockSignals(False)

        mic_idx = self._mic_combo.findData(cfg.audio_device_index)
        self._mic_combo.setCurrentIndex(mic_idx if mic_idx >= 0 else 0)
        self._double_tap_spin.setValue(cfg.double_tap_ms)

        self._whisper_combo.setCurrentText(cfg.whisper_model)
        self._whisper_dev_combo.setCurrentText(cfg.whisper_device)
        self._whisper_ct_combo.setCurrentText(cfg.whisper_compute_type)

        self._auto_style_check.setChecked(cfg.auto_style_by_app)
        self._messenger_strip_period_check.setChecked(cfg.messenger_strip_trailing_period)
        for cat_key, combo in self._cat_style_combos.items():
            target_style = cfg.category_styles.get(cat_key, DEFAULT_CATEGORY_STYLES[cat_key])
            cidx = combo.findData(target_style)
            if cidx >= 0:
                combo.setCurrentIndex(cidx)
        self._app_overrides_edit.setPlainText(format_app_style_overrides(cfg.app_style_overrides))
        self._pending_style_instructions = dict(cfg.style_custom_instructions)
        self._load_style_instruction_editor(self._editing_style_key)
        self._update_auto_style_controls_enabled(cfg.auto_style_by_app)

        self._llm_check.setChecked(cfg.llm_enabled)
        b_idx = self._llm_backend_combo.findData(cfg.llm_backend)
        if b_idx >= 0:
            self._llm_backend_combo.setCurrentIndex(b_idx)
        self._refresh_embedded_model_items(select_id=cfg.embedded_llm_model_id)
        self._custom_gguf_edit.setText(cfg.embedded_llm_custom_path)
        self._custom_gguf_edit.setVisible(bool(cfg.embedded_llm_custom_path.strip()))
        self._update_llm_backend_visibility()

        self._ollama_url_edit.setText(cfg.ollama_url)
        self._set_combo_text(self._ollama_model_combo, cfg.ollama_model)
        self._keep_alive_combo.setCurrentText(cfg.ollama_keep_alive)
        self._prompt_edit.setPlainText(cfg.llm_system_prompt)
        self._vocab_edit.setPlainText(cfg.custom_vocabulary)

        self._restore_cb_check.setChecked(cfg.paste_restore_clipboard)
        self._sounds_check.setChecked(cfg.play_sounds)
        self._autostart_check.setChecked(cfg.autostart_enabled)
        self._show_idle_bar_check.setChecked(cfg.show_idle_bar)
        self._live_preview_check.setChecked(cfg.live_preview_enabled)

        self._theme_combo.blockSignals(True)
        t_idx = self._theme_combo.findData(cfg.theme)
        self._theme_combo.setCurrentIndex(t_idx if t_idx >= 0 else 0)
        self._theme_combo.blockSignals(False)

        self._silence_auto_stop_check.setChecked(cfg.silence_auto_stop_enabled)
        self._silence_auto_stop_spin.setValue(cfg.silence_auto_stop_sec)
        self._silence_auto_stop_spin.setEnabled(cfg.silence_auto_stop_enabled)
        self.apply_theme(cfg.theme)

    def _save_clicked(self) -> None:
        self._flush_current_style_instruction()
        category_styles = dict(DEFAULT_CATEGORY_STYLES)
        category_styles.update(self._config.category_styles)
        for cat_key, combo in self._cat_style_combos.items():
            chosen = combo.currentData()
            if chosen:
                category_styles[cat_key] = str(chosen)

        profile = self._profile_combo.currentData() or "custom"
        base = (
            apply_performance_profile(self._config, profile)
            if profile in ("quality", "light_gpu", "zero_vram")
            else self._config
        )
        llm_backend = str(self._llm_backend_combo.currentData() or "embedded")
        embedded_model_id = str(
            self._embedded_model_combo.currentData() or self._config.embedded_llm_model_id
        )
        language = normalize_language(str(self._language_combo.currentData() or DEFAULT_LANGUAGE))
        updated = base.model_copy(
            update={
                "performance_profile": profile,
                "audio_device_index": self._mic_combo.currentData(),
                "double_tap_ms": self._double_tap_spin.value(),
                "whisper_model": self._whisper_combo.currentText().strip() or "small",
                "whisper_device": self._whisper_dev_combo.currentText(),
                "whisper_compute_type": self._whisper_ct_combo.currentText(),
                "auto_style_by_app": self._auto_style_check.isChecked(),
                "messenger_strip_trailing_period": self._messenger_strip_period_check.isChecked(),
                "category_styles": category_styles,
                "app_style_overrides": parse_app_style_overrides(
                    self._app_overrides_edit.toPlainText()
                ),
                "style_custom_instructions": dict(self._pending_style_instructions),
                "llm_enabled": self._llm_check.isChecked(),
                "llm_backend": llm_backend,
                "embedded_llm_model_id": embedded_model_id,
                "embedded_llm_custom_path": self._custom_gguf_edit.text().strip(),
                "ollama_url": self._ollama_url_edit.text().strip() or "http://127.0.0.1:11434",
                "ollama_model": self._ollama_model_combo.currentText().strip() or "gemma4:e4b",
                "ollama_keep_alive": self._keep_alive_combo.currentText().strip() or "2m",
                "llm_system_prompt": self._prompt_edit.toPlainText().strip(),
                "custom_vocabulary": self._vocab_edit.toPlainText().strip(),
                "auto_space_before_paste": True,
                "paste_restore_clipboard": self._restore_cb_check.isChecked(),
                "play_sounds": self._sounds_check.isChecked(),
                "autostart_enabled": self._autostart_check.isChecked(),
                "show_idle_bar": self._show_idle_bar_check.isChecked(),
                "live_preview_enabled": self._live_preview_check.isChecked(),
                "language": language,
                "theme": str(self._theme_combo.currentData() or "system"),
                "silence_auto_stop_enabled": self._silence_auto_stop_check.isChecked(),
                "silence_auto_stop_sec": round(self._silence_auto_stop_spin.value(), 1),
            }
        )
        self._config = updated
        self.apply_language(updated.language)
        self.apply_theme(updated.theme)
        self.config_saved.emit(updated)
        self._save_status.setText(tr("settings.saved_ok", self._lang))

    def show_and_focus(self) -> None:
        self._populate_mics()
        self._refresh_embedded_model_items(select_id=self._config.embedded_llm_model_id)
        self.refresh_history()
        self.refresh_insights()
        if self._config.llm_backend == "ollama" and not self._cached_models:
            self._refresh_ollama_models()
        self.show()
        self.raise_()
        self.activateWindow()
