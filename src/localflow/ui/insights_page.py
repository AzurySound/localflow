"""Wispr Flow-inspired Usage Insights page with custom QPainter gauges, bars, and heatmap."""

from __future__ import annotations

from datetime import date, timedelta

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from localflow.i18n import (
    DEFAULT_LANGUAGE,
    LanguageCode,
    get_months,
    get_weekdays,
    normalize_language,
    plural_streak_days,
    tr,
)
from localflow.insights import CategoryStat, DailyActivity, UsageInsights, compute_usage_insights
from localflow.storage import TranscriptionEntry
from localflow.ui.theme import ResolvedTheme

# Exact Wispr Flow teal ramp for usage bars and heatmap
_TEAL_BAR_COLORS = (
    QColor("#115e59"),
    QColor("#2a9d8f"),
    QColor("#45b7a5"),
    QColor("#5ec7b7"),
    QColor("#7dd3c7"),
)
_HEATMAP_EMPTY_DARK = QColor("#2d2d32")
_HEATMAP_FUTURE_DARK = QColor("#222226")
_HEATMAP_LEVELS_DARK = (
    QColor("#115e59"),
    QColor("#1f9d8b"),
    QColor("#5eead4"),
    QColor("#ccfbf1"),
)
_HEATMAP_EMPTY_LIGHT = QColor("#e4e4ea")
_HEATMAP_FUTURE_LIGHT = QColor("#efeff4")
_HEATMAP_LEVELS_LIGHT = (
    QColor("#99f6e4"),
    QColor("#2dd4bf"),
    QColor("#0d9488"),
    QColor("#115e59"),
)


class _ArcGaugeWidget(QWidget):
    """Semi-circular teal speed gauge (matching Wispr Flow WPM Top X% widget)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._wpm = 0
        self._top_percent = 100
        self._theme: ResolvedTheme = "dark"
        self._lang: LanguageCode = DEFAULT_LANGUAGE
        self.setFixedHeight(86)
        self.setMinimumWidth(148)

    def set_theme(self, theme: ResolvedTheme) -> None:
        self._theme = theme
        self.update()

    def set_language(self, lang: str) -> None:
        self._lang = normalize_language(lang)
        self.update()

    def set_metrics(self, wpm: int, top_percent: int) -> None:
        self._wpm = max(0, wpm)
        self._top_percent = max(1, min(100, top_percent))
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        is_light = self._theme == "light"

        stroke = 8.0
        diam = min(self.width() - stroke * 2 - 16.0, 96.0)
        diam = max(70.0, diam)
        x = (self.width() - diam) / 2.0
        y = 6.0
        arc_rect = QRectF(x, y, diam, diam)

        # Background track (200-degree wide semi-arc)
        track_color = QColor("#e4e4ea") if is_light else QColor("#333338")
        bg_pen = QPen(track_color, stroke, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(bg_pen)
        p.drawArc(arc_rect, 190 * 16, -200 * 16)

        # Active teal progress (higher WPM / lower Top% => fuller arc)
        ratio = 0.0 if self._wpm <= 0 else max(0.12, min(0.95, (100.0 - self._top_percent + 10.0) / 100.0))
        if ratio > 0.0:
            arc_color = QColor("#0d9488") if is_light else QColor("#147d73")
            fg_pen = QPen(arc_color, stroke, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            p.setPen(fg_pen)
            p.drawArc(arc_rect, 190 * 16, int(-200 * ratio * 16))

        # Center labels
        if self._wpm > 0:
            p.setPen(QColor("#63636e") if is_light else QColor("#a1a1aa"))
            p.setFont(QFont("Segoe UI", 8, QFont.Weight.Medium))
            p.drawText(QRectF(x, y + diam * 0.20, diam, 15), Qt.AlignmentFlag.AlignCenter, "Top")

            p.setPen(QColor("#111113") if is_light else QColor("#fafafa"))
            p.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            p.drawText(
                QRectF(x, y + diam * 0.38, diam, 22),
                Qt.AlignmentFlag.AlignCenter,
                f"{self._top_percent}%",
            )
        else:
            p.setPen(QColor("#71717a"))
            p.setFont(QFont("Segoe UI", 8.5, QFont.Weight.Medium))
            p.drawText(
                QRectF(x, y + diam * 0.28, diam, 20),
                Qt.AlignmentFlag.AlignCenter,
                tr("insights.no_data", self._lang),
            )


class _UsageBarsWidget(QWidget):
    """Horizontal teal percentage bars matching Wispr Flow Desktop Usage card."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._categories: list[CategoryStat] = []
        self._items: list[tuple[str, int, str]] = []  # (icon, percent, right_label)
        self._theme: ResolvedTheme = "dark"
        self._lang: LanguageCode = DEFAULT_LANGUAGE
        self.setMinimumHeight(186)

    def set_theme(self, theme: ResolvedTheme) -> None:
        self._theme = theme
        self.update()

    def set_language(self, lang: str) -> None:
        self._lang = normalize_language(lang)
        self._rebuild_items()

    def _rebuild_items(self) -> None:
        icon_map = {
            "messenger": "💬",
            "ide": "⌨",
            "docs": "📄",
            "command": "✦",
            "general": "∞",
        }
        self._items = [
            (
                icon_map.get(c.key, "●"),
                c.percent,
                f"{c.count}  {tr(f'insights.cat.{c.key}', self._lang)}",
            )
            for c in self._categories
        ]
        self.setMinimumHeight(max(140, len(self._items) * 34 + 6))
        self.update()

    def set_categories(self, categories: list[CategoryStat]) -> None:
        self._categories = list(categories)
        self._rebuild_items()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        is_light = self._theme == "light"

        if not self._items:
            p.setPen(QColor("#71717a"))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, tr("insights.no_data_yet", self._lang))
            return

        row_h = 34.0
        bar_h = 22.0
        icon_w = 24.0
        bar_start_x = icon_w + 6.0
        max_bar_w = max(84.0, (self.width() - bar_start_x) * 0.46)
        min_pill_w = 34.0

        icon_font = QFont("Segoe UI Symbol", 10)
        pct_font = QFont("Segoe UI", 8.5, QFont.Weight.Bold)
        lbl_font = QFont("Segoe UI", 8, QFont.Weight.Bold)

        for idx, (icon, pct, right_label) in enumerate(self._items):
            y = idx * row_h + (row_h - bar_h) / 2.0

            # Left icon
            p.setPen(QColor("#52525b") if is_light else QColor("#a1a1aa"))
            p.setFont(icon_font)
            p.drawText(QRectF(0.0, y, icon_w, bar_h), Qt.AlignmentFlag.AlignCenter, icon)

            # Teal pill bar
            frac = max(0.0, min(1.0, pct / 100.0))
            bar_w = min_pill_w if pct <= 0 else max(min_pill_w + 10.0, max_bar_w * frac)
            color = _TEAL_BAR_COLORS[min(idx, len(_TEAL_BAR_COLORS) - 1)]
            bar_rect = QRectF(bar_start_x, y, bar_w, bar_h)

            path = QPainterPath()
            path.addRoundedRect(bar_rect, 5.0, 5.0)
            p.fillPath(path, color)

            # Percentage inside pill
            text_color = QColor("#0b1215") if idx >= 2 or pct <= 0 else QColor("#f4f4f5")
            p.setPen(text_color)
            p.setFont(pct_font)
            p.drawText(bar_rect, Qt.AlignmentFlag.AlignCenter, f"{pct}%")

            # Right uppercase label
            label_x = bar_start_x + bar_w + 10.0
            label_w = max(20.0, self.width() - label_x)
            if is_light:
                lbl_color = QColor("#27272a") if pct > 0 else QColor("#71717a")
            else:
                lbl_color = QColor("#d4d4d8") if pct > 0 else QColor("#8a8a93")
            p.setPen(lbl_color)
            p.setFont(lbl_font)
            elided = p.fontMetrics().elidedText(right_label, Qt.TextElideMode.ElideRight, int(label_w))
            p.drawText(
                QRectF(label_x, y, label_w, bar_h),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                elided,
            )


class _ActivityHeatmapWidget(QWidget):
    """7xN GitHub/Wispr Flow-style daily activity heatmap with hover tooltips."""

    WEEKS = 18

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._daily: dict[date, DailyActivity] = {}
        self._end_date: date = date.today()
        self._week_offset: int = 0
        self._cell_rects: list[tuple[QRectF, date]] = []
        self._theme: ResolvedTheme = "dark"
        self._lang: LanguageCode = DEFAULT_LANGUAGE
        self.setMouseTracking(True)
        self.setMinimumHeight(168)

    def set_theme(self, theme: ResolvedTheme) -> None:
        self._theme = theme
        self.update()

    def set_language(self, lang: str) -> None:
        self._lang = normalize_language(lang)
        self.update()

    def set_data(self, daily: dict[date, DailyActivity], end_date: date | None = None) -> None:
        self._daily = daily
        self._end_date = end_date or date.today()
        self.update()

    def shift_weeks(self, delta: int) -> None:
        self._week_offset = min(0, self._week_offset + delta)
        self.update()

    def _grid_start_sunday(self) -> date:
        anchor = self._end_date + timedelta(weeks=self._week_offset)
        # Convert Python weekday (Mon=0..Sun=6) to Sunday-based row index (Sun=0..Sat=6)
        sun_offset = (anchor.weekday() + 1) % 7
        current_week_sunday = anchor - timedelta(days=sun_offset)
        return current_week_sunday - timedelta(weeks=self.WEEKS - 1)

    def _color_for_words(self, words: int, max_words: int) -> QColor:
        is_light = self._theme == "light"
        levels = _HEATMAP_LEVELS_LIGHT if is_light else _HEATMAP_LEVELS_DARK
        empty_color = _HEATMAP_EMPTY_LIGHT if is_light else _HEATMAP_EMPTY_DARK
        if words <= 0:
            return empty_color
        if max_words <= 20:
            return levels[3]
        ratio = words / float(max_words)
        if ratio >= 0.75:
            return levels[3]
        if ratio >= 0.45:
            return levels[2]
        if ratio >= 0.20:
            return levels[1]
        return levels[0]

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        is_light = self._theme == "light"
        muted_color = QColor("#63636e") if is_light else QColor("#8a8a93")
        future_color = _HEATMAP_FUTURE_LIGHT if is_light else _HEATMAP_FUTURE_DARK
        levels = _HEATMAP_LEVELS_LIGHT if is_light else _HEATMAP_LEVELS_DARK
        months = get_months(self._lang)
        days = get_weekdays(self._lang)

        left_margin = 30.0
        top_margin = 24.0
        bottom_margin = 24.0
        avail_w = max(120.0, self.width() - left_margin - 6.0)
        avail_h = max(90.0, self.height() - top_margin - bottom_margin)

        gap = 3.5
        cell_w = (avail_w - gap * (self.WEEKS - 1)) / self.WEEKS
        cell_h = (avail_h - gap * 6) / 7.0
        cell = max(7.0, min(14.0, cell_w, cell_h))
        step = cell + gap

        start_sun = self._grid_start_sunday()
        max_words = max((a.words for a in self._daily.values()), default=0)

        # Month headers: evaluate month by the end of each week column (Saturday)
        # and prefer the first full month over a 1-column partial tail at col=0.
        p.setPen(muted_color)
        p.setFont(QFont("Segoe UI", 8))
        month_positions: list[tuple[float, str]] = []
        last_month = -1
        for col in range(self.WEEKS):
            col_end_day = start_sun + timedelta(days=col * 7 + 6)
            if col_end_day.month != last_month:
                last_month = col_end_day.month
                mx = left_margin + col * step
                if month_positions and (mx - month_positions[-1][0]) < 34.0:
                    # Replace a cramped partial-month label at col=0 with the new full month
                    month_positions[-1] = (mx, months[col_end_day.month - 1])
                else:
                    month_positions.append((mx, months[col_end_day.month - 1]))

        for mx, mlabel in month_positions:
            p.drawText(
                QRectF(mx, 2.0, 38.0, 16.0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                mlabel,
            )

        # Weekday labels along the left
        p.setFont(QFont("Segoe UI", 7.5))
        for row, label in enumerate(days):
            ry = top_margin + row * step
            p.drawText(
                QRectF(0.0, ry, left_margin - 5.0, cell),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

        # Grid cells
        self._cell_rects = []
        p.setPen(Qt.PenStyle.NoPen)
        for col in range(self.WEEKS):
            for row in range(7):
                day = start_sun + timedelta(days=col * 7 + row)
                rect = QRectF(left_margin + col * step, top_margin + row * step, cell, cell)
                if day > self._end_date:
                    p.setBrush(future_color)
                else:
                    act = self._daily.get(day)
                    words = act.words if act else 0
                    p.setBrush(self._color_for_words(words, max_words))
                    self._cell_rects.append((rect, day))
                p.drawRoundedRect(rect, 2.5, 2.5)

        # Bottom legend: "Less ◼ ◼ ◼ ◼ More"
        leg_y = top_margin + 7 * step + 7.0
        p.setPen(muted_color)
        p.setFont(QFont("Segoe UI", 8))
        less_lbl = tr("insights.less", self._lang)
        more_lbl = tr("insights.more", self._lang)
        p.drawText(QRectF(0.0, leg_y - 2.0, 46.0, 16.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, less_lbl)

        swatch_x = 49.0
        p.setPen(Qt.PenStyle.NoPen)
        for idx, col_c in enumerate(levels):
            p.setBrush(col_c)
            p.drawRoundedRect(QRectF(swatch_x + idx * 14.0, leg_y, 10.0, 10.0), 2.0, 2.0)

        p.setPen(muted_color)
        p.drawText(
            QRectF(swatch_x + 4 * 14.0 + 4.0, leg_y - 2.0, 52.0, 16.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            more_lbl,
        )

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        pt = QPointF(event.position())
        months = get_months(self._lang)
        for rect, day in self._cell_rects:
            if rect.contains(pt):
                act = self._daily.get(day)
                d_str = f"{day.day} {months[day.month - 1]} {day.year}"
                if act and act.dictations > 0:
                    tip = tr(
                        "insights.tooltip_active",
                        self._lang,
                        date=d_str,
                        dictations=act.dictations,
                        words=act.words,
                    )
                else:
                    tip = tr("insights.tooltip_empty", self._lang, date=d_str)
                QToolTip.showText(event.globalPosition().toPoint(), tip, self)
                return
        QToolTip.hideText()
        super().mouseMoveEvent(event)


class InsightsPage(QWidget):
    """Usage Insights page styled after Wispr Flow Insights (without Share button)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._insights: UsageInsights | None = None
        self._resolved_theme: ResolvedTheme = "dark"
        self._lang: LanguageCode = DEFAULT_LANGUAGE

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 18)
        outer.setSpacing(12)

        # Header (no Share button, no sub-tabs — single Usage dashboard)
        self._title_lbl = QLabel(tr("insights.title", self._lang))
        self._title_lbl.setObjectName("pageTitle")
        outer.addWidget(self._title_lbl)

        div = QFrame()
        div.setObjectName("divider")
        outer.addWidget(div)

        # Scrollable usage dashboard
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._usage_view = self._build_usage_view()
        scroll.setWidget(self._usage_view)
        outer.addWidget(scroll, stretch=1)
        self.set_theme("dark")
        self.set_language(DEFAULT_LANGUAGE)

    def set_language(self, lang: str) -> None:
        self._lang = normalize_language(lang)
        self._title_lbl.setText(tr("insights.title", self._lang))
        self._wpm_sub.setText(tr("insights.wpm_hdr", self._lang))
        self._fixes_sub.setText(tr("insights.ai_fixes_hdr", self._lang))
        self._words_sub.setText(tr("insights.total_words_hdr", self._lang))
        self._desk_lbl.setText(tr("insights.desktop", self._lang))
        self._usage_title.setText(tr("insights.usage_title", self._lang))
        self._prev_btn.setToolTip(tr("insights.prev_period", self._lang))
        self._next_btn.setToolTip(tr("insights.next_period", self._lang))
        self._wpm_gauge.set_language(self._lang)
        self._usage_bars.set_language(self._lang)
        self._heatmap.set_language(self._lang)
        self._render_metrics()

    def set_theme(self, theme: ResolvedTheme) -> None:
        self._resolved_theme = theme
        is_light = theme == "light"
        strong = "#111113" if is_light else "#fafafa"
        body = "#3f3f46" if is_light else "#d4d4d8"
        desk = "#18181b" if is_light else "#e4e4e7"

        kpi_css = f"color:{strong}; font-size:20pt; font-weight:700;"
        self._wpm_val.setStyleSheet(kpi_css)
        self._fixes_val.setStyleSheet(kpi_css)
        self._total_words_val.setStyleSheet(kpi_css)

        sub_css = f"color:{body}; font-size:9pt;"
        self._words_corrected_lbl.setStyleSheet(sub_css)
        self._dict_fixes_lbl.setStyleSheet(sub_css)

        self._desk_lbl.setStyleSheet(f"color:{desk}; font-weight:600; font-size:9pt;")
        if is_light:
            self._time_saved_badge.setStyleSheet(
                "background-color:#ffffff; color:#27272a; border:1px solid #d4d4dc; "
                "border-radius:7px; padding:5px 10px; font-size:8.5pt; font-weight:600;"
            )
        else:
            self._time_saved_badge.setStyleSheet(
                "background-color:#151517; color:#d4d4d8; border:1px solid #2e2e34; "
                "border-radius:7px; padding:5px 10px; font-size:8.5pt; font-weight:600;"
            )

        card_title_css = f"color:{strong}; font-size:13pt; font-weight:700;"
        self._usage_title.setStyleSheet(card_title_css)
        self._streak_title.setStyleSheet(card_title_css)

        self._wpm_gauge.set_theme(theme)
        self._usage_bars.set_theme(theme)
        self._heatmap.set_theme(theme)

    def _make_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("sectionCard")
        return card

    # ------------------------------------------------------------------
    # Usage dashboard (Wispr Flow Insights layout)
    # ------------------------------------------------------------------
    def _build_usage_view(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(0, 6, 4, 6)
        root.setSpacing(14)

        # Top row: 3 KPI cards
        top_row = QHBoxLayout()
        top_row.setSpacing(14)

        # Card 1: WPM + Semi-circle Gauge
        wpm_card = self._make_card()
        wpm_l = QVBoxLayout(wpm_card)
        wpm_l.setContentsMargins(16, 14, 16, 10)
        wpm_l.setSpacing(2)
        self._wpm_val = QLabel("0")
        self._wpm_val.setStyleSheet("color:#fafafa; font-size:20pt; font-weight:700;")
        wpm_l.addWidget(self._wpm_val)
        self._wpm_sub = QLabel(tr("insights.wpm_hdr", self._lang))
        self._wpm_sub.setObjectName("sectionHeader")
        wpm_l.addWidget(self._wpm_sub)
        self._wpm_gauge = _ArcGaugeWidget()
        wpm_l.addWidget(self._wpm_gauge)
        top_row.addWidget(wpm_card, stretch=2)

        # Card 2: Fixes made by LocalFlow
        fixes_card = self._make_card()
        fixes_l = QVBoxLayout(fixes_card)
        fixes_l.setContentsMargins(16, 14, 16, 14)
        fixes_l.setSpacing(6)
        self._fixes_val = QLabel("0")
        self._fixes_val.setStyleSheet("color:#fafafa; font-size:20pt; font-weight:700;")
        fixes_l.addWidget(self._fixes_val)
        self._fixes_sub = QLabel(tr("insights.ai_fixes_hdr", self._lang))
        self._fixes_sub.setObjectName("sectionHeader")
        fixes_l.addWidget(self._fixes_sub)

        f_div = QFrame()
        f_div.setObjectName("divider")
        fixes_l.addWidget(f_div)
        fixes_l.addSpacing(4)

        self._words_corrected_lbl = QLabel(tr("insights.words_corrected", self._lang, count=0))
        self._words_corrected_lbl.setStyleSheet("color:#d4d4d8; font-size:9pt;")
        fixes_l.addWidget(self._words_corrected_lbl)

        self._dict_fixes_lbl = QLabel(tr("insights.dict_fixes", self._lang, count=0))
        self._dict_fixes_lbl.setStyleSheet("color:#d4d4d8; font-size:9pt;")
        fixes_l.addWidget(self._dict_fixes_lbl)
        fixes_l.addStretch()
        top_row.addWidget(fixes_card, stretch=2)

        # Card 3: Total Words Dictated
        words_card = self._make_card()
        words_l = QVBoxLayout(words_card)
        words_l.setContentsMargins(16, 14, 16, 14)
        words_l.setSpacing(6)
        self._total_words_val = QLabel("0")
        self._total_words_val.setStyleSheet("color:#fafafa; font-size:20pt; font-weight:700;")
        words_l.addWidget(self._total_words_val)
        self._words_sub = QLabel(tr("insights.total_words_hdr", self._lang))
        self._words_sub.setObjectName("sectionHeader")
        words_l.addWidget(self._words_sub)

        w_div = QFrame()
        w_div.setObjectName("divider")
        words_l.addWidget(w_div)
        words_l.addSpacing(4)

        foot_row = QHBoxLayout()
        left_foot = QVBoxLayout()
        left_foot.setSpacing(2)
        self._desk_lbl = QLabel(tr("insights.desktop", self._lang))
        self._desk_lbl.setStyleSheet("color:#e4e4e7; font-weight:600; font-size:9pt;")
        left_foot.addWidget(self._desk_lbl)
        self._desktop_sub_lbl = QLabel(tr("insights.desktop_sub", self._lang, count=0, mins=0.0))
        self._desktop_sub_lbl.setObjectName("mutedMeta")
        left_foot.addWidget(self._desktop_sub_lbl)
        foot_row.addLayout(left_foot)
        foot_row.addStretch()

        self._time_saved_badge = QLabel(tr("insights.time_saved_min", self._lang, mins=0.0))
        self._time_saved_badge.setStyleSheet(
            "background-color:#151517; color:#d4d4d8; border:1px solid #2e2e34; "
            "border-radius:7px; padding:5px 10px; font-size:8.5pt; font-weight:600;"
        )
        foot_row.addWidget(self._time_saved_badge, alignment=Qt.AlignmentFlag.AlignVCenter)
        words_l.addLayout(foot_row)
        words_l.addStretch()
        top_row.addWidget(words_card, stretch=3)

        root.addLayout(top_row)

        # Bottom row: 2 large cards (Desktop Usage + Streak Heatmap)
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(14)

        # Left bottom card: Desktop Usage
        usage_card = self._make_card()
        usage_l = QVBoxLayout(usage_card)
        usage_l.setContentsMargins(16, 16, 16, 14)
        usage_l.setSpacing(10)

        u_hdr = QHBoxLayout()
        self._usage_title = QLabel(tr("insights.usage_title", self._lang))
        self._usage_title.setStyleSheet("color:#fafafa; font-size:13pt; font-weight:700;")
        u_hdr.addWidget(self._usage_title)
        u_hdr.addStretch()
        self._apps_used_lbl = QLabel(tr("insights.apps_count", self._lang, count=0))
        self._apps_used_lbl.setObjectName("sectionHeader")
        u_hdr.addWidget(self._apps_used_lbl)
        usage_l.addLayout(u_hdr)

        self._usage_bars = _UsageBarsWidget()
        usage_l.addWidget(self._usage_bars, stretch=1)
        bottom_row.addWidget(usage_card, stretch=1)

        # Right bottom card: Streak + Heatmap
        streak_card = self._make_card()
        streak_l = QVBoxLayout(streak_card)
        streak_l.setContentsMargins(16, 16, 16, 14)
        streak_l.setSpacing(6)

        s_hdr = QHBoxLayout()
        self._streak_title = QLabel(plural_streak_days(0, self._lang))
        self._streak_title.setStyleSheet("color:#fafafa; font-size:13pt; font-weight:700;")
        s_hdr.addWidget(self._streak_title)
        s_hdr.addStretch()
        self._longest_streak_lbl = QLabel(tr("insights.best_streak", self._lang, days=0))
        self._longest_streak_lbl.setObjectName("sectionHeader")
        s_hdr.addWidget(self._longest_streak_lbl)
        s_hdr.addSpacing(6)

        self._prev_btn = QPushButton("‹")
        self._prev_btn.setObjectName("ghostBtn")
        self._prev_btn.setFixedSize(22, 22)
        self._prev_btn.setToolTip(tr("insights.prev_period", self._lang))
        self._prev_btn.clicked.connect(lambda: self._heatmap.shift_weeks(-4))
        self._next_btn = QPushButton("›")
        self._next_btn.setObjectName("ghostBtn")
        self._next_btn.setFixedSize(22, 22)
        self._next_btn.setToolTip(tr("insights.next_period", self._lang))
        self._next_btn.clicked.connect(lambda: self._heatmap.shift_weeks(4))
        s_hdr.addWidget(self._prev_btn)
        s_hdr.addWidget(self._next_btn)
        streak_l.addLayout(s_hdr)

        self._heatmap = _ActivityHeatmapWidget()
        streak_l.addWidget(self._heatmap, stretch=1)
        bottom_row.addWidget(streak_card, stretch=1)

        root.addLayout(bottom_row, stretch=1)
        return w

    def _render_metrics(self) -> None:
        ins = self._insights
        if ins is None:
            self._words_corrected_lbl.setText(tr("insights.words_corrected", self._lang, count=0))
            self._dict_fixes_lbl.setText(tr("insights.dict_fixes", self._lang, count=0))
            self._desktop_sub_lbl.setText(tr("insights.desktop_sub", self._lang, count=0, mins=0.0))
            self._time_saved_badge.setText(tr("insights.time_saved_min", self._lang, mins=0.0))
            self._apps_used_lbl.setText(tr("insights.apps_count", self._lang, count=0))
            self._streak_title.setText(plural_streak_days(0, self._lang))
            self._longest_streak_lbl.setText(tr("insights.best_streak", self._lang, days=0))
            return

        self._wpm_val.setText(str(ins.wpm))
        self._wpm_gauge.set_metrics(ins.wpm, ins.top_percent)

        self._fixes_val.setText(str(ins.total_fixes))
        self._words_corrected_lbl.setText(
            tr("insights.words_corrected", self._lang, count=ins.words_corrected)
        )
        self._dict_fixes_lbl.setText(
            tr("insights.dict_fixes", self._lang, count=ins.dictionary_fixes)
        )

        self._total_words_val.setText(f"{ins.total_words:,}".replace(",", " "))
        total_min = ins.total_duration_sec / 60.0
        self._desktop_sub_lbl.setText(
            tr("insights.desktop_sub", self._lang, count=ins.total_dictations, mins=total_min)
        )
        if ins.time_saved_min >= 1.0:
            self._time_saved_badge.setText(
                tr("insights.time_saved_min", self._lang, mins=ins.time_saved_min)
            )
        else:
            saved_sec = int(round(ins.time_saved_min * 60.0))
            self._time_saved_badge.setText(
                tr("insights.time_saved_sec", self._lang, secs=saved_sec)
            )

        self._apps_used_lbl.setText(
            tr("insights.apps_count", self._lang, count=ins.unique_apps_count)
        )
        self._usage_bars.set_categories(ins.categories)

        self._streak_title.setText(plural_streak_days(ins.current_streak_days, self._lang))
        self._longest_streak_lbl.setText(
            tr("insights.best_streak", self._lang, days=ins.longest_streak_days)
        )

    # ------------------------------------------------------------------
    # Refresh with latest entries from HistoryStorage
    # ------------------------------------------------------------------
    def refresh(self, entries: list[TranscriptionEntry], *, today: date | None = None) -> UsageInsights:
        ins = compute_usage_insights(entries, today=today, lang=self._lang)
        self._insights = ins
        self._render_metrics()
        self._heatmap.set_data(ins.daily_activity, end_date=today)
        return ins
