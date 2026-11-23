from __future__ import annotations

from datetime import date, datetime, timezone
import sqlite3
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from localflow.config import AppConfig
from localflow.insights import compute_usage_insights, count_word_fixes
from localflow.storage import HistoryStorage, TranscriptionEntry
from localflow.ui.insights_page import InsightsPage
from localflow.ui.settings_window import SettingsWindow


def test_count_word_fixes_basic() -> None:
    assert count_word_fixes("Привет мир", "Привет мир!") == 0
    # Removed 2 filler words ("ну", "типа")
    assert count_word_fixes("ну типа привет мир", "Привет мир.") == 2
    # Replaced "среду" -> "пятницу"
    assert count_word_fixes("встреча в среду", "Встреча в пятницу.") == 1


def test_compute_usage_insights_wpm_fixes_categories_and_streaks() -> None:
    d_today = date(2026, 11, 22)
    entries = [
        TranscriptionEntry(
            id=1,
            created_at=datetime(2026, 11, 20, 10, 0, tzinfo=timezone.utc).isoformat(),
            duration_sec=30.0,
            raw_text="ну типа привет как дела на работе сегодня вечером",
            final_text="Привет, как дела на работе сегодня вечером?",
            llm_model="gemma4:e4b",
            used_llm=True,
            app_category="messenger",
            app_name="Telegram",
            is_command=False,
            dict_fixes=0,
        ),
        TranscriptionEntry(
            id=2,
            created_at=datetime(2026, 11, 21, 14, 0, tzinfo=timezone.utc).isoformat(),
            duration_sec=20.0,
            raw_text="сделай рефакторинг курс в файле",
            final_text="Сделай рефакторинг курсор в файле.",
            llm_model="gemma4:e4b",
            used_llm=True,
            app_category="ide_terminal",
            app_name="Antigravity",
            is_command=False,
            dict_fixes=1,
        ),
        TranscriptionEntry(
            id=3,
            created_at=datetime(2026, 11, 22, 16, 0, tzinfo=timezone.utc).isoformat(),
            duration_sec=10.0,
            raw_text="переведи на английский",
            final_text="Translate to English please",
            llm_model="gemma4:e4b",
            used_llm=True,
            app_category="ide_terminal",
            app_name="VS Code",
            is_command=True,
            dict_fixes=0,
        ),
    ]

    ins = compute_usage_insights(entries, today=d_today)
    assert ins.total_dictations == 3
    assert ins.total_words == 16  # 7 + 5 + 4
    assert ins.wpm == 16  # 16 words / 60s * 60
    assert ins.words_corrected == 2  # "ну", "типа"
    assert ins.dictionary_fixes == 1  # "курс -> курсор"
    assert ins.total_fixes == 3
    assert ins.current_streak_days == 3
    assert ins.longest_streak_days == 3
    assert ins.unique_apps_count == 3
    assert ins.llm_usage_pct == 100

    cat_by_key = {c.key: c for c in ins.categories}
    assert cat_by_key["messenger"].count == 1
    assert cat_by_key["ide"].count == 1
    assert cat_by_key["command"].count == 1
    assert sum(c.percent for c in ins.categories) in (99, 100, 101)


def test_storage_migrates_legacy_schema_without_losing_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            duration_sec REAL NOT NULL,
            raw_text TEXT NOT NULL,
            final_text TEXT NOT NULL,
            llm_model TEXT,
            used_llm INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        INSERT INTO transcriptions (created_at, duration_sec, raw_text, final_text, llm_model, used_llm)
        VALUES ('2026-11-22T12:00:00+00:00', 4.0, 'привет', 'Привет.', 'gemma4:e4b', 1)
        """
    )
    conn.commit()
    conn.close()

    storage = HistoryStorage(db_path=db_path)
    rows = storage.list_all()
    assert len(rows) == 1
    assert rows[0].final_text == "Привет."
    assert rows[0].app_category == "general"
    assert rows[0].dict_fixes == 0

    storage.add_entry(
        duration_sec=5.0,
        raw_text="тест",
        final_text="Тест.",
        llm_model="gemma4:e4b",
        used_llm=True,
        app_category="ide_terminal",
        app_name="Antigravity",
        dict_fixes=2,
    )
    rows2 = storage.list_all()
    assert len(rows2) == 2
    assert rows2[0].app_category == "ide_terminal"
    assert rows2[0].app_name == "Antigravity"
    assert rows2[0].dict_fixes == 2


def test_insights_page_renders_and_navigates_from_settings_window(tmp_path: Path, monkeypatch, qtbot) -> None:
    monkeypatch.setattr("localflow.ui.settings_window.list_input_devices", lambda: [])
    storage = HistoryStorage(db_path=tmp_path / "history.db")
    storage.add_entry(
        duration_sec=6.0,
        raw_text="ну типа давай сделаем релиз",
        final_text="Давай сделаем релиз.",
        llm_model="gemma4:e4b",
        used_llm=True,
        app_category="messenger",
        app_name="Telegram",
        dict_fixes=1,
    )

    win = SettingsWindow(AppConfig(ollama_model="gemma4:e4b"), storage)
    qtbot.addWidget(win)
    win._navigate_to("insights")
    assert win._tabs.currentWidget() is win._insights_tab
    assert win._nav_insights.isChecked() is True
    assert win._insights_tab._total_words_val.text() == "3"
    assert win._insights_tab._fixes_val.text() == "2"

    # Single-page Usage view without sub-tabs or Voice & models tab
    assert not hasattr(win._insights_tab, "_tab_usage_btn")
    assert not hasattr(win._insights_tab, "_tab_voice_btn")
    assert not hasattr(win._insights_tab, "_stack")

    # Render onto QPixmap without error
    win.show()
    pix = QPixmap(win._insights_tab.size())
    pix.fill(Qt.GlobalColor.transparent)
    win._insights_tab.render(pix)
    win.close()
