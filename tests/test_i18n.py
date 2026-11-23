from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel

from localflow.config import (
    AppConfig,
    DEFAULT_LLM_SYSTEM_PROMPT_EN,
    DEFAULT_LLM_SYSTEM_PROMPT_RU,
    create_first_run_config,
    load_config,
    save_config,
)
from localflow.i18n import (
    DEFAULT_LANGUAGE,
    TRANSLATIONS,
    normalize_language,
    plural_streak_days,
    tr,
)
from localflow.storage import HistoryStorage
from localflow.ui.overlay_pill import FloatingPillWidget
from localflow.ui.settings_window import SettingsWindow


def test_i18n_catalog_key_parity_and_defaults() -> None:
    en_keys = set(TRANSLATIONS["en"].keys())
    ru_keys = set(TRANSLATIONS["ru"].keys())
    assert en_keys == ru_keys, f"Key mismatch: en-ru={en_keys - ru_keys}, ru-en={ru_keys - en_keys}"

    assert DEFAULT_LANGUAGE == "en"
    assert AppConfig().language == "en"
    assert create_first_run_config(has_cuda=True).language == "en"
    assert create_first_run_config(has_cuda=False).language == "en"

    assert normalize_language(None) == "en"
    assert normalize_language("en") == "en"
    assert normalize_language("ru") == "ru"
    assert normalize_language("ru-RU") == "ru"
    assert normalize_language("de") == "en"

    assert plural_streak_days(0, "en") == "0 days streak"
    assert plural_streak_days(1, "en") == "1 day streak"
    assert plural_streak_days(5, "en") == "5 days streak"
    assert plural_streak_days(1, "ru") == "1 день подряд"
    assert plural_streak_days(2, "ru") == "2 дня подряд"
    assert plural_streak_days(5, "ru") == "5 дней подряд"
    assert plural_streak_days(11, "ru") == "11 дней подряд"


def test_settings_window_live_language_switch_and_persistence(
    tmp_path: Path, monkeypatch, qtbot
) -> None:
    monkeypatch.setattr("localflow.ui.settings_window.list_input_devices", lambda: [])
    storage = HistoryStorage(db_path=tmp_path / "hist.db")
    storage.add_entry(
        duration_sec=3.0,
        raw_text="hello world",
        final_text="Hello world.",
        llm_model="gemma4:e4b",
        used_llm=True,
        app_category="messenger",
        app_name="Telegram",
        style_profile="messenger_alive",
    )

    win = SettingsWindow(AppConfig(), storage)
    qtbot.addWidget(win)
    win.refresh_history()
    win.refresh_insights()
    win._navigate_to("settings")
    win.show()

    # English primary default
    assert win._lang == "en"
    assert win._language_combo.currentData() == "en"
    assert win.windowTitle() == "LocalFlow — Hub & Settings"
    assert "History" in win._nav_history.text()
    assert "Settings" in win._nav_settings.text()
    assert win._save_btn.text() == "Save settings"
    assert win._double_tap_spin.suffix() == " ms"
    assert win._silence_auto_stop_spin.suffix() == " s"
    assert win._prompt_edit.toPlainText() == DEFAULT_LLM_SYSTEM_PROMPT_EN
    assert win._insights_tab._usage_title.text() == "Desktop Usage"
    assert win._settings_scroll.horizontalScrollBar().maximum() == 0

    # Switch live to Russian -> emits preview, retranslates everything + default prompt
    with qtbot.waitSignal(win.language_preview_requested, timeout=1000) as preview_blocker:
        win._language_combo.setCurrentIndex(win._language_combo.findData("ru"))
    assert preview_blocker.args == ["ru"]
    assert win._lang == "ru"
    assert win.windowTitle() == "LocalFlow — Hub и настройки"
    assert "История" in win._nav_history.text()
    assert "Настройки" in win._nav_settings.text()
    assert win._save_btn.text() == "Сохранить настройки"
    assert win._double_tap_spin.suffix() == " мс"
    assert win._silence_auto_stop_spin.suffix() == " с"
    assert win._prompt_edit.toPlainText() == DEFAULT_LLM_SYSTEM_PROMPT_RU
    assert win._insights_tab._usage_title.text() == "Использование на ПК"
    hk_texts = [lbl.text() for lbl in win._hotkeys_card.findChildren(QLabel)]
    assert any("ГОРЯЧИЕ КЛАВИШИ" in t for t in hk_texts)
    assert win._settings_scroll.horizontalScrollBar().maximum() == 0

    # Custom LLM prompt is preserved across language switches
    win._prompt_edit.setPlainText("My custom system prompt")
    win._language_combo.setCurrentIndex(win._language_combo.findData("en"))
    assert win._prompt_edit.toPlainText() == "My custom system prompt"

    # Switch back to Russian, save, and verify config.json roundtrip
    win._language_combo.setCurrentIndex(win._language_combo.findData("ru"))
    with qtbot.waitSignal(win.config_saved, timeout=1000) as save_blocker:
        win._save_btn.click()
    saved: AppConfig = save_blocker.args[0]
    assert saved.language == "ru"
    assert win._save_status.text() == "Настройки сохранены."

    cfg_path = tmp_path / "config.json"
    save_config(saved, cfg_path)
    reloaded = load_config(cfg_path)
    assert reloaded.language == "ru"

    win2 = SettingsWindow(reloaded, storage)
    qtbot.addWidget(win2)
    assert win2._lang == "ru"
    assert win2._language_combo.currentData() == "ru"
    assert win2._save_btn.text() == "Сохранить настройки"


def test_overlay_pill_language_switching(qtbot) -> None:
    pill = FloatingPillWidget()
    qtbot.addWidget(pill)

    pill.show_polishing()
    assert pill._processing_label == tr("pill.polishing", "en")

    pill.set_language("ru")
    pill.show_polishing()
    assert pill._processing_label == tr("pill.polishing", "ru")

    pill.set_language("en")
    pill.show_polishing()
    assert pill._processing_label == "Polishing"
    pill.hide_pill()
