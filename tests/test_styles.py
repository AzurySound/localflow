from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from localflow.config import AppConfig, load_config, save_config
from localflow.injector.window_context import classify_window
from localflow.insights import compute_usage_insights
from localflow.llm.ollama_polisher import OllamaPolisher, build_polish_system_prompt
from localflow.storage import HistoryStorage
from localflow.styles import (
    DEFAULT_CATEGORY_STYLES,
    DEFAULT_STYLE_INSTRUCTIONS,
    STYLE_PROFILES,
    apply_style_postprocessing,
    format_app_style_overrides,
    get_style_label,
    parse_app_style_overrides,
    resolve_style_profile,
)
from localflow.ui.overlay_pill import FloatingPillWidget, PillMode
from localflow.ui.settings_window import SettingsWindow


@pytest.mark.parametrize(
    ("proc", "title", "expected_cat", "expected_browser"),
    [
        ("chrome.exe", "ChatGPT - Google Chrome", "ai_chat", True),
        ("msedge.exe", "Claude — Microsoft Edge", "ai_chat", True),
        ("chrome.exe", "Gemini - Google Chrome", "ai_chat", True),
        ("firefox.exe", "Perplexity — Mozilla Firefox", "ai_chat", True),
        ("brave.exe", "DeepSeek Chat - Brave", "ai_chat", True),
        ("chrome.exe", "Google AI Studio - Google Chrome", "ai_chat", True),
        ("claude.exe", "Claude", "ai_chat", False),
        ("chatgpt.exe", "ChatGPT", "ai_chat", False),
        ("chrome.exe", "GitHub Issues - Google Chrome", "general", True),
        ("chrome.exe", "Telegram Web - Google Chrome", "messenger", True),
        ("chrome.exe", "Inbox - Gmail - Google Chrome", "mail_docs", True),
        ("Antigravity.exe", "chat — Antigravity", "ide_terminal", False),
    ],
)
def test_classify_window_ai_chat_and_is_browser(
    proc: str, title: str, expected_cat: str, expected_browser: bool
) -> None:
    ctx = classify_window(proc, title)
    assert ctx.category == expected_cat
    assert ctx.is_browser is expected_browser


def test_resolve_style_profile_app_routing_and_overrides() -> None:
    tg = classify_window("Telegram.exe", "Telegram")
    assert resolve_style_profile(tg).key == "messenger_alive"
    assert resolve_style_profile(tg).strip_trailing_period is True
    assert (
        resolve_style_profile(tg, messenger_strip_trailing_period=False).strip_trailing_period
        is False
    )

    antigravity = classify_window("Antigravity.exe", "chat — Antigravity")
    assert resolve_style_profile(antigravity).key == "ai_prompt"

    vscode = classify_window("Code.exe", "app.py — Visual Studio Code")
    assert resolve_style_profile(vscode).key == "ide_technical"

    chatgpt = classify_window("chrome.exe", "ChatGPT - Google Chrome")
    assert resolve_style_profile(chatgpt).key == "ai_prompt"

    chrome_generic = classify_window("chrome.exe", "Python docs - Google Chrome")
    assert resolve_style_profile(chrome_generic).key == "ai_prompt"
    assert (
        resolve_style_profile(
            chrome_generic,
            category_styles={**DEFAULT_CATEGORY_STYLES, "browser": "browser_query"},
        ).key
        == "browser_query"
    )

    gmail = classify_window("chrome.exe", "Inbox - Gmail - Google Chrome")
    assert resolve_style_profile(gmail).key == "mail_docs"

    term = classify_window("WindowsTerminal.exe", "PowerShell")
    raw_profile = resolve_style_profile(term, app_style_overrides={"WindowsTerminal": "raw"})
    assert raw_profile.key == "raw"
    assert raw_profile.use_llm is False

    # auto_style_by_app=False forces default style
    assert resolve_style_profile(tg, auto_style_by_app=False).key == "default"
    assert resolve_style_profile(antigravity, auto_style_by_app=False).key == "default"

    # Custom instructions override default instruction
    custom_prof = resolve_style_profile(
        tg,
        custom_instructions={"messenger_alive": "Пиши супер коротко и живо."},
    )
    assert custom_prof.instruction == "Пиши супер коротко и живо."


def test_apply_style_postprocessing_trailing_period_rules() -> None:
    alive = STYLE_PROFILES["messenger_alive"]
    prompt = STYLE_PROFILES["ai_prompt"]

    # Short single sentence drops single trailing period
    assert apply_style_postprocessing("Привет, как дела.", alive) == "Привет, как дела"
    # Preserves ?, !, ...
    assert apply_style_postprocessing("Как дела?", alive) == "Как дела?"
    assert apply_style_postprocessing("Круто!", alive) == "Круто!"
    assert apply_style_postprocessing("Ну такое...", alive) == "Ну такое..."
    # Preserves multi-sentence and multiline
    assert apply_style_postprocessing("Привет. Как дела.", alive) == "Привет. Как дела."
    assert apply_style_postprocessing("Первая\nВторая.", alive) == "Первая\nВторая."
    # Preserves > 12 words
    long_msg = "Один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать тринадцать."
    assert apply_style_postprocessing(long_msg, alive) == long_msg
    # Non-stripping profile preserves trailing period
    assert apply_style_postprocessing("Сделай план.", prompt) == "Сделай план."


def test_parse_and_format_app_style_overrides() -> None:
    raw = "# comment\nAntigravity = ai_prompt\nTerminal -> raw\nBrokenLine\nApp = unknown_style\n"
    parsed = parse_app_style_overrides(raw)
    assert parsed == {"Antigravity": "ai_prompt", "Terminal": "raw"}
    formatted = format_app_style_overrides(parsed)
    assert parse_app_style_overrides(formatted) == parsed


def test_polisher_uses_style_profile_and_skips_http_for_raw() -> None:
    captured_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(payload)
        return httpx.Response(200, json={"message": {"content": "Ок, погнали."}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    polisher = OllamaPolisher(
        system_prompt="Базовый промпт.",
        vocabulary="вайп кожу -> вайбкожу",
        client=client,
    )

    # 1. messenger_alive sends alive instruction, pre-applies vocabulary, and strips trailing period
    tg = classify_window("Telegram.exe", "Telegram")
    alive = resolve_style_profile(tg)
    res_tg = polisher.polish("ну ок я вайп кожу", window_context=tg, style_profile=alive)
    assert res_tg.used_llm is True
    assert res_tg.text == "Ок, погнали"
    assert len(captured_payloads) == 1
    sys_msg = captured_payloads[0]["messages"][0]["content"]
    user_msg = captured_payloads[0]["messages"][1]["content"]
    assert "preserve the message's energy" in sys_msg
    assert "вайбкожу" in user_msg
    assert captured_payloads[0]["options"]["temperature"] == pytest.approx(0.05)

    # 2. ai_prompt uses temperature=0.05 and num_predict >= 256 for multi-word requests
    ai_ctx = classify_window("chrome.exe", "ChatGPT - Google Chrome")
    prompt_prof = resolve_style_profile(ai_ctx)
    res_ai = polisher.polish(
        "сделай план рефакторинга и добавь тесты",
        window_context=ai_ctx,
        style_profile=prompt_prof,
    )
    assert res_ai.used_llm is True
    assert res_ai.text == "Ок, погнали."
    assert captured_payloads[1]["options"]["temperature"] == pytest.approx(0.05)
    assert captured_payloads[1]["options"]["num_predict"] >= 256
    assert "impersonal infinitive" in captured_payloads[1]["messages"][0]["content"]

    # 2b. Ultra-short 1-2 word prompt ("Проанализируй.") falls back to gentle default instruction
    polisher.polish("Проанализируй.", window_context=ai_ctx, style_profile=prompt_prof)
    short_sys = captured_payloads[2]["messages"][0]["content"]
    assert "Style (general)" in short_sys
    assert "3+ distinct" not in short_sys

    # 3. raw profile skips HTTP completely and applies vocabulary replacements
    res_raw = polisher.polish("я сижу вайп кожу", style_profile=STYLE_PROFILES["raw"])
    assert res_raw.used_llm is False
    assert res_raw.text == "я сижу вайбкожу"
    assert len(captured_payloads) == 3

    # 4. build_polish_system_prompt helper
    built = build_polish_system_prompt("Base", window_context=tg, style_profile=alive)
    assert "Active window: Telegram" in built
    assert alive.instruction in built


def test_load_config_auto_migrates_legacy_default_prompts_but_preserves_custom(
    tmp_path: Path,
) -> None:
    from localflow.config import (
        DEFAULT_LLM_SYSTEM_PROMPT,
        DEFAULT_WHISPER_PROMPT,
        _LEGACY_LLM_SYSTEM_PROMPTS,
        _LEGACY_WHISPER_PROMPTS,
    )

    legacy_path = tmp_path / "legacy_config.json"
    legacy_path.write_text(
        json.dumps(
            {
                "llm_system_prompt": next(iter(_LEGACY_LLM_SYSTEM_PROMPTS)),
                "whisper_initial_prompt": next(iter(_LEGACY_WHISPER_PROMPTS)),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    migrated = load_config(legacy_path)
    assert migrated.llm_system_prompt == DEFAULT_LLM_SYSTEM_PROMPT
    assert migrated.whisper_initial_prompt == DEFAULT_WHISPER_PROMPT

    custom_path = tmp_path / "custom_config.json"
    custom_path.write_text(
        json.dumps(
            {
                "llm_system_prompt": "Мой персональный промпт",
                "whisper_initial_prompt": "Мой персональный Whisper промпт",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    kept = load_config(custom_path)
    assert kept.llm_system_prompt == "Мой персональный промпт"
    assert kept.whisper_initial_prompt == "Мой персональный Whisper промпт"


def test_history_storage_migrates_legacy_db_and_stores_style_profile(tmp_path: Path) -> None:
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
            used_llm INTEGER NOT NULL,
            app_category TEXT NOT NULL DEFAULT 'general',
            app_name TEXT NOT NULL DEFAULT '',
            is_command INTEGER NOT NULL DEFAULT 0,
            dict_fixes INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT INTO transcriptions (created_at, duration_sec, raw_text, final_text, used_llm) VALUES (?, ?, ?, ?, ?)",
        ("2026-11-20T10:00:00+00:00", 2.0, "old raw", "Old final", 1),
    )
    conn.commit()
    conn.close()

    storage = HistoryStorage(db_path=db_path)
    storage.add_entry(
        duration_sec=3.0,
        raw_text="привет",
        final_text="Привет",
        llm_model="gemma4:e4b",
        used_llm=True,
        app_category="ai_chat",
        app_name="ChatGPT",
        style_profile="ai_prompt",
    )

    entries = storage.list_recent(limit=10)
    assert len(entries) == 2
    assert entries[0].style_profile == "ai_prompt"
    assert entries[0].app_category == "ai_chat"
    assert entries[1].final_text == "Old final"
    assert entries[1].style_profile == ""

    # Insights maps ai_chat into IDE & AI prompts bucket
    insights = compute_usage_insights(storage.list_all())
    ide_bucket = next(b for b in insights.categories if b.key == "ide")
    assert ide_bucket.count == 1


def test_overlay_pill_show_polishing_custom_label(qtbot) -> None:
    pill = FloatingPillWidget()
    qtbot.addWidget(pill)

    pill.show_polishing("Чат")
    assert pill._mode == PillMode.POLISHING
    assert pill._processing_label == "Чат"

    pill.show_polishing("Промпт")
    assert pill._processing_label == "Промпт"

    pill.show_polishing()
    assert pill._processing_label == "Polishing"


def test_settings_window_styles_roundtrip_and_no_horizontal_overflow(
    tmp_path: Path, monkeypatch, qtbot
) -> None:
    monkeypatch.setattr("localflow.ui.settings_window.list_input_devices", lambda: [])
    storage = HistoryStorage(db_path=tmp_path / "history.db")
    storage.add_entry(
        duration_sec=2.4,
        raw_text="привет как дела",
        final_text="Привет, как дела",
        llm_model="gemma4:e4b",
        used_llm=True,
        app_category="messenger",
        app_name="Telegram",
        style_profile="messenger_alive",
    )

    win = SettingsWindow(AppConfig(), storage)
    qtbot.addWidget(win)
    win.refresh_history()
    assert get_style_label("messenger_alive") in win._history_list.item(0).text()
    assert "Style: Casual (Messenger)" in win._inspector_meta.text()
    assert "Telegram" in win._inspector_meta.text()

    # Load a very long Ollama model name and ensure zero horizontal overflow across all config pages
    win._on_models_loaded(
        ["gemma4:e4b", "SetneufPT/Qwen3.8-27B_Q4_100k_24GB-GPU:latest"]
    )
    win.resize(960, 630)
    win.show()

    win._navigate_to("vocab")
    qtbot.wait(30)
    assert win._tabs.currentWidget() is win._vocab_tab
    assert win._vocab_scroll.horizontalScrollBar().maximum() == 0

    win._navigate_to("style")
    qtbot.wait(30)
    assert win._tabs.currentWidget() is win._style_tab
    assert win._style_scroll.horizontalScrollBar().maximum() == 0
    assert win._styles_card.minimumSizeHint().width() <= 440

    win._navigate_to("settings")
    qtbot.wait(30)
    assert win._tabs.currentWidget() is win._settings_tab
    assert win._settings_scroll.horizontalScrollBar().maximum() == 0

    win._navigate_to("style")

    # Modify style settings and save
    win._messenger_strip_period_check.setChecked(False)
    browser_combo = win._cat_style_combos["browser"]
    browser_combo.setCurrentIndex(browser_combo.findData("browser_query"))
    win._app_overrides_edit.setPlainText("Antigravity = ai_prompt\nTerminal = raw")
    win._style_instruction_edit.setPlainText("Мой кастомный стиль для мессенджера.")

    with qtbot.waitSignal(win.config_saved, timeout=1000) as blocker:
        win._save_btn.click()

    saved: AppConfig = blocker.args[0]
    assert saved.auto_style_by_app is True
    assert saved.messenger_strip_trailing_period is False
    assert saved.category_styles["browser"] == "browser_query"
    assert saved.app_style_overrides == {"Antigravity": "ai_prompt", "Terminal": "raw"}
    assert saved.style_custom_instructions == {
        "messenger_alive": "Мой кастомный стиль для мессенджера."
    }

    cfg_path = tmp_path / "config.json"
    save_config(saved, cfg_path)
    reloaded = load_config(cfg_path)

    win2 = SettingsWindow(reloaded, storage)
    qtbot.addWidget(win2)
    assert win2._messenger_strip_period_check.isChecked() is False
    assert win2._cat_style_combos["browser"].currentData() == "browser_query"
    assert parse_app_style_overrides(win2._app_overrides_edit.toPlainText()) == {
        "Antigravity": "ai_prompt",
        "Terminal": "raw",
    }
    assert win2._style_instruction_edit.toPlainText() == "Мой кастомный стиль для мессенджера."

    # Reset current style instruction restores default
    win2._reset_style_btn.click()
    assert win2._style_instruction_edit.toPlainText() == DEFAULT_STYLE_INSTRUCTIONS["messenger_alive"]


def test_controller_e2e_app_aware_styles_messenger_and_raw_override(
    tmp_path: Path, monkeypatch, qapp, qtbot
) -> None:
    from test_controller_e2e import _ControllerHarness

    harness = _ControllerHarness(tmp_path, monkeypatch)
    cfg = AppConfig(
        ollama_warmup_on_start=False,
        custom_vocabulary="вайп кожу -> вайбкожу",
        app_style_overrides={"Antigravity": "ai_prompt", "WindowsTerminal": "raw"},
    )
    ctrl = harness.create_controller(qapp, qtbot, initial_config=cfg)
    try:
        # 1. Telegram -> messenger_alive, strips trailing period, saves style_profile
        harness.active_window_ctx = classify_window("Telegram.exe", "Telegram")
        ctrl._asr.transcribe_fn = lambda audio, prompt: "короче погнали сегодня в кино"
        harness.ollama_handler = lambda payload: httpx.Response(
            200, json={"message": {"content": "Короче, погнали сегодня в кино."}}
        )
        ctrl._hotkey.toggle_requested.emit()
        ctrl._recorder.next_chunked_duration = 1.2
        ctrl._recorder.next_tail_audio = np.ones(8000, dtype=np.float32) * 0.05
        ctrl._hotkey.toggle_requested.emit()

        qtbot.waitUntil(lambda: ctrl._state.value == "idle", timeout=3000)
        assert len(harness.ollama_requests) == 1
        assert "preserve the message's energy" in harness.ollama_requests[0]["messages"][0]["content"]
        assert ctrl._injector.inject_calls[-1]["text"] == "Короче, погнали сегодня в кино"
        recent = ctrl._storage.list_recent(limit=1)
        assert recent[0].style_profile == "messenger_alive"

        # 2. WindowsTerminal override=raw -> skips Ollama, applies vocabulary replacement
        harness.active_window_ctx = classify_window("WindowsTerminal.exe", "PowerShell")
        ctrl._asr.transcribe_fn = lambda audio, prompt: "я вайп кожу утилиту"
        ctrl._hotkey.toggle_requested.emit()
        ctrl._recorder.next_chunked_duration = 1.1
        ctrl._recorder.next_tail_audio = np.ones(8000, dtype=np.float32) * 0.05
        ctrl._hotkey.toggle_requested.emit()

        qtbot.waitUntil(lambda: ctrl._state.value == "idle", timeout=3000)
        assert len(harness.ollama_requests) == 1  # No new Ollama call!
        assert ctrl._injector.inject_calls[-1]["text"] == "я вайбкожу утилиту"
        recent = ctrl._storage.list_recent(limit=1)
        assert recent[0].style_profile == "raw"
        assert recent[0].used_llm is False
    finally:
        ctrl.shutdown()
        ctrl._pill.close()
        ctrl._settings.close()
        ctrl._tray.deleteLater()
        qapp.processEvents()
