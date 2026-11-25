"""Regression tests for Smart Cleanup evaluation improvements (P0, P1, P2)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx

from localflow.config import (
    DEFAULT_LLM_SYSTEM_PROMPT_EN,
    DEFAULT_WHISPER_PROMPT,
    AppConfig,
    create_first_run_config,
    load_config,
)
from localflow.llm.embedded_polisher import EmbeddedLlamaPolisher
from localflow.llm.llama_server import LlamaServerManager
from localflow.llm.ollama_polisher import (
    OllamaPolisher,
    should_bypass_llm,
    validate_polish_output,
)
from localflow.storage import HistoryStorage
from localflow.styles import STYLE_PROFILES
from localflow.ui.context_menu import QuickMenuCallbacks, QuickMenuState, build_quick_menu
from localflow.ui.settings_window import SettingsWindow
from localflow.vocabulary import (
    apply_vocabulary_replacements_with_count,
    format_vocabulary_for_llm,
    parse_vocabulary,
)


def test_config_migrations_and_ctx_floors(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    legacy_payload = {
        "ollama_num_ctx": 1024,
        "whisper_initial_prompt": (
            "API, backend, frontend, PR, commit, Docker, Kubernetes, Python, React, TypeScript, CapsLock."
        ),
        "language": "en",
    }
    cfg_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    loaded = load_config(cfg_path)
    assert loaded.ollama_num_ctx == 4096
    assert loaded.whisper_initial_prompt == DEFAULT_WHISPER_PROMPT
    assert "CapsLock" not in loaded.whisper_initial_prompt
    assert "Non-execution" in DEFAULT_LLM_SYSTEM_PROMPT_EN

    first_gpu = create_first_run_config(has_cuda=True)
    assert first_gpu.ollama_num_ctx == 4096
    assert first_gpu.ollama_keep_alive == "10m"

    mgr = LlamaServerManager(
        binary_path=tmp_path / "llama-server.exe",
        model_path=tmp_path / "model.gguf",
        num_ctx=1024,
    )
    assert mgr.num_ctx == 2048


def test_validate_polish_output_rejects_truncation_and_length_finish() -> None:
    # Real case ID 197: 378 chars -> 28 chars
    raw_long = (
        "Вот, в конце октября заболел ОРВИ, проболел где-то полторы недели, стал лучше себя чувствовать, "
        "но через неделю появился стабильный кашель, и каждый день я с тех пор кашляю, хотя сейчас тенденция "
        "такая, что я кашляю меньше, но после переезда снова больше."
    )
    err_shrink = validate_polish_output("В конце октября заболел ОРВИ", raw_long)
    assert err_shrink is not None and "truncated" in err_shrink.lower()

    err_len = validate_polish_output("Хороший текст.", "Хороший текст для проверки.", finish_reason="length")
    assert err_len is not None and "length" in err_len.lower()

    # Legitimate short self-correction should pass
    assert validate_polish_output("Он упал после этого?", "Ну, то есть он упал, да, после этого?") is None


def test_ollama_and_embedded_reject_finish_reason_length(tmp_path: Path) -> None:
    raw_text = "Это достаточно длинное предложение для проверки обрезки контекста при генерации ответа."

    def ollama_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"content": "Это достаточно длинное"}, "done_reason": "length"},
        )

    ol_polisher = OllamaPolisher(
        client=httpx.Client(transport=httpx.MockTransport(ollama_handler)),
    )
    ol_res = ol_polisher.polish(raw_text)
    assert not ol_res.used_llm
    assert ol_res.text == raw_text
    assert ol_res.error and "length" in ol_res.error

    fake_mgr = SimpleNamespace(
        ensure_started=lambda: "http://127.0.0.1:9999",
        touch_activity=lambda: None,
    )

    def emb_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Это достаточно длинное"}, "finish_reason": "length"}]
            },
        )

    emb_polisher = EmbeddedLlamaPolisher(
        server_manager=fake_mgr,  # type: ignore[arg-type]
        client=httpx.Client(transport=httpx.MockTransport(emb_handler)),
    )
    emb_res = emb_polisher.polish(raw_text)
    assert not emb_res.used_llm
    assert emb_res.text == raw_text
    assert emb_res.error and "length" in emb_res.error


def test_should_bypass_llm_fast_path() -> None:
    assert should_bypass_llm("Давай.", STYLE_PROFILES["ai_prompt"]) is True
    assert should_bypass_llm("Привет, как дела?", STYLE_PROFILES["messenger_alive"]) is True
    assert should_bypass_llm("Ну, то есть он упал?", STYLE_PROFILES["ai_prompt"]) is False
    assert should_bypass_llm("Давай... проверим.", STYLE_PROFILES["ai_prompt"]) is False
    assert (
        should_bypass_llm(
            "Сделай кнопку зелёной и проверь валидацию формы.",
            STYLE_PROFILES["ai_prompt"],
        )
        is False
    )


def test_vocabulary_wildcard_stems_count_and_localization() -> None:
    vocab = parse_vocabulary("Antigravity\nвайпкод* -> вайбкод*")
    out, count = apply_vocabulary_replacements_with_count(
        "Я сегодня весь день вайпкодил и хочу вайпкодить дальше.",
        vocab.replacements,
    )
    assert out == "Я сегодня весь день вайбкодил и хочу вайбкодить дальше."
    assert count == 2

    en_block = format_vocabulary_for_llm(vocab, lang="en")
    assert en_block.startswith("Preferred terms & spelling: Antigravity.")
    assert "Mandatory replacements:" in en_block

    ru_block = format_vocabulary_for_llm(vocab, lang="ru")
    assert ru_block.startswith("Предпочитаемые термины и написание: Antigravity.")
    assert "Обязательные автозамены:" in ru_block


def test_history_storage_update_and_inspector_diff_html(qapp, tmp_path: Path) -> None:
    db = HistoryStorage(tmp_path / "history.db")
    entry_id = db.add_entry(
        duration_sec=5.0,
        raw_text="ну типа привет мир",
        final_text="ну типа привет мир",
        llm_model=None,
        used_llm=False,
    )
    db.update_entry_cleanup(
        entry_id,
        final_text="Привет, мир.",
        llm_model="gemma4-e4b",
        used_llm=True,
    )
    entries = db.list_recent(limit=5)
    assert len(entries) == 1
    assert entries[0].final_text == "Привет, мир."
    assert entries[0].used_llm is True
    assert entries[0].llm_model == "gemma4-e4b"

    diff_html = SettingsWindow.format_history_inspector_html(
        "ну типа привет мир",
        "Привет, мир.",
        lang="en",
        theme="dark",
    )
    assert "line-through" in diff_html
    assert "CHANGES:" in diff_html
    assert "Привет, мир." in diff_html


def test_context_menu_copy_raw_asr_visibility_and_callback(qapp) -> None:
    copied: list[str] = []
    callbacks = QuickMenuCallbacks(
        on_select_mic=lambda _idx: None,
        on_select_profile=lambda _p: None,
        on_toggle_llm=lambda _e: None,
        on_toggle_live_preview=lambda _e: None,
        on_toggle_idle_bar=lambda _e: None,
        on_snooze_1h=lambda: None,
        on_unsnooze=lambda: None,
        on_reset_position=lambda: None,
        on_copy_last_dictation=lambda: copied.append("final"),
        on_copy_last_raw=lambda: copied.append("raw"),
        on_open_settings=lambda: None,
        on_quit=lambda: None,
    )

    menu_same = build_quick_menu(
        QuickMenuState("ASR ready", None, "light_gpu", True, True, True, False, "Hello", "Hello"),
        callbacks,
        lang="en",
    )
    assert not any(a.text() == "Copy Raw ASR" for a in menu_same.actions())

    menu_diff = build_quick_menu(
        QuickMenuState(
            "ASR ready", None, "light_gpu", True, True, True, False, "Hello.", "um hello"
        ),
        callbacks,
        lang="en",
    )
    raw_actions = [a for a in menu_diff.actions() if a.text() == "Copy Raw ASR"]
    assert len(raw_actions) == 1
    raw_actions[0].trigger()
    assert copied == ["raw"]
