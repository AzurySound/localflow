"""Internationalization (i18n) catalog and helpers for LocalFlow (English primary + Russian)."""

from __future__ import annotations

from typing import Literal

LanguageCode = Literal["en", "ru"]
DEFAULT_LANGUAGE: LanguageCode = "en"

LANGUAGE_CHOICE_ITEMS: tuple[tuple[str, LanguageCode], ...] = (
    ("English", "en"),
    ("Русский", "ru"),
)

_MONTHS: dict[LanguageCode, tuple[str, ...]] = {
    "en": ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"),
    "ru": ("Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"),
}

_WEEKDAYS: dict[LanguageCode, tuple[str, ...]] = {
    "en": ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"),
    "ru": ("Вс", "Пн", "Вт", "Ср", "Чт", "Пт", "Сб"),
}


def normalize_language(lang: str | None) -> LanguageCode:
    """Normalize a language string into 'en' (default) or 'ru'."""
    if (lang or "").strip().lower().startswith("ru"):
        return "ru"
    return "en"


def get_months(lang: str | None = None) -> tuple[str, ...]:
    return _MONTHS[normalize_language(lang)]


def get_weekdays(lang: str | None = None) -> tuple[str, ...]:
    return _WEEKDAYS[normalize_language(lang)]


def plural_streak_days(n: int, lang: str | None = None) -> str:
    code = normalize_language(lang)
    if code == "ru":
        n_abs = abs(n)
        if n_abs % 10 == 1 and n_abs % 100 != 11:
            return f"{n} день подряд"
        if 2 <= n_abs % 10 <= 4 and not (12 <= n_abs % 100 <= 14):
            return f"{n} дня подряд"
        return f"{n} дней подряд"
    return f"{n} day streak" if abs(n) == 1 else f"{n} days streak"


TRANSLATIONS: dict[LanguageCode, dict[str, str]] = {
    "en": {
        # Window & Navigation
        "window.title": "LocalFlow — Hub & Settings",
        "nav.history": "◫   History",
        "nav.insights": "▁▃▆  Insights",
        "nav.vocab": "▤   Dictionary",
        "nav.style": "Tt  Style && Prompt",
        "nav.settings": "⚙   Settings",
        # Sidebar cards
        "sidebar.profile_hdr": "PROFILE",
        "sidebar.hotkeys_hdr": "HOTKEYS",
        "sidebar.asr_status": "ASR: {model}",
        "sidebar.llm_local": "LLM: {model} (local)",
        "sidebar.llm_ollama": "LLM: {model}",
        "sidebar.llm_off": "LLM: off",
        "sidebar.hk_start": "●  Start: 2× CapsLock",
        "sidebar.hk_stop": "●  Stop: 1× CapsLock",
        "sidebar.hk_cancel": "●  Cancel: Esc",
        "sidebar.hk_command": "●  Command: Ctrl + 2× Caps",
        # Theme choices
        "theme.system": "System (auto)",
        "theme.dark": "Dark",
        "theme.light": "Light",
        # VRAM profiles (Settings dropdown)
        "profile.light_gpu.long": "Light GPU (~3 GB VRAM: int8_float16 + auto-unload)",
        "profile.quality.long": "Quality (~12 GB VRAM: float16 + warmup)",
        "profile.zero_vram.long": "0 VRAM / CPU (0 MB VRAM: Whisper CPU + Ollama CPU)",
        "profile.custom.long": "Custom",
        # VRAM profiles (Menu)
        "profile.quality.short": "Quality (~12 GB VRAM)",
        "profile.light_gpu.short": "Light GPU (~3 GB VRAM)",
        "profile.zero_vram.short": "0 VRAM / CPU (0 MB)",
        # History page
        "history.title": "History",
        "history.refresh": "Refresh",
        "history.delete": "Delete",
        "history.delete_count": "Delete ({count})",
        "history.delete_tooltip": "Delete selected entries (Ctrl/Shift + click, Delete)",
        "history.clear": "Clear",
        "history.filter_all": "All",
        "history.filter_llm": "LLM Cleaned",
        "history.filter_raw": "RAW",
        "history.search_placeholder": "Search...",
        "history.stats": "Dictations: {count} • Words: {words}",
        "history.inspector_title": "Dictation Details",
        "history.inspector_empty_meta": "Select an entry from the list",
        "history.inspector_overview": "OVERVIEW",
        "history.inspector_placeholder": "Final text and raw speech transcript (ASR) will appear here...",
        "history.copy_text": "Copy text",
        "history.multi_title": "Selected entries: {count}",
        "history.multi_meta": "Words: {words} · Duration: {sec:.1f}s",
        "history.multi_hint": (
            "Click \"Delete\", press the Delete key, or right-click "
            "to remove the selected entries from History and Insights."
        ),
        "history.type_command": "Voice Command",
        "history.type_dictation": "Dictation",
        "history.type_raw": "Raw Dictation",
        "history.meta_style": "Style: {label}",
        "history.preview_body": "Final text:\n{final}\n\nRaw ASR:\n{raw}",
        "history.ctx_copy": "Copy text",
        "history.ctx_delete_one": "Delete entry",
        "history.ctx_delete_many": "Delete selected ({count})",
        "history.confirm_delete_title": "Delete from history?",
        "history.confirm_delete_one": "Delete the selected entry?",
        "history.confirm_delete_many": "Delete selected entries ({count})?",
        "history.confirm_delete_info": (
            "The selected dictations and commands will be removed from History and Usage Insights."
        ),
        "history.confirm_clear_title": "Clear history?",
        "history.confirm_clear_text": "Delete all dictation history ({count} entries)?",
        "history.confirm_clear_info": (
            "All saved dictations and Usage Insights statistics will be permanently deleted."
        ),
        "common.cancel": "Cancel",
        # Insights page
        "insights.title": "Insights",
        "insights.wpm_hdr": "WORDS PER MINUTE",
        "insights.ai_fixes_hdr": "AI FIXES",
        "insights.words_corrected": "<b>{count}</b> words corrected",
        "insights.dict_fixes": "<b>{count}</b> dictionary fixes",
        "insights.total_words_hdr": "TOTAL WORDS DICTATED",
        "insights.desktop": "🖥  Desktop",
        "insights.desktop_sub": "{count} dict. · {mins:.1f} min",
        "insights.time_saved_min": "~{mins:.0f} min saved",
        "insights.time_saved_sec": "~{secs} s saved",
        "insights.usage_title": "Desktop Usage",
        "insights.apps_count": "APPS | {count}",
        "insights.best_streak": "BEST | {days} D.",
        "insights.prev_period": "Previous period",
        "insights.next_period": "Next period",
        "insights.no_data": "No data",
        "insights.no_data_yet": "No data to display yet",
        "insights.less": "Less",
        "insights.more": "More",
        "insights.tooltip_active": "{date}: {dictations} dict. · {words} words",
        "insights.tooltip_empty": "{date}: no dictations",
        "insights.cat.messenger": "MESSENGERS",
        "insights.cat.ide": "IDE & AI PROMPTS",
        "insights.cat.docs": "DOCS & MAIL",
        "insights.cat.command": "VOICE COMMANDS",
        "insights.cat.general": "OTHER TASKS",
        # Dictionary page
        "vocab.title": "Dictionary",
        "vocab.subtitle": (
            "Personal terms improve Whisper and LLM recognition, and replacement rules "
            "automatically fix frequent mistakes."
        ),
        "vocab.card_hdr": "TERMS & AUTO-REPLACEMENTS",
        "vocab.label": "Dictionary:",
        "vocab.placeholder": "Antigravity, Wispr Flow, Ollama, PySide6\ncurse -> cursor",
        "vocab.hint": (
            "List terms separated by commas or new lines. "
            "For auto-replacement use the format: typo -> correct (or →)."
        ),
        # Style & Prompt page
        "style.title": "Style & Prompt",
        "style.subtitle": "Automatic style adaptation per active app and base LLM system prompt.",
        "style.card_hdr": "APP-AWARE AUTO STYLES",
        "style.auto_check": "Automatically change style based on the active app",
        "style.strip_period_check": "Drop trailing period for short messenger replies",
        "style.cat.messenger": "Messengers:",
        "style.cat.ai_chat": "AI Chats:",
        "style.cat.browser": "Browser:",
        "style.cat.ide_terminal": "IDE & Terminal:",
        "style.cat.mail_docs": "Mail & Docs:",
        "style.overrides_label": "Overrides:",
        "style.overrides_hint": (
            "App name = messenger_alive | ai_prompt | ide_technical | mail_docs | browser_query | default | raw"
        ),
        "style.reset_btn": "Reset style",
        "style.instruction_label": "Instruction:",
        "style.base_prompt_hdr": "BASE LLM PROMPT",
        "style.prompt_label": "LLM prompt:",
        "style.reset_prompt_btn": "Reset prompt",
        # Style labels & pill labels
        "style.name.default": "Standard",
        "style.name.messenger_alive": "Casual (Messenger)",
        "style.name.ai_prompt": "Prompt Packaging",
        "style.name.ide_technical": "Technical (IDE)",
        "style.name.mail_docs": "Document / Email",
        "style.name.browser_query": "Short Query",
        "style.name.raw": "No LLM (RAW)",
        "style.name.command": "Voice Command",
        "style.pill.default": "Polishing",
        "style.pill.messenger_alive": "Chat",
        "style.pill.ai_prompt": "Prompt",
        "style.pill.ide_technical": "Code",
        "style.pill.mail_docs": "Doc",
        "style.pill.browser_query": "Query",
        "style.pill.raw": "RAW",
        # Settings page
        "settings.title": "Settings",
        "settings.subtitle": "Whisper speech engine, local LLM, microphone, language, appearance, and Flow Bar.",
        "settings.perf_hdr": "PERFORMANCE & ASR",
        "settings.vram_label": "VRAM mode:",
        "settings.whisper_label": "Whisper:",
        "settings.whisper_hint": (
            "Default for a quick start: small (~460 MB). "
            "Recommended for best accuracy: large-v3-turbo (~1.6 GB)."
        ),
        "settings.audio_hdr": "MICROPHONE & ACTIVATION",
        "settings.mic_label": "Microphone:",
        "settings.mic_default": "System default",
        "settings.mic_default_suffix": " (default)",
        "settings.double_tap_label": "CapsLock interval:",
        "settings.ms_suffix": " ms",
        "settings.silence_label": "Silence timer:",
        "settings.silence_check": "Auto-stop after silence",
        "settings.sec_suffix": " s",
        "settings.llm_hdr": "SMART CLEANUP & LLM ENGINE",
        "settings.llm_enable": "Enable smart text cleanup (LLM)",
        "settings.llm_engine_label": "LLM engine:",
        "settings.llm_backend_embedded": "Embedded engine (standalone, no Ollama)",
        "settings.llm_backend_ollama": "External Ollama (localhost)",
        "settings.model_gguf_label": "Model (GGUF):",
        "settings.download_model": "Download model",
        "settings.downloaded_ok": "Downloaded ✓",
        "settings.download_recommended": "Download recommended models",
        "settings.download_recommended_tip": (
            "Downloads the recommended LLM (Gemma 3 4B ~2.5 GB) and switches Whisper to large-v3-turbo"
        ),
        "settings.browse_gguf": "Custom .gguf…",
        "settings.custom_gguf_placeholder": "Optional: path to your local .gguf file",
        "settings.browse_gguf_title": "Select a local GGUF model",
        "settings.ollama_url_label": "Ollama URL:",
        "settings.ollama_model_label": "Ollama model:",
        "settings.refresh_models": "Refresh models",
        "settings.keep_alive_label": "VRAM keep-alive:",
        "settings.ux_hdr": "FLOW BAR & BEHAVIOR",
        "settings.language_label": "Language:",
        "settings.theme_label": "Appearance:",
        "settings.show_idle_bar": "Show resting Flow Bar",
        "settings.reset_pos": "Reset position",
        "settings.live_preview": "Live word preview above the pill",
        "settings.restore_cb": "Restore clipboard after paste",
        "settings.sounds_check": "Start and finish sound cues",
        "settings.test_sound": "Test sound",
        "settings.autostart": "Launch LocalFlow at Windows startup",
        "settings.save_btn": "Save settings",
        "settings.saved_ok": "Settings saved.",
        "settings.loading_ollama": "Loading Ollama models...",
        "settings.found_ollama": "Found Ollama models: {count}",
        "settings.ollama_err": "Failed to connect to Ollama: {err}",
        "settings.preset_applied": "Applied preset \"{name}\" — click Save.",
        "settings.gguf_selected": "Selected local .gguf — click Save.",
        "settings.model_ready": "Model and engine are ready.",
        "settings.model_no_runtime": "Model downloaded (llama.cpp engine will download on first run).",
        "settings.model_not_downloaded": "Model not downloaded yet — click \"Download model\".",
        "settings.dl_preparing": "Preparing download of {name}…",
        "settings.dl_cancelled": "Download cancelled.",
        "settings.dl_cancelling": "Cancelling download…",
        "settings.dl_error": "Download error: {err}",
        "settings.dl_rec_done": "Recommended models ready ({label} + large-v3-turbo).",
        "settings.dl_one_done": "Model {label} downloaded and activated.",
        "settings.delete_model_title": "Delete model?",
        "settings.delete_model_text": "Delete model file {name}?",
        "settings.delete_model_info": "The GGUF file will be removed from disk. You can re-download it anytime.",
        "settings.model_deleted": "Model {name} deleted.",
        # Embedded model catalog metadata
        "catalog.qwen25-1.5b.badge": "Default · ~1.0 GB",
        "catalog.qwen25-1.5b.desc": "Lightest model for a quick start — runs even on laptops without a GPU.",
        "catalog.gemma3-4b.badge": "Recommended · ~2.5 GB",
        "catalog.gemma3-4b.desc": "Best balance of speed and multilingual accuracy (4–6 GB VRAM or CPU).",
        "catalog.gemma4-e4b.badge": "Max quality · ~5.3 GB",
        "catalog.gemma4-e4b.desc": "Highest quality cleanup, auto-styles, and complex prompts (8+ GB VRAM).",
        # Context menu & Tray
        "menu.asr_ready": "ASR: Ready",
        "menu.asr_loading_generic": "ASR: Loading model...",
        "menu.asr_loading_model": "ASR: Loading {model}...",
        "menu.asr_load_error": "ASR: Load error",
        "menu.microphone": "Microphone",
        "menu.vram_mode": "VRAM Mode",
        "menu.smart_cleanup": "Smart cleanup (LLM)",
        "menu.live_preview": "Live word preview",
        "menu.flow_bar": "Flow Bar",
        "menu.show_idle_bar": "Show resting bar",
        "menu.show_bar_now": "Show bar now",
        "menu.snooze_1h": "Hide for 1 hour",
        "menu.reset_pos": "Reset position",
        "menu.copy_to_clipboard": "Copy to clipboard",
        "menu.copy_tooltip": "Copy: {snippet}",
        "menu.settings": "Settings",
        "menu.quit": "Quit",
        "tray.tooltip": "LocalFlow — Double-tap CapsLock to dictate",
        # Overlay pill & Controller notices
        "pill.transcribing": "Transcribing",
        "pill.polishing": "Polishing",
        "pill.command": "Command",
        "pill.copy_tooltip": "Copy last dictation",
        "notice.asr_error": "ASR error: {err}",
        "notice.no_last_dictation": "No saved dictation",
        "notice.last_copied": "Last dictation copied",
        "notice.model_loading": "Model is still loading...",
        "notice.select_text_cmd": "Select text for a command",
        "notice.mic_error": "Mic error: {err}",
        "notice.cancelled": "Cancelled",
        "notice.cmd_not_recognized": "Command not recognized",
        "notice.speech_not_recognized": "Speech not recognized",
        "notice.llm_cmd_unavailable": "LLM unavailable — command cancelled",
        "notice.error": "Error: {err}",
        "notice.download_llm_raw": "Download LLM in Settings — raw text",
        "notice.embedded_llm_raw": "Embedded LLM error — raw text",
        "notice.ollama_raw": "Ollama unavailable — raw text pasted",
        "notice.mic_switched": "Mic: {name}",
        "notice.mic_system_short": "System",
        "notice.mode_switched": "Mode: {profile}",
        # Downloader stages
        "dl.stage_default": "Downloading",
        "dl.stage_search_llama": "Finding llama.cpp engine...",
        "dl.stage_llama": "Downloading LLM engine (llama.cpp)",
        "dl.stage_extract_llama": "Extracting LLM engine...",
        "dl.stage_model": "Downloading {name}",
    },
    "ru": {
        # Window & Navigation
        "window.title": "LocalFlow — Hub и настройки",
        "nav.history": "◫   История",
        "nav.insights": "▁▃▆  Insights",
        "nav.vocab": "▤   Словарь",
        "nav.style": "Tt  Стиль и промпт",
        "nav.settings": "⚙   Настройки",
        # Sidebar cards
        "sidebar.profile_hdr": "ПРОФИЛЬ",
        "sidebar.hotkeys_hdr": "ГОРЯЧИЕ КЛАВИШИ",
        "sidebar.asr_status": "ASR: {model}",
        "sidebar.llm_local": "LLM: {model} (лок.)",
        "sidebar.llm_ollama": "LLM: {model}",
        "sidebar.llm_off": "LLM: выключена",
        "sidebar.hk_start": "●  Старт: 2× CapsLock",
        "sidebar.hk_stop": "●  Стоп: 1× CapsLock",
        "sidebar.hk_cancel": "●  Отмена: Esc",
        "sidebar.hk_command": "●  Команда: Ctrl + 2× Caps",
        # Theme choices
        "theme.system": "Как в системе (авто)",
        "theme.dark": "Тёмная",
        "theme.light": "Светлая",
        # VRAM profiles (Settings dropdown)
        "profile.light_gpu.long": "Лёгкий GPU (~3 GB VRAM: int8_float16 + автовыгрузка)",
        "profile.quality.long": "Качество (~12 GB VRAM: float16 + прогрев)",
        "profile.zero_vram.long": "0 VRAM / CPU (0 MB VRAM: Whisper CPU + Ollama CPU)",
        "profile.custom.long": "Пользовательский",
        # VRAM profiles (Menu)
        "profile.quality.short": "Качество (~12 GB VRAM)",
        "profile.light_gpu.short": "Лёгкий GPU (~3 GB VRAM)",
        "profile.zero_vram.short": "0 VRAM / CPU (0 MB)",
        # History page
        "history.title": "История",
        "history.refresh": "Обновить",
        "history.delete": "Удалить",
        "history.delete_count": "Удалить ({count})",
        "history.delete_tooltip": "Удалить выбранные записи (Ctrl/Shift + клик, Delete)",
        "history.clear": "Очистить",
        "history.filter_all": "Все",
        "history.filter_llm": "С очисткой LLM",
        "history.filter_raw": "RAW",
        "history.search_placeholder": "Поиск...",
        "history.stats": "Диктовок: {count} • Слов: {words}",
        "history.inspector_title": "Детали диктовки",
        "history.inspector_empty_meta": "Выберите запись из списка",
        "history.inspector_overview": "ОБЗОР",
        "history.inspector_placeholder": "Здесь отобразится финальный текст и исходная расшифровка (ASR)...",
        "history.copy_text": "Копировать текст",
        "history.multi_title": "Выбрано записей: {count}",
        "history.multi_meta": "Слов: {words} · Длительность: {sec:.1f}s",
        "history.multi_hint": (
            "Нажмите «Удалить», клавишу Delete или правую кнопку мыши, "
            "чтобы удалить выбранные записи из истории и Insights."
        ),
        "history.type_command": "Голосовая команда",
        "history.type_dictation": "Диктовка",
        "history.type_raw": "Сырая диктовка",
        "history.meta_style": "Стиль: {label}",
        "history.preview_body": "Финальный текст:\n{final}\n\nСырой ASR:\n{raw}",
        "history.ctx_copy": "Копировать текст",
        "history.ctx_delete_one": "Удалить запись",
        "history.ctx_delete_many": "Удалить выбранное ({count})",
        "history.confirm_delete_title": "Удалить из истории?",
        "history.confirm_delete_one": "Удалить выбранную запись?",
        "history.confirm_delete_many": "Удалить выбранные записи ({count} шт.)?",
        "history.confirm_delete_info": (
            "Выбранные диктовки и команды будут удалены из истории и статистики Usage Insights."
        ),
        "history.confirm_clear_title": "Очистить историю?",
        "history.confirm_clear_text": "Удалить всю историю диктовок ({count} зап.)?",
        "history.confirm_clear_info": (
            "Все сохранённые диктовки и статистика Usage Insights будут удалены без возможности восстановления."
        ),
        "common.cancel": "Отмена",
        # Insights page
        "insights.title": "Insights",
        "insights.wpm_hdr": "СЛОВ В МИНУТУ",
        "insights.ai_fixes_hdr": "ИСПРАВЛЕНИЙ AI",
        "insights.words_corrected": "<b>{count}</b> слов исправлено",
        "insights.dict_fixes": "<b>{count}</b> правок словаря",
        "insights.total_words_hdr": "ВСЕГО ПРОДИКТОВАНО СЛОВ",
        "insights.desktop": "🖥  Десктоп",
        "insights.desktop_sub": "{count} дикт. · {mins:.1f} мин",
        "insights.time_saved_min": "~{mins:.0f} мин сэкономлено",
        "insights.time_saved_sec": "~{secs} с сэкономлено",
        "insights.usage_title": "Использование на ПК",
        "insights.apps_count": "ПРИЛОЖЕНИЙ | {count}",
        "insights.best_streak": "РЕКОРД | {days} ДН.",
        "insights.prev_period": "Предыдущий период",
        "insights.next_period": "Следующий период",
        "insights.no_data": "Нет данных",
        "insights.no_data_yet": "Пока нет данных для отображения",
        "insights.less": "Меньше",
        "insights.more": "Больше",
        "insights.tooltip_active": "{date}: {dictations} дикт. · {words} слов",
        "insights.tooltip_empty": "{date}: нет диктовок",
        "insights.cat.messenger": "МЕССЕНДЖЕРЫ",
        "insights.cat.ide": "IDE И AI-ПРОМПТЫ",
        "insights.cat.docs": "ДОКУМЕНТЫ И ПОЧТА",
        "insights.cat.command": "ГОЛОСОВЫЕ КОМАНДЫ",
        "insights.cat.general": "ПРОЧИЕ ЗАДАЧИ",
        # Dictionary page
        "vocab.title": "Словарь",
        "vocab.subtitle": (
            "Персональные термины улучшают распознавание Whisper и LLM, а правила замен "
            "автоматически исправляют частые ошибки."
        ),
        "vocab.card_hdr": "ТЕРМИНЫ И АВТОЗАМЕНЫ",
        "vocab.label": "Словарь:",
        "vocab.placeholder": "Antigravity, Wispr Flow, Ollama, PySide6\nкурс -> курсор",
        "vocab.hint": (
            "Термины перечисляйте через запятую или с новой строки. "
            "Для автозамены используйте формат: ошибка -> правильно (или →)."
        ),
        # Style & Prompt page
        "style.title": "Стиль и промпт",
        "style.subtitle": "Автоматическая смена стиля по активному приложению и базовый системный промпт LLM.",
        "style.card_hdr": "АВТО-СТИЛИ ПО ПРИЛОЖЕНИЯМ",
        "style.auto_check": "Автоматически менять стиль по активному приложению",
        "style.strip_period_check": "Без точки в конце коротких фраз в мессенджерах",
        "style.cat.messenger": "Мессенджеры:",
        "style.cat.ai_chat": "ИИ-чаты:",
        "style.cat.browser": "Браузер:",
        "style.cat.ide_terminal": "IDE и терминал:",
        "style.cat.mail_docs": "Почта и тексты:",
        "style.overrides_label": "Исключения:",
        "style.overrides_hint": (
            "Имя приложения = messenger_alive | ai_prompt | ide_technical | mail_docs | browser_query | default | raw"
        ),
        "style.reset_btn": "Сбросить стиль",
        "style.instruction_label": "Инструкция:",
        "style.base_prompt_hdr": "БАЗОВЫЙ ПРОМПТ LLM",
        "style.prompt_label": "Промпт LLM:",
        "style.reset_prompt_btn": "Сбросить промпт",
        # Style labels & pill labels
        "style.name.default": "Стандартный",
        "style.name.messenger_alive": "Живой (мессенджер)",
        "style.name.ai_prompt": "Упаковка в промпт",
        "style.name.ide_technical": "Технический (IDE)",
        "style.name.mail_docs": "Документ / Письмо",
        "style.name.browser_query": "Краткий запрос",
        "style.name.raw": "Без LLM (RAW)",
        "style.name.command": "Голосовая команда",
        "style.pill.default": "Очищаю",
        "style.pill.messenger_alive": "Чат",
        "style.pill.ai_prompt": "Промпт",
        "style.pill.ide_technical": "Код",
        "style.pill.mail_docs": "Текст",
        "style.pill.browser_query": "Запрос",
        "style.pill.raw": "RAW",
        # Settings page
        "settings.title": "Настройки",
        "settings.subtitle": "Движок распознавания Whisper, локальная LLM, микрофон, язык, тема и поведение Flow Bar.",
        "settings.perf_hdr": "ПРОИЗВОДИТЕЛЬНОСТЬ И ASR",
        "settings.vram_label": "Режим VRAM:",
        "settings.whisper_label": "Whisper:",
        "settings.whisper_hint": (
            "По умолчанию для быстрого старта: small (~460 MB). "
            "Рекомендуется для максимального качества: large-v3-turbo (~1.6 GB)."
        ),
        "settings.audio_hdr": "МИКРОФОН И АКТИВАЦИЯ",
        "settings.mic_label": "Микрофон:",
        "settings.mic_default": "Системный по умолчанию",
        "settings.mic_default_suffix": " (по умолчанию)",
        "settings.double_tap_label": "Интервал CapsLock:",
        "settings.ms_suffix": " мс",
        "settings.silence_label": "Таймер тишины:",
        "settings.silence_check": "Авто-стоп после тишины",
        "settings.sec_suffix": " с",
        "settings.llm_hdr": "УМНАЯ ОЧИСТКА И ДВИЖОК LLM",
        "settings.llm_enable": "Включить умную очистку текста (LLM)",
        "settings.llm_engine_label": "Движок LLM:",
        "settings.llm_backend_embedded": "Встроенный движок (автономно, без Ollama)",
        "settings.llm_backend_ollama": "Внешняя Ollama (localhost)",
        "settings.model_gguf_label": "Модель (GGUF):",
        "settings.download_model": "Скачать модель",
        "settings.downloaded_ok": "Скачана ✓",
        "settings.download_recommended": "Скачать рекомендуемые модели",
        "settings.download_recommended_tip": (
            "Скачает рекомендуемую LLM (Gemma 3 4B ~2.5 GB) и переключит Whisper на large-v3-turbo"
        ),
        "settings.browse_gguf": "Свой .gguf…",
        "settings.custom_gguf_placeholder": "Необязательно: путь к своему локальному .gguf файлу",
        "settings.browse_gguf_title": "Выберите локальную модель GGUF",
        "settings.ollama_url_label": "Ollama URL:",
        "settings.ollama_model_label": "Модель Ollama:",
        "settings.refresh_models": "Обновить модели",
        "settings.keep_alive_label": "Удержание в VRAM:",
        "settings.ux_hdr": "FLOW BAR И ПОВЕДЕНИЕ",
        "settings.language_label": "Язык интерфейса:",
        "settings.theme_label": "Тема оформления:",
        "settings.show_idle_bar": "Показывать полоску в покое (Flow Bar)",
        "settings.reset_pos": "Сбросить позицию",
        "settings.live_preview": "Живое превью последних слов над пилюлей",
        "settings.restore_cb": "Восстанавливать буфер обмена после вставки",
        "settings.sounds_check": "Звуковые сигналы старта и завершения",
        "settings.test_sound": "Проверить звук",
        "settings.autostart": "Запускать LocalFlow при входе в Windows",
        "settings.save_btn": "Сохранить настройки",
        "settings.saved_ok": "Настройки сохранены.",
        "settings.loading_ollama": "Загрузка списка моделей Ollama...",
        "settings.found_ollama": "Найдено моделей в Ollama: {count}",
        "settings.ollama_err": "Не удалось подключиться к Ollama: {err}",
        "settings.preset_applied": "Применён пресет «{name}» — нажмите Сохранить.",
        "settings.gguf_selected": "Выбран локальный .gguf — нажмите Сохранить.",
        "settings.model_ready": "Модель и движок готовы к работе.",
        "settings.model_no_runtime": "Модель скачана (движок llama.cpp догрузится при первом запуске).",
        "settings.model_not_downloaded": "Модель ещё не скачана — нажмите «Скачать модель».",
        "settings.dl_preparing": "Подготовка загрузки {name}…",
        "settings.dl_cancelled": "Загрузка отменена.",
        "settings.dl_cancelling": "Отмена загрузки…",
        "settings.dl_error": "Ошибка загрузки: {err}",
        "settings.dl_rec_done": "Рекомендуемые модели готовы ({label} + large-v3-turbo).",
        "settings.dl_one_done": "Модель {label} загружена и активирована.",
        "settings.delete_model_title": "Удалить модель?",
        "settings.delete_model_text": "Удалить файл модели {name}?",
        "settings.delete_model_info": "Файл GGUF будет удалён с диска. Вы сможете скачать его повторно в любой момент.",
        "settings.model_deleted": "Модель {name} удалена.",
        # Embedded model catalog metadata
        "catalog.qwen25-1.5b.badge": "По умолчанию · ~1.0 GB",
        "catalog.qwen25-1.5b.desc": "Самая лёгкая модель для быстрого старта — работает даже на ноутбуках без GPU.",
        "catalog.gemma3-4b.badge": "Рекомендуется · ~2.5 GB",
        "catalog.gemma3-4b.desc": "Оптимальный баланс скорости и качества текста (4–6 GB VRAM или CPU).",
        "catalog.gemma4-e4b.badge": "Максимум качества · ~5.3 GB",
        "catalog.gemma4-e4b.desc": "Лучшее качество коррекции, авто-стилей и сложных промптов (8+ GB VRAM).",
        # Context menu & Tray
        "menu.asr_ready": "ASR: Готов",
        "menu.asr_loading_generic": "ASR: Загрузка модели...",
        "menu.asr_loading_model": "ASR: Загрузка {model}...",
        "menu.asr_load_error": "ASR: Ошибка загрузки",
        "menu.microphone": "Микрофон",
        "menu.vram_mode": "Режим VRAM",
        "menu.smart_cleanup": "Умная очистка (LLM)",
        "menu.live_preview": "Живое превью слов",
        "menu.flow_bar": "Flow Bar (полоска)",
        "menu.show_idle_bar": "Показывать полоску в покое",
        "menu.show_bar_now": "Показать полоску сейчас",
        "menu.snooze_1h": "Скрыть на 1 час",
        "menu.reset_pos": "Сбросить позицию",
        "menu.copy_to_clipboard": "Скопировать в буфер",
        "menu.copy_tooltip": "Копировать: {snippet}",
        "menu.settings": "Настройки",
        "menu.quit": "Выход",
        "tray.tooltip": "LocalFlow — 2× CapsLock для диктовки",
        # Overlay pill & Controller notices
        "pill.transcribing": "Распознаю",
        "pill.polishing": "Очищаю",
        "pill.command": "Команда",
        "pill.copy_tooltip": "Скопировать последнюю диктовку",
        "notice.asr_error": "Ошибка ASR: {err}",
        "notice.no_last_dictation": "Нет сохранённой диктовки",
        "notice.last_copied": "Последняя диктовка скопирована",
        "notice.model_loading": "Модель ещё загружается...",
        "notice.select_text_cmd": "Выделите текст для команды",
        "notice.mic_error": "Микрофон: {err}",
        "notice.cancelled": "Отменено",
        "notice.cmd_not_recognized": "Команда не распознана",
        "notice.speech_not_recognized": "Речь не распознана",
        "notice.llm_cmd_unavailable": "LLM недоступна — команда отменена",
        "notice.error": "Ошибка: {err}",
        "notice.download_llm_raw": "Скачайте LLM в Настройках — сырой текст",
        "notice.embedded_llm_raw": "Ошибка встроенной LLM — сырой текст",
        "notice.ollama_raw": "Ollama недоступна — вставлен сырой текст",
        "notice.mic_switched": "Микрофон: {name}",
        "notice.mic_system_short": "Системный",
        "notice.mode_switched": "Режим: {profile}",
        # Downloader stages
        "dl.stage_default": "Загрузка",
        "dl.stage_search_llama": "Поиск движка llama.cpp...",
        "dl.stage_llama": "Загрузка движка LLM (llama.cpp)",
        "dl.stage_extract_llama": "Распаковка движка LLM...",
        "dl.stage_model": "Загрузка {name}",
    },
}


def tr(key: str, lang: str | None = None, **kwargs: object) -> str:
    """Translate key into the requested language ('en' primary, 'ru' secondary)."""
    code = normalize_language(lang)
    table = TRANSLATIONS.get(code, TRANSLATIONS["en"])
    template = table.get(key)
    if template is None:
        template = TRANSLATIONS["en"].get(key, key)
    if kwargs:
        return template.format(**kwargs)
    return template
