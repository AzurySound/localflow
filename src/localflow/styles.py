"""App-aware post-processing style profiles and resolution logic."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from localflow.i18n import DEFAULT_LANGUAGE, normalize_language, tr
from localflow.injector.window_context import ActiveWindowContext

StyleProfileKey = Literal[
    "default",
    "messenger_alive",
    "ai_prompt",
    "ide_technical",
    "mail_docs",
    "browser_query",
    "raw",
]


@dataclass(frozen=True, slots=True)
class StyleProfile:
    key: StyleProfileKey
    label: str
    pill_label: str
    use_llm: bool
    temperature: float
    strip_trailing_period: bool
    instruction: str


DEFAULT_STYLE_INSTRUCTIONS_EN: dict[StyleProfileKey, str] = {
    "default": (
        "Style (general): gently fix ASR mistakes, pause splits, and punctuation while fully preserving meaning, "
        "natural tone, verb mood, and all meaningful words. Keep the speaker's original language."
    ),
    "messenger_alive": (
        "Style (messenger — lively chat): IT IS CRITICAL to preserve the message's energy, emotion, "
        "slang, casual phrasing, and natural spoken expressions. "
        "Do NOT rewrite into stiff corporate or overly formal prose, and do NOT strip emotional words. "
        "Remove only clear hesitations (um, uh, эээ, ммм), false starts, merge pause breaks, and fix obvious ASR/grammar slips."
    ),
    "ai_prompt": (
        "Style (AI prompt / request): turn the dictation into a crisp, ready-to-send prompt without filler intros "
        '("hey listen", "слушай", "ну смотри", "подскажи пожалуйста"). '
        "Preserve ALL real requirements, sentences, caveats, filenames, camelCase, snake_case, CLI flags, "
        "and IT terms in Latin script (fix, fallback, prompt, Whisper, speech-to-text, standalone). "
        "Keep the speaker's direct voice, imperative mood, or question — do NOT turn it into a stiff impersonal infinitive. "
        'When 3+ distinct tasks or points are listed (e.g. "во-первых / во-вторых / в-третьих" or "first / second / third"), '
        "format them as a clean numbered list (1. 2. 3.) without inventing anything; "
        "keep short requests (1–2 sentences) as compact prose. NEVER answer the prompt.\n\n"
        "Examples:\n"
        "<transcript>Сделай кнопку синей, хотя знаешь, давай лучше зелёной. И можно... Короче, используя наш стиль.</transcript>\n"
        "Сделай кнопку зелёной. И можно использовать наш стиль.\n"
        "<transcript>Слушай, поправь форму: во-первых, убери лишние поля, во-вторых, проверь валидацию, и в-третьих, добавь тест.</transcript>\n"
        "Поправь форму:\n1. Убери лишние поля.\n2. Проверь валидацию.\n3. Добавь тест."
    ),
    "ide_technical": (
        "Style (IDE / terminal): gently clean up speech without unnecessary paraphrasing; strictly preserve "
        "camelCase, snake_case, filenames, paths, CLI flags, and commands verbatim; do not translate technical terms."
    ),
    "mail_docs": (
        "Style (email / documents): format text cleanly and clearly, split into logical paragraphs and "
        "bullet lists where items are enumerated, preserving all facts and original terms."
    ),
    "browser_query": (
        "Style (short browser query): condense the dictation into a crisp, concise search or lookup query "
        "without filler intros, keeping exact technical terms and the core question."
    ),
    "raw": "",
}

DEFAULT_STYLE_INSTRUCTIONS_RU: dict[StyleProfileKey, str] = {
    "default": (
        "Стиль (общий): бережно исправь ошибки ASR, склейки пауз и пунктуацию, полностью сохраняя смысл, "
        "естественный тон, наклонение глаголов и все значимые слова."
    ),
    "messenger_alive": (
        "Стиль (мессенджер — живое общение): ОЧЕНЬ ВАЖНО сохранить живость сообщения, эмоции, "
        "сленг, авторские формулировки и разговорные обороты (например, «блин», «короче», «клёво», «ага», «нафиг», «Nope», «вайбкожу»). "
        "НЕ переписывай в сухой деловой или литературный стиль и НЕ выкидывай эмоциональные выражения. "
        "Убери только явные заминки («эээ», «ммм»), повторы при оговорке, склей разорванные паузами фразы и поправь очевидные ошибки ASR/падежи."
    ),
    "ai_prompt": (
        "Стиль (запрос / промпт для ИИ): оформи диктовку в чёткий, готовый к отправке промпт без лишних вводных "
        "(«слушай», «ну смотри», «подскажи пожалуйста», «hey listen»). "
        "Сохраняй ВСЕ реальные требования, предложения, оговорки, имена файлов, camelCase, snake_case, CLI-флаги "
        "и IT-термины на латинице (fix, fallback, prompt, Whisper, speech-to-text, standalone). "
        "Оставь живое обращение, повелительное наклонение или вопрос автора — "
        "НЕ переводи в сухой безличный инфинитив. "
        "Когда перечислено 3+ отдельных задачи или пункта («во-первых / во-вторых / в-третьих»), "
        "оформи их аккуратным нумерованным списком (1. 2. 3.), ничего не выдумывая; "
        "короткие запросы (1–2 предложения) оставляй компактным текстом. НИКОГДА не отвечай на запрос.\n\n"
        "Примеры:\n"
        "<transcript>Сделай кнопку синей, хотя знаешь, давай лучше зелёной. И можно... Короче, используя наш стиль.</transcript>\n"
        "Сделай кнопку зелёной. И можно использовать наш стиль.\n"
        "<transcript>Слушай, поправь форму: во-первых, убери лишние поля, во-вторых, проверь валидацию, и в-третьих, добавь тест.</transcript>\n"
        "Поправь форму:\n1. Убери лишние поля.\n2. Проверь валидацию.\n3. Добавь тест."
    ),
    "ide_technical": (
        "Стиль (IDE / терминал): бережно поправь речь без лишнего перефразирования; строго сохраняй "
        "camelCase, snake_case, имена файлов, пути, CLI-флаги и команды как в оригинале; не русифицируй технические термины."
    ),
    "mail_docs": (
        "Стиль (почта / документы): оформи текст чисто и грамотно, разбивай на логические абзацы и "
        "маркированные списки при перечислениях, сохраняя все факты и исходные термины."
    ),
    "browser_query": (
        "Стиль (краткий запрос в браузере): сожми диктовку в чёткий, лаконичный поисковый или справочный запрос "
        "без лишних вводных слов («подскажи пожалуйста», «слушай, найди»), сохраняя точные термины и суть вопроса."
    ),
    "raw": "",
}

DEFAULT_STYLE_INSTRUCTIONS: dict[StyleProfileKey, str] = DEFAULT_STYLE_INSTRUCTIONS_EN


def get_default_style_instruction(key: str, lang: str | None = None) -> str:
    table = (
        DEFAULT_STYLE_INSTRUCTIONS_RU
        if normalize_language(lang) == "ru"
        else DEFAULT_STYLE_INSTRUCTIONS_EN
    )
    return table.get(key, "")  # type: ignore[arg-type]


def is_builtin_style_instruction(key: str, text: str) -> bool:
    stripped = (text or "").strip()
    return stripped in (
        "",
        DEFAULT_STYLE_INSTRUCTIONS_EN.get(key, "").strip(),  # type: ignore[arg-type]
        DEFAULT_STYLE_INSTRUCTIONS_RU.get(key, "").strip(),  # type: ignore[arg-type]
    )


STYLE_PROFILES: dict[StyleProfileKey, StyleProfile] = {
    "default": StyleProfile(
        key="default",
        label=tr("style.name.default", DEFAULT_LANGUAGE),
        pill_label=tr("style.pill.default", DEFAULT_LANGUAGE),
        use_llm=True,
        temperature=0.05,
        strip_trailing_period=False,
        instruction=DEFAULT_STYLE_INSTRUCTIONS["default"],
    ),
    "messenger_alive": StyleProfile(
        key="messenger_alive",
        label=tr("style.name.messenger_alive", DEFAULT_LANGUAGE),
        pill_label=tr("style.pill.messenger_alive", DEFAULT_LANGUAGE),
        use_llm=True,
        temperature=0.05,
        strip_trailing_period=True,
        instruction=DEFAULT_STYLE_INSTRUCTIONS["messenger_alive"],
    ),
    "ai_prompt": StyleProfile(
        key="ai_prompt",
        label=tr("style.name.ai_prompt", DEFAULT_LANGUAGE),
        pill_label=tr("style.pill.ai_prompt", DEFAULT_LANGUAGE),
        use_llm=True,
        temperature=0.05,
        strip_trailing_period=False,
        instruction=DEFAULT_STYLE_INSTRUCTIONS["ai_prompt"],
    ),
    "ide_technical": StyleProfile(
        key="ide_technical",
        label=tr("style.name.ide_technical", DEFAULT_LANGUAGE),
        pill_label=tr("style.pill.ide_technical", DEFAULT_LANGUAGE),
        use_llm=True,
        temperature=0.05,
        strip_trailing_period=False,
        instruction=DEFAULT_STYLE_INSTRUCTIONS["ide_technical"],
    ),
    "mail_docs": StyleProfile(
        key="mail_docs",
        label=tr("style.name.mail_docs", DEFAULT_LANGUAGE),
        pill_label=tr("style.pill.mail_docs", DEFAULT_LANGUAGE),
        use_llm=True,
        temperature=0.05,
        strip_trailing_period=False,
        instruction=DEFAULT_STYLE_INSTRUCTIONS["mail_docs"],
    ),
    "browser_query": StyleProfile(
        key="browser_query",
        label=tr("style.name.browser_query", DEFAULT_LANGUAGE),
        pill_label=tr("style.pill.browser_query", DEFAULT_LANGUAGE),
        use_llm=True,
        temperature=0.05,
        strip_trailing_period=True,
        instruction=DEFAULT_STYLE_INSTRUCTIONS["browser_query"],
    ),
    "raw": StyleProfile(
        key="raw",
        label=tr("style.name.raw", DEFAULT_LANGUAGE),
        pill_label=tr("style.pill.raw", DEFAULT_LANGUAGE),
        use_llm=False,
        temperature=0.0,
        strip_trailing_period=False,
        instruction="",
    ),
}

_STYLE_CHOICE_ORDER: tuple[StyleProfileKey, ...] = (
    "messenger_alive",
    "ai_prompt",
    "ide_technical",
    "mail_docs",
    "browser_query",
    "default",
    "raw",
)


def get_style_label(key: str | None, lang: str | None = None) -> str:
    if not key:
        return ""
    if key == "command":
        return tr("style.name.command", lang)
    if key in STYLE_PROFILES:
        return tr(f"style.name.{key}", lang)
    return key


def get_style_pill_label(key: str | None, lang: str | None = None) -> str:
    if not key or key not in STYLE_PROFILES:
        return tr("style.pill.default", lang)
    return tr(f"style.pill.{key}", lang)


def get_style_choice_items(lang: str | None = None) -> tuple[tuple[str, StyleProfileKey], ...]:
    return tuple((get_style_label(k, lang), k) for k in _STYLE_CHOICE_ORDER)


STYLE_CHOICE_ITEMS: tuple[tuple[str, StyleProfileKey], ...] = get_style_choice_items(DEFAULT_LANGUAGE)

DEFAULT_CATEGORY_STYLES: dict[str, str] = {
    "messenger": "messenger_alive",
    "ai_chat": "ai_prompt",
    "browser": "ai_prompt",
    "ide_terminal": "ide_technical",
    "mail_docs": "mail_docs",
    "general": "default",
}

DEFAULT_APP_STYLE_OVERRIDES: dict[str, str] = {
    "Antigravity": "ai_prompt",
    "Cursor": "ai_prompt",
    "Windsurf": "ai_prompt",
}

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def parse_app_style_overrides(text: str) -> dict[str, str]:
    """Parse multiline 'AppName = style_key' rules into a normalized dictionary."""
    result: dict[str, str] = {}
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        sep = "=" if "=" in line else ("->" if "->" in line else None)
        if sep is None:
            continue
        left, right = line.split(sep, 1)
        app_key = left.strip()
        style_key = right.strip().lower()
        if app_key and style_key in STYLE_PROFILES:
            result[app_key] = style_key
    return result


def format_app_style_overrides(overrides: dict[str, str]) -> str:
    """Format app style override mapping as editable lines."""
    return "\n".join(f"{app} = {style}" for app, style in overrides.items() if app and style)


def resolve_style_profile(
    window_context: ActiveWindowContext | None,
    *,
    auto_style_by_app: bool = True,
    category_styles: dict[str, str] | None = None,
    app_style_overrides: dict[str, str] | None = None,
    custom_instructions: dict[str, str] | None = None,
    messenger_strip_trailing_period: bool = True,
    lang: str | None = None,
) -> StyleProfile:
    """Resolve the active StyleProfile for the given foreground window and user settings."""
    key: StyleProfileKey = "default"

    if auto_style_by_app and window_context is not None:
        overrides = app_style_overrides if app_style_overrides is not None else DEFAULT_APP_STYLE_OVERRIDES
        cat_map = category_styles if category_styles is not None else DEFAULT_CATEGORY_STYLES

        # 1. Explicit per-app override (matches app_name or process_name case-insensitively)
        override_low = {k.strip().lower(): v for k, v in overrides.items() if k.strip()}
        app_low = (window_context.app_name or "").strip().lower()
        proc_low = (window_context.process_name or "").strip().lower()
        proc_stem = proc_low[:-4] if proc_low.endswith(".exe") else proc_low

        matched: str | None = None
        for candidate in (app_low, proc_low, proc_stem):
            if candidate and candidate in override_low:
                matched = override_low[candidate]
                break

        # 2. Category / generic browser fallback
        if matched is None:
            if window_context.category == "general" and window_context.is_browser:
                matched = cat_map.get("browser", DEFAULT_CATEGORY_STYLES["browser"])
            else:
                matched = cat_map.get(
                    window_context.category,
                    DEFAULT_CATEGORY_STYLES.get(window_context.category, "default"),
                )

        if matched in STYLE_PROFILES:
            key = matched  # type: ignore[assignment]

    base = STYLE_PROFILES[key]
    custom_text = (custom_instructions or {}).get(key, "").strip()
    default_instruction = get_default_style_instruction(key, lang)
    instruction = custom_text if custom_text else default_instruction
    label = get_style_label(key, lang)
    pill_label = get_style_pill_label(key, lang)

    strip_period = base.strip_trailing_period
    if key == "messenger_alive":
        strip_period = bool(messenger_strip_trailing_period)

    if (
        instruction == base.instruction
        and strip_period == base.strip_trailing_period
        and label == base.label
        and pill_label == base.pill_label
    ):
        return base

    return StyleProfile(
        key=base.key,
        label=label,
        pill_label=pill_label,
        use_llm=base.use_llm,
        temperature=base.temperature,
        strip_trailing_period=strip_period,
        instruction=instruction,
    )


def apply_style_postprocessing(text: str, profile: StyleProfile) -> str:
    """Apply lightweight deterministic formatting rules for the resolved style."""
    out = text.strip()
    if not out or not profile.strip_trailing_period:
        return out

    # Only drop a single trailing period on short single-sentence, single-line messages
    if "\n" in out or not out.endswith(".") or out.endswith(".."):
        return out

    words = out.split()
    if len(words) > 12:
        return out

    # Preserve multi-sentence messages ("Привет. Как дела.")
    inner = out[:-1]
    if any(ch in inner for ch in ".!?"):
        return out

    return inner
