"""Local LLM post-processing via Ollama HTTP API."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import httpx

from localflow.injector.window_context import (
    ActiveWindowContext,
    WindowCategory,
    classify_window,
    format_window_context_line,
)
from localflow.styles import (
    DEFAULT_STYLE_INSTRUCTIONS,
    StyleProfile,
    apply_style_postprocessing,
    get_standalone_ai_prompt_system,
    is_builtin_style_instruction,
    strip_final_period,
)
from localflow.vocabulary import (
    ParsedVocabulary,
    apply_vocabulary_replacements,
    format_vocabulary_for_llm,
    parse_vocabulary,
)

_CATEGORY_INSTRUCTIONS: dict[WindowCategory, str] = {
    "ide_terminal": (
        "Format (IDE/terminal): strictly preserve camelCase, snake_case, filenames, "
        "CLI flags, and commands verbatim; do not translate technical terms."
    ),
    "messenger": (
        "Format (messenger): use a natural conversational tone without stiff formality, preserving lively speech and emotion."
    ),
    "mail_docs": (
        "Format (email/documents): structure text cleanly into paragraphs and bullet lists when items are listed."
    ),
    "ai_chat": (
        "Format (AI chat/prompt): package the dictation as a clear structured prompt without filler; "
        "keep all technical terms and never answer the prompt itself."
    ),
    "general": "Format (general): neutral, clean, natural style.",
}

_TRANSFORM_SYSTEM_PROMPT = (
    "You are a precise text editor. Transform <selected_text> strictly following the spoken instruction in <command>.\n"
    "Rules:\n"
    "1. Interpret minor voice-recognition typos in <command> by intent (e.g. 'на форме как промпт' = 'оформи как промпт').\n"
    "2. If asked to condense/tighten ('ужми', 'сократи', 'condense', 'for prompt'), actively remove fluff and filler while keeping all key facts.\n"
    "3. Never use long dashes ('—' or '–'); use commas, colons, or a short hyphen '-' instead.\n"
    "4. Keep original language unless translation is requested. Return ONLY the transformed text: no tags, quotes, or commentary."
)

_KNOWN_CATEGORIES: frozenset[str] = frozenset(
    {"ide_terminal", "messenger", "mail_docs", "ai_chat", "general"}
)
_CATEGORY_MARKER_RE = re.compile(r"\bcategory=(ide_terminal|messenger|mail_docs|ai_chat|general)\b")


_SYSTEM_LEAK_MARKERS: tuple[str, ...] = (
    "active window:",
    "активное окно:",
    "category=ide_terminal",
    "category=messenger",
    "category=mail_docs",
    "category=ai_chat",
    "category=general",
    "core rules:",
    "базовые правила:",
    "обязательные автозамены:",
    "style (ai prompt",
    "style (general):",
    "style (messenger",
    "стиль (общий):",
    "стиль (мессенджер",
    "стиль (запрос / промпт",
    "examples:\n",
    "примеры:\n",
)

_TRANSCRIPT_WRAPPER_RE = re.compile(
    r"^\s*<transcript>\s*(.*?)\s*</transcript>\s*$",
    re.DOTALL | re.IGNORECASE,
)


def format_polish_user_message(text: str) -> str:
    """Wrap raw ASR transcript in explicit tags so small LLMs edit instead of continuing."""
    return f"<transcript>\n{text}\n</transcript>"


_LONG_DASH_CHARS = "—–―‒"
_COMMA_LONG_DASH_RE = re.compile(r",\s*[—–―‒]\s*")
_DIGIT_LONG_DASH_RE = re.compile(r"(?<=\d)[—–―‒](?=\d)")
_WORD_LONG_DASH_RE = re.compile(r"(?<=\w)[—―](?=\w)")
_ANY_LONG_DASH_RE = re.compile(r"[—–―‒]")


def strip_long_dashes(text: str) -> str:
    """Replace long typographic dashes (—, –, ―, ‒) with commas or short hyphens (-)."""
    if not text or not any(ch in text for ch in _LONG_DASH_CHARS):
        return text
    out = _COMMA_LONG_DASH_RE.sub(", ", text)
    out = _DIGIT_LONG_DASH_RE.sub("-", out)
    out = _WORD_LONG_DASH_RE.sub(" - ", out)
    out = _ANY_LONG_DASH_RE.sub("-", out)
    return out


def is_local_ollama_url(url: str) -> bool:
    """Return True when url points to localhost / loopback."""
    try:
        host = (urlparse(url).hostname or "").strip().lower()
    except Exception:
        return False
    return host in {"127.0.0.1", "localhost", "::1"}


def find_ollama_launch_command() -> list[str] | None:
    """Locate the Ollama desktop app or CLI binary to start Ollama locally."""
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            app_exe = Path(local_app_data) / "Programs" / "Ollama" / "ollama app.exe"
            if app_exe.is_file():
                return [str(app_exe)]
            cli_exe = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
            if cli_exe.is_file():
                return [str(cli_exe), "serve"]
    which = shutil.which("ollama")
    if which:
        return [which, "serve"]
    return None


def launch_ollama_background(cmd: list[str] | None = None) -> bool:
    """Start Ollama detached in the background; returns True if spawn succeeded."""
    launch_cmd = cmd if cmd is not None else find_ollama_launch_command()
    if not launch_cmd:
        return False
    kwargs: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200) | getattr(
            subprocess, "DETACHED_PROCESS", 0x00000008
        )
        if len(launch_cmd) > 1 and launch_cmd[1] == "serve":
            flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        kwargs["creationflags"] = flags
    try:
        subprocess.Popen(launch_cmd, **kwargs)  # type: ignore[arg-type]
        return True
    except Exception:
        return False


def strip_transcript_wrapper(content: str) -> str:
    """Remove accidental <transcript>...</transcript> wrapper echoed by small models."""
    match = _TRANSCRIPT_WRAPPER_RE.match(content)
    if match:
        return match.group(1).strip()
    return content.strip()


_BYPASS_CUE_RE = re.compile(
    r"(?:^|\b)(?:"
    r"ну|типа|короче|в\s+общем|как\s+его|как\s+бы|то\s+есть|ой\s+нет|вернее|"
    r"ээ+|мм+|um+|uh+|erm+|like|you\s+know|i\s+mean|actually"
    r")(?:\b|$)",
    re.IGNORECASE,
)


def should_bypass_llm(text: str, style_profile: StyleProfile | None = None) -> bool:
    """Return True when a short, clean transcript needs zero LLM cleanup."""
    if style_profile is not None and not style_profile.use_llm:
        return True
    stripped = (text or "").strip()
    if not stripped:
        return True
    words = stripped.split()
    if len(words) > 5:
        return False
    if "..." in stripped or "…" in stripped:
        return False
    if _BYPASS_CUE_RE.search(stripped):
        return False
    return True


def validate_polish_output(
    content: str,
    prepared_text: str,
    *,
    finish_reason: str | None = None,
    style_key: str | None = None,
) -> str | None:
    """Return an error reason if LLM polish output leaked system prompt, truncated, or hallucinated."""
    if finish_reason == "length":
        return "LLM output truncated (finish_reason=length)"

    low = content.lower()
    for marker in _SYSTEM_LEAK_MARKERS:
        if marker in low:
            return f"LLM output leaked system prompt ({marker})"

    # Guard against severe mid-sentence truncation on multi-clause / long inputs.
    # For ai_prompt we explicitly ask the LLM to condense verbose speech (~35-50% shorter),
    # so allow compression down to 0.25 of source length.
    src_len = len(prepared_text.strip())
    min_ratio = 0.25 if style_key == "ai_prompt" else 0.45
    if src_len >= 80 and len(content.strip()) < src_len * min_ratio:
        return "LLM output unexpectedly truncated input text"

    # Guard against small-model continuation hallucinations (e.g. 4-word input -> 400-char essay)
    if src_len > 0 and len(content) > max(src_len * 2.5 + 48, 120):
        return "LLM output unexpectedly expanded input text"
    return None


def _normalize_vocabulary(vocabulary: ParsedVocabulary | str | None) -> ParsedVocabulary:
    if vocabulary is None:
        return ParsedVocabulary(terms=(), replacements=())
    if isinstance(vocabulary, ParsedVocabulary):
        return vocabulary
    return parse_vocabulary(vocabulary)


def _resolve_window_context(
    window_context: ActiveWindowContext | str | None,
) -> tuple[str, WindowCategory | None]:
    if window_context is None:
        return ("", None)
    if isinstance(window_context, ActiveWindowContext):
        return (format_window_context_line(window_context), window_context.category)

    single_line = re.sub(r"[\r\n\t]+", " ", window_context)
    single_line = re.sub(r"\s+", " ", single_line).strip()
    if not single_line:
        return ("", None)

    low = single_line.lower()
    if low in _KNOWN_CATEGORIES:
        return (f"Активное окно: категория {low}", cast(WindowCategory, low))

    match = _CATEGORY_MARKER_RE.search(low)
    if match:
        category = cast(WindowCategory, match.group(1))
    else:
        category = classify_window("", single_line).category
    return (single_line, category)


def build_polish_system_prompt(
    base_prompt: str,
    *,
    window_context: ActiveWindowContext | str | None = None,
    vocabulary: ParsedVocabulary | str | None = None,
    style_profile: StyleProfile | None = None,
    lang: str | None = None,
) -> str:
    """Build system prompt enriched with active window context, style profile, and personal vocabulary."""
    style_instruction = style_profile.instruction.strip() if style_profile is not None else ""
    if style_profile is not None and isinstance(window_context, ActiveWindowContext):
        # When an explicit style profile is resolved, avoid noisy process/title/category metadata
        # that distracts small local models and increases prompt-leak risk.
        app_name = (window_context.app_name or "").strip()
        ctx_line = f"Active window: {app_name}" if app_name else ""
        category = window_context.category
    else:
        ctx_line, category = _resolve_window_context(window_context)
    vocab = _normalize_vocabulary(vocabulary)
    vocab_block = format_vocabulary_for_llm(vocab, lang=lang).strip()

    use_standalone_ai_prompt = (
        style_profile is not None
        and style_profile.key == "ai_prompt"
        and is_builtin_style_instruction("ai_prompt", style_profile.instruction)
    )
    if use_standalone_ai_prompt:
        parts = [get_standalone_ai_prompt_system(lang).strip()]
        if ctx_line:
            parts.append(ctx_line)
        if vocab_block:
            parts.append(vocab_block)
        return "\n\n".join(parts)

    if not ctx_line and category is None and not vocab_block and not style_instruction:
        return base_prompt

    parts: list[str] = []
    if base_prompt.strip():
        parts.append(base_prompt.strip())
    if ctx_line:
        parts.append(ctx_line)
    if style_instruction:
        parts.append(style_instruction)
    elif category is not None:
        parts.append(_CATEGORY_INSTRUCTIONS[category])
    if vocab_block:
        parts.append(vocab_block)
    return "\n\n".join(parts)


def _build_transform_system_prompt(
    *,
    window_context: ActiveWindowContext | str | None = None,
    vocabulary: ParsedVocabulary | str | None = None,
    lang: str | None = None,
) -> str:
    return build_polish_system_prompt(
        _TRANSFORM_SYSTEM_PROMPT,
        window_context=window_context,
        vocabulary=vocabulary,
        lang=lang,
    )


@dataclass(frozen=True, slots=True)
class PolishResult:
    text: str
    used_llm: bool
    error: str | None = None


class OllamaPolisher:
    """Cleans up raw ASR transcripts using a local Ollama model with keep-alive pooling."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5:14b",
        system_prompt: str = "",
        timeout_sec: float = 35.0,
        keep_alive: str = "10m",
        num_ctx: int = 4096,
        num_gpu: int | None = None,
        client: httpx.Client | None = None,
        vocabulary: ParsedVocabulary | str | None = None,
        lang: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.timeout_sec = timeout_sec
        self.keep_alive = keep_alive
        self.num_ctx = max(2048, int(num_ctx))
        self.num_gpu = num_gpu
        self.vocabulary = _normalize_vocabulary(vocabulary)
        self.lang = lang
        self._warmup_lock = threading.Lock()
        self._warming = False
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else httpx.Client(timeout=self.timeout_sec, trust_env=False)
        )

    def close(self) -> None:
        """Close the internal HTTP connection pool if owned by this instance."""
        if self._owns_client:
            try:
                self._client.close()
            except Exception:
                pass

    def list_models(self) -> list[str]:
        """Query /api/tags to list locally installed Ollama models."""
        resp = self._client.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        return [str(m["name"]) for m in models if "name" in m]

    def is_reachable(self, timeout: float = 0.5) -> bool:
        """Return True if the Ollama HTTP server responds cleanly within timeout seconds."""
        try:
            resp = self._client.get(f"{self.base_url}/api/version", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def _build_options(
        self,
        raw_text: str | None = None,
        *,
        num_predict: int | None = None,
        temperature: float = 0.05,
    ) -> dict[str, object]:
        opts: dict[str, object] = {
            "temperature": temperature,
            "num_ctx": self.num_ctx,
        }
        if num_predict is not None:
            opts["num_predict"] = num_predict
        elif raw_text is not None:
            opts["num_predict"] = max(256, min(2048, len(raw_text) * 2))
        if self.num_gpu is not None:
            opts["num_gpu"] = self.num_gpu
        return opts

    def warmup(self) -> None:
        """Pre-load the model into memory so the first dictation is instant."""
        with self._warmup_lock:
            if self._warming:
                return
            self._warming = True
        try:
            self._client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "",
                    "keep_alive": self.keep_alive,
                    "options": self._build_options(),
                },
                timeout=60.0,
            )
        except Exception:
            pass
        finally:
            with self._warmup_lock:
                self._warming = False

    def unload(self) -> None:
        """Ask Ollama to immediately evict the current model from VRAM."""
        try:
            self._client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": "", "keep_alive": 0},
                timeout=5.0,
            )
        except Exception:
            pass

    def polish(
        self,
        raw_text: str,
        *,
        window_context: ActiveWindowContext | str | None = None,
        style_profile: StyleProfile | None = None,
    ) -> PolishResult:
        """Clean up raw_text via Ollama; falls back to raw_text on any error."""
        stripped = raw_text.strip()
        if not stripped:
            return PolishResult(text="", used_llm=False)

        prepared_text = (
            apply_vocabulary_replacements(stripped, self.vocabulary.replacements)
            if self.vocabulary.replacements
            else stripped
        )
        if style_profile is not None and not style_profile.use_llm:
            return PolishResult(
                text=apply_style_postprocessing(prepared_text, style_profile),
                used_llm=False,
            )

        # Protect ultra-short 1-2 word commands ("Проанализируй.") from prompt-packaging rewrites
        effective_style = style_profile
        is_short_prompt_guard = (
            style_profile is not None
            and style_profile.key == "ai_prompt"
            and len(prepared_text.split()) <= 2
        )
        if is_short_prompt_guard and style_profile is not None:
            effective_style = StyleProfile(
                key=style_profile.key,
                label=style_profile.label,
                pill_label=style_profile.pill_label,
                use_llm=style_profile.use_llm,
                temperature=style_profile.temperature,
                strip_trailing_period=style_profile.strip_trailing_period,
                instruction=DEFAULT_STYLE_INSTRUCTIONS["default"],
            )

        system_content = build_polish_system_prompt(
            self.system_prompt,
            window_context=window_context,
            vocabulary=self.vocabulary,
            style_profile=effective_style,
            lang=self.lang,
        )
        temperature = effective_style.temperature if effective_style is not None else 0.05
        num_predict = None
        if effective_style is not None and effective_style.key == "ai_prompt" and not is_short_prompt_guard:
            num_predict = max(256, min(2048, len(prepared_text) * 3))

        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": self._build_options(
                prepared_text, num_predict=num_predict, temperature=temperature
            ),
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": format_polish_user_message(prepared_text)},
            ],
        }

        try:
            resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message") if isinstance(data, dict) else None
            raw_content = msg.get("content") if isinstance(msg, dict) else None
            done_reason = data.get("done_reason") if isinstance(data, dict) else None
            finish_reason = done_reason if isinstance(done_reason, str) else None
            content = strip_transcript_wrapper(raw_content) if isinstance(raw_content, str) else ""
            if not content:
                return PolishResult(text=prepared_text, used_llm=False, error="Empty LLM response")
            leak_err = validate_polish_output(
                content,
                prepared_text,
                finish_reason=finish_reason,
                style_key=effective_style.key if effective_style is not None else None,
            )
            if leak_err is not None:
                return PolishResult(text=prepared_text, used_llm=False, error=leak_err)
            if self.vocabulary.replacements:
                replaced = apply_vocabulary_replacements(
                    content, self.vocabulary.replacements
                ).strip()
                if replaced:
                    content = replaced
            if effective_style is not None:
                content = apply_style_postprocessing(content, effective_style)
            else:
                content = strip_final_period(content)
            content = strip_long_dashes(content)
            return PolishResult(text=content, used_llm=True)
        except Exception as exc:
            return PolishResult(text=prepared_text, used_llm=False, error=str(exc))

    def transform_selection(
        self,
        selected_text: str,
        command: str,
        *,
        window_context: ActiveWindowContext | str | None = None,
    ) -> PolishResult:
        """Transform selected_text according to a voice command; never falls back to command."""
        if not selected_text.strip():
            return PolishResult(text="", used_llm=False, error="Empty selected text")

        clean_command = command.strip()
        if not clean_command:
            return PolishResult(text="", used_llm=False, error="Empty command")

        num_predict = max(256, min(2048, len(selected_text) * 3))
        system_content = _build_transform_system_prompt(
            window_context=window_context,
            vocabulary=self.vocabulary,
            lang=self.lang,
        )
        user_content = (
            f"<selected_text>\n{selected_text}\n</selected_text>\n\n"
            f"<command>\n{clean_command}\n</command>"
        )

        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": self._build_options(num_predict=num_predict),
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
        }

        try:
            resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message") if isinstance(data, dict) else None
            raw_content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(raw_content, str) or not raw_content.strip():
                return PolishResult(text="", used_llm=False, error="Empty LLM response")
            if "\n" in raw_content or selected_text[:1] in {" ", "\t"}:
                content = raw_content.strip("\r\n")
            else:
                content = raw_content.strip()
            content = strip_long_dashes(content)
            return PolishResult(text=content, used_llm=True)
        except Exception as exc:
            return PolishResult(text="", used_llm=False, error=str(exc))

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        """Run a general-purpose chat completion (for Notetaker summary & Q&A)."""
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": self._build_options(num_predict=max_tokens, temperature=temperature),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        resp = self._client.post(f"{self.base_url}/api/chat", json=payload, timeout=max(self.timeout_sec, 60.0))
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message") if isinstance(data, dict) else None
        raw_content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(raw_content, str):
            return ""
        return strip_long_dashes(raw_content.strip())
