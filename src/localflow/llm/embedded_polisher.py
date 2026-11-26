"""Standalone embedded LLM polisher backed by a managed llama-server.exe process."""

from __future__ import annotations

from pathlib import Path
import threading

import httpx

from localflow.injector.window_context import ActiveWindowContext
from localflow.llm.llama_server import LlamaServerManager
from localflow.llm.model_catalog import (
    EMBEDDED_LLM_MODELS,
    find_llama_server_binary,
    get_model_path,
)
from localflow.llm.ollama_polisher import (
    PolishResult,
    _build_transform_system_prompt,
    _normalize_vocabulary,
    build_polish_system_prompt,
    format_polish_user_message,
    strip_long_dashes,
    strip_transcript_wrapper,
    validate_polish_output,
)
from localflow.styles import DEFAULT_STYLE_INSTRUCTIONS, StyleProfile, apply_style_postprocessing
from localflow.vocabulary import ParsedVocabulary, apply_vocabulary_replacements


class EmbeddedLlamaPolisher:
    """Drop-in polisher matching OllamaPolisher's API using local llama-server + GGUF."""

    def __init__(
        self,
        *,
        model_id: str = "qwen25-1.5b",
        custom_model_path: str = "",
        system_prompt: str = "",
        timeout_sec: float = 45.0,
        keep_alive: str = "10m",
        num_ctx: int = 4096,
        num_gpu: int | None = None,
        vocabulary: ParsedVocabulary | str | None = None,
        lang: str | None = None,
        base_dir: Path | None = None,
        server_manager: LlamaServerManager | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.model_id = model_id
        self.custom_model_path = custom_model_path
        self.system_prompt = system_prompt
        self.timeout_sec = timeout_sec
        self.keep_alive = keep_alive
        self.num_ctx = max(2048, int(num_ctx))
        self.num_gpu = num_gpu
        self.vocabulary = _normalize_vocabulary(vocabulary)
        self.lang = lang
        self._base_dir = base_dir
        self._lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._warming = False
        self._manager = server_manager
        self._owns_manager = server_manager is None
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else httpx.Client(timeout=self.timeout_sec, trust_env=False)
        )

    def _get_or_create_manager(self) -> LlamaServerManager:
        with self._lock:
            if self._manager is not None:
                return self._manager
            binary = find_llama_server_binary(self._base_dir)
            if binary is None:
                raise FileNotFoundError(
                    "Embedded LLM engine is not installed — open Settings"
                )
            model_path = get_model_path(
                self.model_id,
                custom_path=self.custom_model_path,
                base_dir=self._base_dir,
            )
            if model_path is None or not model_path.is_file():
                raise FileNotFoundError(
                    "Embedded model is not downloaded — download a model in Settings"
                )
            self._manager = LlamaServerManager(
                binary_path=binary,
                model_path=model_path,
                num_ctx=self.num_ctx,
                num_gpu=self.num_gpu,
                keep_alive=self.keep_alive,
            )
            return self._manager

    def list_models(self) -> list[str]:
        return [m.id for m in EMBEDDED_LLM_MODELS]

    def warmup(self) -> None:
        with self._warmup_lock:
            if self._warming:
                return
            self._warming = True
        try:
            mgr = self._get_or_create_manager()
            mgr.ensure_started()
            mgr.touch_activity()
        except Exception:
            pass
        finally:
            with self._warmup_lock:
                self._warming = False

    def unload(self) -> None:
        with self._lock:
            mgr = self._manager
        if mgr is not None:
            mgr.stop()

    def close(self) -> None:
        with self._lock:
            mgr = self._manager
            self._manager = None
        if mgr is not None and self._owns_manager:
            mgr.close()
        if self._owns_client:
            try:
                self._client.close()
            except Exception:
                pass

    def _chat_completion_with_meta(
        self,
        *,
        system_content: str,
        user_content: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, str | None]:
        mgr = self._get_or_create_manager()
        base_url = mgr.ensure_started()
        try:
            payload = {
                "model": self.model_id,
                "stream": False,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
            }
            resp = self._client.post(f"{base_url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if not isinstance(choices, list) or not choices:
                return ("", None)
            first = choices[0] if isinstance(choices[0], dict) else {}
            msg = first.get("message")
            content = msg.get("content") if isinstance(msg, dict) else None
            finish_reason = first.get("finish_reason")
            return (
                content if isinstance(content, str) else "",
                finish_reason if isinstance(finish_reason, str) else None,
            )
        finally:
            mgr.touch_activity()

    def _chat_completion(
        self,
        *,
        system_content: str,
        user_content: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        content, _finish_reason = self._chat_completion_with_meta(
            system_content=system_content,
            user_content=user_content,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return content

    def polish(
        self,
        raw_text: str,
        *,
        window_context: ActiveWindowContext | str | None = None,
        style_profile: StyleProfile | None = None,
    ) -> PolishResult:
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
        if effective_style is not None and effective_style.key == "ai_prompt" and not is_short_prompt_guard:
            max_tokens = max(256, min(2048, len(prepared_text) * 3))
        else:
            max_tokens = max(256, min(2048, len(prepared_text) * 2))

        try:
            raw_content, finish_reason = self._chat_completion_with_meta(
                system_content=system_content,
                user_content=format_polish_user_message(prepared_text),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = strip_transcript_wrapper(raw_content)
            if not content:
                return PolishResult(text=prepared_text, used_llm=False, error="Empty LLM response")
            leak_err = validate_polish_output(content, prepared_text, finish_reason=finish_reason)
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
        if not selected_text.strip():
            return PolishResult(text="", used_llm=False, error="Empty selected text")

        clean_command = command.strip()
        if not clean_command:
            return PolishResult(text="", used_llm=False, error="Empty command")

        max_tokens = max(256, min(2048, len(selected_text) * 3))
        system_content = _build_transform_system_prompt(
            window_context=window_context,
            vocabulary=self.vocabulary,
            lang=self.lang,
        )
        user_content = (
            f"<selected_text>\n{selected_text}\n</selected_text>\n\n"
            f"<command>\n{clean_command}\n</command>"
        )

        try:
            raw_content = self._chat_completion(
                system_content=system_content,
                user_content=user_content,
                temperature=0.05,
                max_tokens=max_tokens,
            )
            if not raw_content.strip():
                return PolishResult(text="", used_llm=False, error="Empty LLM response")
            if "\n" in raw_content or selected_text[:1] in {" ", "\t"}:
                content = raw_content.strip("\r\n")
            else:
                content = raw_content.strip()
            content = strip_long_dashes(content)
            return PolishResult(text=content, used_llm=True)
        except Exception as exc:
            return PolishResult(text="", used_llm=False, error=str(exc))
