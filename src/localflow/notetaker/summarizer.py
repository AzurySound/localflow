"""Local LLM Map-Reduce meeting summarizer with Action Items extraction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Protocol

from localflow.i18n import normalize_language
from localflow.llm.ollama_polisher import strip_long_dashes
from localflow.notetaker.storage import MeetingStorage, format_timestamp


class TextGenerator(Protocol):
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class SummaryResult:
    title: str
    summary_md: str
    action_items: list[tuple[str, str]]
    used_map_reduce: bool = False
    too_short: bool = False


def target_language_name(lang: str | None) -> str:
    """Return explicit English language name matching app language setting."""
    return "Russian" if normalize_language(lang) == "ru" else "English"


def is_meeting_too_short(storage: MeetingStorage, meeting_id: int, *, min_words: int = 12) -> bool:
    """Return True when total words across transcript and user thoughts are below threshold."""
    meeting = storage.get_meeting(meeting_id)
    if meeting is None:
        return True
    words = len(meeting.my_thoughts.split())
    for u in storage.get_utterances(meeting_id):
        words += len(u.text.split())
        if words >= min_words:
            return False
    return words < min_words


_SECTION_TITLE_RE = re.compile(r"(?im)^(?:TITLE|ЗАГОЛОВОК)\s*:\s*(.+)$")
_ACTION_LINE_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*(?:\[[ xX]\]\s*)?(.+)$")
_OWNER_TAIL_RE = re.compile(r"^(.*?)\s*(?:\(([^()]{1,40})\)|\[([^\[\]]{1,40})\])\s*$")


def _build_summary_system_prompt(target_lang: str) -> str:
    if target_lang == "Russian":
        return (
            "Ты — элитный AI-ассистент встреч в стиле Wispr Flow. Составь живое, конкретное и отлично "
            "структурированное саммари.\n"
            "ЯЗЫК: пиши ВЕСЬ ответ (TITLE, SUMMARY и ACTION_ITEMS) строго на русском языке, "
            "независимо от языка реплик в транскрипте (технические/IT-термины сохраняй как в оригинале).\n"
            "СТИЛЬ И ТОЧНОСТЬ:\n"
            "1. ЗАПРЕЩЕНЫ канцеляризмы и пассивные штампы («Обсуждались аспекты...», «Были подняты вопросы...», "
            "«Обсуждено, что...», «Было отмечено...», «Определена договоренность...»). Пиши кратко, по делу, "
            "активными глаголами.\n"
            "2. СОХРАНЯЙ конкретику: все цифры, проценты, сроки, имена людей, названия систем/инструментов (Excel, ВОР, "
            "векторная база, слепые тесты и т.д.).\n"
            "3. Если в <my_thoughts> есть заметки пользователя — обязательно сделай на них акцент в саммари.\n"
            "4. НИКОГДА не используй длинные тире (em-dash или en-dash); используй запятую, двоеточие или короткий дефис '-'.\n"
            "СТРУКТУРА ОТВЕТА (соблюдай точные заголовки):\n"
            "TITLE: <короткий точный заголовок 3-6 слов>\n"
            "SUMMARY:\n"
            "<1-2 предложения главной сути встречи (TL;DR)>\n\n"
            "### Ключевые моменты\n"
            "- <тема/факт с деталями и цифрами>\n"
            "ACTION_ITEMS:\n"
            "- <конкретное следующее действие и срок, если назван> (<имя ответственного, например Вы или Собеседник 1>)\n"
            "(если реальных задач или договорённостей о действиях нет, напиши под ACTION_ITEMS только слово NONE)\n\n"
            "Пример хорошего вывода:\n"
            "TITLE: Пресейл-обсуждение сверки ВОР и точности\n"
            "SUMMARY:\n"
            "Разобрали требования к автоматической проверке спецификаций и договорились о критериях точности для старта контракта.\n\n"
            "### Ключевые моменты\n"
            "- Входные данные: система работает с векторной базой и принимает ВОР через Excel для приведения к нужной форме.\n"
            "- Контроль качества: целевой порог точности составляет 95% по каждому разделу на 2-3 комплектах, замер через ручную разметку и слепые тесты.\n"
            "ACTION_ITEMS:\n"
            "- Собрать обратную связь стейкхолдеров по материалам к понедельнику (Вы)"
        )
    return (
        "You are an expert Wispr Flow meeting notetaker. Produce a crisp, concrete, well-structured summary.\n"
        "LANGUAGE: Write the ENTIRE output (TITLE, SUMMARY, ACTION_ITEMS) strictly in English, "
        "regardless of source transcript languages (keep domain/IT terms intact).\n"
        "STYLE & PRECISION:\n"
        "1. FORBIDDEN passive filler phrases ('Aspects were discussed...', 'Questions were raised...', 'It was noted that...'). "
        "Write directly with active verbs.\n"
        "2. PRESERVE concrete details: numbers, percentages, deadlines, names, tools, and technical terms.\n"
        "3. Prioritize any highlights inside <my_thoughts>.\n"
        "4. NEVER use long dashes (em-dash or en-dash); use commas, colons, or a short hyphen '-'.\n"
        "FORMAT (exact headers):\n"
        "TITLE: <short 3-6 word meeting title>\n"
        "SUMMARY:\n"
        "<1-2 sentence TL;DR>\n\n"
        "### Key Points\n"
        "- <concrete takeaway with numbers/details>\n"
        "ACTION_ITEMS:\n"
        "- <concrete task + deadline if any> (<owner, e.g. You or Speaker 1>)\n"
        "(if there are no real tasks, output ONLY NONE under ACTION_ITEMS)"
    )


_STOP_ACTION_PHRASES: frozenset[str] = frozenset(
    {
        "none",
        "нет",
        "n/a",
        "-",
        "нет задач",
        "задач нет",
        "задачи не выявлены",
        "нет конкретных задач",
        "нет конкретных задач, требующих немедленного выполнения",
        "no action items",
        "no specific action items",
        "nothing",
    }
)
class MeetingSummarizer:
    """Summarizes short and long meeting transcripts via single-pass or Map-Reduce."""

    def __init__(self, llm: TextGenerator, *, single_pass_char_limit: int = 7500) -> None:
        self.llm = llm
        self.single_pass_char_limit = max(2000, single_pass_char_limit)

    def summarize_meeting(
        self,
        storage: MeetingStorage,
        meeting_id: int,
        *,
        lang: str = "en",
        force: bool = False,
        on_step: Callable[[str, int, int], None] | None = None,
    ) -> SummaryResult:
        meeting = storage.get_meeting(meeting_id)
        if meeting is None:
            raise ValueError(f"Meeting {meeting_id} not found")

        speakers = {s.speaker_key: s.display_name for s in storage.get_speakers(meeting_id)}
        utterances = storage.get_utterances(meeting_id)
        lines = [
            f"[{format_timestamp(u.start_sec)}] {speakers.get(u.speaker_key, u.speaker_key)}: {u.text}"
            for u in utterances
            if u.text.strip()
        ]
        my_thoughts = meeting.my_thoughts.strip()
        if not lines and not my_thoughts:
            fallback_title = meeting.title or "Empty Meeting"
            storage.update_meeting(meeting_id, status="ready")
            return SummaryResult(
                title=fallback_title,
                summary_md="",
                action_items=[],
                used_map_reduce=False,
                too_short=False,
            )

        if not force and is_meeting_too_short(storage, meeting_id):
            storage.update_meeting(meeting_id, summary_md="", status="ready")
            return SummaryResult(
                title=meeting.title,
                summary_md="",
                action_items=[],
                used_map_reduce=False,
                too_short=True,
            )

        transcript = "\n".join(lines)

        if len(transcript) <= self.single_pass_char_limit:
            raw = self._summarize_single_pass(transcript, my_thoughts, lang=lang, on_step=on_step)
            used_mr = False
        else:
            raw = self._summarize_map_reduce(lines, my_thoughts, lang=lang, on_step=on_step)
            used_mr = True

        parsed = self._parse_summary_output(raw, fallback_title=meeting.title)
        storage.update_meeting(
            meeting_id,
            title=parsed.title,
            summary_md=parsed.summary_md,
            status="ready",
        )
        storage.replace_action_items(meeting_id, parsed.action_items)
        return SummaryResult(
            title=parsed.title,
            summary_md=parsed.summary_md,
            action_items=parsed.action_items,
            used_map_reduce=used_mr,
            too_short=False,
        )

    def _summarize_single_pass(
        self,
        transcript: str,
        my_thoughts: str,
        *,
        lang: str,
        on_step: Callable[[str, int, int], None] | None = None,
    ) -> str:
        if on_step is not None:
            on_step("analyze", 1, 2)
        target_lang = target_language_name(lang)
        thoughts_block = f"\n\n<my_thoughts>\n{my_thoughts}\n</my_thoughts>" if my_thoughts else ""
        system_prompt = _build_summary_system_prompt(target_lang)
        user_prompt = f"<transcript>\n{transcript}\n</transcript>{thoughts_block}"
        out = self.llm.generate_text(system_prompt, user_prompt, temperature=0.15, max_tokens=1400)
        if on_step is not None:
            on_step("write", 2, 2)
        return out

    def _summarize_map_reduce(
        self,
        lines: list[str],
        my_thoughts: str,
        *,
        lang: str,
        on_step: Callable[[str, int, int], None] | None = None,
    ) -> str:
        target_lang = target_language_name(lang)
        chunks: list[str] = []
        buf: list[str] = []
        buf_len = 0
        chunk_limit = max(2000, self.single_pass_char_limit - 1200)
        for line in lines:
            if buf and buf_len + len(line) + 1 > chunk_limit:
                chunks.append("\n".join(buf))
                buf = [line]
                buf_len = len(line)
            else:
                buf.append(line)
                buf_len += len(line) + 1
        if buf:
            chunks.append("\n".join(buf))

        total_steps = len(chunks) + 1
        map_system = (
            "Extract concrete facts, numbers, tools, decisions, and tasks with speaker names from this meeting segment. "
            f"CRITICAL LANGUAGE REQUIREMENT: Write strictly in {target_lang} regardless of transcript language. "
            "Avoid passive bureaucratic phrasing. Never use long dashes; use '-' instead."
        )
        partials: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            if on_step is not None:
                on_step("analyze", idx, total_steps)
            part = self.llm.generate_text(
                map_system,
                f"<segment index='{idx}/{len(chunks)}'>\n{chunk}\n</segment>",
                temperature=0.15,
                max_tokens=700,
            )
            if part.strip():
                partials.append(f"### Part {idx}\n{part.strip()}")

        combined = "\n\n".join(partials)
        thoughts_block = f"\n\n<my_thoughts>\n{my_thoughts}\n</my_thoughts>" if my_thoughts else ""
        if on_step is not None:
            on_step("write", total_steps, total_steps)
        reduce_system = _build_summary_system_prompt(target_lang)
        return self.llm.generate_text(
            reduce_system,
            f"<partial_notes>\n{combined}\n</partial_notes>{thoughts_block}",
            temperature=0.15,
            max_tokens=1500,
        )

    @staticmethod
    def _parse_summary_output(raw: str, *, fallback_title: str) -> SummaryResult:
        clean = strip_long_dashes(raw.strip())
        if not clean:
            return SummaryResult(title=fallback_title, summary_md="", action_items=[])

        title = fallback_title
        title_match = _SECTION_TITLE_RE.search(clean)
        if title_match:
            candidate = title_match.group(1).strip().strip('"').strip("'")
            if candidate:
                title = candidate[:90]

        # Split SUMMARY and ACTION_ITEMS sections
        summary_text = clean
        actions_text = ""

        summary_split = re.split(r"(?im)^(?:SUMMARY|РЕЗЮМЕ)\s*:\s*$", clean, maxsplit=1)
        if len(summary_split) == 2:
            after_summary = summary_split[1]
        else:
            after_summary = _SECTION_TITLE_RE.sub("", clean, count=1).strip()

        action_split = re.split(r"(?im)^(?:ACTION_ITEMS|ACTION ITEMS|ЗАДАЧИ)\s*:\s*$", after_summary, maxsplit=1)
        if len(action_split) == 2:
            summary_text = action_split[0].strip()
            actions_text = action_split[1].strip()
        else:
            summary_text = after_summary.strip()

        action_items: list[tuple[str, str]] = []
        if actions_text:
            for line in actions_text.splitlines():
                m = _ACTION_LINE_RE.match(line)
                if not m:
                    continue
                item_body = m.group(1).strip()
                norm_body = item_body.rstrip(".!").strip().lower()
                if not item_body or norm_body in _STOP_ACTION_PHRASES or "нет конкретных задач" in norm_body:
                    continue
                owner = ""
                om = _OWNER_TAIL_RE.match(item_body)
                if om:
                    item_body = om.group(1).strip()
                    owner = (om.group(2) or om.group(3) or "").strip()
                if item_body:
                    action_items.append((strip_long_dashes(item_body), strip_long_dashes(owner)))

        return SummaryResult(
            title=strip_long_dashes(title),
            summary_md=strip_long_dashes(summary_text),
            action_items=action_items,
        )
