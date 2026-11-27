"""Local FTS5 + LLM Question Answering ('Ask AI') across single or all meetings."""

from __future__ import annotations

from dataclasses import dataclass
import re

from localflow.i18n import normalize_language
from localflow.llm.ollama_polisher import strip_long_dashes
from localflow.notetaker.summarizer import TextGenerator
from localflow.notetaker.storage import MeetingRecord, MeetingStorage, format_timestamp

_CATCHUP_HEADER_RE = re.compile(
    r"^(?:обсудили|обсуждалось(?:\s+следующее)?|что\s+обсуждалось|discussed|topics\s+discussed|summary)\s*:?$",
    re.IGNORECASE,
)
_CATCHUP_FILLER_LEAD_RE = re.compile(
    r"^(?:обсуждение(?:\s+(?:темы|вопроса|того))?|обсудили|обсуждалось(?:\s+что)?|"
    r"разговор(?:\s+(?:о|об|про))?|упоминание(?:\s+(?:о|об))?|"
    r"discussion\s+(?:of|about)|talking\s+about|mention(?:ing)?\s+of|discussed)\s*[:,-]?\s+",
    re.IGNORECASE,
)


def sanitize_plain_bullets(text: str, *, lang: str | None = None) -> str:
    """Convert catch-up output into a single header + short plain-text '-' list without asterisks or boilerplate."""
    cleaned = strip_long_dashes(text or "")
    cleaned = cleaned.replace("**", "").replace("__", "").replace("*", "").replace("`", "")
    # Split inline "Обсудили: - Тема..." onto header line + bullet line
    cleaned = re.sub(
        r"(?im)^(обсудили|обсуждалось(?:\s+следующее)?|что\s+обсуждалось|discussed|topics\s+discussed)\s*:\s*(?=-|\S)",
        r"\1:\n",
        cleaned,
    )
    bullets: list[str] = []
    for raw in cleaned.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^(?:[-•·]+|\d+[.)])\s+", "", line).strip()
        if not line or _CATCHUP_HEADER_RE.match(line):
            continue
        # Strip repeated lead-ins like "Обсуждение ..." or "Discussion of ..."
        prev = None
        while prev != line:
            prev = line
            line = _CATCHUP_FILLER_LEAD_RE.sub("", line).strip()
        if not line:
            continue
        if line[0].isalpha() and line[0].islower():
            line = line[0].upper() + line[1:]
        bullets.append(f"- {line}")
    if not bullets:
        return ""
    is_ru = (
        normalize_language(lang) == "ru"
        if lang
        else bool(re.search(r"[А-Яа-яЁё]", "\n".join(bullets)))
    )
    header = "Обсудили:" if is_ru else "Discussed:"
    return "\n".join([header, *bullets])


@dataclass(frozen=True, slots=True)
class QACitation:
    meeting_id: int
    title: str
    date_label: str


@dataclass(frozen=True, slots=True)
class QAResult:
    answer: str
    citations: list[QACitation]


class MeetingQAEngine:
    """Answers questions grounded in one meeting or across all stored meetings via SQLite FTS5."""

    def __init__(self, storage: MeetingStorage, llm: TextGenerator) -> None:
        self.storage = storage
        self.llm = llm

    def answer_question(
        self,
        question: str,
        *,
        meeting_id: int | None = None,
        max_context_chars: int = 8000,
        lang: str | None = None,
        plain_bullets: bool = False,
    ) -> QAResult:
        clean_q = question.strip()
        if not clean_q:
            return QAResult(answer="", citations=[])

        citations: list[QACitation] = []
        context_blocks: list[str] = []

        if meeting_id is not None:
            meeting = self.storage.get_meeting(meeting_id)
            if meeting is None:
                return QAResult(answer="", citations=[])
            citations.append(
                QACitation(
                    meeting_id=meeting.id,
                    title=meeting.title,
                    date_label=meeting.created_at[:10],
                )
            )
            if plain_bullets:
                return self._answer_catchup_plain_bullets(
                    meeting,
                    clean_q,
                    citations,
                    lang=lang,
                )
            context_blocks.append(
                self._format_single_meeting_context(
                    meeting,
                    max_chars=max_context_chars,
                    prefer_recent=False,
                )
            )
        else:
            hits = self.storage.search_fts(clean_q, limit=5)
            meetings: list[MeetingRecord] = []
            for hit in hits:
                m = self.storage.get_meeting(hit.meeting_id)
                if m is not None:
                    meetings.append(m)
            if not meetings:
                meetings = self.storage.list_meetings(limit=3)

            per_meeting_budget = max(1200, max_context_chars // max(1, len(meetings)))
            for m in meetings:
                citations.append(
                    QACitation(
                        meeting_id=m.id,
                        title=m.title,
                        date_label=m.created_at[:10],
                    )
                )
                context_blocks.append(self._format_single_meeting_context(m, max_chars=per_meeting_budget))

        if not context_blocks:
            return QAResult(answer="", citations=[])

        lang_hint = ""
        if lang:
            from localflow.notetaker.summarizer import target_language_name

            lang_hint = (
                f"Respond in {target_language_name(lang)} unless the user explicitly asked in another language. "
            )
        if plain_bullets:
            system_prompt, _map_sys, _reduce_sys = self._build_catchup_prompts(lang)
            user_prompt = (
                f"<meetings_context>\n{chr(10).join(context_blocks)}\n</meetings_context>\n\n"
                f"<question>\n{clean_q}\n</question>"
            )
            answer = self.llm.generate_text(system_prompt, user_prompt, temperature=0.1, max_tokens=550)
            return QAResult(answer=sanitize_plain_bullets(answer, lang=lang), citations=citations)

        system_prompt = (
            "You are a helpful local meeting assistant. Answer the user's question strictly based on the provided "
            "meeting notes and transcripts. Answer in the same language as the user's question. "
            f"{lang_hint}"
            "When referencing information across multiple meetings, cite [Meeting Title, Date]. "
            "If the answer is not in the notes, say so honestly. "
            "Never use long dashes (em-dash or en-dash); use commas, colons, or a short hyphen '-' instead."
        )
        user_prompt = (
            f"<meetings_context>\n{chr(10).join(context_blocks)}\n</meetings_context>\n\n"
            f"<question>\n{clean_q}\n</question>"
        )
        answer = self.llm.generate_text(system_prompt, user_prompt, temperature=0.15, max_tokens=1000)
        return QAResult(answer=strip_long_dashes(answer.strip()), citations=citations)

    @staticmethod
    def _build_catchup_prompts(lang: str | None) -> tuple[str, str, str]:
        is_ru = normalize_language(lang) == "ru"
        if is_ru:
            single_sys = (
                "Подведи краткие промежуточные итоги ВСЕЙ беседы целиком по каждой отдельной теме.\n"
                "ПРАВИЛА:\n"
                "1. Выдели ВСЕ отдельные основные темы беседы (от 2 в коротком чате до 6-9 в длинном - не ограничивайся 3-4 пунктами, если тем больше).\n"
                "2. По каждой теме запиши 1 короткий пункт с реальным итогом, выводом, цифрами, именами или исходом (выиграли/проиграли, подошло/не подошло, о чём договорились).\n"
                "3. Игнорируй явные ослышки ASR и случайные фразы на чужих языках.\n"
                "4. ФОРМАТ (строго plain text):\n"
                "   Первая строка: ровно \"Обсудили:\"\n"
                "   Далее: - <Тема>: <короткий точный итог, 8-18 слов>\n"
                "5. ЗАПРЕЩЕНО:\n"
                "   - начинать пункты с «Обсуждение», «Разговор», «Упоминание», «Обсудили», «Обсуждалось», «Участники»;\n"
                "   - ЛЮБОЕ форматирование: **, *, #, `, длинные тире;\n"
                "   - вступления и заключения.\n\n"
                "Пример ответа:\n"
                "Обсудили:\n"
                "- Сбой оплаты: нашли причину в таймауте шлюза и откатили релиз 2.4\n"
                "- График патча: Алексей проверит фикс к 18:00 и выкатит обновление"
            )
            map_sys = (
                "Извлеки ВСЕ отдельные смысловые темы из этого фрагмента разговора и подведи точный итог по каждой теме.\n"
                "Правила:\n"
                "- Пиши строго на русском языке, коротко и конкретно, активными глаголами (кто что сказал, решил, оспорил, к чему пришли).\n"
                "- Сохраняй реальные исходы, имена, цифры, время и ключевые термины. Игнорируй ослышки ASR.\n"
                "- ЗАПРЕЩЕНЫ штампы («Обсуждалось...», «Было отмечено...», «Обсуждение...») и длинные тире.\n"
                "- Формат: список строк `- <Тема>: <конкретные факты и итог>`."
            )
            reduce_sys = (
                "Собери единый краткий промежуточный итог беседы по всем темам от начала до текущего момента.\n"
                "ПРАВИЛА:\n"
                "1. Включи ВСЕ основные темы из контекста (2-3 для короткой беседы, 5-9 для насыщенной) - НЕ обрезай до 3-4 пунктов, если тем больше.\n"
                "2. Объединяй дубли одной и той же темы в один точный пункт с финальным итогом/выводом (сохраняй конфликты, решения, цифры, имена и исходы).\n"
                "3. ФОРМАТ (строго plain text):\n"
                "   Первая строка: ровно \"Обсудили:\"\n"
                "   Далее по строке на тему: - <Тема>: <короткий точный итог, 8-18 слов>\n"
                "4. ЗАПРЕЩЕНО:\n"
                "   - начинать пункты с «Обсуждение», «Разговор», «Упоминание», «Обсудили», «Обсуждалось», «Участники»;\n"
                "   - ЛЮБОЕ форматирование: **, *, #, `, длинные тире;\n"
                "   - вводные и завершающие фразы."
            )
            return single_sys, map_sys, reduce_sys

        single_sys = (
            "Summarize what has been discussed across the ENTIRE conversation so far, organized by distinct topic.\n"
            "RULES:\n"
            "1. Identify EVERY distinct main topic covered from start to current moment (2-3 for a short chat, 5-9 for a dense multi-topic call - do NOT cap at 3-4).\n"
            "2. For each topic, write 1 short line stating the concrete outcome, conclusion, numbers, names, or decision.\n"
            "3. Ignore obvious ASR glitches.\n"
            "4. STRICT PLAIN-TEXT FORMAT:\n"
            "   First line: exactly \"Discussed:\"\n"
            "   Below: - <Topic>: <concise outcome, 8-18 words>\n"
            "5. FORBIDDEN:\n"
            "   - starting bullets with 'Discussion of', 'Talking about', 'Mention of', 'Discussed', 'Participants';\n"
            "   - ANY markdown formatting: **, *, #, `, or long dashes;\n"
            "   - any intro or outro prose."
        )
        map_sys = (
            "Extract EVERY distinct topic from this conversation segment with its concrete facts, names, outcomes, and decisions. "
            "Ignore ASR glitches. Write concise '- <Topic>: <concrete outcome>' lines in English with NO markdown and NO long dashes."
        )
        reduce_sys = (
            "Consolidate the segment notes into a single concise catch-up list covering ALL major topics discussed so far.\n"
            "RULES:\n"
            "1. Include ALL distinct main topics (2-3 short, 5-9 dense - do NOT drop earlier/later themes or cap at 3-4).\n"
            "2. Merge duplicate notes for the same topic into one line with its latest outcome.\n"
            "3. STRICT PLAIN-TEXT FORMAT:\n"
            "   First line: exactly \"Discussed:\"\n"
            "   Below: - <Topic>: <concise outcome, 8-18 words>\n"
            "4. FORBIDDEN: 'Discussion of', 'Talking about', 'Mention of', bold (**), asterisks (*), #, `, long dashes, intro/outro prose."
        )
        return single_sys, map_sys, reduce_sys

    def _answer_catchup_plain_bullets(
        self,
        meeting: MeetingRecord,
        clean_q: str,
        citations: list[QACitation],
        *,
        lang: str | None,
        chunk_char_limit: int = 4500,
    ) -> QAResult:
        speakers = {s.speaker_key: s.display_name for s in self.storage.get_speakers(meeting.id)}
        utterances = self.storage.get_utterances(meeting.id)
        lines = [
            f"[{format_timestamp(u.start_sec)}] {speakers.get(u.speaker_key, u.speaker_key)}: {u.text}"
            for u in utterances
            if u.text.strip()
        ]
        my_thoughts = meeting.my_thoughts.strip()
        if not lines and not my_thoughts:
            return QAResult(answer="", citations=citations)

        single_sys, map_sys, reduce_sys = self._build_catchup_prompts(lang)
        thoughts_block = f"\n\n<my_thoughts>\n{my_thoughts}\n</my_thoughts>" if my_thoughts else ""
        full_transcript = "\n".join(lines)

        if len(full_transcript) <= chunk_char_limit:
            user_prompt = (
                f"<meetings_context>\n=== {meeting.title} ===\nTranscript:\n{full_transcript}{thoughts_block}\n</meetings_context>\n\n"
                f"<question>\n{clean_q}\n</question>"
            )
            answer = self.llm.generate_text(single_sys, user_prompt, temperature=0.1, max_tokens=550)
            return QAResult(answer=sanitize_plain_bullets(answer, lang=lang), citations=citations)

        # Chronological chunked extraction across the entire conversation so far
        chunk_target = max(2000, chunk_char_limit - 300)
        chunks: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for line in lines:
            if buf and buf_len + len(line) + 1 > chunk_target:
                chunks.append("\n".join(buf))
                buf = [line]
                buf_len = len(line)
            else:
                buf.append(line)
                buf_len += len(line) + 1
        if buf:
            chunks.append("\n".join(buf))
        if len(chunks) > 10:
            # Keep balanced coverage if call is exceptionally long
            step = len(chunks) / 10.0
            chunks = [chunks[min(len(chunks) - 1, int(i * step))] for i in range(10)]

        partials: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            part = self.llm.generate_text(
                map_sys,
                f"<segment index='{idx}/{len(chunks)}'>\n{chunk}\n</segment>",
                temperature=0.1,
                max_tokens=500,
            )
            if part.strip():
                partials.append(f"Part {idx}:\n{part.strip()}")

        combined = "\n\n".join(partials)
        reduce_user = (
            f"<partial_notes>\n{combined}\n</partial_notes>{thoughts_block}\n\n"
            f"<question>\n{clean_q}\n</question>"
        )
        answer = self.llm.generate_text(reduce_sys, reduce_user, temperature=0.1, max_tokens=600)
        return QAResult(answer=sanitize_plain_bullets(answer, lang=lang), citations=citations)

    def _format_single_meeting_context(
        self,
        meeting: MeetingRecord,
        *,
        max_chars: int,
        prefer_recent: bool = False,
    ) -> str:
        speakers = {s.speaker_key: s.display_name for s in self.storage.get_speakers(meeting.id)}
        utterances = self.storage.get_utterances(meeting.id)
        actions = self.storage.get_action_items(meeting.id)

        parts: list[str] = [f"=== {meeting.title} ({meeting.created_at[:10]}) ==="]
        if meeting.summary_md.strip():
            parts.append(f"Summary:\n{meeting.summary_md.strip()}")
        if actions:
            act_lines = [
                f"- [{'x' if a.done else ' '}] {a.text}" + (f" ({a.owner})" if a.owner else "")
                for a in actions
            ]
            parts.append("Action Items:\n" + "\n".join(act_lines))
        if meeting.my_thoughts.strip():
            parts.append(f"My Thoughts:\n{meeting.my_thoughts.strip()}")

        header_text = "\n\n".join(parts)
        remaining = max(0, max_chars - len(header_text) - 40)
        if utterances and remaining > 200:
            t_lines: list[str] = []
            used = 0
            iter_utts = reversed(utterances) if prefer_recent else utterances
            for u in iter_utts:
                spk = speakers.get(u.speaker_key, u.speaker_key)
                line = f"[{format_timestamp(u.start_sec)}] {spk}: {u.text}"
                if used + len(line) + 1 > remaining:
                    break
                t_lines.append(line)
                used += len(line) + 1
            if prefer_recent:
                t_lines.reverse()
            if t_lines:
                parts.append("Transcript:\n" + "\n".join(t_lines))

        return "\n\n".join(parts)[:max_chars]
