"""Local FTS5 + LLM Question Answering ('Ask AI') across single or all meetings."""

from __future__ import annotations

from dataclasses import dataclass

from localflow.llm.ollama_polisher import strip_long_dashes
from localflow.notetaker.summarizer import TextGenerator
from localflow.notetaker.storage import MeetingRecord, MeetingStorage, format_timestamp


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
            context_blocks.append(self._format_single_meeting_context(meeting, max_chars=max_context_chars))
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

    def _format_single_meeting_context(self, meeting: MeetingRecord, *, max_chars: int) -> str:
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
            for u in utterances:
                spk = speakers.get(u.speaker_key, u.speaker_key)
                line = f"[{format_timestamp(u.start_sec)}] {spk}: {u.text}"
                if used + len(line) + 1 > remaining:
                    break
                t_lines.append(line)
                used += len(line) + 1
            if t_lines:
                parts.append("Transcript:\n" + "\n".join(t_lines))

        return "\n\n".join(parts)[:max_chars]
