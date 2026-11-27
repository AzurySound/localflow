from __future__ import annotations

from pathlib import Path

from localflow.notetaker.qa_engine import MeetingQAEngine
from localflow.notetaker.storage import MeetingStorage


class _FakeQALLM:
    def __init__(self) -> None:
        self.last_user_prompt = ""

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.15,
        max_tokens: int = 1000,
    ) -> str:
        self.last_user_prompt = user_prompt
        return "Gena agreed to deploy the hotfix on Friday — no blockers."


def test_qa_engine_single_and_all_meetings(tmp_path: Path) -> None:
    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    m1 = storage.create_meeting(title="Release Sync")
    storage.ensure_speaker(m1.id, "spk_1", display_name="Gena")
    storage.add_utterance(m1.id, speaker_key="spk_1", channel="system", start_sec=2.0, end_sec=5.0, text="I will deploy the hotfix on Friday.")

    m2 = storage.create_meeting(title="Design Review")
    storage.ensure_speaker(m2.id, "you", display_name="You", is_self=True)
    storage.add_utterance(m2.id, speaker_key="you", channel="mic", start_sec=1.0, end_sec=4.0, text="Let's polish the sidebar spacing.")

    llm = _FakeQALLM()
    qa = MeetingQAEngine(storage, llm)

    # Single-meeting Q&A
    res1 = qa.answer_question("When is the hotfix?", meeting_id=m1.id)
    assert "—" not in res1.answer
    assert len(res1.citations) == 1
    assert res1.citations[0].meeting_id == m1.id
    assert "Release Sync" in llm.last_user_prompt

    # Cross-meeting FTS5 Q&A
    res_all = qa.answer_question("hotfix", meeting_id=None)
    assert len(res_all.citations) >= 1
    assert any(c.meeting_id == m1.id for c in res_all.citations)


def test_qa_engine_plain_bullets_strips_markdown_and_asterisks(tmp_path: Path) -> None:
    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    m = storage.create_meeting(title="Standup")
    storage.ensure_speaker(m.id, "spk_1", display_name="Alex")
    for i in range(4):
        storage.add_utterance(
            m.id,
            speaker_key="spk_1",
            channel="system",
            start_sec=float(i * 10),
            end_sec=float(i * 10 + 4),
            text=f"Utterance {i} about topic {i} with detailed context and release metrics.",
        )

    class _MarkdownLLMFake:
        def __init__(self) -> None:
            self.last_system = ""
            self.last_user = ""
            self.users: list[str] = []
            self.last_max_tokens = 0

        def generate_text(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.15, max_tokens: int = 1000) -> str:
            self.last_system = system_prompt
            self.last_user = user_prompt
            self.users.append(user_prompt)
            self.last_max_tokens = max_tokens
            if "Обсудили:" in system_prompt:
                return (
                    "Обсудили: - Обсуждение различий между **квитанцией** и чеком\n"
                    "- Разговор о налоговом `извещении` — без штрафов\n"
                    "• Упоминание о слове *сегменты*"
                )
            if "<segment " in user_prompt:
                return "- Topic from segment: concrete fact"
            return "### Summary\n* **Release**: shipped `v0.3.1` — stable\n• Discussion of *next* steps: monitor logs"

    llm = _MarkdownLLMFake()
    qa = MeetingQAEngine(storage, llm)
    res_en = qa.answer_question("What did I miss?", meeting_id=m.id, lang="en", plain_bullets=True)
    assert llm.last_max_tokens >= 400
    assert "*" not in res_en.answer
    assert "`" not in res_en.answer
    assert "—" not in res_en.answer
    assert "#" not in res_en.answer
    assert res_en.answer == (
        "Discussed:\n"
        "- Release: shipped v0.3.1 - stable\n"
        "- Next steps: monitor logs"
    )

    # Russian prompt + sanitizer splits inline "Обсудили: - ..." and drops "Обсуждение ..." / "Разговор о ..."
    res_ru = qa.answer_question("Что я пропустил?", meeting_id=m.id, lang="ru", plain_bullets=True)
    assert res_ru.answer == (
        "Обсудили:\n"
        "- Различий между квитанцией и чеком\n"
        "- Налоговом извещении - без штрафов\n"
        "- Слове сегменты"
    )

    # Long meeting (> 4500 chars) triggers chunked Map-Reduce so BOTH earliest & latest topics are analyzed
    m_long = storage.create_meeting(title="Long Sync")
    storage.ensure_speaker(m_long.id, "spk_1", display_name="Alex")
    for idx in range(30):
        storage.add_utterance(
            m_long.id,
            speaker_key="spk_1",
            channel="system",
            start_sec=float(idx * 30),
            end_sec=float(idx * 30 + 10),
            text=f"MESSAGE_INDEX_{idx}_" + ("context word details " * 9),
        )
    llm.users.clear()
    res_long = qa.answer_question("Catch up", meeting_id=m_long.id, lang="en", plain_bullets=True)
    assert len(llm.users) >= 3  # 2+ map segments + 1 reduce synthesis
    all_inputs = "\n".join(llm.users)
    assert "MESSAGE_INDEX_0_" in all_inputs
    assert "MESSAGE_INDEX_29_" in all_inputs
    assert res_long.answer.startswith("Discussed:\n- ")
