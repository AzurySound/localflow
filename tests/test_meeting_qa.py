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
