from __future__ import annotations

from pathlib import Path

from localflow.notetaker.storage import MeetingStorage
from localflow.notetaker.summarizer import MeetingSummarizer


class _FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        self.calls.append((system_prompt, user_prompt))
        if "<segment " in user_prompt:
            return "- Discussed API migration and tests."
        return (
            "TITLE: API Migration Sync — Draft\n"
            "SUMMARY:\n"
            "- Agreed to migrate endpoints — without downtime.\n"
            "- Incorporated user thoughts.\n"
            "ACTION_ITEMS:\n"
            "- Update client SDK (Gena)\n"
            "- Run load tests (You)\n"
        )


def test_summarizer_single_pass_and_no_long_dashes(tmp_path: Path) -> None:
    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    m = storage.create_meeting(title="New Meeting")
    storage.ensure_speaker(m.id, "you", display_name="You", is_self=True)
    storage.ensure_speaker(m.id, "spk_1", display_name="Gena", is_self=False)
    storage.add_utterance(m.id, speaker_key="you", channel="mic", start_sec=0.0, end_sec=2.0, text="Let's plan the API migration.")
    storage.add_utterance(m.id, speaker_key="spk_1", channel="system", start_sec=3.0, end_sec=6.0, text="I will update the client SDK.")
    storage.update_meeting(m.id, my_thoughts="Remember zero downtime")

    llm = _FakeLLM()
    summarizer = MeetingSummarizer(llm, single_pass_char_limit=5000)
    res = summarizer.summarize_meeting(storage, m.id)

    assert res.used_map_reduce is False
    assert "—" not in res.title
    assert "—" not in res.summary_md
    assert len(res.action_items) == 2
    assert res.action_items[0] == ("Update client SDK", "Gena")
    assert "Remember zero downtime" in llm.calls[0][1]

    reloaded = storage.get_meeting(m.id)
    assert reloaded is not None
    assert reloaded.title == res.title
    assert len(storage.get_action_items(m.id)) == 2


def test_summarizer_map_reduce_for_long_transcript(tmp_path: Path) -> None:
    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    m = storage.create_meeting(title="Long Call")
    storage.ensure_speaker(m.id, "you", display_name="You", is_self=True)
    for i in range(60):
        storage.add_utterance(
            m.id,
            speaker_key="you",
            channel="mic",
            start_sec=float(i * 10),
            end_sec=float(i * 10 + 5),
            text=f"Detailed architectural discussion point number {i} about database sharding and caching.",
        )

    llm = _FakeLLM()
    summarizer = MeetingSummarizer(llm, single_pass_char_limit=2200)
    res = summarizer.summarize_meeting(storage, m.id)

    assert res.used_map_reduce is True
    assert len(llm.calls) >= 3  # Multiple map calls + 1 reduce call


def test_summarizer_language_lock_and_too_short_guard(tmp_path: Path) -> None:
    storage = MeetingStorage(db_path=tmp_path / "meetings.db")
    m = storage.create_meeting(title="New note")
    storage.ensure_speaker(m.id, "you", display_name="You", is_self=True)
    storage.add_utterance(m.id, speaker_key="you", channel="mic", start_sec=0.0, end_sec=1.5, text="Quick test check.")

    llm = _FakeLLM()
    summarizer = MeetingSummarizer(llm, single_pass_char_limit=5000)

    # Below 12 words with force=False returns too_short=True without calling LLM
    res_short = summarizer.summarize_meeting(storage, m.id, lang="ru", force=False)
    assert res_short.too_short is True
    assert len(llm.calls) == 0

    # With force=True (or longer meeting) it forces Russian in system prompt and reports steps
    steps: list[tuple[str, int, int]] = []
    res_forced = summarizer.summarize_meeting(
        storage,
        m.id,
        lang="ru",
        force=True,
        on_step=lambda stage, step, total: steps.append((stage, step, total)),
    )
    assert res_forced.too_short is False
    assert len(llm.calls) == 1
    assert "строго на русском языке" in llm.calls[0][0]
    assert "ЗАПРЕЩЕНЫ канцеляризмы" in llm.calls[0][0]
    assert steps == [("analyze", 1, 2), ("write", 2, 2)]

    # Verify lang="en" locks to English
    res_en = summarizer.summarize_meeting(storage, m.id, lang="en", force=True)
    assert res_en.too_short is False
    assert "strictly in English" in llm.calls[1][0]

    # Stop-filter drops bogus placeholder action items like "Нет конкретных задач..."
    parsed = MeetingSummarizer._parse_summary_output(
        "TITLE: Тест\nSUMMARY:\nКратко\nACTION_ITEMS:\n- Нет конкретных задач, требующих немедленного выполнения.\n- Проверить логи (Вы)",
        fallback_title="Тест",
    )
    assert parsed.action_items == [("Проверить логи", "Вы")]

    # Summary markdown formatter guarantees a blank line between opening paragraph and heading
    formatted = MeetingSummarizer._parse_summary_output(
        "TITLE: Итоги\nSUMMARY: Короткое введение.\n### Ключевые моменты\n- Деталь 1",
        fallback_title="Итоги",
    )
    assert formatted.summary_md == "Короткое введение.\n\n### Ключевые моменты\n- Деталь 1"
    assert MeetingSummarizer(llm).single_pass_char_limit <= 5000

