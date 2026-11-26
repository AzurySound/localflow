from __future__ import annotations

from pathlib import Path

from localflow.notetaker.storage import MeetingStorage


def test_meeting_crud_speakers_utterances_and_fts(tmp_path: Path) -> None:
    db_path = tmp_path / "meetings.db"
    storage = MeetingStorage(db_path=db_path)

    m = storage.create_meeting(title="Sprint Planning", audio_mode="dual")
    assert m.id > 0
    assert m.title == "Sprint Planning"

    storage.ensure_speaker(m.id, "you", display_name="You", color_idx=0, is_self=True)
    storage.ensure_speaker(m.id, "spk_1", display_name="Speaker 1", color_idx=1, is_self=False)
    storage.ensure_speaker(m.id, "spk_2", display_name="Speaker 2", color_idx=2, is_self=False)

    storage.add_utterance(m.id, speaker_key="you", channel="mic", start_sec=1.0, end_sec=3.0, text="Let's review Kubernetes deployment.")
    storage.add_utterance(m.id, speaker_key="spk_1", channel="system", start_sec=4.0, end_sec=8.0, text="Gena will update the Helm chart.")
    storage.add_utterance(m.id, speaker_key="spk_2", channel="system", start_sec=9.0, end_sec=12.0, text="Sounds good, I will test staging.")

    # Rename & merge speakers
    storage.rename_speaker(m.id, "spk_1", "Gena")
    storage.merge_speakers(m.id, "spk_2", "spk_1")
    speakers = {s.speaker_key: s.display_name for s in storage.get_speakers(m.id)}
    assert speakers["spk_1"] == "Gena"
    assert "spk_2" not in speakers

    utts = storage.get_utterances(m.id)
    assert len(utts) == 3
    assert all(u.speaker_key in {"you", "spk_1"} for u in utts)

    # Action items & checkbox toggle
    items = storage.replace_action_items(m.id, [("Update Helm chart", "Gena"), ("Verify staging", "You")])
    assert len(items) == 2
    assert items[0].done is False
    storage.set_action_item_done(items[0].id, True)
    reloaded_items = storage.get_action_items(m.id)
    assert reloaded_items[0].done is True
    assert reloaded_items[1].done is False

    # FTS5 search across transcript & thoughts
    storage.update_meeting(m.id, my_thoughts="Check rollback script before Friday", summary_md="- Reviewed Helm chart.")
    hits = storage.search_fts("Kubernetes")
    assert len(hits) == 1
    assert hits[0].meeting_id == m.id

    hits_thoughts = storage.search_fts("rollback")
    assert len(hits_thoughts) == 1

    # Export Markdown
    md = storage.export_markdown(m.id)
    assert "# Sprint Planning" in md
    assert "- [x] Update Helm chart (Gena)" in md
    assert "**[00:04] Gena:** Gena will update the Helm chart." in md

    # Soft delete hides from list & FTS
    storage.delete_meeting(m.id, hard=False)
    assert storage.get_meeting(m.id) is None
    assert storage.list_meetings() == []
    assert storage.search_fts("Kubernetes") == []
