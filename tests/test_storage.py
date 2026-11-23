import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from localflow.config import AppConfig
from localflow.storage import HistoryStorage
from localflow.ui.settings_window import SettingsWindow


class TestHistoryStorage(unittest.TestCase):
    def test_add_list_delete_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "history.db"
            storage = HistoryStorage(db_path=db_path)

            entry_id = storage.add_entry(
                duration_sec=3.4,
                raw_text="ну привет",
                final_text="Привет.",
                llm_model="qwen2.5:7b-instruct",
                used_llm=True,
            )
            items = storage.list_recent()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].id, entry_id)
            self.assertEqual(items[0].final_text, "Привет.")
            self.assertTrue(items[0].used_llm)

            # Soft delete hides from list_recent() and list_all(), but preserves in list_all(include_deleted=True)
            storage.delete_entry(entry_id, hard=False)
            self.assertEqual(storage.list_recent(), [])
            self.assertEqual(storage.list_all(), [])
            all_rows = storage.list_all(include_deleted=True)
            self.assertEqual(len(all_rows), 1)
            self.assertTrue(all_rows[0].is_deleted)

            # Hard delete physically removes the row
            storage.delete_entry(entry_id, hard=True)
            self.assertEqual(storage.list_all(include_deleted=True), [])

            storage.add_entry(
                duration_sec=1.2,
                raw_text="test",
                final_text="Test.",
                llm_model=None,
                used_llm=False,
            )
            storage.clear(hard=False)
            self.assertEqual(storage.list_recent(), [])
            self.assertEqual(len(storage.list_all(include_deleted=True)), 1)

            storage.clear(hard=True)
            self.assertEqual(storage.list_all(include_deleted=True), [])

    def test_auto_cleanup_older_than_minutes_hours_days_soft_and_hard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "history.db"
            storage = HistoryStorage(db_path=db_path)
            now = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)

            with storage._connect() as conn:
                for offset, text in (
                    (timedelta(minutes=5), "5m ago"),
                    (timedelta(minutes=45), "45m ago"),
                    (timedelta(hours=5), "5h ago"),
                    (timedelta(days=10), "10d ago"),
                ):
                    ts = (now - offset).isoformat()
                    conn.execute(
                        """
                        INSERT INTO transcriptions (
                            created_at, duration_sec, raw_text, final_text, llm_model, used_llm
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (ts, 2.0, text, text, "gemma3-4b", 1),
                    )
                conn.commit()

            # 1. Soft auto-cleanup older than 30 minutes -> hides 45m, 5h, 10d (3 rows), keeps 5m
            removed_soft = storage.apply_auto_cleanup(
                enabled=True,
                value=30,
                unit="minutes",
                hard=False,
                now=now,
            )
            self.assertEqual(removed_soft, 3)
            visible = storage.list_recent()
            self.assertEqual([e.final_text for e in visible], ["5m ago"])
            # All 4 still exist in DB for Insights
            self.assertEqual(len(storage.list_all(include_deleted=True)), 4)

            # 2. Hard auto-cleanup older than 2 hours -> physically deletes 5h and 10d from DB
            removed_hard = storage.apply_auto_cleanup(
                enabled=True,
                value=2,
                unit="hours",
                hard=True,
                now=now,
            )
            self.assertEqual(removed_hard, 2)
            self.assertEqual(len(storage.list_all(include_deleted=True)), 2)


def test_history_ui_auto_delete_and_hard_delete_controls(qapp, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("localflow.ui.settings_window.save_config", lambda cfg, path=None: None)
    storage = HistoryStorage(db_path=tmp_path / "history.db")
    now = datetime.now(timezone.utc)
    with storage._connect() as conn:
        for offset, text in (
            (timedelta(minutes=2), "Fresh entry"),
            (timedelta(days=3), "Old entry"),
        ):
            conn.execute(
                """
                INSERT INTO transcriptions (
                    created_at, duration_sec, raw_text, final_text, llm_model, used_llm
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                ((now - offset).isoformat(), 3.0, text, text, "gemma3-4b", 1),
            )
        conn.commit()

    win = SettingsWindow(AppConfig(language="en"), storage)
    win.refresh_history()
    assert win._history_list.count() == 2
    assert win._auto_delete_check.isChecked() is False
    assert win._auto_delete_spin.isEnabled() is True
    assert win._auto_delete_unit_combo.isEnabled() is True

    # 1. Turning auto-delete ON when matching entries exist prompts confirmation;
    # cancelling reverts the checkbox and preserves all entries
    prompted: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        win,
        "_confirm_auto_delete_enable",
        lambda n, *, hard=False: prompted.append((n, hard)) or False,
    )
    win._auto_delete_spin.setValue(1)
    u_idx = win._auto_delete_unit_combo.findData("days")
    win._auto_delete_unit_combo.setCurrentIndex(u_idx)
    win._auto_delete_check.setChecked(True)
    assert prompted == [(1, False)]
    assert win._auto_delete_check.isChecked() is False
    assert win._history_list.count() == 2

    # 2. Confirming enables auto-delete older than 1 day (soft mode by default)
    monkeypatch.setattr(
        win,
        "_confirm_auto_delete_enable",
        lambda n, *, hard=False: prompted.append((n, hard)) or True,
    )
    win._auto_delete_check.setChecked(True)

    assert prompted == [(1, False), (1, False)]
    assert win._auto_delete_check.isChecked() is True
    assert win._history_list.count() == 1
    assert len(storage.list_recent()) == 1
    assert len(storage.list_all(include_deleted=True)) == 2

    # Switch to hard delete and trigger auto-cleanup for older than 1 minute -> removes both from DB
    win._hard_delete_check.setChecked(True)
    u_min = win._auto_delete_unit_combo.findData("minutes")
    win._auto_delete_unit_combo.setCurrentIndex(u_min)
    win.refresh_history()
    assert win._history_list.count() == 0
    assert storage.list_all(include_deleted=True) == []
    win.close()


if __name__ == "__main__":
    unittest.main()
