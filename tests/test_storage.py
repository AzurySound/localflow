import tempfile
import unittest
from pathlib import Path

from localflow.storage import HistoryStorage


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

            storage.delete_entry(entry_id)
            self.assertEqual(storage.list_recent(), [])

            storage.add_entry(
                duration_sec=1.2,
                raw_text="test",
                final_text="Test.",
                llm_model=None,
                used_llm=False,
            )
            storage.clear()
            self.assertEqual(storage.list_recent(), [])


if __name__ == "__main__":
    unittest.main()
