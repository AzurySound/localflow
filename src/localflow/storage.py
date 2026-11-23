from __future__ import annotations

from contextlib import contextmanager
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from localflow.config import get_app_data_dir


@dataclass(frozen=True, slots=True)
class TranscriptionEntry:
    id: int
    created_at: str
    duration_sec: float
    raw_text: str
    final_text: str
    llm_model: str | None
    used_llm: bool
    app_category: str = "general"
    app_name: str = ""
    is_command: bool = False
    dict_fixes: int = 0
    style_profile: str = ""


class HistoryStorage:
    """SQLite storage for voice dictations and local usage insights."""

    def __init__(self, db_path: Path | None = None, max_entries: int = 2000) -> None:
        self.db_path = db_path or (get_app_data_dir() / "history.db")
        self.max_entries = max(10, max_entries)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    duration_sec REAL NOT NULL,
                    raw_text TEXT NOT NULL,
                    final_text TEXT NOT NULL,
                    llm_model TEXT,
                    used_llm INTEGER NOT NULL DEFAULT 0,
                    app_category TEXT NOT NULL DEFAULT 'general',
                    app_name TEXT NOT NULL DEFAULT '',
                    is_command INTEGER NOT NULL DEFAULT 0,
                    dict_fixes INTEGER NOT NULL DEFAULT 0,
                    style_profile TEXT NOT NULL DEFAULT ''
                )
                """
            )
            # Safe schema migration for pre-existing history.db files
            existing_cols = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(transcriptions)").fetchall()
            }
            migrations = [
                ("app_category", "TEXT NOT NULL DEFAULT 'general'"),
                ("app_name", "TEXT NOT NULL DEFAULT ''"),
                ("is_command", "INTEGER NOT NULL DEFAULT 0"),
                ("dict_fixes", "INTEGER NOT NULL DEFAULT 0"),
                ("style_profile", "TEXT NOT NULL DEFAULT ''"),
            ]
            for col_name, col_def in migrations:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE transcriptions ADD COLUMN {col_name} {col_def}")

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_transcriptions_created_at ON transcriptions(created_at DESC)"
            )
            conn.commit()

    def add_entry(
        self,
        *,
        duration_sec: float,
        raw_text: str,
        final_text: str,
        llm_model: str | None,
        used_llm: bool,
        app_category: str = "general",
        app_name: str = "",
        is_command: bool = False,
        dict_fixes: int = 0,
        style_profile: str = "",
    ) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO transcriptions (
                    created_at, duration_sec, raw_text, final_text, llm_model, used_llm,
                    app_category, app_name, is_command, dict_fixes, style_profile
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    duration_sec,
                    raw_text,
                    final_text,
                    llm_model,
                    int(used_llm),
                    app_category or "general",
                    app_name or "",
                    int(is_command),
                    max(0, int(dict_fixes)),
                    style_profile or "",
                ),
            )
            conn.execute(
                """
                DELETE FROM transcriptions
                WHERE id NOT IN (
                    SELECT id FROM transcriptions ORDER BY id DESC LIMIT ?
                )
                """,
                (self.max_entries,),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_recent(self, limit: int = 100) -> list[TranscriptionEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, duration_sec, raw_text, final_text, llm_model, used_llm,
                       app_category, app_name, is_command, dict_fixes, style_profile
                FROM transcriptions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            TranscriptionEntry(
                id=int(row["id"]),
                created_at=str(row["created_at"]),
                duration_sec=float(row["duration_sec"]),
                raw_text=str(row["raw_text"]),
                final_text=str(row["final_text"]),
                llm_model=row["llm_model"],
                used_llm=bool(row["used_llm"]),
                app_category=str(row["app_category"] or "general"),
                app_name=str(row["app_name"] or ""),
                is_command=bool(row["is_command"]),
                dict_fixes=int(row["dict_fixes"] or 0),
                style_profile=str(row["style_profile"] or ""),
            )
            for row in rows
        ]

    def list_all(self) -> list[TranscriptionEntry]:
        return self.list_recent(limit=self.max_entries)

    def delete_entry(self, entry_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM transcriptions WHERE id = ?", (entry_id,))
            conn.commit()

    def delete_entries(self, entry_ids: list[int] | tuple[int, ...] | set[int]) -> int:
        ids = [int(i) for i in entry_ids]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            cur = conn.execute(
                f"DELETE FROM transcriptions WHERE id IN ({placeholders})",
                ids,
            )
            conn.commit()
            return int(cur.rowcount)

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM transcriptions")
            conn.commit()
