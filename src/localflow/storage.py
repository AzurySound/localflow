from __future__ import annotations

from contextlib import contextmanager
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
    is_deleted: bool = False


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
                    style_profile TEXT NOT NULL DEFAULT '',
                    is_deleted INTEGER NOT NULL DEFAULT 0
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
                ("is_deleted", "INTEGER NOT NULL DEFAULT 0"),
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

    def list_recent(
        self,
        limit: int = 100,
        *,
        include_deleted: bool = False,
    ) -> list[TranscriptionEntry]:
        where_clause = "" if include_deleted else "WHERE is_deleted = 0"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, created_at, duration_sec, raw_text, final_text, llm_model, used_llm,
                       app_category, app_name, is_command, dict_fixes, style_profile, is_deleted
                FROM transcriptions
                {where_clause}
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
                is_deleted=bool(row["is_deleted"]),
            )
            for row in rows
        ]

    def list_all(self, *, include_deleted: bool = False) -> list[TranscriptionEntry]:
        return self.list_recent(limit=self.max_entries, include_deleted=include_deleted)

    def update_entry_cleanup(
        self,
        entry_id: int,
        *,
        final_text: str,
        llm_model: str | None,
        used_llm: bool,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE transcriptions
                SET final_text = ?, llm_model = ?, used_llm = ?
                WHERE id = ?
                """,
                (final_text, llm_model, int(used_llm), int(entry_id)),
            )
            conn.commit()

    def delete_entry(self, entry_id: int, *, hard: bool = False) -> None:
        with self._connect() as conn:
            if hard:
                conn.execute("DELETE FROM transcriptions WHERE id = ?", (entry_id,))
            else:
                conn.execute(
                    "UPDATE transcriptions SET is_deleted = 1 WHERE id = ?",
                    (entry_id,),
                )
            conn.commit()

    def delete_entries(
        self,
        entry_ids: list[int] | tuple[int, ...] | set[int],
        *,
        hard: bool = False,
    ) -> int:
        ids = [int(i) for i in entry_ids]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            if hard:
                cur = conn.execute(
                    f"DELETE FROM transcriptions WHERE id IN ({placeholders})",
                    ids,
                )
            else:
                cur = conn.execute(
                    f"UPDATE transcriptions SET is_deleted = 1 WHERE id IN ({placeholders}) AND is_deleted = 0",
                    ids,
                )
            conn.commit()
            return int(cur.rowcount)

    @staticmethod
    def _parse_utc_timestamp(ts: str) -> datetime | None:
        raw = (ts or "").strip()
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _find_ids_older_than(
        self,
        cutoff_utc: datetime | str,
        *,
        hard: bool = False,
    ) -> list[int]:
        if isinstance(cutoff_utc, str):
            cutoff_dt = self._parse_utc_timestamp(cutoff_utc)
        elif cutoff_utc.tzinfo is None:
            cutoff_dt = cutoff_utc.replace(tzinfo=timezone.utc)
        else:
            cutoff_dt = cutoff_utc.astimezone(timezone.utc)
        if cutoff_dt is None:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, is_deleted FROM transcriptions"
            ).fetchall()

        target_ids: list[int] = []
        for row in rows:
            if not hard and bool(row["is_deleted"]):
                continue
            dt = self._parse_utc_timestamp(str(row["created_at"]))
            if dt is not None and dt < cutoff_dt:
                target_ids.append(int(row["id"]))
        return target_ids

    def delete_older_than(
        self,
        cutoff_utc: datetime | str,
        *,
        hard: bool = False,
    ) -> int:
        """Soft-delete or hard-delete entries older than cutoff_utc."""
        target_ids = self._find_ids_older_than(cutoff_utc, hard=hard)
        if not target_ids:
            return 0
        return self.delete_entries(target_ids, hard=hard)

    @staticmethod
    def _compute_auto_cleanup_cutoff(
        value: int,
        unit: str,
        *,
        now: datetime | None = None,
    ) -> datetime:
        val = max(1, int(value))
        norm_unit = (unit or "days").strip().lower()
        if norm_unit == "minutes":
            delta = timedelta(minutes=val)
        elif norm_unit == "hours":
            delta = timedelta(hours=val)
        else:
            delta = timedelta(days=val)
        ref_now = now if now is not None else datetime.now(timezone.utc)
        if ref_now.tzinfo is None:
            ref_now = ref_now.replace(tzinfo=timezone.utc)
        return ref_now - delta

    def count_auto_cleanup(
        self,
        *,
        enabled: bool,
        value: int,
        unit: str,
        hard: bool = False,
        now: datetime | None = None,
    ) -> int:
        """Return how many existing records match the auto-cleanup cutoff right now."""
        if not enabled:
            return 0
        cutoff = self._compute_auto_cleanup_cutoff(value, unit, now=now)
        return len(self._find_ids_older_than(cutoff, hard=hard))

    def apply_auto_cleanup(
        self,
        *,
        enabled: bool,
        value: int,
        unit: str,
        hard: bool = False,
        now: datetime | None = None,
    ) -> int:
        """Apply automatic age-based history cleanup when enabled."""
        if not enabled:
            return 0
        cutoff = self._compute_auto_cleanup_cutoff(value, unit, now=now)
        return self.delete_older_than(cutoff, hard=hard)

    def clear(self, *, hard: bool = False) -> None:
        with self._connect() as conn:
            if hard:
                conn.execute("DELETE FROM transcriptions")
            else:
                conn.execute("UPDATE transcriptions SET is_deleted = 1 WHERE is_deleted = 0")
            conn.commit()
