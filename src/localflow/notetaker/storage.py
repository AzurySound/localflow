"""SQLite + FTS5 persistence for LocalFlow Notetaker meetings, speakers, utterances, and action items."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
from typing import Iterator

from localflow.config import get_app_data_dir


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass(frozen=True, slots=True)
class MeetingRecord:
    id: int
    title: str
    created_at: str
    ended_at: str | None
    duration_sec: float
    audio_mode: str
    status: str
    summary_md: str
    my_thoughts: str
    llm_model: str | None
    is_deleted: bool = False


@dataclass(frozen=True, slots=True)
class MeetingSpeaker:
    id: int
    meeting_id: int
    speaker_key: str
    display_name: str
    color_idx: int
    is_self: bool


@dataclass(frozen=True, slots=True)
class MeetingUtterance:
    id: int
    meeting_id: int
    speaker_key: str
    channel: str
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True, slots=True)
class MeetingActionItem:
    id: int
    meeting_id: int
    text: str
    owner: str
    done: bool
    sort_order: int


@dataclass(frozen=True, slots=True)
class MeetingSearchHit:
    meeting_id: int
    title: str
    created_at: str
    snippet: str
    score: float


_FTS_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _sanitize_fts_query(query: str) -> str:
    tokens = _FTS_TOKEN_RE.findall(query.strip())
    if not tokens:
        return ""
    return " OR ".join(f'"{tok}"*' for tok in tokens[:12])


class MeetingStorage:
    """SQLite WAL + FTS5 storage for meetings, diarized utterances, action items, and notes."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or (get_app_data_dir() / "meetings.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=8.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
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
                CREATE TABLE IF NOT EXISTS meetings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_sec REAL NOT NULL DEFAULT 0.0,
                    audio_mode TEXT NOT NULL DEFAULT 'dual',
                    status TEXT NOT NULL DEFAULT 'recording',
                    summary_md TEXT NOT NULL DEFAULT '',
                    my_thoughts TEXT NOT NULL DEFAULT '',
                    llm_model TEXT,
                    is_deleted INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meeting_speakers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                    speaker_key TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    color_idx INTEGER NOT NULL DEFAULT 0,
                    is_self INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(meeting_id, speaker_key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meeting_utterances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                    speaker_key TEXT NOT NULL,
                    channel TEXT NOT NULL DEFAULT 'system',
                    start_sec REAL NOT NULL DEFAULT 0.0,
                    end_sec REAL NOT NULL DEFAULT 0.0,
                    text TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meeting_action_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                    text TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT '',
                    done INTEGER NOT NULL DEFAULT 0,
                    sort_order INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS meetings_fts USING fts5(
                    meeting_id UNINDEXED,
                    title,
                    summary_md,
                    my_thoughts,
                    transcript_text,
                    tokenize='unicode61'
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_meetings_created_at ON meetings(created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_utterances_meeting ON meeting_utterances(meeting_id, start_sec)"
            )
            conn.commit()

    def _row_to_meeting(self, row: sqlite3.Row) -> MeetingRecord:
        return MeetingRecord(
            id=int(row["id"]),
            title=str(row["title"]),
            created_at=str(row["created_at"]),
            ended_at=str(row["ended_at"]) if row["ended_at"] is not None else None,
            duration_sec=float(row["duration_sec"]),
            audio_mode=str(row["audio_mode"]),
            status=str(row["status"]),
            summary_md=str(row["summary_md"]),
            my_thoughts=str(row["my_thoughts"]),
            llm_model=str(row["llm_model"]) if row["llm_model"] is not None else None,
            is_deleted=bool(row["is_deleted"]),
        )

    def _sync_fts(self, conn: sqlite3.Connection, meeting_id: int) -> None:
        row = conn.execute(
            "SELECT id, title, summary_md, my_thoughts, is_deleted FROM meetings WHERE id = ?",
            (meeting_id,),
        ).fetchone()
        conn.execute("DELETE FROM meetings_fts WHERE meeting_id = ?", (meeting_id,))
        if row is None or bool(row["is_deleted"]):
            return
        utterances = conn.execute(
            "SELECT speaker_key, text FROM meeting_utterances WHERE meeting_id = ? ORDER BY start_sec ASC, id ASC",
            (meeting_id,),
        ).fetchall()
        speakers = {
            str(s["speaker_key"]): str(s["display_name"])
            for s in conn.execute(
                "SELECT speaker_key, display_name FROM meeting_speakers WHERE meeting_id = ?",
                (meeting_id,),
            ).fetchall()
        }
        transcript_text = "\n".join(
            f"{speakers.get(str(u['speaker_key']), str(u['speaker_key']))}: {u['text']}"
            for u in utterances
        )
        conn.execute(
            "INSERT INTO meetings_fts (meeting_id, title, summary_md, my_thoughts, transcript_text) VALUES (?, ?, ?, ?, ?)",
            (
                meeting_id,
                str(row["title"]),
                str(row["summary_md"]),
                str(row["my_thoughts"]),
                transcript_text,
            ),
        )

    def create_meeting(
        self,
        *,
        title: str = "New Meeting",
        audio_mode: str = "dual",
        status: str = "recording",
        created_at: str | None = None,
    ) -> MeetingRecord:
        ts = created_at or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO meetings (title, created_at, audio_mode, status)
                VALUES (?, ?, ?, ?)
                """,
                (title, ts, audio_mode, status),
            )
            meeting_id = int(cur.lastrowid)
            self._sync_fts(conn, meeting_id)
            conn.commit()
            row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
            assert row is not None
            return self._row_to_meeting(row)

    def get_meeting(self, meeting_id: int) -> MeetingRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM meetings WHERE id = ? AND is_deleted = 0",
                (meeting_id,),
            ).fetchone()
            return self._row_to_meeting(row) if row is not None else None

    def list_meetings(self, *, search: str = "", limit: int = 100) -> list[MeetingRecord]:
        with self._connect() as conn:
            q = search.strip()
            if q:
                fts_q = _sanitize_fts_query(q)
                if fts_q:
                    rows = conn.execute(
                        """
                        SELECT m.* FROM meetings m
                        JOIN meetings_fts f ON f.meeting_id = m.id
                        WHERE m.is_deleted = 0 AND meetings_fts MATCH ?
                        ORDER BY bm25(meetings_fts), m.created_at DESC
                        LIMIT ?
                        """,
                        (fts_q, max(1, limit)),
                    ).fetchall()
                    return [self._row_to_meeting(r) for r in rows]
                like_q = f"%{q}%"
                rows = conn.execute(
                    """
                    SELECT * FROM meetings
                    WHERE is_deleted = 0 AND (title LIKE ? OR summary_md LIKE ? OR my_thoughts LIKE ?)
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (like_q, like_q, like_q, max(1, limit)),
                ).fetchall()
                return [self._row_to_meeting(r) for r in rows]

            rows = conn.execute(
                "SELECT * FROM meetings WHERE is_deleted = 0 ORDER BY created_at DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
            return [self._row_to_meeting(r) for r in rows]

    def update_meeting(
        self,
        meeting_id: int,
        *,
        title: str | None = None,
        ended_at: str | None = None,
        duration_sec: float | None = None,
        status: str | None = None,
        summary_md: str | None = None,
        my_thoughts: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        fields: list[str] = []
        values: list[object] = []
        if title is not None:
            fields.append("title = ?")
            values.append(title)
        if ended_at is not None:
            fields.append("ended_at = ?")
            values.append(ended_at)
        if duration_sec is not None:
            fields.append("duration_sec = ?")
            values.append(float(duration_sec))
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if summary_md is not None:
            fields.append("summary_md = ?")
            values.append(summary_md)
        if my_thoughts is not None:
            fields.append("my_thoughts = ?")
            values.append(my_thoughts)
        if llm_model is not None:
            fields.append("llm_model = ?")
            values.append(llm_model)
        if not fields:
            return
        values.append(meeting_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE meetings SET {', '.join(fields)} WHERE id = ?", values)
            self._sync_fts(conn, meeting_id)
            conn.commit()

    def delete_meeting(self, meeting_id: int, *, hard: bool = False) -> None:
        with self._connect() as conn:
            if hard:
                conn.execute("DELETE FROM meetings_fts WHERE meeting_id = ?", (meeting_id,))
                conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
            else:
                conn.execute("UPDATE meetings SET is_deleted = 1 WHERE id = ?", (meeting_id,))
                self._sync_fts(conn, meeting_id)
            conn.commit()

    def ensure_speaker(
        self,
        meeting_id: int,
        speaker_key: str,
        *,
        display_name: str,
        color_idx: int = 0,
        is_self: bool = False,
    ) -> MeetingSpeaker:
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM meeting_speakers WHERE meeting_id = ? AND speaker_key = ?",
                (meeting_id, speaker_key),
            ).fetchone()
            if existing is not None:
                return MeetingSpeaker(
                    id=int(existing["id"]),
                    meeting_id=int(existing["meeting_id"]),
                    speaker_key=str(existing["speaker_key"]),
                    display_name=str(existing["display_name"]),
                    color_idx=int(existing["color_idx"]),
                    is_self=bool(existing["is_self"]),
                )
            cur = conn.execute(
                """
                INSERT INTO meeting_speakers (meeting_id, speaker_key, display_name, color_idx, is_self)
                VALUES (?, ?, ?, ?, ?)
                """,
                (meeting_id, speaker_key, display_name, color_idx, int(is_self)),
            )
            conn.commit()
            return MeetingSpeaker(
                id=int(cur.lastrowid),
                meeting_id=meeting_id,
                speaker_key=speaker_key,
                display_name=display_name,
                color_idx=color_idx,
                is_self=is_self,
            )

    def get_speakers(self, meeting_id: int) -> list[MeetingSpeaker]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM meeting_speakers WHERE meeting_id = ? ORDER BY is_self DESC, id ASC",
                (meeting_id,),
            ).fetchall()
            return [
                MeetingSpeaker(
                    id=int(r["id"]),
                    meeting_id=int(r["meeting_id"]),
                    speaker_key=str(r["speaker_key"]),
                    display_name=str(r["display_name"]),
                    color_idx=int(r["color_idx"]),
                    is_self=bool(r["is_self"]),
                )
                for r in rows
            ]

    def rename_speaker(self, meeting_id: int, speaker_key: str, new_display_name: str) -> None:
        clean = new_display_name.strip()
        if not clean:
            return
        with self._connect() as conn:
            conn.execute(
                "UPDATE meeting_speakers SET display_name = ? WHERE meeting_id = ? AND speaker_key = ?",
                (clean, meeting_id, speaker_key),
            )
            self._sync_fts(conn, meeting_id)
            conn.commit()

    def merge_speakers(self, meeting_id: int, source_key: str, target_key: str) -> None:
        """Merge all utterances from source_key into target_key and remove source speaker."""
        if not source_key or not target_key or source_key == target_key:
            return
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE meeting_utterances
                SET speaker_key = ?
                WHERE meeting_id = ? AND speaker_key = ?
                """,
                (target_key, meeting_id, source_key),
            )
            conn.execute(
                "DELETE FROM meeting_speakers WHERE meeting_id = ? AND speaker_key = ?",
                (meeting_id, source_key),
            )
            self._sync_fts(conn, meeting_id)
            conn.commit()

    def add_utterance(
        self,
        meeting_id: int,
        *,
        speaker_key: str,
        channel: str,
        start_sec: float,
        end_sec: float,
        text: str,
    ) -> MeetingUtterance:
        clean = text.strip()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO meeting_utterances (meeting_id, speaker_key, channel, start_sec, end_sec, text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (meeting_id, speaker_key, channel, float(start_sec), float(end_sec), clean),
            )
            utt_id = int(cur.lastrowid)
            self._sync_fts(conn, meeting_id)
            conn.commit()
            return MeetingUtterance(
                id=utt_id,
                meeting_id=meeting_id,
                speaker_key=speaker_key,
                channel=channel,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                text=clean,
            )

    def update_utterance_speaker(self, utterance_id: int, new_speaker_key: str) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT meeting_id FROM meeting_utterances WHERE id = ?",
                (utterance_id,),
            ).fetchone()
            if row is None:
                return
            meeting_id = int(row["meeting_id"])
            conn.execute(
                "UPDATE meeting_utterances SET speaker_key = ? WHERE id = ?",
                (new_speaker_key, utterance_id),
            )
            self._sync_fts(conn, meeting_id)
            conn.commit()

    def apply_speaker_remap(self, meeting_id: int, utterance_to_speaker: dict[int, str], speaker_names: dict[str, str]) -> None:
        """Batch-update utterance speaker assignments after post-call agglomerative clustering."""
        with self._connect() as conn:
            for utt_id, spk_key in utterance_to_speaker.items():
                conn.execute(
                    "UPDATE meeting_utterances SET speaker_key = ? WHERE id = ? AND meeting_id = ?",
                    (spk_key, int(utt_id), meeting_id),
                )
            # Prune non-self speakers with no remaining utterances
            used_keys = {
                str(r["speaker_key"])
                for r in conn.execute(
                    "SELECT DISTINCT speaker_key FROM meeting_utterances WHERE meeting_id = ?",
                    (meeting_id,),
                ).fetchall()
            }
            existing_speakers = conn.execute(
                "SELECT speaker_key, is_self FROM meeting_speakers WHERE meeting_id = ?",
                (meeting_id,),
            ).fetchall()
            for row in existing_speakers:
                key = str(row["speaker_key"])
                if not bool(row["is_self"]) and key not in used_keys:
                    conn.execute(
                        "DELETE FROM meeting_speakers WHERE meeting_id = ? AND speaker_key = ?",
                        (meeting_id, key),
                    )
            for idx, (key, name) in enumerate(speaker_names.items(), start=1):
                if key in used_keys:
                    existing = conn.execute(
                        "SELECT id FROM meeting_speakers WHERE meeting_id = ? AND speaker_key = ?",
                        (meeting_id, key),
                    ).fetchone()
                    if existing is None:
                        conn.execute(
                            """
                            INSERT INTO meeting_speakers (meeting_id, speaker_key, display_name, color_idx, is_self)
                            VALUES (?, ?, ?, ?, 0)
                            """,
                            (meeting_id, key, name, idx),
                        )
            self._sync_fts(conn, meeting_id)
            conn.commit()

    def get_utterances(self, meeting_id: int) -> list[MeetingUtterance]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM meeting_utterances WHERE meeting_id = ? ORDER BY start_sec ASC, id ASC",
                (meeting_id,),
            ).fetchall()
            return [
                MeetingUtterance(
                    id=int(r["id"]),
                    meeting_id=int(r["meeting_id"]),
                    speaker_key=str(r["speaker_key"]),
                    channel=str(r["channel"]),
                    start_sec=float(r["start_sec"]),
                    end_sec=float(r["end_sec"]),
                    text=str(r["text"]),
                )
                for r in rows
            ]

    def replace_action_items(self, meeting_id: int, items: list[tuple[str, str]]) -> list[MeetingActionItem]:
        """Replace all action items for meeting_id. items = [(text, owner), ...]."""
        with self._connect() as conn:
            conn.execute("DELETE FROM meeting_action_items WHERE meeting_id = ?", (meeting_id,))
            created: list[MeetingActionItem] = []
            for idx, (text, owner) in enumerate(items):
                clean_text = text.strip()
                if not clean_text:
                    continue
                cur = conn.execute(
                    """
                    INSERT INTO meeting_action_items (meeting_id, text, owner, done, sort_order)
                    VALUES (?, ?, ?, 0, ?)
                    """,
                    (meeting_id, clean_text, owner.strip(), idx),
                )
                created.append(
                    MeetingActionItem(
                        id=int(cur.lastrowid),
                        meeting_id=meeting_id,
                        text=clean_text,
                        owner=owner.strip(),
                        done=False,
                        sort_order=idx,
                    )
                )
            conn.commit()
            return created

    def get_action_items(self, meeting_id: int) -> list[MeetingActionItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM meeting_action_items WHERE meeting_id = ? ORDER BY sort_order ASC, id ASC",
                (meeting_id,),
            ).fetchall()
            return [
                MeetingActionItem(
                    id=int(r["id"]),
                    meeting_id=int(r["meeting_id"]),
                    text=str(r["text"]),
                    owner=str(r["owner"]),
                    done=bool(r["done"]),
                    sort_order=int(r["sort_order"]),
                )
                for r in rows
            ]

    def set_action_item_done(self, item_id: int, done: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE meeting_action_items SET done = ? WHERE id = ?",
                (int(done), item_id),
            )
            conn.commit()

    def search_fts(
        self,
        query: str,
        *,
        meeting_id: int | None = None,
        limit: int = 8,
    ) -> list[MeetingSearchHit]:
        """Search meetings via SQLite FTS5 BM25 ranking."""
        fts_q = _sanitize_fts_query(query)
        if not fts_q:
            return []
        with self._connect() as conn:
            if meeting_id is not None:
                rows = conn.execute(
                    """
                    SELECT m.id, m.title, m.created_at,
                           snippet(meetings_fts, -1, '', '', ' ... ', 32) AS snip,
                           bm25(meetings_fts) AS rank_score
                    FROM meetings_fts
                    JOIN meetings m ON m.id = meetings_fts.meeting_id
                    WHERE m.is_deleted = 0 AND m.id = ? AND meetings_fts MATCH ?
                    ORDER BY rank_score ASC
                    LIMIT ?
                    """,
                    (meeting_id, fts_q, max(1, limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT m.id, m.title, m.created_at,
                           snippet(meetings_fts, -1, '', '', ' ... ', 32) AS snip,
                           bm25(meetings_fts) AS rank_score
                    FROM meetings_fts
                    JOIN meetings m ON m.id = meetings_fts.meeting_id
                    WHERE m.is_deleted = 0 AND meetings_fts MATCH ?
                    ORDER BY rank_score ASC
                    LIMIT ?
                    """,
                    (fts_q, max(1, limit)),
                ).fetchall()
            return [
                MeetingSearchHit(
                    meeting_id=int(r["id"]),
                    title=str(r["title"]),
                    created_at=str(r["created_at"]),
                    snippet=str(r["snip"] or ""),
                    score=float(r["rank_score"] or 0.0),
                )
                for r in rows
            ]

    def export_markdown(self, meeting_id: int) -> str:
        """Render a complete clean Markdown export of a meeting."""
        meeting = self.get_meeting(meeting_id)
        if meeting is None:
            return ""
        speakers = {s.speaker_key: s.display_name for s in self.get_speakers(meeting_id)}
        utterances = self.get_utterances(meeting_id)
        actions = self.get_action_items(meeting_id)

        lines: list[str] = [
            f"# {meeting.title}",
            "",
            f"- Date: {meeting.created_at[:19].replace('T', ' ')}",
            f"- Duration: {format_timestamp(meeting.duration_sec)}",
            f"- Mode: {meeting.audio_mode}",
            "",
        ]
        if meeting.summary_md.strip():
            lines.extend(["## Summary", "", meeting.summary_md.strip(), ""])
        if actions:
            lines.append("## Action Items")
            lines.append("")
            for item in actions:
                mark = "x" if item.done else " "
                owner_part = f" ({item.owner})" if item.owner else ""
                lines.append(f"- [{mark}] {item.text}{owner_part}")
            lines.append("")
        if meeting.my_thoughts.strip():
            lines.extend(["## My Thoughts", "", meeting.my_thoughts.strip(), ""])
        if utterances:
            lines.append("## Transcript")
            lines.append("")
            for u in utterances:
                spk = speakers.get(u.speaker_key, u.speaker_key)
                ts = format_timestamp(u.start_sec)
                lines.append(f"**[{ts}] {spk}:** {u.text}")
            lines.append("")
        return "\n".join(lines).strip() + "\n"
