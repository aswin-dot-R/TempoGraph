"""frame_captions schema, WAL concurrency, and DB helpers.

Covers PS1 acceptance:
- frame_captions table exists on new DBs and is added to legacy DBs by migration;
- WAL journal mode is active for concurrent walker/verifier writes;
- insert/fetch/verdict round-trip works end to end;
- two threads writing via separate connections incur no SQLite lock errors.
"""

import sqlite3
import sys
import threading
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage import TempoGraphDB

LEGACY_FRAMES_SQL = """
CREATE TABLE frames (
    frame_idx INTEGER PRIMARY KEY,
    timestamp_ms INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    is_keyframe INTEGER NOT NULL,
    delta_score REAL NOT NULL
);
"""


def _has_table(db_path: Path, name: str) -> bool:
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        )
        return cur.fetchone() is not None


def _journal_mode(db_path: Path) -> str:
    with sqlite3.connect(str(db_path)) as conn:
        return conn.execute("PRAGMA journal_mode").fetchone()[0]


class TestSchema:
    def test_fresh_db_has_frame_captions_table(self, tmp_path):
        db = TempoGraphDB(tmp_path / "t.db")
        db.close()
        assert _has_table(tmp_path / "t.db", "frame_captions")

    def test_legacy_db_migrates_and_old_data_survives(self, tmp_path):
        db_path = tmp_path / "t.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(LEGACY_FRAMES_SQL)
            conn.execute(
                "INSERT INTO frames (frame_idx, timestamp_ms, image_path, "
                "is_keyframe, delta_score) VALUES (0, 100, 'img_0.jpg', 1, 0.5)"
            )
            conn.commit()

        assert not _has_table(db_path, "frame_captions")
        db = TempoGraphDB(db_path)
        assert _has_table(db_path, "frame_captions")

        rows = db._conn.execute("SELECT * FROM frames").fetchall()
        assert len(rows) == 1
        assert rows[0]["image_path"] == "img_0.jpg"
        db.close()

    def test_wal_mode_is_active(self, tmp_path):
        db = TempoGraphDB(tmp_path / "t.db")
        db.close()
        assert _journal_mode(tmp_path / "t.db") == "wal"


class TestHelperRoundTrip:
    def test_insert_fetch_verdict_lifecycle(self, tmp_path):
        db = TempoGraphDB(tmp_path / "t.db")
        db.insert_frame(0, 0, "frame_0.jpg", True, 1.0)

        # Insert a non-escalated caption (should NOT show in unverified escalations)
        db.insert_frame_caption(
            frame_idx=0,
            caption="A person walks into the room.",
            change_line="00:01:23",
            walker_model="qwen-9b-walker",
            escalated=False,
        )
        db.insert_frame_caption(
            frame_idx=1,
            caption="The person sits down.",
            change_line=None,
            walker_model="qwen-9b-walker",
            escalated=True,
        )

        # Only the escalated row should be returned
        unverified = db.fetch_unverified_escalations()
        assert len(unverified) == 1
        assert unverified[0]["frame_idx"] == 1
        assert unverified[0]["escalated"] == 1

        # Verdict: agrees
        db.save_caption_verdict(
            frame_idx=1,
            verifier_caption="A person sits down at a table.",
            verifier_agrees=True,
            verifier_model="qwen-35b-verifier",
        )

        # Escalation no longer returned; counts reflect the update
        assert len(db.fetch_unverified_escalations()) == 0
        assert db.count_frame_captions() == (2, 1, 1)
        row = db.get_frame_caption(1)
        assert row is not None
        assert row["verifier_agrees"] == 1
        assert row["verified_at"] is not None
        assert row["verifier_caption"] == "A person sits down at a table."

        db.close()

    def test_get_frame_caption_returns_none_for_missing(self, tmp_path):
        db = TempoGraphDB(tmp_path / "t.db")
        assert db.get_frame_caption(999) is None
        db.close()


class TestWALConcurrency:
    def test_two_threads_two_connections_no_lock_errors(self, tmp_path):
        db_path = tmp_path / "t.db"
        errors_a: list = []
        errors_b: list = []

        # Main thread: insert 200 frames rows (required FK dependency).
        db = TempoGraphDB(db_path)
        for i in range(200):
            db.insert_frame(i, i * 100, f"frame_{i}.jpg", False, 0.0)
        db.close()

        def thread_a():
            try:
                db_a = TempoGraphDB(db_path)
                for i in range(200):
                    db_a.insert_frame_caption(
                        frame_idx=i,
                        caption=f"Walker caption for frame {i}",
                        change_line=None,
                        walker_model="qwen-9b-walker",
                        escalated=(i % 3 == 0),
                    )
                db_a.close()
            except Exception as e:
                errors_a.append(e)

        def thread_b():
            try:
                db_b = TempoGraphDB(db_path)
                for _ in range(50):
                    escalations = db_b.fetch_unverified_escalations(limit=10)
                    for esc in escalations:
                        db_b.save_caption_verdict(
                            frame_idx=esc["frame_idx"],
                            verifier_caption=f"Verifier: frame {esc['frame_idx']}",
                            verifier_agrees=True,
                            verifier_model="qwen-35b-verifier",
                        )
                db_b.close()
            except Exception as e:
                errors_b.append(e)

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert not errors_a, f"thread A raised: {errors_a}"
        assert not errors_b, f"thread B raised: {errors_b}"

        # Final counts should be consistent: all 200 inserted,
        # escalated ones verified (200 // 3 = 66 escalated, but the exact
        # number depends on interleaving — just assert reasonable bounds).
        db = TempoGraphDB(db_path)
        total, escalated, verified = db.count_frame_captions()
        assert total == 200
        assert 66 <= escalated <= 67  # floor/ceil(200/3)
        assert verified <= escalated  # can't verify more than escalated
        db.close()
