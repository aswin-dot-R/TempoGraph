"""Tests for ``ui.live_view`` — ``fetch_live_state`` headless, no streamlit runtime."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sqlite3
import pytest

from ui.live_view import fetch_live_state


# ── helpers ──────────────────────────────────────────────────────────


def _make_db(tmp_path: Path, rows=None) -> Path:
    """Create a test database with frames and frame_captions tables.

    Args:
        tmp_path: pytest tmp_path directory.
        rows: Optional list of dicts with frame caption data.

    Returns:
        The db_path.
    """
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Drop existing tables to avoid conflicts
    conn.execute("DROP TABLE IF EXISTS frames")
    conn.execute("DROP TABLE IF EXISTS frame_captions")
    conn.execute("DROP TABLE IF EXISTS audio_segments")

    conn.execute(
        """
        CREATE TABLE frames (
            frame_idx INTEGER PRIMARY KEY,
            timestamp_ms INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            is_keyframe INTEGER NOT NULL,
            delta_score REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE frame_captions (
            frame_idx INTEGER PRIMARY KEY,
            caption TEXT NOT NULL,
            change_line TEXT,
            walker_model TEXT NOT NULL,
            escalated INTEGER NOT NULL DEFAULT 0,
            verifier_caption TEXT,
            verifier_agrees INTEGER,
            verifier_model TEXT,
            created_at TEXT NOT NULL,
            verified_at TEXT,
            prompt TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE audio_segments (
            segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL,
            text TEXT NOT NULL
        )
        """
    )

    if rows:
        for i, row in enumerate(rows):
            conn.execute(
                "INSERT INTO frames VALUES (?,?,?,?,?)",
                (
                    row["frame_idx"],
                    row.get("timestamp_ms", 0),
                    row["image_path"],
                    row.get("is_keyframe", 0),
                    row["delta_score"],
                ),
            )
            created_at = (
                datetime(2025, 1, 1, 0, 0, 0) + timedelta(seconds=i * 10)
            ).isoformat()
            verified_at = (
                None
                if not row.get("verified_at", False)
                else (
                    (
                        datetime(2025, 1, 1, 0, 0, 0) + timedelta(seconds=i * 10)
                    ).isoformat()
                )
            )
            conn.execute(
                """
                INSERT INTO frame_captions
                (frame_idx, caption, change_line, walker_model, escalated,
                 verifier_caption, verifier_agrees, verifier_model,
                 created_at, verified_at, prompt)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row["frame_idx"],
                    row["caption"],
                    row.get("change_line", ""),
                    row.get("walker_model", ""),
                    row.get("escalated", 0),
                    row.get("verifier_caption", ""),
                    row.get("verifier_agrees", None),
                    row.get("verifier_model", ""),
                    created_at,
                    verified_at,
                    row.get("prompt", ""),
                ),
            )

        # Insert audio segments if requested
        if rows is not None:
            for i, row in enumerate(rows):
                start_ms = row.get("timestamp_ms", 0) - 500
                end_ms = row.get("timestamp_ms", 0) + 500
                conn.execute(
                    "INSERT INTO audio_segments (start_ms, end_ms, text) VALUES (?,?,?)",
                    (start_ms, end_ms, f"audio segment for frame {row['frame_idx']}"),
                )

    conn.commit()
    conn.close()
    return db_path


# ── Tests ───────────────────────────────────────────────────────────


class TestFetchLiveState:
    """Tests for ``fetch_live_state``."""

    def test_missing_db_file(self, tmp_path):
        """Missing DB file → None."""
        db_path = tmp_path / "nonexistent.db"
        state = fetch_live_state(db_path)
        assert state is None

    def test_empty_frame_captions(self, tmp_path):
        """Empty frame_captions table → None."""
        db_path = _make_db(tmp_path)
        # No rows inserted — frame_captions is empty
        state = fetch_live_state(db_path)
        assert state is None

    def test_populated_fixture(self, tmp_path):
        """Populated fixture → correct current, recent ordering, transcript, verdicts, counts."""
        rows = [
            {
                "frame_idx": 1,
                "image_path": "/tmp/frame_001.jpg",
                "timestamp_ms": 1000,
                "delta_score": 0.1,
                "caption": "a cat on a mat",
                "change_line": "cat appeared",
                "walker_model": "test-9b",
                "escalated": False,
                "prompt": "prompt1",
            },
            {
                "frame_idx": 2,
                "image_path": "/tmp/frame_002.jpg",
                "timestamp_ms": 2000,
                "delta_score": 0.2,
                "caption": "a dog on a bed",
                "change_line": "dog appeared",
                "walker_model": "test-9b",
                "escalated": True,
                "prompt": "prompt2",
            },
            {
                "frame_idx": 3,
                "image_path": "/tmp/frame_003.jpg",
                "timestamp_ms": 3000,
                "delta_score": 0.3,
                "caption": "a bird in a tree",
                "change_line": "bird appeared",
                "walker_model": "test-9b",
                "escalated": False,
                "prompt": "prompt3",
                "verified_at": True,
                "verifier_agrees": 1,
            },
        ]

        db_path = _make_db(tmp_path, rows=rows)

        state = fetch_live_state(db_path, n=10)

        assert state is not None
        assert "current" in state
        assert "recent" in state
        assert "transcript" in state
        assert "verdicts" in state
        assert "counts" in state

        # current should be the highest frame_idx (frame 3, since it has verified_at)
        assert state["current"]["frame_idx"] == 3
        assert state["current"]["caption"] == "a bird in a tree"
        assert state["current"]["change_line"] == "bird appeared"
        assert state["current"]["timestamp_ms"] == 3000
        assert state["current"]["image_path"] == "/tmp/frame_003.jpg"
        assert state["current"]["prompt"] == "prompt3"

        # recent should have all 3 frames, newest first
        assert len(state["recent"]) == 3
        assert state["recent"][0]["frame_idx"] == 3
        assert state["recent"][0]["caption"] == "a bird in a tree"
        assert state["recent"][1]["frame_idx"] == 2
        assert state["recent"][1]["caption"] == "a dog on a bed"
        assert state["recent"][2]["frame_idx"] == 1
        assert state["recent"][2]["caption"] == "a cat on a mat"

        # transcript: audio segments overlapping ±1 s around current timestamp
        # (3000 → window [2000, 4000]): segments [1500,2500] and [2500,3500]
        # overlap; [500,1500] does not.
        assert len(state["transcript"]) == 2
        # The segment for frame 3 spans 2500-3500, which overlaps [2000, 4000]
        assert any("frame 3" in seg["text"] for seg in state["transcript"])

        # verdicts: last 5 verified rows (only frame 3 is verified)
        assert len(state["verdicts"]) == 1
        assert state["verdicts"][0]["frame_idx"] == 3
        assert state["verdicts"][0]["verifier_agrees"] is True
        assert state["verdicts"][0]["verifier_model"] == ""

        # counts
        assert state["counts"]["captioned"] == 3
        assert state["counts"]["escalated"] == 1
        assert state["counts"]["verified"] == 1

    def test_n_limit(self, tmp_path):
        """n parameter limits the number of recent frames."""
        rows = [
            {
                "frame_idx": i,
                "image_path": f"/tmp/frame_{i}.jpg",
                "timestamp_ms": i * 1000,
                "delta_score": 0.1,
                "caption": f"frame {i}",
                "change_line": "",
                "walker_model": "test-9b",
                "prompt": "",
            }
            for i in range(1, 21)
        ]
        db_path = _make_db(tmp_path, rows=rows)

        state = fetch_live_state(db_path, n=5)
        assert len(state["recent"]) == 5
        assert state["recent"][0]["frame_idx"] == 20
        assert state["recent"][-1]["frame_idx"] == 16

    def test_readonly_connection(self, tmp_path):
        """After fetch_live_state, a writer connection can still INSERT (no lingering lock)."""
        rows = [
            {
                "frame_idx": 1,
                "image_path": "/tmp/frame_001.jpg",
                "timestamp_ms": 1000,
                "delta_score": 0.1,
                "caption": "test",
                "change_line": "",
                "walker_model": "",
                "prompt": "",
            }
        ]
        db_path = _make_db(tmp_path, rows=rows)

        # Fetch live state (read-only)
        state = fetch_live_state(db_path)
        assert state is not None

        # Now open a writer connection and INSERT a new row
        writer_conn = sqlite3.connect(str(db_path))
        writer_conn.execute(
            "INSERT INTO frame_captions (frame_idx, caption, change_line, walker_model, created_at, prompt) VALUES (99, 'new row', '', '', '2025-01-01T00:00:00', '')"
        )
        writer_conn.commit()

        # Verify the writer can read the new row
        row = writer_conn.execute(
            "SELECT caption FROM frame_captions WHERE frame_idx = 99"
        ).fetchone()
        assert row[0] == "new row"

        writer_conn.close()

    def test_missing_table(self, tmp_path):
        """Missing required tables → None."""
        db_path = tmp_path / "no_tables.db"
        db_path.touch()
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE dummy (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        state = fetch_live_state(db_path)
        assert state is None
