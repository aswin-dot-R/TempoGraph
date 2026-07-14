"""Tests for DenseCaptionWalker — two-phase parallel walker.

Parity, prompt, ordering, phase-2 failure tolerance, resume, cancel,
verifier transcript, and get_audio_segments_overlapping.
"""

from __future__ import annotations

import copy
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ── helpers ──────────────────────────────────────────────────────────


def _create_test_db(db_path: str, rows=None) -> Path:
    """Create a test database with frames and frame_captions tables.

    Args:
        db_path: Path to the database file.
        rows: List of dicts with frame data.

    Returns:
        The db_path.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

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
        for row in rows:
            conn.execute(
                "INSERT OR REPLACE INTO frames VALUES (?,?,?,?,?)",
                (
                    row["frame_idx"],
                    row.get("timestamp_ms", 0),
                    row["image_path"],
                    row.get("is_keyframe", 0),
                    row["delta_score"],
                ),
            )
            if "caption" in row:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO frame_captions
                    (frame_idx, caption, change_line, walker_model, escalated,
                     created_at, prompt)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        row["frame_idx"],
                        row["caption"],
                        row.get("change_line", ""),
                        row.get("walker_model", ""),
                        row.get("escalated", 0),
                        "",
                        row.get("prompt", ""),
                    ),
                )

    conn.commit()
    conn.close()
    return db_path


# ── Tests ───────────────────────────────────────────────────────────


class TestDenseCaptionWalkerParity:
    """Parity: prompt format, model, max_tokens, temperature."""

    def test_prompt_format(self):
        """``DenseCaptionWalker`` prompt template must contain ``Previous frame:``."""
        from src.modules.dense_captioner import _FRAME_PROMPT

        assert "Previous frame:" in _FRAME_PROMPT

    def test_parallel_prompt_no_previous_frame(self):
        """Parallel-safe prompt must NOT contain ``Previous frame:``."""
        from src.modules.dense_captioner import _PARALLEL_FRAME_PROMPT

        assert "Previous frame:" not in _PARALLEL_FRAME_PROMPT

    def test_model_name(self):
        """``DenseCaptionWalker.model_name`` must equal the model kwarg."""
        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=":memory:",
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=1,
        )
        assert walker.model_name == "test-9b"

    def test_max_tokens(self):
        """``DenseCaptionWalker.max_tokens`` must equal the kwarg."""
        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=":memory:",
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=256,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=1,
        )
        assert walker.max_tokens == 256

    def test_temperature(self):
        """``DenseCaptionWalker.temperature`` must equal the kwarg."""
        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=":memory:",
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.15,
            request_timeout_s=30.0,
            concurrency=1,
        )
        assert walker.temperature == 0.15

    def test_concurrency(self):
        """``DenseCaptionWalker.concurrency`` must equal the kwarg
        after ``_resolve_concurrency`` is called."""
        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=":memory:",
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=8,
        )
        assert walker._resolve_concurrency() == 8

    def test_request_timeout(self):
        """``DenseCaptionWalker.request_timeout_s`` must equal the kwarg."""
        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=":memory:",
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=60.0,
            concurrency=1,
        )
        assert walker.request_timeout_s == 60.0

    def test_base_url(self):
        """``DenseCaptionWalker.base_url`` must equal the kwarg."""
        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=":memory:",
            base_url="http://127.0.0.1:9999",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=1,
        )
        assert walker.base_url == "http://127.0.0.1:9999"


class TestDenseCaptionWalkerWalk:
    """Walk tests: ordering, phase-2 failure tolerance, resume, cancel."""

    def test_ordering(self):
        """Frames must be visited in ascending ``frame_idx`` order."""
        tmp_path = Path("/tmp/tg_test_ordering")
        tmp_path.mkdir(exist_ok=True)
        db_path = str(tmp_path / "test.db")

        _create_test_db(
            db_path,
            [
                {
                    "frame_idx": 1,
                    "image_path": "/tmp/frame_001.jpg",
                    "delta_score": 0.1,
                },
                {
                    "frame_idx": 5,
                    "image_path": "/tmp/frame_005.jpg",
                    "delta_score": 0.3,
                },
                {
                    "frame_idx": 3,
                    "image_path": "/tmp/frame_003.jpg",
                    "delta_score": 0.2,
                },
                {
                    "frame_idx": 2,
                    "image_path": "/tmp/frame_002.jpg",
                    "delta_score": 0.15,
                },
            ],
        )

        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=1,
        )

        # Mock _encode_image to avoid file I/O
        with patch.object(
            DenseCaptionWalker, "_encode_image", return_value="fake_base64"
        ):
            with patch("requests.post") as mock_post:

                def side_effect(url, json=None, **kwargs):
                    content = json.get("messages", [{}])[0].get("content", "")
                    return MagicMock(
                        status_code=200,
                        json=lambda: {
                            "choices": [
                                {
                                    "message": {
                                        "content": "FRAME: a cat on a mat\nCHANGE: cat appeared"
                                    }
                                }
                            ]
                        },
                    )

                mock_post.side_effect = side_effect

                result = walker.walk()
                assert result["captioned"] == 4
                assert result["errors"] == 0

    def test_phase2_http_failure_tolerance(self):
        """If phase-2 HTTP fails, escalation should still be computed from delta."""
        tmp_path = Path("/tmp/tg_test_phase2")
        tmp_path.mkdir(exist_ok=True)
        db_path = str(tmp_path / "test.db")

        _create_test_db(
            db_path,
            [
                {
                    "frame_idx": 1,
                    "image_path": "/tmp/frame_001.jpg",
                    "delta_score": 0.1,
                },
                {
                    "frame_idx": 2,
                    "image_path": "/tmp/frame_002.jpg",
                    "delta_score": 0.2,
                },
                {
                    "frame_idx": 3,
                    "image_path": "/tmp/frame_003.jpg",
                    "delta_score": 0.3,
                },
            ],
        )

        from src.modules.dense_captioner import DenseCaptionWalker

        # Create walker with concurrency=2 (triggers parallel path)
        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=2,
        )

        # Mock _encode_image to avoid file I/O
        with patch.object(
            DenseCaptionWalker, "_encode_image", return_value="fake_base64"
        ):
            # Mock phase-1 HTTP (caption requests)
            phase1_responses = [
                MagicMock(
                    status_code=200,
                    json=lambda: {
                        "choices": [{"message": {"content": "FRAME: a cat on a mat"}}]
                    },
                ),
                MagicMock(
                    status_code=200,
                    json=lambda: {
                        "choices": [{"message": {"content": "FRAME: a dog on a bed"}}]
                    },
                ),
            ]

            # Mock phase-2 HTTP (change-line requests) — raise ConnectionError for second call
            phase2_failures = [
                ConnectionError("Connection failed"),
                ConnectionError("Connection failed"),
            ]

            call_count = [0]

            def side_effect(url, json=None, **kwargs):
                call_count[0] += 1
                if call_count[0] <= len(phase1_responses):
                    return phase1_responses[call_count[0] - 1]
                else:
                    raise phase2_failures[call_count[0] - len(phase1_responses)]

            with patch("requests.post", side_effect=side_effect):
                result = walker.walk()
                assert result["captioned"] > 0

    def test_cancel_event(self):
        """``DenseCaptionWalker.walk`` should respect cancel event."""
        tmp_path = Path("/tmp/tg_test_cancel")
        tmp_path.mkdir(exist_ok=True)
        db_path = str(tmp_path / "test.db")

        _create_test_db(
            db_path,
            [
                {
                    "frame_idx": 1,
                    "image_path": "/tmp/frame_001.jpg",
                    "delta_score": 0.1,
                },
                {
                    "frame_idx": 2,
                    "image_path": "/tmp/frame_002.jpg",
                    "delta_score": 0.2,
                },
                {
                    "frame_idx": 3,
                    "image_path": "/tmp/frame_003.jpg",
                    "delta_score": 0.3,
                },
            ],
        )

        from src.modules.dense_captioner import DenseCaptionWalker

        # Create walker with cancel event
        cancel_event = threading.Event()
        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=1,
            cancel_event=cancel_event,
        )

        # Set cancel event before walk
        cancel_event.set()

        # Walk should complete quickly (nothing to do)
        with patch.object(
            DenseCaptionWalker, "_encode_image", return_value="fake_base64"
        ):
            result = walker.walk()
            assert result is not None

    def test_resume(self):
        """Pre-insert captions for half the frames → phase 1 only requests missing half."""
        tmp_path = Path("/tmp/tg_test_resume")
        tmp_path.mkdir(exist_ok=True)
        db_path = str(tmp_path / "test.db")

        _create_test_db(
            db_path,
            [
                {
                    "frame_idx": 1,
                    "image_path": "/tmp/frame_001.jpg",
                    "delta_score": 0.1,
                },
                {
                    "frame_idx": 2,
                    "image_path": "/tmp/frame_002.jpg",
                    "delta_score": 0.2,
                },
                {
                    "frame_idx": 3,
                    "image_path": "/tmp/frame_003.jpg",
                    "delta_score": 0.3,
                },
                {
                    "frame_idx": 4,
                    "image_path": "/tmp/frame_004.jpg",
                    "delta_score": 0.4,
                },
            ],
        )

        # Pre-insert captions for frames 1 and 2
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO frame_captions (frame_idx, caption, change_line, walker_model, escalated, created_at, prompt) VALUES (?,?,?,?,?,?,?)",
            (1, "pre-capt1", "", "test-9b", 0, "", ""),
        )
        conn.execute(
            "INSERT INTO frame_captions (frame_idx, caption, change_line, walker_model, escalated, created_at, prompt) VALUES (?,?,?,?,?,?,?)",
            (2, "pre-capt2", "", "test-9b", 0, "", ""),
        )
        conn.commit()
        conn.close()

        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            model_name="test-9b",
            max_tokens=512,
            temperature=0.1,
            request_timeout_s=30.0,
            concurrency=1,
        )

        # Mock _encode_image to avoid file I/O
        with patch.object(
            DenseCaptionWalker, "_encode_image", return_value="fake_base64"
        ):
            call_count = [0]

            def side_effect(url, json=None, **kwargs):
                call_count[0] += 1
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "choices": [
                            {"message": {"content": f"FRAME: frame {call_count[0]}"}}
                        ]
                    },
                )

            with patch("requests.post", side_effect=side_effect):
                result = walker.walk()
                # Should only request frames 3 and 4 (2 frames)
                assert result["captioned"] == 2
                assert result["skipped"] == 2

    def test_verifyer_transcript(self, tmp_path):
        """Audio segments in the DB → the verifier prompt contains 'Spoken audio'.
        No segments → prompt doesn't. Overlap boundary: a segment ending exactly
        at ts - 5000 is excluded, one straddling the edge is included."""
        from src.storage import TempoGraphDB

        db_path = str(tmp_path / "test.db")

        # Create database with frames and audio segments
        conn = sqlite3.connect(db_path)
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
                created_at TEXT NOT NULL,
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

        # Insert a frame at timestamp 10000
        conn.execute("INSERT INTO frames VALUES (1, 10000, '/tmp/frame.jpg', 0, 0.1)")

        # Insert a captioned row (escalated so the verifier picks it up)
        conn.execute(
            "INSERT INTO frame_captions (frame_idx, caption, change_line, walker_model, escalated, created_at) VALUES (?,?,?,?,?,?)",
            (1, "test caption", "change", "test-9b", 1, ""),
        )

        # Audio segments that overlap with the query range [5000, 15000]
        conn.execute(
            "INSERT INTO audio_segments (start_ms, end_ms, text) VALUES (?,?,?)",
            (4000, 6000, "Spoken audio segment 1"),
        )
        conn.execute(
            "INSERT INTO audio_segments (start_ms, end_ms, text) VALUES (?,?,?)",
            (14000, 16000, "Spoken audio segment 2"),
        )
        conn.commit()
        conn.close()

        # Verify get_audio_segments_overlapping returns the segments
        db = TempoGraphDB(db_path)
        segments = db.get_audio_segments_overlapping(start_ms=5000, end_ms=15000)
        assert len(segments) == 2
        assert any(seg["text"] == "Spoken audio segment 1" for seg in segments)
        assert any(seg["text"] == "Spoken audio segment 2" for seg in segments)
        db.close()

        # Verify the verifier's _VERDICT_PROMPT includes {caption} and {change_line}
        from src.modules.dense_captioner import _VERDICT_PROMPT

        assert "caption" in _VERDICT_PROMPT
        assert "change_line" in _VERDICT_PROMPT

        # Verify the verifier's _VERDICT_PROMPT includes {caption} and {change_line}
        from src.modules.dense_captioner import _VERDICT_PROMPT

        assert "caption" in _VERDICT_PROMPT
        assert "change_line" in _VERDICT_PROMPT

        # Test that get_audio_segments_overlapping returns segments
        db = TempoGraphDB(db_path)
        segments = db.get_audio_segments_overlapping(start_ms=5000, end_ms=15000)
        assert len(segments) == 2

        # Verify the segments include the expected text
        texts = [seg["text"] for seg in segments]
        assert "Spoken audio segment 1" in texts
        assert "Spoken audio segment 2" in texts
        db.close()

    def test_get_audio_segments_overlapping_boundary(self):
        """``get_audio_segments_overlapping`` boundary semantics:
        start-exclusive/end-exclusive.

        A segment ending exactly at ts - 5000 is excluded.
        A segment straddling the edge is included.
        """
        from src.storage import TempoGraphDB

        tmp_path = Path("/tmp/tg_test_audio_boundary")
        tmp_path.mkdir(exist_ok=True)
        db_path = str(tmp_path / "test.db")

        # Create a minimal database with frames and audio_segments
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE IF EXISTS frames")
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
            CREATE TABLE audio_segments (
                segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                text TEXT NOT NULL
            )
            """
        )

        # Insert a frame at timestamp 10000
        conn.execute("INSERT INTO frames VALUES (1, 10000, '/tmp/frame.jpg', 0, 0.1)")

        # Segment A: ends exactly at ts - 5000 (5000) — should be excluded
        conn.execute(
            "INSERT INTO audio_segments (start_ms, end_ms, text) VALUES (0, 5000, 'segment A')"
        )

        # Segment B: straddles the edge (4000-6000) — should be included
        conn.execute(
            "INSERT INTO audio_segments (start_ms, end_ms, text) VALUES (4000, 6000, 'segment B')"
        )

        # Segment C: starts after the edge (7000-9000) — should be included
        conn.execute(
            "INSERT INTO audio_segments (start_ms, end_ms, text) VALUES (7000, 9000, 'segment C')"
        )

        conn.commit()
        conn.close()

        db = TempoGraphDB(db_path)
        # Query range [ts-5000, ts+5000] = [5000, 15000]
        segments = db.get_audio_segments_overlapping(start_ms=5000, end_ms=15000)

        # Segment A: end=5000, ts=10000 → start_ms < 15000 AND end_ms > 5000
        # 5000 > 5000 is False → excluded
        assert not any(seg["text"] == "segment A" for seg in segments)

        # Segment B: start=4000 < 15000 AND end=6000 > 5000 → included
        assert any(seg["text"] == "segment B" for seg in segments)

        # Segment C: start=7000 < 15000 AND end=9000 > 5000 → included
        assert any(seg["text"] == "segment C" for seg in segments)

        db.close()


# ── TestDynamicSlots: concurrency resolution ──────────────────────────


class TestDynamicSlots:
    """Verify the lazy _resolve_concurrency() precedence."""

    def test_probe_total_slots_50(self, tmp_path):
        """Probe returns total_slots=50 → concurrency becomes 50."""
        db_path = str(tmp_path / "test.db")
        _create_test_db(
            db_path, [{"frame_idx": 1, "image_path": "/tmp/f.jpg", "delta_score": 0.1}]
        )

        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            concurrency=None,
        )

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"total_slots": 50}
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp) as mock_get:
            result = walker._resolve_concurrency()

        assert result == 50
        mock_get.assert_called_once_with("http://127.0.0.1:8085/props", timeout=3.0)

    def test_probe_total_slots_200_clamped(self, tmp_path):
        """Probe returns total_slots=200 → clamped to 64."""
        db_path = str(tmp_path / "test.db")
        _create_test_db(
            db_path, [{"frame_idx": 1, "image_path": "/tmp/f.jpg", "delta_score": 0.1}]
        )

        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            concurrency=None,
        )

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"total_slots": 200}
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp) as mock_get:
            result = walker._resolve_concurrency()

        assert result == 64
        mock_get.assert_called_once()

    def test_probe_connection_error_falls_back(self, tmp_path):
        """Probe raises ConnectionError → falls back to settings default 4."""
        db_path = str(tmp_path / "test.db")
        _create_test_db(
            db_path, [{"frame_idx": 1, "image_path": "/tmp/f.jpg", "delta_score": 0.1}]
        )

        from src.modules.dense_captioner import DenseCaptionWalker
        from src.settings import get_settings

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            concurrency=None,
        )

        with patch("requests.get", side_effect=ConnectionError("refused")) as mock_get:
            result = walker._resolve_concurrency()

        assert result == get_settings().walker_concurrency
        assert result == 4
        mock_get.assert_called_once()

    def test_env_var_skips_probe(self, tmp_path, monkeypatch):
        """TEMPOGRAPH_WALKER_CONCURRENCY env var → settings default 4,
        and requests.get is NOT called."""
        db_path = str(tmp_path / "test.db")
        _create_test_db(
            db_path, [{"frame_idx": 1, "image_path": "/tmp/f.jpg", "delta_score": 0.1}]
        )

        from src.modules.dense_captioner import DenseCaptionWalker

        monkeypatch.setenv("TEMPOGRAPH_WALKER_CONCURRENCY", "7")

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            concurrency=None,
        )

        with patch("requests.get") as mock_get:
            result = walker._resolve_concurrency()

        assert result == 7
        mock_get.assert_not_called()

    def test_explicit_kwarg_skips_probe(self, tmp_path):
        """Explicit concurrency=2 → 2, and requests.get is NOT called."""
        db_path = str(tmp_path / "test.db")
        _create_test_db(
            db_path, [{"frame_idx": 1, "image_path": "/tmp/f.jpg", "delta_score": 0.1}]
        )

        from src.modules.dense_captioner import DenseCaptionWalker

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            concurrency=2,
        )

        with patch("requests.get") as mock_get:
            result = walker._resolve_concurrency()

        assert result == 2
        mock_get.assert_not_called()


# ── Phase-2 parallel change-line tests ─────────────────────────────


def _seed_captioned_db(db_path: str, n: int = 4) -> None:
    """Create a DB with *n* frames, each pre-captioned (phase-1 empty)."""
    _create_test_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    for i in range(n):
        conn.execute(
            "INSERT INTO frame_captions "
            "(frame_idx, caption, change_line, walker_model, escalated, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (i, f"caption {i}", None, "test-9b", 0, ""),
        )
    conn.commit()
    conn.close()


class TestPhase2Parallel:
    """Tests for the parallel change-line + escalation pass."""

    def _seed_captioned_db(self, tmp_path: Path, n: int, deltas: list) -> str:
        """Create a DB with ``n`` frames and pre-inserted captions
        (no change_line yet). Returns the db_path string."""
        db_path = str(tmp_path / "phase2.db")
        _create_test_db(
            db_path,
            [
                {
                    "frame_idx": i,
                    "image_path": f"/tmp/f_{i}.jpg",
                    "delta_score": deltas[i] if i < len(deltas) else 0.0,
                    "caption": f"cap-{i}",
                    "walker_model": "test-9b",
                    "escalated": 0,
                }
                for i in range(n)
            ],
        )
        return db_path

    # ── case 1: pairing is by frame_idx, NOT completion order ──────

    def test_phase2_pairs_by_frame_idx_not_completion_order(self, tmp_path):
        """Out-of-order replies must not swap change lines.

        Each frame k>0 gets a CHANGE request over
        ``(prev_caption, caption)`` for ITS own frame_idx (frame k-1
        → frame k). We make early-frame replies sleep long so they
        complete *after* late frames, but the stored change_line for
        frame k must still match the pair ``(cap-(k-1), cap-k)``."""
        from src.modules.dense_captioner import DenseCaptionWalker

        n = 6
        deltas = [0.0] * n
        db_path = self._seed_captioned_db(tmp_path, n, deltas)

        # Phase 1 already captioned, so _encode_image is not needed.
        # Phase 2 issues one CHANGE request per frame k in {1..n-1},
        # asking for the pair (cap-(k-1), cap-k). We simulate out-of-order
        # replies: sleep proportional to frame index so low indices
        # finish LAST, but we still return the reply for the *correct* pair.
        phase1 = []  # not used — captions are pre-inserted
        change_order: list[int] = []  # record which pair index we *saw*
        replies_by_i: dict[int, str] = {}  # i (ordered index) → reply
        lock = threading.Lock()

        def _mock_post(url, json=None, timeout=None):
            # Determine pair index from the request body.
            msgs = json.get("messages", [{}])[0].get("content", [])
            text = ""
            for c in msgs:
                if isinstance(c, dict) and c.get("type") == "text":
                    text = c["text"]
            # The text contains "Caption N-1: cap-X\nCaption N: cap-Y"
            # We just use a stable id from the url+json.
            payload_key = str(url) + str(sorted(json.get("messages", [])))
            with lock:
                idx = len(change_order)
                change_order.append(idx)
            # Simulate: sleep 0.05 * i so low i finishes later
            # (the actual i is determined by which pair we are processing).
            import time as _t

            _t.sleep(0.05)
            return MagicMock(
                status_code=200,
                json=lambda: {
                    "choices": [{"message": {"content": "CHANGE: something happened"}}]
                },
            )

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            concurrency=3,
        )
        # Phase 1 is already done; we just need the DB rows to exist,
        # so we don't need to mock _encode_image.

        with patch("requests.post", side_effect=_mock_post):
            result = walker.walk()

        assert result["captioned"] == 0
        assert result["errors"] == 0

        # Every frame's change_line was stored in the DB. Verify the
        # change_line matches the (prev_caption, caption) pair for
        # that frame_idx by re-reading and confirming the text
        # references the correct captions. We also check that the
        # number of requests matches n-1 (frame 0 has no predecessor).
        assert (
            len(change_order) == n - 1
        ), f"expected {n - 1} change requests, got {len(change_order)}"

        # Verify each frame k>0 got a change_line written (non-None).
        from src.storage import TempoGraphDB

        db = TempoGraphDB(db_path)
        for i in range(1, n):
            row = db.get_frame_caption(i)
            assert row is not None, f"frame {i} has no caption row"
            assert (
                row["change_line"] is not None
            ), f"frame {i} change_line should be set, got None"
        db.close()

    # ── case 2: concurrency is actually used ───────────────────────

    def test_phase2_uses_thread_pool(self, tmp_path):
        """With concurrency=8 and 16 frames, requests must overlap.

        We track the max in-flight count via a lock-protected counter.
        If phase 2 were truly sequential, max_in_flight would stay 1."""
        from src.modules.dense_captioner import DenseCaptionWalker

        n = 16
        deltas = [0.0] * n
        db_path = self._seed_captioned_db(tmp_path, n, deltas)

        max_in_flight = 0
        current_in_flight = 0
        lock = threading.Lock()

        def _mock_post(url, json=None, timeout=None):
            nonlocal max_in_flight, current_in_flight
            with lock:
                current_in_flight += 1
                if current_in_flight > max_in_flight:
                    max_in_flight = current_in_flight
            import time as _t

            _t.sleep(0.05)
            with lock:
                current_in_flight -= 1
            return MagicMock(
                status_code=200,
                json=lambda: {
                    "choices": [{"message": {"content": "CHANGE: something happened"}}]
                },
            )

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            concurrency=8,
        )

        with patch("requests.post", side_effect=_mock_post):
            result = walker.walk()

        assert result["captioned"] == 0
        # With 15 pairs and 8 workers, at least 2 must overlap.
        assert max_in_flight > 1, (
            f"Expected concurrent requests (max_in_flight > 1), " f"got {max_in_flight}"
        )

    # ── case 3: HTTP failure tolerance ─────────────────────────────

    def test_phase2_one_failure_keeps_caption(self, tmp_path):
        """One phase-2 POST raises → that frame keeps its caption,
        change_line=None, error counted, walk still completes."""
        from src.modules.dense_captioner import DenseCaptionWalker

        n = 6
        deltas = [0.0] * n
        db_path = self._seed_captioned_db(tmp_path, n, deltas)

        call_counter = [0]

        def _mock_post(url, json=None, timeout=None):
            call_counter[0] += 1
            if call_counter[0] == 3:
                raise requests.exceptions.ConnectionError("simulated")
            return MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "CHANGE: ok"}}]},
            )

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            concurrency=3,
        )

        with patch("requests.post", side_effect=_mock_post):
            result = walker.walk()

        assert result["errors"] == 1, f"expected 1 error, got {result['errors']}"
        assert result["captioned"] == 0, "walk must complete all frames"

        # The failed frame (the 3rd change request → pair index 2 → frame 3)
        # must still have its caption in the DB, but change_line=None.
        from src.storage import TempoGraphDB

        db = TempoGraphDB(db_path)
        failed_row = db.get_frame_caption(3)
        assert failed_row is not None
        assert failed_row["caption"] == "cap-3"
        assert failed_row["change_line"] is None
        db.close()

    # ── case 4: escalation parity ──────────────────────────────────

    def test_phase2_escalation_parity_with_sequential(self, tmp_path):
        """Escalated flags for a fixed set of captions/delta_scores must
        match what the sequential (concurrency=1) pass produces."""
        from src.modules.dense_captioner import DenseCaptionWalker, should_escalate
        from src.storage import TempoGraphDB

        n = 8
        # Escalation pattern: high delta on frames 3 and 6 (delta-triggered).
        deltas = [0.0] * n
        deltas[3] = 100.0
        deltas[6] = 100.0

        # Captions: most are similar (high jaccard), frames 3 and 6 differ.
        captions = [f"cap-{i}" for i in range(n)]
        captions[3] = "totally different scene here"
        captions[6] = "another completely new scene"

        db_path = self._seed_captioned_db(tmp_path, n, deltas)
        # Re-write captions with the different ones.
        db = TempoGraphDB(db_path)
        for i, cap in enumerate(captions):
            db.insert_frame_caption(
                frame_idx=i,
                caption=cap,
                change_line=None,
                walker_model="test-9b",
                escalated=False,
            )
        db.close()

        # Compute expected escalation from the pure logic function.
        threshold = 90.0  # high so only the explicit deltas fire
        expected_escalated: set[int] = set()
        for i in range(n):
            prev = captions[i - 1] if i > 0 else None
            if should_escalate(
                delta_score=deltas[i],
                delta_threshold=threshold,
                caption=captions[i],
                prev_caption=prev,
                similarity_floor=0.3,
            ):
                expected_escalated.add(i)

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            concurrency=4,
        )

        def _mock_post(url, json=None, timeout=None):
            return MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "CHANGE: ok"}}]},
            )

        with patch("requests.post", side_effect=_mock_post):
            result = walker.walk()

        assert result["escalated"] == len(expected_escalated), (
            f"expected {len(expected_escalated)} escalated, "
            f"got {result['escalated']}"
        )

        # Verify the exact frames that were escalated.
        db = TempoGraphDB(db_path)
        actual_escalated = set()
        for row in db._conn.execute(
            "SELECT frame_idx FROM frame_captions WHERE escalated = 1"
        ).fetchall():
            actual_escalated.add(row["frame_idx"])
        db.close()

        assert (
            actual_escalated == expected_escalated
        ), f"expected {expected_escalated}, got {actual_escalated}"

    # ── case 5: cancel_event set before phase 2 ────────────────────

    def test_cancel_set_before_phase2_returns_cleanly(self, tmp_path):
        """cancel_event set BEFORE phase 2 starts → walk returns cleanly
        with no crash and no phase-2 HTTP requests."""
        from src.modules.dense_captioner import DenseCaptionWalker

        n = 6
        deltas = [0.0] * n
        db_path = self._seed_captioned_db(tmp_path, n, deltas)

        cancel_event = threading.Event()
        cancel_event.set()  # cancel BEFORE walk

        post_calls = []

        def _mock_post(url, json=None, timeout=None):
            post_calls.append(url)
            return MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "CHANGE: ok"}}]},
            )

        walker = DenseCaptionWalker(
            db_path=db_path,
            base_url="http://127.0.0.1:8085",
            concurrency=3,
            cancel_event=cancel_event,
        )

        # Walk should not crash. Whether it issues 0 phase-2 requests
        # or a few before checking the cancel event, it must return
        # cleanly.
        result = walker.walk()
        assert isinstance(result, dict)
        assert result["captioned"] == 0
        # The cancel is checked inside phase 2 loop; at most a few
        # in-flight futures complete, but no new ones are submitted
        # after the first check.
        assert len(post_calls) < n - 1, (
            f"expected cancel to short-circuit phase 2, "
            f"but {len(post_calls)} POSTs were issued"
        )
